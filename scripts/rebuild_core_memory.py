#!/usr/bin/env python3
"""에이전트의 메시지 히스토리를 분석해서 core memory(human/persona 블록)를 채우는 스크립트.

메시지를 80,000자 단위로 배치 분할해서 claude -p를 여러 번 호출,
각 배치 결과를 합산 후 블록에 저장.

사용법:
    python scripts/rebuild_core_memory.py                     # 대화형으로 에이전트 선택
    python scripts/rebuild_core_memory.py --agent Open-LLM-VTuber
    python scripts/rebuild_core_memory.py --all               # 전체 에이전트
    python scripts/rebuild_core_memory.py --all --dry-run
"""

import argparse
import json
import re
import subprocess

import httpx

LETTA_BASE_URL = "http://localhost:8283"
LETTA_TOKEN = "test_server_token"

MAX_BATCH_CHARS = 80_000

HUMAN_TOTAL_LIMIT = 250
PERSONA_TOTAL_LIMIT = 1000

REBUILD_PROMPT = """\
당신은 아래 시스템 프롬프트로 정의된 AI 에이전트입니다:

<system_prompt>
{system_prompt}
</system_prompt>

아래는 당신과 사용자의 대화 기록입니다 (전체 {batch_index}/{batch_total} 구간).
이 구간에서 "이건 기억해둬야겠다"고 느낄 만한 것들을 두 블록으로 정리해주세요.

당신에게는 **archival memory(장기 저장소)**도 있습니다. 상세한 에피소드, 후원 기록, 시청자 정보 등은 archival에 별도로 저장되어 있고 언제든 키워드로 검색해 꺼낼 수 있습니다.

따라서 core memory(human/persona 블록)에는 **검색 없이 즉시 참조해야 할 것만** 남겨야 합니다:
- 매 대화에서 전제로 깔려야 할 사실 (성별, 직업, 관계 등)
- 잊으면 실례가 되는 핵심 정보
- 자신의 정체성·말투·행동 패턴에 영향을 주는 경험

아래에 해당하는 것은 core memory에 넣지 마세요 (archival에 있거나 있어야 할 내용):
- 특정 날짜·이벤트의 상세 내용 (방송 요약, 후원 기록, 에피소드 등)
- 이미 "핵심 사실"로 요약된 것의 원본 상세 내용
- 드물게 필요하거나 검색으로 찾으면 되는 정보

판단 기준:
- **human 블록**: 이 사람과 다음 대화에서 처음부터 알고 있어야 할 핵심 사실. 검색 없이 바로 써야 하는 것만.
- **persona 블록**: 자신이 어떤 AI인지를 규정하는 정체성·패턴·결론. 매 대화에서 전제가 되어야 할 것만.

글자 수 제한 (반드시 준수):
- human 블록: 이 구간에서 최대 {human_limit}자
- persona 블록: 이 구간에서 최대 {persona_limit}자

제한을 넘으면 덜 중요한 것부터 생략하세요. 항목별로 간결하게. 아래 구분자를 정확히 지켜서 출력하세요:

===HUMAN===
(human 블록 내용)
===PERSONA===
(persona 블록 내용)

기억할 게 없으면 각 섹션을 비워두세요.

대화 기록:
{transcript}"""


def get_all_agents() -> list:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    return resp.json()


def get_agent_detail(agent_id: str) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    return resp.json()


def get_all_messages(agent_id: str) -> list:
    all_messages = []
    cursor = None
    with httpx.Client(timeout=30) as client:
        while True:
            params = {"limit": 500, "order": "asc"}
            if cursor:
                params["after"] = cursor
            resp = client.get(
                f"{LETTA_BASE_URL}/v1/agents/{agent_id}/messages/",
                headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
                params=params,
                follow_redirects=True,
            )
            resp.raise_for_status()
            page = resp.json()
            page = page if isinstance(page, list) else page.get("messages", [])
            if not page:
                break
            all_messages.extend(page)
            if len(page) < 500:
                break
            cursor = page[-1]["id"]
    return all_messages


def build_transcript_lines(messages: list) -> list[str]:
    lines = []
    for m in messages:
        msg_type = m.get("message_type", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        content = (content or "").strip()
        if not content:
            continue
        if msg_type == "assistant_message":
            lines.append(f"AI: {content}")
        elif msg_type == "user_message":
            lines.append(f"사용자: {content}")
    return lines


def split_into_batches(lines: list[str], max_chars: int) -> list[str]:
    """줄 목록을 max_chars 이내의 텍스트 배치로 분할."""
    batches = []
    current: list[str] = []
    current_chars = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current and current_chars + line_len > max_chars:
            batches.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_len

    if current:
        batches.append("\n".join(current))

    return batches


def ask_claude(prompt: str, retries: int = 3, delay: float = 5.0) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, cwd="/tmp",
        )
        if result.returncode != 0:
            last_err = RuntimeError(f"claude CLI 실패: {result.stderr.strip()}")
            print(f"    재시도 {attempt}/{retries} ({last_err})")
            import time; time.sleep(delay)
            continue
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            last_err = RuntimeError(f"JSON 파싱 실패: {result.stdout[:200]}")
            print(f"    재시도 {attempt}/{retries} ({last_err})")
            import time; time.sleep(delay)
            continue
        if data.get("is_error"):
            last_err = RuntimeError(f"claude CLI 오류: {data.get('result')}")
            print(f"    재시도 {attempt}/{retries} ({last_err})")
            import time; time.sleep(delay)
            continue
        return (data.get("result") or "").strip()
    raise last_err or RuntimeError("claude CLI 실패 (알 수 없는 오류)")


def parse_blocks(result: str) -> tuple[str, str]:
    """===HUMAN=== / ===PERSONA=== 구분자로 두 블록 내용 분리."""
    m = re.search(r"===HUMAN===(.*?)===PERSONA===(.*?)$", result, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return result.strip(), ""


def get_block_id(agent_id: str, label: str) -> str | None:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/core-memory/blocks/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    for block in resp.json():
        if block.get("label") == label:
            return block["id"]
    return None


def update_block(block_id: str, value: str) -> None:
    with httpx.Client(timeout=10) as client:
        resp = client.patch(
            f"{LETTA_BASE_URL}/v1/blocks/{block_id}/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}",
                     "Content-Type": "application/json"},
            json={"value": value},
            follow_redirects=True,
        )
        resp.raise_for_status()


def rebuild_agent(agent: dict, dry_run: bool) -> None:
    agent_id = agent["id"]
    agent_name = agent["name"]
    print(f"\n{'='*50}")
    print(f"대상: {agent_name} ({agent_id})")

    detail = get_agent_detail(agent_id)
    system_prompt = (detail.get("system") or "").strip()
    if not system_prompt:
        print("  시스템 프롬프트 없음 — 건너뜀.")
        return

    print("메시지 수집 중...")
    messages = get_all_messages(agent_id)
    target = [m for m in messages if m.get("message_type") in ("assistant_message", "user_message")]
    print(f"{len(target)}개 메시지 수집 완료")

    if len(target) < 10:
        print("메시지가 너무 적어 건너뜁니다.")
        return

    lines = build_transcript_lines(target)
    batches = split_into_batches(lines, MAX_BATCH_CHARS)
    total_chars = sum(len(b) for b in batches)
    n = len(batches)
    human_per_batch = HUMAN_TOTAL_LIMIT // n
    persona_per_batch = PERSONA_TOTAL_LIMIT // n
    print(f"트랜스크립트: {total_chars:,}자 → {n}개 배치")
    print(f"배치당 한도: human {human_per_batch}자 / persona {persona_per_batch}자")

    all_human: list[str] = []
    all_persona: list[str] = []

    for i, batch_text in enumerate(batches, 1):
        print(f"  [배치 {i}/{n}] {len(batch_text):,}자 분석 중...")
        try:
            result = ask_claude(REBUILD_PROMPT.format(
                system_prompt=system_prompt,
                transcript=batch_text,
                batch_index=i,
                batch_total=n,
                human_limit=human_per_batch,
                persona_limit=persona_per_batch,
            ))
        except RuntimeError as e:
            print(f"    오류: {e}")
            continue
        human_val, persona_val = parse_blocks(result)
        if human_val:
            all_human.append(human_val)
        if persona_val:
            all_persona.append(persona_val)

    # 합산 후 전체 한도 초과 시 잘라냄 (안전장치)
    human_value = "\n\n".join(all_human)[:HUMAN_TOTAL_LIMIT]
    persona_value = "\n\n".join(all_persona)[:PERSONA_TOTAL_LIMIT]

    print(f"\n[human] {len(human_value)}자\n{human_value}\n")
    print(f"[persona] {len(persona_value)}자\n{persona_value}\n")

    if dry_run:
        print("--dry-run 모드: 블록 업데이트 건너뜀.")
        return

    for label, value in [("human", human_value), ("persona", persona_value)]:
        if not value:
            print(f"[{label}] 내용 없음 — 건너뜀.")
            continue
        block_id = get_block_id(agent_id, label)
        if block_id:
            update_block(block_id, value)
            print(f"[{label}] 블록 업데이트 완료.")
        else:
            print(f"[{label}] 블록 없음 — 건너뜀.")


def main():
    parser = argparse.ArgumentParser(description="메시지 히스토리로 core memory 재구축")
    parser.add_argument("--agent", help="에이전트 이름 (생략 시 대화형 선택)")
    parser.add_argument("--all", action="store_true", help="모든 에이전트 처리")
    parser.add_argument("--dry-run", action="store_true", help="블록 업데이트 없이 결과만 출력")
    args = parser.parse_args()

    agents = get_all_agents()

    if args.all:
        for agent in agents:
            rebuild_agent(agent, args.dry_run)
    elif args.agent:
        agent = next((a for a in agents if a["name"] == args.agent), None)
        if not agent:
            raise SystemExit(f"에이전트 '{args.agent}'를 찾을 수 없습니다.")
        rebuild_agent(agent, args.dry_run)
    else:
        print("\n에이전트 목록:")
        for i, a in enumerate(agents, 1):
            print(f"  {i}. {a['name']}  ({a['id']})")
        choice = int(input("번호 선택: ").strip()) - 1
        rebuild_agent(agents[choice], args.dry_run)


if __name__ == "__main__":
    main()
