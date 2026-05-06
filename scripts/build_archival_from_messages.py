"""
메시지 히스토리 → Archival Memory 리빌더

기존 archival memory를 모두 삭제하고, 메시지 히스토리를 80,000자 단위 배치로
나눠 claude -p를 여러 번 호출해 빠짐없이 기억을 추출 후 저장.

사용법:
    python scripts/build_archival_from_messages.py
    python scripts/build_archival_from_messages.py --agent Open-LLM-VTuber
    python scripts/build_archival_from_messages.py --all
    python scripts/build_archival_from_messages.py --all --dry-run
"""

import argparse
import json
import subprocess

import httpx

LETTA_BASE_URL = "http://localhost:8283"
LETTA_TOKEN = "test_server_token"

MAX_BATCH_CHARS = 80_000

REBUILD_PROMPT = """\
당신은 아래 시스템 프롬프트로 정의된 AI 에이전트입니다:

<system_prompt>
{system_prompt}
</system_prompt>

아래는 당신과 사용자의 대화 기록입니다. 이 대화를 당신의 캐릭터 관점에서 읽고,
나중에 다시 참고하기 위해 archival memory에 저장할 기억 목록을 추출해주세요.

추출 기준 (에이전트 캐릭터 관점):
- 사람/시청자에 대해 새로 알게 된 것 (이름, 특성, 특별한 사건)
- 에이전트 자신에게 일어난 중요한 사건이나 경험
- 후원/구독 이벤트: 누가, 얼마를, 어떤 말과 함께 (인상적인 교류)
- 반복되는 패턴이나 선호사항
- 나중에 잊으면 어색할 것 같은 맥락 정보

추출하지 않을 것:
- 일회성 잡담, 지나가는 말
- 이미 명백한 캐릭터 설정에 있는 내용
- 시스템 메시지, 오류, 기술적인 운영 내용

반드시 아래 JSON 형식만 출력하세요 (설명 없이):
[
  {{
    "text": "기억 내용 (구체적으로, 문장 형태로)",
    "tags": ["태그1", "태그2"]
  }},
  ...
]

기억할 게 없으면 빈 배열 [] 를 출력하세요.

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


def parse_memories(response: str) -> list[dict]:
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(response[start:end + 1])
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and "text" in m]
    except json.JSONDecodeError:
        pass
    return []


def get_all_archival(agent_id: str) -> list:
    all_items = []
    cursor = None
    with httpx.Client(timeout=30) as client:
        while True:
            params = {"limit": 100, "ascending": "true"}
            if cursor:
                params["after"] = cursor
            resp = client.get(
                f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory",
                headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
                params=params,
                follow_redirects=True,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            all_items.extend(page)
            if len(page) < 100:
                break
            cursor = page[-1]["id"]
    return all_items


def delete_archival(agent_id: str, memory_id: str) -> None:
    with httpx.Client(timeout=10) as client:
        resp = client.delete(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory/{memory_id}",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()


def insert_archival(agent_id: str, text: str, tags: list[str] | None = None) -> None:
    body: dict = {"text": text}
    if tags:
        body["tags"] = tags
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            json=body,
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

    if len(target) < 5:
        print("메시지가 너무 적어 건너뜁니다.")
        return

    lines = build_transcript_lines(target)
    batches = split_into_batches(lines, MAX_BATCH_CHARS)
    total_chars = sum(len(b) for b in batches)
    print(f"트랜스크립트: {total_chars:,}자 → {len(batches)}개 배치")

    # 모든 배치에서 기억 추출
    all_memories: list[dict] = []
    for i, batch_text in enumerate(batches, 1):
        print(f"  [배치 {i}/{len(batches)}] {len(batch_text):,}자 분석 중...")
        try:
            response = ask_claude(REBUILD_PROMPT.format(
                system_prompt=system_prompt,
                transcript=batch_text,
            ))
        except RuntimeError as e:
            print(f"    오류: {e}")
            continue
        memories = parse_memories(response)
        print(f"    추출: {len(memories)}개")
        all_memories.extend(memories)

    print(f"\n총 추출된 기억: {len(all_memories)}개")
    for mem in all_memories:
        tags = mem.get("tags", [])
        text = mem.get("text", "")
        preview = text[:80] + ("..." if len(text) > 80 else "")
        print(f"  → [{', '.join(tags)}] {preview}")

    if dry_run:
        print("\n--dry-run 모드: 저장 건너뜀.")
        return

    # 기존 archival 전부 삭제
    existing = get_all_archival(agent_id)
    if existing:
        print(f"\n기존 archival {len(existing)}개 삭제 중...")
        for item in existing:
            delete_archival(agent_id, item["id"])
        print("삭제 완료.")

    # 새 기억 저장
    saved = 0
    for mem in all_memories:
        text = mem.get("text", "").strip()
        tags = mem.get("tags") or None
        if not text:
            continue
        insert_archival(agent_id, text, tags)
        saved += 1

    print(f"archival memory 저장 완료: {saved}개")


def main():
    parser = argparse.ArgumentParser(description="메시지 히스토리로 archival memory 재구축")
    parser.add_argument("--agent", help="에이전트 이름 (생략 시 대화형 선택)")
    parser.add_argument("--all", action="store_true", help="모든 에이전트 처리")
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 결과만 출력")
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
