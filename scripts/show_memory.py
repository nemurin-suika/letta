#!/usr/bin/env python3
"""에이전트의 core memory와 archival memory를 출력하는 스크립트.

사용법:
    python scripts/show_memory.py <에이전트_이름>
    python scripts/show_memory.py <에이전트_이름> --archival-limit 50
"""

import argparse
import sys

import httpx

LETTA_BASE_URL = "http://localhost:8283"
LETTA_TOKEN = "test_server_token"


def get_all_agents() -> list:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    return resp.json()


def get_core_memory_blocks(agent_id: str) -> list:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/core-memory/blocks/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    return resp.json()


def get_archival_memory(agent_id: str, limit: int | None) -> list:
    all_passages = []
    cursor = None
    with httpx.Client(timeout=30) as client:
        while True:
            page_size = 100 if limit is None else min(limit - len(all_passages), 100)
            params = {"limit": page_size, "ascending": "true"}
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
            page = page if isinstance(page, list) else page.get("passages", [])
            if not page:
                break
            all_passages.extend(page)
            if limit is not None and len(all_passages) >= limit:
                break
            if len(page) < page_size:
                break
            cursor = page[-1]["id"]
    return all_passages


def print_core_memory(blocks: list) -> None:
    print("=" * 60)
    print("CORE MEMORY")
    print("=" * 60)
    if not blocks:
        print("  (없음)")
        return
    for block in blocks:
        label = block.get("label", "?")
        value = block.get("value") or ""
        char_limit = block.get("limit", 0)
        read_only = block.get("read_only", False)
        flags = []
        if read_only:
            flags.append("read-only")
        flags_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"\n[{label}]{flags_str}  ({len(value)}/{char_limit}자)")
        print("-" * 40)
        print(value if value else "(비어 있음)")


def print_archival_memory(passages: list) -> None:
    print("\n" + "=" * 60)
    print(f"ARCHIVAL MEMORY  ({len(passages)}개)")
    print("=" * 60)
    if not passages:
        print("  (없음)")
        return
    for i, p in enumerate(passages, 1):
        pid = p.get("id", "")
        created_at = p.get("created_at", "")[:19].replace("T", " ")
        tags = p.get("tags") or []
        tags_str = f"  tags: {', '.join(tags)}" if tags else ""
        print(f"\n[{i}] {created_at}  id: {pid}{tags_str}")
        print("-" * 40)
        print(p.get("text", ""))


def main():
    parser = argparse.ArgumentParser(description="에이전트 메모리 조회")
    parser.add_argument("agent", help="에이전트 이름")
    parser.add_argument("--archival-limit", type=int, default=None, metavar="N",
                        help="archival memory 최대 조회 개수 (생략 시 전체)")
    args = parser.parse_args()

    agents = get_all_agents()
    agent = next((a for a in agents if a["name"] == args.agent), None)
    if not agent:
        names = [a["name"] for a in agents]
        print(f"에이전트 '{args.agent}'를 찾을 수 없습니다.", file=sys.stderr)
        print(f"존재하는 에이전트: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)

    agent_id = agent["id"]
    print(f"에이전트: {agent['name']}  ({agent_id})")

    blocks = get_core_memory_blocks(agent_id)
    print_core_memory(blocks)

    passages = get_archival_memory(agent_id, args.archival_limit)
    print_archival_memory(passages)


if __name__ == "__main__":
    main()
