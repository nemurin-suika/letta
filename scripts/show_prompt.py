#!/usr/bin/env python3
"""에이전트의 시스템 프롬프트를 출력하는 스크립트.

사용법:
    python scripts/show_prompt.py <에이전트_이름>
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


def get_agent_detail(agent_id: str) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{LETTA_BASE_URL}/v1/agents/{agent_id}/",
            headers={"Authorization": f"Bearer {LETTA_TOKEN}"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="에이전트 시스템 프롬프트 조회")
    parser.add_argument("agent", help="에이전트 이름")
    args = parser.parse_args()

    agents = get_all_agents()
    agent = next((a for a in agents if a["name"] == args.agent), None)
    if not agent:
        names = [a["name"] for a in agents]
        print(f"에이전트 '{args.agent}'를 찾을 수 없습니다.", file=sys.stderr)
        print(f"존재하는 에이전트: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)

    detail = get_agent_detail(agent["id"])
    system_prompt = (detail.get("system") or "").strip()

    if not system_prompt:
        print("(시스템 프롬프트 없음)", file=sys.stderr)
        sys.exit(1)

    print(system_prompt)


if __name__ == "__main__":
    main()
