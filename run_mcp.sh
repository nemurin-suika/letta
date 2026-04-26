#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LETTA_URL="http://localhost:8283"

# ── MCP 서버 시작 (Docker 외부, SSE 모드) ──────────────────────────────────
# mcp-proxy: stdio MCP 서버를 SSE HTTP 서버로 변환
#   time MCP        → http://localhost:8001/sse
#   claude-code MCP → http://localhost:8002/sse

echo "[MCP] time 서버 시작 (port 8001)..."
uvx mcp-proxy --sse-port 8001 -- uvx mcp-server-time --local-timezone Asia/Seoul \
  > /tmp/mcp_time.log 2>&1 &
MCP_TIME_PID=$!

echo "[MCP] claude-code 서버 시작 (port 8002)..."
CLAUDE_CODE_CWD="/Users/nemulin/develop/playground" \
  uvx mcp-proxy --sse-port 8002 -- \
  uv --directory "$SCRIPT_DIR" run python mcp_servers/claude_code_server.py \
  > /tmp/mcp_claude.log 2>&1 &
MCP_CLAUDE_PID=$!

# Ctrl+C (SIGINT) / SIGTERM 시 포트 8001·8002를 점유한 프로세스를 모두 종료
cleanup() {
  echo ""
  echo "[MCP] 서버 종료 중..."
  # uvx가 fork한 자식 프로세스까지 잡기 위해 포트로 직접 kill
  lsof -ti tcp:8001 | xargs kill -9 2>/dev/null
  lsof -ti tcp:8002 | xargs kill -9 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# ── Letta 준비 대기 후 MCP 서버 등록 ──────────────────────────────────────

echo "[Letta] 서버 준비 대기 중 (최대 90초)..."
for i in $(seq 1 18); do
  sleep 5
  if curl -sf "$LETTA_URL/v1/health" > /dev/null 2>&1; then
    echo "[Letta] 서버 준비 완료 (${i}×5초 경과)"
    break
  fi
  echo "[Letta] 대기 중... (${i}/18)"
done

# ── MCP 서버 등록 ─────────────────────────────────────────────────────────
# Docker 내부에서 host.docker.internal:PORT 로 호스트의 MCP SSE 서버에 접근
echo ""
echo "[MCP] time 서버 등록 중..."
curl -sf -X POST "$LETTA_URL/v1/mcp-servers/" \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "time",
    "config": {
      "mcp_server_type": "sse",
      "server_url": "http://host.docker.internal:8001/sse"
    }
  }' | python3 -m json.tool 2>/dev/null || echo "(등록 실패 - 이미 등록되어 있을 수 있음)"

echo ""
echo "[MCP] claude-code 서버 등록 중..."
curl -sf -X POST "$LETTA_URL/v1/mcp-servers/" \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "claude-code",
    "config": {
      "mcp_server_type": "sse",
      "server_url": "http://host.docker.internal:8002/sse"
    }
  }' | python3 -m json.tool 2>/dev/null || echo "(등록 실패 - 이미 등록되어 있을 수 있음)"

echo ""
echo "[완료] MCP 서버 등록 완료"
echo "  - time MCP:        http://host.docker.internal:8001/sse"
echo "  - claude-code MCP: http://host.docker.internal:8002/sse"
echo "  - Letta UI:        http://localhost:8283"
echo ""
echo "[MCP] 실행 중. 종료하려면 Ctrl+C를 누르세요."

# Ctrl+C까지 대기 (sleep을 짧게 쪼개서 trap이 빠르게 반응하도록)
while true; do
  sleep 1
done
