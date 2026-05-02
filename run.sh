#!/bin/bash

source ~/.secrets

# Gemini API 키 (embedding 모델용) - 429 발생 시 자동으로 다음 키로 전환됨
GEMINI_API_KEY="${GEMINI_KEYS[0]}"
GEMINI_API_KEYS="$(IFS=,; echo "${GEMINI_KEYS[*]}")"

# ── Letta Docker 실행 ────────────────────────────────────────────────────────
# --add-host=host.docker.internal:host-gateway
#   macOS Docker Desktop: host.docker.internal은 자동으로 사용 가능하지만 명시적으로 추가
#   Linux: 이 옵션 없이는 host.docker.internal이 컨테이너에서 사용 불가

mkdir -p ~/.letta/.persist/pgdata

# ── Claude CLI 로그인 확인 ────────────────────────────────────────────────────
# letta-claude-auth 볼륨에 저장된 인증 정보로 로그인 상태 확인
# 미로그인 상태면 대화형 로그인 후 볼륨에 저장
_CLAUDE_AUTH_OUT=$(docker run --rm \
  -v letta-claude-auth:/root/.claude \
  letta/letta:nemurin \
  claude -p "ping" --output-format json 2>/dev/null)

if echo "$_CLAUDE_AUTH_OUT" | grep -qi "not logged in"; then
    echo "[Claude] 로그인이 필요합니다. 브라우저 인증을 진행합니다..."
    docker run --rm -it \
      -v letta-claude-auth:/root/.claude \
      letta/letta:nemurin \
      claude login
    echo "[Claude] 로그인 완료"
else
    echo "[Claude] 로그인 확인됨"
fi

# Claude CLI fallback 설정 (Google 429 시 호스트의 claude -p 서브프로세스로 전환)
LLM_FALLBACK_CONFIG='{"model": "claude-cli", "model_endpoint_type": "claude_cli", "context_window": 200000}'

echo "[Letta] Docker 컨테이너 시작..."
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -v letta-claude-auth:/root/.claude \
  -p 8283:8283 \
  --add-host=host.docker.internal:host-gateway \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e GEMINI_API_KEYS="$GEMINI_API_KEYS" \
  -e LLM_FALLBACK_CONFIG="$LLM_FALLBACK_CONFIG" \
  letta/letta:nemurin
