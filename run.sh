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

echo "[Letta] Docker 컨테이너 시작..."
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  --add-host=host.docker.internal:host-gateway \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e GEMINI_API_KEYS="$GEMINI_API_KEYS" \
  letta/letta:nemurin
