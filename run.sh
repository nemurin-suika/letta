#!/bin/bash

source ~/.secrets

# Gemini API 키 (embedding 모델용) - 429 발생 시 자동으로 다음 키로 전환됨
GEMINI_API_KEY="${GEMINI_KEYS[0]}"
GEMINI_API_KEYS="$(IFS=,; echo "${GEMINI_KEYS[*]}")"

LLM_FALLBACK_CONFIG='{"model": "claude-cli", "model_endpoint_type": "claude_cli", "context_window": 72000}'

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
PG_DATA="/opt/homebrew/var/postgresql@17"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_DB="postgres"
PG_PORT="5432"

PGDATA_DIR="$HOME/.letta/.persist/pgdata"  # Docker 모드용 데이터 디렉터리
PG_CONTAINER="letta-postgres"

# ── 실행 모드 파싱 ────────────────────────────────────────────────────────────
# ./run.sh          → Docker 모드 (기본)
# ./run.sh --local  → 로컬 직접 실행 모드
MODE="local"
for arg in "$@"; do
    case $arg in
        --local)  MODE="local"  ;;
        --docker) MODE="docker" ;;
    esac
done

# =============================================================================
if [ "$MODE" = "local" ]; then
# ── 로컬 직접 실행 모드 ───────────────────────────────────────────────────────

    export PATH="$PG_BIN:$PATH"
    mkdir -p ~/.letta/logs/claude_cli_debug

    # ── Claude CLI 로그인 확인 ──────────────────────────────────────────────
    if ! command -v claude &>/dev/null; then
        echo "[Claude] ERROR: 'claude' CLI를 찾을 수 없습니다. Claude Code CLI를 설치하세요."
        exit 1
    fi
    _CLAUDE_OUT=$(claude -p "ping" --output-format json 2>/dev/null)
    if echo "$_CLAUDE_OUT" | grep -qi "not logged in"; then
        echo "[Claude] 로그인이 필요합니다..."
        claude login
        echo "[Claude] 로그인 완료"
    else
        echo "[Claude] 로그인 확인됨"
    fi

    # ── PostgreSQL 시작 (Homebrew 네이티브) ────────────────────────────────
    if pg_ctl -D "$PG_DATA" status >/dev/null 2>&1; then
        echo "[DB] PostgreSQL 이미 실행 중"
    else
        echo "[DB] PostgreSQL 시작 중..."
        pg_ctl -D "$PG_DATA" -l "$PG_DATA/server.log" start
        sleep 1
    fi

    # postgres 유저/DB/익스텐션이 없으면 초기 설정
    psql -U "$PG_USER" -d "$PG_DB" -c "SELECT 1" >/dev/null 2>&1 || {
        echo "[DB] 초기 설정 중..."
        psql postgres -c "CREATE USER $PG_USER WITH SUPERUSER PASSWORD '$PG_PASSWORD';" 2>/dev/null || true
        psql postgres -c "CREATE DATABASE $PG_DB OWNER $PG_USER;" 2>/dev/null || true
        psql -U "$PG_USER" -d "$PG_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
    }

    # PG 접속 정보는 ~/.letta/pg_uri 파일에서 자동으로 로드됨

    # ── DB 마이그레이션 ─────────────────────────────────────────────────────
    echo "[DB] 데이터베이스 마이그레이션 중..."
    uv run alembic upgrade head

    # ── Redis 없음 → 단일 워커 모드 (NoopAsyncRedisClient 자동 사용) ────────
    echo "[Redis] Redis 미사용 → 단일 워커 모드로 실행"

    # ── Letta 서버 시작 ─────────────────────────────────────────────────────
    export GEMINI_API_KEY="$GEMINI_API_KEY"
    export GEMINI_API_KEYS="$GEMINI_API_KEYS"
    export LLM_FALLBACK_CONFIG="$LLM_FALLBACK_CONFIG"
    export LETTA_CLAUDE_CLI_DEBUG_DIR="${HOME}/.letta/logs/claude_cli_debug"

    echo "[Letta] 로컬 서버 시작 (http://0.0.0.0:8283)..."
    exec uv run letta server --host 0.0.0.0 --port 8283

# =============================================================================
else
# ── Docker 실행 모드 (기존 동작) ──────────────────────────────────────────────

    mkdir -p "$PGDATA_DIR"
    mkdir -p ~/.letta/logs/claude_cli_debug

    # Claude CLI 로그인 확인
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

    # --add-host=host.docker.internal:host-gateway
    #   macOS Docker Desktop: host.docker.internal은 자동으로 사용 가능하지만 명시적으로 추가
    #   Linux: 이 옵션 없이는 host.docker.internal이 컨테이너에서 사용 불가
    echo "[Letta] Docker 컨테이너 시작..."
    docker run \
      -v "$PGDATA_DIR:/var/lib/postgresql/data" \
      -v letta-claude-auth:/root/.claude \
      -v ~/.letta/logs/claude_cli_debug:/app/logs/claude_cli_debug \
      -p 8283:8283 \
      --add-host=host.docker.internal:host-gateway \
      -e GEMINI_API_KEY="$GEMINI_API_KEY" \
      -e GEMINI_API_KEYS="$GEMINI_API_KEYS" \
      -e LLM_FALLBACK_CONFIG="$LLM_FALLBACK_CONFIG" \
      letta/letta:nemurin

fi
