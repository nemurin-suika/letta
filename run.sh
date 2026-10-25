#!/bin/bash

REAL_HOME="${MIZUKI_REAL_HOME:-~}"
AI_HOME="${MIZUKI_AI_HOME:-${REAL_HOME}/ai}"
export HOME="$AI_HOME"
export CODEX_HOME="${CODEX_HOME:-${REAL_HOME}/.codex}"

# Optional Gemini variables are read from the environment only. The current
# setup uses ollama/bge-m3 for embeddings, so no secrets file is required.
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
GEMINI_API_KEYS="${GEMINI_API_KEYS:-$GEMINI_API_KEY}"

if [ -z "${LLM_FALLBACK_CONFIG:-}" ]; then
    LLM_FALLBACK_CONFIG='{"model": "gpt-5.5", "model_endpoint_type": "codex_cli", "context_window": 128000}'
fi

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
# Runtime data stays under ~/ai so the company laptop setup does not write to
# the real home directory. CODEX_HOME still points at the real Codex config.
PG_DATA="${MIZUKI_PG_DATA:-$HOME/.mizuki/pgdata}"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_DB="postgres"
PG_PORT="5432"

PGDATA_DIR="$HOME/.letta/.persist/pgdata"  # Docker 모드용 데이터 디렉터리
PG_CONTAINER="letta-postgres"
UV_RUN=(uv run --extra postgres --extra redis --extra server)

# Kept for compatibility if the model is switched back to claude_cli.
export VTUBER_CLAUDE_HOOK_SETTINGS=~/ai/Open-LLM-VTuber-v1.2.1-en/scripts/claude_hook_settings.json
export VTUBER_APPROVAL_URL=http://127.0.0.1:12393/tool-approval


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
    mkdir -p "$HOME/.letta/logs/claude_cli_debug"
    mkdir -p "$HOME/.letta/logs/codex_cli_debug"

    # ── Codex CLI 확인 ─────────────────────────────────────────────────────
    if ! command -v codex &>/dev/null; then
        echo "[Codex] ERROR: 'codex' CLI를 찾을 수 없습니다. Codex CLI를 설치하세요."
        exit 1
    fi
    codex --version </dev/null
    echo "[Codex] CODEX_HOME=${CODEX_HOME}"

    # ── PostgreSQL 시작 (Homebrew 네이티브) ────────────────────────────────
    if pg_ctl -D "$PG_DATA" status >/dev/null 2>&1; then
        echo "[DB] PostgreSQL 이미 실행 중"
    else
        if [ ! -s "$PG_DATA/PG_VERSION" ]; then
            echo "[DB] PostgreSQL 데이터 디렉터리 초기화 중: $PG_DATA"
            mkdir -p "$PG_DATA"
            initdb -D "$PG_DATA" --encoding=UTF8 --locale=en_US.UTF-8
        fi
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

    # ~/.letta/pg_uri 파일로는 pydantic-settings가 불안정하므로 환경변수로 명시
    # sslmode=disable: 로컬 접속 시 db.py가 ssl=require를 강제하는 것을 방지
    export LETTA_PG_URI="postgresql://${PG_USER}:${PG_PASSWORD}@localhost:${PG_PORT}/${PG_DB}"

    # ── DB 마이그레이션 ─────────────────────────────────────────────────────
    echo "[DB] 데이터베이스 마이그레이션 중..."
    "${UV_RUN[@]}" alembic upgrade head

    # ── Redis 시작 ──────────────────────────────────────────────────────────
    REDIS_HOST="127.0.0.1"
    REDIS_PORT="6379"
    if ! command -v redis-server &>/dev/null; then
        echo "[Redis] ERROR: redis-server를 찾을 수 없습니다. 'brew install redis' 로 설치하세요."
        exit 1
    fi
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
        echo "[Redis] 이미 실행 중 (${REDIS_HOST}:${REDIS_PORT})"
    else
        echo "[Redis] 시작 중..."
        mkdir -p "$HOME/.letta/logs"
        redis-server --daemonize yes \
            --bind "$REDIS_HOST" \
            --port "$REDIS_PORT" \
            --logfile "${HOME}/.letta/logs/redis.log" \
            --loglevel notice
        # 준비될 때까지 대기 (최대 10초)
        for _i in $(seq 1 20); do
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG" && break
            sleep 0.5
        done
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG" \
            && echo "[Redis] 시작 완료" \
            || { echo "[Redis] ERROR: 시작 실패. $HOME/.letta/logs/redis.log 확인"; exit 1; }
    fi

    # ── CPU 코어 수 자동 감지 → 워커 수 설정 ─────────────────────────────────
    _WORKERS=1
    echo "[Letta] 워커 수: ${_WORKERS}"

    # ── Letta 서버 시작 ─────────────────────────────────────────────────────
    export LETTA_DIR="${MIZUKI_LETTA_DIR:-${HOME}/.mizuki/letta}"
    mkdir -p "$LETTA_DIR"
    export GEMINI_API_KEY="$GEMINI_API_KEY"
    export GEMINI_API_KEYS="$GEMINI_API_KEYS"
    export LLM_FALLBACK_CONFIG="$LLM_FALLBACK_CONFIG"
    # Logs are local to the company laptop setup.
    LOG_DIR="${HOME}/.letta/logs"
    export LETTA_WORKER_LOG_DIR="${HOME}/.letta/logs"
    export LETTA_CLAUDE_CLI_DEBUG_DIR="${HOME}/.letta/logs/claude_cli_debug"
    export LETTA_CODEX_CLI_DEBUG_DIR="${HOME}/.letta/logs/codex_cli_debug"
    export LETTA_REDIS_HOST="$REDIS_HOST"
    export LETTA_REDIS_PORT="$REDIS_PORT"
    export LETTA_UVICORN_WORKERS="${_WORKERS}"

    mkdir -p "$LOG_DIR"
    _START_TIME=$(date +%Y%m%d_%H%M%S)
    export LETTA_START_TIME="$_START_TIME"
    echo "[Letta] 로컬 서버 시작 (http://0.0.0.0:8283, workers=${_WORKERS}, redis=${REDIS_HOST}:${REDIS_PORT})..."
    echo "[Letta] 워커 로그 경로: ${LOG_DIR}/letta-worker-${_START_TIME}-<PID>.log"
    exec "${UV_RUN[@]}" letta server --host 0.0.0.0 --port 8283 2>&1 | tee -a "${LOG_DIR}/main-${_START_TIME}.log"

# =============================================================================
else
# ── Docker 실행 모드 (기존 동작) ──────────────────────────────────────────────

    echo "[Letta] ERROR: Docker mode is not configured for codex_cli. Use ./run.sh --local."
    exit 1

fi
