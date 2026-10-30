#!/bin/bash

# ── 머신별 설정 ───────────────────────────────────────────────────────────────
# 기본값은 집(맥미니 / claude_cli). 다른 머신은 이 파일을 고치지 말고
# ~/.mizuki_env 를 만들어 아래 변수만 덮어쓴다. 리포 밖 파일이라 git 충돌 없음.
#
#   MIZUKI_REAL_HOME            실제 홈 (기본: 현재 HOME)
#   MIZUKI_AI_HOME              런타임 데이터 홈 (기본: REAL_HOME)
#   MIZUKI_SERVICE_HOME         서비스 HOME. 있으면 MIZUKI_AI_HOME 보다 우선.
#                               _mz_setup/letta.sh 는 MIZUKI_AI_HOME 을 리포 루트(~/develop)
#                               로 넘기므로, 집(데이터가 실제 홈)은 plist 에서 이 값을 준다.
#   MIZUKI_LLM_CLI              claude | codex (기본: claude)
#   MIZUKI_SECRETS_FILE         시크릿 파일 (기본: REAL_HOME/.secrets, 없으면 생략)
#   LLM_FALLBACK_CONFIG         폴백 모델 JSON
#   VTUBER_CLAUDE_HOOK_SETTINGS 툴 승인 hook 설정 파일
#   VTUBER_APPROVAL_URL         툴 승인 서버 주소
#   CODEX_HOME                  codex 설정 디렉터리 (codex 사용 시)
#
# 회사 노트북 예시:
#   MIZUKI_REAL_HOME=~
#   MIZUKI_AI_HOME=~/ai
#   MIZUKI_LLM_CLI=codex
#   LLM_FALLBACK_CONFIG='{"model": "gpt-5.5", "model_endpoint_type": "codex_cli", "context_window": 128000}'
#   VTUBER_CLAUDE_HOOK_SETTINGS=~/ai/Open-LLM-VTuber-v1.2.1-en/scripts/claude_hook_settings.json
#   VTUBER_APPROVAL_URL=http://127.0.0.1:12393/tool-approval
[ -r "$HOME/.mizuki_env" ] && . "$HOME/.mizuki_env"

REAL_HOME="${MIZUKI_REAL_HOME:-$HOME}"
AI_HOME="${MIZUKI_SERVICE_HOME:-${MIZUKI_AI_HOME:-$REAL_HOME}}"
export HOME="$AI_HOME"
export CODEX_HOME="${CODEX_HOME:-${REAL_HOME}/.codex}"
MIZUKI_LLM_CLI="${MIZUKI_LLM_CLI:-claude}"

# Gemini API 키 (embedding 폴백용) - 429 발생 시 자동으로 다음 키로 전환됨.
# 현재 임베딩은 ollama/bge-m3 라서 없어도 동작한다.
MIZUKI_SECRETS_FILE="${MIZUKI_SECRETS_FILE:-${REAL_HOME}/.secrets}"
if [ -r "$MIZUKI_SECRETS_FILE" ]; then
    . "$MIZUKI_SECRETS_FILE"
    if [ -n "${GEMINI_KEYS:-}" ]; then
        GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_KEYS[0]}}"
        GEMINI_API_KEYS="${GEMINI_API_KEYS:-$(IFS=,; echo "${GEMINI_KEYS[*]}")}"
    fi
fi
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
GEMINI_API_KEYS="${GEMINI_API_KEYS:-$GEMINI_API_KEY}"

if [ -z "${LLM_FALLBACK_CONFIG:-}" ]; then
    LLM_FALLBACK_CONFIG='{"model": "claude-cli", "model_endpoint_type": "claude_cli", "context_window": 72000}'
fi

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
# ★ 데이터 경로는 OneDrive 밖(로컬)이어야 한다 ★
# launchd 로 기동되는 프로세스는 OneDrive 파일 제공자 영역을 읽을 수 없다.
#   pg_ctl -D <OneDrive 경로> status  → 영구 블록(응답 없음) 실측 확인
# 그래서 pgdata/LETTA_DIR 은 ~/.mizuki 아래로 옮기고, OneDrive 에는
# 1시간 주기 pg_dump 백업만 올린다(moe.nemurin.mizuki-backup).
# 회사 노트북은 MIZUKI_AI_HOME 으로 HOME 을 ~/ai 로 돌려 실제 홈을 건드리지 않는다.
PG_DATA="${MIZUKI_PG_DATA:-$HOME/.mizuki/pgdata}"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_DB="postgres"
PG_PORT="5432"

PGDATA_DIR="$HOME/.letta/.persist/pgdata"  # Docker 모드용 데이터 디렉터리
PG_CONTAINER="letta-postgres"
UV_RUN=(uv run --extra postgres --extra redis --extra server)

# export 필수 — CLI 서브프로세스(claude_cli_client.py / codex_cli_client.py 가
# os.environ.get 으로 읽음)까지 전달돼야 hook 이 붙는다.
export VTUBER_CLAUDE_HOOK_SETTINGS="${VTUBER_CLAUDE_HOOK_SETTINGS:-${REAL_HOME}/develop/Open-LLM-VTuber-v1.2.1-en/scripts/claude_hook_settings.json}"
# 승인 서버 주소 — 집 백엔드는 conf.yaml host(192.168.0.5)에만 바인드되므로
# 127.0.0.1 로는 연결되지 않는다. 백엔드 host 를 바꾸면 이 값도 같이 맞춘다.
export VTUBER_APPROVAL_URL="${VTUBER_APPROVAL_URL:-http://192.168.0.5:12393/tool-approval}"


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

    # ── LLM CLI 확인 (MIZUKI_LLM_CLI = claude | codex) ─────────────────────
    if [ "$MIZUKI_LLM_CLI" = "codex" ]; then
        if ! command -v codex &>/dev/null; then
            echo "[Codex] ERROR: 'codex' CLI를 찾을 수 없습니다. Codex CLI를 설치하세요."
            exit 1
        fi
        codex --version </dev/null
        echo "[Codex] CODEX_HOME=${CODEX_HOME}"
    else
        if ! command -v claude &>/dev/null; then
            echo "[Claude] ERROR: 'claude' CLI를 찾을 수 없습니다. Claude Code CLI를 설치하세요."
            exit 1
        fi
        # `claude -p "ping"`은 stdin이 TTY면 EOF를 기다리며 hang된다. 게다가 실제 API 호출이라
        # 매 기동마다 CLAUDE.md 전체가 컨텍스트로 올라가 수 초 + 실비가 나간다.
        # `claude auth status --json`은 네트워크/토큰 소모 없이 즉시 로그인 여부만 반환한다.
        _CLAUDE_AUTH=$(claude auth status --json </dev/null 2>/dev/null)
        if echo "$_CLAUDE_AUTH" | grep -q '"loggedIn"[[:space:]]*:[[:space:]]*true'; then
            echo "[Claude] 로그인 확인됨"
        else
            echo "[Claude] 로그인이 필요합니다..."
            claude auth login
            echo "[Claude] 로그인 완료"
        fi
    fi

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

    # Docker 모드는 현재 구성에서 쓰지 않는다(집·회사 둘 다 --local).
    # 과거 docker run 블록은 git 히스토리와 _mz_backup/letta-run.sh.home-20261026 에 있다.
    echo "[Letta] ERROR: Docker 모드는 현재 구성에서 지원하지 않습니다. ./run.sh --local 을 쓰세요."
    exit 1

fi
