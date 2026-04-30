#!/usr/bin/env bash
set -euo pipefail

MODEL="dolphin3:latest"
OLLAMA_PORT=11434

# ollama 설치 확인
if ! command -v ollama &>/dev/null; then
    echo "[ERROR] ollama가 설치되어 있지 않습니다."
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# 이미 실행 중이면 종료
if curl -sf "http://localhost:${OLLAMA_PORT}" &>/dev/null; then
    echo "[INFO] 실행 중인 ollama 서버를 종료합니다..."
    pkill -x ollama 2>/dev/null || true
    # 종료 대기 (최대 10초)
    for i in $(seq 1 10); do
        if ! curl -sf "http://localhost:${OLLAMA_PORT}" &>/dev/null; then
            break
        fi
        sleep 1
    done
    echo "[INFO] ollama 서버 종료 완료"
fi

# 서버 시작
echo "[INFO] ollama 서버 시작 중..."
ollama serve &
OLLAMA_PID=$!

# 서버 준비 대기 (최대 15초)
for i in $(seq 1 15); do
    if curl -sf "http://localhost:${OLLAMA_PORT}" &>/dev/null; then
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:${OLLAMA_PORT}" &>/dev/null; then
    echo "[ERROR] ollama 서버 시작 실패"
    exit 1
fi
echo "[INFO] ollama 서버 시작 완료 (PID: ${OLLAMA_PID})"

# 모델 다운로드 (없는 경우에만)
if ollama list | grep -q "^${MODEL}"; then
    echo "[INFO] 모델 ${MODEL} 이미 존재합니다"
else
    echo "[INFO] 모델 ${MODEL} 다운로드 중..."
    ollama pull "${MODEL}"
fi

echo ""
echo "[INFO] 준비 완료"
echo "  모델    : ${MODEL}"
echo "  엔드포인트: http://localhost:${OLLAMA_PORT}/v1"
echo ""
echo "  Letta 환경변수:"
echo "    LLM_FALLBACK_CONFIG='{\"model\": \"${MODEL}\", \"model_endpoint_type\": \"openai\", \"model_endpoint\": \"http://localhost:${OLLAMA_PORT}/v1\", \"context_window\": 32768}'"
echo ""
echo "[INFO] Ctrl+C 로 종료"
wait "${OLLAMA_PID}"
