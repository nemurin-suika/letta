#!/bin/bash

BASE_URL="http://localhost:8283/v1/agents/"

if [ -n "$1" ]; then
    curl -v "${BASE_URL}$1" -X DELETE
else
    echo "agent ID가 없어 모든 agent를 조회하고 삭제합니다..."
    AGENT_IDS=$(curl -s "${BASE_URL}" | jq -r '.[].id')

    if [ -z "$AGENT_IDS" ]; then
        echo "삭제할 agent가 없습니다."
        exit 0
    fi

    echo "삭제 대상 agent IDs:"
    echo "$AGENT_IDS"
    echo ""

    echo "$AGENT_IDS" | while read -r AGENT_ID; do
        echo "삭제 중: $AGENT_ID"
        curl -s -X DELETE "${BASE_URL}${AGENT_ID}"
        echo "완료: $AGENT_ID"
    done

    echo "모든 agent 삭제 완료."
fi