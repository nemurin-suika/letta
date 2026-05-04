#!/bin/bash
# PostgreSQL 데이터 전체 초기화 (데이터 디렉터리 삭제 후 재생성)

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
PG_DATA="/opt/homebrew/var/postgresql@17"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_DB="postgres"

export PATH="$PG_BIN:$PATH"

echo "[reset] PostgreSQL 정지..."
pg_ctl -D "$PG_DATA" stop 2>/dev/null || true

echo "[reset] 데이터 디렉터리 삭제: $PG_DATA"
rm -rf "$PG_DATA"

echo "[reset] initdb로 재초기화..."
initdb --locale=en_US.UTF-8 -E UTF-8 "$PG_DATA"

echo "[reset] PostgreSQL 시작..."
pg_ctl -D "$PG_DATA" -l "$PG_DATA/server.log" start
sleep 1

echo "[reset] postgres 유저/DB/pgvector 설정..."
psql postgres -c "CREATE USER $PG_USER WITH SUPERUSER PASSWORD '$PG_PASSWORD';" 2>/dev/null || true
psql postgres -c "CREATE DATABASE $PG_DB OWNER $PG_USER;" 2>/dev/null || true
psql -U "$PG_USER" -d "$PG_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "[reset] 완료. ./run.sh 로 서버를 다시 시작하세요."
