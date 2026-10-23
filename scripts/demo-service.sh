#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LUNA_UID="$(id -u)" LUNA_GID="$(id -g)"
cd "$PROJECT_DIR"
COMPOSE=(docker compose -f "$PROJECT_DIR/deploy/demo.compose.yaml")
case "${1:-}" in
  start)
    [[ -f .env ]] || cp .env.example .env
    mkdir -p data web/node_modules
    "${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 300
    echo 'Tianji Luna MuJoCo: http://192.168.3.4:8350/'
    ;;
  stop) "${COMPOSE[@]}" stop ;;
  status) "${COMPOSE[@]}" ps ;;
  logs) "${COMPOSE[@]}" logs --tail 80 --follow ;;
  *) echo 'Usage: scripts/demo-service.sh {start|stop|status|logs}' >&2; exit 2 ;;
esac
