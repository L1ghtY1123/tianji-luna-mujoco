#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: scripts/demo-service.sh {start|stop|status|logs} [all|web|algorithms]' >&2
}
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi
ACTION="$1"
case "$ACTION" in
  start|stop|status|logs) ;;
  *) usage; exit 2 ;;
esac
GROUP="${2:-all}"
case "$GROUP" in
  all) SERVICES=(web curobo sam3 graspgen foundationpose) ;;
  web) SERVICES=(web) ;;
  algorithms) SERVICES=(curobo sam3 graspgen foundationpose) ;;
  *) usage; exit 2 ;;
esac

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LUNA_UID="$(id -u)" LUNA_GID="$(id -g)"
cd "$PROJECT_DIR"
COMPOSE=(docker compose --env-file "$PROJECT_DIR/.env" -f "$PROJECT_DIR/deploy/demo.compose.yaml")
case "$ACTION" in
  start)
    [[ -f .env ]] || cp .env.example .env
    mkdir -p data web/node_modules
    "${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 300 "${SERVICES[@]}"
    if [[ "$GROUP" == algorithms ]]; then
      echo 'Tianji Luna algorithms ready: curobo sam3 graspgen foundationpose'
    else
      echo 'Tianji Luna MuJoCo: http://192.168.3.4:8350/'
    fi
    ;;
  stop) "${COMPOSE[@]}" stop "${SERVICES[@]}" ;;
  status) "${COMPOSE[@]}" ps "${SERVICES[@]}" ;;
  logs) "${COMPOSE[@]}" logs --tail 80 --follow "${SERVICES[@]}" ;;
esac
