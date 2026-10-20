#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "用法: $0 {start [0.2.0-guangyu]|stop|status|logs}" >&2
}
ACTION="${1:-}"
case "$ACTION" in
  start) [[ $# -le 2 ]] || { usage; exit 2; } ;;
  stop|status|logs) [[ $# -eq 1 ]] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac
export JEV_ALGORITHM_IMAGE_TAG="${2:-0.2.0-guangyu}"
export JEV_CUROBO_IMAGE_TAG="${JEV_CUROBO_IMAGE_TAG:-0.2.0-guangyu}"
if [[ ! "$JEV_ALGORITHM_IMAGE_TAG" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ || "$JEV_ALGORITHM_IMAGE_TAG" != *guangyu* ]]; then
  echo "镜像版本必须是包含 guangyu 的有效 Docker tag。" >&2
  exit 2
fi
if [[ ! "$JEV_CUROBO_IMAGE_TAG" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ || "$JEV_CUROBO_IMAGE_TAG" != *guangyu* ]]; then
  echo "cuRobo 镜像版本必须是包含 guangyu 的有效 Docker tag。" >&2
  exit 2
fi
cd "$PROJECT_DIR"
COMPOSE=(docker compose -f "$PROJECT_DIR/deploy/algorithms.compose.yaml")

case "$ACTION" in
  start)
    if ! docker image inspect "jodell-curobo:$JEV_CUROBO_IMAGE_TAG" >/dev/null 2>&1; then
      echo "缺少 jodell-curobo:$JEV_CUROBO_IMAGE_TAG；启动不会自动构建或下载镜像。" >&2
      exit 1
    fi
    if ! "${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 300 sam3 curobo graspgen; then
      "${COMPOSE[@]}" logs --tail 80 sam3 curobo graspgen >&2
      exit 1
    fi
    ;;
  stop) "${COMPOSE[@]}" stop sam3 curobo graspgen ;;
  status) "${COMPOSE[@]}" ps sam3 curobo graspgen ;;
  logs) "${COMPOSE[@]}" logs --follow --tail 100 sam3 curobo graspgen ;;
esac
