#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  echo "用法: $0 {start [0.4.0-experiments-guangyu]|build [0.4.0-experiments-guangyu]|stop|status|logs}" >&2
}
ACTION="${1:-}"
case "$ACTION" in
  start|build) ;;
  stop|status|logs) [[ $# -eq 1 ]] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac
shift
DEV_ALIAS=0
TAG_SET=0
export JEV_WEB_IMAGE_TAG="0.4.0-experiments-guangyu"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      # Compatibility only: source reload is always enabled.
      [[ $DEV_ALIAS -eq 0 ]] || { usage; exit 2; }
      DEV_ALIAS=1
      ;;
    -*) usage; exit 2 ;;
    *)
      [[ $TAG_SET -eq 0 ]] || { usage; exit 2; }
      JEV_WEB_IMAGE_TAG="$1"
      TAG_SET=1
      ;;
  esac
  shift
done
if [[ ! "$JEV_WEB_IMAGE_TAG" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ || "$JEV_WEB_IMAGE_TAG" != *guangyu* ]]; then
  echo "镜像版本必须是包含 guangyu 的有效 Docker tag。" >&2
  exit 2
fi
export JEV_UID="$(id -u)" JEV_GID="$(id -g)"
cd "$PROJECT_DIR"
COMPOSE=(docker compose -f "$PROJECT_DIR/deploy/web.compose.yaml")
WEB_IMAGE="tianji-jev-mujoco:$JEV_WEB_IMAGE_TAG"

case "$ACTION" in
  start)
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
      echo "缺少 $PROJECT_DIR/.env；请参考 .env.example 配置 TYPESAFE_API_KEY。" >&2
      exit 1
    fi
    mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/web/node_modules"
    mkdir -p "${JEV_EXPERIMENT_DIR:-/home/guangyu/projects/tianji-experiments}"
    if ! docker image inspect "$WEB_IMAGE" >/dev/null 2>&1; then
      echo "缺少 $WEB_IMAGE；启动不会自动构建或下载镜像。需要构建时请明确使用 build。" >&2
      exit 1
    fi
    # One-time migration from the former two-container web deployment.
    LEGACY_FRONTEND="jev-mujoco-frontend-dev-guangyu"
    if docker container inspect "$LEGACY_FRONTEND" >/dev/null 2>&1; then
      LEGACY_PROJECT="$(docker container inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$LEGACY_FRONTEND")"
      if [[ "$LEGACY_PROJECT" != "tianji-jev-mujoco-web-guangyu" ]]; then
        echo "旧前端容器 $LEGACY_FRONTEND 的归属不匹配，未删除；无法占用网页端口 8340。" >&2
        exit 1
      fi
      docker rm -f "$LEGACY_FRONTEND" >/dev/null
    fi
    if ! "${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 120 web; then
      "${COMPOSE[@]}" logs --tail 80 web >&2
      exit 1
    fi
    echo "网页和 MuJoCo 已在 Docker 中启动：http://192.168.3.4:8340/"
    echo "前端自动更新；后端代码重载会重置当前仿真和任务，参数与日志保留。"
    ;;
  build) "${COMPOSE[@]}" build web ;;
  stop) "${COMPOSE[@]}" stop web ;;
  status) "${COMPOSE[@]}" ps web ;;
  logs) "${COMPOSE[@]}" logs --follow --tail 100 web ;;
esac
