#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
cp -a .web-node_modules/. web/node_modules/
export JEV_DEV_BACKEND_URL=http://127.0.0.1:8341
export XDG_CACHE_HOME="/tmp/jev-cache-$(id -u)"
mkdir -p "$XDG_CACHE_HOME"

BACKEND_PID=""
FRONTEND_PID=""
stop_children() {
  trap - TERM INT
  for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
    if [[ -n "$pid" ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
trap stop_children EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

python -m uvicorn jev_mujoco.app:create_app --factory --host 127.0.0.1 --port 8341 \
  --reload --reload-dir "$PROJECT_DIR/src" --reload-dir "$PROJECT_DIR/skills" \
  --reload-include '*.py' --reload-include '*.md' --reload-include '*.json' \
  --timeout-graceful-shutdown 10 &
BACKEND_PID=$!
(
  cd web
  exec node node_modules/vite/bin/vite.js --host 0.0.0.0 --port 8340 --strictPort
) &
FRONTEND_PID=$!

# A web service exiting must not leave a half-working container behind.
if wait -n "$BACKEND_PID" "$FRONTEND_PID"; then
  echo "网页进程意外退出，停止网页容器。" >&2
  exit 1
else
  exit "$?"
fi
