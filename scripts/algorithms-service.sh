#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Usage: scripts/algorithms-service.sh {start|stop|status|logs}' >&2
  exit 2
fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/demo-service.sh" "$1" algorithms
