#!/usr/bin/env bash
set -euo pipefail
# This independent copy owns only the demo Compose project.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/demo-service.sh" "$@"
