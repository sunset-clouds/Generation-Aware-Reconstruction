#!/usr/bin/env bash
# Compatibility entrypoint; use download_sit_checkpoints.sh.
set -euo pipefail
exec bash "$(dirname "$0")/download_sit_checkpoints.sh" "$@"
