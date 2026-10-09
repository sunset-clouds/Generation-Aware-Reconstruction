#!/usr/bin/env bash
# Compatibility entrypoint; use download_hf_checkpoint.sh.
set -euo pipefail
exec bash "$(dirname "$0")/download_hf_checkpoint.sh" "$@"
