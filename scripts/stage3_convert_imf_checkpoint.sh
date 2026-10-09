#!/usr/bin/env bash
# Compatibility entrypoint; use convert_imf_checkpoint.sh.
set -euo pipefail
exec bash "$(dirname "$0")/convert_imf_checkpoint.sh" "$@"
