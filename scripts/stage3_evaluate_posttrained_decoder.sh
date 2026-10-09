#!/usr/bin/env bash
# Compatibility entrypoint; use evaluate_decoder.sh.
set -euo pipefail
exec bash "$(dirname "$0")/evaluate_decoder.sh" "$@"
