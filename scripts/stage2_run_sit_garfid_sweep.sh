#!/usr/bin/env bash
# Compatibility entrypoint; use evaluate_garfid_sweep.sh.
set -euo pipefail
exec bash "$(dirname "$0")/evaluate_garfid_sweep.sh" "$@"
