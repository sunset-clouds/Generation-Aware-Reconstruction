#!/usr/bin/env bash
# Compatibility entrypoint; use evaluate_garfid.sh.
set -euo pipefail
exec bash "$(dirname "$0")/evaluate_garfid.sh" "$@"
