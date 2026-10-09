#!/usr/bin/env bash
# Compatibility entrypoint; use smoke_test_garfid.sh.
set -euo pipefail
exec bash "$(dirname "$0")/smoke_test_garfid.sh" "$@"
