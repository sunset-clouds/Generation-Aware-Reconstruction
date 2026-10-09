#!/usr/bin/env bash
# Compatibility entrypoint; use compute_correlations.sh.
set -euo pipefail
exec bash "$(dirname "$0")/compute_correlations.sh" "$@"
