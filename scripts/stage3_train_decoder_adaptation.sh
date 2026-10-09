#!/usr/bin/env bash
# Compatibility entrypoint; use train_decoder_adaptation.sh.
set -euo pipefail
exec bash "$(dirname "$0")/train_decoder_adaptation.sh" "$@"
