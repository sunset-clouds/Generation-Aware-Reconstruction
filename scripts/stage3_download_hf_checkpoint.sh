#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}

REPO_ID=${1:?usage: stage3_download_hf_checkpoint.sh HF_REPO_ID HF_FILENAME OUTPUT_PATH}
FILENAME=${2:?usage: stage3_download_hf_checkpoint.sh HF_REPO_ID HF_FILENAME OUTPUT_PATH}
OUTPUT_PATH=${3:?usage: stage3_download_hf_checkpoint.sh HF_REPO_ID HF_FILENAME OUTPUT_PATH}

"${PYTHON}" - "$REPO_ID" "$FILENAME" "$OUTPUT_PATH" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id, filename, output_path = sys.argv[1:4]
target = Path(output_path).expanduser()
target.parent.mkdir(parents=True, exist_ok=True)
cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"))
shutil.copy2(cached_path, target)
print(f"Saved {repo_id}/{filename} -> {target}")
PY
