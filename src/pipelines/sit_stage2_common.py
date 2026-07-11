from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT
WORKSPACE_ROOT = PROJECT_ROOT
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "assets" / "sit_checkpoint_registry_template.csv"
DEFAULT_IFID_REPO_ROOT = PROJECT_ROOT / "third_party" / "ifid"
REQUIRED_IFID_RUNTIME_FILES = (
    "ifid/sit/sit.py",
    "ifid/vae/utils.py",
)
REQUIRED_CONTINUOUS_RUNTIME_FILES = (
    "ifid/vae/cont.py",
    "cont/modelling/tokenizer.py",
    "cont/modelling/jet.py",
)
SHIFTED_REGISTRY_KEYS = [
    "imagenet_train_dir",
    "imagenet_val_dir",
    "label_map_json",
    "fid_reference_file",
    "load_policy",
    "load_check",
    "inference_check",
    "rfid_check",
    "ifid_check",
    "gfid_check",
    "garfid_check",
    "pcc_srcc_check",
    "notes",
]


def normalize_registry_row(row: Dict[str, str]) -> Dict[str, str]:
    normalized = dict(row)
    extra_values = normalized.get(None) or []
    looks_shifted = bool(extra_values) and (
        str(normalized.get("load_policy", "")).endswith(".npz")
        or str(normalized.get("fid_reference_file", "")).endswith(".json")
    )
    if looks_shifted:
        current_values = [str(normalized.get(key, "")) for key in SHIFTED_REGISTRY_KEYS]
        shifted_values = current_values[1:] + [str(extra_values[0])]
        for key, value in zip(SHIFTED_REGISTRY_KEYS, shifted_values):
            normalized[key] = value
        normalized["__normalized_misaligned_csv"] = "true"
    normalized.pop(None, None)
    return normalized


def load_registry_rows(registry_path: Optional[str] = None) -> List[Dict[str, str]]:
    path = Path(registry_path or DEFAULT_REGISTRY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Registry CSV not found: {path}")
    with path.open("r", newline="") as handle:
        return [normalize_registry_row(row) for row in csv.DictReader(handle)]


def truthy_csv(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "run"}


def row_should_run(row: Dict[str, str]) -> bool:
    return truthy_csv(row.get("should_run", "yes"))


def normalize_group_ids(group_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not group_ids:
        return None
    normalized = {str(group_id).strip() for group_id in group_ids if str(group_id).strip()}
    return normalized or None


def filter_rows(
    rows: List[Dict[str, str]],
    *,
    group_ids: Optional[Iterable[str]] = None,
    only_should_run: bool = False,
) -> List[Dict[str, str]]:
    normalized_group_ids = normalize_group_ids(group_ids)
    filtered = []
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        if normalized_group_ids and group_id not in normalized_group_ids:
            continue
        if only_should_run and not row_should_run(row):
            continue
        filtered.append(row)
    return filtered


def resolve_user_path(path_value: str, extra_roots: Optional[Iterable[Path]] = None) -> str:
    if not path_value:
        return path_value
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())
    search_roots = [PROJECT_ROOT, SRC_ROOT]
    if extra_roots:
        search_roots.extend(extra_roots)
    for root in search_roots:
        combined = (root / path_value).resolve()
        if combined.exists():
            return str(combined)
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate)


def resolve_reference_path(path_value: str, ifid_repo_root: str) -> str:
    resolved = resolve_user_path(
        path_value,
        extra_roots=[
            Path(ifid_repo_root),
            Path(ifid_repo_root) / "metric",
            SRC_ROOT / "metric",
            PROJECT_ROOT / "metric",
        ],
    )
    return resolved


def resolve_stage2_vae_config_path(
    path_value: str,
    *,
    ifid_repo_root: str,
    exp_path: str = "",
) -> str:
    extra_roots = [
        Path(ifid_repo_root),
        Path(ifid_repo_root) / "configs",
        SRC_ROOT,
        PROJECT_ROOT,
    ]
    if exp_path:
        extra_roots.insert(0, Path(exp_path))
    return resolve_user_path(path_value, extra_roots=extra_roots)


def missing_ifid_runtime_files(ifid_repo_root: str) -> List[str]:
    root = Path(resolve_user_path(ifid_repo_root))
    if not root.exists():
        return list(REQUIRED_IFID_RUNTIME_FILES)
    missing = []
    for rel_path in REQUIRED_IFID_RUNTIME_FILES:
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def missing_continuous_tokenizer_runtime_files(ifid_repo_root: str) -> List[str]:
    root = Path(resolve_user_path(ifid_repo_root))
    if not root.exists():
        return list(REQUIRED_CONTINUOUS_RUNTIME_FILES)
    missing = []
    for rel_path in REQUIRED_CONTINUOUS_RUNTIME_FILES:
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def path_exists(path_value: str) -> bool:
    return bool(path_value) and Path(path_value).exists()


def split_piped_values(raw_value: str) -> List[str]:
    return [part.strip() for part in str(raw_value or "").split("|") if part.strip()]


def classify_runtime_failure(output: str) -> str:
    lowered = output.lower()
    if "args.json not found" in lowered:
        return "missing_args_json"
    if "registry csv not found" in lowered:
        return "missing_registry"
    if "ifid repo root not found" in lowered or "does not look like the ifid repo root" in lowered:
        return "missing_ifid_repo"
    if "incomplete ifid repo" in lowered or "missing required files under ifid repo" in lowered:
        return "missing_ifid_runtime"
    if "missing continuous tokenizer runtime files" in lowered:
        return "missing_continuous_runtime"
    if "unknown unexpected keys" in lowered:
        return "unknown_unexpected_keys"
    if "missing keys" in lowered and "checkpoint load reported missing keys" in lowered:
        return "missing_checkpoint_keys"
    if "dataset_dir is missing" in lowered:
        return "missing_dataset_dir"
    if "label_map_json is missing" in lowered:
        return "missing_label_map"
    if "fid reference file not found" in lowered:
        return "missing_fid_reference"
    if "missing vae_config" in lowered:
        return "missing_vae_config"
    if "could not resolve path" in lowered:
        return "unresolved_path"
    if "could not infer vae_type" in lowered:
        return "missing_vae_type"
    if "no module named" in lowered:
        return "missing_dependency"
    if "out of memory" in lowered:
        return "oom"
    return "runtime_failure"


def summarize_check_status(metadata_missing: List[str], asset_missing: List[str]) -> str:
    if not metadata_missing and not asset_missing:
        return "ready"
    if metadata_missing and asset_missing:
        return "missing_metadata_and_assets"
    if metadata_missing:
        return "missing_metadata"
    return "missing_assets"


def make_output_root(default_name: str) -> Path:
    output_root = PROJECT_ROOT / "results" / default_name
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def first_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return str(Path(candidate).resolve())
    return None
