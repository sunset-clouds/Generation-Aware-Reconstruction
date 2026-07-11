from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.sit_stage2_common import (
    DEFAULT_IFID_REPO_ROOT,
    filter_rows,
    load_registry_rows,
    make_output_root,
    missing_continuous_tokenizer_runtime_files,
    missing_ifid_runtime_files,
    path_exists,
    resolve_reference_path,
    resolve_stage2_vae_config_path,
    resolve_user_path,
    summarize_check_status,
)


REQUIRED_IDENTITY_FIELDS = [
    "group_id",
    "vae_name",
    "vae_config",
    "vae_type",
    "latent_kind",
    "sit_scale",
    "sit_model",
    "ifid_exp_name",
]
REQUIRED_RUNTIME_FIELDS = [
    "cfg_modes",
    "load_policy",
    "imagenet_val_dir",
    "label_map_json",
    "fid_reference_file",
]
BASELINE_FIELDS = [
    "expected_ifid",
    "expected_gfid_wo_cfg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage 2 SiT registry completeness and on-disk assets."
    )
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=str(DEFAULT_IFID_REPO_ROOT))
    parser.add_argument("--group_ids", nargs="*", default=None)
    parser.add_argument("--only_should_run", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _resolve_args_path(row: Dict[str, str], exp_path: str, ifid_repo_root: str) -> str:
    args_json = str(row.get("args_json", "")).strip()
    if args_json:
        return resolve_user_path(args_json, extra_roots=[Path(exp_path), Path(ifid_repo_root)])
    if exp_path:
        return str((Path(exp_path) / "args.json").resolve())
    return ""


def _resolve_ckpt_path(row: Dict[str, str], exp_path: str, ifid_repo_root: str) -> str:
    ckpt_file = str(row.get("ckpt_file", "")).strip()
    if ckpt_file:
        return resolve_user_path(ckpt_file, extra_roots=[Path(exp_path), Path(ifid_repo_root)])
    ckpt_step = str(row.get("ckpt_step", "")).strip()
    if exp_path and ckpt_step:
        return str((Path(exp_path) / "checkpoints" / f"{int(ckpt_step):07d}.pt").resolve())
    return ""


def _append_missing_if_blank(target: List[str], row: Dict[str, str], field_names: List[str]) -> None:
    for field_name in field_names:
        if not str(row.get(field_name, "")).strip():
            target.append(field_name)


def _render_markdown(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Stage2 Registry Audit",
        "",
        f"- `registry_path`: `{summary['registry_path']}`",
        f"- `ifid_repo_root`: `{summary['ifid_repo_root']}`",
        f"- `rows_scanned`: `{summary['rows_scanned']}`",
        f"- `ready_rows`: `{summary['ready_rows']}`",
        f"- `rows_missing_metadata`: `{summary['rows_missing_metadata']}`",
        f"- `rows_missing_assets`: `{summary['rows_missing_assets']}`",
        f"- `rows_missing_both`: `{summary['rows_missing_both']}`",
        f"- `missing_expected_group_ids`: `{', '.join(summary['missing_expected_group_ids']) or 'none'}`",
        "",
        "## Row Status",
    ]
    for row in rows:
        lines.append(
            f"- `{row['group_id']}`: `{row['check_status']}`; "
            f"metadata_missing={row['metadata_missing_count']}; "
            f"asset_missing={row['asset_missing_count']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = filter_rows(
        load_registry_rows(args.registry_path),
        group_ids=args.group_ids,
        only_should_run=args.only_should_run,
    )
    ifid_repo_root = resolve_user_path(args.ifid_repo_root)

    audited_rows: List[Dict[str, object]] = []
    missing_ifid_runtime = missing_ifid_runtime_files(ifid_repo_root)
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        metadata_missing: List[str] = []
        asset_missing: List[str] = []
        warnings: List[str] = []

        _append_missing_if_blank(metadata_missing, row, REQUIRED_IDENTITY_FIELDS + REQUIRED_RUNTIME_FIELDS)

        exp_path_value = str(row.get("exp_path", "")).strip()
        resolved_exp_path = resolve_user_path(exp_path_value, extra_roots=[Path(ifid_repo_root)]) if exp_path_value else ""
        if not exp_path_value:
            metadata_missing.append("exp_path")
        elif not path_exists(resolved_exp_path):
            asset_missing.append("exp_path")

        vae_config_value = str(row.get("vae_config", "")).strip()
        resolved_vae_config = ""
        if vae_config_value:
            resolved_vae_config = resolve_stage2_vae_config_path(
                vae_config_value,
                ifid_repo_root=ifid_repo_root,
                exp_path=resolved_exp_path,
            )
            if not path_exists(resolved_vae_config):
                asset_missing.append("vae_config")

        if missing_ifid_runtime:
            asset_missing.extend(f"ifid_runtime:{rel_path}" for rel_path in missing_ifid_runtime)
        vae_type = str(row.get("vae_type", "")).strip().lower()
        if vae_type in {"softvq", "maetok"}:
            missing_continuous_runtime = missing_continuous_tokenizer_runtime_files(ifid_repo_root)
            if missing_continuous_runtime:
                asset_missing.extend(
                    f"continuous_runtime:{rel_path}" for rel_path in missing_continuous_runtime
                )

        if not str(row.get("args_json", "")).strip() and not exp_path_value:
            metadata_missing.append("args_json_or_exp_path")
        resolved_args_path = _resolve_args_path(row, resolved_exp_path, ifid_repo_root)
        if resolved_args_path and not path_exists(resolved_args_path):
            asset_missing.append("args_json")

        ckpt_step = str(row.get("ckpt_step", "")).strip()
        ckpt_file = str(row.get("ckpt_file", "")).strip()
        if not ckpt_step and not ckpt_file:
            metadata_missing.append("ckpt_step_or_ckpt_file")
        resolved_ckpt_path = _resolve_ckpt_path(row, resolved_exp_path, ifid_repo_root)
        if resolved_ckpt_path and not path_exists(resolved_ckpt_path):
            asset_missing.append("checkpoint")

        resolved_dataset_dir = resolve_user_path(str(row.get("imagenet_val_dir", "")).strip())
        if resolved_dataset_dir and not path_exists(resolved_dataset_dir):
            asset_missing.append("imagenet_val_dir")

        resolved_label_map = resolve_user_path(str(row.get("label_map_json", "")).strip())
        if resolved_label_map and not path_exists(resolved_label_map):
            asset_missing.append("label_map_json")

        fid_reference_value = str(row.get("fid_reference_file", "")).strip()
        resolved_fid_reference = resolve_reference_path(fid_reference_value, ifid_repo_root) if fid_reference_value else ""
        if resolved_fid_reference and not path_exists(resolved_fid_reference):
            asset_missing.append("fid_reference_file")

        for field_name in BASELINE_FIELDS:
            if not str(row.get(field_name, "")).strip():
                warnings.append(f"missing_{field_name}")
        if not str(row.get("expected_gfid_w_cfg", "")).strip():
            warnings.append("missing_expected_gfid_w_cfg")

        check_status = summarize_check_status(metadata_missing, asset_missing)
        audited_rows.append(
            {
                "group_id": group_id,
                "check_status": check_status,
                "metadata_missing_count": len(metadata_missing),
                "asset_missing_count": len(asset_missing),
                "metadata_missing": metadata_missing,
                "asset_missing": asset_missing,
                "warnings": warnings,
                "resolved_exp_path": resolved_exp_path,
                "resolved_vae_config": resolved_vae_config,
                "resolved_args_json": resolved_args_path,
                "resolved_ckpt_file": resolved_ckpt_path,
                "resolved_imagenet_val_dir": resolved_dataset_dir,
                "resolved_label_map_json": resolved_label_map,
                "resolved_fid_reference_file": resolved_fid_reference,
            }
        )

    summary = {
        "registry_path": str(Path(args.registry_path or PROJECT_ROOT / "assets" / "sit_checkpoint_registry_template.csv").resolve()),
        "ifid_repo_root": ifid_repo_root,
        "rows_scanned": len(audited_rows),
        "ready_rows": sum(1 for row in audited_rows if row["check_status"] == "ready"),
        "rows_missing_metadata": sum(1 for row in audited_rows if row["check_status"] == "missing_metadata"),
        "rows_missing_assets": sum(1 for row in audited_rows if row["check_status"] == "missing_assets"),
        "rows_missing_both": sum(1 for row in audited_rows if row["check_status"] == "missing_metadata_and_assets"),
        "missing_expected_group_ids": [],
    }

    family_scales: Dict[str, set[str]] = {}
    for row in audited_rows:
        group_id = str(row["group_id"])
        if "_" not in group_id:
            continue
        family, scale = group_id.rsplit("_", 1)
        family_scales.setdefault(family, set()).add(scale)
    missing_expected_group_ids: List[str] = []
    for family, scales in sorted(family_scales.items()):
        for expected_scale in ("b", "xl"):
            if expected_scale not in scales:
                missing_expected_group_ids.append(f"{family}_{expected_scale}")
    summary["missing_expected_group_ids"] = missing_expected_group_ids

    output_dir = Path(args.output_dir or make_output_root("stage2_registry_audit")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "registry_audit.json"
    csv_path = output_dir / "registry_audit.csv"
    md_path = output_dir / "registry_audit.md"

    with json_path.open("w") as handle:
        json.dump({"summary": summary, "rows": audited_rows}, handle, indent=2)

    csv_fieldnames = [
        "group_id",
        "check_status",
        "metadata_missing_count",
        "asset_missing_count",
        "metadata_missing",
        "asset_missing",
        "warnings",
        "resolved_exp_path",
        "resolved_vae_config",
        "resolved_args_json",
        "resolved_ckpt_file",
        "resolved_imagenet_val_dir",
        "resolved_label_map_json",
        "resolved_fid_reference_file",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in audited_rows:
            serializable = dict(row)
            for key in ("metadata_missing", "asset_missing", "warnings"):
                serializable[key] = "|".join(serializable[key])
            writer.writerow(serializable)

    md_path.write_text(_render_markdown(summary, audited_rows))

    print("=" * 72)
    print("Stage2 registry audit written")
    print(f"rows_scanned   : {summary['rows_scanned']}")
    print(f"ready_rows     : {summary['ready_rows']}")
    print(f"output_json    : {json_path}")
    print(f"output_csv     : {csv_path}")
    print(f"output_md      : {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
