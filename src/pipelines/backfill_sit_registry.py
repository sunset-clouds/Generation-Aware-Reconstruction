from __future__ import annotations

import argparse
import os
import csv
import shutil
from pathlib import Path
import sys
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.sit_stage2_common import (  # noqa: E402
    DEFAULT_IFID_REPO_ROOT,
    DEFAULT_REGISTRY_PATH,
    filter_rows,
    load_registry_rows,
    make_output_root,
    normalize_group_ids,
)


SERVER_STAGE2_ASSETS_ROOT = os.environ.get("STAGE2_ASSETS_ROOT", "./assets/stage2_assets")
SERVER_IMAGENET_TRAIN_DIR = "/public/test/data/imagenet/train"
SERVER_IMAGENET_VAL_DIR = "/public/test/data/imagenet/val"
SERVER_LABEL_MAP_JSON = os.environ.get("LABEL_MAP_JSON", "src/scripts/imagenet_val_dir_to_index.json")
SERVER_FID_REFERENCE_FILE = os.environ.get("FID_REFERENCE_FILE", "assets/VIRTUAL_imagenet256_labeled.npz")
SERVER_IFID_REPO_ROOT = os.environ.get("IFID_ROOT", "third_party/ifid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Stage 2 registry rows with server-side paths and checkpoint metadata."
    )
    parser.add_argument("--registry_path", type=str, default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--write", action="store_true", help="Write the updated CSV to disk.")
    parser.add_argument("--no_backup", action="store_true", help="Do not create .bak when overwriting the input CSV.")
    parser.add_argument("--group_ids", nargs="*", default=None)
    parser.add_argument("--only_should_run", action="store_true")
    parser.add_argument("--stage2_assets_root", type=str, default=SERVER_STAGE2_ASSETS_ROOT)
    parser.add_argument("--imagenet_train_dir", type=str, default=SERVER_IMAGENET_TRAIN_DIR)
    parser.add_argument("--imagenet_val_dir", type=str, default=SERVER_IMAGENET_VAL_DIR)
    parser.add_argument("--label_map_json", type=str, default=SERVER_LABEL_MAP_JSON)
    parser.add_argument("--fid_reference_file", type=str, default=SERVER_FID_REFERENCE_FILE)
    parser.add_argument("--ifid_repo_root", type=str, default=SERVER_IFID_REPO_ROOT)
    parser.add_argument("--default_ckpt_step", type=int, default=400000)
    parser.add_argument("--force_paths", action="store_true", help="Overwrite existing path fields for selected rows.")
    parser.add_argument("--fill_hf_source", action="store_true", help="Also fill hf_or_local_source using --ifid_repo_root.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for the preview report.")
    return parser.parse_args()


def _selected_group_ids(args: argparse.Namespace) -> Optional[set[str]]:
    return normalize_group_ids(args.group_ids)


def _should_modify_row(row: Dict[str, str], *, selected_group_ids: Optional[set[str]], only_should_run: bool) -> bool:
    group_id = str(row.get("group_id", "")).strip()
    if selected_group_ids and group_id not in selected_group_ids:
        return False
    if only_should_run and str(row.get("should_run", "")).strip().lower() not in {"yes", "true", "1", "y", "run"}:
        return False
    return True


def _set_value(
    row: Dict[str, str],
    field_name: str,
    value: str,
    *,
    force: bool,
    changes: List[str],
) -> None:
    current = str(row.get(field_name, "")).strip()
    if force or not current:
        if current != value:
            row[field_name] = value
            changes.append(field_name)


def _build_exp_path(stage2_assets_root: str, ifid_exp_name: str) -> str:
    return str((Path(stage2_assets_root) / ifid_exp_name).resolve())


def _build_args_json(exp_path: str) -> str:
    return str((Path(exp_path) / "args.json").resolve())


def _maybe_build_hf_source(ifid_repo_root: str, ifid_exp_name: str) -> str:
    return str((Path(ifid_repo_root) / ifid_exp_name).resolve())


def _render_preview(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Stage2 Registry Backfill Preview",
        "",
        f"- `registry_path`: `{summary['registry_path']}`",
        f"- `target_output`: `{summary['target_output']}`",
        f"- `rows_scanned`: `{summary['rows_scanned']}`",
        f"- `rows_selected`: `{summary['rows_selected']}`",
        f"- `rows_changed`: `{summary['rows_changed']}`",
        "",
        "## Changed Rows",
    ]
    if not rows:
        lines.append("- none")
    for row in rows:
        lines.append(
            f"- `{row['group_id']}`: changed `{', '.join(row['changed_fields']) or 'none'}`; "
            f"`exp_path={row['exp_path']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_path).resolve()
    rows = load_registry_rows(str(registry_path))
    selected_group_ids = _selected_group_ids(args)
    filtered_rows = filter_rows(
        rows,
        group_ids=args.group_ids,
        only_should_run=args.only_should_run,
    )
    filtered_lookup = {str(row.get("group_id", "")).strip() for row in filtered_rows}

    updated_rows: List[Dict[str, str]] = []
    changed_rows: List[Dict[str, object]] = []
    for row in rows:
        row_copy = dict(row)
        group_id = str(row_copy.get("group_id", "")).strip()
        changed_fields: List[str] = []
        if group_id in filtered_lookup and _should_modify_row(
            row_copy,
            selected_group_ids=selected_group_ids,
            only_should_run=args.only_should_run,
        ):
            ifid_exp_name = str(row_copy.get("ifid_exp_name", "")).strip()
            exp_path = _build_exp_path(args.stage2_assets_root, ifid_exp_name)
            args_json = _build_args_json(exp_path)

            _set_value(row_copy, "exp_path", exp_path, force=args.force_paths, changes=changed_fields)
            _set_value(row_copy, "args_json", args_json, force=args.force_paths, changes=changed_fields)
            _set_value(
                row_copy,
                "ckpt_step",
                str(args.default_ckpt_step),
                force=False,
                changes=changed_fields,
            )
            _set_value(row_copy, "imagenet_train_dir", args.imagenet_train_dir, force=args.force_paths, changes=changed_fields)
            _set_value(row_copy, "imagenet_val_dir", args.imagenet_val_dir, force=args.force_paths, changes=changed_fields)
            _set_value(row_copy, "label_map_json", args.label_map_json, force=args.force_paths, changes=changed_fields)
            _set_value(
                row_copy,
                "fid_reference_file",
                args.fid_reference_file,
                force=args.force_paths,
                changes=changed_fields,
            )
            if args.fill_hf_source:
                _set_value(
                    row_copy,
                    "hf_or_local_source",
                    _maybe_build_hf_source(args.ifid_repo_root, ifid_exp_name),
                    force=args.force_paths,
                    changes=changed_fields,
                )

        updated_rows.append(row_copy)
        if changed_fields:
            changed_rows.append(
                {
                    "group_id": group_id,
                    "changed_fields": changed_fields,
                    "exp_path": row_copy.get("exp_path", ""),
                    "args_json": row_copy.get("args_json", ""),
                    "ckpt_step": row_copy.get("ckpt_step", ""),
                }
            )

    fieldnames = [name for name in updated_rows[0].keys() if name != "__normalized_misaligned_csv"]
    target_output = Path(args.output_path).resolve() if args.output_path else registry_path

    preview_root = Path(args.output_dir or make_output_root("stage2_registry_backfill_preview")).resolve()
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_csv = preview_root / "registry_backfill_preview.csv"
    preview_md = preview_root / "registry_backfill_preview.md"

    with preview_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_rows:
            serializable = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(serializable)

    summary = {
        "registry_path": str(registry_path),
        "target_output": str(target_output),
        "rows_scanned": len(rows),
        "rows_selected": len(filtered_rows),
        "rows_changed": len(changed_rows),
    }
    preview_md.write_text(_render_preview(summary, changed_rows))

    if args.write:
        if target_output == registry_path and not args.no_backup:
            backup_path = registry_path.with_suffix(registry_path.suffix + ".bak")
            shutil.copyfile(registry_path, backup_path)
        with target_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in updated_rows:
                serializable = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(serializable)

    print("=" * 72)
    print("Stage2 registry backfill preview written")
    print(f"registry_path  : {registry_path}")
    print(f"target_output  : {target_output}")
    print(f"rows_selected  : {len(filtered_rows)}")
    print(f"rows_changed   : {len(changed_rows)}")
    print(f"preview_csv    : {preview_csv}")
    print(f"preview_md     : {preview_md}")
    print(f"write_applied  : {args.write}")
    print("=" * 72)


if __name__ == "__main__":
    main()
