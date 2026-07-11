from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.sit_stage2_common import (
    DEFAULT_IFID_REPO_ROOT,
    classify_runtime_failure,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch smoke runner for Stage 2 SiT checkpoints. This runner never computes full metrics."
    )
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=str(DEFAULT_IFID_REPO_ROOT))
    parser.add_argument("--group_ids", nargs="*", default=None)
    parser.add_argument("--only_should_run", action="store_true")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--eta_t", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--sampling_mode", type=str, choices=("ode", "sde"), default="sde")
    parser.add_argument("--step_policy", type=str, choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--min_num_steps", type=int, default=8)
    parser.add_argument("--max_num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_missing_keys", action="store_true")
    parser.add_argument("--allow_unknown_unexpected_keys", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
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


def _audit_row(row: Dict[str, str], ifid_repo_root: str) -> Tuple[List[str], List[str]]:
    metadata_missing: List[str] = []
    asset_missing: List[str] = []

    for field_name in (
        "group_id",
        "vae_name",
        "vae_config",
        "vae_type",
        "latent_kind",
        "sit_scale",
        "sit_model",
        "ifid_exp_name",
        "cfg_modes",
        "load_policy",
        "imagenet_val_dir",
        "label_map_json",
        "fid_reference_file",
    ):
        if not str(row.get(field_name, "")).strip():
            metadata_missing.append(field_name)

    exp_path_value = str(row.get("exp_path", "")).strip()
    resolved_exp_path = resolve_user_path(exp_path_value, extra_roots=[Path(ifid_repo_root)]) if exp_path_value else ""
    if not exp_path_value:
        metadata_missing.append("exp_path")
    elif not path_exists(resolved_exp_path):
        asset_missing.append("exp_path")

    vae_config_value = str(row.get("vae_config", "")).strip()
    if vae_config_value:
        resolved_vae_config = resolve_stage2_vae_config_path(
            vae_config_value,
            ifid_repo_root=ifid_repo_root,
            exp_path=resolved_exp_path,
        )
        if not path_exists(resolved_vae_config):
            asset_missing.append("vae_config")

    missing_ifid_runtime = missing_ifid_runtime_files(ifid_repo_root)
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
    resolved_ckpt = _resolve_ckpt_path(row, resolved_exp_path, ifid_repo_root)
    if resolved_ckpt and not path_exists(resolved_ckpt):
        asset_missing.append("checkpoint")

    dataset_dir = resolve_user_path(str(row.get("imagenet_val_dir", "")).strip())
    if dataset_dir and not path_exists(dataset_dir):
        asset_missing.append("imagenet_val_dir")
    label_map_json = resolve_user_path(str(row.get("label_map_json", "")).strip())
    if label_map_json and not path_exists(label_map_json):
        asset_missing.append("label_map_json")
    fid_reference = str(row.get("fid_reference_file", "")).strip()
    resolved_fid_reference = resolve_reference_path(fid_reference, ifid_repo_root) if fid_reference else ""
    if resolved_fid_reference and not path_exists(resolved_fid_reference):
        asset_missing.append("fid_reference_file")

    return metadata_missing, asset_missing


def _build_command(args: argparse.Namespace, group_id: str, output_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "pipelines" / "smoke_test_sit_gar.py"),
        "--group_id",
        group_id,
        "--registry_path",
        str(Path(args.registry_path or PROJECT_ROOT / "assets" / "sit_checkpoint_registry_template.csv").resolve()),
        "--ifid_repo_root",
        resolve_user_path(args.ifid_repo_root),
        "--output_dir",
        str(output_dir),
        "--num_images",
        str(args.num_images),
        "--eta_t",
        str(args.eta_t),
        "--num_steps",
        str(args.num_steps),
        "--sampling_mode",
        args.sampling_mode,
        "--step_policy",
        args.step_policy,
        "--min_num_steps",
        str(args.min_num_steps),
        "--cfg_scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
    ]
    if args.max_num_steps is not None:
        command.extend(["--max_num_steps", str(args.max_num_steps)])
    if args.use_cfg:
        command.append("--use_cfg")
    if args.device:
        command.extend(["--device", args.device])
    if args.allow_missing_keys:
        command.append("--allow_missing_keys")
    if args.allow_unknown_unexpected_keys:
        command.append("--allow_unknown_unexpected_keys")
    return command


def _infer_checks(returncode: int, output: str) -> Tuple[str, str]:
    if returncode == 0:
        return "pass", "pass"
    if any(marker in output for marker in ("Loading dataset from", "Saved grid to", "Visualizing")):
        return "pass", "fail"
    return "fail", "not_run"


def _render_markdown(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Stage2 Batch Smoke Summary",
        "",
        f"- `registry_path`: `{summary['registry_path']}`",
        f"- `ifid_repo_root`: `{summary['ifid_repo_root']}`",
        f"- `rows_requested`: `{summary['rows_requested']}`",
        f"- `rows_executed`: `{summary['rows_executed']}`",
        f"- `rows_succeeded`: `{summary['rows_succeeded']}`",
        f"- `rows_failed`: `{summary['rows_failed']}`",
        f"- `rows_skipped_precheck`: `{summary['rows_skipped_precheck']}`",
        f"- `full_metrics_launched`: `0`",
        "",
        "## Per Group",
    ]
    for row in rows:
        lines.append(
            f"- `{row['group_id']}`: status=`{row['status']}`, "
            f"load_check=`{row['load_check']}`, inference_check=`{row['inference_check']}`, "
            f"failure_category=`{row['failure_category']}`"
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
    output_root = Path(args.output_dir or make_output_root("stage2_smoke_batch")).resolve()
    logs_dir = output_root / "logs"
    runs_dir = output_root / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        metadata_missing, asset_missing = _audit_row(row, ifid_repo_root)
        precheck_status = summarize_check_status(metadata_missing, asset_missing)
        run_output_dir = runs_dir / group_id
        log_path = logs_dir / f"{group_id}.log"

        if precheck_status != "ready":
            result = {
                "group_id": group_id,
                "status": f"skipped_{precheck_status}",
                "load_check": "not_run",
                "inference_check": "not_run",
                "failure_category": precheck_status,
                "metadata_missing": metadata_missing,
                "asset_missing": asset_missing,
                "output_dir": str(run_output_dir),
                "log_path": str(log_path),
                "return_code": None,
            }
            log_path.write_text(
                "Precheck blocked smoke run.\n"
                f"metadata_missing={metadata_missing}\n"
                f"asset_missing={asset_missing}\n"
            )
            results.append(result)
            continue

        command = _build_command(args, group_id, run_output_dir)
        if args.dry_run:
            log_path.write_text("DRY RUN\n" + " ".join(command) + "\n")
            results.append(
                {
                    "group_id": group_id,
                    "status": "dry_run",
                    "load_check": "not_run",
                    "inference_check": "not_run",
                    "failure_category": "dry_run",
                    "metadata_missing": [],
                    "asset_missing": [],
                    "output_dir": str(run_output_dir),
                    "log_path": str(log_path),
                    "return_code": None,
                }
            )
            continue

        completed = subprocess.run(
            command,
            cwd=str(SRC_ROOT),
            text=True,
            capture_output=True,
        )
        combined_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        log_path.write_text(combined_output)
        load_check, inference_check = _infer_checks(completed.returncode, combined_output)
        failure_category = "none" if completed.returncode == 0 else classify_runtime_failure(combined_output)
        results.append(
            {
                "group_id": group_id,
                "status": "success" if completed.returncode == 0 else "failed",
                "load_check": load_check,
                "inference_check": inference_check,
                "failure_category": failure_category,
                "metadata_missing": [],
                "asset_missing": [],
                "output_dir": str(run_output_dir),
                "log_path": str(log_path),
                "return_code": completed.returncode,
            }
        )
        if completed.returncode != 0 and args.stop_on_failure:
            break

    summary = {
        "registry_path": str(Path(args.registry_path or PROJECT_ROOT / "assets" / "sit_checkpoint_registry_template.csv").resolve()),
        "ifid_repo_root": ifid_repo_root,
        "rows_requested": len(rows),
        "rows_executed": sum(1 for row in results if row["status"] in {"success", "failed"}),
        "rows_succeeded": sum(1 for row in results if row["status"] == "success"),
        "rows_failed": sum(1 for row in results if row["status"] == "failed"),
        "rows_skipped_precheck": sum(1 for row in results if str(row["status"]).startswith("skipped_")),
        "full_metrics_launched": 0,
    }

    summary_json = output_root / "batch_smoke_summary.json"
    summary_csv = output_root / "batch_smoke_summary.csv"
    summary_md = output_root / "batch_smoke_summary.md"

    with summary_json.open("w") as handle:
        json.dump({"summary": summary, "rows": results}, handle, indent=2)

    csv_fieldnames = [
        "group_id",
        "status",
        "load_check",
        "inference_check",
        "failure_category",
        "metadata_missing",
        "asset_missing",
        "output_dir",
        "log_path",
        "return_code",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in results:
            serializable = dict(row)
            serializable["metadata_missing"] = "|".join(serializable["metadata_missing"])
            serializable["asset_missing"] = "|".join(serializable["asset_missing"])
            writer.writerow(serializable)

    summary_md.write_text(_render_markdown(summary, results))

    print("=" * 72)
    print("Stage2 batch smoke summary written")
    print(f"rows_requested : {summary['rows_requested']}")
    print(f"rows_executed  : {summary['rows_executed']}")
    print(f"rows_succeeded : {summary['rows_succeeded']}")
    print(f"rows_failed    : {summary['rows_failed']}")
    print(f"output_json    : {summary_json}")
    print(f"output_csv     : {summary_csv}")
    print(f"output_md      : {summary_md}")
    print("=" * 72)


if __name__ == "__main__":
    main()
