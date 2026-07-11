from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.sit_stage2_common import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    filter_rows,
    load_registry_rows,
    make_output_root,
)


DEFAULT_REPO_ID = os.environ.get("STAGE2_ASSET_REPO_ID", "")
DEFAULT_STAGE2_ASSETS_ROOT = os.environ.get("STAGE2_ASSETS_ROOT", "./assets/stage2_assets")
HF_SUBDIR_OVERRIDES = {
    "sit-xl-sdvae-400k": "sit-xl-sdvae-0421-400k",
    "sit-b-fluxvae-400k": "sit-b-flux-shift-400k",
    "sit-xl-fluxvae-400k": "sit-xl-flux-shift-400k",
    "sit-b-qwvae-400k": "sit-b-qw-shift-400k",
    "sit-xl-qwvae-400k": "sit-xl-qw-shift-400k",
    "sit-b-sd3vae-400k": "sit-b-sd3-400k",
    "sit-xl-sd3vae-400k": "sit-xl-sd3-shift-400k",
    "sit-xl-invae-400k": "sit-xl-dinov2-b-enc8-repae-invae-0.0-0.0-400k",
    "sit-b-repaevae-400k": "sit-b-repae-400k",
    "sit-xl-repaevae-400k": "sit-xl-repae-400k",
    "sit-xl-vavae64-400k": "sit-xl-vavae64-shift-400k",
    "sit-xl-maetok-400k": "sit-XL-maetok-400k",
    "sit-xl-softvq-400k": "sit-XL-softvq-400k",
    "sit-b-rae-400k": "sit-b-rae-shift-400k",
    "sit-xl-rae-400k": "sit-xl-rae-shift-400k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stage 2 SiT args/checkpoints from Hugging Face according to the registry."
    )
    parser.add_argument("--registry_path", type=str, default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--stage2_assets_root", type=str, default=DEFAULT_STAGE2_ASSETS_ROOT)
    parser.add_argument("--group_ids", nargs="*", default=None)
    parser.add_argument("--only_should_run", action="store_true")
    parser.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", ""))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--default_ckpt_step", type=int, default=400000)
    parser.add_argument("--include_log", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def resolve_hf_subdir(ifid_exp_name: str) -> str:
    return HF_SUBDIR_OVERRIDES.get(ifid_exp_name, ifid_exp_name)


def resolve_target_dir(row: Dict[str, str], stage2_assets_root: str) -> Path:
    exp_path = str(row.get("exp_path", "")).strip()
    if exp_path:
        return Path(exp_path).expanduser().resolve()
    ifid_exp_name = str(row.get("ifid_exp_name", "")).strip()
    return (Path(stage2_assets_root) / ifid_exp_name).resolve()


def resolve_checkpoint_name(row: Dict[str, str], default_ckpt_step: int) -> str:
    ckpt_file = str(row.get("ckpt_file", "")).strip()
    if ckpt_file:
        return Path(ckpt_file).name
    ckpt_step_raw = str(row.get("ckpt_step", "")).strip()
    ckpt_step = int(ckpt_step_raw) if ckpt_step_raw else int(default_ckpt_step)
    return f"{ckpt_step:07d}.pt"


def expected_remote_files(row: Dict[str, str], hf_subdir: str, default_ckpt_step: int, include_log: bool) -> List[str]:
    remote_files = [
        f"{hf_subdir}/args.json",
        f"{hf_subdir}/checkpoints/{resolve_checkpoint_name(row, default_ckpt_step)}",
    ]
    if include_log:
        remote_files.append(f"{hf_subdir}/log.txt")
    return remote_files


def destination_for_remote(target_dir: Path, hf_subdir: str, remote_path: str) -> Path:
    relative_path = Path(remote_path).relative_to(hf_subdir)
    return (target_dir / relative_path).resolve()


def build_api(args: argparse.Namespace):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore

    api = HfApi(endpoint=args.hf_endpoint.rstrip("/") if args.hf_endpoint else None)
    return api, hf_hub_download


def list_repo_files(api, repo_id: str, hf_token: str) -> List[str]:
    kwargs = {"repo_id": repo_id, "repo_type": "model"}
    if hf_token:
        kwargs["token"] = hf_token
    return list(api.list_repo_files(**kwargs))


def download_one_file(
    *,
    hf_hub_download,
    repo_id: str,
    remote_path: str,
    hf_token: str,
    destination: Path,
    overwrite_existing: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite_existing:
        return "skipped_existing"
    kwargs = {
        "repo_id": repo_id,
        "filename": remote_path,
        "repo_type": "model",
    }
    if hf_token:
        kwargs["token"] = hf_token
    cached_path = Path(hf_hub_download(**kwargs))
    shutil.copy2(cached_path, destination)
    return "downloaded"


def render_markdown(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Stage2 HF Download Summary",
        "",
        f"- `repo_id`: `{summary['repo_id']}`",
        f"- `rows_requested`: `{summary['rows_requested']}`",
        f"- `rows_processed`: `{summary['rows_processed']}`",
        f"- `rows_success`: `{summary['rows_success']}`",
        f"- `rows_failed`: `{summary['rows_failed']}`",
        f"- `dry_run`: `{summary['dry_run']}`",
        "",
        "## Per Group",
    ]
    for row in rows:
        lines.append(
            f"- `{row['group_id']}`: status=`{row['status']}`, hf_subdir=`{row['hf_subdir']}`, "
            f"target_dir=`{row['target_dir']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit(
            "Missing Hugging Face repo id. Pass --repo_id or set STAGE2_ASSET_REPO_ID."
        )
    rows = filter_rows(
        load_registry_rows(args.registry_path),
        group_ids=args.group_ids,
        only_should_run=args.only_should_run,
    )
    api, hf_hub_download = build_api(args)
    repo_files = set(list_repo_files(api, args.repo_id, args.hf_token))

    output_dir = Path(args.output_dir or make_output_root("stage2_hf_downloads")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_summary: List[Dict[str, object]] = []

    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        ifid_exp_name = str(row.get("ifid_exp_name", "")).strip()
        hf_subdir = resolve_hf_subdir(ifid_exp_name)
        target_dir = resolve_target_dir(row, args.stage2_assets_root)
        remote_files = expected_remote_files(row, hf_subdir, args.default_ckpt_step, args.include_log)

        missing_remote = [remote_file for remote_file in remote_files if remote_file not in repo_files]
        file_actions: List[Dict[str, str]] = []
        status = "dry_run" if args.dry_run else "success"

        if missing_remote:
            status = "missing_remote_files"
        else:
            for remote_file in remote_files:
                destination = destination_for_remote(target_dir, hf_subdir, remote_file)
                if args.dry_run:
                    action = "would_download" if not destination.exists() else "would_skip_existing"
                else:
                    action = download_one_file(
                        hf_hub_download=hf_hub_download,
                        repo_id=args.repo_id,
                        remote_path=remote_file,
                        hf_token=args.hf_token,
                        destination=destination,
                        overwrite_existing=args.overwrite_existing,
                    )
                file_actions.append(
                    {
                        "remote_file": remote_file,
                        "destination": str(destination),
                        "action": action,
                    }
                )

        rows_summary.append(
            {
                "group_id": group_id,
                "ifid_exp_name": ifid_exp_name,
                "hf_subdir": hf_subdir,
                "target_dir": str(target_dir),
                "status": status,
                "missing_remote_files": missing_remote,
                "file_actions": file_actions,
            }
        )

    summary = {
        "repo_id": args.repo_id,
        "rows_requested": len(rows),
        "rows_processed": len(rows_summary),
        "rows_success": sum(1 for row in rows_summary if row["status"] in {"success", "dry_run"}),
        "rows_failed": sum(1 for row in rows_summary if row["status"] == "missing_remote_files"),
        "dry_run": args.dry_run,
    }

    json_path = output_dir / "download_summary.json"
    csv_path = output_dir / "download_summary.csv"
    md_path = output_dir / "download_summary.md"

    with json_path.open("w") as handle:
        json.dump({"summary": summary, "rows": rows_summary}, handle, indent=2)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_id",
                "ifid_exp_name",
                "hf_subdir",
                "target_dir",
                "status",
                "missing_remote_files",
            ],
        )
        writer.writeheader()
        for row in rows_summary:
            writer.writerow(
                {
                    "group_id": row["group_id"],
                    "ifid_exp_name": row["ifid_exp_name"],
                    "hf_subdir": row["hf_subdir"],
                    "target_dir": row["target_dir"],
                    "status": row["status"],
                    "missing_remote_files": "|".join(row["missing_remote_files"]),
                }
            )

    md_path.write_text(render_markdown(summary, rows_summary))

    print("=" * 72)
    print("Stage2 HF download summary written")
    print(f"repo_id       : {args.repo_id}")
    print(f"rows_processed: {summary['rows_processed']}")
    print(f"rows_success  : {summary['rows_success']}")
    print(f"rows_failed   : {summary['rows_failed']}")
    print(f"output_json   : {json_path}")
    print(f"output_csv    : {csv_path}")
    print(f"output_md     : {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
