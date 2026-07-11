from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.sit_stage2_common import (
    DEFAULT_IFID_REPO_ROOT,
    load_registry_rows,
    make_output_root,
    path_exists,
    resolve_reference_path,
    resolve_user_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a baseline-alignment report for one SiT checkpoint group."
    )
    parser.add_argument("--group_id", type=str, default="sdvae_b")
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=str(DEFAULT_IFID_REPO_ROOT))
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--label_map_json", type=str, default=None)
    parser.add_argument("--fid_reference_file", type=str, default=None)
    parser.add_argument("--adapter_summary_json", type=str, default=None)
    parser.add_argument("--official_ifid", type=float, default=None)
    parser.add_argument("--official_gfid_nocfg", type=float, default=None)
    parser.add_argument("--official_gfid_cfg", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_load_json(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def _guess_adapter_summary_path(group_id: str) -> Optional[str]:
    candidates = [
        REPO_ROOT / "results" / f"garfid_single_{group_id}_eta03" / "summary.json",
        REPO_ROOT / "results" / "sit_garfid_single" / f"{group_id}_eta0.3" / "summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _gap(actual: Optional[float], expected: Optional[float]) -> Optional[float]:
    if actual is None or expected is None:
        return None
    return actual - expected


def _collect_alignment_notes(
    *,
    adapter_summary: Optional[Dict[str, Any]],
    official_generate_exists: bool,
    official_evalvae_exists: bool,
    fid_reference_path: str,
) -> List[str]:
    notes: List[str] = []
    if not official_generate_exists:
        notes.append("`generate.py` is missing under the IFID repo root, so official gFID reproduction cannot run from this checkout yet.")
    if not official_evalvae_exists:
        notes.append("`evalvae.py` is missing under the IFID repo root, so official iFID reproduction requires a fuller IFID checkout or a mirrored copy.")
    if not path_exists(fid_reference_path):
        notes.append("`VIRTUAL_imagenet256_labeled.npz` is unresolved on disk; gFID comparisons are blocked until the reference path is fixed.")
    if adapter_summary is None:
        notes.append("No adapter summary was provided or auto-discovered, so this report only fixes the canonical assets and expected paper baselines.")
    if adapter_summary:
        num_samples = adapter_summary.get("num_samples")
        if isinstance(num_samples, int) and num_samples < 50000:
            notes.append(
                f"The current adapter summary uses only {num_samples} samples. gFID at this sample count is a sanity number, not a paper-comparable baseline."
            )
        num_steps = adapter_summary.get("num_steps")
        if isinstance(num_steps, int) and num_steps < 250:
            notes.append(
                f"The current adapter summary uses {num_steps} denoising steps; official gFID runs typically use many more steps, so generated-image quality is not yet aligned."
            )
        if adapter_summary.get("use_cfg"):
            notes.append("The adapter summary was produced with CFG enabled; compare it only to registry `expected_gfid_w_cfg` values.")
        else:
            notes.append("The adapter summary was produced without CFG; compare it only to registry `expected_gfid_wo_cfg` values.")
    notes.append("Canonical Stage 2 report alignment should use the IFID-style gFID reference `VIRTUAL_imagenet256_labeled.npz` rather than the iMF `jit_in256_stats` or TPU/JAX stats.")
    notes.append("Registry `expected_ifid` and `expected_gfid_*` should be treated as paper-report baselines; `rFID` currently has no official baseline column and must be tracked as an internal Stage 2 metric.")
    return notes


def _make_command_block(
    *,
    ifid_repo_root: str,
    dataset_dir: str,
    label_map_json: str,
    fid_reference_file: str,
    group_id: str,
    registry_path: str,
    adapter_summary_path: Optional[str],
    exp_path: str,
    ckpt_step: str,
) -> Dict[str, str]:
    exp_path_value = exp_path or "<fill_exp_path>"
    ckpt_step_value = ckpt_step or "<fill_ckpt_step>"
    official_gfid = f"""cd {ifid_repo_root}
torchrun --standalone --nnodes=1 --nproc_per_node=1 generate.py \\
  --num-fid-samples 50000 \\
  --mode sde \\
  --num-steps 250 \\
  --cfg-scale 1.0 \\
  --guidance-high 1.0 \\
  --guidance-low 0.0 \\
  --sample-dir ./samples \\
  --exp-path {exp_path_value} \\
  --fid-reference-file {fid_reference_file} \\
  --train-steps {ckpt_step_value}
"""
    adapter_garfid = f"""cd {REPO_ROOT}
${{PYTHON:-python3}} -u pipelines/run_sit_garfid_single.py \\
  --group_id {group_id} \\
  --registry_path {registry_path} \\
  --ifid_repo_root {ifid_repo_root} \\
  --dataset_dir {dataset_dir} \\
  --label_map_json {label_map_json} \\
  --fid_reference_file {fid_reference_file} \\
  --output_dir {REPO_ROOT / "results" / f"garfid_single_{group_id}_eta03"} \\
  --num_samples 50000 \\
  --batch_size 8 \\
  --eta_t 0.3 \\
  --num_steps 250 \\
  --save_npz
"""
    align_report = f"""cd {REPO_ROOT}
${{PYTHON:-python3}} -u pipelines/align_sit_baseline.py \\
  --group_id {group_id} \\
  --registry_path {registry_path} \\
  --ifid_repo_root {ifid_repo_root} \\
  --dataset_dir {dataset_dir} \\
  --label_map_json {label_map_json} \\
  --fid_reference_file {fid_reference_file}"""
    if adapter_summary_path:
        align_report += f" \\\n+  --adapter_summary_json {adapter_summary_path}"
    align_report += "\n"
    return {
        "official_gfid_nocfg": official_gfid,
        "adapter_garfid_single": adapter_garfid,
        "alignment_report_refresh": align_report,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    assets = report["canonical_assets"]
    expected = report["expected_metrics"]
    adapter = report.get("adapter_summary", {})
    commands = report["recommended_commands"]
    lines = [
        "# Stage2 Baseline Alignment",
        "",
        f"## Group",
        f"- `group_id`: `{report['group_id']}`",
        f"- `registry_path`: `{report['registry_path']}`",
        "",
        "## Canonical Assets",
        f"- `ifid_repo_root`: `{assets['ifid_repo_root']}`",
        f"- `dataset_dir`: `{assets['dataset_dir']}`",
        f"- `label_map_json`: `{assets['label_map_json']}`",
        f"- `fid_reference_file`: `{assets['fid_reference_file']}`",
        "",
        "## Canonical Policy",
        "- Official report alignment uses IFID-style `generate.py` and `VIRTUAL_imagenet256_labeled.npz`.",
        "- `rFID` is treated as an internal Stage 2 baseline unless a paper-side report value is explicitly added to the registry.",
        "- `torch-fidelity + jit_in256_stats.npz` is useful for iMF comparisons, but not the canonical Stage 2 SiT baseline in this workflow.",
        "",
        "## Expected Metrics",
        f"- `expected_ifid`: {_format_float(expected['expected_ifid'])}",
        f"- `expected_gfid_nocfg`: {_format_float(expected['expected_gfid_nocfg'])}",
        f"- `expected_gfid_cfg`: {_format_float(expected['expected_gfid_cfg'])}",
        "",
        "## Adapter Summary",
        f"- `summary_path`: `{adapter.get('path', 'n/a')}`",
        f"- `num_samples`: `{adapter.get('num_samples', 'n/a')}`",
        f"- `num_steps`: `{adapter.get('num_steps', 'n/a')}`",
        f"- `use_cfg`: `{adapter.get('use_cfg', 'n/a')}`",
        f"- `vanilla_rfid`: {_format_float(adapter.get('vanilla_rfid'))}",
        f"- `garfid`: {_format_float(adapter.get('garfid'))}",
        f"- `gfid`: {_format_float(adapter.get('gfid'))}",
        f"- `gfid_minus_expected_gfid_nocfg`: {_format_float(adapter.get('gfid_minus_expected_gfid_nocfg'))}",
        "",
        "## Alignment Notes",
    ]
    for note in report["alignment_notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Recommended Commands",
            "### Official gFID noCFG",
            "```bash",
            commands["official_gfid_nocfg"].rstrip(),
            "```",
            "",
            "### Adapter GAR-FID single point",
            "```bash",
            commands["adapter_garfid_single"].rstrip(),
            "```",
            "",
            "### Refresh this alignment report",
            "```bash",
            commands["alignment_report_refresh"].rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_registry_rows(args.registry_path)
    row = next((item for item in rows if str(item.get("group_id", "")).strip() == args.group_id), None)
    if row is None:
        raise KeyError(f"group_id={args.group_id!r} not found in registry")
    registry_path = str(Path(args.registry_path or REPO_ROOT / "sit_checkpoint_registry_template.csv").resolve())
    ifid_repo_root = resolve_user_path(args.ifid_repo_root)
    dataset_dir = resolve_user_path(args.dataset_dir or row.get("imagenet_val_dir") or "")
    label_map_json = resolve_user_path(args.label_map_json or row.get("label_map_json") or "")
    fid_reference_file = resolve_reference_path(
        args.fid_reference_file or row.get("fid_reference_file") or "VIRTUAL_imagenet256_labeled.npz",
        ifid_repo_root,
    )

    adapter_summary_path = (
        resolve_user_path(args.adapter_summary_json)
        if args.adapter_summary_json
        else _guess_adapter_summary_path(args.group_id)
    )
    adapter_summary_json = _maybe_load_json(adapter_summary_path)
    adapter_metrics = (adapter_summary_json or {}).get("metrics", {})
    adapter_section = {
        "path": adapter_summary_path,
        "num_samples": (adapter_summary_json or {}).get("num_samples"),
        "num_steps": (adapter_summary_json or {}).get("num_steps"),
        "use_cfg": (adapter_summary_json or {}).get("use_cfg"),
        "vanilla_rfid": _safe_float(adapter_metrics.get("vanilla_rfid")),
        "garfid": _safe_float(adapter_metrics.get("garfid")),
        "gfid": _safe_float(adapter_metrics.get("gfid")),
    }

    expected_ifid = _safe_float(args.official_ifid if args.official_ifid is not None else row.get("expected_ifid"))
    expected_gfid_nocfg = _safe_float(
        args.official_gfid_nocfg if args.official_gfid_nocfg is not None else row.get("expected_gfid_wo_cfg")
    )
    expected_gfid_cfg = _safe_float(
        args.official_gfid_cfg if args.official_gfid_cfg is not None else row.get("expected_gfid_w_cfg")
    )
    adapter_section["gfid_minus_expected_gfid_nocfg"] = _gap(adapter_section["gfid"], expected_gfid_nocfg)

    official_generate_path = Path(ifid_repo_root) / "generate.py"
    official_evalvae_path = Path(ifid_repo_root) / "evalvae.py"

    commands = _make_command_block(
        ifid_repo_root=ifid_repo_root,
        dataset_dir=dataset_dir or "<fill_dataset_dir>",
        label_map_json=label_map_json or "<fill_label_map_json>",
        fid_reference_file=fid_reference_file or "<fill_fid_reference_file>",
        group_id=args.group_id,
        registry_path=registry_path,
        adapter_summary_path=adapter_summary_path,
        exp_path=str(row.get("exp_path", "")).strip(),
        ckpt_step=str(row.get("ckpt_step", "")).strip(),
    )

    report = {
        "group_id": args.group_id,
        "registry_path": registry_path,
        "canonical_assets": {
            "ifid_repo_root": ifid_repo_root,
            "dataset_dir": dataset_dir,
            "label_map_json": label_map_json,
            "fid_reference_file": fid_reference_file,
        },
        "expected_metrics": {
            "expected_ifid": expected_ifid,
            "expected_gfid_nocfg": expected_gfid_nocfg,
            "expected_gfid_cfg": expected_gfid_cfg,
        },
        "adapter_summary": adapter_section,
        "official_paths": {
            "generate_py": str(official_generate_path),
            "generate_py_exists": official_generate_path.exists(),
            "evalvae_py": str(official_evalvae_path),
            "evalvae_py_exists": official_evalvae_path.exists(),
        },
        "alignment_notes": _collect_alignment_notes(
            adapter_summary=adapter_summary_json,
            official_generate_exists=official_generate_path.exists(),
            official_evalvae_exists=official_evalvae_path.exists(),
            fid_reference_path=fid_reference_file,
        ),
        "recommended_commands": commands,
    }

    output_dir = Path(args.output_dir or make_output_root("stage2_baseline_alignment")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{args.group_id}_baseline_alignment.json"
    output_md = output_dir / f"{args.group_id}_baseline_alignment.md"
    with output_json.open("w") as handle:
        json.dump(report, handle, indent=2)
    output_md.write_text(_render_markdown(report))

    print("=" * 72)
    print("Stage2 baseline alignment report written")
    print(f"group_id      : {args.group_id}")
    print(f"output_json   : {output_json}")
    print(f"output_md     : {output_md}")
    print("=" * 72)


if __name__ == "__main__":
    main()
