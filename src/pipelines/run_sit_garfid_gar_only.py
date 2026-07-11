from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT
WORKSPACE_ROOT = PROJECT_ROOT
DEFAULT_IFID_REPO_ROOT = PROJECT_ROOT / "third_party" / "ifid"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrations.sit import SitGarAdapter, load_registry_row  # noqa: E402
from pipelines.run_sit_garfid_single import (  # noqa: E402
    collect_names,
    compute_fid,
    fail_on_load_issues,
    make_grid,
    resolve_user_path,
    save_npz,
    save_uint8_batch,
    seed_everything,
    to_uint8_images,
    resize_batch,
)
from scripts.imagenet_val_labeled_dataset import PairedImageNetDataset  # noqa: E402


PREVIEW_COLUMNS = ("original", "gar_recon")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-model single-noise GAR-only validation for IFID SiT checkpoints. "
            "This runner only saves GAR reconstructions and only computes GAR-FID."
        )
    )
    parser.add_argument("--group_id", type=str, required=True, help="Registry row id.")
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--label_map_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--reference_image_dir",
        type=str,
        default=None,
        help=(
            "Existing directory of original PNGs to use as FID reference. "
            "When omitted, this runner writes `input_png` locally."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eta_t", type=float, default=0.3)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--sampling_mode", type=str, choices=("ode", "sde"), default="sde")
    parser.add_argument("--step_policy", type=str, choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--min_num_steps", type=int, default=8)
    parser.add_argument("--max_num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fid_batch_size", type=int, default=50)
    parser.add_argument("--fid_num_workers", type=int, default=8)
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save NHWC uint8 NPZ files for GAR reconstructions (and originals if written locally).",
    )
    parser.add_argument("--preview_count", type=int, default=0, help="Rows to include in preview grid.")
    parser.add_argument("--allow_missing_keys", action="store_true")
    parser.add_argument("--allow_unknown_unexpected_keys", action="store_true")
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Resume an interrupted run by skipping samples whose output PNGs already exist.",
    )
    return parser.parse_args()


def count_pngs(path: Path) -> int:
    return sum(1 for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() == ".png")


def output_image_path(output_dir: Path, name: str, idx: int, column: str) -> Path:
    stem = Path(name).stem.replace(" ", "_")
    return output_dir / f"{int(idx):06d}_{stem}_{column}.png"


def complete_png(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def batch_outputs_complete(
    *,
    names: List[str],
    indices: List[int],
    gar_image_dir: Path,
    reference_image_dir: Path,
    save_reference_png: bool,
) -> bool:
    for name, idx in zip(names, indices):
        if save_reference_png and not complete_png(output_image_path(reference_image_dir, name, idx, "original")):
            return False
        if not complete_png(output_image_path(gar_image_dir, name, idx, "gar_recon")):
            return False
    return True


def load_dataset_batch(
    dataset: PairedImageNetDataset,
    indices: List[int],
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    float_tensors = []
    uint8_images = []
    labels = []
    for idx in indices:
        img_tensor, arr_uint8, _, label = dataset[idx]
        float_tensors.append(img_tensor)
        uint8_images.append(arr_uint8)
        labels.append(label)
    return (
        torch.stack(float_tensors, dim=0),
        np.stack(uint8_images, axis=0),
        torch.tensor(labels, dtype=torch.long),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.resume_existing and args.save_npz:
        raise ValueError("--resume_existing is not currently compatible with --save_npz.")

    row = load_registry_row(args.group_id, registry_path=args.registry_path)
    ifid_repo_root = resolve_user_path(
        args.ifid_repo_root or "", extra_roots=[PROJECT_ROOT]
    ) if args.ifid_repo_root else str(DEFAULT_IFID_REPO_ROOT.resolve())
    dataset_dir = resolve_user_path(
        args.dataset_dir or row.get("imagenet_val_dir") or "",
        extra_roots=[Path(ifid_repo_root)],
    )
    label_map_json = resolve_user_path(
        args.label_map_json or row.get("label_map_json") or "",
        extra_roots=[Path(ifid_repo_root)],
    )
    if not dataset_dir:
        raise ValueError("dataset_dir is missing. Fill the registry row or pass --dataset_dir.")
    if not label_map_json:
        raise ValueError("label_map_json is missing. Fill the registry row or pass --label_map_json.")

    default_output_stem = f"{args.group_id}_eta{args.eta_t}_{args.sampling_mode}_{args.step_policy}_gar_only"
    default_output_dir = PROJECT_ROOT / "results" / "sit_garfid_gar_only" / default_output_stem
    output_dir = Path(args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_image_dir:
        reference_image_dir = Path(resolve_user_path(args.reference_image_dir)).resolve()
        if not reference_image_dir.exists():
            raise FileNotFoundError(f"reference_image_dir does not exist: {reference_image_dir}")
        save_reference_png = False
        reference_mode = "reuse_existing"
    else:
        reference_image_dir = (output_dir / "input_png").resolve()
        reference_image_dir.mkdir(parents=True, exist_ok=True)
        save_reference_png = True
        reference_mode = "write_local"

    gar_image_dir = (output_dir / "gar_recon_png").resolve()
    gar_image_dir.mkdir(parents=True, exist_ok=True)

    adapter = SitGarAdapter.from_group_id(
        args.group_id,
        device=args.device,
        registry_path=args.registry_path,
        ifid_repo_root=ifid_repo_root,
        default_num_steps=args.num_steps,
        default_cfg_scale=args.cfg_scale,
    )
    fail_on_load_issues(
        adapter.bundle.load_report,
        allow_missing_keys=args.allow_missing_keys,
        allow_unknown_unexpected_keys=args.allow_unknown_unexpected_keys,
    )
    gar_plan = adapter.resolve_sampling_plan(
        args.eta_t,
        args.num_steps,
        args.sampling_mode,
        step_policy=args.step_policy,
        min_num_steps=args.min_num_steps,
        max_num_steps=args.max_num_steps,
    )

    dataset = PairedImageNetDataset(
        dataset_dir,
        num_samples=args.num_samples,
        label_map_json=label_map_json,
    )
    expected_samples = len(dataset)
    if not save_reference_png:
        reference_png_count = count_pngs(reference_image_dir)
        if reference_png_count != expected_samples:
            raise ValueError(
                "reference_image_dir sample count mismatch: "
                f"expected {expected_samples} PNGs, found {reference_png_count} in {reference_image_dir}"
            )

    npz_buffers: Dict[str, List[np.ndarray]] = {"gar_recon": []}
    if args.save_npz and save_reference_png:
        npz_buffers["original"] = []

    preview_rows: List[List[np.ndarray]] = []
    preview_remaining = max(0, args.preview_count)
    processed = 0
    resumed_existing_samples = 0
    resumed_existing_batches = 0

    batch_starts = list(range(0, len(dataset), args.batch_size))
    with torch.no_grad():
        for batch_start in tqdm(batch_starts, desc="Run GAR-FID GAR-only point"):
            indices_list = list(range(batch_start, min(batch_start + args.batch_size, len(dataset))))
            names = collect_names(dataset, indices_list)
            if args.resume_existing and batch_outputs_complete(
                names=names,
                indices=indices_list,
                gar_image_dir=gar_image_dir,
                reference_image_dir=reference_image_dir,
                save_reference_png=save_reference_png,
            ):
                resumed_existing_samples += len(indices_list)
                resumed_existing_batches += 1
                processed += len(indices_list)
                continue

            batch_float, batch_uint8, batch_labels = load_dataset_batch(dataset, indices_list)
            batch_float = batch_float.to(adapter.device)
            batch_labels = batch_labels.to(adapter.device)

            gar_outputs = adapter.paired_reconstruct(
                batch_float,
                eta_t=args.eta_t,
                labels=batch_labels,
                num_steps=args.num_steps,
                use_cfg=args.use_cfg,
                sampling_mode=args.sampling_mode,
                step_policy=args.step_policy,
                min_num_steps=args.min_num_steps,
                max_num_steps=args.max_num_steps,
            )

            images_by_column = {
                "original": batch_uint8,
                "gar_recon": to_uint8_images(gar_outputs.x_rec),
            }
            target_hw = images_by_column["gar_recon"].shape[1:3]
            for key, images in list(images_by_column.items()):
                images_by_column[key] = resize_batch(images, target_hw)

            if save_reference_png:
                save_uint8_batch(reference_image_dir, names, indices_list, images_by_column["original"], "original")
                if args.save_npz:
                    npz_buffers["original"].append(images_by_column["original"])
            save_uint8_batch(gar_image_dir, names, indices_list, images_by_column["gar_recon"], "gar_recon")
            if args.save_npz:
                npz_buffers["gar_recon"].append(images_by_column["gar_recon"])

            if preview_remaining > 0:
                take = min(preview_remaining, len(indices_list))
                for i in range(take):
                    preview_rows.append([images_by_column[column][i] for column in PREVIEW_COLUMNS])
                preview_remaining -= take

            processed += len(indices_list)

    if processed != expected_samples:
        raise RuntimeError(
            f"Expected to account for {expected_samples} samples, but only accounted for {processed}."
        )

    if args.save_npz:
        gar_images = (
            np.concatenate(npz_buffers["gar_recon"], axis=0)
            if npz_buffers["gar_recon"]
            else np.empty((0, 256, 256, 3), dtype=np.uint8)
        )
        save_npz(output_dir / f"GAR_reconstruction_t{args.eta_t}_{args.group_id}.npz", gar_images)
        if save_reference_png:
            original_images = (
                np.concatenate(npz_buffers["original"], axis=0)
                if npz_buffers["original"]
                else np.empty((0, 256, 256, 3), dtype=np.uint8)
            )
            save_npz(output_dir / "Input.npz", original_images)

    if preview_rows:
        preview_grid = make_grid(preview_rows)
        preview_path = output_dir / "preview_grid.png"
        Image.fromarray(preview_grid).save(preview_path)
    else:
        preview_path = None

    device_for_fid = str(adapter.device)
    garfid = compute_fid(
        ifid_repo_root,
        str(reference_image_dir),
        str(gar_image_dir),
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
        device=device_for_fid,
        sample_count=processed,
    )

    summary = {
        "group_id": args.group_id,
        "seed": args.seed,
        "device": device_for_fid,
        "dataset_dir": dataset_dir,
        "label_map_json": label_map_json,
        "num_samples": processed,
        "batch_size": args.batch_size,
        "eta_t": args.eta_t,
        "sampling_mode": args.sampling_mode,
        "step_policy": args.step_policy,
        "num_steps": args.num_steps,
        "min_num_steps": args.min_num_steps,
        "max_num_steps": args.max_num_steps,
        "tshift": adapter.tshift,
        "t_raw_start": gar_plan.t_raw_start,
        "t_shifted_start": gar_plan.t_shifted_start,
        "time_map_abs_error": gar_plan.time_map_abs_error,
        "resolved_num_steps": gar_plan.resolved_num_steps,
        "gar_sampling_plan": gar_plan.as_dict(),
        "use_cfg": args.use_cfg,
        "cfg_scale": args.cfg_scale,
        "resume_existing": args.resume_existing,
        "resumed_existing_samples": resumed_existing_samples,
        "resumed_existing_batches": resumed_existing_batches,
        "reference_mode": reference_mode,
        "reference_image_dir": str(reference_image_dir),
        "image_dirs": {
            "reference": str(reference_image_dir),
            "gar_recon": str(gar_image_dir),
        },
        "spec": {
            "vae_name": adapter.spec.vae_name,
            "vae_config": adapter.spec.vae_config,
            "vae_type": adapter.spec.vae_type,
            "latent_kind": adapter.spec.latent_kind,
            "sit_scale": adapter.spec.sit_scale,
            "sit_model": adapter.spec.sit_model,
            "exp_path": adapter.spec.exp_path,
            "checkpoint_path": adapter.bundle.checkpoint_path,
            "latent_shape": adapter.bundle.latent_shape,
        },
        "load_report": adapter.bundle.load_report,
        "metrics": {
            "garfid": garfid,
        },
    }
    if preview_path is not None:
        summary["preview_grid"] = str(preview_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print("SiT GAR-only single-point validation finished")
    print(f"group_id      : {args.group_id}")
    print(f"sampling_mode : {args.sampling_mode}")
    print(f"step_policy   : {args.step_policy}")
    print(f"tshift        : {adapter.tshift:.6f}")
    print(f"eta_t         : {args.eta_t}")
    print(f"t_raw_start   : {gar_plan.t_raw_start:.6f}")
    print(f"t_shift_start : {gar_plan.t_shifted_start:.6f}")
    print(f"time_map_err  : {gar_plan.time_map_abs_error:.6e}")
    print(f"num_steps     : requested={args.num_steps} resolved={gar_plan.resolved_num_steps}")
    print(f"num_samples   : {processed}")
    print(f"resumed_exist : samples={resumed_existing_samples} batches={resumed_existing_batches}")
    print(f"reference_dir : {reference_image_dir}")
    print(f"output_dir    : {output_dir}")
    print(f"garfid        : {garfid:.4f}")
    print(f"summary       : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
