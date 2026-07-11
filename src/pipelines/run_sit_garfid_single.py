from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT
WORKSPACE_ROOT = PROJECT_ROOT
DEFAULT_IFID_REPO_ROOT = PROJECT_ROOT / "third_party" / "ifid"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from integrations.sit import SitGarAdapter, load_registry_row  # noqa: E402
from scripts.imagenet_val_labeled_dataset import PairedImageNetDataset  # noqa: E402


COLUMN_NAMES = ("original", "vae_recon", "gar_recon", "sit_gen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-model single-noise GAR-FID validation for IFID SiT checkpoints."
    )
    parser.add_argument("--group_id", type=str, required=True, help="Registry row id.")
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--label_map_json", type=str, default=None)
    parser.add_argument("--fid_reference_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
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
    parser.add_argument("--save_npz", action="store_true", help="Also save NHWC uint8 NPZ files.")
    parser.add_argument("--preview_count", type=int, default=4, help="Rows to include in preview grid.")
    parser.add_argument("--allow_missing_keys", action="store_true")
    parser.add_argument("--allow_unknown_unexpected_keys", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_user_path(path_value: str, extra_roots: Iterable[Path] | None = None) -> str:
    if not path_value:
        return path_value
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())
    roots = [PROJECT_ROOT, SRC_ROOT]
    if extra_roots:
        roots.extend(extra_roots)
    for root in roots:
        combined = (root / path_value).resolve()
        if combined.exists():
            return str(combined)
    return path_value


def resolve_reference_path(
    path_value: str,
    *,
    ifid_repo_root: str,
) -> str:
    extra_roots = [
        Path(ifid_repo_root),
        Path(ifid_repo_root) / "metric",
        SRC_ROOT / "metric",
        PROJECT_ROOT / "metric",
    ]
    resolved = resolve_user_path(path_value, extra_roots=extra_roots)
    if not Path(resolved).exists():
        raise FileNotFoundError(
            f"FID reference file not found: {path_value!r} -> {resolved!r}"
        )
    return resolved


def to_uint8_images(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to(torch.float32).clamp(-1.0, 1.0)
    x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    return x


def resize_batch(images: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for image in images:
        if image.shape[:2] == (target_h, target_w):
            resized.append(image)
            continue
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_w, target_h), Image.LANCZOS)
        resized.append(np.asarray(pil_image))
    return np.stack(resized, axis=0)


def save_uint8_batch(
    output_dir: Path,
    names: List[str],
    indices: List[int],
    images: np.ndarray,
    column: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, idx, image in zip(names, indices, images):
        stem = Path(name).stem.replace(" ", "_")
        path = output_dir / f"{int(idx):06d}_{stem}_{column}.png"
        Image.fromarray(image).save(path)


def npz_name_for(column: str, *, eta_t: float, use_cfg: bool, group_id: str) -> str:
    cfg_tag = "CFG" if use_cfg else "noCFG"
    if column == "original":
        return "Input.npz"
    if column == "vae_recon":
        return "VAE_reconstruction.npz"
    if column == "gar_recon":
        return f"GAR_reconstruction_t{eta_t}_{cfg_tag}_{group_id}.npz"
    if column == "sit_gen":
        return f"Generated_{cfg_tag}_{group_id}.npz"
    raise KeyError(column)


def save_npz(path: Path, images: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, arr_0=images)


def make_grid(rows: List[List[np.ndarray]], pad: int = 4, pad_color: int = 220) -> np.ndarray:
    if not rows or not rows[0]:
        raise ValueError("rows must contain at least one image")
    h, w, c = rows[0][0].shape
    n_rows = len(rows)
    n_cols = len(rows[0])
    grid = np.full(
        (n_rows * h + (n_rows + 1) * pad, n_cols * w + (n_cols + 1) * pad, c),
        fill_value=pad_color,
        dtype=np.uint8,
    )
    for r, row in enumerate(rows):
        for c_idx, img in enumerate(row):
            y0 = pad + r * (h + pad)
            x0 = pad + c_idx * (w + pad)
            grid[y0 : y0 + h, x0 : x0 + w] = img
    return grid


def fail_on_load_issues(
    load_report: Dict[str, object],
    *,
    allow_missing_keys: bool,
    allow_unknown_unexpected_keys: bool,
) -> None:
    missing = list(load_report.get("missing_keys", []))
    unknown_unexpected = list(load_report.get("unknown_unexpected_keys", []))
    if missing and not allow_missing_keys:
        raise RuntimeError(
            "Checkpoint load reported missing keys.\n"
            f"Missing keys: {missing}"
        )
    if unknown_unexpected and not allow_unknown_unexpected_keys:
        raise RuntimeError(
            "Checkpoint load reported unknown unexpected keys.\n"
            f"Unknown unexpected keys: {unknown_unexpected}"
        )


def compute_fid(
    ifid_repo_root: str,
    reference_path: str,
    sample_path: str,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
    sample_count: int,
) -> float:
    if str(ifid_repo_root) not in sys.path:
        sys.path.insert(0, str(ifid_repo_root))
    from ifid.fid.fid import calculate_fid_given_paths  # type: ignore

    return float(
        calculate_fid_given_paths(
            [reference_path, sample_path],
            batch_size=batch_size,
            dims=2048,
            device=device,
            num_workers=num_workers,
            sp_len=sample_count,
        )
    )


def collect_names(dataset: PairedImageNetDataset, indices: List[int]) -> List[str]:
    names = []
    for idx in indices:
        names.append(os.path.basename(dataset.samples[int(idx)][0]))
    return names


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

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

    fid_reference_value = (
        args.fid_reference_file
        or row.get("fid_reference_file")
        or "VIRTUAL_imagenet256_labeled.npz"
    )
    fid_reference_file = resolve_reference_path(
        str(fid_reference_value),
        ifid_repo_root=ifid_repo_root,
    )

    default_output_stem = f"{args.group_id}_eta{args.eta_t}_{args.sampling_mode}_{args.step_policy}"
    default_output_dir = PROJECT_ROOT / "results" / "sit_garfid_single" / default_output_stem
    output_dir = Path(args.output_dir or default_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    gen_plan = adapter.resolve_sampling_plan(
        1.0,
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
    )

    image_dirs = {
        "original": output_dir / "input_png",
        "vae_recon": output_dir / "vae_recon_png",
        "gar_recon": output_dir / "gar_recon_png",
        "sit_gen": output_dir / "sit_gen_png",
    }
    for path in image_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    npz_buffers = {column: [] for column in COLUMN_NAMES} if args.save_npz else None
    preview_rows: List[List[np.ndarray]] = []
    preview_remaining = max(0, args.preview_count)

    generator = torch.Generator(device=str(adapter.device))
    generator.manual_seed(args.seed)

    processed = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Run GAR-FID single point"):
            batch_float, batch_uint8, batch_indices, batch_labels = batch
            batch_float = batch_float.to(adapter.device)
            batch_labels = batch_labels.to(adapter.device)

            z_vae = adapter.encode_to_vae_latents(batch_float, labels=batch_labels)
            x_vae_rec = adapter.decode_from_vae_latents(z_vae, labels=batch_labels)
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
            z_gen = adapter.generate(
                batch_size=batch_float.shape[0],
                labels=batch_labels,
                num_steps=args.num_steps,
                use_cfg=args.use_cfg,
                device=str(adapter.device),
                generator=generator,
                sampling_mode=args.sampling_mode,
                step_policy=args.step_policy,
                min_num_steps=args.min_num_steps,
                max_num_steps=args.max_num_steps,
            )
            x_gen = adapter.decode_from_model_latents(z_gen, labels=batch_labels)

            images_by_column = {
                "original": batch_uint8.numpy(),
                "vae_recon": to_uint8_images(x_vae_rec),
                "gar_recon": to_uint8_images(gar_outputs.x_rec),
                "sit_gen": to_uint8_images(x_gen),
            }
            target_hw = images_by_column["vae_recon"].shape[1:3]
            for key, images in list(images_by_column.items()):
                images_by_column[key] = resize_batch(images, target_hw)

            indices_list = [int(i) for i in batch_indices.tolist()]
            names = collect_names(dataset, indices_list)
            for column, images in images_by_column.items():
                save_uint8_batch(image_dirs[column], names, indices_list, images, column)
                if npz_buffers is not None:
                    npz_buffers[column].append(images)

            if preview_remaining > 0:
                take = min(preview_remaining, len(indices_list))
                for i in range(take):
                    preview_rows.append([images_by_column[column][i] for column in COLUMN_NAMES])
                preview_remaining -= take

            processed += len(indices_list)

    if npz_buffers is not None:
        for column, chunks in npz_buffers.items():
            images = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 256, 256, 3), dtype=np.uint8)
            save_npz(output_dir / npz_name_for(column, eta_t=args.eta_t, use_cfg=args.use_cfg, group_id=args.group_id), images)

    if preview_rows:
        preview_grid = make_grid(preview_rows)
        preview_path = output_dir / "preview_grid.png"
        Image.fromarray(preview_grid).save(preview_path)
    else:
        preview_path = None

    device_for_fid = str(adapter.device)
    vanilla_rfid = compute_fid(
        ifid_repo_root,
        str(image_dirs["original"]),
        str(image_dirs["vae_recon"]),
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
        device=device_for_fid,
        sample_count=processed,
    )
    garfid = compute_fid(
        ifid_repo_root,
        str(image_dirs["original"]),
        str(image_dirs["gar_recon"]),
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
        device=device_for_fid,
        sample_count=processed,
    )
    gfid = compute_fid(
        ifid_repo_root,
        fid_reference_file,
        str(image_dirs["sit_gen"]),
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
        device=device_for_fid,
        sample_count=processed,
    )

    metrics = {
        "vanilla_rfid": vanilla_rfid,
        "garfid": garfid,
        "gfid": gfid,
        "gar_minus_vanilla_rfid": garfid - vanilla_rfid,
    }

    summary = {
        "group_id": args.group_id,
        "seed": args.seed,
        "device": device_for_fid,
        "dataset_dir": dataset_dir,
        "label_map_json": label_map_json,
        "fid_reference_file": fid_reference_file,
        "num_samples": processed,
        "batch_size": args.batch_size,
        "eta_t": args.eta_t,
        "sampling_mode": args.sampling_mode,
        "step_policy": args.step_policy,
        "num_steps": args.num_steps,
        "min_num_steps": args.min_num_steps,
        "max_num_steps": args.max_num_steps,
        "tshift": adapter.tshift,
        "t_raw_start": gar_outputs.t_raw_start,
        "t_shifted_start": gar_outputs.t_shifted_start,
        "time_map_abs_error": gar_outputs.time_map_abs_error,
        "resolved_num_steps": gar_outputs.resolved_num_steps,
        "gar_sampling_plan": gar_plan.as_dict(),
        "gen_sampling_plan": gen_plan.as_dict(),
        "use_cfg": args.use_cfg,
        "cfg_scale": args.cfg_scale,
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
        "image_dirs": {key: str(path) for key, path in image_dirs.items()},
        "metrics": metrics,
    }
    if preview_path is not None:
        summary["preview_grid"] = str(preview_path)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print("SiT GAR single-point validation finished")
    print(f"group_id      : {args.group_id}")
    print(f"sampling_mode : {args.sampling_mode}")
    print(f"step_policy   : {args.step_policy}")
    print(f"tshift        : {adapter.tshift:.6f}")
    print(f"eta_t         : {args.eta_t}")
    print(f"t_raw_start   : {gar_outputs.t_raw_start:.6f}")
    print(f"t_shift_start : {gar_outputs.t_shifted_start:.6f}")
    print(f"time_map_err  : {gar_outputs.time_map_abs_error:.6e}")
    print(f"num_steps     : requested={args.num_steps} resolved={gar_outputs.resolved_num_steps}")
    print(f"gen_steps     : requested={gen_plan.requested_num_steps} resolved={gen_plan.resolved_num_steps}")
    print(f"num_samples   : {processed}")
    print(f"output_dir    : {output_dir}")
    print(f"vanilla_rfid  : {vanilla_rfid:.4f}")
    print(f"garfid        : {garfid:.4f}")
    print(f"gfid          : {gfid:.4f}")
    print(f"summary       : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
