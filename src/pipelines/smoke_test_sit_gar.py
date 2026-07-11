from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch


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
        description="Minimal smoke test for iFID SiT checkpoints inside GAR-FID."
    )
    parser.add_argument("--group_id", type=str, required=True, help="Registry row id.")
    parser.add_argument("--registry_path", type=str, default=None)
    parser.add_argument("--ifid_repo_root", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--label_map_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=4)
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
    parser.add_argument("--allow_missing_keys", action="store_true")
    parser.add_argument("--allow_unknown_unexpected_keys", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_uint8_images(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to(torch.float32).clamp(-1.0, 1.0)
    x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    return x


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


def save_column_images(
    output_dir: Path,
    names: List[str],
    images_by_column: Dict[str, np.ndarray],
) -> Dict[str, List[str]]:
    saved_paths: Dict[str, List[str]] = {}
    for column, images in images_by_column.items():
        column_dir = output_dir / column
        column_dir.mkdir(parents=True, exist_ok=True)
        saved_paths[column] = []
        for idx, (name, image) in enumerate(zip(names, images)):
            safe_name = Path(name).stem.replace(" ", "_")
            path = column_dir / f"{idx:02d}_{safe_name}_{column}.png"
            Image.fromarray(image).save(path)
            saved_paths[column].append(str(path))
    return saved_paths


def resolve_user_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())
    for root in (PROJECT_ROOT, SRC_ROOT):
        combined = (root / path_value).resolve()
        if combined.exists():
            return str(combined)
    return path_value


def collect_batch(dataset: PairedImageNetDataset, num_images: int) -> Dict[str, object]:
    actual_n = min(num_images, len(dataset))
    float_tensors = []
    uint8_images = []
    labels = []
    names = []
    indices = []
    for i in range(actual_n):
        img_tensor, arr_uint8, idx, label = dataset[i]
        float_tensors.append(img_tensor)
        uint8_images.append(arr_uint8)
        labels.append(label)
        indices.append(idx)
        names.append(os.path.basename(dataset.samples[i][0]))
    return {
        "float_batch": torch.stack(float_tensors, dim=0),
        "uint8_batch": np.stack(uint8_images, axis=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "indices": indices,
        "names": names,
    }


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
            "Checkpoint load reported missing keys. "
            "Refuse to continue in smoke test mode.\n"
            f"Missing keys: {missing}"
        )
    if unknown_unexpected and not allow_unknown_unexpected_keys:
        raise RuntimeError(
            "Checkpoint load reported unknown unexpected keys. "
            "Refuse to continue in smoke test mode.\n"
            f"Unknown unexpected keys: {unknown_unexpected}"
        )


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().to(torch.float32)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def resize_batch(images: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    resized = []
    for image in images:
        if image.shape[0] == target_h and image.shape[1] == target_w:
            resized.append(image)
            continue
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_w, target_h), Image.LANCZOS)
        resized.append(np.asarray(pil_image))
    return np.stack(resized, axis=0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    row = load_registry_row(args.group_id, registry_path=args.registry_path)
    dataset_dir = args.dataset_dir or row.get("imagenet_val_dir") or ""
    label_map_json = args.label_map_json or row.get("label_map_json") or ""
    if not dataset_dir:
        raise ValueError("dataset_dir is missing. Fill the registry row or pass --dataset_dir.")
    if not label_map_json:
        raise ValueError("label_map_json is missing. Fill the registry row or pass --label_map_json.")
    dataset_dir = resolve_user_path(dataset_dir)
    label_map_json = resolve_user_path(label_map_json)

    default_output_stem = f"{args.group_id}_eta{args.eta_t}_{args.sampling_mode}_{args.step_policy}"
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "results" / "smoke_test_sit_gar" / default_output_stem
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = SitGarAdapter.from_group_id(
        args.group_id,
        device=args.device,
        registry_path=args.registry_path,
        ifid_repo_root=args.ifid_repo_root or str(DEFAULT_IFID_REPO_ROOT),
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
        num_samples=args.num_images,
        label_map_json=label_map_json,
    )
    batch = collect_batch(dataset, args.num_images)
    x = batch["float_batch"].to(adapter.device)
    labels = batch["labels"].to(adapter.device)
    names = batch["names"]
    original_uint8 = batch["uint8_batch"]

    generator = torch.Generator(device=str(adapter.device))
    generator.manual_seed(args.seed)

    with torch.no_grad():
        z_vae = adapter.encode_to_vae_latents(x, labels=labels)
        x_vae_rec = adapter.decode_from_vae_latents(z_vae, labels=labels)
        gar_outputs = adapter.paired_reconstruct(
            x,
            eta_t=args.eta_t,
            labels=labels,
            num_steps=args.num_steps,
            use_cfg=args.use_cfg,
            sampling_mode=args.sampling_mode,
            step_policy=args.step_policy,
            min_num_steps=args.min_num_steps,
            max_num_steps=args.max_num_steps,
        )
        z_gen = adapter.generate(
            batch_size=x.shape[0],
            labels=labels,
            num_steps=args.num_steps,
            use_cfg=args.use_cfg,
            device=str(adapter.device),
            generator=generator,
            sampling_mode=args.sampling_mode,
            step_policy=args.step_policy,
            min_num_steps=args.min_num_steps,
            max_num_steps=args.max_num_steps,
        )
        x_gen = adapter.decode_from_model_latents(z_gen, labels=labels)

    images_by_column = {
        "original": original_uint8,
        "vae_recon": to_uint8_images(x_vae_rec),
        "gar_recon": to_uint8_images(gar_outputs.x_rec),
        "sit_gen": to_uint8_images(x_gen),
    }
    target_hw = images_by_column["vae_recon"].shape[1:3]
    for key, images in list(images_by_column.items()):
        images_by_column[key] = resize_batch(images, target_hw)
    saved_paths = save_column_images(output_dir, names, images_by_column)

    rows = []
    for idx in range(len(names)):
        rows.append([images_by_column[column][idx] for column in COLUMN_NAMES])
    grid = make_grid(rows)
    grid_path = output_dir / "smoke_grid.png"
    Image.fromarray(grid).save(grid_path)

    columns_txt = output_dir / "columns.txt"
    with columns_txt.open("w") as handle:
        for i, column in enumerate(COLUMN_NAMES):
            handle.write(f"Column {i}: {column}\n")
        handle.write(f"\neta_t={args.eta_t}\n")
        handle.write(f"sampling_mode={args.sampling_mode}\n")
        handle.write(f"step_policy={args.step_policy}\n")
        handle.write(f"num_steps={args.num_steps}\n")
        handle.write(f"resolved_gar_num_steps={gar_plan.resolved_num_steps}\n")
        handle.write(f"resolved_gen_num_steps={gen_plan.resolved_num_steps}\n")
        handle.write(f"t_raw_start={gar_plan.t_raw_start}\n")
        handle.write(f"t_shifted_start={gar_plan.t_shifted_start}\n")
        handle.write(f"time_map_abs_error={gar_plan.time_map_abs_error}\n")
        handle.write(f"use_cfg={args.use_cfg}\n")
        handle.write(f"cfg_scale={args.cfg_scale}\n")

    summary = {
        "group_id": args.group_id,
        "seed": args.seed,
        "device": str(adapter.device),
        "dataset_dir": dataset_dir,
        "label_map_json": label_map_json,
        "num_images": len(names),
        "eta_t": args.eta_t,
        "sampling_mode": args.sampling_mode,
        "step_policy": args.step_policy,
        "num_steps": args.num_steps,
        "min_num_steps": args.min_num_steps,
        "max_num_steps": args.max_num_steps,
        "resolved_num_steps": gar_outputs.resolved_num_steps,
        "use_cfg": args.use_cfg,
        "cfg_scale": args.cfg_scale,
        "t_raw_start": gar_outputs.t_raw_start,
        "t_shifted_start": gar_outputs.t_shifted_start,
        "time_map_abs_error": gar_outputs.time_map_abs_error,
        "gar_sampling_plan": gar_plan.as_dict(),
        "gen_sampling_plan": gen_plan.as_dict(),
        "spec": {
            "vae_name": adapter.spec.vae_name,
            "vae_config": adapter.spec.vae_config,
            "vae_type": adapter.spec.vae_type,
            "latent_kind": adapter.spec.latent_kind,
            "sit_scale": adapter.spec.sit_scale,
            "sit_model": adapter.spec.sit_model,
            "exp_path": adapter.spec.exp_path,
            "checkpoint_path": adapter.bundle.checkpoint_path,
        },
        "load_report": adapter.bundle.load_report,
        "labels": labels.cpu().tolist(),
        "names": names,
        "latent_shape": adapter.bundle.latent_shape,
        "latent_stats": {
            "latents_scale": summarize_tensor(adapter.bundle.latent_stats["latents_scale"]),
            "latents_bias": summarize_tensor(adapter.bundle.latent_stats["latents_bias"]),
        },
        "tensor_stats": {
            "z_vae": summarize_tensor(z_vae),
            "z_model": summarize_tensor(gar_outputs.z_model),
            "z_noisy": summarize_tensor(gar_outputs.z_noisy),
            "z_denoised_model": summarize_tensor(gar_outputs.z_denoised_model),
            "x_vae_rec": summarize_tensor(x_vae_rec),
            "x_gar_rec": summarize_tensor(gar_outputs.x_rec),
            "x_sit_gen": summarize_tensor(x_gen),
        },
        "saved_paths": {
            **saved_paths,
            "grid": str(grid_path),
            "columns_txt": str(columns_txt),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print("SiT GAR smoke test finished")
    print(f"group_id      : {args.group_id}")
    print(f"sampling_mode : {args.sampling_mode}")
    print(f"step_policy   : {args.step_policy}")
    print(f"num_steps     : requested={args.num_steps} resolved={gar_outputs.resolved_num_steps}")
    print(f"t_raw_start   : {gar_outputs.t_raw_start:.6f}")
    print(f"t_shift_start : {gar_outputs.t_shifted_start:.6f}")
    print(f"time_map_err  : {gar_outputs.time_map_abs_error:.6e}")
    print(f"output_dir    : {output_dir}")
    print(f"checkpoint    : {adapter.bundle.checkpoint_path}")
    print(f"dataset_dir   : {dataset_dir}")
    print(f"grid          : {grid_path}")
    print(f"summary       : {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
