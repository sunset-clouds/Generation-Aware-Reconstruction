"""
Evaluate reconstruction metrics for decoder adaptation.

Default denoising labels: **exact** class per image (``--label_map_json`` / PairedImageNetDataset).
Legacy options: ``--label_mode image_order`` or ``random`` (not recommended).

Supports three reconstruction modes:
  - vanilla_rfid:   x -> Encoder -> Decoder
  - our_rfid_nocfg: x -> Encoder -> add_noise(t) -> iMF denoise(no CFG) -> Decoder
  - our_rfid_cfg:   x -> Encoder -> add_noise(t) -> iMF denoise(CFG) -> Decoder

Outputs:
  - reconstruction_metrics.csv
  - reconstruction_metrics.json
  - optional NPZ files compatible with scripts/compute_fid.py

Example:
  python scripts/eval_reconstruction_metrics.py \
    --model_type iMF-B-2 \
    --pretrained_imf_pytorch checkpoints/pytorch/iMF-B-2.pth \
    --dataset_dir /path/to/imagenet/val \
    --output_dir ./results/recon_metrics/imf-b-2 \
    --modes vanilla_rfid,our_rfid_nocfg \
    --noise_levels 0.2 \
    --save_npz
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric.metric import LPIPS, PSNR, SSIM
from models.model import TokenizerFlowComposition
from scripts.generate_eval_images import save_npz, to_uint8
from scripts.imagenet_dir_label_map import DEFAULT_LABEL_MAP_JSON
from scripts.imagenet_val_labeled_dataset import PairedImageNetDataset


ALL_MODES = ("vanilla_rfid", "our_rfid_nocfg", "our_rfid_cfg")


def build_model(args, device: torch.device):
    args.use_pytorch_imf = True
    args.vae_type = "mse"
    model = TokenizerFlowComposition(args).to(device)
    model.eval_mode()

    if getattr(model, "vae_wrapper", None) is not None:
        model.vae_wrapper.compiled_decode = model.vae_wrapper.vae.decode
    return model


def prepare_labels(
    label_mode: str,
    idx: torch.Tensor,
    all_labels: torch.Tensor,
    batch_labels: Optional[torch.Tensor],
    device: torch.device,
):
    if label_mode == "exact":
        if batch_labels is None:
            raise ValueError("label_mode=exact requires labels from PairedImageNetDataset.")
        return batch_labels.to(device)
    if label_mode == "image_order":
        # Legacy synthetic schedule (does NOT match alphabetical val folder order).
        idx = idx.to(all_labels.device)
        return all_labels[idx].to(device)
    if label_mode == "random":
        return torch.randint(0, 1000, (idx.shape[0],), device=device)
    raise ValueError(f"Unknown label mode: {label_mode}")


@torch.no_grad()
def reconstruct_batch(
    model,
    batch_float: torch.Tensor,
    idx: torch.Tensor,
    mode: str,
    noise_level: float,
    label_mode: str,
    all_labels: torch.Tensor,
    batch_labels: Optional[torch.Tensor],
    device: torch.device,
):
    model_u = model.module if hasattr(model, "module") else model

    if mode == "vanilla_rfid":
        return model_u.vae_reconstruction(batch_float)

    z_real = model_u.tokenizer.encode(batch_float)
    z_noisy, _ = model_u.diffusion_pytorch.add_noise(z_real, noise_level)
    labels = prepare_labels(label_mode, idx, all_labels, batch_labels, device)
    z_den = model_u.diffusion_pytorch.denoise(
        z_noisy,
        noise_level,
        labels,
        num_steps=1,
        use_cfg=(mode == "our_rfid_cfg"),
    )
    return model_u.decode_latent(z_den)


def evaluate_mode(
    model,
    dataloader,
    mode: str,
    noise_level: float,
    label_mode: str,
    all_labels: torch.Tensor,
    device: torch.device,
    save_recon_npz: bool,
    output_dir: str,
    model_type: str,
    use_per_image_labels: bool,
):
    psnr_metric = PSNR(device=device)
    ssim_metric = SSIM()
    lpips_metric = LPIPS(device=device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_mse = 0.0
    total_count = 0

    input_images = []
    output_images = []
    input_saved = os.path.exists(os.path.join(output_dir, "Input.npz"))

    desc = mode if mode == "vanilla_rfid" else f"{mode}@t={noise_level}"
    for batch in tqdm(dataloader, desc=f"Recon {desc}"):
        if use_per_image_labels:
            batch_float, batch_uint8, idx, blab = batch
        else:
            batch_float, batch_uint8, idx = batch
            blab = None
        batch_float = batch_float.to(device)
        idx = idx.to(device)

        x_rec = reconstruct_batch(
            model=model,
            batch_float=batch_float,
            idx=idx,
            mode=mode,
            noise_level=noise_level,
            label_mode=label_mode,
            all_labels=all_labels,
            batch_labels=blab,
            device=device,
        )

        x_norm = torch.clamp((batch_float + 1.0) / 2.0, 0.0, 1.0)
        x_rec_norm = torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)

        batch_psnr = psnr_metric(x_norm, x_rec_norm)
        batch_ssim = ssim_metric(x_norm, x_rec_norm)
        batch_lpips = lpips_metric(x_norm, x_rec_norm)
        batch_mse = F.mse_loss(x_norm, x_rec_norm, reduction="none").mean(dim=(1, 2, 3))

        total_psnr += float(batch_psnr.sum().item())
        total_ssim += float(batch_ssim.sum().item())
        total_lpips += float(batch_lpips.sum().item())
        total_mse += float(batch_mse.sum().item())
        total_count += batch_float.shape[0]

        if save_recon_npz:
            if not input_saved:
                input_images.append(batch_uint8.numpy())
            output_images.append(to_uint8(x_rec))

    if save_recon_npz:
        if not input_saved:
            save_npz(np.concatenate(input_images, axis=0), os.path.join(output_dir, "Input.npz"))

        if mode == "vanilla_rfid":
            fname = "VAE_reconstruction.npz"
        else:
            cfg_tag = "CFG" if mode == "our_rfid_cfg" else "noCFG"
            fname = f"Denoising_t{noise_level}_{cfg_tag}_{model_type}.npz"
        save_npz(np.concatenate(output_images, axis=0), os.path.join(output_dir, fname))

    return {
        "mode": mode,
        "noise_level": None if mode == "vanilla_rfid" else float(noise_level),
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
        "lpips": total_lpips / total_count,
        "mse": total_mse / total_count,
        "num_samples": total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS for reconstruction modes.")
    parser.add_argument("--model_type", type=str, required=True, choices=["iMF-B-2", "iMF-M-2", "iMF-L-2", "iMF-XL-2"])
    parser.add_argument("--pretrained_imf_pytorch", type=str, required=True)
    parser.add_argument("--pretrained_decoder", type=str, default="",
                        help="Optional post-trained decoder checkpoint.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--label_map_json",
        type=str,
        default=DEFAULT_LABEL_MAP_JSON,
        help="Exact val folder -> class index JSON (default: scripts/imagenet_val_dir_to_index.json).",
    )
    parser.add_argument("--output_dir", type=str, default="./results/reconstruction_metrics")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--modes",
        type=str,
        default="vanilla_rfid,our_rfid_nocfg",
        help="Comma-separated from: vanilla_rfid,our_rfid_nocfg,our_rfid_cfg",
    )
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="0.2",
        help="Comma-separated noise levels for GAR modes.",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="exact",
        choices=("exact", "image_order", "random"),
        help="Denoising labels: exact=folder map (default); image_order/rand=legacy.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save Input.npz / VAE_reconstruction.npz / Denoising_t*.npz for compute_fid.py.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ALL_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Choices: {ALL_MODES}")
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

    use_exact = args.label_mode == "exact"
    if use_exact:
        dataset = PairedImageNetDataset(
            args.dataset_dir,
            args.num_samples,
            label_map_json=args.label_map_json,
        )
    else:
        from scripts.generate_eval_images import ImageNetDataset

        dataset = ImageNetDataset(args.dataset_dir, args.num_samples)
    n_img = len(dataset)
    if n_img == 0:
        raise RuntimeError(f"No images found under {args.dataset_dir!r}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
    )

    effective_n = len(dataset)
    num_classes = 1000
    per_class = max(1, effective_n // num_classes)
    all_labels = torch.arange(num_classes).repeat_interleave(per_class)[:effective_n]

    print("=" * 70)
    print("Reconstruction Metrics Evaluation")
    print("=" * 70)
    print(f"Model:       {args.model_type}")
    print(f"Dataset:     {args.dataset_dir}")
    print(f"Num samples: {effective_n}")
    print(f"Modes:       {modes}")
    print(f"Noise:       {noise_levels}")
    print(f"Label mode:  {args.label_mode}" + ("  (per-image from label_map_json)" if use_exact else ""))
    print(f"Save NPZ:    {args.save_npz}")
    print("=" * 70)

    model = build_model(args, device)

    if args.pretrained_decoder:
        print(f"Loading post-trained decoder from {args.pretrained_decoder}")
        checkpoint = torch.load(args.pretrained_decoder, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        tokenizer_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            if key.startswith("tokenizer."):
                tokenizer_state_dict[key] = value

        model_state = model.state_dict()
        loaded = 0
        for key, value in tokenizer_state_dict.items():
            if key in model_state:
                model_state[key] = value
                loaded += 1
        model.load_state_dict(model_state)
        print(f"Loaded {loaded} tokenizer weights")

    rows: List[Dict] = []
    for mode in modes:
        if mode == "vanilla_rfid":
            rows.append(
                evaluate_mode(
                    model=model,
                    dataloader=dataloader,
                    mode=mode,
                    noise_level=0.0,
                    label_mode=args.label_mode,
                    all_labels=all_labels,
                    device=device,
                    save_recon_npz=args.save_npz,
                    output_dir=args.output_dir,
                    model_type=args.model_type,
                    use_per_image_labels=use_exact,
                )
            )
        else:
            for noise_level in noise_levels:
                rows.append(
                    evaluate_mode(
                        model=model,
                        dataloader=dataloader,
                        mode=mode,
                        noise_level=noise_level,
                        label_mode=args.label_mode,
                        all_labels=all_labels,
                        device=device,
                        save_recon_npz=args.save_npz,
                        output_dir=args.output_dir,
                        model_type=args.model_type,
                        use_per_image_labels=use_exact,
                    )
                )

    for row in rows:
        noise_str = "-" if row["noise_level"] is None else f"{row['noise_level']:.2f}"
        print(
            f"{row['mode']:15s} noise={noise_str:>5s}  "
            f"PSNR={row['psnr']:.4f}  SSIM={row['ssim']:.4f}  "
            f"LPIPS={row['lpips']:.4f}  MSE={row['mse']:.6f}"
        )

    csv_path = os.path.join(args.output_dir, "reconstruction_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "noise_level", "psnr", "ssim", "lpips", "mse", "num_samples"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "reconstruction_metrics.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved JSON to {json_path}")
    if args.save_npz:
        print("Saved NPZ files compatible with scripts/compute_fid.py")


if __name__ == "__main__":
    main()
