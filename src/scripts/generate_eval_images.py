"""
Generate images for FID evaluation. Supports DDP for multi-GPU.

Five evaluation modes (selectable via --modes):
  vanilla_rfid:    x -> Encoder -> z -> Decoder -> x'
  our_rfid_nocfg:  x -> Encoder -> z -> noise(t) -> Denoise(no CFG) -> Decoder -> x'
                   (uses **true class labels** from --label_map_json per val image)
  our_rfid_cfg:    x -> Encoder -> z -> noise(t) -> Denoise(CFG) -> Decoder -> x'
                   (same true labels as nocfg)
  gfid_nocfg:      noise -> Generate(no CFG) -> Decoder -> x
  gfid_cfg:        noise -> Generate(CFG) -> Decoder -> x

Output NPZ files (uint8 NHWC, key='arr_0'):
  Input.npz                           -- reference images (for rFID)
  VAE_reconstruction.npz              -- vanilla rFID
  Denoising_t{t}_noCFG_{model}.npz    -- our rFID without CFG
  Denoising_t{t}_CFG_{model}.npz      -- our rFID with CFG
  Generated_noCFG_{model}.npz         -- gFID without CFG
  Generated_CFG_{model}.npz           -- gFID with CFG

Usage:
    # Official .pth checkpoint (recommended):
    torchrun --nproc_per_node=4 scripts/generate_eval_images.py \
        --model_type iMF-B-2 \
        --pretrained_imf_pytorch checkpoints/pytorch/iMF-B-2.pth \
        --dataset_dir /path/to/imagenet/val \
        --output_dir ./results/fid/baseline \
        --modes all --num_samples 50000

    # All modes, single GPU:
    python scripts/generate_eval_images.py \
        --model_type iMF-B-2 \
        --pretrained_imf_pytorch checkpoints/pytorch/iMF-B-2.pth \
        --dataset_dir /path/to/imagenet/val \
        --output_dir ./results/fid/baseline

    # Selective modes, multi-GPU:
    torchrun --nproc_per_node=4 scripts/generate_eval_images.py \
        --model_type iMF-B-2 \
        --pretrained_imf_pytorch checkpoints/pytorch/iMF-B-2.pth \
        --dataset_dir /path/to/imagenet/val \
        --output_dir ./results/fid/baseline \
        --modes gfid_cfg,gfid_nocfg \
        --num_samples 50000

    # With post-trained decoder:
    torchrun --nproc_per_node=4 scripts/generate_eval_images.py \
        --model_type iMF-B-2 \
        --pretrained_imf_pytorch checkpoints/pytorch/iMF-B-2.pth \
        --pretrained_decoder /path/to/decoder_ckpt.pth.tar \
        --dataset_dir /path/to/imagenet/val \
        --output_dir ./results/fid/ours
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.imagenet_val_crop import center_crop_arr
from scripts.imagenet_val_paths import list_imagenet_val_image_paths
from scripts.imagenet_dir_label_map import DEFAULT_LABEL_MAP_JSON
from scripts.imagenet_val_labeled_dataset import PairedImageNetDataset

from models.model import TokenizerFlowComposition
from models.diffusion_pytorch import DiffusionModelPyTorch
from models.tokenizer import Tokenizer

ALL_MODES = [
    'vanilla_rfid',
    'our_rfid_nocfg',
    'our_rfid_cfg',
    'gfid_nocfg',
    'gfid_cfg',
]


# ============================================================
# Distributed helpers
# ============================================================

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ============================================================
# Dataset
# ============================================================


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_samples=50000, image_size=256):
        self.image_size = image_size
        self.image_paths = list_imagenet_val_image_paths(root_dir, num_samples)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        arr = center_crop_arr(img, self.image_size)
        arr_uint8 = arr.copy()
        arr_float = arr.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr_float).permute(2, 0, 1), arr_uint8, idx


# ============================================================
# Utilities
# ============================================================

def to_uint8(tensor):
    """Convert [-1, 1] tensor to uint8 numpy array (NHWC)."""
    if tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    x = tensor.cpu().numpy()
    x = (x + 1.0) / 2.0
    return np.round(np.clip(x * 255.0, 0, 255)).astype(np.uint8)


class MockArgs:
    """Minimal args for model initialization."""
    def __init__(self, model_type, pretrained_imf_pytorch, pretrained_decoder=None,
                 latent_size=32, vae_type='mse', cfg_omega=None,
                 cfg_t_min=None, cfg_t_max=None):
        self.model_type = model_type
        self.pretrained_imf_pytorch = pretrained_imf_pytorch
        self.pretrained_decoder = pretrained_decoder if pretrained_decoder else ""
        self.latent_size = latent_size
        self.vae_type = vae_type
        if cfg_omega is not None:
            self.cfg_omega = cfg_omega
        if cfg_t_min is not None:
            self.cfg_t_min = cfg_t_min
        if cfg_t_max is not None:
            self.cfg_t_max = cfg_t_max
        self.use_pytorch_imf = True
        self.maximum_noise_level = 0.2
        self.normalized = False
        self.stage = 'eval'
        self._skip_diffusion_load = False
        self.debug_mode = False


def save_npz(images, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, arr_0=images)
    print_main(f"Saved {len(images)} images to {path}")


def gather_images(local_images, local_indices, world_size, rank, total_samples):
    if world_size == 1:
        return local_images

    all_images_list = [None] * world_size
    all_indices_list = [None] * world_size

    dist.gather_object(local_images, all_images_list if rank == 0 else None, dst=0)
    dist.gather_object(local_indices, all_indices_list if rank == 0 else None, dst=0)

    if rank == 0:
        combined_images = np.concatenate(all_images_list, axis=0)
        combined_indices = []
        for idx_list in all_indices_list:
            combined_indices.extend(idx_list)
        sorted_order = np.argsort(combined_indices)
        return combined_images[sorted_order][:total_samples]

    return None


# ============================================================
# Generation functions
# ============================================================

@torch.no_grad()
def generate_reconstruction(model, dataloader, device, mode, noise_level=0.2,
                            world_size=1, rank=0):
    """Generate reconstructed images for rFID modes."""
    model_u = model.module if hasattr(model, 'module') else model

    local_input_images = []
    local_output_images = []
    local_indices = []

    desc = f"Generating ({mode})" if rank == 0 else None
    iterator = tqdm(dataloader, desc=desc, disable=(rank != 0))

    for batch in iterator:
        if len(batch) == 4:
            batch_float, batch_uint8, indices, batch_labels = batch
            batch_labels = batch_labels.to(device)
        else:
            batch_float, batch_uint8, indices = batch
            batch_labels = None
        batch_float = batch_float.to(device)
        B = batch_float.shape[0]

        local_input_images.append(batch_uint8.numpy())
        local_indices.extend(indices.tolist())

        if mode == 'vanilla_rfid':
            x_rec = model_u.vae_reconstruction(batch_float)
            local_output_images.append(to_uint8(x_rec))

        elif mode == 'our_rfid_nocfg':
            z_real = model_u.tokenizer.encode(batch_float)
            z_noisy, _ = model_u.diffusion_pytorch.add_noise(z_real, noise_level)
            labels = batch_labels if batch_labels is not None else torch.randint(0, 1000, (B,), device=device)
            z_den = model_u.diffusion_pytorch.denoise(
                z_noisy, noise_level, labels, num_steps=1, use_cfg=False
            )
            x_rec = model_u.decode_latent(z_den)
            local_output_images.append(to_uint8(x_rec))

        elif mode == 'our_rfid_cfg':
            z_real = model_u.tokenizer.encode(batch_float)
            z_noisy, _ = model_u.diffusion_pytorch.add_noise(z_real, noise_level)
            labels = batch_labels if batch_labels is not None else torch.randint(0, 1000, (B,), device=device)
            z_den = model_u.diffusion_pytorch.denoise(
                z_noisy, noise_level, labels, num_steps=1, use_cfg=True
            )
            x_rec = model_u.decode_latent(z_den)
            local_output_images.append(to_uint8(x_rec))

    local_input_images = np.concatenate(local_input_images, axis=0)
    local_output_images = np.concatenate(local_output_images, axis=0)
    return local_input_images, local_output_images, local_indices


@torch.no_grad()
def generate_samples_distributed(model, num_samples, batch_size, device, use_cfg=True,
                                 world_size=1, rank=0):
    """Generate samples from pure noise for gFID (distributed)."""
    model_u = model.module if hasattr(model, 'module') else model

    samples_per_rank = (num_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, num_samples)
    local_num_samples = end_idx - start_idx

    if local_num_samples <= 0:
        return np.array([]), []

    output_images = []
    local_indices = list(range(start_idx, end_idx))
    num_batches = (local_num_samples + batch_size - 1) // batch_size

    # For the standard 50k ImageNet setting this gives 50 samples per class.
    # For small smoke tests, repeat the class cycle so every requested sample has a label.
    num_classes = 1000
    repeats_per_class = max(1, (num_samples + num_classes - 1) // num_classes)
    all_labels = torch.arange(num_classes).repeat_interleave(repeats_per_class)[:num_samples]
    if all_labels.numel() < num_samples:
        all_labels = torch.arange(num_classes).repeat((num_samples + num_classes - 1) // num_classes)[:num_samples]
    local_labels = all_labels[local_indices].to(device)

    cfg_str = "with CFG" if use_cfg else "no CFG"
    iterator = tqdm(range(num_batches), desc=f"Generating ({cfg_str})", disable=(rank != 0))

    generator = torch.Generator(device=device)

    for i in iterator:
        B = min(batch_size, local_num_samples - i * batch_size)
        batch_labels = local_labels[i * batch_size : i * batch_size + B]
        batch_indices = local_indices[i * batch_size : i * batch_size + B]
        
        # Seeds for reproducibility (match official: seeds = sample_idx ^ initial_seed)
        seeds = (torch.tensor(batch_indices, device=device, dtype=torch.int64) ^ 42)
        generator.manual_seed(42 + batch_indices[0])
        
        z_gen = model_u.diffusion_pytorch.generate(
            B, labels=batch_labels, num_steps=1, use_cfg=use_cfg, device=device,
            generator=generator, seeds=seeds
        )
        x_gen = model_u.decode_latent(z_gen)
        output_images.append(to_uint8(x_gen))

    output_images = np.concatenate(output_images, axis=0)[:local_num_samples]
    return output_images, local_indices


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate images for FID evaluation (DDP)")
    parser.add_argument('--model_type', type=str, default='iMF-B-2',
                        choices=['iMF-B-2', 'iMF-M-2', 'iMF-L-2', 'iMF-XL-2'])
    parser.add_argument('--pretrained_imf_pytorch', type=str, required=True,
                        help='Path to PyTorch iMF checkpoint')
    parser.add_argument('--pretrained_decoder', type=str, default=None,
                        help='Path to post-trained decoder checkpoint')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Path to ImageNet validation directory (required for rFID modes)')
    parser.add_argument(
        '--label_map_json',
        type=str,
        default=DEFAULT_LABEL_MAP_JSON,
        help='For rFID: exact val folder -> class index JSON (PairedImageNetDataset).',
    )
    parser.add_argument('--output_dir', type=str, default='./results/fid',
                        help='Output directory for NPZ files')
    parser.add_argument('--modes', type=str, default='all',
                        help='Comma-separated modes: vanilla_rfid, our_rfid_nocfg, '
                             'our_rfid_cfg, gfid_nocfg, gfid_cfg, or "all"')
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--noise_levels', type=str, default='0.2',
                        help='Comma-separated noise levels for denoising rFID')
    parser.add_argument('--vae_type', type=str, default='mse',
                        choices=['mse', 'ema'])
    parser.add_argument('--cfg_omega', type=float, default=None,
                        help='Override iMF CFG scale omega for CFG modes')
    parser.add_argument('--cfg_t_min', type=float, default=None,
                        help='Override lower endpoint of CFG guidance interval')
    parser.add_argument('--cfg_t_max', type=float, default=None,
                        help='Override upper endpoint of CFG guidance interval')
    args = parser.parse_args()

    if (args.cfg_t_min is None) != (args.cfg_t_max is None):
        print("ERROR: --cfg_t_min and --cfg_t_max must be provided together")
        sys.exit(1)
    if args.cfg_t_min is not None and not (0.0 <= args.cfg_t_min <= args.cfg_t_max <= 1.0):
        print("ERROR: require 0 <= --cfg_t_min <= --cfg_t_max <= 1")
        sys.exit(1)

    # Parse modes
    if args.modes.strip().lower() == 'all':
        modes = ALL_MODES
    else:
        modes = [m.strip() for m in args.modes.split(',')]
        for m in modes:
            if m not in ALL_MODES:
                print(f"ERROR: Unknown mode '{m}'. Available: {ALL_MODES}")
                sys.exit(1)

    noise_levels = [float(x) for x in args.noise_levels.split(',')]
    needs_dataset = any(m in modes for m in ['vanilla_rfid', 'our_rfid_nocfg', 'our_rfid_cfg'])

    if needs_dataset and not args.dataset_dir:
        print("ERROR: --dataset_dir required for rFID modes")
        sys.exit(1)

    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print_main("=" * 70)
    print_main("Generate Eval Images")
    print_main("=" * 70)
    print_main(f"Model:       {args.model_type}")
    print_main(f"Modes:       {modes}")
    print_main(f"Num samples: {args.num_samples}")
    print_main(f"Output:      {args.output_dir}")
    print_main(f"GPUs:        {world_size}")
    if args.pretrained_decoder:
        print_main(f"Decoder:     {args.pretrained_decoder}")
    if args.cfg_omega is not None or args.cfg_t_min is not None:
        print_main(
            "CFG override: "
            f"omega={args.cfg_omega if args.cfg_omega is not None else 'default'}, "
            f"t_min={args.cfg_t_min if args.cfg_t_min is not None else 'default'}, "
            f"t_max={args.cfg_t_max if args.cfg_t_max is not None else 'default'}"
        )
    print_main("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True) if rank == 0 else None
    if world_size > 1:
        dist.barrier()

    # Build model
    mock_args = MockArgs(
        model_type=args.model_type,
        pretrained_imf_pytorch=args.pretrained_imf_pytorch,
        pretrained_decoder=args.pretrained_decoder,
        vae_type=args.vae_type,
        cfg_omega=args.cfg_omega,
        cfg_t_min=args.cfg_t_min,
        cfg_t_max=args.cfg_t_max,
    )
    model = TokenizerFlowComposition(mock_args)

    if args.pretrained_decoder:
        print_main(f"Loading post-trained decoder from {args.pretrained_decoder}")
        checkpoint = torch.load(args.pretrained_decoder, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        tokenizer_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if k.startswith('tokenizer.'):
                tokenizer_state_dict[k] = v

        model_state = model.state_dict()
        for k, v in tokenizer_state_dict.items():
            if k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state)
        print_main(f"Loaded {len(tokenizer_state_dict)} tokenizer weights")

    model = model.to(device)
    model.eval()

    # ============================================================
    # rFID modes (need ImageNet dataset)
    # ============================================================
    rfid_modes = [m for m in modes if m in ['vanilla_rfid', 'our_rfid_nocfg', 'our_rfid_cfg']]

    if rfid_modes:
        dataset = PairedImageNetDataset(
            args.dataset_dir,
            args.num_samples,
            label_map_json=args.label_map_json,
        )
        print_main(f"Loaded {len(dataset)} images from {args.dataset_dir} ({dataset.label_mapping_description})")

        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            sampler = None

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True
        )

        input_saved = False
        epoch_counter = 0

        for mode in rfid_modes:
            if world_size > 1 and sampler is not None:
                sampler.set_epoch(epoch_counter)
                epoch_counter += 1

            if mode == 'vanilla_rfid':
                local_inp, local_out, local_idx = generate_reconstruction(
                    model, dataloader, device, mode='vanilla_rfid',
                    world_size=world_size, rank=rank
                )
                if world_size > 1:
                    dist.barrier()
                    inp_imgs = gather_images(local_inp, local_idx, world_size, rank, args.num_samples)
                    out_imgs = gather_images(local_out, local_idx, world_size, rank, args.num_samples)
                else:
                    inp_imgs, out_imgs = local_inp, local_out

                if rank == 0:
                    if not input_saved:
                        save_npz(inp_imgs, os.path.join(args.output_dir, 'Input.npz'))
                        input_saved = True
                    save_npz(out_imgs, os.path.join(args.output_dir, 'VAE_reconstruction.npz'))

                if world_size > 1:
                    dist.barrier()

            elif mode in ('our_rfid_nocfg', 'our_rfid_cfg'):
                for noise_level in noise_levels:
                    local_inp, local_out, local_idx = generate_reconstruction(
                        model, dataloader, device, mode=mode,
                        noise_level=noise_level,
                        world_size=world_size, rank=rank
                    )
                    if world_size > 1:
                        dist.barrier()
                        inp_imgs = gather_images(local_inp, local_idx, world_size, rank, args.num_samples)
                        out_imgs = gather_images(local_out, local_idx, world_size, rank, args.num_samples)
                    else:
                        inp_imgs, out_imgs = local_inp, local_out

                    if rank == 0:
                        if not input_saved:
                            save_npz(inp_imgs, os.path.join(args.output_dir, 'Input.npz'))
                            input_saved = True

                        cfg_tag = "noCFG" if mode == 'our_rfid_nocfg' else "CFG"
                        fname = f"Denoising_t{noise_level}_{cfg_tag}_{args.model_type}.npz"
                        save_npz(out_imgs, os.path.join(args.output_dir, fname))

                    if world_size > 1:
                        dist.barrier()

    # ============================================================
    # gFID modes (generate from noise)
    # ============================================================
    if 'gfid_cfg' in modes:
        local_gen, local_idx = generate_samples_distributed(
            model, args.num_samples, args.batch_size, device,
            use_cfg=True, world_size=world_size, rank=rank
        )
        if world_size > 1:
            dist.barrier()
            gen_imgs = gather_images(local_gen, local_idx, world_size, rank, args.num_samples)
        else:
            gen_imgs = local_gen

        if rank == 0:
            save_npz(gen_imgs, os.path.join(
                args.output_dir, f'Generated_CFG_{args.model_type}.npz'))

        if world_size > 1:
            dist.barrier()

    if 'gfid_nocfg' in modes:
        local_gen, local_idx = generate_samples_distributed(
            model, args.num_samples, args.batch_size, device,
            use_cfg=False, world_size=world_size, rank=rank
        )
        if world_size > 1:
            dist.barrier()
            gen_imgs = gather_images(local_gen, local_idx, world_size, rank, args.num_samples)
        else:
            gen_imgs = local_gen

        if rank == 0:
            save_npz(gen_imgs, os.path.join(
                args.output_dir, f'Generated_noCFG_{args.model_type}.npz'))

        if world_size > 1:
            dist.barrier()

    # ============================================================
    # Summary
    # ============================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
        print("\nGenerated files:")
        for f in sorted(os.listdir(args.output_dir)):
            if f.endswith('.npz'):
                fpath = os.path.join(args.output_dir, f)
                size_gb = os.path.getsize(fpath) / 1e9
                print(f"  {f} ({size_gb:.2f} GB)")
        print("=" * 70)
        print("\nNext step: compute FID using scripts/compute_fid.py")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
