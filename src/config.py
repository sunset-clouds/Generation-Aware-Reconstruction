import os
import random
import argparse

import numpy as np
import torch
from models.imf_torch.registry import model_defaults

try:
    import ruamel.yaml as yaml
except ModuleNotFoundError:
    import yaml

def parse_arg():
    parser = argparse.ArgumentParser(description='Generation-aware reconstruction and decoder adaptation.')

    # ============================================================
    # Dataset Configuration
    # ============================================================
    parser.add_argument('--dataset_dir', default="/path/to/dataset/", type=str, 
                        help='the directory of dataset')
    parser.add_argument('--dataset_name', default='ImageNet', 
                        help='the name of dataset', 
                        choices=['ImageNet', 'FFHQ', 'CelebAHQ', 'Churches'])
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, 
                        help='number of data loader workers')
    parser.add_argument('--resolution', type=int, default=256, 
                        help='resolution of images')
    
    # ============================================================
    # Model Hyperparameters
    # ============================================================
    # VAE configuration
    parser.add_argument('--latent_size', type=int, default=32, 
                        help='resolution of latent (256/8=32 for SD VAE)')
    parser.add_argument('--vae_type', default='mse', type=str, 
                        choices=['mse', 'ema'],
                        help='VAE type from HuggingFace')
    
    # Diffusion model configuration
    parser.add_argument('--model_type', default='iMF-B-2', type=str,
                        choices=['iMF-B-2', 'iMF-M-2', 'iMF-L-2', 'iMF-XL-2'],
                        help='iMF model variant')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='number of sampling steps (NFE)')
    parser.add_argument('--maximum_noise_level', type=float, default=0.4, 
                        choices=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                        help='Maximum noise level to add to the encoder latent')
    parser.add_argument('--minimum_noise_level', type=float, default=0.0,
                        help='Minimum noise level when sampling random noise. '
                             'Only used when --fixed_noise is not set.')
    parser.add_argument('--fixed_noise', action='store_true', default=False,
                        help='use fixed noise level instead of random U[min, max]. '
                             'When set, every batch uses exactly maximum_noise_level.')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='whether or not normalize the noised latent')                  
    
    # CFG parameters (will be auto-set based on model_type if not specified)
    parser.add_argument('--omega', type=float, default=None,
                        help='CFG scale (auto-set if None)')
    parser.add_argument('--t_min', type=float, default=None,
                        help='CFG interval start (auto-set if None)')
    parser.add_argument('--t_max', type=float, default=None,
                        help='CFG interval end (auto-set if None)')

    # ============================================================
    # Loss Hyperparameters (for training decoder)
    # ============================================================
    parser.add_argument('--rec_weight', type=float, default=1.0,
                        help='weight for reconstruction loss (MSE)')
    parser.add_argument('--perceptual_weight', type=float, default=1.0,
                        help='weight for perceptual loss (LPIPS)')
    parser.add_argument('--disc_weight', type=float, default=0.7,
                        help='weight for discriminator loss (for reconstruction)')
    parser.add_argument('--disc_start_epoch', type=int, default=0,
                        help='epoch to start discriminator training')
    parser.add_argument('--lecam_loss_weight', type=float, default=0.001,
                        help='weight for LeCAM regularization')
    parser.add_argument('--disc_cr_loss_weight', type=float, default=4.0, help='disc_cr_loss_weight')
    parser.add_argument('--gram_loss_weight', type=float, default=10.0, help="gram_loss weight")
    
    # ============================================================
    # Training Configuration
    # ============================================================
    parser.add_argument('--stage', default='eval', type=str,
                        choices=['eval', 'train_decoder'],
                        help='stage: eval for FID analysis, train_decoder for decoder finetuning')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs for ImageNet.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate for decoder refinement')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay for optimizer')
    parser.add_argument('--eval_epochs', type=int, default=1,
                        help='epochs between evaluations (0 = every epoch)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to a post-training checkpoint to resume from')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='limit training steps per epoch for smoke tests (0 = no limit)')
    parser.add_argument('--max_eval_steps', type=int, default=0,
                        help='limit eval steps for smoke tests (0 = no limit)')
    
    # Debug mode flags
    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='debug mode: disable noise+iMF and GAN, only use rec loss (for sanity check)')
    parser.add_argument('--disable_gan', action='store_true', default=False,
                        help='disable GAN loss (only rec + perceptual loss)')
    parser.add_argument('--_skip_diffusion_load', action='store_true', default=False,
                        help='skip loading diffusion model (for debug)')

    # ============================================================
    # Checkpoint and Results Paths
    # ============================================================
    parser.add_argument('--saver_name_pre', default='', type=str,
                        help='prefix for saved checkpoint names')
    parser.add_argument('--checkpoint_dir', default="./checkpoints/", type=str,
                        help='directory for saving checkpoints')
    parser.add_argument('--results_dir', default="./results/", type=str,
                        help='directory for saving results')
    parser.add_argument('--saver_dir', default="./saver/", type=str,
                        help='directory for saving training records')
    parser.add_argument('--yaml_dir', default="./yaml/", type=str,
                        help='directory for saving yaml file')
    parser.add_argument('--pretrained_imf_pytorch', default="", type=str,
                        help='path to official or converted iMF PyTorch checkpoint')
    parser.add_argument('--pretrained_decoder', default="", type=str,
                        help='path to post-trained decoder checkpoint (for evaluation)')

    # ============================================================
    # Seed Configuration
    # ============================================================
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    # ============================================================
    # Logging Configuration
    # ============================================================
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='decoder-adaptation',
                        help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='',
                        help='wandb run name (auto-generated if empty)')

    # ============================================================
    # Distributed Training
    # ============================================================
    parser.add_argument('--local-rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend.')
    args = parser.parse_args()

    if not args.pretrained_imf_pytorch and not args._skip_diffusion_load and not args.debug_mode:
        parser.error('--pretrained_imf_pytorch is required unless diffusion loading is disabled')

    # Handle distributed training environment variables
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.workers = min(max(0, args.workers), args.batch_size)
    
    # Create yaml directory
    os.makedirs(args.yaml_dir, exist_ok=True)
    os.makedirs(args.saver_dir, exist_ok=True)

    # ============================================================
    # Auto-set CFG parameters based on model type
    # ============================================================
    cfg_defaults = model_defaults(args.model_type)
    
    if args.omega is None:
        args.omega = cfg_defaults['omega']
    if args.t_min is None:
        args.t_min = cfg_defaults['t_min']
    if args.t_max is None:
        args.t_max = cfg_defaults['t_max']
    
    # ============================================================
    # Create directories
    # ============================================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Generate saver_name_pre if not provided
    if not args.saver_name_pre:
        args.saver_name_pre = 'IMF_Decoder_GAP_{}_{}_{}_{}_{}'.format(
            args.dataset_name, args.model_type, args.num_steps,
            args.maximum_noise_level, args.normalized
        )

    dict_args = vars(args)
    config_name = args.saver_name_pre+'.yaml'
    with open(os.path.join(args.yaml_dir, config_name), 'w', encoding='utf-8') as f:
        if hasattr(yaml, 'YAML'):
            file_yaml = yaml.YAML()
            file_yaml.dump(dict_args, f)
        else:
            yaml.safe_dump(dict_args, f, sort_keys=False)

    # ============================================================
    # Set random seed
    # ============================================================
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return args


if __name__ == '__main__':
    args = parse_arg()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
