"""
Post-train VAE Decoder to bridge the distribution gap.

This script trains the VAE decoder on iMF-processed latents while keeping
the encoder and diffusion model frozen.

Pipeline:
    x (GT) → Encoder (frozen) → z_real 
                                  ↓
                            add_noise(t)
                                  ↓
                               z_noisy
                                  ↓
                        iMF Model (frozen)
                                  ↓
                              z_denoised
                                  ↓
                        Decoder (trainable) → x_rec
                                  ↓
                        Loss(x_rec, x) + GAN Loss

Usage:
    torchrun --nproc_per_node=4 post_train.py \
        --model_type iMF-B-2 \
        --pretrained_imf_pytorch /path/to/checkpoint.pt \
        --dataset_dir /path/to/imagenet
"""

import os
import sys
import time
import warnings

# Suppress warnings before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCHDYNAMO_LOGLEVEL"] = "INFO"
warnings.filterwarnings('ignore')

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from utils.util import Logger, LossManager, save_checkpoint
from utils.distributed import init_distributed_mode
from data.dataloader import build_dataloader
from models.model import TokenizerFlowComposition
from models.losses import PostTrainingLoss
from eval_tokenizer import eval_one_epoch

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def is_main_process():
    """Check if current process is main (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def main_worker(args):
    """Main training worker."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Force train_decoder stage for post-training
    args.stage = 'train_decoder'
    
    # ============================================================
    # Setup DDP
    # ============================================================
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    if is_main_process():
        print("=" * 70)
        print("Post-training VAE Decoder")
        print("=" * 70)
        print(f"Model: {args.model_type}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Maximum noise level: {args.maximum_noise_level}")
        print(f"Normalized: {args.normalized}")
        print("=" * 70)
        
        # Initialize wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.saver_name_pre
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=vars(args),
            )
            print(f"[wandb] Initialized: {args.wandb_project}/{wandb_run_name}")
    
    # ============================================================
    # Initialize Model
    # ============================================================
    model = TokenizerFlowComposition(args)
    
    if is_main_process():
        tokenizer_para = sum(p.numel() for p in model.tokenizer.parameters())
        print(f"Tokenizer Parameters: {tokenizer_para / 1e6:.2f}M")
        
        if model.diffusion_pytorch is not None:
            print(f"Diffusion Model: {args.model_type} (frozen)")
    
    model = model.to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # ============================================================
    # Initialize Loss
    # ============================================================
    post_loss = PostTrainingLoss(args).to(device)
    
    if is_main_process():
        if post_loss.discriminator is not None:
            disc_para_count = sum(p.numel() for p in post_loss.discriminator.parameters())
            print(f"Discriminator Parameters: {disc_para_count / 1e6:.2f}M")
        else:
            print("Discriminator disabled")
    
    # ============================================================
    # Setup Optimizers
    # ============================================================
    model_para = list(model.tokenizer.vae.decoder.parameters())
    disc_para = list(post_loss.discriminator.parameters()) if post_loss.discriminator is not None else []
    
    optimizer = torch.optim.AdamW(
        model_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    optimizer_disc = (
        torch.optim.AdamW(disc_para, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        if disc_para else None
    )

    resume_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if post_loss.discriminator is not None and 'discriminator' in checkpoint:
            post_loss.discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_disc is not None and 'optimizer_disc' in checkpoint:
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        resume_epoch = int(checkpoint.get('epoch', 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {resume_epoch}")
    
    # ============================================================
    # Build Dataloaders
    # ============================================================
    train_dataloader, train_sampler, len_train_set = build_dataloader(args, split='train')
    val_dataloader, _, len_val_set = build_dataloader(args, split='val')
    
    if is_main_process():
        print(f"Train size: {len_train_set}, Val size: {len_val_set}")
    
    # ============================================================
    # Wrap with DDP
    # ============================================================
    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()
    model.module.tokenizer.vae.encoder.eval()
    if model.module.diffusion_pytorch is not None:
        model.module.diffusion_pytorch.eval()
    
    post_loss = post_loss.to(device)
    if any(parameter.requires_grad for parameter in post_loss.parameters()):
        post_loss = DDP(post_loss, device_ids=[args.gpu])
    post_loss_module = post_loss.module if isinstance(post_loss, DDP) else post_loss
    post_loss.train()
    post_loss_module.perceptual_loss.eval()
    post_loss_module.gram_loss.eval()
    
    # ============================================================
    # Training Loop
    # ============================================================
    results_eval = {'epoch': [], 'psnr': [], 'ssim': [], 'lpips': [], 'rec_loss': []}
    train_loss = LossManager()
    
    if is_main_process():
        print("\nStart training...")
    
    start_epoch = resume_epoch + 1
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        if is_main_process():
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{args.epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'=' * 60}")
        
        start_time = time.time()
        
        for step, (x, labels) in enumerate(train_dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # ============================================================
                # Forward pass
                # ============================================================
                optimizer.zero_grad()
                x_rec = model(x, labels=labels)
                
                # ============================================================
                # Generator loss
                # ============================================================
                gen_loss, gen_loss_pack = post_loss(
                    x, x_rec, 
                    optimizer_idx=0, 
                    cur_epoch=epoch, 
                    last_layer=model.module.tokenizer.last_layer
                )
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                optimizer.step()
                
                # ============================================================
                # Discriminator loss (only when active)
                # ============================================================
                disc_active = optimizer_disc is not None and epoch >= args.disc_start_epoch
                if disc_active:
                    optimizer_disc.zero_grad()
                    d_loss, d_loss_pack = post_loss(
                        x, x_rec.detach(), 
                        optimizer_idx=1, 
                        cur_epoch=epoch
                    )
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                    optimizer_disc.step()
                    
                    # Merge loss packs
                    for k, v in d_loss_pack.items():
                        gen_loss_pack[k] = v
                
                torch.cuda.synchronize()
            
            train_loss.add_loss(gen_loss_pack)
            
            if is_main_process() and (step < 10 or (step + 1) % 10 == 0):
                print(train_loss.pprint(
                    window=50, 
                    prefix=f'Epoch [{epoch}/{args.epochs}] Iter [{step+1}/{len(train_dataloader)}]'
                ))
                
                # Log to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    global_step = (epoch - 1) * len(train_dataloader) + step
                    log_dict = {
                        'train/gen_loss': train_loss.get_avg('gen_loss', window=50),
                        'train/rec_loss': train_loss.get_avg('rec_loss', window=50),
                        'train/lpips_loss': train_loss.get_avg('lpips_loss', window=50),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                    }
                    if disc_active:
                        log_dict['train/d_loss'] = train_loss.get_avg('d_loss', window=50)
                        log_dict['train/g_loss'] = train_loss.get_avg('g_loss', window=50)
                    wandb.log(log_dict, step=global_step)

            if args.max_train_steps > 0 and (step + 1) >= args.max_train_steps:
                if is_main_process():
                    print(f"Stopping epoch early after {args.max_train_steps} train steps.")
                break
        
        train_loss.clear()
        epoch_time = time.time() - start_time
        
        if is_main_process():
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        # ============================================================
        # Save checkpoint
        # ============================================================
        if epoch % args.eval_epochs == 0 and is_main_process():
            model.train()
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'checkpoint-{args.saver_name_pre}-{epoch}.pth.tar'
            )
            save_checkpoint({
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'discriminator': (
                    post_loss_module.discriminator.state_dict()
                    if post_loss_module.discriminator is not None else None
                ),
                'optimizer_disc': optimizer_disc.state_dict() if optimizer_disc is not None else None,
                'args': vars(args),
            }, is_best=False, filename=checkpoint_path)
        
        dist.barrier()
        
        # ============================================================
        # Evaluation
        # ============================================================
        if epoch % args.eval_epochs == 0:
            with torch.no_grad():
                results_pack = eval_one_epoch(args, model, epoch, val_dataloader, len_val_set)
            
            if is_main_process():
                results_eval['epoch'].append(epoch)
                results_eval['psnr'].append(results_pack.psnr)
                results_eval['ssim'].append(results_pack.ssim)
                results_eval['lpips'].append(results_pack.lpips)
                results_eval['rec_loss'].append(results_pack.rec_loss)
                
                print(f"\nEvaluation Results:")
                print(f"  PSNR: {results_pack.psnr:.2f}")
                print(f"  SSIM: {results_pack.ssim:.4f}")
                print(f"  LPIPS: {results_pack.lpips:.4f}")
                print(f"  Rec Loss: {results_pack.rec_loss:.4f}")
                
                # Log eval to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'eval/psnr': results_pack.psnr,
                        'eval/ssim': results_pack.ssim,
                        'eval/lpips': results_pack.lpips,
                        'eval/rec_loss': results_pack.rec_loss,
                        'epoch': epoch,
                    })
                
                # Save results
                import pandas as pd

                results_val_len = len(results_eval['epoch'])
                data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_len + 1))
                data_frame.to_csv(
                    f'{args.results_dir}/eval_{args.saver_name_pre}_rec_results.csv', 
                    index_label='index'
                )
    
    # ============================================================
    # Final checkpoint
    # ============================================================
    if is_main_process():
        print("\n" + "=" * 70)
        print("Training complete! Saving final checkpoint...")
        print("=" * 70)
        
        checkpoint_path = os.path.join(
            args.checkpoint_dir, 
            f'checkpoint-{args.saver_name_pre}-final.pth.tar'
        )
        save_checkpoint({
            'epoch': args.epochs,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'discriminator': (
                post_loss_module.discriminator.state_dict()
                if post_loss_module.discriminator is not None else None
            ),
            'optimizer_disc': optimizer_disc.state_dict() if optimizer_disc is not None else None,
            'args': vars(args),
        }, is_best=False, filename=checkpoint_path)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    os.environ['NCCL_TIMEOUT_IN_MS'] = '7200000'
    
    args = config.parse_arg()
    dict_args = vars(args)
    
    # Setup logging
    os.makedirs(args.saver_dir, exist_ok=True)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    
    if is_main_process():
        print("Configuration:")
        for k, v in dict_args.items():
            print(f"  {k}: {v}")
        print()
    
    main_worker(args)
