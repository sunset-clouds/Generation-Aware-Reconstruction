"""
Evaluation functions for the tokenizer model.

Computes PSNR, SSIM, LPIPS, and reconstruction loss on validation set.
Supports distributed evaluation with all_reduce.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import distributed as tdist
from tqdm import tqdm

from utils.util import Pack
def is_main_process():
    """Check if current process is main (rank 0)."""
    if tdist.is_available() and tdist.is_initialized():
        return tdist.get_rank() == 0
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def eval_one_epoch(args, model, epoch, val_dataloader, len_val_set):
    """
    Evaluate model on validation set.
    
    Args:
        args: Configuration
        model: DDP-wrapped model
        epoch: Current epoch
        val_dataloader: Validation dataloader
        len_val_set: Total size of validation set
    Returns:
        Pack with psnr, ssim, lpips, rec_loss
    """
    from metric.metric import PSNR, LPIPS, SSIM

    model.eval()
    
    # Initialize metrics
    device = next(model.parameters()).device
    psnr_metric = PSNR(device=device)
    ssim_metric = SSIM()
    lpips_metric = LPIPS(device=device)
    
    ssim, psnr, lpips, rec_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0
    
    for step, (x, labels) in enumerate(tqdm(val_dataloader, desc=f"Eval epoch {epoch}", disable=not is_main_process())):
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        x = x.cuda(local_rank, non_blocking=True)
        labels = labels.cuda(local_rank, non_blocking=True)
        batch_size = x.size(0)
        
        with torch.no_grad():
            # Get reconstruction and loss
            x_rec, rec_loss_eval = model.module.collect_eval_info(x, labels=labels)
            
            # Normalize to [0, 1] for metrics
            x_norm = (x + 1.0) / 2.0
            x_rec_norm = (x_rec + 1.0) / 2.0
            x_rec_norm = torch.clamp(x_rec_norm, 0, 1)
            
            # Compute metrics
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()
            batch_rec_loss = rec_loss_eval.detach() * batch_size
            batch_total_num = torch.tensor(batch_size, device=device, dtype=torch.float32)
        
        # All-reduce across GPUs
        if tdist.is_initialized():
            handler1 = tdist.all_reduce(batch_lpips, async_op=True)
            handler2 = tdist.all_reduce(batch_psnr, async_op=True)
            handler3 = tdist.all_reduce(batch_ssim, async_op=True)
            handler4 = tdist.all_reduce(batch_rec_loss, async_op=True)
            handler5 = tdist.all_reduce(batch_total_num, async_op=True)
            handler1.wait()
            handler2.wait()
            handler3.wait()
            handler4.wait()
            handler5.wait()
        
        ssim += batch_ssim.item()
        psnr += batch_psnr.item()
        lpips += batch_lpips.item()
        rec_loss += batch_rec_loss.item()
        total_num += batch_total_num.item()

        if args.max_eval_steps > 0 and (step + 1) >= args.max_eval_steps:
            break
    
    # Compute averages
    if total_num <= 0:
        raise RuntimeError("Evaluation produced zero samples on this rank.")
    metric_num = total_num if args.max_eval_steps > 0 else len_val_set
    eval_psnr = psnr / metric_num
    eval_ssim = ssim / metric_num
    eval_lpips = lpips / metric_num
    eval_rec_loss = rec_loss / total_num
    
    # Restore training mode
    model.train()
    model.module.tokenizer.vae.encoder.eval()
    if model.module.diffusion_pytorch is not None:
        model.module.diffusion_pytorch.eval()
    
    return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)
