"""
Distributed training utilities.

Same as VQ-Transplant/VAR/utils/distributed.py
"""

import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initialize distributed training.
    
    Args:
        args: Arguments with local_rank, etc.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = getattr(args, 'dist_backend', 'nccl')
    
    print(f'| distributed init (rank {args.rank}): gpu {args.gpu}', flush=True)
    
    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """Get world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
