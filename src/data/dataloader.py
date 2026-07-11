"""
Data loading utilities for decoder adaptation.

Follows the same structure as VQ-Transplant/VAR/data/dataloader.py
"""
import os
import sys  
import torch
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode, transforms
from .augmentation import random_crop_arr, center_crop_arr
from .lsun_church import LSUNChurchesDataset
from .lsun_bedroom import LSUNBedroomsDataset

DATASET_PATHS = {
    "ImageNet": "imagenet",
    "FFHQ": "FFHQ",
    "CelebAHQ": "CelebAHQ",
    "Churches": "LSUN-Churches",
    "Bedrooms":"LSUN-Bedrooms",
}

def build_train_transform(resolution=256):
    """Build training transform with random crop and flip."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def build_eval_transform(resolution=256):
    """Build evaluation transform with center crop."""
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def build_dataloader(args, split='train'):
    """
    Build dataloader for training or evaluation.
    
    Args:
        args: Configuration arguments
        split: 'train' or 'val'
        
    Returns:
        dataloader: DataLoader object
        sampler: DistributedSampler (if distributed)
        dataset_size: Size of the dataset
    """
    if split == 'train':
        transform = build_train_transform(args.resolution)
    else:
        transform = build_eval_transform(args.resolution)
    
    if args.dataset_name == "ImageNet":
        # Support both structures:
        # 1. dataset_dir/imagenet/train, dataset_dir/imagenet/val
        # 2. dataset_dir/train, dataset_dir/validation
        split_name = 'train' if split == 'train' else 'validation'
        
        # Try direct path first (e.g., ${IMAGENET_TRAIN})
        data_path = os.path.join(args.dataset_dir, split_name)
        if not os.path.exists(data_path):
            # Try with 'val' instead of 'validation'
            data_path = os.path.join(args.dataset_dir, 'val' if split != 'train' else 'train')
        if not os.path.exists(data_path):
            # Try nested structure (dataset_dir/imagenet/train)
            data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name], split_name)
        if not os.path.exists(data_path):
            data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name], 'val' if split != 'train' else 'train')
        
        print(f"Loading ImageNet from: {data_path}")
        dataset = ImageFolder(root=data_path, transform=transform)
    elif args.dataset_name in ["FFHQ", "CelebAHQ"]:
        dataset = ImageFolder(root=data_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    print(f"Dataset: {args.dataset_name}, split: {split}, size: {len(dataset)}")
    
    # Check if distributed
    distributed = torch.distributed.is_initialized()
    
    # Use 'spawn' multiprocessing to avoid JAX fork issues
    # Or set num_workers=0 for single-process loading (safer with JAX)
    num_workers = args.workers if not hasattr(args, 'use_jax') or not args.use_jax else 0
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(split == 'train'))
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0),
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0),
        )
    
    return dataloader, sampler, len(dataset)


def load_dataset(args, batch_size=16, split='val'):
    """
    Load dataset for evaluation (non-distributed).
    
    Args:
        args: Configuration arguments
        batch_size: Batch size
        split: 'train' or 'val'
        
    Returns:
        dataloader: DataLoader object
        dataset_size: Size of the dataset
    """
    data_path = os.path.join(args.dataset_dir, DATASET_PATHS[args.dataset_name])
    transform = build_eval_transform(args.resolution)
    
    if args.dataset_name == "ImageNet":
        dataset = ImageFolder(
            root=os.path.join(data_path, 'val' if split == 'val' else 'train'),
            transform=transform
        )
    elif args.dataset_name in ["FFHQ", "CelebAHQ"]:
        dataset = ImageFolder(root=data_path, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )
    
    return dataloader, len(dataset)
