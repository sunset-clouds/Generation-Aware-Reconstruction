"""
FID computation utilities.

Uses the same implementation as iMF codebase (JAX-based InceptionV3).
"""

import os
import sys
import numpy as np

# Add imf to path for imports
IMF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'imf')
if IMF_PATH not in sys.path:
    sys.path.insert(0, IMF_PATH)

from utils.fid_util import (
    compute_fid,
    build_jax_inception,
    get_reference,
    compute_stats,
    compute_inception_score,
)


def load_fid_stats(stats_path):
    """
    Load precomputed FID statistics.
    
    Args:
        stats_path: Path to .npz file with mu and sigma
        
    Returns:
        dict with 'mu' and 'sigma'
    """
    return get_reference(stats_path)


def save_fid_stats(mu, sigma, stats_path):
    """
    Save FID statistics to file.
    
    Args:
        mu: Mean
        sigma: Covariance
        stats_path: Output path
    """
    np.savez(stats_path, ref_mu=mu, ref_sigma=sigma)
    print(f"Saved FID stats to {stats_path}")


class FIDEvaluator:
    """
    FID evaluator using Inception network (same as iMF).
    
    Usage:
        evaluator = FIDEvaluator(ref_stats_path)
        fid = evaluator.compute_fid(generated_images)
    """
    
    def __init__(self, ref_stats_path, batch_size=200, fid_samples=50000):
        """
        Initialize FID evaluator.
        
        Args:
            ref_stats_path: Path to reference statistics (.npz)
            batch_size: Batch size for Inception
            fid_samples: Number of samples to use for FID
        """
        self.batch_size = batch_size
        self.fid_samples = fid_samples
        
        # Load reference statistics
        print(f"Loading reference stats from {ref_stats_path}")
        self.ref_stats = load_fid_stats(ref_stats_path)
        
        # Build Inception network
        print("Building Inception network...")
        self.inception_net = build_jax_inception(batch_size=batch_size)
        print("Inception network ready")
    
    def compute_stats_from_images(self, images):
        """
        Compute FID statistics for a set of images.
        
        Args:
            images: numpy array of shape (N, H, W, 3) with values in [0, 255] (uint8)
            
        Returns:
            dict with 'mu', 'sigma', and optionally 'logits'
        """
        return compute_stats(
            images,
            self.inception_net,
            batch_size=self.batch_size,
            fid_samples=min(self.fid_samples, len(images)),
        )
    
    def compute_fid_from_images(self, images):
        """
        Compute FID between generated images and reference.
        
        Args:
            images: numpy array of shape (N, H, W, 3) with values in [0, 255] (uint8)
            
        Returns:
            FID score
        """
        stats = self.compute_stats_from_images(images)
        fid = compute_fid(
            self.ref_stats["mu"], stats["mu"],
            self.ref_stats["sigma"], stats["sigma"]
        )
        return fid
    
    def compute_inception_score(self, images, splits=10):
        """
        Compute Inception Score for images.
        
        Args:
            images: numpy array of shape (N, H, W, 3) with values in [0, 255] (uint8)
            splits: Number of splits for IS computation
            
        Returns:
            (is_mean, is_std)
        """
        stats = self.compute_stats_from_images(images)
        is_mean, is_std = compute_inception_score(stats["logits"], splits=splits)
        return is_mean, is_std


# Convenience function for quick FID computation
def evaluate_fid(images, ref_stats_path, batch_size=200, fid_samples=50000):
    """
    Quick FID evaluation.
    
    Args:
        images: numpy array of shape (N, H, W, 3) with values in [0, 255]
        ref_stats_path: Path to reference statistics
        batch_size: Batch size for Inception
        fid_samples: Number of samples for FID
        
    Returns:
        FID score
    """
    evaluator = FIDEvaluator(ref_stats_path, batch_size, fid_samples)
    return evaluator.compute_fid_from_images(images)
