"""
PyTorch Diffusion Model wrapper for iMF.

This is a pure PyTorch implementation for converted iMF checkpoints.
Implements the DiffusionInterface for pluggable eval pipeline support.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from models.imf_torch.converted_arch import MODEL_CONFIGS, MiT_PyTorch
from models.imf_torch.registry import model_defaults


class DiffusionInterface(ABC):
    """
    Abstract interface for diffusion models used in the eval pipeline.
    All diffusion models (iMF, SiT, etc.) must implement this interface
    so that generate_eval_images.py and compute_fid.py work generically.
    """

    @abstractmethod
    def generate(self, batch_size: int, labels: Optional[torch.Tensor] = None,
                 num_steps: int = 1, use_cfg: bool = True,
                 device: str = 'cuda', generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate latents from pure noise. Returns NCHW tensor."""
        ...

    @abstractmethod
    def denoise(self, z_noisy: torch.Tensor, noise_level: float,
                labels: Optional[torch.Tensor] = None,
                num_steps: int = 1) -> torch.Tensor:
        """Denoise a noisy latent. Returns NCHW tensor."""
        ...

    @staticmethod
    @abstractmethod
    def add_noise(z: torch.Tensor, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latent. Returns (z_noisy, noise)."""
        ...


class DiffusionModelPyTorch(nn.Module, DiffusionInterface):
    """
    PyTorch implementation of iMF diffusion model.
    Implements DiffusionInterface for the unified eval pipeline.
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.checkpoint_path = getattr(args, 'pretrained_imf_pytorch', '')
        self.latent_size = getattr(args, 'latent_size', 32)
        self.num_classes = 1000
        
        # Get CFG parameters. Command-line eval jobs can override these through
        # MockArgs without mutating the official defaults below.
        cfg_params = model_defaults(self.model_type)
        self.omega = float(getattr(args, 'cfg_omega', cfg_params['omega']))
        self.t_min = float(getattr(args, 'cfg_t_min', cfg_params['t_min']))
        self.t_max = float(getattr(args, 'cfg_t_max', cfg_params['t_max']))

        # Create model
        model_cfg = MODEL_CONFIGS[self.model_type]
        self.model = MiT_PyTorch(
            input_size=self.latent_size,
            patch_size=2,
            in_channels=4,
            hidden_size=model_cfg['hidden_size'],
            num_shared_blocks=model_cfg['num_shared_blocks'],
            num_head_blocks=model_cfg['num_head_blocks'],
            num_heads=model_cfg['num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            num_classes=self.num_classes,
        )
        
        self._is_loaded = False
    
    def load(self, checkpoint_path=None):
        """Load pretrained weights."""
        if self._is_loaded:
            return
        
        path = checkpoint_path or self.checkpoint_path
        if not path:
            raise ValueError("No checkpoint path provided")
        
        print(f"[DiffusionModelPyTorch] Loading from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self._is_loaded = True
        print(f"[DiffusionModelPyTorch] Loaded {self.model_type}")
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        super().eval()
        self.model.eval()
        return self
    
    @staticmethod
    def add_noise(z, noise_level):
        """
        Add noise using flow matching interpolation.
        z_t = (1 - t) * z + t * noise
        """
        noise = torch.randn_like(z)
        z_t = (1 - noise_level) * z + noise_level * noise
        return z_t, noise
    
    @torch.no_grad()
    def denoise(self, z_noisy, noise_level, labels=None, num_steps=1, use_cfg=True):
        """
        Denoise from z_noisy at given noise level.
        
        Args:
            z_noisy: Noisy latent, (B, C, H, W) in NCHW format (PyTorch convention)
            noise_level: Starting noise level
            labels: Class labels
            num_steps: Number of sampling steps
            use_cfg: Whether to use CFG
            
        Returns:
            z_denoised: Denoised latent in NCHW format
        """
        if not self._is_loaded:
            self.load()
        
        device = z_noisy.device
        B = z_noisy.shape[0]
        
        # Convert NCHW -> NHWC for model (MiT_PyTorch uses NHWC)
        z = z_noisy.permute(0, 2, 3, 1)
        
        # Random labels if not provided
        if labels is None:
            labels = torch.randint(0, self.num_classes, (B,), device=device)
        
        # Time steps
        t_steps = torch.linspace(noise_level, 0.0, num_steps + 1, device=device)
        omega = self.omega if use_cfg else 1.0
        t_min = self.t_min if use_cfg else 0.0
        t_max = self.t_max if use_cfg else 1.0
        
        # Sampling loop
        for i in range(num_steps):
            z = self.model.sample_one_step(z, labels, i, t_steps, omega, t_min, t_max)
        
        # Convert back NHWC -> NCHW
        return z.permute(0, 3, 1, 2)
    
    @torch.no_grad()
    def generate(self, batch_size, labels=None, num_steps=1, use_cfg=True, device='cuda', generator=None, seeds=None):
        """
        Generate from pure noise.
        
        Args:
            batch_size: Number of samples
            labels: Class labels
            num_steps: Number of sampling steps
            use_cfg: Whether to use CFG
            device: Device
            generator: Optional torch.Generator for deterministic noise
            
        Returns:
            z_gen: Generated latent in NCHW format
        """
        if not self._is_loaded:
            self.load()
        
        # Start from pure noise (NHWC)
        z = torch.randn(batch_size, self.latent_size, self.latent_size, 4, device=device, generator=generator)
        
        # Random labels if not provided
        if labels is None:
            labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        
        # Time steps
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        # CFG parameters
        omega = self.omega if use_cfg else 1.0
        t_min = self.t_min if use_cfg else 0.0
        t_max = self.t_max if use_cfg else 1.0
        
        # Sampling loop
        for i in range(num_steps):
            z = self.model.sample_one_step(z, labels, i, t_steps, omega, t_min, t_max)
        
        # Convert NHWC -> NCHW
        return z.permute(0, 3, 1, 2)
    
    @torch.no_grad()
    def denoise_from_encoder_latent(self, z_real, maximum_noise_level, normalized=False,
                                     labels=None, num_steps=1, use_cfg=False,
                                     fixed_noise=False, minimum_noise_level=0.0):
        """
        Denoise encoder output latent.
        
        Args:
            z_real: Encoder output, (B, C, H, W) in NCHW format
            maximum_noise_level: Noise level. When fixed_noise=True, used as-is;
                                 otherwise samples t ~ U[min, max].
            normalized: Whether to normalize z_noisy
            labels: Class labels
            num_steps: Number of steps
            use_cfg: Whether to use CFG (False for reconstruction, True for generation)
            fixed_noise: If True, use maximum_noise_level directly instead of sampling
            minimum_noise_level: Lower bound for random noise sampling
            
        Returns:
            z_denoised: Denoised latent in NCHW format
        """
        if not self._is_loaded:
            self.load()
        
        device = z_real.device
        B = z_real.shape[0]
        
        # Fixed or random noise level
        if fixed_noise:
            noise_level = maximum_noise_level
        else:
            noise_span = maximum_noise_level - minimum_noise_level
            noise_level = minimum_noise_level + torch.rand(1, device=device).item() * noise_span
        
        # Add noise
        z_noisy, _ = self.add_noise(z_real, noise_level)
        
        # Optional normalization
        if normalized:
            mean = z_noisy.mean(dim=(1, 2, 3), keepdim=True)
            std = z_noisy.std(dim=(1, 2, 3), keepdim=True)
            z_noisy_norm = (z_noisy - mean) / (std + 1e-8)
        else:
            z_noisy_norm = z_noisy
            mean, std = None, None
        
        # Denoise
        z_denoised = self.denoise(z_noisy_norm, noise_level, labels, num_steps, use_cfg=use_cfg)
        
        # Un-normalize
        if normalized and mean is not None:
            z_denoised = z_denoised * std + mean
        
        return z_denoised
