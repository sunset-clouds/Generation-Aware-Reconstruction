"""
Combined model for decoder post-training.

This module combines:
1. DiffusionModel (frozen iMF) - for denoising latents
2. Tokenizer (VAE with trainable decoder) - for encoding/decoding

Following the same structure as VQ-Transplant's var_model.py

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.tokenizer import Tokenizer


# ============================================================
# JAX <-> PyTorch Conversion Utilities (only used for JAX backend)
# ============================================================

def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    import jax.numpy as jnp
    return jnp.array(tensor.detach().cpu().numpy())


def jax_to_torch(jax_array, device):
    """Convert JAX array to PyTorch tensor."""
    return torch.from_numpy(np.array(jax_array)).to(device)


class TokenizerFlowComposition(nn.Module):
    """
    Main model for decoder post-training.
    
    Combines:
    - Tokenizer (VAE): encoder (frozen) + decoder (trainable)
    - DiffusionModel (iMF): frozen, used for denoising
    
    Supports both JAX and PyTorch backends for iMF.
    """
    
    def __init__(self, args):
        """
        Initialize combined model.
        
        Args:
            args: Configuration with model_type, pretrained_imf, etc.
        """
        super().__init__()
        self.args = args
        
        # Check if we should skip diffusion model loading (for debug mode)
        skip_diffusion = getattr(args, '_skip_diffusion_load', False) or getattr(args, 'debug_mode', False)
        
        # Check if we should use PyTorch version
        use_pytorch = getattr(args, 'use_pytorch_imf', False)
        self.use_pytorch_imf = use_pytorch
        
        # Initialize components
        print("[TokenizerFlowComposition] Initializing components...")
        
        # Diffusion model - skip in debug mode
        if skip_diffusion:
            print("[TokenizerFlowComposition] Skipping diffusion model (debug mode)")
            self.diffusion = None
            self.diffusion_pytorch = None
            self.vae_wrapper = None
        elif use_pytorch:
            ckpt_path = getattr(args, 'pretrained_imf_pytorch', '')
            use_official = ckpt_path.endswith('.pth')
            if use_official:
                from models.imf_official import OfficialImfWrapper
                from models.vae_wrapper import VAEWrapper
                imf_torch_path = getattr(args, 'imf_torch_path', None)
                print("[TokenizerFlowComposition] Using official iMeanFlow (.pth)")
                self.diffusion = None
                self.diffusion_pytorch = OfficialImfWrapper(
                    args.model_type, ckpt_path, imf_torch_path
                )
                vae_type = getattr(args, 'vae_type', 'mse')
                self.vae_wrapper = VAEWrapper(decode_batch_size=64, vae_type=vae_type)
            else:
                from models.diffusion_pytorch import DiffusionModelPyTorch
                self.vae_wrapper = None
                print("[TokenizerFlowComposition] Using JAX-converted PyTorch iMF (.pt)")
                self.diffusion = None
                self.diffusion_pytorch = DiffusionModelPyTorch(args)
                self.diffusion_pytorch.load()
            self.diffusion_pytorch.eval()
        else:
            # Use JAX version (original)
            from models.diffusion import DiffusionModel
            print("[TokenizerFlowComposition] Using JAX iMF")
            self.diffusion = DiffusionModel(args)
            self.diffusion.load()
            self.diffusion_pytorch = None
            self.vae_wrapper = None
        
        # Tokenizer (PyTorch, decoder trainable)
        self.tokenizer = Tokenizer(args)
        
        print("[TokenizerFlowComposition] Initialized")
    
    def parameters(self):
        """Return all parameters (for counting)."""
        return self.tokenizer.parameters()
    
    def train_decoder(self):
        """Set decoder to training mode, encoder frozen."""
        self.tokenizer.train_decoder()
    
    def eval_mode(self):
        """Set entire model to evaluation mode."""
        self.tokenizer.eval()
        if self.diffusion_pytorch is not None:
            self.diffusion_pytorch.eval()
        if getattr(self, 'vae_wrapper', None) is not None:
            self.vae_wrapper.eval()
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.tokenizer = self.tokenizer.to(device)
        if self.diffusion_pytorch is not None:
            self.diffusion_pytorch = self.diffusion_pytorch.to(device)
        if getattr(self, 'vae_wrapper', None) is not None:
            self.vae_wrapper = self.vae_wrapper.to(device)
        return self

    def decode_latent(self, z):
        """Decode latent to image. Uses VAEWrapper when .pth for FID consistency."""
        if getattr(self, 'vae_wrapper', None) is not None:
            return self.vae_wrapper.decode(z).clamp(-1, 1)
        return self.tokenizer.decode(z).clamp(-1, 1)
    
    # ============================================================
    # Pipeline 1: Vanilla VAE Reconstruction (for rFID baseline)
    # x → Encoder → z_real → Decoder → x_rec
    # ============================================================
    def vae_reconstruction(self, x):
        """
        Vanilla VAE reconstruction (no iMF).
        Used for computing baseline rFID.
        """
        with torch.no_grad():
            z_real = self.tokenizer.encode(x)
        x_rec = self.decode_latent(z_real)
        return x_rec
    
    # ============================================================
    # Pipeline 2: iMF Generation without CFG (for gFID baseline)
    # noise → iMF → z_gen → Decoder → x_gen
    # ============================================================
    def imf_image_generation_without_cfg(self, batch_size, labels=None, rng=None):
        """
        Generate images using iMF without CFG.
        Used for computing gFID without CFG.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        z_gen = self.diffusion.generate_without_cfg(batch_size, labels, rng)
        
        # Convert to PyTorch (NHWC -> NCHW)
        z_gen_torch = jax_to_torch(z_gen, next(self.tokenizer.parameters()).device)
        z_gen_torch = z_gen_torch.permute(0, 3, 1, 2).float()
        
        x_gen = self.tokenizer.decode(z_gen_torch)
        return x_gen.clamp(-1, 1)
    
    # ============================================================
    # Pipeline 3: iMF Generation with CFG (for gFID with CFG)
    # noise → iMF (CFG) → z_gen → Decoder → x_gen
    # ============================================================
    def imf_image_generation_with_cfg(self, batch_size, labels=None, rng=None):
        """
        Generate images using iMF with CFG.
        Used for computing gFID with CFG.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        z_gen = self.diffusion.generate_with_cfg(batch_size, labels, rng)
        
        # Convert to PyTorch (NHWC -> NCHW)
        z_gen_torch = jax_to_torch(z_gen, next(self.tokenizer.parameters()).device)
        z_gen_torch = z_gen_torch.permute(0, 3, 1, 2).float()
        
        x_gen = self.tokenizer.decode(z_gen_torch)
        return x_gen.clamp(-1, 1)
    
    # ============================================================
    # Pipeline 4: Our Proposed Reconstruction (for training)
    # x → Encoder → z_real → add_noise(t) → z_noisy → iMF → z_denoised → Decoder → x_rec
    # ============================================================
    def vae_imf_reconstruction(self, x, rng=None, labels=None, use_cfg=False):
        """
        Our proposed reconstruction pipeline for training.
        
        Args:
            x: Input images, shape (B, 3, H, W) in range [-1, 1]
            rng: JAX random key for noise (only used for JAX backend)
            labels: Class labels for conditional generation
            use_cfg: Whether to use CFG for denoising (False for reconstruction)
            
        Returns:
            x_rec: Reconstructed images
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode (frozen)
        with torch.no_grad():
            z_real = self.tokenizer.encode(x)  # (B, 4, H/8, W/8) NCHW
        
        # Use PyTorch or JAX backend
        if self.use_pytorch_imf and self.diffusion_pytorch is not None:
            # PyTorch backend - much faster, no memory conflicts
            with torch.no_grad():
                z_denoised = self.diffusion_pytorch.denoise_from_encoder_latent(
                    z_real,
                    maximum_noise_level=self.args.maximum_noise_level,
                    normalized=getattr(self.args, 'normalized', False),
                    labels=labels,
                    num_steps=1,
                    use_cfg=use_cfg,
                    fixed_noise=getattr(self.args, 'fixed_noise', False),
                    minimum_noise_level=getattr(self.args, 'minimum_noise_level', 0.0)
                )
        elif self.diffusion is not None:
            # JAX backend
            import jax
            if rng is None:
                rng = jax.random.PRNGKey(0)
            
            # Convert to JAX (NCHW -> NHWC)
            z_real_jax = torch_to_jax(z_real.permute(0, 2, 3, 1))
            
            # Denoise with iMF
            z_denoised_jax = self.diffusion.denoise_from_encoder_latent(
                z_real_jax,
                maximum_noise_level=self.args.maximum_noise_level,
                normalized=getattr(self.args, 'normalized', False),
                labels=labels,
                rng=rng
            )
            
            # Convert back to PyTorch (NHWC -> NCHW)
            z_denoised = jax_to_torch(z_denoised_jax, device).permute(0, 3, 1, 2).float()
        else:
            # No diffusion (debug mode) - directly use z_real
            z_denoised = z_real
        
        # Decode (trainable)
        x_rec = self.tokenizer.decode(z_denoised)
        return x_rec.clamp(-1, 1)
    
    def forward(self, x, rng=None, labels=None):
        """
        Forward pass for training.
        
        Args:
            x: Input images, shape (B, 3, H, W) in range [-1, 1]
            rng: JAX random key (only used for JAX backend)
            labels: Class labels (must be provided from dataloader)
            
        Returns:
            x_rec: Reconstructed images from our pipeline
        """
        batch_size = x.size(0)
        device = x.device
        
        assert labels is not None, (
            "labels must be provided from the dataloader, "
            "do not use random labels for conditional denoising"
        )
        
        # For JAX backend, still need rng
        rng_rec = None
        if not self.use_pytorch_imf and self.diffusion is not None:
            import jax
            if rng is None:
                rng = jax.random.PRNGKey(0)
            rng, rng_rec = jax.random.split(rng)
            # Convert labels to JAX
            labels_jax = jax.numpy.array(labels.cpu().numpy())
        else:
            labels_jax = labels
        
        # Reconstruction
        x_rec = self.vae_imf_reconstruction(x, rng=rng_rec, labels=labels_jax if not self.use_pytorch_imf else labels)
        
        return x_rec
    
    def collect_eval_info(self, x, rng=None, labels=None):
        """
        Collect evaluation metrics.
        
        Returns:
            x_rec: Reconstructed images
            rec_loss: Reconstruction MSE loss
        """
        x_rec = self.vae_imf_reconstruction(x, rng=rng, labels=labels)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
    
    @property
    def last_layer(self):
        """Get last decoder layer for adaptive weight calculation."""
        return self.tokenizer.last_layer
