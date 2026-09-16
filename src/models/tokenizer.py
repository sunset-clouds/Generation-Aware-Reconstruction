"""
Tokenizer (VAE) wrapper for encoding/decoding images.

This module provides:
1. VAE encoder (frozen) - for encoding real images to latents
2. VAE decoder (trainable) - for decoding latents to images

Following the same structure as VQ-Transplant's tokenizer design.
Uses Stable Diffusion VAE from HuggingFace.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from utils.hf_paths import resolve_sd_vae_source


# VAE normalization constants (same as iMF, SD VAE)
# Shape: (1, 4, 1, 1) for NCHW format
VAE_MEAN = torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409]).view(1, 4, 1, 1)
VAE_STD = torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003]).view(1, 4, 1, 1)


class Tokenizer(nn.Module):
    """
    VAE-based tokenizer for images.
    
    Encoder is frozen, Decoder is trainable for post-training.
    Uses Stable Diffusion VAE (sd-vae-ft-mse or sd-vae-ft-ema).
    """
    
    def __init__(self, args):
        """
        Initialize tokenizer from args.
        
        Args:
            args: Configuration with vae_type, stage, etc.
        """
        super().__init__()
        self.args = args
        self.vae_type = getattr(args, 'vae_type', 'mse')
        self.stage = getattr(args, 'stage', 'train_decoder')
        
        print(f"[Tokenizer] Loading VAE (type={self.vae_type})...")
        vae_source, load_kwargs, load_message = resolve_sd_vae_source(self.vae_type)
        print(load_message)
        self.vae = AutoencoderKL.from_pretrained(vae_source, **load_kwargs)
        
        # Register normalization buffers
        self.register_buffer('mean', VAE_MEAN)
        self.register_buffer('std', VAE_STD)
        
        # Setup trainable/frozen parts based on stage
        self._setup_training_mode()
        
        print(f"[Tokenizer] Loaded VAE, stage={self.stage}")
    
    def _setup_training_mode(self):
        """Configure which parts are trainable based on stage."""
        if self.stage == 'train_decoder':
            # Freeze encoder
            for param in self.vae.encoder.parameters():
                param.requires_grad = False
            self.vae.encoder.eval()
            
            # Keep decoder trainable
            for param in self.vae.decoder.parameters():
                param.requires_grad = True
            self.vae.decoder.train()
            
            print("[Tokenizer] Encoder frozen, Decoder trainable")
        else:
            # Evaluation mode - everything frozen
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            
            print("[Tokenizer] All frozen (eval mode)")
    
    @torch.no_grad()
    def encode(self, x):
        """
        Encode image to normalized latent.
        
        Args:
            x: Image tensor, shape (B, 3, H, W) in range [-1, 1]
            
        Returns:
            Normalized latent, shape (B, 4, H/8, W/8)
        """
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.mean  # Use mean, not sample
        return (z - self.mean) / self.std
    
    def decode(self, z):
        """
        Decode normalized latent to image.
        
        Args:
            z: Normalized latent, shape (B, 4, H/8, W/8)
            
        Returns:
            Decoded image, shape (B, 3, H, W) in range [-1, 1]
        """
        z_denorm = z * self.std + self.mean
        return self.vae.decode(z_denorm).sample
    
    def get_decoder_params(self):
        """Get trainable decoder parameters."""
        decoder = self.vae.decoder
        # Handle DDP wrapper
        if hasattr(decoder, 'module'):
            decoder = decoder.module
        return list(decoder.parameters())
    
    def train_decoder(self):
        """Set decoder to training mode."""
        self.vae.decoder.train()
        
    def eval_decoder(self):
        """Set decoder to evaluation mode."""
        self.vae.decoder.eval()
    
    def forward(self, x):
        """
        Full encode-decode forward pass.
        
        Args:
            x: Image tensor, shape (B, 3, H, W) in range [-1, 1]
            
        Returns:
            Reconstructed image, shape (B, 3, H, W)
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
    
    @property
    def last_layer(self):
        """Get last decoder layer for adaptive weight calculation."""
        decoder = self.vae.decoder
        if hasattr(decoder, 'module'):
            decoder = decoder.module
        return decoder.conv_out.weight
