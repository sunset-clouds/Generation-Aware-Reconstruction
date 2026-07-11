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
import numpy as np
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
    
    def create_jax_encode_decode(self):
        """
        Create JAX-based encode/decode functions for evaluation.
        
        Uses FlaxAutoencoderKL for encoding (since encoder is always frozen).
        This is useful when we want to use JAX for encoding but PyTorch decoder
        for post-trained decoder evaluation.
        
        Returns:
            (vae, encode_fn, decode_fn): JAX VAE, encode function, decode function
        """
        import jax
        import jax.numpy as jnp
        from diffusers.models import FlaxAutoencoderKL
        
        # VAE normalization constants (BHWC format for JAX)
        vae_mean = jnp.array([0.86488, -0.27787343, 0.21616915, 0.3738409]).reshape(1, 1, 1, -1)
        vae_std = jnp.array([4.85503674, 5.31922414, 3.93725398, 3.9870003]).reshape(1, 1, 1, -1)
        
        # Load JAX VAE
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            f"pcuenq/sd-vae-ft-{self.vae_type}-flax"
        )
        
        @jax.jit
        def encode_fn(image):
            """Encode image to normalized latent. Input: BHWC [-1,1], Output: BHWC normalized latent"""
            image_bchw = jnp.transpose(image, (0, 3, 1, 2))
            latent_dist = vae.apply(
                {"params": vae_params}, image_bchw,
                method=FlaxAutoencoderKL.encode,
            ).latent_dist
            latent = latent_dist.mean
            return (latent - vae_mean) / vae_std
        
        @jax.jit
        def decode_fn(latent_normalized):
            """Decode normalized latent to image. Input: BHWC normalized latent, Output: BHWC [-1,1]"""
            latent = latent_normalized * vae_std + vae_mean
            latent_bchw = jnp.transpose(latent, (0, 3, 1, 2))
            decoded = vae.apply(
                {"params": vae_params}, latent_bchw,
                method=FlaxAutoencoderKL.decode,
            ).sample
            if decoded.shape[1] == 3:
                decoded = jnp.transpose(decoded, (0, 2, 3, 1))
            return decoded
        
        return vae, encode_fn, decode_fn

    @property
    def last_layer(self):
        """Get last decoder layer for adaptive weight calculation."""
        decoder = self.vae.decoder
        if hasattr(decoder, 'module'):
            decoder = decoder.module
        return decoder.conv_out.weight


class JAXTokenizer:
    """
    JAX-based tokenizer for evaluation (compatible with iMF pipeline).
    
    Uses FlaxAutoencoderKL for faster inference on multiple GPUs.
    """
    
    def __init__(self, vae_type='mse', latent_size=32, decode_batch_size=8):
        """
        Initialize JAX tokenizer.
        
        Args:
            vae_type: 'mse' or 'ema'
            latent_size: Latent resolution
            decode_batch_size: Batch size for pmap
        """
        import jax
        import jax.numpy as jnp
        from functools import partial
        from flax import jax_utils
        from diffusers.models import FlaxAutoencoderKL
        
        self.vae_type = vae_type
        self.latent_size = latent_size
        self.decode_batch_size = decode_batch_size
        
        # Normalization constants
        self.mean = jnp.array([0.86488, -0.27787343, 0.21616915, 0.3738409]).reshape(1, -1, 1, 1)
        self.std = jnp.array([4.85503674, 5.31922414, 3.93725398, 3.9870003]).reshape(1, -1, 1, 1)
        
        # Load VAE
        print(f"[JAXTokenizer] Loading VAE (type={vae_type})...")
        self.vae, self.vae_params = FlaxAutoencoderKL.from_pretrained(
            f"pcuenq/sd-vae-ft-{vae_type}-flax"
        )
        
        # Compile decode function
        self._compile_decode_fn()
        print("[JAXTokenizer] Loaded and compiled")
    
    def _compile_decode_fn(self):
        """Compile pmap-based decode function."""
        import jax
        import jax.numpy as jnp
        from functools import partial
        from flax import jax_utils
        from diffusers.models import FlaxAutoencoderKL
        
        def dist_prepare_batch_data(batch):
            local_device_count = jax.local_device_count()
            return_dict = {}
            for k, v in batch.items():
                v = v.reshape((local_device_count, -1) + v.shape[1:])
                return_dict[k] = v
            return return_dict
        
        z_dummy = jnp.ones((
            jax.local_device_count(),
            self.decode_batch_size,
            4,
            self.latent_size,
            self.latent_size,
        ))
        
        p_vae_variable = jax_utils.replicate({"params": self.vae_params})
        p_decode_fn = partial(self.vae.apply, method=FlaxAutoencoderKL.decode)
        p_decode_fn = jax.pmap(p_decode_fn, axis_name="batch")
        
        lowered = p_decode_fn.lower(p_vae_variable, z_dummy)
        compiled_decode_fn = lowered.compile()
        
        def call_compiled_decode(x, p_func, var):
            x = dist_prepare_batch_data(dict(x=x))["x"]
            x = p_func(var, x)
            x = x.sample
            x = x.reshape((-1,) + x.shape[2:])
            return dict(sample=x)
        
        self._decode_fn = partial(
            call_compiled_decode, p_func=compiled_decode_fn, var=p_vae_variable
        )
    
    def decode(self, latents_nchw):
        """
        Decode latents to images.
        
        Args:
            latents_nchw: Normalized latent, shape (B, 4, H, W) in NCHW format
            
        Returns:
            Decoded images in range [-1, 1], shape (B, H, W, 3)
        """
        latents = latents_nchw * self.std + self.mean
        return self._decode_fn(latents)["sample"]
    
    def decode_nhwc(self, latents_nhwc):
        """
        Decode latents in NHWC format.
        
        Args:
            latents_nhwc: Normalized latent, shape (B, H, W, 4) in NHWC format
            
        Returns:
            Decoded images in range [-1, 1], shape (B, H, W, 3)
        """
        latents_nchw = latents_nhwc.transpose((0, 3, 1, 2))
        return self.decode(latents_nchw)


    
