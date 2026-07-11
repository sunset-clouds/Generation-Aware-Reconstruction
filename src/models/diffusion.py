"""
Diffusion Model wrapper for iMF (improved MeanFlow).

This module provides a unified interface to the frozen iMF model.
Following the same structure as VQ-Transplant's models.

The diffusion model is used to:
1. Generate samples from pure noise (z_t=1 -> z_t=0)
2. Denoise partially noisy latents (z_t -> z_0)
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax.training import checkpoints

# Use local iMF implementation
from models.imf import iMeanFlow, generate


# Model configurations (CFG parameters for each variant)
MODEL_CONFIGS = {
    'iMF-B-2': {
        'model_str': 'MiT_B_2',
        'omega': 8.0,
        't_min': 0.4,
        't_max': 0.65,
    },
    'iMF-M-2': {
        'model_str': 'MiT_M_2',
        'omega': 10.5,
        't_min': 0.4,
        't_max': 0.6,
    },
    'iMF-L-2': {
        'model_str': 'MiT_L_2',
        'omega': 10.5,
        't_min': 0.4,
        't_max': 0.6,
    },
    'iMF-XL-2': {
        'model_str': 'MiT_XL_2',
        'omega': 8.0,
        't_min': 0.42,
        't_max': 0.62,
    },
}


class DiffusionModel:
    """
    Frozen iMF diffusion model for generation and denoising.
    
    This is always frozen during decoder post-training.
    Uses JAX/Flax backend.
    """
    
    def __init__(self, args):
        """
        Initialize diffusion model from args.
        
        Args:
            args: Configuration with model_type, pretrained_imf, etc.
        """
        self.args = args
        self.model_type = args.model_type
        self.checkpoint_path = args.pretrained_imf
        self.latent_size = getattr(args, 'latent_size', 32)
        self.num_classes = 1000  # ImageNet
        
        # Get model configuration
        if self.model_type in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[self.model_type]
            self.model_str = cfg['model_str']
            self.omega = cfg['omega']
            self.t_min = cfg['t_min']
            self.t_max = cfg['t_max']
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Model state (lazy loading)
        self._model = None
        self._params = None
        self._is_loaded = False
    
    def load(self):
        """Load and compile the model."""
        if self._is_loaded:
            return
        
        print(f"[DiffusionModel] Initializing {self.model_str}...")
        
        # Create model
        self._model = iMeanFlow(
            model_str=self.model_str,
            num_classes=self.num_classes,
            eval=True,
        )
        
        # Initialize with dummy input
        rng = jax.random.PRNGKey(0)
        dummy_z = jnp.ones((1, self.latent_size, self.latent_size, 4))
        dummy_t = jnp.ones((1,))
        dummy_y = jnp.zeros((1,), dtype=jnp.int32)
        
        rng, init_rng = jax.random.split(rng)
        self._params = self._model.init(init_rng, dummy_z, dummy_t, dummy_y)
        
        # Load checkpoint
        print(f"[DiffusionModel] Loading checkpoint from {self.checkpoint_path}...")
        
        from flax.training import train_state
        import optax
        
        class TrainState(train_state.TrainState):
            ema_params: dict = None
        
        # Must match the optimizer used during training (adamw with b2=0.95),
        # otherwise restore_checkpoint silently fails due to state tree mismatch.
        dummy_tx = optax.adamw(learning_rate=1e-4, weight_decay=0, b2=0.95)
        state = TrainState.create(
            apply_fn=self._model.apply,
            params=self._params['params'],
            tx=dummy_tx,
            ema_params=self._params['params'],
        )
        
        # Restore checkpoint
        state = checkpoints.restore_checkpoint(self.checkpoint_path, state)
        
        if state.step == 0:
            raise RuntimeError(
                f"[DiffusionModel] Checkpoint did NOT load (step=0). "
                f"Check path: {self.checkpoint_path}"
            )
        print(f"[DiffusionModel] Loaded from step {state.step}")
        
        # Use EMA params if available
        if hasattr(state, 'ema_params') and state.ema_params is not None:
            self._params = {'params': state.ema_params}
            print("[DiffusionModel] Using EMA parameters")
        else:
            self._params = {'params': state.params}
            print("[DiffusionModel] Using regular parameters")
        self._is_loaded = True
    
    @property
    def model(self):
        """Get the underlying model (for external JIT compilation)."""
        if not self._is_loaded:
            self.load()
        return self._model
    
    @property
    def params(self):
        """Get model parameters."""
        if not self._is_loaded:
            self.load()
        return self._params
    
    def create_denoise_fn(self, noise_level, num_steps=1):
        """
        Create a JIT-compiled denoise function for a specific noise level.
        
        Args:
            noise_level: Starting noise level (0 to 1)
            num_steps: Number of sampling steps
            
        Returns:
            JIT-compiled function: (z_noisy, labels) -> z_denoised
        """
        if not self._is_loaded:
            self.load()
        
        model = self._model
        params = self._params
        t_steps = jnp.linspace(noise_level, 0.0, num_steps + 1)
        omega = self.omega
        t_min = self.t_min
        t_max = self.t_max
        
        @jax.jit
        def denoise_fn(z_noisy, labels):
            def step_fn(i, z_i):
                return model.apply(
                    params, z_i, labels, i, t_steps,
                    jnp.array(omega), jnp.array(t_min), jnp.array(t_max),
                    method=model.sample_one_step
                )
            return jax.lax.fori_loop(0, num_steps, step_fn, z_noisy)
        
        return denoise_fn
    
    def create_generate_fn(self, num_steps=1):
        """
        Create a JIT-compiled generation function (from pure noise).
        
        Args:
            num_steps: Number of sampling steps
            
        Returns:
            JIT-compiled function: (z_init, labels) -> z_gen
        """
        return self.create_denoise_fn(noise_level=1.0, num_steps=num_steps)
    
    @staticmethod
    def add_noise(z, noise_level, rng): 
        """
        Add noise to latent using flow matching interpolation.
        
        z_t = (1 - t) * z + t * noise
        
        Args:
            z: Clean latent, shape (B, H, W, 4)
            noise_level: Noise level t in [0, 1]
            rng: JAX random key
            
        Returns:
            Noisy latent z_t, noise
        """
        noise = jax.random.normal(rng, z.shape)
        z_t = (1 - noise_level) * z + noise_level * noise
        return z_t, noise
    
    # ================================================================
    # Generation Functions (from pure noise)
    # ================================================================
    
    def generate_without_cfg(self, batch_size, labels=None, rng=None, num_steps=1):
        """
        Generate latents from pure noise WITHOUT CFG.
        
        Pipeline:
            white noise → iMF (no CFG) → z_denoised
        
        Args:
            batch_size: Number of samples to generate
            labels: Class labels, shape (B,). If None, random labels are used.
            rng: JAX random key
            num_steps: Number of sampling steps
            
        Returns:
            z_denoised: Generated latents, shape (B, H, W, 4) in NHWC format
        """
        if not self._is_loaded:
            self.load()
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        rng, rng_noise, rng_label = jax.random.split(rng, 3)
        
        # Start from pure noise
        z_init = jax.random.normal(rng_noise, (batch_size, self.latent_size, self.latent_size, 4))
        
        # Random labels if not provided
        if labels is None:
            labels = jax.random.randint(rng_label, (batch_size,), 0, self.num_classes)
        
        # Create denoise function (omega=1 means no CFG)
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
        model = self._model
        params = self._params
        
        @jax.jit
        def denoise_fn(z_noisy, labels):
            def step_fn(i, z_i):
                return model.apply(
                    params, z_i, labels, i, t_steps,
                    jnp.array(1.0),  # omega=1: no CFG
                    jnp.array(0.0),  # t_min (not used when omega=1)
                    jnp.array(1.0),  # t_max (not used when omega=1)
                    method=model.sample_one_step
                )
            return jax.lax.fori_loop(0, num_steps, step_fn, z_noisy)
        
        z_denoised = denoise_fn(z_init, labels)
        return z_denoised
    
    def generate_with_cfg(self, batch_size, labels=None, rng=None, num_steps=1):
        """
        Generate latents from pure noise WITH CFG.
        
        Pipeline:
            white noise → iMF (with CFG) → z_denoised
        
        Args:
            batch_size: Number of samples to generate
            labels: Class labels, shape (B,). If None, random labels are used.
            rng: JAX random key
            num_steps: Number of sampling steps
            
        Returns:
            z_denoised: Generated latents, shape (B, H, W, 4) in NHWC format
        """
        if not self._is_loaded:
            self.load()
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        rng, rng_noise, rng_label = jax.random.split(rng, 3)
        
        # Start from pure noise
        z_init = jax.random.normal(rng_noise, (batch_size, self.latent_size, self.latent_size, 4))
        
        # Random labels if not provided
        if labels is None:
            labels = jax.random.randint(rng_label, (batch_size,), 0, self.num_classes)
        
        # Create denoise function with CFG
        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)
        model = self._model
        params = self._params
        omega = self.omega
        t_min = self.t_min
        t_max = self.t_max
        
        @jax.jit
        def denoise_fn(z_noisy, labels):
            def step_fn(i, z_i):
                return model.apply(
                    params, z_i, labels, i, t_steps,
                    jnp.array(omega),
                    jnp.array(t_min),
                    jnp.array(t_max),
                    method=model.sample_one_step
                )
            return jax.lax.fori_loop(0, num_steps, step_fn, z_noisy)
        
        z_denoised = denoise_fn(z_init, labels)
        return z_denoised
    
    # ================================================================
    # Reconstruction Functions (from encoder latent)
    # ================================================================
    
    def denoise_from_encoder_latent(self, z_real, maximum_noise_level, normalized=False, 
                                     labels=None, rng=None, num_steps=1):
        """
        Denoise encoder output latent for our proposed training pipeline.
        
        Pipeline:
            z_real → add_noise(t) → z_noisy → iMF → z_denoised
        
        The noise level is sampled uniformly from [0, maximum_noise_level].
        
        Args:
            z_real: Encoder output latent, shape (B, H, W, 4) in NHWC format
            maximum_noise_level: Maximum noise level (noise_level ~ U[0, max])
            normalized: Whether to normalize z_noisy before denoising
            labels: Class labels, shape (B,). If None, random labels are used.
            rng: JAX random key
            num_steps: Number of sampling steps
            
        Returns:
            z_denoised: Denoised latents, shape (B, H, W, 4) in NHWC format
        """
        if not self._is_loaded:
            self.load()
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        batch_size = z_real.shape[0]
        rng, rng_noise, rng_t, rng_label = jax.random.split(rng, 4)
        
        # Sample noise level uniformly from [0, maximum_noise_level]
        noise_level = jax.random.uniform(rng_t, (1,), minval=0.0, maxval=maximum_noise_level)[0]
        
        # Add noise
        z_noisy, noise = self.add_noise(z_real, noise_level, rng_noise)
        
        # Optional normalization
        if normalized:
            # Normalize to standard Gaussian
            z_mean = jnp.mean(z_noisy, axis=(1, 2, 3), keepdims=True)
            z_std = jnp.std(z_noisy, axis=(1, 2, 3), keepdims=True)
            z_noisy_norm = (z_noisy - z_mean) / (z_std + 1e-8)
        else:
            z_noisy_norm = z_noisy
            z_mean = None
            z_std = None
        
        # Random labels if not provided
        if labels is None:
            labels = jax.random.randint(rng_label, (batch_size,), 0, self.num_classes)
        
        # Denoise
        denoise_fn = self.create_denoise_fn(noise_level, num_steps)
        z_denoised = denoise_fn(z_noisy_norm, labels)
        
        # Un-normalize if needed
        if normalized and z_mean is not None:
            z_denoised = z_denoised * z_std + z_mean
        
        return z_denoised
    
    def denoise_from_encoder_latent_fixed_noise(self, z_real, noise_level, 
                                                 labels=None, rng=None, num_steps=1):
        """
        Denoise encoder output latent with a FIXED noise level.
        
        Used for evaluation (rFID at specific noise levels).
        
        Args:
            z_real: Encoder output latent, shape (B, H, W, 4) in NHWC format
            noise_level: Fixed noise level
            labels: Class labels, shape (B,). If None, random labels are used.
            rng: JAX random key
            num_steps: Number of sampling steps
            
        Returns:
            z_denoised: Denoised latents, shape (B, H, W, 4) in NHWC format
        """
        if not self._is_loaded:
            self.load()
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        batch_size = z_real.shape[0]
        rng, rng_noise, rng_label = jax.random.split(rng, 3)
        
        # Add noise at fixed level
        z_noisy, noise = self.add_noise(z_real, noise_level, rng_noise)
        
        # Random labels if not provided
        if labels is None:
            labels = jax.random.randint(rng_label, (batch_size,), 0, self.num_classes)
        
        # Denoise
        denoise_fn = self.create_denoise_fn(noise_level, num_steps)
        z_denoised = denoise_fn(z_noisy, labels)
        
        return z_denoised
    


