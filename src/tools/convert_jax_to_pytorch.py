"""
Convert iMF JAX/Flax checkpoint to PyTorch.

Usage:
    python tools/convert_jax_to_pytorch.py \
        --jax_checkpoint /path/to/checkpoint_0 \
        --model_type iMF-B-2 \
        --output_path checkpoints/pytorch/iMF-B-2.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax


# ============================================================
# PyTorch Model Components
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size
    
    def timestep_embedding(self, t, max_period=10000):
        half = self.freq_embed_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, t):
        embed = self.timestep_embedding(t)
        return self.mlp(embed)


def apply_rotary_pos_emb(x, freqs_cos, freqs_sin):
    """
    Apply RoPE using complex multiplication (matching JAX implementation).
    x: (B, L, num_heads, head_dim)
    freqs_cos, freqs_sin: (L, head_dim // 2)
    """
    # Reshape to complex view: (B, L, num_heads, head_dim // 2, 2)
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # Create complex numbers
    x_complex = torch.view_as_complex(x_r)  # (B, L, num_heads, head_dim // 2)
    
    # Get frequencies for this sequence length
    cos = freqs_cos[:x.shape[1]]  # (L, head_dim // 2)
    sin = freqs_sin[:x.shape[1]]  # (L, head_dim // 2)
    freqs_complex = torch.complex(cos, sin)  # (L, head_dim // 2)
    
    # Broadcast: (1, L, 1, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # Complex multiplication
    x_rotated = x_complex * freqs_complex
    
    # Convert back to real
    x_out = torch.view_as_real(x_rotated)  # (B, L, num_heads, head_dim // 2, 2)
    x_out = x_out.reshape(x.shape)
    
    return x_out.to(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    
    def forward(self, x, freqs_cos, freqs_sin):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = apply_rotary_pos_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_pos_emb(k, freqs_cos, freqs_sin)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, self.hidden_size)
        
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio * 2 / 3)  # SwiGLU uses 2/3
        self.mlp = SwiGLUMlp(hidden_size, mlp_hidden)
        
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_features)
    
    def forward(self, x):
        return self.linear(self.norm(x))


class MiT_PyTorch(nn.Module):
    """PyTorch implementation of iMF MiT model."""
    
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        num_shared_blocks=4,
        num_head_blocks=8,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_shared_blocks = num_shared_blocks
        self.num_head_blocks = num_head_blocks
        
        self.num_patches = (input_size // patch_size) ** 2
        self.head_dim = hidden_size // num_heads
        
        # Learnable tokens
        self.time_tokens = nn.Parameter(torch.zeros(4, hidden_size))
        self.class_tokens = nn.Parameter(torch.zeros(8, hidden_size))
        self.omega_tokens = nn.Parameter(torch.zeros(4, hidden_size))
        self.t_min_tokens = nn.Parameter(torch.zeros(2, hidden_size))
        self.t_max_tokens = nn.Parameter(torch.zeros(2, hidden_size))
        
        # Patch embedder
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Embedders
        self.h_embedder = TimestepEmbedder(hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size)
        self.cfg_t_start_embedder = TimestepEmbedder(hidden_size)
        self.cfg_t_end_embedder = TimestepEmbedder(hidden_size)
        
        # Class embedding
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        
        # Shared blocks
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_shared_blocks)
        ])
        
        # U-head blocks (for velocity prediction)
        self.u_heads = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_head_blocks)
        ])
        
        # Final layers
        out_dim = patch_size * patch_size * in_channels
        self.u_final_layer = FinalLayer(hidden_size, out_dim)
        self.v_final_layer = FinalLayer(hidden_size, out_dim)
        
        # RoPE frequencies
        max_seq_len = self.num_patches + 20  # extra for tokens
        self.register_buffer('freqs_cos', torch.zeros(max_seq_len, self.head_dim // 2))
        self.register_buffer('freqs_sin', torch.zeros(max_seq_len, self.head_dim // 2))
        self._init_rope()
    
    def _init_rope(self):
        dim = self.head_dim
        max_seq_len = self.freqs_cos.shape[0]
        theta = 10000.0
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        self.freqs_cos.copy_(torch.cos(freqs))
        self.freqs_sin.copy_(torch.sin(freqs))
    
    def unpatchify(self, x, h, w):
        """Convert patches back to image."""
        p = self.patch_size
        c = self.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, h * p, w * p, c)
        return x
    
    def forward(self, z_t, t, labels, omega=1.0, t_min=0.0, t_max=1.0):
        """
        Forward pass.
        
        Args:
            z_t: Noisy latent, (B, H, W, C) NHWC format
            t: Timestep, (B,)
            labels: Class labels, (B,)
            omega: CFG scale
            t_min, t_max: CFG interval
        """
        B = z_t.shape[0]
        h = w = self.input_size // self.patch_size
        
        # Patch embed: (B, H, W, C) -> (B, C, H, W) -> conv -> (B, hidden, h, w) -> (B, h*w, hidden)
        x = z_t.permute(0, 3, 1, 2)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Build tokens
        t_tensor = t.float()
        omega_tensor = torch.full((B,), omega, device=z_t.device, dtype=z_t.dtype)
        t_min_tensor = torch.full((B,), t_min, device=z_t.device, dtype=z_t.dtype)
        t_max_tensor = torch.full((B,), t_max, device=z_t.device, dtype=z_t.dtype)
        
        # Time tokens + embedding
        time_tok = self.time_tokens.unsqueeze(0).expand(B, -1, -1)
        time_tok = time_tok + self.h_embedder(t_tensor).unsqueeze(1)
        
        # Class tokens + embedding
        class_tok = self.class_tokens.unsqueeze(0).expand(B, -1, -1)
        class_tok = class_tok + self.y_embedder(labels).unsqueeze(1)
        
        # Omega tokens (JAX uses 1 - 1/omega for embedding)
        omega_tok = self.omega_tokens.unsqueeze(0).expand(B, -1, -1)
        omega_embed_input = 1.0 - 1.0 / omega_tensor  # Match JAX: omega_embedder(1 - 1/w)
        omega_tok = omega_tok + self.omega_embedder(omega_embed_input).unsqueeze(1)
        
        # t_min tokens
        t_min_tok = self.t_min_tokens.unsqueeze(0).expand(B, -1, -1)
        t_min_tok = t_min_tok + self.cfg_t_start_embedder(t_min_tensor).unsqueeze(1)
        
        # t_max tokens
        t_max_tok = self.t_max_tokens.unsqueeze(0).expand(B, -1, -1)
        t_max_tok = t_max_tok + self.cfg_t_end_embedder(t_max_tensor).unsqueeze(1)
        
        # Concatenate: [class, omega, t_min, t_max, time, patches] (matching JAX order)
        x = torch.cat([class_tok, omega_tok, t_min_tok, t_max_tok, time_tok, x], dim=1)
        
        # Shared blocks
        for block in self.shared_blocks:
            x = block(x, self.freqs_cos, self.freqs_sin)
        
        # U-head blocks
        for block in self.u_heads:
            x = block(x, self.freqs_cos, self.freqs_sin)
        
        # Remove condition tokens, keep only patch tokens
        num_cond_tokens = 4 + 8 + 4 + 2 + 2  # time + class + omega + t_min + t_max = 20
        x_patches = x[:, num_cond_tokens:, :]
        
        # Final layer
        x_out = self.u_final_layer(x_patches)
        
        # Unpatchify
        velocity = self.unpatchify(x_out, h, w)
        
        return velocity
    
    def sample_one_step(self, z_t, labels, step_idx, t_steps, omega, t_min, t_max):
        """
        One sampling step (matching JAX implementation).
        
        In iMF, CFG is handled internally through omega/t_min/t_max embeddings.
        The model learns to apply guidance based on these conditioning values.
        
        For CFG (omega > 1): We need to do classifier-free guidance mixing externally
        because the model was trained with conditional and unconditional passes.
        """
        t_cur = t_steps[step_idx]
        t_next = t_steps[step_idx + 1]
        dt = t_next - t_cur  # dt is negative (going from t=1 to t=0)
        
        B = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype
        
        t_cur_val = t_cur.item() if isinstance(t_cur, torch.Tensor) else t_cur
        h = t_cur_val - (t_next.item() if isinstance(t_next, torch.Tensor) else t_next)  # h = t - r
        
        t_batch = torch.full((B,), t_cur_val, device=device, dtype=dtype)
        h_batch = torch.full((B,), h, device=device, dtype=dtype)
        
        # Check if CFG should be applied (omega > 1 and within interval)
        apply_cfg = omega > 1.0 and t_min <= t_cur_val <= t_max
        
        if apply_cfg:
            # Conditional forward pass
            u_cond = self.forward(z_t, h_batch, labels, omega, t_min, t_max)
            
            # Unconditional forward pass (null class = num_classes)
            null_labels = torch.full_like(labels, self.num_classes)
            u_uncond = self.forward(z_t, h_batch, null_labels, omega, t_min, t_max)
            
            # CFG mixing: u = u_uncond + omega * (u_cond - u_uncond)
            u = u_uncond + omega * (u_cond - u_uncond)
        else:
            # No CFG, just conditional pass
            u = self.forward(z_t, h_batch, labels, omega, t_min, t_max)
        
        # Update: z_t = z_t - h * u (equivalent to z_t + dt * u since dt = -h)
        z_t = z_t - h * u
        return z_t


# Model configurations
MODEL_CONFIGS = {
    'iMF-B-2': {
        'hidden_size': 768,
        'num_shared_blocks': 4,    # depth=12, aux_head_depth=8 → shared=4
        'num_head_blocks': 8,
        'num_heads': 12,
        'mlp_ratio': 4.0,          # PyTorch 4.0 * 2/3 == JAX 8/3
    },
    'iMF-M-2': {
        'hidden_size': 768,
        'num_shared_blocks': 16,   # depth=24, aux_head_depth=8 → shared=16
        'num_head_blocks': 8,
        'num_heads': 12,
        'mlp_ratio': 4.0,
    },
    'iMF-L-2': {
        'hidden_size': 1024,
        'num_shared_blocks': 24,   # depth=32, aux_head_depth=8 → shared=24
        'num_head_blocks': 8,
        'num_heads': 16,
        'mlp_ratio': 4.0,
    },
    'iMF-XL-2': {
        'hidden_size': 1024,
        'num_shared_blocks': 40,   # depth=48, aux_head_depth=8 → shared=40
        'num_head_blocks': 8,
        'num_heads': 16,
        'mlp_ratio': 4.0,
    },
}


def load_jax_checkpoint(checkpoint_path, model_type):
    """Load JAX checkpoint."""
    from models.imf import iMeanFlow
    
    model_str = f"MiT_{model_type.split('-')[1]}_{model_type.split('-')[2]}"
    
    jax_model = iMeanFlow(model_str=model_str, num_classes=1000, eval=True)
    
    rng = jax.random.PRNGKey(0)
    dummy_z = jnp.ones((1, 32, 32, 4))
    dummy_t = jnp.ones((1,))
    dummy_y = jnp.zeros((1,), dtype=jnp.int32)
    
    params = jax_model.init(rng, dummy_z, dummy_t, dummy_y)
    
    class TrainState(train_state.TrainState):
        ema_params: dict = None
    
    # Must match the optimizer used during training (adamw with b2=0.95),
    # otherwise restore_checkpoint silently fails due to state tree mismatch.
    dummy_tx = optax.adamw(learning_rate=1e-4, weight_decay=0, b2=0.95)
    state = TrainState.create(
        apply_fn=jax_model.apply,
        params=params['params'],
        tx=dummy_tx,
        ema_params=params['params'],
    )
    
    state = checkpoints.restore_checkpoint(checkpoint_path, state)
    
    if state.step == 0:
        raise RuntimeError(
            f"Checkpoint did NOT load (step=0). Check path: {checkpoint_path}\n"
            "The path should be the checkpoint DIRECTORY (e.g., /path/to/iMF-B-2/), "
            "not the checkpoint file itself."
        )
    print(f"Checkpoint loaded from step {state.step}")
    
    if hasattr(state, 'ema_params') and state.ema_params is not None:
        print("Using EMA parameters")
        return state.ema_params
    else:
        print("Using regular parameters")
        return state.params


def get_nested(d, keys):
    """Get nested dict value."""
    for k in keys:
        d = d[k]
    return d


def convert_params(jax_params, model_type):
    """Convert JAX parameters to PyTorch state dict."""
    cfg = MODEL_CONFIGS[model_type]
    pytorch_state_dict = OrderedDict()
    
    def jax_to_torch(arr):
        return torch.from_numpy(np.array(arr))
    
    net = jax_params['net']
    
    # Learnable tokens
    pytorch_state_dict['time_tokens'] = jax_to_torch(net['time_tokens'])
    pytorch_state_dict['class_tokens'] = jax_to_torch(net['class_tokens'])
    pytorch_state_dict['omega_tokens'] = jax_to_torch(net['omega_tokens'])
    pytorch_state_dict['t_min_tokens'] = jax_to_torch(net['t_min_tokens'])
    pytorch_state_dict['t_max_tokens'] = jax_to_torch(net['t_max_tokens'])
    
    # x_embedder (Conv2d)
    kernel = net['x_embedder']['proj']['kernel']
    pytorch_state_dict['x_embedder.weight'] = jax_to_torch(kernel).permute(3, 2, 0, 1)
    pytorch_state_dict['x_embedder.bias'] = jax_to_torch(net['x_embedder']['proj']['bias'])
    
    # h_embedder (timestep)
    h_emb = net['h_embedder']['mlp']
    pytorch_state_dict['h_embedder.mlp.0.weight'] = jax_to_torch(h_emb['layers_0']['_flax_linear']['kernel']).T
    pytorch_state_dict['h_embedder.mlp.0.bias'] = jax_to_torch(h_emb['layers_0']['_flax_linear']['bias'])
    pytorch_state_dict['h_embedder.mlp.2.weight'] = jax_to_torch(h_emb['layers_2']['_flax_linear']['kernel']).T
    pytorch_state_dict['h_embedder.mlp.2.bias'] = jax_to_torch(h_emb['layers_2']['_flax_linear']['bias'])
    
    # omega_embedder
    o_emb = net['omega_embedder']['mlp']
    pytorch_state_dict['omega_embedder.mlp.0.weight'] = jax_to_torch(o_emb['layers_0']['_flax_linear']['kernel']).T
    pytorch_state_dict['omega_embedder.mlp.0.bias'] = jax_to_torch(o_emb['layers_0']['_flax_linear']['bias'])
    pytorch_state_dict['omega_embedder.mlp.2.weight'] = jax_to_torch(o_emb['layers_2']['_flax_linear']['kernel']).T
    pytorch_state_dict['omega_embedder.mlp.2.bias'] = jax_to_torch(o_emb['layers_2']['_flax_linear']['bias'])
    
    # cfg_t_start_embedder
    ts_emb = net['cfg_t_start_embedder']['mlp']
    pytorch_state_dict['cfg_t_start_embedder.mlp.0.weight'] = jax_to_torch(ts_emb['layers_0']['_flax_linear']['kernel']).T
    pytorch_state_dict['cfg_t_start_embedder.mlp.0.bias'] = jax_to_torch(ts_emb['layers_0']['_flax_linear']['bias'])
    pytorch_state_dict['cfg_t_start_embedder.mlp.2.weight'] = jax_to_torch(ts_emb['layers_2']['_flax_linear']['kernel']).T
    pytorch_state_dict['cfg_t_start_embedder.mlp.2.bias'] = jax_to_torch(ts_emb['layers_2']['_flax_linear']['bias'])
    
    # cfg_t_end_embedder
    te_emb = net['cfg_t_end_embedder']['mlp']
    pytorch_state_dict['cfg_t_end_embedder.mlp.0.weight'] = jax_to_torch(te_emb['layers_0']['_flax_linear']['kernel']).T
    pytorch_state_dict['cfg_t_end_embedder.mlp.0.bias'] = jax_to_torch(te_emb['layers_0']['_flax_linear']['bias'])
    pytorch_state_dict['cfg_t_end_embedder.mlp.2.weight'] = jax_to_torch(te_emb['layers_2']['_flax_linear']['kernel']).T
    pytorch_state_dict['cfg_t_end_embedder.mlp.2.bias'] = jax_to_torch(te_emb['layers_2']['_flax_linear']['bias'])
    
    # y_embedder
    pytorch_state_dict['y_embedder.weight'] = jax_to_torch(net['y_embedder']['embedding_table']['_flax_embedding']['embedding'])
    
    # Shared blocks
    for i in range(cfg['num_shared_blocks']):
        jax_block = net[f'shared_blocks_{i}']
        prefix = f'shared_blocks.{i}'
        
        pytorch_state_dict[f'{prefix}.norm1.weight'] = jax_to_torch(jax_block['norm1']['kernel'])
        pytorch_state_dict[f'{prefix}.attn.q_proj.weight'] = jax_to_torch(jax_block['attn']['q_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.k_proj.weight'] = jax_to_torch(jax_block['attn']['k_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.v_proj.weight'] = jax_to_torch(jax_block['attn']['v_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.out_proj.weight'] = jax_to_torch(jax_block['attn']['out_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.q_norm.weight'] = jax_to_torch(jax_block['attn']['q_norm']['kernel'])
        pytorch_state_dict[f'{prefix}.attn.k_norm.weight'] = jax_to_torch(jax_block['attn']['k_norm']['kernel'])
        pytorch_state_dict[f'{prefix}.norm2.weight'] = jax_to_torch(jax_block['norm2']['kernel'])
        pytorch_state_dict[f'{prefix}.mlp.w1.weight'] = jax_to_torch(jax_block['mlp']['w1']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.mlp.w2.weight'] = jax_to_torch(jax_block['mlp']['w2']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.mlp.w3.weight'] = jax_to_torch(jax_block['mlp']['w3']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn_scale'] = jax_to_torch(jax_block['attn_scale'])
        pytorch_state_dict[f'{prefix}.mlp_scale'] = jax_to_torch(jax_block['mlp_scale'])
    
    # U-head blocks
    for i in range(cfg['num_head_blocks']):
        jax_block = net[f'u_heads_{i}']
        prefix = f'u_heads.{i}'
        
        pytorch_state_dict[f'{prefix}.norm1.weight'] = jax_to_torch(jax_block['norm1']['kernel'])
        pytorch_state_dict[f'{prefix}.attn.q_proj.weight'] = jax_to_torch(jax_block['attn']['q_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.k_proj.weight'] = jax_to_torch(jax_block['attn']['k_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.v_proj.weight'] = jax_to_torch(jax_block['attn']['v_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.out_proj.weight'] = jax_to_torch(jax_block['attn']['out_proj']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn.q_norm.weight'] = jax_to_torch(jax_block['attn']['q_norm']['kernel'])
        pytorch_state_dict[f'{prefix}.attn.k_norm.weight'] = jax_to_torch(jax_block['attn']['k_norm']['kernel'])
        pytorch_state_dict[f'{prefix}.norm2.weight'] = jax_to_torch(jax_block['norm2']['kernel'])
        pytorch_state_dict[f'{prefix}.mlp.w1.weight'] = jax_to_torch(jax_block['mlp']['w1']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.mlp.w2.weight'] = jax_to_torch(jax_block['mlp']['w2']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.mlp.w3.weight'] = jax_to_torch(jax_block['mlp']['w3']['_flax_linear']['kernel']).T
        pytorch_state_dict[f'{prefix}.attn_scale'] = jax_to_torch(jax_block['attn_scale'])
        pytorch_state_dict[f'{prefix}.mlp_scale'] = jax_to_torch(jax_block['mlp_scale'])
    
    # Final layers
    pytorch_state_dict['u_final_layer.norm.weight'] = jax_to_torch(net['u_final_layer']['norm']['kernel'])
    pytorch_state_dict['u_final_layer.linear.weight'] = jax_to_torch(net['u_final_layer']['linear']['_flax_linear']['kernel']).T
    pytorch_state_dict['u_final_layer.linear.bias'] = jax_to_torch(net['u_final_layer']['linear']['_flax_linear']['bias'])
    
    pytorch_state_dict['v_final_layer.norm.weight'] = jax_to_torch(net['v_final_layer']['norm']['kernel'])
    pytorch_state_dict['v_final_layer.linear.weight'] = jax_to_torch(net['v_final_layer']['linear']['_flax_linear']['kernel']).T
    pytorch_state_dict['v_final_layer.linear.bias'] = jax_to_torch(net['v_final_layer']['linear']['_flax_linear']['bias'])
    
    return pytorch_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jax_checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='iMF-B-2',
                        choices=['iMF-B-2', 'iMF-M-2', 'iMF-L-2', 'iMF-XL-2'])
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading JAX checkpoint from {args.jax_checkpoint}...")
    jax_params = load_jax_checkpoint(args.jax_checkpoint, args.model_type)
    
    print(f"Creating PyTorch model ({args.model_type})...")
    cfg = MODEL_CONFIGS[args.model_type]
    pytorch_model = MiT_PyTorch(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=cfg['hidden_size'],
        num_shared_blocks=cfg['num_shared_blocks'],
        num_head_blocks=cfg['num_head_blocks'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        num_classes=1000,
    )
    
    print("Converting parameters...")
    pytorch_state_dict = convert_params(jax_params, args.model_type)
    
    # Load state dict
    missing, unexpected = pytorch_model.load_state_dict(pytorch_state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        'model_type': args.model_type,
        'state_dict': pytorch_model.state_dict(),
        'config': cfg,
    }, args.output_path)
    
    print(f"\nSaved PyTorch checkpoint to {args.output_path}")


if __name__ == '__main__':
    main()
