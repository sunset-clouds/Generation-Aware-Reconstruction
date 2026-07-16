"""
Convert iMF JAX/Flax checkpoint to PyTorch.

Usage:
    python src/tools/convert_jax_to_pytorch.py \
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
from collections import OrderedDict

import jax
import jax.numpy as jnp
from flax.training import checkpoints, train_state
import optax

from models.imf_torch.converted_arch import MODEL_CONFIGS, MiT_PyTorch


def load_jax_checkpoint(checkpoint_path, model_type):
    """Load JAX checkpoint."""
    from tools.jax_conversion.imf import iMeanFlow
    
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
