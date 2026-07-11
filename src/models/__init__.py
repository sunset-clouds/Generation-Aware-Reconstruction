"""
Models for iMF-GAP.

Structure follows VQ-Transplant convention:
- diffusion.py: DiffusionModel (frozen iMF)
- tokenizer.py: Tokenizer (VAE with trainable decoder)
- model.py: TokenizerFlowComposition (integrates diffusion + tokenizer)
- loss.py: PostTrainingLoss (discriminator + losses)

For training:
    from models import get_model, get_post_training_loss
    model = get_model()(args, device)
    loss_fn = get_post_training_loss()(args)

For evaluation:
    from models import get_diffusion_model, get_tokenizer
    diffusion = get_diffusion_model()(args)
    tokenizer = get_tokenizer()(args)
"""

# Lazy imports to avoid JAX initialization issues
def get_diffusion_model():
    """Get DiffusionModel class (frozen iMF)."""
    from models.diffusion import DiffusionModel
    return DiffusionModel


def get_tokenizer():
    """Get Tokenizer class (VAE with trainable decoder)."""
    from models.tokenizer import Tokenizer
    return Tokenizer


def get_jax_tokenizer():
    """Get JAXTokenizer class (for fast multi-GPU evaluation)."""
    from models.tokenizer import JAXTokenizer
    return JAXTokenizer


def get_model():
    """Get TokenizerFlowComposition class (main model for training)."""
    from models.model import TokenizerFlowComposition
    return TokenizerFlowComposition


def get_post_training_loss():
    """Get PostTrainingLoss class (discriminator + losses)."""
    from models.loss import PostTrainingLoss
    return PostTrainingLoss


# Legacy API for backwards compatibility
def get_combined_model():
    """Legacy: Get CombinedModel class."""
    from models.model import TokenizerFlowComposition
    return TokenizerFlowComposition


def get_training_loss():
    """Legacy: Get TrainingLoss class."""
    from models.loss import PostTrainingLoss
    return PostTrainingLoss


# Conversion utilities
def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    import jax.numpy as jnp
    return jnp.array(tensor.detach().cpu().numpy())


def jax_to_torch(jax_array, device):
    """Convert JAX array to PyTorch tensor."""
    import torch
    import numpy as np
    return torch.from_numpy(np.array(jax_array)).to(device)
