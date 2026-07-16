"""
Models for iMF-GAP.

Structure follows the GAR-FID training convention:
- diffusion_pytorch.py / imf_official.py: frozen iMF wrappers
- tokenizer.py: Tokenizer (VAE with trainable decoder)
- model.py: TokenizerFlowComposition (integrates diffusion + tokenizer)
- losses.py: PostTrainingLoss (discriminator + losses)

For training:
    from models import get_model, get_post_training_loss
    model = get_model()(args, device)
    loss_fn = get_post_training_loss()(args)

JAX is not a runtime backend. Legacy Flax checkpoints can be converted with
``src/tools/convert_jax_to_pytorch.py`` before training or evaluation.
"""

def get_tokenizer():
    """Get Tokenizer class (VAE with trainable decoder)."""
    from models.tokenizer import Tokenizer
    return Tokenizer


def get_model():
    """Get TokenizerFlowComposition class (main model for training)."""
    from models.model import TokenizerFlowComposition
    return TokenizerFlowComposition


def get_post_training_loss():
    """Get PostTrainingLoss class (discriminator + losses)."""
    from models.losses import PostTrainingLoss
    return PostTrainingLoss
