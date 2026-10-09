"""
Models for generation-aware reconstruction and decoder adaptation.

Structure follows the GAR-FID training convention:
- imf_official.py: frozen official iMF .pth wrapper
- tokenizer.py: Tokenizer (VAE with trainable decoder)
- model.py: TokenizerFlowComposition (integrates diffusion + tokenizer)
- losses.py: PostTrainingLoss (discriminator + losses)

For training:
    from models import get_model, get_post_training_loss
    model = get_model()(args)
    loss_fn = get_post_training_loss()(args)

Training and evaluation require official iMF .pth checkpoints.
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
