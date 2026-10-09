"""Shared diffusion interface for GAR-FID adapters."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class DiffusionInterface(ABC):
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
