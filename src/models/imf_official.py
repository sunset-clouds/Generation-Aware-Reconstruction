"""
Wrapper for official iMeanFlow PyTorch model (.pth checkpoint).

Uses imeantflow-torch's iMeanFlow and matches the evaluate.py interface.
Provides the same API as DiffusionModelPyTorch for generate_eval_images.py.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# Model name mapping: iMF-B-2 -> imfDiT_B_2
MODEL_STR_MAP = {
    'iMF-B-2': 'imfDiT_B_2',
    'iMF-M-2': 'imfDiT_M_2',
    'iMF-L-2': 'imfDiT_L_2',
    'iMF-XL-2': 'imfDiT_XL_2',
}

# CFG params from imeantflow-torch README
CFG_PARAMS = {
    'iMF-B-2': {'omega': 8.0, 't_min': 0.4, 't_max': 0.65},
    'iMF-M-2': {'omega': 10.5, 't_min': 0.4, 't_max': 0.6},
    'iMF-L-2': {'omega': 10.5, 't_min': 0.4, 't_max': 0.6},
    'iMF-XL-2': {'omega': 8.0, 't_min': 0.42, 't_max': 0.62},
}


class BatchGenerator:
    """Deterministic noise generator matching imeantflow-torch's tu.BatchGenerator."""
    def __init__(self, device, seeds):
        self.device = device
        if hasattr(seeds, 'cpu'):
            seeds = seeds.cpu().tolist()
        self.generators = [
            torch.Generator("cpu").manual_seed(int(s) % (1 << 32)) for s in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([
            torch.randn(size[1:], generator=gen, **kwargs).to(self.device)
            for gen in self.generators
        ])

    def randint(self, low, high, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([
            torch.randint(low, high, size=size[1:], generator=gen, **kwargs).to(self.device)
            for gen in self.generators
        ])


class OfficialImfWrapper(nn.Module):
    """
    Wrapper for official iMeanFlow (.pth) providing DiffusionInterface-compatible API.
    """

    def __init__(self, model_type: str, checkpoint_path: str, imf_torch_path: Optional[str] = None):
        super().__init__()
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.latent_size = 32
        self.num_classes = 1000

        cfg = CFG_PARAMS[model_type]
        self.omega = cfg['omega']
        self.t_min = cfg['t_min']
        self.t_max = cfg['t_max']

        # Use embedded imf_torch (avoids 'models' namespace conflict)
        from models.imf_torch.imf import iMeanFlow
        model_str = MODEL_STR_MAP[model_type]
        self.model = iMeanFlow(model_str, eval=True)
        self._load_checkpoint()
        self._is_loaded = True

    def _load_checkpoint(self):
        print(f"[OfficialImfWrapper] Loading from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt)
        print(f"[OfficialImfWrapper] Loaded {self.model_type}")

    def to(self, device):
        super().to(device)
        self.model = self.model.to(device)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    @staticmethod
    def add_noise(z: torch.Tensor, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flow matching: z_t = (1 - t) * z + t * noise"""
        noise = torch.randn_like(z)
        z_t = (1 - noise_level) * z + noise_level * noise
        return z_t, noise

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        use_cfg: bool = True,
        device: str = 'cuda',
        generator: Optional[torch.Generator] = None,
        seeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate latents from noise. Returns NCHW tensor (normalized latent)."""
        omega = self.omega if use_cfg else 1.0
        t_min = self.t_min if use_cfg else 0.0
        t_max = self.t_max if use_cfg else 1.0

        # Build seeds for BatchGenerator (match official evaluate.py: seeds = sample_idx ^ initial_seed)
        if seeds is None:
            seeds = (torch.arange(batch_size, device=device, dtype=torch.int64) ^ 42)
        rng = BatchGenerator(device, seeds)

        if labels is None:
            labels = rng.randint(0, self.num_classes, (batch_size,))

        z_gen = self.model.generate(
            n_sample=batch_size,
            rng=rng,
            num_steps=num_steps,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        return z_gen  # Already NCHW (B, 4, 32, 32), normalized latent

    @torch.no_grad()
    def denoise(
        self,
        z_noisy: torch.Tensor,
        noise_level: float,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        use_cfg: bool = True,
    ) -> torch.Tensor:
        """Denoise from z_noisy at noise_level. z_noisy is NCHW normalized latent."""
        device = z_noisy.device
        B = z_noisy.shape[0]
        if labels is None:
            labels = torch.randint(0, self.num_classes, (B,), device=device)

        t_steps = torch.linspace(noise_level, 0.0, num_steps + 1, device=device)
        omega_val = self.omega if use_cfg else 1.0
        t_min_val = self.t_min if use_cfg else 0.0
        t_max_val = self.t_max if use_cfg else 1.0
        omega = torch.tensor(omega_val, dtype=z_noisy.dtype, device=device)
        t_min = torch.tensor(t_min_val, dtype=z_noisy.dtype, device=device)
        t_max = torch.tensor(t_max_val, dtype=z_noisy.dtype, device=device)

        z_t = z_noisy
        for i in range(num_steps):
            z_t = self.model.sample_one_step(z_t, labels, i, t_steps, omega, t_min, t_max)
        return z_t
