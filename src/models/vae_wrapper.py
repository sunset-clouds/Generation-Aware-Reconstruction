"""
VAE decode wrapper matching imeanflow-torch's VAEWrapper.
Uses torch.compile and channels_last for bit-exact match with official FID.
"""

import torch
from diffusers.models import AutoencoderKL
from utils.hf_paths import resolve_sd_vae_source


class VAEWrapper(torch.nn.Module):
    """
    VAE decode-only wrapper matching imeanflow-torch evaluate.py.
    Uses torch.compile + channels_last for reproducibility.
    """

    def __init__(self, decode_batch_size=64, vae_type='mse', dtype=torch.float32):
        super().__init__()
        vae_source, load_kwargs, load_message = resolve_sd_vae_source(vae_type)
        print(load_message.replace("[HFPaths]", "[VAEWrapper]"))
        vae = AutoencoderKL.from_pretrained(
            vae_source,
            torch_dtype=dtype,
            **load_kwargs,
        )
        del vae.encoder
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        self.vae = vae
        self.latent_size = 32
        self.register_buffer(
            'mean',
            torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409]).view(1, 4, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003]).view(1, 4, 1, 1)
        )
        self.vae.to(memory_format=torch.channels_last)
        try:
            self.compiled_decode = torch.compile(
                self.vae.decode, mode="reduce-overhead", fullgraph=True
            )
        except Exception:
            self.compiled_decode = self.vae.decode

    def decode(self, latents):
        """Decode normalized latent (B,4,H,W) to image (B,3,H,W) in [-1,1]."""
        assert latents.shape[1:] == (4, self.latent_size, self.latent_size)
        z = latents * self.std + self.mean
        z = z.contiguous(memory_format=torch.channels_last)
        out = self.compiled_decode(z)["sample"]
        return out.contiguous(memory_format=torch.channels_last)
