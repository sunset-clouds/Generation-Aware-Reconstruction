import torch
import torch.nn as nn
from pathlib import Path
from ifid.vae.rae_module.rae import instantiate_rae_from_config, parse_rae_configs


class RAE(nn.Module):
    def __init__(self, rae_config, *args, **kwargs):
        super().__init__()
        if isinstance(rae_config, str) and not Path(rae_config).exists():
            fallback = Path(__file__).resolve().parents[2] / "configs" / Path(rae_config).name
            if fallback.exists():
                rae_config = str(fallback)
        rae_config, *_ = parse_rae_configs(rae_config)
        self.rae = instantiate_rae_from_config(rae_config)

    def encode(
        self,
        x,
        *args, **kwargs
    ):
        # x -> [0, 1]
        x = (x + 1.0) / 2.0
        z = self.rae.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        xhat = self.rae.decode(z)
        xhat = xhat * 2.0 - 1.0
        return xhat


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/RAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
