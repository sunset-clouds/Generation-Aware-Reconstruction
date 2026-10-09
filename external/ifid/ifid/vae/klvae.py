from ifid.vae.autoencoder import AutoencoderKL
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import os


def _unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("state_dict", "model", "ema"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            ckpt = value
            break
    return ckpt


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixed_keys = [key for key in state_dict.keys() if key.startswith(prefix)]
    if not prefixed_keys:
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value
        else:
            stripped[key] = value
    return stripped


class KLVAE(nn.Module):
    def __init__(self, embed_dim, ch_mult, ckpt_path, *args, **kwargs):
        super().__init__()

        local_override = os.environ.get("SDVAE_CKPT", "")
        if local_override and not os.path.exists(ckpt_path):
            ckpt_path = local_override
        if not os.path.exists(ckpt_path):
            repo_id, fname = ckpt_path.rsplit("/", 1)
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
            )
        vae_ckpt = torch.load(ckpt_path, map_location="cpu")
        vae_ckpt = _unwrap_state_dict(vae_ckpt)
        for prefix in ("vae.", "model.", "module.", "autoencoder.", "first_stage_model."):
            vae_ckpt = _strip_prefix_if_present(vae_ckpt, prefix)
        self.vae = AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            use_variational=True,
        )
        self.vae.load_state_dict(vae_ckpt, strict=False)

    def encode(self, x, *args, **kwargs):
        return self.vae.encode(x, sample=True)

    def decode(self, z, *args, **kwargs):
        return self.vae.decode(z).sample


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/SDVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/REPAEVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/INVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/VAVAE.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/VAVAE64.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
