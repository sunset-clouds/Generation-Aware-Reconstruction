import os

import torch
import torch.nn as nn

from ifid.vae.continous_tokenizer.tokenizer import FlowModel, ModelArgs
from ifid.vae.continous_tokenizer.tokenizer import SoftVQModel, AEModel

import warnings
warnings.filterwarnings("ignore", message=".*timm_vit.*")


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_local_files_only(kwargs) -> bool:
    if "local_files_only" in kwargs:
        return bool(kwargs.pop("local_files_only"))
    return _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE")


def _default_softvq_repo() -> str:
    return str(
        os.environ.get(
            "SOFTVQ_HF_REPO",
            "SoftVQVAE/softvq-l-64",
        )
    )


def _softvq_wants_tc_permute() -> bool:
    """If true, map SoftVQ latents to (B, tokens, channels) by transposing the last two dims when ndim==3.

    This must NOT be enabled as a global default: different SiT SoftVQ checkpoints may already match
    the underlying SoftVQ model layout, and an unconditional transpose is exactly what can trigger
    latent input_size mismatch against SiT pos_embed.
    """
    return str(os.environ.get("SOFTVQ_LATENT_PERMUTE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "tc",
        "tokens_channels",
    }


def _softvq_tc_transpose_if_needed(z: torch.Tensor) -> torch.Tensor:
    if not _softvq_wants_tc_permute():
        return z
    if z.ndim == 3:
        return z.transpose(1, 2).contiguous()
    return z

class ContinuousTokenizerVAE(nn.Module):
    def __init__(self, ckpt_file, *args, **kwargs):
        super().__init__()

        self.supports_y = False
        self.num_classes = 0

        modelargs = ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4],
            decoder_ch_mult=[1, 1, 2, 2, 4],
            num_latent_tokens=256,
            codebook_embed_dim=32,
            entropy_loss_ratio=0.0,
            enc_type="flow",
            encoder_model="vit_base_patch14_dinov2.lvd142m",
            dec_type="vit",
            decoder_model="vit_tinytiny_patch14_dinov2_movq2",
            use_ape=False,
            use_rope=True,
            rope_mixed=True,
            rope_theta=10.0,
            enc_token_drop=0.4,
            enc_token_drop_max=0.6,
            aux_loss_mask=True,
            aux_hog_dec=False,
            aux_dino_dec=False,
            aux_clip_dec=False,
            repa=True,
            repa_model="vit_large_patch14_dinov2.lvd142m",
            repa_patch_size=16,
            repa_proj_dim=1024,
            repa_loss_weight=0.1,
            repa_align="repeat",
        )

        self.model = FlowModel(config=modelargs)

        flow_ckpt = torch.load(ckpt_file, map_location="cpu")["model"]
        missing, unexpected = self.model.load_state_dict(flow_ckpt, strict=False)
        print("missing", missing)
        print("unexp", unexpected)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode(self, x, *args, **kwargs):
        return self.model.encode(x)

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


class SOFTVQ(nn.Module):
    def __init__(self, hf_repo=None, *args, **kwargs):
        super().__init__()
        if hf_repo is None:
            hf_repo = _default_softvq_repo()
        local_files_only = _resolve_local_files_only(kwargs)
        try:
            self.model = SoftVQModel.from_pretrained(
                hf_repo,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Failed to load SOFTVQ repo {hf_repo!r} from local cache. "
                    "Pre-download it on a login node first, or retry without offline mode."
                ) from exc
            raise

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        # Default: return SoftVQ latents as produced by the HF model. If a specific SiT checkpoint
        # family requires a (B, tokens, channels) layout and your SoftVQ head outputs (B, channels, tokens)
        # (or the converse), set SOFTVQ_LATENT_PERMUTE=1 to apply the 3D transpose.
        return _softvq_tc_transpose_if_needed(z)

    def decode(self, z, *args, **kwargs):
        return self.model.decode(_softvq_tc_transpose_if_needed(z))


class MAETOK(nn.Module):
    def __init__(self, hf_repo="MAETok/maetok-b-128", *args, **kwargs):
        super().__init__()
        local_files_only = _resolve_local_files_only(kwargs)
        try:
            self.model = AEModel.from_pretrained(
                hf_repo,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Failed to load MAETOK repo {hf_repo!r} from local cache. "
                    "Pre-download it on a login node first, or retry without offline mode."
                ) from exc
            raise

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


class FLOWEXPORT(nn.Module):
    def __init__(self, hf_repo="MAETok/maetok-b-128", *args, **kwargs):
        super().__init__()
        local_files_only = _resolve_local_files_only(kwargs)
        self.model = AEModel.from_pretrained(
            hf_repo,
            local_files_only=local_files_only,
        )

    def encode(self, x, *args, **kwargs):
        z, _, _ = self.model.encode(x)
        return z

    def decode(self, z, *args, **kwargs):
        return self.model.decode(z)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from ifid.vae.utils import instantiate_from_config

    configs = OmegaConf.load("../../configs/SOFTVQ.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)

    configs = OmegaConf.load("../../configs/MAETOK.yaml")
    sdvae = instantiate_from_config(configs)
    z = sdvae.encode(torch.randn([1, 3, 256, 256]))
    xhat = sdvae.decode(z)
    print(z.shape, xhat.shape)
