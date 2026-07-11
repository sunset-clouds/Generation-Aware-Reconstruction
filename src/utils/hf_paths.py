import os
from pathlib import Path
from typing import Dict, Tuple


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def hf_offline_mode() -> bool:
    return _env_flag("HF_HUB_OFFLINE") or _env_flag("TRANSFORMERS_OFFLINE")


def resolve_sd_vae_source(vae_type: str = "mse") -> Tuple[str, Dict[str, object], str]:
    """
    Resolve where Stable Diffusion VAE should be loaded from.

    Priority:
    1. Explicit local path via env vars.
    2. HF repo id (online or local cache, depending on offline flags).
    """
    env_candidates = (
        "SD_VAE_PATH",
        "IMF_SD_VAE_PATH",
        "HF_SD_VAE_PATH",
    )
    for env_name in env_candidates:
        raw_value = os.getenv(env_name)
        if not raw_value:
            continue
        path = Path(raw_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"{env_name} is set but does not exist: {path}"
            )
        return (
            str(path),
            {"local_files_only": True},
            f"[HFPaths] Loading SD VAE from {env_name}={path}",
        )

    repo_id = f"stabilityai/sd-vae-ft-{vae_type}"
    kwargs: Dict[str, object] = {}
    if hf_offline_mode():
        kwargs["local_files_only"] = True
        message = f"[HFPaths] Loading SD VAE from local HF cache: {repo_id}"
    else:
        message = f"[HFPaths] Loading SD VAE from Hugging Face repo: {repo_id}"
    return repo_id, kwargs, message
