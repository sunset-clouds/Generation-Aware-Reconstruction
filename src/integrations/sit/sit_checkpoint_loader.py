from __future__ import annotations

import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT
IMF_GAP_ROOT = PROJECT_ROOT
DEFAULT_IFID_REPO_ROOT = PROJECT_ROOT / "third_party" / "ifid"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "assets" / "sit_checkpoint_registry_template.csv"
KNOWN_REPA_KEYWORDS = ("repa", "projector", "linear", "proj")
SHIFTED_REGISTRY_KEYS = [
    "imagenet_train_dir",
    "imagenet_val_dir",
    "label_map_json",
    "fid_reference_file",
    "load_policy",
    "load_check",
    "inference_check",
    "rfid_check",
    "ifid_check",
    "gfid_check",
    "garfid_check",
    "pcc_srcc_check",
    "notes",
]
VAE_TYPE_HINTS = {
    "sdvae": "mse",
    "eqvae": "eqvae",
    "invae": "mse",
    "repaevae": "mse",
    "vavae": "mse",
    "vavae64": "mse",
    "fluxvae": "flux",
    "sd3vae": "sd3",
    "qwvae": "qwen",
    "dmvae": "dmvae",
    "detok": "detok",
    "maetok": "maetok",
    "softvq": "softvq",
    "rae": "rae",
}
REQUIRED_IFID_RUNTIME_FILES = (
    "ifid/sit/sit.py",
    "ifid/vae/utils.py",
)
REQUIRED_CONTINUOUS_RUNTIME_FILES = (
    "ifid/vae/cont.py",
    "cont/modelling/tokenizer.py",
    "cont/modelling/jet.py",
)


@dataclass
class SitCheckpointSpec:
    group_id: str
    vae_name: str
    vae_config: str
    vae_type: str
    latent_kind: str
    sit_scale: str
    sit_model: str
    ifid_exp_name: str
    exp_path: str
    args_json: Optional[str] = None
    ckpt_step: Optional[int] = None
    ckpt_file: Optional[str] = None
    cfg_modes: str = "nocfg|cfg"
    load_policy: str = "strict_false_allow_known"
    notes: str = ""
    ifid_repo_root: str = str(DEFAULT_IFID_REPO_ROOT)
    row_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_registry_row(
        cls,
        row: Mapping[str, Any],
        *,
        ifid_repo_root: Optional[str] = None,
    ) -> "SitCheckpointSpec":
        row_dict = dict(row)
        ckpt_step = row_dict.get("ckpt_step") or None
        return cls(
            group_id=str(row_dict.get("group_id", "")).strip(),
            vae_name=str(row_dict.get("vae_name", "")).strip(),
            vae_config=str(row_dict.get("vae_config", "")).strip(),
            vae_type=str(row_dict.get("vae_type", "")).strip(),
            latent_kind=str(row_dict.get("latent_kind", "")).strip(),
            sit_scale=str(row_dict.get("sit_scale", "")).strip(),
            sit_model=str(row_dict.get("sit_model", "")).strip(),
            ifid_exp_name=str(row_dict.get("ifid_exp_name", "")).strip(),
            exp_path=str(row_dict.get("exp_path", "")).strip(),
            args_json=_clean_optional(row_dict.get("args_json")),
            ckpt_step=int(ckpt_step) if ckpt_step not in (None, "") else None,
            ckpt_file=_clean_optional(row_dict.get("ckpt_file")),
            cfg_modes=str(row_dict.get("cfg_modes", "nocfg|cfg")).strip() or "nocfg|cfg",
            load_policy=str(row_dict.get("load_policy", "strict_false_allow_known")).strip()
            or "strict_false_allow_known",
            notes=str(row_dict.get("notes", "")).strip(),
            ifid_repo_root=str(ifid_repo_root or row_dict.get("ifid_repo_root") or DEFAULT_IFID_REPO_ROOT),
            row_data=row_dict,
        )


@dataclass
class SitCheckpointBundle:
    spec: SitCheckpointSpec
    args_dict: Dict[str, Any]
    vae: torch.nn.Module
    sit_model: torch.nn.Module
    checkpoint_path: str
    latent_shape: List[int]
    latent_size: int
    in_channels: int
    vae_1d: bool
    resolution: int
    tshift: float
    latent_stats: Dict[str, torch.Tensor]
    load_report: Dict[str, Any]
    ifid_modules: Dict[str, Any]


def _clean_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_registry_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    extra_values = normalized.get(None) or []
    looks_shifted = bool(extra_values) and (
        str(normalized.get("load_policy", "")).endswith(".npz")
        or str(normalized.get("fid_reference_file", "")).endswith(".json")
    )
    if looks_shifted:
        current_values = [str(normalized.get(key, "")) for key in SHIFTED_REGISTRY_KEYS]
        shifted_values = current_values[1:] + [str(extra_values[0])]
        for key, value in zip(SHIFTED_REGISTRY_KEYS, shifted_values):
            normalized[key] = value
    normalized.pop(None, None)
    return normalized


def resolve_ifid_repo_root(ifid_repo_root: Optional[str] = None) -> str:
    candidate = Path(ifid_repo_root or os.environ.get("IFID_REPO_ROOT", DEFAULT_IFID_REPO_ROOT))
    if not candidate.exists():
        raise FileNotFoundError(
            f"IFID repo root not found: {candidate}. "
            "Pass ifid_repo_root explicitly or set IFID_REPO_ROOT."
        )
    if not (candidate / "ifid").exists():
        raise FileNotFoundError(
            f"{candidate} does not look like the IFID repo root because `ifid/` is missing."
        )
    return str(candidate.resolve())


@contextmanager
def _prepend_sys_path(path: str) -> Iterator[None]:
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def load_registry_row(
    group_id: str,
    *,
    registry_path: Optional[str] = None,
) -> Dict[str, str]:
    path = Path(registry_path or DEFAULT_REGISTRY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Registry CSV not found: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = _normalize_registry_row(row)
            if str(normalized.get("group_id", "")).strip() == group_id:
                return normalized
    raise KeyError(f"group_id={group_id!r} not found in registry {path}")


def _load_args_json(spec: SitCheckpointSpec) -> Dict[str, Any]:
    args_json = spec.args_json or os.path.join(spec.exp_path, "args.json")
    args_path = _resolve_candidate_path(
        args_json,
        [Path(spec.exp_path), PROJECT_ROOT, SRC_ROOT, Path(spec.ifid_repo_root)],
    )
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_path}")
    with args_path.open("r") as handle:
        return json.load(handle)


def _resolve_candidate_path(path_value: str, search_roots: List[Path]) -> Path:
    path_candidate = Path(path_value)
    if path_candidate.is_absolute() and path_candidate.exists():
        return path_candidate
    for root in search_roots:
        combined = (root / path_value).resolve()
        if combined.exists():
            return combined
    if path_candidate.exists():
        return path_candidate.resolve()
    raise FileNotFoundError(f"Could not resolve path {path_value!r} under {[str(r) for r in search_roots]}")


def _infer_vae_type(spec: SitCheckpointSpec, args_dict: Mapping[str, Any]) -> str:
    if spec.vae_type:
        return spec.vae_type
    for key in ("vae_type",):
        value = _clean_optional(args_dict.get(key))
        if value:
            return value
    config_name = Path(spec.vae_config or str(args_dict.get("vae_config", ""))).stem.lower()
    for hint, vae_type in VAE_TYPE_HINTS.items():
        if hint in config_name:
            return vae_type
    raise ValueError(
        f"Could not infer vae_type for group_id={spec.group_id}. "
        "Fill it in the registry before running experiments."
    )


def _load_ifid_modules(ifid_repo_root: str) -> Dict[str, Any]:
    missing_runtime = [
        rel_path
        for rel_path in REQUIRED_IFID_RUNTIME_FILES
        if not (Path(ifid_repo_root) / rel_path).exists()
    ]
    if missing_runtime:
        raise FileNotFoundError(
            "Incomplete IFID repo: missing required files under IFID repo root "
            f"{ifid_repo_root}: {missing_runtime}"
        )
    with _prepend_sys_path(ifid_repo_root):
        try:
            from ifid.sit.sit import SiT_models  # type: ignore
            from ifid.sit.samplers import compute_diffusion, get_score_from_velocity  # type: ignore
            from ifid.vae.utils import instantiate_from_config  # type: ignore
            from omegaconf import OmegaConf  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import IFID runtime modules. "
                f"Check that {ifid_repo_root} contains a complete IFID checkout "
                "with tokenizer dependencies installed."
            ) from exc

    return {
        "SiT_models": SiT_models,
        "compute_diffusion": compute_diffusion,
        "get_score_from_velocity": get_score_from_velocity,
        "instantiate_from_config": instantiate_from_config,
        "OmegaConf": OmegaConf,
    }


def _extract_encoded_tensor(encoded: Any) -> Optional[torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        return encoded
    if isinstance(encoded, (list, tuple)):
        for item in encoded:
            tensor = _extract_encoded_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(encoded, dict):
        for key in ("latents", "z", "latent", "samples", "sample"):
            if key in encoded:
                tensor = _extract_encoded_tensor(encoded[key])
                if tensor is not None:
                    return tensor
        for value in encoded.values():
            tensor = _extract_encoded_tensor(value)
            if tensor is not None:
                return tensor
        return None
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is not None:
        for attr in ("mode", "mean", "sample"):
            candidate = getattr(latent_dist, attr, None)
            if callable(candidate):
                tensor = _extract_encoded_tensor(candidate())
            else:
                tensor = _extract_encoded_tensor(candidate)
            if tensor is not None:
                return tensor
    for attr in ("latents", "z", "latent", "mean"):
        if hasattr(encoded, attr):
            tensor = _extract_encoded_tensor(getattr(encoded, attr))
            if tensor is not None:
                return tensor
    sample_fn = getattr(encoded, "sample", None)
    if callable(sample_fn):
        tensor = _extract_encoded_tensor(sample_fn())
        if tensor is not None:
            return tensor
    return None


def _encode_vae(vae: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        try:
            encoded = vae.encode(batch)
        except TypeError:
            encoded = vae.encode(batch, None)
    tensor = _extract_encoded_tensor(encoded)
    if tensor is None:
        raise TypeError(
            "Expected `vae.encode` to return or contain a Tensor, "
            f"got {type(encoded)!r}"
        )
    return tensor


def _classify_unexpected_keys(unexpected_keys: List[str]) -> Dict[str, Any]:
    known = []
    unknown = []
    for key in unexpected_keys:
        lowered = key.lower()
        if any(token in lowered for token in KNOWN_REPA_KEYWORDS):
            known.append(key)
        else:
            unknown.append(key)
    return {
        "known_unexpected_keys": known,
        "unknown_unexpected_keys": unknown,
    }


def _infer_input_size_from_pos_embed(
    sit_model_name: str,
    pos_embed: Any,
) -> Optional[int]:
    if not isinstance(pos_embed, torch.Tensor) or pos_embed.ndim != 3:
        return None
    num_tokens = int(pos_embed.shape[1])
    if sit_model_name.endswith("1D"):
        return num_tokens
    if "/" not in sit_model_name:
        return None
    patch_suffix = sit_model_name.rsplit("/", 1)[-1]
    try:
        patch_size = int(patch_suffix)
    except ValueError:
        return None
    side = math.isqrt(num_tokens)
    if side * side != num_tokens:
        return None
    return side * patch_size


class SitCheckpointLoader:
    """Controlled loader for IFID SiT checkpoints."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        ifid_repo_root: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        self.ifid_repo_root = resolve_ifid_repo_root(ifid_repo_root)
        self.ifid_modules = _load_ifid_modules(self.ifid_repo_root)

    def load_from_registry_row(self, row: Mapping[str, Any]) -> SitCheckpointBundle:
        return self.load_bundle(SitCheckpointSpec.from_registry_row(row, ifid_repo_root=self.ifid_repo_root))

    def load_from_group_id(
        self,
        group_id: str,
        *,
        registry_path: Optional[str] = None,
    ) -> SitCheckpointBundle:
        row = load_registry_row(group_id, registry_path=registry_path)
        return self.load_from_registry_row(row)

    def load_bundle(self, spec: SitCheckpointSpec) -> SitCheckpointBundle:
        args_dict = _load_args_json(spec)

        search_roots = [
            Path(spec.exp_path),
            PROJECT_ROOT,
            SRC_ROOT,
            Path(self.ifid_repo_root),
        ]
        vae_config_value = spec.vae_config or str(args_dict.get("vae_config", "")).strip()
        if not vae_config_value:
            raise ValueError(f"Missing vae_config for group_id={spec.group_id}")
        vae_config_path = _resolve_candidate_path(vae_config_value, search_roots)
        spec.vae_config = str(vae_config_path)
        spec.vae_type = _infer_vae_type(spec, args_dict)
        if spec.vae_type in {"softvq", "maetok"}:
            missing_runtime = [
                rel_path
                for rel_path in REQUIRED_CONTINUOUS_RUNTIME_FILES
                if not (Path(self.ifid_repo_root) / rel_path).exists()
            ]
            if missing_runtime:
                raise FileNotFoundError(
                    "Missing continuous tokenizer runtime files under IFID repo "
                    f"{self.ifid_repo_root}: {missing_runtime}"
                )

        checkpoint_path = self._resolve_checkpoint_path(spec)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
        instantiate_from_config = self.ifid_modules["instantiate_from_config"]
        OmegaConf = self.ifid_modules["OmegaConf"]
        SiT_models = self.ifid_modules["SiT_models"]

        vae = instantiate_from_config(OmegaConf.load(spec.vae_config)).to(self.device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)

        resolution = int(args_dict.get("resolution", 256))
        fake_in = torch.zeros((1, 3, resolution, resolution), device=self.device)
        fake_z = _encode_vae(vae, fake_in)[0]

        if fake_z.ndim == 3:
            latent_size = int(fake_z.shape[-1])
            in_channels = int(fake_z.shape[0])
            vae_1d = False
        elif fake_z.ndim == 2:
            latent_size = int(fake_z.shape[0])
            in_channels = int(fake_z.shape[-1])
            vae_1d = True
        else:
            raise ValueError(
                f"Unsupported latent shape {tuple(fake_z.shape)} for group_id={spec.group_id}"
            )

        expected_input_size = _infer_input_size_from_pos_embed(
            spec.sit_model,
            state_dict.get("pos_embed"),
        )
        if expected_input_size is not None and expected_input_size != latent_size:
            raise RuntimeError(
                f"Latent input_size mismatch for group_id={spec.group_id}: "
                f"VAE config {spec.vae_config} produced latent shape {tuple(fake_z.shape)} "
                f"(input_size={latent_size}), but checkpoint {checkpoint_path} expects "
                f"input_size={expected_input_size} from pos_embed for {spec.sit_model}. "
                "This usually means the tokenizer config downsampling ratio does not match "
                "the SiT checkpoint family."
            )

        tshift = math.sqrt(float(fake_z.numel()) / 4096.0)
        block_kwargs = {
            "fused_attn": bool(args_dict.get("fused_attn", True)),
            "qk_norm": bool(args_dict.get("qk_norm", False)),
        }
        sit_model = SiT_models[spec.sit_model](
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=int(args_dict.get("num_classes", 1000)),
            class_dropout_prob=float(args_dict.get("cfg_prob", 0.1)),
            bn_momentum=float(args_dict.get("bn_momentum", 0.1)),
            tshift=tshift,
            **block_kwargs,
        ).to(self.device)
        sit_model.eval()

        load_result = sit_model.load_state_dict(state_dict, strict=False)
        unexpected_report = _classify_unexpected_keys(list(load_result.unexpected_keys))

        bundle = SitCheckpointBundle(
            spec=spec,
            args_dict=args_dict,
            vae=vae,
            sit_model=sit_model,
            checkpoint_path=str(checkpoint_path),
            latent_shape=list(fake_z.shape),
            latent_size=latent_size,
            in_channels=in_channels,
            vae_1d=vae_1d,
            resolution=resolution,
            tshift=tshift,
            latent_stats=sit_model.extract_latents_stats(),
            load_report={
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
                **unexpected_report,
            },
            ifid_modules=self.ifid_modules,
        )
        return bundle

    def _resolve_checkpoint_path(self, spec: SitCheckpointSpec) -> Path:
        if spec.ckpt_file:
            return _resolve_candidate_path(
                spec.ckpt_file,
                [Path(spec.exp_path), PROJECT_ROOT, SRC_ROOT, Path(self.ifid_repo_root)],
            )
        if spec.ckpt_step is None:
            raise ValueError(
                f"Missing ckpt_file and ckpt_step for group_id={spec.group_id}. "
                "Fill one of them in the registry."
            )
        checkpoint_name = f"{int(spec.ckpt_step):07d}.pt"
        checkpoint_path = Path(spec.exp_path) / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path.resolve()
