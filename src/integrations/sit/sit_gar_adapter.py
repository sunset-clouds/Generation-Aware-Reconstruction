from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.diffusion_pytorch import DiffusionInterface

from .sit_checkpoint_loader import SitCheckpointBundle, SitCheckpointLoader


SDE_RAW_MIN_TIME = 0.04


@dataclass
class GarDenoiseOutputs:
    eta_t: float
    sampling_mode: str
    t_raw_start: float
    t_shifted_start: float
    time_map_abs_error: float
    requested_num_steps: int
    resolved_num_steps: int
    step_policy: str
    sde_uses_short_schedule: bool
    z_vae: torch.Tensor
    z_model: torch.Tensor
    z_noisy: torch.Tensor
    z_denoised_model: torch.Tensor
    z_denoised_vae: torch.Tensor
    x_rec: torch.Tensor


@dataclass(frozen=True)
class SamplingPlan:
    noise_level: float
    sampling_mode: str
    step_policy: str
    requested_num_steps: int
    resolved_num_steps: int
    min_num_steps: int
    max_num_steps: int
    t_raw_start: float
    t_shifted_start: float
    time_map_abs_error: float
    sde_uses_short_schedule: bool

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


class SitGarAdapter(nn.Module, DiffusionInterface):
    """
    Bridge GAR-style reconstruction onto IFID SiT checkpoints.

    Naming convention:
    - `vae latents` are the raw latents expected by the IFID VAE decoder.
    - `model latents` are the normalized latents seen by SiT after applying the
      batch-norm statistics stored in the checkpoint.
    """

    def __init__(
        self,
        bundle: SitCheckpointBundle,
        *,
        device: Optional[str] = None,
        default_num_steps: int = 50,
        default_cfg_scale: float = 1.0,
        guidance_low: float = 0.0,
        guidance_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.spec = bundle.spec
        first_param_device = next(bundle.vae.parameters()).device
        self.device = torch.device(device) if device is not None else first_param_device
        self.vae = bundle.vae.to(self.device)
        self.model = bundle.sit_model.to(self.device)
        self.default_num_steps = default_num_steps
        self.default_cfg_scale = default_cfg_scale
        self.guidance_low = guidance_low
        self.guidance_high = guidance_high

        self.vae.eval()
        self.model.eval()

    @property
    def tshift(self) -> float:
        return float(self.bundle.tshift)

    @classmethod
    def from_group_id(
        cls,
        group_id: str,
        *,
        device: str = "cuda",
        registry_path: Optional[str] = None,
        ifid_repo_root: Optional[str] = None,
        default_num_steps: int = 50,
        default_cfg_scale: float = 1.0,
    ) -> "SitGarAdapter":
        loader = SitCheckpointLoader(device=device, ifid_repo_root=ifid_repo_root)
        bundle = loader.load_from_group_id(group_id, registry_path=registry_path)
        return cls(
            bundle,
            device=device,
            default_num_steps=default_num_steps,
            default_cfg_scale=default_cfg_scale,
        )

    def _stats_like(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.bundle.latent_stats["latents_scale"].to(device=z.device, dtype=z.dtype)
        bias = self.bundle.latent_stats["latents_bias"].to(device=z.device, dtype=z.dtype)
        if z.ndim == 4:
            return scale.view(1, -1, 1, 1), bias.view(1, -1, 1, 1)
        if z.ndim == 3:
            return scale.view(1, 1, -1), bias.view(1, 1, -1)
        raise ValueError(f"Unsupported latent rank {z.ndim}; expected 3 or 4 dims")

    def normalize_latents(self, z_vae: torch.Tensor) -> torch.Tensor:
        scale, bias = self._stats_like(z_vae)
        return (z_vae - bias) * scale

    def denormalize_latents(self, z_model: torch.Tensor) -> torch.Tensor:
        scale, bias = self._stats_like(z_model)
        return z_model / scale + bias

    def encode_to_vae_latents(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            try:
                return self.vae.encode(x, labels.to(self.device) if labels is not None else labels)
            except TypeError:
                return self.vae.encode(x)

    def encode_to_model_latents(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.normalize_latents(self.encode_to_vae_latents(x, labels=labels))

    def decode_from_vae_latents(
        self,
        z_vae: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_vae = z_vae.to(self.device)
        with torch.no_grad():
            try:
                return self.vae.decode(z_vae, labels.to(self.device) if labels is not None else labels)
            except TypeError:
                return self.vae.decode(z_vae)

    def decode_from_model_latents(
        self,
        z_model: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decode_from_vae_latents(self.denormalize_latents(z_model), labels=labels)

    def raw_timestep_from_eta(self, eta_t: float) -> float:
        eta_t = float(eta_t)
        if not 0.0 <= eta_t <= 1.0:
            raise ValueError(f"Expected eta_t in [0, 1], got {eta_t}")
        tshift = self.tshift
        denom = tshift - (tshift - 1.0) * eta_t
        if denom <= 0.0:
            raise ValueError(
                f"Could not invert shift_time for eta_t={eta_t} with tshift={tshift}"
            )
        return eta_t / denom

    def shifted_timestep_from_raw(self, t_raw: float) -> float:
        t_value = torch.tensor([float(t_raw)], dtype=torch.float64, device=self.device)
        shifted = self.model.shift_time(t_value)
        return float(shifted[0].item())

    def resolve_sampling_plan(
        self,
        noise_level: float,
        num_steps: int,
        sampling_mode: str,
        *,
        step_policy: str = "fixed",
        min_num_steps: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ) -> SamplingPlan:
        normalized_mode = str(sampling_mode).strip().lower()
        if normalized_mode not in {"ode", "sde"}:
            raise ValueError(
                f"Unsupported sampling_mode={sampling_mode!r}; expected 'ode' or 'sde'"
            )
        normalized_policy = str(step_policy).strip().lower()
        if normalized_policy not in {"fixed", "adaptive"}:
            raise ValueError(
                f"Unsupported step_policy={step_policy!r}; expected 'fixed' or 'adaptive'"
            )

        requested_num_steps = int(num_steps or self.default_num_steps)
        if requested_num_steps <= 0:
            raise ValueError(f"Expected num_steps > 0, got {requested_num_steps}")

        t_raw_start = self.raw_timestep_from_eta(noise_level)
        t_shifted_start = self.shifted_timestep_from_raw(t_raw_start)
        time_map_abs_error = abs(t_shifted_start - float(noise_level))
        sde_uses_short_schedule = normalized_mode == "sde" and t_raw_start <= SDE_RAW_MIN_TIME

        if normalized_policy == "fixed":
            resolved_num_steps = requested_num_steps
            min_steps = requested_num_steps
            max_steps = requested_num_steps
        else:
            max_steps = int(max_num_steps if max_num_steps is not None else requested_num_steps)
            if max_steps <= 0:
                raise ValueError(f"Expected max_num_steps > 0, got {max_steps}")
            if min_num_steps is None:
                min_steps = 8 if normalized_mode == "sde" else max(1, max_steps // 4)
            else:
                min_steps = int(min_num_steps)
            if normalized_mode == "sde":
                min_steps = max(2, min_steps)
            else:
                min_steps = max(1, min_steps)
            if min_steps > max_steps:
                raise ValueError(
                    f"Expected min_num_steps <= max_num_steps, got {min_steps} > {max_steps}"
                )
            if min_steps == max_steps:
                resolved_num_steps = max_steps
            else:
                raw_fraction = min(max(t_raw_start, 0.0), 1.0)
                resolved_num_steps = int(round(min_steps + raw_fraction * (max_steps - min_steps)))
                resolved_num_steps = max(min_steps, min(max_steps, resolved_num_steps))

        return SamplingPlan(
            noise_level=float(noise_level),
            sampling_mode=normalized_mode,
            step_policy=normalized_policy,
            requested_num_steps=requested_num_steps,
            resolved_num_steps=resolved_num_steps,
            min_num_steps=min_steps,
            max_num_steps=max_steps,
            t_raw_start=t_raw_start,
            t_shifted_start=t_shifted_start,
            time_map_abs_error=time_map_abs_error,
            sde_uses_short_schedule=sde_uses_short_schedule,
        )

    @staticmethod
    def add_noise(
        z: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise using the effective eta_t seen by SiT after shift_time()."""
        noise = torch.randn_like(z)
        z_noisy = (1.0 - noise_level) * z + noise_level * noise
        return z_noisy, noise

    def add_noise_to_vae_latents(
        self,
        z_vae: torch.Tensor,
        noise_level: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_model = self.normalize_latents(z_vae)
        return self.add_noise(z_model, noise_level)

    def _prepare_labels(
        self,
        batch_size: int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is not None:
            return labels.to(device=device, dtype=torch.long)
        return torch.randint(0, self.bundle.args_dict.get("num_classes", 1000), (batch_size,), device=device)

    def _sample_shape(self, batch_size: int) -> tuple[int, ...]:
        if self.bundle.vae_1d:
            return (batch_size, self.bundle.latent_size, self.bundle.in_channels)
        return (batch_size, self.bundle.in_channels, self.bundle.latent_size, self.bundle.latent_size)

    def _ode_denoise(
        self,
        z_model_noisy: torch.Tensor,
        labels: torch.Tensor,
        *,
        plan: SamplingPlan,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, SamplingPlan]:
        dtype_in = z_model_noisy.dtype
        t_steps = torch.linspace(
            plan.t_raw_start,
            0.0,
            plan.resolved_num_steps + 1,
            dtype=torch.float64,
            device=z_model_noisy.device,
        )
        t_steps = self.model.shift_time(t_steps)
        x_next = z_model_noisy.to(torch.float64)
        y_null = None
        if cfg_scale > 1.0:
            y_null = torch.full_like(labels, self.bundle.args_dict.get("num_classes", 1000))

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = x_next
            if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
                model_input = torch.cat([x_cur, x_cur], dim=0)
                y_cur = torch.cat([labels, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = labels
            time_input = torch.full(
                (model_input.shape[0],),
                fill_value=float(t_cur),
                device=model_input.device,
                dtype=torch.float64,
            )
            d_cur = self.model.inference(
                model_input.to(dtype=dtype_in),
                time_input.to(dtype=dtype_in),
                y_cur,
            ).to(torch.float64)
            if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
                d_cond, d_uncond = d_cur.chunk(2)
                d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur
        return x_next.to(dtype=dtype_in), plan

    def _sde_denoise(
        self,
        z_model_noisy: torch.Tensor,
        labels: torch.Tensor,
        *,
        plan: SamplingPlan,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, SamplingPlan]:
        get_score_from_velocity = self.bundle.ifid_modules["get_score_from_velocity"]
        compute_diffusion = self.bundle.ifid_modules["compute_diffusion"]

        dtype_in = z_model_noisy.dtype
        if plan.sde_uses_short_schedule:
            t_steps = torch.linspace(
                plan.t_raw_start,
                0.0,
                plan.resolved_num_steps + 1,
                dtype=torch.float64,
                device=z_model_noisy.device,
            )
            t_steps = self.model.shift_time(t_steps)
        else:
            t_steps = torch.linspace(
                plan.t_raw_start,
                SDE_RAW_MIN_TIME,
                plan.resolved_num_steps,
                dtype=torch.float64,
                device=z_model_noisy.device,
            )
            t_steps = self.model.shift_time(t_steps)
            t_steps = torch.cat(
                [
                    t_steps,
                    torch.tensor([0.0], dtype=torch.float64, device=z_model_noisy.device),
                ]
            )

        x_next = z_model_noisy.to(torch.float64)
        y_null = None
        if cfg_scale > 1.0:
            y_null = torch.full_like(labels, self.bundle.args_dict.get("num_classes", 1000))

        for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
                model_input = torch.cat([x_cur, x_cur], dim=0)
                y_cur = torch.cat([labels, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = labels

            time_input = torch.full(
                (model_input.shape[0],),
                fill_value=float(t_cur),
                device=model_input.device,
                dtype=torch.float64,
            )
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            v_cur = self.model.inference(
                model_input.to(dtype=dtype_in),
                time_input.to(dtype=dtype_in),
                y_cur,
            ).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type="linear")
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
                d_cond, d_uncond = d_cur.chunk(2)
                d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

        t_cur, t_next = t_steps[-2], t_steps[-1]
        dt = t_next - t_cur
        x_cur = x_next
        if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
            model_input = torch.cat([x_cur, x_cur], dim=0)
            y_cur = torch.cat([labels, y_null], dim=0)
        else:
            model_input = x_cur
            y_cur = labels

        time_input = torch.full(
            (model_input.shape[0],),
            fill_value=float(t_cur),
            device=model_input.device,
            dtype=torch.float64,
        )
        v_cur = self.model.inference(
            model_input.to(dtype=dtype_in),
            time_input.to(dtype=dtype_in),
            y_cur,
        ).to(torch.float64)
        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type="linear")
        diffusion = compute_diffusion(t_cur)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        if cfg_scale > 1.0 and self.guidance_low <= float(t_cur) <= self.guidance_high:
            d_cond, d_uncond = d_cur.chunk(2)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)

        mean_x = x_cur + dt * d_cur
        return mean_x.to(dtype=dtype_in), plan

    def _denoise_impl(
        self,
        z_model_noisy: torch.Tensor,
        noise_level: float,
        labels: torch.Tensor,
        *,
        num_steps: int,
        cfg_scale: float,
        sampling_mode: str,
        step_policy: str = "fixed",
        min_num_steps: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, SamplingPlan]:
        plan = self.resolve_sampling_plan(
            noise_level,
            num_steps,
            sampling_mode,
            step_policy=step_policy,
            min_num_steps=min_num_steps,
            max_num_steps=max_num_steps,
        )
        if plan.sampling_mode == "ode":
            return self._ode_denoise(
                z_model_noisy,
                labels,
                plan=plan,
                cfg_scale=cfg_scale,
            )
        if plan.sampling_mode == "sde":
            return self._sde_denoise(
                z_model_noisy,
                labels,
                plan=plan,
                cfg_scale=cfg_scale,
            )
        raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}; expected 'ode' or 'sde'")

    @torch.no_grad()
    def denoise(
        self,
        z_noisy: torch.Tensor,
        noise_level: float,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        use_cfg: bool = False,
        sampling_mode: str = "ode",
        step_policy: str = "fixed",
        min_num_steps: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        prepared_labels = self._prepare_labels(z_noisy.shape[0], z_noisy.device, labels)
        cfg_scale = self.default_cfg_scale if use_cfg else 1.0
        steps = num_steps or self.default_num_steps
        z_denoised, _ = self._denoise_impl(
            z_model_noisy=z_noisy,
            noise_level=noise_level,
            labels=prepared_labels,
            num_steps=steps,
            cfg_scale=cfg_scale,
            sampling_mode=sampling_mode,
            step_policy=step_policy,
            min_num_steps=min_num_steps,
            max_num_steps=max_num_steps,
        )
        return z_denoised

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        use_cfg: bool = True,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
        sampling_mode: str = "ode",
        step_policy: str = "fixed",
        min_num_steps: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        target_device = torch.device(device)
        z0 = torch.randn(self._sample_shape(batch_size), device=target_device, generator=generator)
        prepared_labels = self._prepare_labels(batch_size, target_device, labels)
        cfg_scale = self.default_cfg_scale if use_cfg else 1.0
        steps = num_steps or self.default_num_steps
        z_generated, _ = self._denoise_impl(
            z_model_noisy=z0,
            noise_level=1.0,
            labels=prepared_labels,
            num_steps=steps,
            cfg_scale=cfg_scale,
            sampling_mode=sampling_mode,
            step_policy=step_policy,
            min_num_steps=min_num_steps,
            max_num_steps=max_num_steps,
        )
        return z_generated

    @torch.no_grad()
    def paired_reconstruct(
        self,
        x: torch.Tensor,
        eta_t: float,
        labels: Optional[torch.Tensor] = None,
        *,
        num_steps: Optional[int] = None,
        use_cfg: bool = False,
        sampling_mode: str = "ode",
        step_policy: str = "fixed",
        min_num_steps: Optional[int] = None,
        max_num_steps: Optional[int] = None,
    ) -> GarDenoiseOutputs:
        normalized_mode = str(sampling_mode).strip().lower()
        z_vae = self.encode_to_vae_latents(x, labels=labels)
        z_model = self.normalize_latents(z_vae)
        z_noisy, _ = self.add_noise(z_model, eta_t)
        prepared_labels = self._prepare_labels(z_noisy.shape[0], z_noisy.device, labels)
        cfg_scale = self.default_cfg_scale if use_cfg else 1.0
        z_denoised_model, plan = self._denoise_impl(
            z_model_noisy=z_noisy,
            noise_level=eta_t,
            labels=prepared_labels,
            num_steps=num_steps or self.default_num_steps,
            cfg_scale=cfg_scale,
            sampling_mode=normalized_mode,
            step_policy=step_policy,
            min_num_steps=min_num_steps,
            max_num_steps=max_num_steps,
        )
        z_denoised_vae = self.denormalize_latents(z_denoised_model)
        x_rec = self.decode_from_vae_latents(z_denoised_vae, labels=labels)
        return GarDenoiseOutputs(
            eta_t=eta_t,
            sampling_mode=normalized_mode,
            t_raw_start=plan.t_raw_start,
            t_shifted_start=plan.t_shifted_start,
            time_map_abs_error=plan.time_map_abs_error,
            requested_num_steps=plan.requested_num_steps,
            resolved_num_steps=plan.resolved_num_steps,
            step_policy=plan.step_policy,
            sde_uses_short_schedule=plan.sde_uses_short_schedule,
            z_vae=z_vae,
            z_model=z_model,
            z_noisy=z_noisy,
            z_denoised_model=z_denoised_model,
            z_denoised_vae=z_denoised_vae,
            x_rec=x_rec,
        )
