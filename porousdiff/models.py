"""Continuous scalar load-conditioned 3D Diffusion UNet, Zero-SNR schedules, and EMA."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Sequence

import torch
import torch.nn as nn

try:
    from monai.networks.nets import DiffusionModelUNet
    from monai.networks.schedulers import DDIMScheduler, DDPMScheduler
    from monai.networks.schedulers.scheduler import NoiseSchedules
except ImportError:  # pragma: no cover - legacy fallback for monai-generative<=0.2.3
    from generative.networks.nets import DiffusionModelUNet
    from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
    from generative.networks.schedulers.scheduler import NoiseSchedules


def _enforce_zero_terminal_snr(betas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rescale beta schedule to have exact zero terminal SNR (Lin et al., WACV 2024).

    Eliminates residual signal leakage at `t = T - 1` so sampling from zero-mean
    Gaussian noise does not bias strongly porous microstructures (`~5-35%` solid)
    upward toward `50%` solid fraction.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift and rescale so sqrt(alpha_bar) hits 0 at T-1 and preserves value at t=0
    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T
    alphas_bar_sqrt = alphas_bar_sqrt * (
        alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    )

    alphas_bar = alphas_bar_sqrt**2
    alphas_new = torch.empty_like(alphas_bar)
    alphas_new[0] = alphas_bar[0]
    alphas_new[1:] = alphas_bar[1:] / alphas_bar[:-1].clamp(min=1e-12)
    betas_new = (1.0 - alphas_new).clamp(0.0, 1.0)
    return betas_new, alphas_new, alphas_bar


if "zero_snr_scaled_linear" not in NoiseSchedules:

    @NoiseSchedules.add_def(
        "zero_snr_scaled_linear",
        "Zero-terminal-SNR scaled linear schedule for unbiased porous media generation",
    )
    def _zero_snr_scaled_linear(
        num_train_timesteps: int,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )
        return _enforce_zero_terminal_snr(base_betas)


def build_noise_scheduler(
    sampler: str = "ddpm",
    num_train_timesteps: int = 1000,
    schedule: str = "zero_snr_scaled_linear",
    prediction_type: str = "v_prediction",
) -> DDPMScheduler | DDIMScheduler:
    """Construct a MONAI diffusion scheduler configured for unbiased binary volumes."""
    kind = sampler.lower()
    if kind == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule=schedule,
            prediction_type=prediction_type,
            clip_sample=True,
            clip_sample_min=-1.0,
            clip_sample_max=1.0,
        )
    if kind == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule=schedule,
            prediction_type=prediction_type,
            clip_sample=True,
            clip_sample_min=-1.0,
            clip_sample_max=1.0,
            set_alpha_to_one=False,
        )
    raise ValueError(f"Unknown sampler {sampler!r}; choose 'ddpm' or 'ddim'")


class ContinuousScalarLoadEmbedder(nn.Module):
    """Smooth Fourier/sinusoidal scalar conditioner with a learned null token for CFG.

    Maps normalized continuous load `c in [-1, 1]` smoothly into `embed_dim` so
    unseen intermediate compressions interpolate continuously rather than jumping
    between discrete lookup bins.
    """

    def __init__(self, freq_dim: int, embed_dim: int, max_period: float = 16.0) -> None:
        super().__init__()
        if freq_dim % 2 != 0:
            raise ValueError("freq_dim must be even")
        self.freq_dim = freq_dim
        self.embed_dim = embed_dim

        half = freq_dim // 2
        # Frequencies spanning [pi/2, max_period * pi] cover [-1, 1] smoothly without aliasing
        freqs = torch.exp(
            torch.linspace(math.log(0.5 * math.pi), math.log(max_period * math.pi), half)
        )
        self.register_buffer("freqs", freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim + 1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Accepts `packed` of shape `(B, 2)` containing `[cond_norm, cond_mask]`."""
        cond_norm = packed[:, 0:1].float()
        cond_mask = packed[:, 1:2].float()

        angles = cond_norm * self.freqs.unsqueeze(0)
        fourier = torch.cat([cond_norm, torch.sin(angles), torch.cos(angles)], dim=-1)
        cond_emb = self.mlp(fourier.to(self.null_embedding.dtype))
        null_emb = self.null_embedding.expand(cond_emb.shape[0], -1)
        mask = cond_mask.to(cond_emb.dtype)
        return mask * cond_emb + (1.0 - mask) * null_emb


class ContinuousLoadConditionedUNet(nn.Module):
    """3D Diffusion UNet conditioned at every resolution on continuous mechanical load.

    Fixes the 1-token bottleneck cross-attention degeneracy by:
      1. Injecting a smooth Fourier embedding of the continuous load directly into
         `temb` across every encoder, bottleneck, and decoder ResBlock (`96^3, 48^3, 24^3`),
      2. Concatenating spatially broadcast `[cond_norm * mask, mask]` channels at the input
         (`in_channels = 3`) so full-resolution convolutions directly see target load
         and unambiguously distinguish midpoint load (`0.0, mask=1`) from unconditional CFG (`0.0, mask=0`),
      3. Storing `[load_min, load_max]` inside model buffers so checkpoints carry their
         own physical calibration.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        attention_levels: Sequence[bool] = (False, False, True),
        num_res_blocks: int = 2,
        norm_num_groups: int = 16,
        load_min: float = 0.008,
        load_max: float = 0.053,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if load_max <= load_min:
            raise ValueError("load_max must be strictly greater than load_min")

        unet_params = inspect.signature(DiffusionModelUNet.__init__).parameters
        unet_kwargs: dict[str, object] = {
            "spatial_dims": 3,
            "in_channels": 3,
            "out_channels": 1,
            "attention_levels": tuple(attention_levels),
            "num_res_blocks": num_res_blocks,
            "norm_num_groups": norm_num_groups,
            "with_conditioning": False,
            "num_class_embeds": 1,  # Hook replaced below with ContinuousScalarLoadEmbedder
        }
        if "channels" in unet_params:
            unet_kwargs["channels"] = tuple(channels)
        else:
            unet_kwargs["num_channels"] = tuple(channels)

        if use_flash_attention and "use_flash_attention" in unet_params:
            unet_kwargs["use_flash_attention"] = True

        self.unet = DiffusionModelUNet(**unet_kwargs)
        base_ch = int(channels[0])
        time_embed_dim = base_ch * 4
        self.unet.class_embedding = ContinuousScalarLoadEmbedder(
            freq_dim=max(32, base_ch),
            embed_dim=time_embed_dim,
        )

        self.register_buffer("load_min", torch.tensor(float(load_min), dtype=torch.float32))
        self.register_buffer("load_max", torch.tensor(float(load_max), dtype=torch.float32))

    def normalize_load(self, load_values: torch.Tensor) -> torch.Tensor:
        """Linearly map physical load values into `[-1, 1]` (allowing smooth interpolation/extrapolation)."""
        span = (self.load_max - self.load_min).clamp(min=1e-8)
        return 2.0 * (load_values.float() - self.load_min) / span - 1.0

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        load_values: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity `v` (or `epsilon`) for noisy 3D volume `x` `[B, 1, D, H, W]`."""
        b, _, d, h, w = x.shape
        loads = load_values.view(b).to(device=x.device, dtype=torch.float32)
        cond_norm = self.normalize_load(loads)

        if cond_mask is None:
            mask = torch.ones(b, device=x.device, dtype=torch.float32)
        else:
            mask = cond_mask.view(b).to(device=x.device, dtype=torch.float32)

        masked_cond = (cond_norm * mask).to(dtype=x.dtype)
        mask_ch = mask.to(dtype=x.dtype)

        cond_vol = masked_cond.view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
        mask_vol = mask_ch.view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
        unet_in = torch.cat([x, cond_vol, mask_vol], dim=1)

        packed_labels = torch.stack([cond_norm, mask], dim=-1)
        return self.unet(x=unet_in, timesteps=timesteps, class_labels=packed_labels)

    def forward_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        load_values: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Classifier-free guided prediction combining conditional and unconditional passes."""
        b = x.shape[0]
        if abs(guidance_scale - 1.0) < 1e-6:
            ones = torch.ones(b, device=x.device, dtype=torch.float32)
            return self.forward(x, timesteps, load_values, cond_mask=ones)

        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([timesteps, timesteps], dim=0)
        loads_cat = torch.cat([load_values.view(b), load_values.view(b)], dim=0)
        mask_cat = torch.cat(
            [
                torch.ones(b, device=x.device, dtype=torch.float32),
                torch.zeros(b, device=x.device, dtype=torch.float32),
            ],
            dim=0,
        )
        pred_both = self.forward(x_cat, t_cat, loads_cat, cond_mask=mask_cat)
        pred_cond, pred_uncond = pred_both.chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


class EMAModel:
    """Exponential Moving Average of model parameters with early-step warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.num_updates = 0
        self.shadow_model = copy.deepcopy(model).eval()
        for p in self.shadow_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        eff_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for s_param, param in zip(self.shadow_model.parameters(), model.parameters()):
            s_param.lerp_(param.detach().to(s_param.dtype), 1.0 - eff_decay)
        for s_buf, buf in zip(self.shadow_model.buffers(), model.buffers()):
            s_buf.copy_(buf.detach())

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_state_dict": self.shadow_model.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])  # type: ignore[arg-type]
        self.num_updates = int(state["num_updates"])  # type: ignore[arg-type]
        self.shadow_model.load_state_dict(state["shadow_state_dict"])  # type: ignore[arg-type]
