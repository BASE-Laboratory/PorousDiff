"""Multi-electrode, multi-phase, continuous stress & resolution conditioned 3D Diffusion UNet."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Mapping, Sequence

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
    """Construct a MONAI diffusion scheduler configured for unbiased porous volumes."""
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
    """Smooth Fourier/sinusoidal scalar conditioner with a learned null token for CFG."""

    def __init__(self, freq_dim: int, embed_dim: int, max_period: float = 16.0) -> None:
        super().__init__()
        if freq_dim % 2 != 0:
            raise ValueError("freq_dim must be even")
        self.freq_dim = freq_dim
        self.embed_dim = embed_dim

        half = freq_dim // 2
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


class MultiModalConditionEmbedder(nn.Module):
    """Joint embedder for continuous compressive stress/load, electrode chemistry ID,
    and physical voxel size (`log(dx / dx_ref)`), with learned CFG null embedding."""

    def __init__(
        self,
        num_materials: int,
        freq_dim: int,
        embed_dim: int,
        max_period: float = 16.0,
        cfg_mode: str = "all",
    ) -> None:
        super().__init__()
        if freq_dim % 2 != 0:
            raise ValueError("freq_dim must be even")
        if num_materials < 1:
            raise ValueError("num_materials must be >= 1")
        if cfg_mode not in ("all", "load_only"):
            raise ValueError(f"cfg_mode must be 'all' or 'load_only', got {cfg_mode!r}")

        self.num_materials = num_materials
        self.freq_dim = freq_dim
        self.embed_dim = embed_dim
        self.cfg_mode = cfg_mode

        half = freq_dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(0.5 * math.pi), math.log(max_period * math.pi), half)
        )
        self.register_buffer("freqs", freqs, persistent=False)

        self.material_embedding = nn.Embedding(num_materials, embed_dim)
        nn.init.normal_(self.material_embedding.weight, std=0.02)

        if cfg_mode == "load_only":
            self.scalar_mlp: nn.Module | None = None
            self.load_mlp: nn.Module | None = nn.Sequential(
                nn.Linear(freq_dim + 1, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.vox_mlp: nn.Module | None = nn.Sequential(
                nn.Linear(freq_dim + 1, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            # Concatenate Fourier(load) [freq_dim + 1] + Fourier(log_voxel) [freq_dim + 1]
            self.scalar_mlp = nn.Sequential(
                nn.Linear(2 * (freq_dim + 1), embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.load_mlp = None
            self.vox_mlp = None

        self.fusion_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(1, embed_dim))

    def _fourier_encode(self, val: torch.Tensor) -> torch.Tensor:
        angles = val * self.freqs.unsqueeze(0)
        return torch.cat([val, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        """Accepts `packed` of shape `(B, 4)`: `[load_norm, mat_id, log_vox, cond_mask]`."""
        load_norm = packed[:, 0:1].float()
        mat_ids = packed[:, 1].round().long().clamp(0, self.num_materials - 1)
        log_vox = packed[:, 2:3].float()
        cond_mask = packed[:, 3:4].float()

        dtype = self.null_embedding.dtype
        mask = cond_mask.to(dtype)
        if self.cfg_mode == "load_only":
            assert self.load_mlp is not None and self.vox_mlp is not None
            load_feat = self.load_mlp(self._fourier_encode(load_norm).to(dtype))
            null_load = self.null_embedding.expand(load_feat.shape[0], -1)
            eff_load = mask * load_feat + (1.0 - mask) * null_load
            ctx_feat = self.vox_mlp(self._fourier_encode(log_vox).to(dtype)) + self.material_embedding(mat_ids)
            return self.fusion_mlp(eff_load + ctx_feat)

        assert self.scalar_mlp is not None
        scalar_feat = torch.cat(
            [self._fourier_encode(load_norm), self._fourier_encode(log_vox)], dim=-1
        ).to(dtype)
        fused = self.fusion_mlp(self.scalar_mlp(scalar_feat) + self.material_embedding(mat_ids))
        null_emb = self.null_embedding.expand(fused.shape[0], -1)
        return mask * fused + (1.0 - mask) * null_emb


class ContinuousLoadConditionedUNet(nn.Module):
    """3D Diffusion UNet conditioned on continuous compression load, electrode material,
    and physical voxel resolution across 2-phase (`1 ch`) or 3-phase (`2 ch`) microstructures.

    Defaults (`num_phases=2, num_materials=1`) preserve exact backward compatibility with
    single-material binary checkpoints while `num_phases=3` or `num_materials>1` activates
    hierarchical multi-phase + multi-electrode conditioning.
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
        conditioning_mode: str = "multiscale_fourier",
        num_phases: int = 2,
        num_materials: int = 1,
        ref_voxel_size_um: float = 1.0,
        use_ordinal_material_channel: bool = False,
        cfg_mode: str = "all",
    ) -> None:
        super().__init__()
        if load_max <= load_min:
            raise ValueError("load_max must be strictly greater than load_min")
        if conditioning_mode not in ("multiscale_fourier", "legacy_crossattn"):
            raise ValueError(
                f"Unknown conditioning_mode {conditioning_mode!r}; "
                "choose 'multiscale_fourier' or 'legacy_crossattn'"
            )
        if num_phases not in (2, 3):
            raise ValueError(f"num_phases must be 2 or 3, got {num_phases}")
        if num_materials < 1:
            raise ValueError(f"num_materials must be >= 1, got {num_materials}")
        if ref_voxel_size_um <= 0:
            raise ValueError("ref_voxel_size_um must be positive")
        if cfg_mode not in ("all", "load_only"):
            raise ValueError(f"cfg_mode must be 'all' or 'load_only', got {cfg_mode!r}")

        self.conditioning_mode = conditioning_mode
        self.num_phases = num_phases
        self.num_materials = num_materials
        self.data_channels = 1 if num_phases == 2 else 2
        self.is_multimodal = (num_phases == 3) or (num_materials > 1)
        self.use_ordinal_material_channel = bool(use_ordinal_material_channel)
        self.cfg_mode = cfg_mode

        unet_params = inspect.signature(DiffusionModelUNet.__init__).parameters
        if conditioning_mode == "multiscale_fourier":
            aux_channels = (4 if self.use_ordinal_material_channel else 3) if self.is_multimodal else 2
            unet_kwargs: dict[str, object] = {
                "spatial_dims": 3,
                "in_channels": self.data_channels + aux_channels,
                "out_channels": self.data_channels,
                "attention_levels": tuple(attention_levels),
                "num_res_blocks": num_res_blocks,
                "norm_num_groups": norm_num_groups,
                "with_conditioning": False,
                "num_class_embeds": 1,
            }
        else:
            unet_kwargs = {
                "spatial_dims": 3,
                "in_channels": self.data_channels,
                "out_channels": self.data_channels,
                "attention_levels": tuple(attention_levels),
                "num_res_blocks": num_res_blocks,
                "norm_num_groups": norm_num_groups,
                "with_conditioning": True,
                "cross_attention_dim": 32,
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
        if conditioning_mode == "multiscale_fourier":
            if self.is_multimodal:
                self.unet.class_embedding = MultiModalConditionEmbedder(
                    num_materials=num_materials,
                    freq_dim=max(32, base_ch),
                    embed_dim=time_embed_dim,
                    cfg_mode=cfg_mode,
                )
            else:
                self.unet.class_embedding = ContinuousScalarLoadEmbedder(
                    freq_dim=max(32, base_ch),
                    embed_dim=time_embed_dim,
                )
            self.legacy_embedder: nn.Module | None = None
        else:
            legacy_in_dim = 3 if self.is_multimodal else 1
            self.legacy_embedder = nn.Sequential(
                nn.Linear(legacy_in_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
            )

        self.register_buffer("load_min", torch.tensor(float(load_min), dtype=torch.float32))
        self.register_buffer("load_max", torch.tensor(float(load_max), dtype=torch.float32))
        self.register_buffer(
            "ref_voxel_size_um", torch.tensor(float(ref_voxel_size_um), dtype=torch.float32)
        )

    def normalize_load(self, load_values: torch.Tensor) -> torch.Tensor:
        """Linearly map physical load/stress values into `[-1, 1]`."""
        span = (self.load_max - self.load_min).clamp(min=1e-8)
        return 2.0 * (load_values.float() - self.load_min) / span - 1.0

    def normalize_voxel_size(self, voxel_sizes: torch.Tensor) -> torch.Tensor:
        """Map physical voxel size into log-ratio `log(dx / dx_ref)`."""
        safe = voxel_sizes.float().clamp(min=1e-4)
        return torch.log(safe / self.ref_voxel_size_um.clamp(min=1e-4))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        load_values: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
        material_ids: torch.Tensor | None = None,
        voxel_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity `v` (or `epsilon`) for noisy 3D volume `x` `[B, C, D, H, W]`."""
        b, _, d, h, w = x.shape
        loads = load_values.view(b).to(device=x.device, dtype=torch.float32)
        cond_norm = self.normalize_load(loads)

        if cond_mask is None:
            mask = torch.ones(b, device=x.device, dtype=torch.float32)
        else:
            mask = cond_mask.view(b).to(device=x.device, dtype=torch.float32)

        if not self.is_multimodal:
            if self.conditioning_mode == "legacy_crossattn":
                assert self.legacy_embedder is not None
                masked_scalar = (cond_norm * mask).unsqueeze(-1)
                ctx = self.legacy_embedder(masked_scalar).unsqueeze(1).to(dtype=x.dtype)
                return self.unet(x=x, timesteps=timesteps, context=ctx)

            masked_cond = (cond_norm * mask).to(dtype=x.dtype)
            mask_ch = mask.to(dtype=x.dtype)
            cond_vol = masked_cond.view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
            mask_vol = mask_ch.view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
            unet_in = torch.cat([x, cond_vol, mask_vol], dim=1)
            packed_labels = torch.stack([cond_norm, mask], dim=-1)
            return self.unet(x=unet_in, timesteps=timesteps, class_labels=packed_labels)

        # Multi-modal / 3-phase forward path: material chemistry conditioned purely via
        # learned categorical embedding in MultiModalConditionEmbedder unless legacy flag set.
        if material_ids is None:
            mats = torch.zeros(b, device=x.device, dtype=torch.long)
        else:
            mats = material_ids.view(b).to(device=x.device, dtype=torch.long)

        if voxel_sizes is None:
            vox = self.ref_voxel_size_um.expand(b).to(device=x.device, dtype=torch.float32)
        else:
            vox = voxel_sizes.view(b).to(device=x.device, dtype=torch.float32)
        log_vox = self.normalize_voxel_size(vox)

        mat_span = max(1, self.num_materials - 1)
        mat_norm = 2.0 * (mats.float() / float(mat_span)) - 1.0 if self.num_materials > 1 else torch.zeros_like(cond_norm)

        ctx_mask = torch.ones_like(mask) if self.cfg_mode == "load_only" else mask
        if self.conditioning_mode == "legacy_crossattn":
            assert self.legacy_embedder is not None
            vec = torch.stack([cond_norm * mask, mat_norm * ctx_mask, log_vox * ctx_mask], dim=-1)
            ctx = self.legacy_embedder(vec).unsqueeze(1).to(dtype=x.dtype)
            return self.unet(x=x, timesteps=timesteps, context=ctx)

        c_load = (cond_norm * mask).to(dtype=x.dtype).view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
        c_vox = (log_vox * ctx_mask).to(dtype=x.dtype).view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
        c_msk = mask.to(dtype=x.dtype).view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
        if self.use_ordinal_material_channel:
            c_mat = (mat_norm * ctx_mask).to(dtype=x.dtype).view(b, 1, 1, 1, 1).expand(b, 1, d, h, w)
            unet_in = torch.cat([x, c_load, c_mat, c_vox, c_msk], dim=1)
        else:
            unet_in = torch.cat([x, c_load, c_vox, c_msk], dim=1)

        packed_mm = torch.stack([cond_norm, mats.float(), log_vox, mask], dim=-1)
        return self.unet(x=unet_in, timesteps=timesteps, class_labels=packed_mm)

    def forward_cfg(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        load_values: torch.Tensor,
        guidance_scale: float = 1.0,
        material_ids: torch.Tensor | None = None,
        voxel_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Classifier-free guided prediction combining conditional and unconditional passes."""
        b = x.shape[0]
        if abs(guidance_scale - 1.0) < 1e-6:
            ones = torch.ones(b, device=x.device, dtype=torch.float32)
            return self.forward(
                x,
                timesteps,
                load_values,
                cond_mask=ones,
                material_ids=material_ids,
                voxel_sizes=voxel_sizes,
            )

        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([timesteps, timesteps], dim=0)
        loads_cat = torch.cat([load_values.view(b), load_values.view(b)], dim=0)
        mats_cat = (
            torch.cat([material_ids.view(b), material_ids.view(b)], dim=0)
            if material_ids is not None
            else None
        )
        vox_cat = (
            torch.cat([voxel_sizes.view(b), voxel_sizes.view(b)], dim=0)
            if voxel_sizes is not None
            else None
        )
        mask_cat = torch.cat(
            [
                torch.ones(b, device=x.device, dtype=torch.float32),
                torch.zeros(b, device=x.device, dtype=torch.float32),
            ],
            dim=0,
        )
        pred_both = self.forward(
            x_cat,
            t_cat,
            loads_cat,
            cond_mask=mask_cat,
            material_ids=mats_cat,
            voxel_sizes=vox_cat,
        )
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


def infer_use_ordinal_material_channel(saved_payload: Mapping[str, object]) -> bool:
    """Resolve `use_ordinal_material_channel` from checkpoint config or state_dict input channels."""
    cfg = saved_payload.get("config", {})
    if isinstance(cfg, Mapping) and "use_ordinal_material_channel" in cfg:
        return bool(cfg["use_ordinal_material_channel"])

    num_phases = int(cfg.get("num_phases", 2)) if isinstance(cfg, Mapping) else 2  # type: ignore[arg-type]
    data_ch = 1 if num_phases == 2 else 2
    model_sd = saved_payload.get("model", {})
    if isinstance(model_sd, Mapping):
        for k, v in model_sd.items():
            if str(k).endswith("conv_in.conv.weight") and isinstance(v, torch.Tensor) and v.ndim == 5:
                return int(v.shape[1]) == data_ch + 4
    return False

