"""Physics-informed 3D diffusion training objective & coating delamination diagnostics.

Provides:
  1. Closed-form single-step clean phase reconstruction `x0_hat` from `v_prediction` or `epsilon`
     on Zero-Terminal-SNR schedules without numerical blow-up at `t = T - 1`.
  2. `PhysicsInformedDiffusionLoss`: combines Min-SNR-gamma timestep-weighted diffusion MSE with
     signal-weighted 3D interfacial gradient sharpness, hierarchical no-floating-coating support
     enforcement (`ReLU(c_coating - c_solid)`), and solid-phase catalyst-to-scaffold loading
     ratio conservation (`phi_coating / phi_solid`).
  3. `compute_coating_retention_diagnostics`: checks experimental XCT states across increasing
     compressive stress to quantify catalyst retention vs. mechanical spallation/delamination.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruct_x0(
    x_t: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    prediction_type: str = "v_prediction",
    clamp_range: tuple[float, float] | None = (-1.5, 1.5),
) -> torch.Tensor:
    """Closed-form clean sample estimate `x0_hat` from noisy `x_t` and network prediction."""
    ab = alphas_cumprod.to(device=x_t.device, dtype=torch.float32)[timesteps]
    while ab.ndim < x_t.ndim:
        ab = ab.unsqueeze(-1)

    sqrt_ab = ab.clamp(min=0.0).sqrt()
    sqrt_one_minus_ab = (1.0 - ab).clamp(min=0.0).sqrt()

    x_f = x_t.float()
    p_f = model_pred.float()

    if prediction_type == "v_prediction":
        x0_hat = sqrt_ab * x_f - sqrt_one_minus_ab * p_f
    elif prediction_type == "epsilon":
        x0_hat = (x_f - sqrt_one_minus_ab * p_f) / sqrt_ab.clamp(min=1e-3)
    else:
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}")

    if clamp_range is not None:
        x0_hat = x0_hat.clamp(clamp_range[0], clamp_range[1])
    return x0_hat


def min_snr_weights(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    snr_gamma: float = 5.0,
    prediction_type: str = "v_prediction",
    zero_snr_floor: float = 0.05,
) -> torch.Tensor:
    """Min-SNR-gamma timestep weights (Hang et al., ICCV 2023) adapted for Zero-Terminal-SNR."""
    if snr_gamma <= 0.0:
        return torch.ones_like(timesteps, dtype=torch.float32)

    ab = alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    snr = ab / (1.0 - ab).clamp(min=1e-6)
    clipped = torch.clamp(snr, max=float(snr_gamma))

    if prediction_type == "v_prediction":
        weights = clipped / (snr + 1.0)
        return weights.clamp(min=float(zero_snr_floor))
    if prediction_type == "epsilon":
        return (clipped / snr.clamp(min=1e-4)).clamp(min=float(zero_snr_floor))
    raise ValueError(f"Unsupported prediction_type={prediction_type!r}")


def spatial_gradient_l1(
    x0_hat: torch.Tensor,
    x0_true: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """3D finite-difference interfacial sharpness loss along `Z`, `Y`, and `X`."""
    dz = (
        (x0_hat[:, :, 1:, :, :] - x0_hat[:, :, :-1, :, :])
        - (x0_true[:, :, 1:, :, :] - x0_true[:, :, :-1, :, :])
    ).abs().mean(dim=(1, 2, 3, 4))
    dy = (
        (x0_hat[:, :, :, 1:, :] - x0_hat[:, :, :, :-1, :])
        - (x0_true[:, :, :, 1:, :] - x0_true[:, :, :, :-1, :])
    ).abs().mean(dim=(1, 2, 3, 4))
    dx = (
        (x0_hat[:, :, :, :, 1:] - x0_hat[:, :, :, :, :-1])
        - (x0_true[:, :, :, :, 1:] - x0_true[:, :, :, :, :-1])
    ).abs().mean(dim=(1, 2, 3, 4))
    grad_err = (dz + dy + dx) / 3.0
    return (sample_weights * grad_err).mean()


def hierarchical_support_penalty(
    x0_hat: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize physically impossible floating catalyst coating (`c_coating > c_solid`)."""
    if x0_hat.shape[1] < 2:
        return torch.zeros((), device=x0_hat.device, dtype=torch.float32)
    violation = F.relu(x0_hat[:, 1] - x0_hat[:, 0]).mean(dim=(1, 2, 3))
    return (sample_weights * violation).mean()


def catalyst_loading_conservation_loss(
    x0_hat: torch.Tensor,
    x0_true: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """Conserve catalyst-to-total-solid volume ratio (`phi_coating / phi_solid`) per patch.

    Handles mixed 2-phase uncoated (`coat_true == 0`) + 3-phase coated batches cleanly without
    dividing rectified `x0_hat` noise by small `solid_hat` on uncoated Virgin patches.
    """
    if x0_hat.shape[1] < 2:
        return torch.zeros((), device=x0_hat.device, dtype=torch.float32)

    solid_hat = ((x0_hat[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=(1, 2, 3))
    coat_hat = ((x0_hat[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=(1, 2, 3))
    solid_true = ((x0_true[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=(1, 2, 3))
    coat_true = ((x0_true[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5).mean(dim=(1, 2, 3))

    is_coated = (coat_true > 1e-4).to(dtype=torch.float32)
    ratio_hat = coat_hat / (solid_hat + 1e-3)
    ratio_true = coat_true / (solid_true + 1e-3)
    coated_err = is_coated * (ratio_hat - ratio_true).pow(2)
    uncoated_err = (1.0 - is_coated) * F.relu(x0_hat[:, 1] + 0.95).mean(dim=(1, 2, 3)).pow(2)
    return (sample_weights * (coated_err + uncoated_err)).mean()


class PhysicsInformedDiffusionLoss(nn.Module):
    """Composite physics-informed 3D porous-electrode diffusion training criterion."""

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        prediction_type: str = "v_prediction",
        snr_gamma: float = 5.0,
        interface_weight: float = 0.05,
        support_weight: float = 0.05,
        loading_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod.detach().clone().to(dtype=torch.float32),
            persistent=False,
        )
        self.prediction_type = prediction_type
        self.snr_gamma = float(snr_gamma)
        self.interface_weight = float(interface_weight)
        self.support_weight = float(support_weight)
        self.loading_weight = float(loading_weight)

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total physics-informed diffusion loss and detached diagnostic components."""
        pred_f = model_pred.float()
        targ_f = target.float()
        x0_f = x_0.float()

        per_sample_mse = (pred_f - targ_f).pow(2).mean(dim=(1, 2, 3, 4))
        w_snr = min_snr_weights(
            self.alphas_cumprod,
            timesteps,
            snr_gamma=self.snr_gamma,
            prediction_type=self.prediction_type,
        )
        loss_diff = (w_snr * per_sample_mse).mean()

        ab_weights = self.alphas_cumprod.to(device=x_t.device)[timesteps].clamp(min=0.05)
        cond_weights = (
            ab_weights * cond_mask.view(-1).to(device=ab_weights.device, dtype=torch.float32)
            if cond_mask is not None
            else ab_weights
        )
        x0_hat = reconstruct_x0(
            x_t=x_t,
            model_pred=pred_f,
            timesteps=timesteps,
            alphas_cumprod=self.alphas_cumprod,
            prediction_type=self.prediction_type,
        )

        loss_iface = (
            spatial_gradient_l1(x0_hat, x0_f, cond_weights)
            if self.interface_weight > 0.0
            else torch.zeros_like(loss_diff)
        )
        loss_support = (
            hierarchical_support_penalty(x0_hat, ab_weights)
            if self.support_weight > 0.0
            else torch.zeros_like(loss_diff)
        )
        loss_loading = (
            catalyst_loading_conservation_loss(x0_hat, x0_f, cond_weights)
            if self.loading_weight > 0.0
            else torch.zeros_like(loss_diff)
        )

        total = (
            loss_diff
            + self.interface_weight * loss_iface
            + self.support_weight * loss_support
            + self.loading_weight * loss_loading
        )
        stats = {
            "loss_total": float(total.detach().item()),
            "loss_diffusion": float(loss_diff.detach().item()),
            "loss_interface": float(loss_iface.detach().item()),
            "loss_support": float(loss_support.detach().item()),
            "loss_loading": float(loss_loading.detach().item()),
        }
        return total, stats


def compute_coating_retention_diagnostics(
    benchmark_report: Mapping[str, object],
    spallation_drop_threshold: float = 0.15,
) -> dict[str, dict[str, object]]:
    """Quantify experimental catalyst-to-carbon mass retention vs. delamination across stress.

    Groups real XCT states by `(material, beamline)`, normalizes each state's
    `coating_to_carbon_ratio` by the lowest-load reference scan, and flags potential
    high-load coating spallation if the ratio drops by more than `spallation_drop_threshold`.
    """
    grouped: dict[tuple[str, str], list[tuple[float, str, float]]] = {}
    for st, raw_entry in benchmark_report.items():
        entry = raw_entry  # type: ignore[assignment]
        mat = str(entry.get("material", "virgin_carbon"))  # type: ignore[union-attr]
        beam = str(entry.get("beamline", "primary"))  # type: ignore[union-attr]
        lv = float(entry.get("load_value", 0.0))  # type: ignore[union-attr]
        real_metrics = entry.get("real", {}).get("metrics", {})  # type: ignore[union-attr]
        if "coating_to_carbon_ratio" not in real_metrics:
            continue
        ratio = float(real_metrics["coating_to_carbon_ratio"]["mean"])
        grouped.setdefault((mat, beam), []).append((lv, str(st), ratio))

    diagnostics: dict[str, dict[str, object]] = {}
    for (mat, beam), items in grouped.items():
        items.sort(key=lambda t: t[0])
        ref_lv, ref_st, ref_ratio = items[0]
        group_key = f"{mat}__{beam}"
        if ref_ratio <= 1e-6:
            diagnostics[group_key] = {
                "material": mat,
                "beamline": beam,
                "reference_state": ref_st,
                "reference_load": ref_lv,
                "is_coated": False,
                "states": {},
            }
            continue

        state_rows: dict[str, object] = {}
        min_ret = 1.0
        for lv, st, ratio in items:
            ret = float(ratio / ref_ratio) if np.isfinite(ratio) else float("nan")
            if np.isfinite(ret):
                min_ret = min(min_ret, ret)
            state_rows[st] = {
                "load_value": lv,
                "coating_to_carbon_ratio": round(ratio, 6),
                "retention_vs_uncompressed": round(ret, 6) if np.isfinite(ret) else float("nan"),
                "possible_spallation": bool(np.isfinite(ret) and ret < (1.0 - spallation_drop_threshold)),
            }

        diagnostics[group_key] = {
            "material": mat,
            "beamline": beam,
            "reference_state": ref_st,
            "reference_load": ref_lv,
            "reference_coating_to_carbon_ratio": round(ref_ratio, 6),
            "min_retention_across_loads": round(min_ret, 6),
            "any_spallation_detected": bool(min_ret < (1.0 - spallation_drop_threshold)),
            "is_coated": True,
            "states": state_rows,
        }
    return diagnostics
