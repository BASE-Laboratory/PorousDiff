"""Virtual in-situ compression via DDIM ODE inversion & seamless >96^3 3D MultiDiffusion.

Implements two zero-extra-beamtime inference capabilities on top of trained PorousDiff UNets:
  1. `virtual_compress_volumes`: Takes a real segmented 3D scan (e.g. benchtop XCT at `0 kPa`
     or lowest synchrotron load `sigma_0`), deterministically inverts it along the conditional
     DDIM probability-flow ODE at `source_load` (safely stopping before Zero-SNR `alpha_bar = 0`),
     and denoises it back to `t = 0` conditioned on any `target_load` `sigma`.
  2. `generate_tiled_3d`: Synthesizes arbitrarily large, seam-free 3D electrode volumes
     (e.g. `96 x 96 x 192` or `160^3`) for OpenImpala REV convergence using overlapped
     3D raised-cosine (Hann) weighted MultiDiffusion without increasing per-step GPU memory.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from porousdiff.baselines import (
    affine_z_compress_to_target,
    compute_mirror_tiled_fraction,
    resolve_compression_ratio,
)
from porousdiff.data import decode_channels_to_phases, encode_phases_to_channels
from porousdiff.models import ContinuousLoadConditionedUNet


def _pred_to_x0_and_eps(
    x_t: torch.Tensor,
    pred: torch.Tensor,
    alpha_bar: float,
    prediction_type: str,
    clip_sample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert `(x_t, pred)` at scalar `alpha_bar` into consistent `(x0_hat, eps_eff)`."""
    ab = float(min(max(alpha_bar, 1e-6), 1.0 - 1e-6))
    sqrt_ab = math.sqrt(ab)
    sqrt_omab = math.sqrt(1.0 - ab)

    x_f = x_t.float()
    p_f = pred.float()

    if prediction_type == "v_prediction":
        x0_hat = sqrt_ab * x_f - sqrt_omab * p_f
    elif prediction_type == "epsilon":
        x0_hat = (x_f - sqrt_omab * p_f) / sqrt_ab
    else:
        raise ValueError(f"Unsupported prediction_type={prediction_type!r}")

    if clip_sample:
        x0_hat = x0_hat.clamp(-1.0, 1.0)
    eps_eff = (x_f - sqrt_ab * x0_hat) / sqrt_omab
    return x0_hat, eps_eff


def _build_safe_ddim_schedule(
    alphas_cumprod: torch.Tensor,
    num_steps: int,
    invert_strength: float,
    min_alpha_bar: float = 1e-4,
) -> list[int]:
    """Select ascending DDIM timesteps `[t_0, ..., t_K]` strictly avoiding `alpha_bar == 0`."""
    if not 0.05 <= invert_strength <= 1.0:
        raise ValueError(f"invert_strength must be in [0.05, 1.0], got {invert_strength}")
    ab_np = alphas_cumprod.detach().cpu().numpy()
    valid_indices = np.flatnonzero(ab_np >= min_alpha_bar)
    if valid_indices.size < 2:
        raise ValueError("Noise schedule has fewer than 2 timesteps with alpha_bar >= min_alpha_bar")

    max_safe_t = int(valid_indices[-1])
    raw_grid = np.linspace(0, max_safe_t, max(2, int(num_steps)), dtype=np.int64)
    unique_asc = sorted({int(t) for t in raw_grid})
    if len(unique_asc) < 2:
        unique_asc = [0, max_safe_t]

    stop_idx = max(1, min(len(unique_asc) - 1, int(round(invert_strength * (len(unique_asc) - 1)))))
    return unique_asc[: stop_idx + 1]


@torch.inference_mode()
def ddim_invert_latent(
    model: ContinuousLoadConditionedUNet,
    x0: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    source_load: float,
    prediction_type: str = "v_prediction",
    num_steps: int = 50,
    invert_strength: float = 0.60,
    guidance_scale_inv: float = 1.0,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, list[int]]:
    """Deterministic forward DDIM ODE inversion mapping clean `x0` `[B, C, D, H, W]` -> `x_{t_K}`."""
    device = x0.device
    bsz = x0.shape[0]
    ab_cpu = alphas_cumprod.detach().cpu().float()
    asc_timesteps = _build_safe_ddim_schedule(
        ab_cpu, num_steps=num_steps, invert_strength=invert_strength
    )

    loads = torch.full((bsz,), float(source_load), device=device, dtype=torch.float32)
    mats = (
        torch.full((bsz,), int(material_id), device=device, dtype=torch.long)
        if model.is_multimodal
        else None
    )
    voxs = (
        torch.full((bsz,), float(voxel_size_um), device=device, dtype=torch.float32)
        if model.is_multimodal
        else None
    )

    # Lift clean x0 onto the first schedule timestep t_0
    t0 = asc_timesteps[0]
    ab0 = float(ab_cpu[t0].item())
    cur = math.sqrt(ab0) * x0.float()

    for idx in range(len(asc_timesteps) - 1):
        t_cur = asc_timesteps[idx]
        t_next = asc_timesteps[idx + 1]
        t_batch = torch.full((bsz,), t_cur, device=device, dtype=torch.long)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model.forward_cfg(
                cur,
                t_batch,
                loads,
                guidance_scale=guidance_scale_inv,
                material_ids=mats,
                voxel_sizes=voxs,
            )

        # Do NOT clamp x0_hat during forward ODE inversion: on discrete {-1, +1} targets,
        # clamping zeroes out eps_eff whenever |x0_hat| > 1 and collapses latent norm.
        x0_hat, eps_eff = _pred_to_x0_and_eps(
            cur,
            pred,
            float(ab_cpu[t_cur].item()),
            prediction_type=prediction_type,
            clip_sample=False,
        )
        ab_next = float(ab_cpu[t_next].item())
        cur = math.sqrt(ab_next) * x0_hat + math.sqrt(max(1.0 - ab_next, 0.0)) * eps_eff

    return cur, asc_timesteps


@torch.inference_mode()
def ddim_denoise_from_inverted(
    model: ContinuousLoadConditionedUNet,
    x_inv: torch.Tensor,
    asc_timesteps: list[int],
    alphas_cumprod: torch.Tensor,
    target_load: float,
    prediction_type: str = "v_prediction",
    guidance_scale: float = 1.5,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic reverse DDIM ODE denoising from inverted `x_inv` to `target_load`."""
    device = x_inv.device
    bsz = x_inv.shape[0]
    ab_cpu = alphas_cumprod.detach().cpu().float()

    loads = torch.full((bsz,), float(target_load), device=device, dtype=torch.float32)
    mats = (
        torch.full((bsz,), int(material_id), device=device, dtype=torch.long)
        if model.is_multimodal
        else None
    )
    voxs = (
        torch.full((bsz,), float(voxel_size_um), device=device, dtype=torch.float32)
        if model.is_multimodal
        else None
    )

    cur = x_inv.clone().float()
    for idx in range(len(asc_timesteps) - 1, 0, -1):
        t_cur = asc_timesteps[idx]
        t_prev = asc_timesteps[idx - 1]
        t_batch = torch.full((bsz,), t_cur, device=device, dtype=torch.long)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model.forward_cfg(
                cur,
                t_batch,
                loads,
                guidance_scale=guidance_scale,
                material_ids=mats,
                voxel_sizes=voxs,
            )

        x0_hat, eps_eff = _pred_to_x0_and_eps(
            cur, pred, float(ab_cpu[t_cur].item()), prediction_type=prediction_type
        )
        ab_prev = float(ab_cpu[t_prev].item())
        cur = math.sqrt(ab_prev) * x0_hat + math.sqrt(max(1.0 - ab_prev, 0.0)) * eps_eff

    # Final projection at t_0 -> clean x0
    t0 = asc_timesteps[0]
    t0_batch = torch.full((bsz,), t0, device=device, dtype=torch.long)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        pred0 = model.forward_cfg(
            cur,
            t0_batch,
            loads,
            guidance_scale=guidance_scale,
            material_ids=mats,
            voxel_sizes=voxs,
        )
    x0_final, _ = _pred_to_x0_and_eps(
        cur, pred0, float(ab_cpu[t0].item()), prediction_type=prediction_type
    )

    raw_np = x0_final.cpu().numpy()
    phases = decode_channels_to_phases(raw_np, enforce_solid_support=True, threshold=0.0)
    probs = np.clip((raw_np[:, 0] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return phases, probs


@torch.inference_mode()
def virtual_compress_volumes(
    model: ContinuousLoadConditionedUNet,
    alphas_cumprod: torch.Tensor,
    source_volumes: np.ndarray,
    source_load: float,
    target_loads: list[float] | tuple[float, ...],
    prediction_type: str = "v_prediction",
    num_steps: int = 50,
    invert_strength: float = 0.60,
    guidance_scale: float = 1.5,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    batch_size: int = 4,
    interface_smooth_sigma: float = 0.25,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> dict[float, np.ndarray]:
    """Transfer real low-load 3D XCT patches `[N, D, H, W]` (or `[D, H, W]`) to `target_loads`.

    Inverts each batch of `source_volumes` ONCE at `source_load` (`invert_strength=0.60`
    retains macroscopic in-plane XY fibre architecture while allowing Z pore collapse) and
    branches the reverse DDIM trajectory across `target_loads`. Note that fixed-cube windows
    represent statistical / mesoscale local microstructure conditioned on input latent structure,
    not Lagrangian material-point tracking.
    """
    vols = np.asarray(source_volumes, dtype=np.uint8)
    single_input = vols.ndim == 3
    if single_input:
        vols = vols[np.newaxis, ...]
    if vols.ndim != 4:
        raise ValueError(f"Expected 3D [D,H,W] or 4D [N,D,H,W] source_volumes, got {vols.shape}")

    device = next(model.parameters()).device
    num_phases = model.num_phases
    per_load_batches: dict[float, list[np.ndarray]] = {float(tl): [] for tl in target_loads}

    for start in range(0, vols.shape[0], max(1, batch_size)):
        chunk = vols[start : start + max(1, batch_size)]
        encoded_list: list[torch.Tensor] = []
        for v in chunk:
            enc = encode_phases_to_channels(v, num_phases=num_phases)
            if interface_smooth_sigma > 0.0:
                enc_np = enc.numpy()
                for ch_i in range(enc_np.shape[0]):
                    enc_np[ch_i] = np.clip(
                        gaussian_filter(enc_np[ch_i], sigma=interface_smooth_sigma), -1.0, 1.0
                    )
                enc = torch.from_numpy(enc_np)
            encoded_list.append(enc)

        x0_batch = torch.stack(encoded_list, dim=0).to(device=device, dtype=torch.float32)
        x_inv, asc_steps = ddim_invert_latent(
            model=model,
            x0=x0_batch,
            alphas_cumprod=alphas_cumprod,
            source_load=source_load,
            prediction_type=prediction_type,
            num_steps=num_steps,
            invert_strength=invert_strength,
            guidance_scale_inv=1.0,
            material_id=material_id,
            voxel_size_um=voxel_size_um,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        for tl in target_loads:
            tl_f = float(tl)
            eff_cfg = 1.0 if abs(tl_f - float(source_load)) < 1e-8 else guidance_scale
            phases_out, _ = ddim_denoise_from_inverted(
                model=model,
                x_inv=x_inv,
                asc_timesteps=asc_steps,
                alphas_cumprod=alphas_cumprod,
                target_load=tl_f,
                prediction_type=prediction_type,
                guidance_scale=eff_cfg,
                material_id=material_id,
                voxel_size_um=voxel_size_um,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            per_load_batches[tl_f].append(phases_out)

    results: dict[float, np.ndarray] = {}
    for tl_f, chunks in per_load_batches.items():
        cat = np.concatenate(chunks, axis=0)
        results[tl_f] = cat[0] if single_input else cat
    return results


def check_load_extrapolation(
    loads: list[float] | tuple[float, ...],
    load_min: float,
    load_max: float,
    tol: float = 1e-6,
) -> dict[str, object]:
    """Identify whether any requested source or target load extrapolates outside `[load_min, load_max]`."""
    out_of_range = [
        float(lv)
        for lv in loads
        if float(lv) < float(load_min) - tol or float(lv) > float(load_max) + tol
    ]
    return {
        "load_min": float(load_min),
        "load_max": float(load_max),
        "is_extrapolated": len(out_of_range) > 0,
        "extrapolated_loads": out_of_range,
    }


def compute_virtual_compression_fidelity(
    source_volumes: np.ndarray,
    compressed_volumes: np.ndarray,
    same_load_reconstructions: np.ndarray | None = None,
    is_target_same_as_source: bool = False,
) -> dict[str, float | bool]:
    """Quantify structural retention, actual deformation, and near-identity copy risk in DDIM virtual compression.

    Computes:
      - `same_load_solid_dice`: Solid-phase Dice overlap (`2|A&B| / (|A|+|B|)`) for `sigma_0 -> sigma_0`
        roundtrip reconstruction (immune to high-porosity accuracy inflation).
      - `paired_xy_column_corr`: Mean Pearson correlation between each input scan's Z-averaged
        XY solid density `mean_Z(source_i >= 1)` and its own compressed output `mean_Z(comp_i >= 1)`.
      - `unpaired_xy_column_corr`: Baseline correlation against unrelated shifted inputs `source_{(i+1)%N}`.
      - `xy_correlation_lift`: `paired_xy_column_corr - unpaired_xy_column_corr` (`> 0` confirms
        specimen-specific structural memory rather than unconditional resampling).
      - `mean_delta_solid_fraction`: `mean(comp >= 1) - mean(source >= 1)` verifying non-trivial compaction.
      - `changed_voxel_fraction_vs_source`: `mean(comp != source)` detecting no-op / identity-copy behaviour.
      - `solid_dice_vs_source`: Solid-phase Dice directly between `source` and `compressed` output.
      - `is_near_identity_copy`: `True` when `not is_target_same_as_source` yet fewer than `1%` of voxels
        changed and `|mean_delta_solid_fraction| < 0.005` (flags under-trained / weak-guidance no-op inversion).
    """
    src = np.asarray(source_volumes)
    cmp = np.asarray(compressed_volumes)
    if src.ndim == 3:
        src = src[np.newaxis, ...]
    if cmp.ndim == 3:
        cmp = cmp[np.newaxis, ...]

    def _solid_dice_batch(a_arr: np.ndarray, b_arr: np.ndarray) -> float:
        dices: list[float] = []
        for a_v, b_v in zip(a_arr, b_arr):
            a_b = a_v >= 1
            b_b = b_v >= 1
            denom = int(a_b.sum()) + int(b_b.sum())
            dices.append(1.0 if denom == 0 else 2.0 * float(np.count_nonzero(a_b & b_b)) / float(denom))
        return float(np.mean(dices)) if dices else float("nan")

    same_dice = float("nan")
    if same_load_reconstructions is not None:
        rec = np.asarray(same_load_reconstructions)
        if rec.ndim == 3:
            rec = rec[np.newaxis, ...]
        if rec.shape[1:] == src.shape[1:]:
            same_dice = _solid_dice_batch(src, rec)
        elif rec.shape[2:] == src.shape[2:] and rec.shape[1] > 0:
            z_idx_r = np.linspace(0, src.shape[1] - 1, rec.shape[1]).round().astype(int)
            same_dice = _solid_dice_batch(src[:, z_idx_r], rec)

    def _corr2d(a: np.ndarray, b: np.ndarray) -> float:
        af = a.ravel().astype(np.float64)
        bf = b.ravel().astype(np.float64)
        sa = float(np.std(af))
        sb = float(np.std(bf))
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.corrcoef(af, bf)[0, 1])

    n = min(src.shape[0], cmp.shape[0])
    paired_corrs: list[float] = []
    unpaired_corrs: list[float] = []
    for i in range(n):
        src_xy = (src[i] >= 1).mean(axis=0)
        cmp_xy = (cmp[i] >= 1).mean(axis=0)
        paired_corrs.append(_corr2d(src_xy, cmp_xy))
        if n >= 2:
            other_xy = (src[(i + 1) % n] >= 1).mean(axis=0)
            unpaired_corrs.append(_corr2d(other_xy, cmp_xy))

    paired_mean = float(np.mean(paired_corrs)) if paired_corrs else float("nan")
    unpaired_mean = float(np.mean(unpaired_corrs)) if unpaired_corrs else float("nan")
    lift = (
        paired_mean - unpaired_mean
        if (np.isfinite(paired_mean) and np.isfinite(unpaired_mean))
        else float("nan")
    )
    src_n = src[:n]
    cmp_n = cmp[:n]
    src_sf = float((src_n >= 1).mean()) if n > 0 else float("nan")
    cmp_sf = float((cmp_n >= 1).mean()) if n > 0 else float("nan")
    delta_sf = cmp_sf - src_sf if (np.isfinite(src_sf) and np.isfinite(cmp_sf)) else float("nan")
    shape_changed = bool(n > 0 and cmp_n.shape[1:] != src_n.shape[1:])
    if n == 0:
        changed_frac = float("nan")
        dice_vs_src = float("nan")
    elif not shape_changed:
        changed_frac = float((cmp_n != src_n).mean())
        dice_vs_src = _solid_dice_batch(src_n, cmp_n)
    elif cmp_n.shape[2:] == src_n.shape[2:] and cmp_n.shape[1] > 0:
        z_idx = np.linspace(0, src_n.shape[1] - 1, cmp_n.shape[1]).round().astype(int)
        src_z_reg = src_n[:, z_idx]
        changed_frac = float((cmp_n != src_z_reg).mean())
        dice_vs_src = _solid_dice_batch(src_z_reg, cmp_n)
    else:
        changed_frac = 1.0
        dice_vs_src = float("nan")

    near_identity = bool(
        (not is_target_same_as_source)
        and (not shape_changed)
        and np.isfinite(changed_frac)
        and changed_frac < 0.01
        and abs(delta_sf) < 0.005
    )
    return {
        "same_load_solid_dice": same_dice,
        "paired_xy_column_corr": paired_mean,
        "unpaired_xy_column_corr": unpaired_mean,
        "xy_correlation_lift": lift,
        "source_solid_fraction_mean": src_sf,
        "compressed_solid_fraction_mean": cmp_sf,
        "mean_delta_solid_fraction": delta_sf,
        "changed_voxel_fraction_vs_source": changed_frac,
        "solid_dice_vs_source": dice_vs_src,
        "is_near_identity_copy": near_identity,
    }


def _compute_tile_starts(full_len: int, patch_size: int, overlap: float) -> list[int]:
    if full_len < patch_size:
        raise ValueError(f"Target dimension {full_len} must be >= patch_size {patch_size}")
    if full_len == patch_size:
        return [0]
    max_stride = max(1, int(round(patch_size * (1.0 - overlap))))
    span = full_len - patch_size
    num_intervals = max(1, math.ceil(span / float(max_stride)))
    raw_starts = [int(round(i * span / float(num_intervals))) for i in range(num_intervals + 1)]
    starts: list[int] = []
    for s in raw_starts:
        if not starts or s > starts[-1]:
            starts.append(s)
    if starts[-1] != span:
        starts.append(span)
    return starts


def _build_3d_tile_weight(
    patch_size: int,
    z0: int,
    y0: int,
    x0: int,
    full_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Smooth 3D raised-cosine window with flat outer-border support for MultiDiffusion."""
    idx = np.arange(patch_size, dtype=np.float32)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * (idx + 0.5) / float(patch_size)))
    hann = np.clip(hann, 1e-3, 1.0)
    half = patch_size // 2

    def _axis_w(start: int, total: int) -> np.ndarray:
        if total == patch_size:
            return np.ones(patch_size, dtype=np.float32)
        w = hann.copy()
        if start == 0:
            w[:half] = 1.0
        if start + patch_size == total:
            w[half:] = 1.0
        return w

    wz = _axis_w(z0, full_shape[0])
    wy = _axis_w(y0, full_shape[1])
    wx = _axis_w(x0, full_shape[2])
    w3d = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return torch.from_numpy(w3d).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _eval_multidiffusion_tiles(
    model: ContinuousLoadConditionedUNet,
    image: torch.Tensor,
    coords: list[tuple[int, int, int]],
    tile_weights: list[torch.Tensor],
    weight_norm: torch.Tensor,
    patch_size: int,
    t_int: int,
    load_value: float,
    guidance_scale: float,
    tile_batch_size: int,
    material_id: int,
    voxel_size_um: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluate overlapped 3D Hann-weighted UNet predictions across arbitrary `(D, H, W)` slabs."""
    device = image.device
    batch_size = image.shape[0]
    pred_accum = torch.zeros_like(image, dtype=torch.float32)
    flat_tiles: list[tuple[int, int, int, int, torch.Tensor, torch.Tensor]] = []
    for b_idx in range(batch_size):
        for (z0, y0, x0), tw in zip(coords, tile_weights):
            crop = image[
                b_idx : b_idx + 1,
                :,
                z0 : z0 + patch_size,
                y0 : y0 + patch_size,
                x0 : x0 + patch_size,
            ]
            flat_tiles.append((b_idx, z0, y0, x0, crop, tw))

    for chunk_start in range(0, len(flat_tiles), max(1, tile_batch_size)):
        sub = flat_tiles[chunk_start : chunk_start + max(1, tile_batch_size)]
        sub_b = len(sub)
        x_sub = torch.cat([item[4] for item in sub], dim=0)
        t_sub = torch.full((sub_b,), t_int, device=device, dtype=torch.long)
        l_sub = torch.full((sub_b,), float(load_value), device=device, dtype=torch.float32)
        m_sub = (
            torch.full((sub_b,), int(material_id), device=device, dtype=torch.long)
            if model.is_multimodal
            else None
        )
        v_sub = (
            torch.full((sub_b,), float(voxel_size_um), device=device, dtype=torch.float32)
            if model.is_multimodal
            else None
        )

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred_sub = model.forward_cfg(
                x_sub,
                t_sub,
                l_sub,
                guidance_scale=guidance_scale,
                material_ids=m_sub,
                voxel_sizes=v_sub,
            ).float()

        for item_i, (b_idx, z0, y0, x0, _, tw) in enumerate(sub):
            pred_accum[
                b_idx : b_idx + 1,
                :,
                z0 : z0 + patch_size,
                y0 : y0 + patch_size,
                x0 : x0 + patch_size,
            ] += pred_sub[item_i : item_i + 1] * tw

    return pred_accum / weight_norm


@torch.inference_mode()
def generate_tiled_3d(
    model: ContinuousLoadConditionedUNet,
    scheduler: object,
    target_shape: tuple[int, int, int],
    patch_size: int,
    load_value: float,
    guidance_scale: float = 1.5,
    overlap: float = 0.5,
    batch_size: int = 1,
    tile_batch_size: int = 4,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Seamless 3D MultiDiffusion synthesizing `target_shape = (D, H, W)` >= `patch_size^3`."""
    if not 0.1 <= overlap <= 0.85:
        raise ValueError(f"overlap must be in [0.1, 0.85], got {overlap}")
    d_full, h_full, w_full = (int(s) for s in target_shape)
    device = next(model.parameters()).device
    data_ch = model.data_channels

    z_starts = _compute_tile_starts(d_full, patch_size, overlap)
    y_starts = _compute_tile_starts(h_full, patch_size, overlap)
    x_starts = _compute_tile_starts(w_full, patch_size, overlap)
    coords = [(z0, y0, x0) for z0 in z_starts for y0 in y_starts for x0 in x_starts]

    tile_weights = [
        _build_3d_tile_weight(patch_size, z0, y0, x0, (d_full, h_full, w_full), device)
        for (z0, y0, x0) in coords
    ]
    weight_norm = torch.zeros((1, 1, d_full, h_full, w_full), device=device, dtype=torch.float32)
    for (z0, y0, x0), tw in zip(coords, tile_weights):
        weight_norm[:, :, z0 : z0 + patch_size, y0 : y0 + patch_size, x0 : x0 + patch_size] += tw
    weight_norm = weight_norm.clamp(min=1e-6)

    image = torch.randn(
        (batch_size, data_ch, d_full, h_full, w_full),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    for timestep in scheduler.timesteps:  # type: ignore[attr-defined]
        t_int = int(timestep)
        blended_pred = _eval_multidiffusion_tiles(
            model=model,
            image=image,
            coords=coords,
            tile_weights=tile_weights,
            weight_norm=weight_norm,
            patch_size=patch_size,
            t_int=t_int,
            load_value=load_value,
            guidance_scale=guidance_scale,
            tile_batch_size=tile_batch_size,
            material_id=material_id,
            voxel_size_um=voxel_size_um,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        next_img = scheduler.step(blended_pred, t_int, image, generator=generator)[0]  # type: ignore[attr-defined]
        image = torch.nan_to_num(next_img, nan=0.0, posinf=1.0, neginf=-1.0)

    raw_np = image.float().cpu().numpy()
    phases = decode_channels_to_phases(raw_np, enforce_solid_support=True, threshold=0.0)
    probs = np.clip((raw_np[:, 0] + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return phases, probs


@torch.inference_mode()
def virtual_compress_lagrangian_slab(
    model: ContinuousLoadConditionedUNet,
    alphas_cumprod: torch.Tensor,
    source_volume: np.ndarray,
    target_load: float,
    patch_size: int,
    target_solid_fraction: float | None = None,
    compression_ratio: float | None = None,
    output_depth: int | None = None,
    prediction_type: str = "v_prediction",
    num_steps: int = 40,
    refine_strength: float = 0.35,
    overlap: float = 0.5,
    guidance_scale: float = 1.5,
    tile_batch_size: int = 4,
    material_id: int = 0,
    voxel_size_um: float = 1.0,
    interface_smooth_sigma: float = 0.25,
    conserve_solid_mass: bool = True,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mass-conserving Lagrangian slab virtual compression via affine Z-prewarping + MultiDiffusion DDIM.

    Maps a tall uncompressed specimen slab `(Z_0, Y, X)` (`Z_0 >= patch_size`) to physical
    compressed height `Z_comp = max(patch_size, round(Z_0 * ratio))`, then runs moderate-noise
    (`refine_strength = 0.35`) 3D Hann-tiled MultiDiffusion DDIM denoising conditioned on
    `target_load` to restore circular carbon-fiber cross-sections and realistic `0-1-2` catalyst
    pinch junctions while preserving exact macroscopic solid/coating mass conservation.
    """
    vol = np.asarray(source_volume, dtype=np.uint8)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D source_volume (Z_0, Y, X), got shape {vol.shape}")
    if vol.shape[1] < patch_size or vol.shape[2] < patch_size:
        raise ValueError(
            f"In-plane dimensions {vol.shape[1:]} must be >= patch_size ({patch_size})"
        )

    phi_0 = float((vol >= 1).mean())
    ratio = resolve_compression_ratio(
        phi_0, target_solid_fraction=target_solid_fraction, compression_ratio=compression_ratio
    )
    ideal_d = int(round(float(vol.shape[0]) * ratio))
    out_d = (
        max(patch_size, int(output_depth))
        if output_depth is not None
        else max(patch_size, ideal_d)
    )
    mirror_frac = compute_mirror_tiled_fraction(
        source_depth=int(vol.shape[0]),
        output_depth=out_d,
        initial_solid_fraction=phi_0,
        target_solid_fraction=target_solid_fraction,
        compression_ratio=ratio if target_solid_fraction is None else None,
    )
    prewarped = affine_z_compress_to_target(
        vol,
        target_solid_fraction=target_solid_fraction,
        compression_ratio=ratio if target_solid_fraction is None else None,
        output_depth=out_d,
    )

    device = next(model.parameters()).device
    num_phases = model.num_phases
    enc = encode_phases_to_channels(prewarped, num_phases=num_phases)
    if interface_smooth_sigma > 0.0:
        enc_np = enc.numpy()
        for ch_i in range(enc_np.shape[0]):
            enc_np[ch_i] = np.clip(
                gaussian_filter(enc_np[ch_i], sigma=interface_smooth_sigma), -1.0, 1.0
            )
        enc = torch.from_numpy(enc_np)
    x0_clean = enc.unsqueeze(0).to(device=device, dtype=torch.float32)

    ab_cpu = alphas_cumprod.detach().cpu().float()
    asc_steps = _build_safe_ddim_schedule(
        ab_cpu, num_steps=num_steps, invert_strength=refine_strength
    )

    d_full, h_full, w_full = prewarped.shape
    z_starts = _compute_tile_starts(d_full, patch_size, overlap)
    y_starts = _compute_tile_starts(h_full, patch_size, overlap)
    x_starts = _compute_tile_starts(w_full, patch_size, overlap)
    coords = [(z0, y0, x0) for z0 in z_starts for y0 in y_starts for x0 in x_starts]
    tile_weights = [
        _build_3d_tile_weight(patch_size, z0, y0, x0, (d_full, h_full, w_full), device)
        for (z0, y0, x0) in coords
    ]
    weight_norm = torch.zeros((1, 1, d_full, h_full, w_full), device=device, dtype=torch.float32)
    for (z0, y0, x0), tw in zip(coords, tile_weights):
        weight_norm[:, :, z0 : z0 + patch_size, y0 : y0 + patch_size, x0 : x0 + patch_size] += tw
    weight_norm = weight_norm.clamp(min=1e-6)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    t_top = asc_steps[-1]
    ab_top = float(ab_cpu[t_top].item())
    eps = torch.randn(x0_clean.shape, generator=gen, device=device, dtype=torch.float32)
    cur = math.sqrt(ab_top) * x0_clean + math.sqrt(max(1.0 - ab_top, 0.0)) * eps

    for idx in range(len(asc_steps) - 1, 0, -1):
        t_cur = asc_steps[idx]
        t_prev = asc_steps[idx - 1]
        blended_pred = _eval_multidiffusion_tiles(
            model=model,
            image=cur,
            coords=coords,
            tile_weights=tile_weights,
            weight_norm=weight_norm,
            patch_size=patch_size,
            t_int=t_cur,
            load_value=target_load,
            guidance_scale=guidance_scale,
            tile_batch_size=tile_batch_size,
            material_id=material_id,
            voxel_size_um=voxel_size_um,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        x0_hat, eps_eff = _pred_to_x0_and_eps(
            cur, blended_pred, float(ab_cpu[t_cur].item()), prediction_type=prediction_type
        )
        ab_prev = float(ab_cpu[t_prev].item())
        cur = math.sqrt(ab_prev) * x0_hat + math.sqrt(max(1.0 - ab_prev, 0.0)) * eps_eff

    t0 = asc_steps[0]
    pred0 = _eval_multidiffusion_tiles(
        model=model,
        image=cur,
        coords=coords,
        tile_weights=tile_weights,
        weight_norm=weight_norm,
        patch_size=patch_size,
        t_int=t0,
        load_value=target_load,
        guidance_scale=guidance_scale,
        tile_batch_size=tile_batch_size,
        material_id=material_id,
        voxel_size_um=voxel_size_um,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    x0_final, _ = _pred_to_x0_and_eps(
        cur, pred0, float(ab_cpu[t0].item()), prediction_type=prediction_type
    )
    raw_np = x0_final.cpu().numpy()

    if conserve_solid_mass:
        n_total = prewarped.size
        expected_phi = (
            min(0.95, float(target_solid_fraction))
            if target_solid_fraction is not None
            else min(0.95, phi_0 / max(ratio, 1e-6))
        )
        n_solid_target = min(n_total, int(round(expected_phi * float(n_total))))
        solid_field = raw_np[0, 0] + 0.05 * x0_clean[0, 0].cpu().numpy()
        out_vol = np.zeros(prewarped.shape, dtype=np.uint8)
        if 0 < n_solid_target < n_total:
            flat_idx = np.argpartition(solid_field.ravel(), -n_solid_target)[-n_solid_target:]
            solid_mask = np.zeros(n_total, dtype=bool)
            solid_mask[flat_idx] = True
            solid_mask_3d = solid_mask.reshape(prewarped.shape)
            out_vol[solid_mask_3d] = 1
            if num_phases == 3 and raw_np.shape[1] >= 2:
                coat_phi_0 = float((vol == 2).mean())
                expected_coat_phi = min(
                    expected_phi,
                    coat_phi_0 * (expected_phi / max(phi_0, 1e-8)) if phi_0 > 0.0 else 0.0,
                )
                n_coat_target = min(n_solid_target, int(round(expected_coat_phi * float(n_total))))
                if n_coat_target > 0:
                    coat_scores = (
                        raw_np[0, 1][solid_mask_3d]
                        + 0.05 * x0_clean[0, 1].cpu().numpy()[solid_mask_3d]
                    )
                    coat_sub = np.argpartition(coat_scores, -n_coat_target)[-n_coat_target:]
                    solid_coords = np.flatnonzero(solid_mask)
                    out_vol.ravel()[solid_coords[coat_sub]] = 2
        elif n_solid_target >= n_total:
            out_vol[...] = 1
    else:
        out_vol = decode_channels_to_phases(raw_np, enforce_solid_support=True, threshold=0.0)[0]

    src_mass = phi_0 * float(vol.shape[0])
    cmp_sf = float((out_vol >= 1).mean())
    cmp_mass = cmp_sf * float(out_vol.shape[0])
    rel_mass_err = abs(cmp_mass - src_mass) / max(src_mass, 1e-8) if mirror_frac == 0.0 else 0.0
    info = {
        "source_depth": float(vol.shape[0]),
        "compressed_depth": float(out_vol.shape[0]),
        "compression_ratio": float(ratio),
        "strain_z": float(1.0 - float(out_vol.shape[0]) / max(float(vol.shape[0]), 1.0)),
        "mirror_tiled_fraction": float(mirror_frac),
        "source_solid_fraction": phi_0,
        "compressed_solid_fraction": cmp_sf,
        "mass_conservation_rel_error": float(rel_mass_err),
    }
    return out_vol, info
