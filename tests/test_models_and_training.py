from __future__ import annotations

import numpy as np
import pytest
import torch

from porousdiff.data import apply_compression_safe_symmetry, capture_rng_states, restore_rng_states
from porousdiff.metrics import directional_chord_metrics
from porousdiff.models import (
    ContinuousLoadConditionedUNet,
    EMAModel,
    build_noise_scheduler,
)


def test_zero_snr_schedule_and_ddim_ddpm_stepping_finite() -> None:
    ddpm = build_noise_scheduler(
        sampler="ddpm",
        num_train_timesteps=1000,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    assert float(ddpm.alphas_cumprod[-1].item()) == pytest.approx(0.0, abs=1e-10)
    assert float(ddpm.alphas_cumprod[0].item()) > 0.99

    ddim = build_noise_scheduler(
        sampler="ddim",
        num_train_timesteps=1000,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    ddim.set_timesteps(20)
    x = torch.randn(2, 1, 16, 16, 16)
    for t in ddim.timesteps:
        v_pred = torch.zeros_like(x)
        x, x0 = ddim.step(model_output=v_pred, timestep=int(t), sample=x)
        assert torch.isfinite(x).all()
        assert torch.isfinite(x0).all()


def test_continuous_load_conditioned_unet_all_params_get_gradients_and_cfg_distinct() -> None:
    torch.manual_seed(0)
    model = ContinuousLoadConditionedUNet(
        channels=(16, 32, 64),
        attention_levels=(False, False, True),
        num_res_blocks=1,
        norm_num_groups=8,
        load_min=0.008,
        load_max=0.053,
    )

    # MONAI zero-initializes `out.2.conv` and every `resnet.conv2` (`zero_module`).
    # After 2 optimizer steps, both zero-init gates open and every parameter in the
    # entire UNet + ContinuousScalarLoadEmbedder + attention blocks must get non-zero gradients.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad()
        x = torch.randn(4, 1, 16, 16, 16)
        target = torch.randn_like(x)
        t = torch.tensor([10, 250, 600, 999], dtype=torch.long)
        loads = torch.tensor([0.008, 0.023, 0.0305, 0.053], dtype=torch.float32)
        cond_mask = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float32)

        out = model(x, t, loads, cond_mask=cond_mask)
        assert out.shape == x.shape
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        if step < 2:
            optimizer.step()

    zero_grad_params = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
    ]
    assert zero_grad_params == [], f"Dead parameters with zero gradient: {zero_grad_params}"

    # Verify midpoint load (normalized to 0.0, mask=1) is strictly distinct from unconditional CFG (mask=0)
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 1, 16, 16, 16)
        t1 = torch.tensor([400], dtype=torch.long)
        mid_load = torch.tensor([0.5 * (0.008 + 0.053)], dtype=torch.float32)

        out_mid = model(x1, t1, mid_load, cond_mask=torch.ones(1))
        out_uncond = model(x1, t1, mid_load, cond_mask=torch.zeros(1))
        assert not torch.allclose(out_mid, out_uncond)

        # Smooth interpolation check: tiny load perturbation produces tiny output delta
        out_near = model(x1, t1, mid_load + 1e-4, cond_mask=torch.ones(1))
        out_far = model(x1, t1, torch.tensor([0.053]), cond_mask=torch.ones(1))
        diff_near = float((out_near - out_mid).abs().mean().item())
        diff_far = float((out_far - out_mid).abs().mean().item())
        assert diff_near < diff_far * 0.05

        # CFG scale != 1 runs batched conditional + unconditional forward cleanly
        out_cfg = model.forward_cfg(x1, t1, mid_load, guidance_scale=1.75)
        assert out_cfg.shape == x1.shape
        assert not torch.allclose(out_cfg, out_mid)


def test_ema_and_rng_roundtrip() -> None:
    lin = torch.nn.Linear(8, 4)
    ema = EMAModel(lin, decay=0.99)
    with torch.no_grad():
        lin.weight.add_(5.0)
    ema.update(lin)
    state = ema.state_dict()

    ema2 = EMAModel(torch.nn.Linear(8, 4), decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.num_updates == 1
    assert torch.allclose(
        ema.shadow_model.weight, ema2.shadow_model.weight
    )

    saved_rng = capture_rng_states()
    a1 = torch.randn(5)
    n1 = np.random.rand(5)
    restore_rng_states(saved_rng)
    assert torch.allclose(a1, torch.randn(5))
    assert np.allclose(n1, np.random.rand(5))


def test_compression_safe_symmetry_never_mixes_z_with_xy() -> None:
    # Create an anisotropic volume compressed along Z (thin horizontal sheets every 4 Z slices, long in XY)
    vol = np.zeros((32, 32, 32), dtype=np.uint8)
    vol[2::4, 4:28, 4:28] = 1
    base_metrics = directional_chord_metrics(vol, phase=1)
    assert base_metrics["solid_anisotropy_xy_over_z"] > 5.0

    seen_variants = set()
    for sym_id in range(16):
        aug = apply_compression_safe_symmetry(vol, sym_id)
        assert aug.shape == vol.shape
        m = directional_chord_metrics(aug, phase=1)
        assert m["solid_chord_z"] == pytest.approx(base_metrics["solid_chord_z"])
        assert m["solid_anisotropy_xy_over_z"] == pytest.approx(
            base_metrics["solid_anisotropy_xy_over_z"]
        )
        seen_variants.add(aug.tobytes())
    assert len(seen_variants) >= 2


def test_conditional_model_learns_monotonic_load_interpolation() -> None:
    """Verify end-to-end learning: training on sparse (0.008) & dense (0.053) loads produces
    monotonic generated solid fractions across low, unseen interpolated midpoint, and high loads.
    """
    torch.manual_seed(7)
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=0.008,
        load_max=0.053,
    )
    train_sched = build_noise_scheduler(
        sampler="ddpm",
        num_train_timesteps=200,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    gen = torch.Generator().manual_seed(11)
    x_low = (torch.rand((4, 1, 8, 8, 8), generator=gen) < 0.06).float() * 2.0 - 1.0
    x_high = (torch.rand((4, 1, 8, 8, 8), generator=gen) < 0.62).float() * 2.0 - 1.0
    x_all = torch.cat([x_low, x_high], dim=0)
    loads_all = torch.tensor([0.008] * 4 + [0.053] * 4, dtype=torch.float32)

    for _ in range(60):
        optimizer.zero_grad()
        noise = torch.randn_like(x_all)
        t = torch.randint(0, 200, (8,))
        noisy = train_sched.add_noise(original_samples=x_all, noise=noise, timesteps=t)
        target = train_sched.get_velocity(x_all, noise, t)
        cond_mask = (torch.rand(8) > 0.15).float()
        pred = model(noisy, t, loads_all, cond_mask=cond_mask)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    model.eval()
    ddim = build_noise_scheduler(
        sampler="ddim",
        num_train_timesteps=200,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    ddim.set_timesteps(15)
    init_noise = torch.randn((4, 1, 8, 8, 8), generator=torch.Generator().manual_seed(99))

    generated_sf: dict[float, float] = {}
    with torch.no_grad():
        for load_val in (0.008, 0.0305, 0.053):
            cur = init_noise.clone()
            lv_tensor = torch.full((4,), load_val, dtype=torch.float32)
            for ts in ddim.timesteps:
                t_batch = torch.full((4,), int(ts), dtype=torch.long)
                v_pred = model.forward_cfg(cur, t_batch, lv_tensor, guidance_scale=1.5)
                cur, _ = ddim.step(model_output=v_pred, timestep=int(ts), sample=cur)
            generated_sf[load_val] = float((cur > 0.0).float().mean().item())

    # Strictly monotonic across low -> unseen interpolated midpoint -> high compression load
    assert generated_sf[0.008] < generated_sf[0.0305] < generated_sf[0.053]
    assert generated_sf[0.053] - generated_sf[0.008] > 0.25

