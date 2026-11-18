"""Unit and end-to-end CLI tests for the 5 v0.4 scientific upgrades:
  1. Deterministic DDIM ODE virtual compression of real uncompressed 3D scans,
  2. Seamless >patch_size^3 Hann-weighted 3D MultiDiffusion volume stitching for OpenImpala REV,
  3. Physics-informed x0_hat diffusion loss + coating delamination diagnostics,
  4. Optimal cell clamping-pressure (sigma*) discovery coupled with OpenImpala transport,
  5. 3D compressive pinch-point & coating coordination metrics wired into K11 DIAD XRD correlation.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from porousdiff.data import encode_phases_to_channels
from porousdiff.losses import (
    PhysicsInformedDiffusionLoss,
    compute_coating_retention_diagnostics,
    hierarchical_support_penalty,
    reconstruct_x0,
)
from porousdiff.metrics import evaluate_multiphase_physics, summarize_metric_records
from porousdiff.models import ContinuousLoadConditionedUNet, build_noise_scheduler
from porousdiff.optimization import (
    compute_transport_and_fom_record,
    find_optimal_clamping_pressure,
    load_openimpala_lookup,
)
from porousdiff.transport_and_xrd import correlate_xrd_with_microstructure
from porousdiff.virtual_compression import generate_tiled_3d, virtual_compress_volumes

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def test_pinch_point_and_solid_tortuosity_metrics(tmp_path: Path) -> None:
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    # Through-plane percolating carbon column at (y=8, x=8)
    vol[:, 8, 8] = 1
    # Sandwiched coating pinched between z=4 and z=6 carbon voxels
    vol[5, 8, 8] = 2
    # Non-sandwiched side coating attached laterally to carbon at (z=10, y=8, x=9)
    vol[10, 8, 9] = 2

    m = evaluate_multiphase_physics(vol, voxel_size=1.5)
    assert m["z_pinch_contact_density"] > 0.0
    assert abs(m["coating_pinch_fraction"] - 0.5) < 1e-6
    assert m["mean_coating_coordination"] >= 1.5
    assert np.isfinite(m["geodesic_solid_tortuosity_z"])
    assert abs(m["geodesic_solid_tortuosity_z"] - 1.0) < 1e-6

    # Verify K11 DIAD XRD correlation automatically correlates pinch-point & solid tortuosity
    report: dict[str, object] = {}
    xrd_csv = tmp_path / "diad_pinch.csv"
    with xrd_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["state", "fwhm_deg"])
        writer.writeheader()
        for idx, pinch_extra in enumerate([0, 1, 2, 3]):
            st = f"cuhcf_load_{idx}"
            v_i = vol.copy()
            for k in range(pinch_extra):
                z_pos = 2 + 2 * k
                v_i[z_pos - 1 : z_pos + 2, 4, 4] = 1
                v_i[z_pos, 4, 4] = 2
            rec = evaluate_multiphase_physics(v_i, voxel_size=1.5)
            summ = summarize_metric_records([rec])
            report[st] = {
                "material": "cuhcf_cathode",
                "load_value": float(50 * (idx + 1)),
                "real": {"metrics": summ},
                "methods": {"porousdiff": {"metrics": summ}},
            }
            writer.writerow({"state": st, "fwhm_deg": round(0.12 + 0.03 * idx, 4)})

    corr_out = correlate_xrd_with_microstructure(
        xrd_csv_path=xrd_csv,
        benchmark_report=report,
        out_dir=tmp_path / "xrd_out",
    )
    fwhm_corrs = corr_out["correlations"]["fwhm_deg"]
    assert "real_coating_pinch_fraction" in fwhm_corrs
    assert "real_z_pinch_contact_density" in fwhm_corrs
    assert fwhm_corrs["real_coating_pinch_fraction"]["pearson_r"] > 0.90
    assert (tmp_path / "xrd_out" / "diad_xrd_microstructure_correlation.png").is_file()


def test_physics_informed_loss_and_delamination_diagnostics() -> None:
    sched = build_noise_scheduler(
        sampler="ddpm",
        num_train_timesteps=100,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    # Verify exact x0 reconstruction even at Zero-Terminal-SNR endpoint t = T - 1 (alpha_bar == 0)
    x0 = torch.randn(2, 2, 8, 8, 8).clamp(-1.0, 1.0)
    noise = torch.randn_like(x0)
    t_ext = torch.tensor([0, 99], dtype=torch.long)
    x_t = sched.add_noise(original_samples=x0, noise=noise, timesteps=t_ext)
    v_true = sched.get_velocity(x0, noise, t_ext)
    x0_rec = reconstruct_x0(
        x_t=x_t,
        model_pred=v_true,
        timesteps=t_ext,
        alphas_cumprod=sched.alphas_cumprod,
        prediction_type="v_prediction",
        clamp_range=None,
    )
    assert torch.allclose(x0_rec, x0, atol=1e-4)

    # Hierarchical support penalty detects floating coating (c_coating > c_solid)
    w = torch.ones(2, dtype=torch.float32)
    valid_hier = torch.stack([torch.ones(2, 4, 4, 4), -torch.ones(2, 4, 4, 4)], dim=1)
    floating_hier = torch.stack([-torch.ones(2, 4, 4, 4), torch.ones(2, 4, 4, 4)], dim=1)
    assert float(hierarchical_support_penalty(valid_hier, w).item()) == 0.0
    assert float(hierarchical_support_penalty(floating_hier, w).item()) > 1.5

    # Full PhysicsInformedDiffusionLoss backward pass with 3-phase UNet
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=10.0,
        load_max=400.0,
        num_phases=3,
        num_materials=3,
    )
    criterion = PhysicsInformedDiffusionLoss(
        alphas_cumprod=sched.alphas_cumprod,
        prediction_type="v_prediction",
    )
    loads = torch.tensor([25.0, 250.0])
    mats = torch.tensor([1, 2])
    voxs = torch.tensor([1.6, 1.6])
    pred = model(x_t, t_ext, loads, material_ids=mats, voxel_sizes=voxs)
    loss, stats = criterion(pred, v_true, x_t, x0, t_ext)
    loss.backward()
    assert np.isfinite(stats["loss_total"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)

    # Coating delamination diagnostics
    mock_report = {
        "tio2_low": {
            "material": "tio2_anode",
            "beamline": "DLS_I12",
            "load_value": 20.0,
            "real": {"metrics": {"coating_to_carbon_ratio": {"mean": 0.40}}},
        },
        "tio2_high": {
            "material": "tio2_anode",
            "beamline": "DLS_I12",
            "load_value": 400.0,
            "real": {"metrics": {"coating_to_carbon_ratio": {"mean": 0.28}}},
        },
    }
    diag = compute_coating_retention_diagnostics(mock_report, spallation_drop_threshold=0.15)
    grp = diag["tio2_anode__DLS_I12"]
    assert grp["any_spallation_detected"] is True
    assert grp["states"]["tio2_high"]["possible_spallation"] is True  # type: ignore[index]


def test_ddim_virtual_compression_trained_fidelity_and_target_shift() -> None:
    from porousdiff.virtual_compression import compute_virtual_compression_fidelity

    torch.manual_seed(17)
    np.random.seed(17)

    # Construct two distinct specimens (A: top-left fibre columns; B: bottom-right fibre columns)
    # at low load (50 kPa, ~11% solid) and compacted high load (300 kPa, ~36% solid)
    vol_a_low = np.zeros((12, 12, 12), dtype=np.uint8)
    vol_a_low[:, 1:5, 1:5] = 1
    vol_b_low = np.zeros((12, 12, 12), dtype=np.uint8)
    vol_b_low[:, 7:11, 7:11] = 1

    vol_a_high = vol_a_low.copy()
    vol_a_high[2:10, 0:7, 0:7] = 1
    vol_b_high = vol_b_low.copy()
    vol_b_high[2:10, 5:12, 5:12] = 1

    train_vols = np.stack([vol_a_low, vol_b_low, vol_a_high, vol_b_high], axis=0)
    x0_batch = torch.stack([encode_phases_to_channels(v, num_phases=2) for v in train_vols], dim=0)
    loads_batch = torch.tensor([50.0, 50.0, 300.0, 300.0], dtype=torch.float32)

    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=50.0,
        load_max=300.0,
        num_phases=2,
        num_materials=1,
    )
    sched = build_noise_scheduler(
        sampler="ddim",
        num_train_timesteps=60,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    model.train()
    for step in range(240):
        opt.zero_grad(set_to_none=True)
        noise = torch.randn_like(x0_batch)
        t = torch.randint(0, 60, (4,), dtype=torch.long)
        mask = (torch.rand(4) >= 0.20).to(torch.float32)
        noisy = sched.add_noise(original_samples=x0_batch, noise=noise, timesteps=t)
        v_target = sched.get_velocity(x0_batch, noise, t)
        pred = model(noisy, t, loads_batch, cond_mask=mask)
        loss = torch.nn.functional.mse_loss(pred, v_target)
        loss.backward()
        opt.step()

    model.eval()
    sources = np.stack([vol_a_low, vol_b_low], axis=0)
    out_060 = virtual_compress_volumes(
        model=model,
        alphas_cumprod=sched.alphas_cumprod,
        source_volumes=sources,
        source_load=50.0,
        target_loads=[50.0, 300.0],
        prediction_type="v_prediction",
        num_steps=20,
        invert_strength=0.60,
        guidance_scale=2.0,
        interface_smooth_sigma=0.0,
    )
    out_075 = virtual_compress_volumes(
        model=model,
        alphas_cumprod=sched.alphas_cumprod,
        source_volumes=sources,
        source_load=50.0,
        target_loads=[50.0, 300.0],
        prediction_type="v_prediction",
        num_steps=20,
        invert_strength=0.75,
        guidance_scale=2.0,
        interface_smooth_sigma=0.0,
    )

    fid_060 = compute_virtual_compression_fidelity(
        source_volumes=sources,
        compressed_volumes=out_060[300.0],
        same_load_reconstructions=out_060[50.0],
    )
    fid_075 = compute_virtual_compression_fidelity(
        source_volumes=sources,
        compressed_volumes=out_075[300.0],
        same_load_reconstructions=out_075[50.0],
    )
    sf_low = float((out_060[50.0] >= 1).mean())
    sf_high_060 = float((out_060[300.0] >= 1).mean())
    sf_high_075 = float((out_075[300.0] >= 1).mean())

    # 1. Same-load roundtrip preserves solid fibres (measured via solid Dice, not sparse background accuracy)
    assert fid_060["same_load_solid_dice"] > 0.85, f"Low roundtrip solid Dice: {fid_060}"
    assert fid_075["same_load_solid_dice"] > 0.85, f"Low roundtrip solid Dice: {fid_075}"
    # 2. Target-load 300 kPa genuinely increases solid fraction vs 50 kPa
    assert sf_high_060 > sf_low + 0.015, f"Expected compaction at 0.60: {sf_low:.3f} -> {sf_high_060:.3f}"
    assert sf_high_075 > sf_low + 0.030, f"Expected stronger compaction at 0.75: {sf_low:.3f} -> {sf_high_075:.3f}"
    # 3. Compressed outputs retain specimen-specific XY structure (A vs B) rather than random resampling
    assert fid_060["xy_correlation_lift"] > 0.50, f"Expected specimen-specific XY lift: {fid_060}"
    assert fid_075["xy_correlation_lift"] > 0.50, f"Expected specimen-specific XY lift: {fid_075}"


def test_seamless_tiled_3d_multidiffusion() -> None:
    torch.manual_seed(11)
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=0.0,
        load_max=500.0,
        num_phases=3,
        num_materials=3,
    ).eval()
    sched = build_noise_scheduler(
        sampler="ddim",
        num_train_timesteps=50,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )
    sched.set_timesteps(4, device=torch.device("cpu"))
    gen = torch.Generator(device="cpu").manual_seed(99)

    phases, probs = generate_tiled_3d(
        model=model,
        scheduler=sched,
        target_shape=(16, 16, 32),
        patch_size=16,
        load_value=150.0,
        guidance_scale=1.2,
        overlap=0.5,
        batch_size=1,
        tile_batch_size=2,
        material_id=2,
        voxel_size_um=1.6,
        generator=gen,
    )
    assert phases.shape == (1, 16, 16, 32)
    assert probs.shape == (1, 16, 16, 32)
    assert np.isfinite(probs).all()
    assert set(np.unique(phases)).issubset({0, 1, 2})
    # Verify encoded channels have valid shape too
    enc = encode_phases_to_channels(phases[0], num_phases=3)
    assert enc.shape == (2, 16, 16, 32)


def test_optimal_clamping_pressure_and_full_cli_pipeline(tmp_path: Path) -> None:
    # 1. Unit test sweet-spot finder on interior-peaked reaction-transport curve
    rng = np.random.default_rng(123)
    synthetic_loads = [20.0, 100.0, 200.0, 350.0, 500.0]
    recs_by_load: dict[float, list[dict[str, float]]] = {}
    for lv in synthetic_loads:
        # Ionic D_eff decreases with compression; Electronic sigma_eff rises steeply then saturates;
        # Active wired interface peaks near ~200 kPa before over-compression occludes pores.
        d_eff = max(0.02, 0.55 * np.exp(-lv / 320.0))
        sig_eff = 0.08 * (1.0 - np.exp(-lv / 95.0))
        act_if = 0.25 * (1.0 - np.exp(-lv / 80.0)) * np.exp(-max(0.0, lv - 180.0) / 450.0)
        recs_by_load[lv] = [
            {
                "solid_fraction": 0.10 + 0.0005 * lv,
                "porosity": 0.90 - 0.0005 * lv,
                "d_eff_pore_z": float(d_eff * (1.0 + rng.normal(0.0, 0.03))),
                "sigma_eff_solid_z": float(sig_eff * (1.0 + rng.normal(0.0, 0.03))),
                "permeability_ck_um2": float(10.0 * d_eff),
                "active_wired_interface": float(act_if * (1.0 + rng.normal(0.0, 0.03))),
                "wired_coating_fraction": 0.9,
                "tpb_density": 0.05,
                "coating_pinch_fraction": 0.02 * (lv / 100.0),
                "used_openimpala": 1.0,
            }
            for _ in range(6)
        ]

    opt = find_optimal_clamping_pressure(recs_by_load, n_bootstrap=100, seed=42)
    assert 120.0 <= float(opt["optimal_load"]) <= 280.0  # type: ignore[arg-type]
    ci_lo, ci_hi = opt["optimal_load_ci95"]  # type: ignore[misc]
    assert ci_lo <= float(opt["optimal_load"]) <= ci_hi  # type: ignore[arg-type]

    # 2. End-to-end CLI smoke test across physics-loss training, tiled sampling,
    #    virtual compression, benchmark with virtual compression, and clamping optimization!
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    patch_size = 16
    version = "v04_smoke"

    scan_defs = [
        ("virgin_low", "virgin_carbon", 20.0, 2, 0.12),
        ("virgin_mid", "virgin_carbon", 180.0, 2, 0.22),
        ("virgin_high", "virgin_carbon", 400.0, 2, 0.34),
        ("tio2_low", "tio2_anode", 25.0, 3, 0.15),
        ("tio2_mid", "tio2_anode", 200.0, 3, 0.25),
        ("tio2_high", "tio2_anode", 420.0, 3, 0.36),
    ]
    manifest_scans = []
    for st, mat, kpa, n_ph, sf in scan_defs:
        pdir_name = f"patches_{st}_{patch_size}_{version}"
        for split, n_p in (("train", 4), ("val", 2)):
            sdir = data_root / pdir_name / split
            sdir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((n_p, patch_size, patch_size, patch_size), dtype=np.uint8)
            for i in range(n_p):
                v = (rng.random((patch_size, patch_size, patch_size)) < sf).astype(np.uint8)
                # Add percolating Z pillar so tortuosities are well-defined
                v[:, 4, 4] = 1
                if n_ph == 3:
                    coat_mask = (v == 1) & (rng.random(v.shape) < 0.30)
                    v[coat_mask] = 2
                arr[i] = v
            np.save(sdir / "patches.npy", arr)
        manifest_scans.append(
            {
                "state": st,
                "material": mat,
                "beamline": "DLS_I12",
                "is_repeat_or_secondary": False,
                "load_mv_per_v": round(kpa / 10000.0, 5),
                "stress_kpa": kpa,
                "effective_voxel_size_um": 3.2,
                "num_phases": n_ph,
                "patch_dir_name": pdir_name,
            }
        )

    manifest_path = data_root / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps({"version": version, "patch_size": patch_size, "scans": manifest_scans}, indent=2)
    )

    exp = "novelty_e2e_16"
    # Train with --physics-loss
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "training" / "train_carbon_conditional_96.py"),
            "--experiment",
            exp,
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_size),
            "--channels",
            "8,16",
            "--attention-levels",
            "0,1",
            "--num-res-blocks",
            "1",
            "--norm-groups",
            "4",
            "--timesteps",
            "40",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--grad-accum",
            "1",
            "--use-stress-kpa",
            "--balanced-sampling",
            "--physics-loss",
            "--allow-cpu",
        ],
        check=True,
    )

    # Sample standard 16^3 patches for all states
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "sample_carbon_conditional_96.py"),
            "--experiment",
            exp,
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--steps",
            "4",
            "--num-samples",
            "2",
            "--batch-size",
            "2",
            "--sweep-loads",
            "",
            "--allow-cpu",
        ],
        check=True,
    )

    # Sample seamless >16^3 tiled volume into a separate test check via virtual_compress_scan CLI too
    input_scan_path = tmp_path / "lab_ct_0kpa.npy"
    src_patch = np.load(data_root / f"patches_tio2_low_{patch_size}_{version}" / "val" / "patches.npy")[0]
    np.save(input_scan_path, src_patch)
    vc_out = tmp_path / "vc_cli_out"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "virtual_compress_scan.py"),
            "--input",
            str(input_scan_path),
            "--experiment",
            exp,
            "--output-root",
            str(out_root),
            "--out-dir",
            str(vc_out),
            "--source-load",
            "25.0",
            "--target-loads",
            "200.0,420.0",
            "--material",
            "tio2_anode",
            "--steps",
            "6",
            "--export-openimpala",
            "--allow-cpu",
        ],
        check=True,
    )
    assert (vc_out / "virtual_compression_panel.png").is_file()
    assert (vc_out / "virtual_compression_metrics.csv").is_file()

    # Benchmark with --include-virtual-compression
    bench_dir = out_root / "results" / exp / "beamtime_benchmark"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff={exp}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_size),
            "--heldout-states",
            "tio2_mid",
            "--include-virtual-compression",
            "--virtual-steps",
            "6",
            "--allow-cpu",
        ],
        check=True,
    )
    bench_report = json.loads((bench_dir / "beamtime_multiplier_report.json").read_text())
    assert "virtual_compression" in bench_report["tio2_mid"]["methods"]
    assert (bench_dir / "coating_retention_diagnostics.json").is_file()

    # Mock OpenImpala transport CSV to test seamless OpenImpala ingestion in optimize_clamping_pressure.py
    oi_csv = bench_dir / "openimpala_transport_parity.csv"
    with oi_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "state", "material", "transport_phase", "d_eff_z"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "real",
                "state": "tio2_mid",
                "material": "tio2_anode",
                "transport_phase": "pore_ionic",
                "d_eff_z": "0.31",
            }
        )
        writer.writerow(
            {
                "method": "real",
                "state": "tio2_mid",
                "material": "tio2_anode",
                "transport_phase": "carbon_electronic",
                "d_eff_z": "0.09",
            }
        )
    oi_map = load_openimpala_lookup(oi_csv)
    assert abs(oi_map[("real", "tio2_mid")]["pore_d_eff_z"] - 0.31) < 1e-6
    rec_with_oi = compute_transport_and_fom_record(
        evaluate_multiphase_physics(src_patch, voxel_size=3.2),
        openimpala_metrics=oi_map[("real", "tio2_mid")],
        voxel_size_um=3.2,
    )
    assert rec_with_oi["used_openimpala"] == 1.0

    # Run Optimal Clamping Pressure CLI with OpenImpala CSV + dense continuous sweep
    opt_dir = out_root / "results" / exp / "clamping_optimization"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "optimize_clamping_pressure.py"),
            "--experiments",
            f"porousdiff={exp}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_size),
            "--openimpala-csv",
            str(oi_csv),
            "--dense-sweep-points",
            "4",
            "--dense-samples-per-load",
            "2",
            "--steps",
            "4",
            "--bootstrap",
            "30",
            "--allow-cpu",
        ],
        check=True,
    )
    opt_summary = json.loads((opt_dir / "optimal_clamping_summary.json").read_text())
    assert "tio2_anode" in opt_summary
    assert "porousdiff" in opt_summary["tio2_anode"]
    assert "porousdiff_dense" in opt_summary["tio2_anode"]
    assert opt_summary["tio2_anode"]["real"]["estimator_mode"] == "geodesic_surrogate"
    assert "weight_sensitivity" in opt_summary["tio2_anode"]["porousdiff"]
    assert "newman_asr" in opt_summary["tio2_anode"]["porousdiff"]
    assert "regime_sensitivity" in opt_summary["tio2_anode"]["porousdiff"]["newman_asr"]
    assert (opt_dir / "clamping_optimization_curves.csv").is_file()
    assert (opt_dir / "optimal_clamping_pressure.png").is_file()


def test_partial_volume_carbon_halo_stripping_preserves_real_tpb() -> None:
    from scipy.ndimage import gaussian_filter
    from porousdiff.preprocessing import segment_slice_multiphase

    # Synthesize a Paganin/PSF-blurred 3-phase slice with:
    #   (A) an isolated TiO2 particle in pore space [y=20:32, x=20:32] (no real carbon fibre),
    #   (B) a real 6-px-wide carbon fibre [y=65:105, x=60:66] with TiO2 coating on its right flank [y=75:95, x=66:73].
    base = np.full((128, 128), 10000.0, dtype=np.float32)
    base[20:32, 20:32] = 28000.0  # Isolated bright coating particle in pore
    base[65:105, 60:66] = 18000.0  # Real 6-px carbon fibre
    base[75:95, 66:73] = 28000.0  # Attached coating deposit creating real 0-1-2 TPBs
    blurred = gaussian_filter(base, sigma=1.1)

    seg_raw, _ = segment_slice_multiphase(
        blurred, delta_carbon=4000.0, delta_coating=12000.0, sigma=0.0, strip_halo_px=0
    )
    seg_clean, _ = segment_slice_multiphase(
        blurred, delta_carbon=4000.0, delta_coating=12000.0, sigma=0.0, strip_halo_px=1
    )

    # Without cleanup, blur creates a fake phase-1 carbon rind around the isolated coating blob
    isolated_raw = seg_raw[15:37, 15:37]
    isolated_clean = seg_clean[15:37, 15:37]
    assert int((isolated_raw == 1).sum()) > 10
    assert int((isolated_clean == 1).sum()) == 0
    assert int((isolated_clean == 2).sum()) > 50

    # Real >= 3 px carbon fibre and its 0-1-2 junction stay intact
    fibre_clean = seg_clean[65:105, 58:76]
    assert int((fibre_clean == 1).sum()) > 180
    assert int((fibre_clean == 2).sum()) > 90
    # Verify genuine 3-phase TPB 2x2 plaquettes exist at the top/bottom coating-fibre-pore corners
    found_tpb = False
    for y in range(70, 100):
        for x in range(58, 74):
            patch2x2 = seg_clean[y : y + 2, x : x + 2]
            if set(np.unique(patch2x2)) == {0, 1, 2}:
                found_tpb = True
                break
    assert found_tpb


def test_thin_scan_auto_xy_sector_split_zero_overlap(tmp_path: Path) -> None:
    import tifffile
    from porousdiff.preprocessing import enumerate_cylinder_xy_positions, split_xy_positions_with_buffer
    from scripts.preprocessing.prepare_local_drive_bundle import stream_bundle_single_scan

    # Verify strict pixel-level non-overlap in split_xy_positions_with_buffer
    xy_pts = enumerate_cylinder_xy_positions((128, 128), patch_size=32, stride_xy=16, radius_fraction=0.95)
    xy_split = split_xy_positions_with_buffer(xy_pts, patch_size=32, val_fraction=0.25)
    assert len(xy_split["train"]) >= 1 and len(xy_split["val"]) >= 1
    for ty, tx in xy_split["train"]:
        for vy, vx in xy_split["val"]:
            overlap_y = (ty < vy + 32) and (vy < ty + 32)
            overlap_x = (tx < vx + 32) and (vx < tx + 32)
            assert not (overlap_y and overlap_x)

    # Synthesize a heavily compressed thin scan (only 40 slices < 3 * raw_window = 96)
    scan_dir = tmp_path / "CuHCF_450kPa"
    scan_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for z in range(40):
        slc = np.full((128, 128), 10000, dtype=np.uint16)
        fibres = rng.random((128, 128)) < 0.18
        coat = fibres & (rng.random((128, 128)) < 0.35)
        slc[fibres] = 18000
        slc[coat] = 28000
        tifffile.imwrite(scan_dir / f"slice_{z:04d}.tif", slc)

    out_dir = tmp_path / "bundle_thin"
    summary = stream_bundle_single_scan(
        scan_dir=scan_dir,
        output_dir=out_dir,
        version="thin_v1",
        auto_multiphase=True,
        delta_threshold=4000.0,
        delta_coating_threshold=12000.0,
        sigma=0.0,
        roi_radius_fraction=0.95,
        patch_size=16,
        spatial_bin=2,
        stride_xy=8,
        stride_z=4,
        max_patches=20,
        min_interval_length=16,
        seam_margin=0,
        split_mode="auto",
    )
    assert summary["split_mode_used"] == "xy"
    pdir = out_dir / str(summary["patch_dir_name"])
    assert (pdir / "train" / "patches.npy").is_file()
    assert (pdir / "val" / "patches.npy").is_file()


def test_newman_asr_dimensional_sweet_spot_and_regime_sensitivity() -> None:
    loads = [0.0, 80.0, 180.0, 320.0, 500.0]
    recs_by_load: dict[float, list[dict[str, float]]] = {}
    for i, lv in enumerate(loads):
        strain_z = min(0.55, lv / 900.0)
        d_eff = max(0.015, 0.58 * np.exp(-lv / 240.0))
        sig_eff = 0.01 + 0.14 * (1.0 - np.exp(-lv / 90.0))
        act_if = 0.04 + 0.22 * (1.0 - np.exp(-lv / 70.0)) * np.exp(-max(0.0, lv - 160.0) / 380.0)
        recs_by_load[lv] = [
            {
                "solid_fraction": 0.12 / (1.0 - strain_z),
                "porosity": 1.0 - 0.12 / (1.0 - strain_z),
                "strain_z": strain_z,
                "d_eff_pore_z": d_eff,
                "sigma_eff_solid_z": sig_eff,
                "permeability_ck_um2": 18.0 * (d_eff**1.5),
                "active_wired_interface": act_if,
                "z_pinch_contact_density": 0.0005 * lv,
            }
            for _ in range(4)
        ]

    res = find_optimal_clamping_pressure(recs_by_load, n_bootstrap=40, uncompressed_thickness_cm=0.30)
    asr = res["newman_asr"]  # type: ignore[index]
    assert 40.0 <= float(asr["optimal_load_asr"]) <= 350.0  # type: ignore[index]
    assert float(asr["min_asr_ohm_cm2"]) > 0.0  # type: ignore[index]
    thick_grid = asr["thickness_cm_grid"]  # type: ignore[index]
    assert thick_grid[0] > thick_grid[-1]
    regimes = asr["regime_sensitivity"]  # type: ignore[index]
    # Mass-transport-limited cells suffer from pore closure earlier than kinetics-limited cells
    assert float(regimes["mass_transport_limited"]["optimal_load_asr"]) <= float(  # type: ignore[index]
        regimes["kinetics_limited"]["optimal_load_asr"]  # type: ignore[index]
    )


def test_mixed_unit_manifest_guard_and_openimpala_template_and_tile_symmetry(tmp_path: Path) -> None:
    import pytest
    from porousdiff.transport_and_xrd import generate_openimpala_job_suite
    from porousdiff.virtual_compression import _compute_tile_starts
    from scripts.training.train_carbon_conditional_96 import load_manifest_specs

    # 1. Symmetric even MultiDiffusion tile spacing
    assert _compute_tile_starts(160, 96, overlap=0.25) == [0, 64]
    starts_200 = _compute_tile_starts(200, 96, overlap=0.5)
    assert starts_200 == [0, 35, 69, 104]

    # 2. Mixed mechanical unit / multi-beamline guard in load_manifest_specs
    bad_manifest = tmp_path / "mixed_uncalibrated.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "scans": [
                    {"state": "i12_a", "beamline": "DLS_I12", "load_unit": "mv_per_v", "load_mv_per_v": 0.01, "stress_kpa": None},
                    {"state": "k11_b", "beamline": "DLS_K11_DIAD", "load_unit": "newton", "load_mv_per_v": 15.0, "stress_kpa": None},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Cannot mix uncalibrated load units"):
        load_manifest_specs(bad_manifest, tmp_path, use_stress_kpa=False)

    # Auto-promotion to stress_kpa (including 0.0 kPa!) when calibrated stress_kpa is present
    good_manifest = tmp_path / "mixed_calibrated.json"
    good_manifest.write_text(
        json.dumps(
            {
                "scans": [
                    {"state": "i12_0kpa", "beamline": "DLS_I12", "load_unit": "mv_per_v", "load_mv_per_v": 0.001, "stress_kpa": 0.0},
                    {"state": "k11_250kpa", "beamline": "DLS_K11_DIAD", "load_unit": "newton", "load_mv_per_v": 12.0, "stress_kpa": 250.0},
                ]
            }
        )
    )
    specs, _, _ = load_manifest_specs(good_manifest, tmp_path, use_stress_kpa=False)
    assert [s["load_value"] for s in specs] == [0.0, 250.0]
    assert all(bool(s["used_stress_kpa"]) for s in specs)

    # 3. Custom OpenImpala user INI template + 3-phase solid_hetero_electronic job
    tpl = tmp_path / "custom_tortuosity.ini"
    tpl.write_text(
        "FILE={tiff_path}\nPHASE={phase}\nLABEL={transport_phase}\nCOND={phase_conductivities}\nOUT={output_prefix}\n"
    )
    fake_tif = tmp_path / "sample.tif"
    fake_tif.write_bytes(b"II*\x00")
    suite = generate_openimpala_job_suite(
        tiff_entries=[
            {
                "method": "porousdiff",
                "state": "tio2_200kpa",
                "material": "tio2_anode",
                "beamline": "DLS_I12",
                "load_value": 200.0,
                "tiff_path": fake_tif,
                "num_phases": 3,
            }
        ],
        suite_dir=tmp_path / "oi_suite",
        template_ini=tpl,
        include_hetero_solid=True,
    )
    assert suite["num_jobs"] == 4
    hetero_ini = (tmp_path / "oi_suite" / "configs" / "porousdiff__tio2_200kpa__sample__solid_hetero_electronic.ini").read_text()
    assert "COND=1:1.0,2:0.01" in hetero_ini


def test_peer_review_bugfixes_units_resume_tortuosity_tall_baselines_and_stats(tmp_path: Path) -> None:
    import pytest
    import tifffile
    from porousdiff.baselines import compute_mirror_tiled_fraction
    from porousdiff.data import resolve_scan_load_value
    from porousdiff.metrics import geodesic_pore_tortuosity
    from porousdiff.preprocessing import audit_binning_topology_preservation
    from scripts.preprocessing.prepare_local_drive_bundle import stream_bundle_single_scan
    from scripts.training.train_carbon_conditional_96 import load_manifest_specs

    # 1. Bug #6: 26-connected Euclidean geodesic tortuosity recovers exact sqrt(2) on a 45-degree pore channel
    diag_vol = np.ones((24, 26, 24), dtype=np.uint8)
    for z in range(24):
        diag_vol[z, z : z + 2, 10:12] = 0
    tau_26 = geodesic_pore_tortuosity(diag_vol, axis=0, connectivity=26)
    tau_6 = geodesic_pore_tortuosity(diag_vol, axis=0, connectivity=6)
    assert abs(tau_26 - float(np.sqrt(2.0))) < 0.015, f"Expected ~1.4142, got {tau_26}"
    assert tau_6 > 1.90

    # 2. Bug #1 & #2: Strict load-unit resolution (preserves 0.0 kPa; raises on missing stress_kpa;
    #    reads checkpoint use_stress_kpa=False even when manifest carries stress_kpa!)
    assert resolve_scan_load_value({"state": "s0", "stress_kpa": 0.0, "load_mv_per_v": 0.008}, True) == 0.0
    missing_kpa_manifest = tmp_path / "missing_kpa.json"
    missing_kpa_manifest.write_text(
        json.dumps(
            {
                "scans": [
                    {"state": "s1", "load_mv_per_v": 0.008, "stress_kpa": 68.0},
                    {"state": "s2", "load_mv_per_v": 0.033, "stress_kpa": None},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="lack calibrated stress_kpa"):
        load_manifest_specs(missing_kpa_manifest, tmp_path, use_stress_kpa=True)

    # Train a model WITHOUT --use-stress-kpa (conditioned on mV/V: 0.010, 0.020, 0.033) on a manifest
    # that ALSO has stress_kpa (85.0, 170.0, 280.5), then verify evaluate & optimize use mV/V (not kPa!)
    rng = np.random.default_rng(99)
    data_root = tmp_path / "unit_data"
    out_root = tmp_path / "unit_out"
    p_sz = 16
    ver = "unit_v1"
    mv_scans = []
    for st, mv, sf in [("v_low", 0.010, 0.12), ("v_mid", 0.020, 0.20), ("v_high", 0.033, 0.30)]:
        pdir = f"patches_{st}_{p_sz}_{ver}"
        for sp, n_p in (("train", 3), ("val", 2)):
            sd = data_root / pdir / sp
            sd.mkdir(parents=True, exist_ok=True)
            arr = (rng.random((n_p, p_sz, p_sz, p_sz)) < sf).astype(np.uint8)
            arr[:, :, 4, 4] = 1
            np.save(sd / "patches.npy", arr)
        mv_scans.append(
            {
                "state": st,
                "material": "virgin_carbon",
                "beamline": "DLS_I12",
                "is_repeat_or_secondary": False,
                "load_unit": "mv_per_v",
                "load_mv_per_v": mv,
                "stress_kpa": round(mv * 8500.0, 2),
                "effective_voxel_size_um": 1.6,
                "num_phases": 2,
                "patch_dir_name": pdir,
            }
        )
    mv_manifest = data_root / "bundle_manifest.json"
    mv_manifest.write_text(json.dumps({"version": ver, "patch_size": p_sz, "scans": mv_scans}, indent=2))

    exp_mv = "mv_unit_check"
    base_train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "training" / "train_carbon_conditional_96.py"),
        "--experiment",
        exp_mv,
        "--data-root",
        str(data_root),
        "--output-root",
        str(out_root),
        "--manifest-json",
        str(mv_manifest),
        "--version",
        ver,
        "--patch-size",
        str(p_sz),
        "--channels",
        "8,16",
        "--attention-levels",
        "0,1",
        "--num-res-blocks",
        "1",
        "--norm-groups",
        "4",
        "--timesteps",
        "20",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--grad-accum",
        "1",
        "--exclude-states",
        "v_mid",
        "--allow-cpu",
    ]
    subprocess.run(base_train_cmd, check=True)

    # 3. Bug #3: --resume MUST refuse if heldout_states (--exclude-states) or --use-stress-kpa changes!
    leaked_cmd = [arg for arg in base_train_cmd if arg not in ("--exclude-states", "v_mid")] + [
        "--epochs",
        "2",
        "--resume",
    ]
    res_leak = subprocess.run(leaked_cmd, capture_output=True, text=True)
    assert res_leak.returncode != 0
    assert "Refusing --resume" in (res_leak.stderr + res_leak.stdout)

    # Sample & evaluate on manifest that contains BOTH stress_kpa and load_mv_per_v
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "sample_carbon_conditional_96.py"),
            "--experiment",
            exp_mv,
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(mv_manifest),
            "--steps",
            "4",
            "--num-samples",
            "2",
            "--batch-size",
            "2",
            "--sweep-loads",
            "",
            "--allow-cpu",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff={exp_mv}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(mv_manifest),
            "--version",
            ver,
            "--patch-size",
            str(p_sz),
            "--heldout-states",
            "v_mid",
            "--include-virtual-compression",
            "--virtual-steps",
            "4",
            "--allow-cpu",
        ],
        check=True,
    )
    mv_rep = json.loads(
        (out_root / "results" / exp_mv / "beamtime_benchmark" / "beamtime_multiplier_report.json").read_text()
    )
    assert mv_rep["v_mid"]["load_unit"] == "mV/V"
    assert abs(float(mv_rep["v_mid"]["load_value"]) - 0.020) < 1e-6
    assert mv_rep["v_mid"]["virtual_compression_diagnostics"]["is_extrapolated"] is False

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "optimize_clamping_pressure.py"),
            "--experiments",
            f"porousdiff={exp_mv}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(mv_manifest),
            "--version",
            ver,
            "--patch-size",
            str(p_sz),
            "--bootstrap",
            "15",
            "--allow-cpu",
        ],
        check=True,
    )
    mv_opt = json.loads(
        (out_root / "results" / exp_mv / "clamping_optimization" / "optimal_clamping_summary.json").read_text()
    )
    assert 0.009 <= float(mv_opt["virgin_carbon"]["porousdiff"]["optimal_load"]) <= 0.034

    # 4. Bug #4: Tall clean validation columns eliminate baseline mirror tiling (mirror_frac == 0.0)
    tall_scan = tmp_path / "Virgin_0p008mVV"
    tall_scan.mkdir(parents=True)
    for z in range(80):
        slc = np.full((96, 96), 10000, dtype=np.uint16)
        slc[rng.random((96, 96)) < 0.14] = 18000
        tifffile.imwrite(tall_scan / f"slice_{z:04d}.tif", slc)
    tall_sum = stream_bundle_single_scan(
        scan_dir=tall_scan,
        output_dir=tmp_path / "tall_bundle",
        version="tall_v1",
        delta_threshold=4000.0,
        sigma=0.0,
        roi_radius_fraction=0.95,
        patch_size=16,
        spatial_bin=1,
        stride_xy=16,
        stride_z=8,
        max_patches=12,
        min_interval_length=16,
        seam_margin=0,
        val_fraction=0.45,
    )
    assert int(tall_sum["val_tall_depth"]) >= 32  # type: ignore[arg-type]
    tall_pdir = tmp_path / "tall_bundle" / str(tall_sum["patch_dir_name"]) / "val" / "tall_patches.npy"
    assert tall_pdir.is_file()
    assert (
        compute_mirror_tiled_fraction(
            source_depth=int(tall_sum["val_tall_depth"]),  # type: ignore[arg-type]
            output_depth=16,
            initial_solid_fraction=0.14,
            target_solid_fraction=0.28,
        )
        == 0.0
    )

    # 5. Mixed-estimator guard in find_optimal_clamping_pressure + partial XRD correlation + topology audit
    mixed_recs = {
        10.0: [{"d_eff_pore_z": 0.5, "sigma_eff_solid_z": 0.05, "active_wired_interface": 0.1, "used_openimpala": 1.0}],
        50.0: [{"d_eff_pore_z": 0.3, "sigma_eff_solid_z": 0.09, "active_wired_interface": 0.2, "used_openimpala": 0.0}],
    }
    with pytest.raises(ValueError, match="Refusing to mix OpenImpala"):
        find_optimal_clamping_pressure(mixed_recs, n_bootstrap=10)

    # Partial correlation controlling for solid_fraction separates pure sf proxies from true pinch excess
    xrd_csv2 = tmp_path / "xrd_partial.csv"
    sfs = [0.10, 0.16, 0.22, 0.28, 0.34]
    pinch_excess = [1.1, 1.8, 1.3, 2.4, 2.0]  # Orthogonal-ish variation to sf
    mock_rep2: dict[str, object] = {}
    with xrd_csv2.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["state", "pure_sf_signal", "true_pinch_signal"])
        wr.writeheader()
        for i, (sf_i, pe_i) in enumerate(zip(sfs, pinch_excess)):
            st_i = f"k11_{i}"
            mock_rep2[st_i] = {
                "material": "cuhcf_cathode",
                "load_value": float(50 * i),
                "real": {
                    "metrics": {
                        "solid_fraction": {"mean": sf_i},
                        "z_pinch_contact_density": {"mean": sf_i * 2.0 + 0.001 * pe_i},
                        "specific_z_pinch_excess": {"mean": pe_i},
                    }
                },
                "methods": {},
            }
            wr.writerow(
                {
                    "state": st_i,
                    "pure_sf_signal": round(10.0 * sf_i, 5),
                    "true_pinch_signal": round(2.0 * sf_i + 3.0 * pe_i, 5),
                }
            )
    p_corr = correlate_xrd_with_microstructure(xrd_csv2, mock_rep2, tmp_path / "xrd_partial_out")
    assert abs(p_corr["correlations"]["pure_sf_signal"]["real_z_pinch_contact_density"]["partial_pearson_r_given_sf"]) < 0.05
    assert p_corr["correlations"]["true_pinch_signal"]["real_specific_z_pinch_excess"]["partial_pearson_r_given_sf"] > 0.95

    from porousdiff.preprocessing import downsample_volume_rev

    raw_3d = np.zeros((16, 16, 16), dtype=np.uint8)
    raw_3d[:, 6:10, 6:10] = 1
    binned_3d = downsample_volume_rev(raw_3d, factor=2)
    topo_audit = audit_binning_topology_preservation(raw_3d, binned_3d)
    assert topo_audit["pore_percolation_agreement"] == 1.0
    assert abs(topo_audit["solid_fraction_abs_err"]) < 1e-6


def test_cli_manifest_state_subset_and_checkpoint_safeguards(tmp_path: Path) -> None:
    """Verify beamline-stripped state matching, typo/ambiguity rejection, relative ckpt lookup, and fallback_root."""
    import pytest

    from porousdiff.data import (
        resolve_checkpoint_path,
        resolve_manifest_state_subset,
        resolve_patch_split_dir,
    )

    scans = [
        {"state": "DLS_I12__tio2_0p008mVV", "source_folder": "tio2_0p008mVV"},
        {"state": "DLS_I12__tio2_0p030mVV", "source_folder": "tio2_0p030mVV"},
        {"state": "DLS_K11__cuhcf_0p030mVV", "source_folder": "cuhcf_0p030mVV"},
    ]
    # Exact and beamline-stripped suffix both resolve to canonical manifest state
    assert resolve_manifest_state_subset(scans, ["tio2_0p030mVV"], "--exclude-states") == {
        "DLS_I12__tio2_0p030mVV"
    }
    assert resolve_manifest_state_subset(scans, ["DLS_I12__tio2_0p008mVV"]) == {
        "DLS_I12__tio2_0p008mVV"
    }
    # Typos raise ValueError rather than silently failing to exclude a held-out state
    with pytest.raises(ValueError, match="Unknown state"):
        resolve_manifest_state_subset(scans, ["tio2_0p030_typo"], "--exclude-states")

    # Ambiguous suffix across multiple beamlines raises ValueError
    dup_scans = [
        {"state": "DLS_I12__tio2_0p030mVV"},
        {"state": "DLS_K11__tio2_0p030mVV"},
    ]
    with pytest.raises(ValueError, match="Ambiguous state"):
        resolve_manifest_state_subset(dup_scans, ["tio2_0p030mVV"], "--heldout-states")

    # Relative --checkpoint filename resolves inside output_root/checkpoints/<experiment>/
    ckpt_dir = tmp_path / "checkpoints" / "exp_demo"
    ckpt_dir.mkdir(parents=True)
    latest = ckpt_dir / "latest.pth"
    latest.write_bytes(b"stub")
    assert resolve_checkpoint_path(tmp_path, "exp_demo", Path("latest.pth")) == latest
    assert resolve_checkpoint_path(tmp_path, "exp_demo", Path("nonexistent.pth")) is None

    # resolve_patch_split_dir falls back to manifest parent directory when data_root lacks the bundle
    bundle_root = tmp_path / "bundle_dir"
    val_dir = bundle_root / "DLS_I12__tio2_0p030mVV_binary_patches24_v1" / "val"
    val_dir.mkdir(parents=True)
    resolved_split = resolve_patch_split_dir(
        tmp_path / "missing_default_data_root",
        "DLS_I12__tio2_0p030mVV",
        24,
        "v1",
        "val",
        patch_dir_name="DLS_I12__tio2_0p030mVV_binary_patches24_v1",
        fallback_root=bundle_root,
    )
    assert resolved_split == val_dir


def test_reviewer_round2_noop_baseline_multiseed_grid_bounds_and_provenance(tmp_path: Path) -> None:
    """Verify no-op baseline, near-identity DDIM copy detection, discrete scan-grid spacing bounds,
    Newman-Tobias parameter note, EMA vs raw weight provenance, and multi-seed error bars."""
    from porousdiff.virtual_compression import compute_virtual_compression_fidelity

    rng = np.random.default_rng(2026)

    # 1. Near-identity copy detection in compute_virtual_compression_fidelity
    src_vols = (rng.random((3, 16, 16, 16)) < 0.14).astype(np.uint8)
    fid_copy = compute_virtual_compression_fidelity(
        source_volumes=src_vols,
        compressed_volumes=src_vols.copy(),
        same_load_reconstructions=src_vols.copy(),
        is_target_same_as_source=False,
    )
    assert fid_copy["changed_voxel_fraction_vs_source"] == 0.0
    assert fid_copy["mean_delta_solid_fraction"] == 0.0
    assert fid_copy["is_near_identity_copy"] is True

    fid_same_target = compute_virtual_compression_fidelity(
        source_volumes=src_vols,
        compressed_volumes=src_vols.copy(),
        is_target_same_as_source=True,
    )
    assert fid_same_target["is_near_identity_copy"] is False

    comp_vols = src_vols.copy()
    comp_vols[:, :6, :6, :6] = 1  # Genuine solid compaction
    fid_comp = compute_virtual_compression_fidelity(
        source_volumes=src_vols,
        compressed_volumes=comp_vols,
        is_target_same_as_source=False,
    )
    assert fid_comp["mean_delta_solid_fraction"] > 0.02
    assert fid_comp["is_near_identity_copy"] is False

    # 2. Discrete grid resolution bounds & explicit Newman-Tobias parameter note on [68, 153, 255, 374] kPa
    four_loads = [68.0, 153.0, 255.0, 374.0]
    recs_4pt: dict[float, list[dict[str, float]]] = {}
    for lv in four_loads:
        # Interior minimum ASR near 153 kPa
        d_eff = max(0.015, 0.55 * np.exp(-lv / 105.0))
        sig_eff = 0.02 + 0.14 * (1.0 - np.exp(-lv / 55.0))
        act_if = 0.10 + 0.24 * (1.0 - np.exp(-lv / 50.0)) * np.exp(-max(0.0, lv - 130.0) / 140.0)
        recs_4pt[lv] = [
            {
                "solid_fraction": 0.11 + 0.0004 * lv,
                "porosity": 0.89 - 0.0004 * lv,
                "strain_z": min(0.50, (lv - 68.0) / 600.0),
                "d_eff_pore_z": d_eff,
                "sigma_eff_solid_z": sig_eff,
                "permeability_ck_um2": 12.0 * d_eff,
                "active_wired_interface": act_if,
                "used_openimpala": 0.0,
            }
            for _ in range(3)
        ]
    opt_4pt = find_optimal_clamping_pressure(recs_4pt, n_bootstrap=20, seed=7)
    asr_4pt = opt_4pt["newman_asr"]  # type: ignore[index]
    assert float(asr_4pt["nearest_discrete_load_asr"]) == 153.0  # type: ignore[index]
    # Adjacent spacings around 153 kPa are (153-68)=85 and (255-153)=102 -> halfwidth = 0.5 * 102 = 51.0 kPa
    assert float(asr_4pt["grid_spacing_halfwidth_asr"]) == 51.0  # type: ignore[index]
    assert "not parameter-free" in str(asr_4pt["parameter_note"])  # type: ignore[index]

    # 3. Multi-seed evaluation (porousdiff=seed_a,seed_b), noop_uncompressed, and EMA provenance audit
    data_root = tmp_path / "ms_data"
    out_root = tmp_path / "ms_out"
    p_sz = 16
    ver = "ms_v1"
    scans = []
    for st, kpa, sf in [("c_68", 68.0, 0.11), ("c_153", 153.0, 0.16), ("c_255", 255.0, 0.23)]:
        pdir = f"patches_{st}_{p_sz}_{ver}"
        for sp, n_p in (("train", 2), ("val", 2)):
            sd = data_root / pdir / sp
            sd.mkdir(parents=True, exist_ok=True)
            arr = (rng.random((n_p, p_sz, p_sz, p_sz)) < sf).astype(np.uint8)
            arr[:, :, 4, 4] = 1
            np.save(sd / "patches.npy", arr)
        scans.append(
            {
                "state": st,
                "material": "virgin_carbon",
                "beamline": "DLS_I12",
                "is_repeat_or_secondary": False,
                "load_mv_per_v": round(kpa / 8500.0, 5),
                "stress_kpa": kpa,
                "strain_z": round((kpa - 68.0) / 500.0, 4),
                "effective_voxel_size_um": 1.6,
                "num_phases": 2,
                "patch_dir_name": pdir,
            }
        )
    manifest_p = data_root / "bundle_manifest.json"
    manifest_p.write_text(json.dumps({"version": ver, "patch_size": p_sz, "scans": scans}, indent=2))

    # Populate synthetic generated samples + sampling_provenance.json for 2 seeds (one EMA, one raw debug)
    for s_idx, (exp_n, w_src, sf_bias) in enumerate([("pd_s42", "ema", -0.01), ("pd_s43", "raw", +0.015)]):
        s_base = out_root / "results" / exp_n / "samples"
        s_base.mkdir(parents=True, exist_ok=True)
        (s_base / "sampling_provenance.json").write_text(
            json.dumps(
                {
                    "experiment": exp_n,
                    "weight_source": w_src,
                    "is_pre_registered_ema": (w_src == "ema"),
                    "seed": 42 + s_idx,
                },
                indent=2,
            )
        )
        for st, _, sf in [("c_68", 68.0, 0.11), ("c_153", 153.0, 0.16), ("c_255", 255.0, 0.23)]:
            st_d = s_base / st
            st_d.mkdir(parents=True, exist_ok=True)
            g_arr = (rng.random((2, p_sz, p_sz, p_sz)) < max(0.05, sf + sf_bias)).astype(np.uint8)
            g_arr[:, :, 4, 4] = 1
            np.save(st_d / "generated_samples.npy", g_arr)

    bench_out = tmp_path / "ms_bench"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            "porousdiff=pd_s42,pd_s43",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--out-dir",
            str(bench_out),
            "--manifest-json",
            str(manifest_p),
            "--version",
            ver,
            "--patch-size",
            str(p_sz),
            "--use-stress-kpa",
            "--heldout-states",
            "c_255",
        ],
        check=True,
    )
    ms_rep = json.loads((bench_out / "beamtime_multiplier_report.json").read_text())
    held_m = ms_rep["c_255"]["methods"]
    assert "noop_uncompressed" in held_m
    assert held_m["noop_uncompressed"]["s2_improvement_over_noop"] == 0.0
    pd_m = held_m["porousdiff"]
    assert pd_m["num_seeds"] == 2
    assert pd_m["s2_rmse_std"] >= 0.0
    assert len(pd_m["per_seed_s2_rmse"]) == 2
    assert pd_m["provenance"]["used_non_ema_debug_weights"] is True
    assert pd_m["metrics"]["solid_fraction"]["seed_std"] > 0.0

    clamp_out = tmp_path / "ms_clamp"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "optimize_clamping_pressure.py"),
            "--experiments",
            "porousdiff=pd_s42,pd_s43",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--out-dir",
            str(clamp_out),
            "--manifest-json",
            str(manifest_p),
            "--version",
            ver,
            "--patch-size",
            str(p_sz),
            "--use-stress-kpa",
            "--bootstrap",
            "15",
        ],
        check=True,
    )
    ms_clamp = json.loads((clamp_out / "optimal_clamping_summary.json").read_text())
    vc_clamp = ms_clamp["virgin_carbon"]
    assert "noop_uncompressed" in vc_clamp
    assert vc_clamp["porousdiff"]["multi_seed"]["num_seeds"] == 2
    assert len(vc_clamp["porousdiff"]["multi_seed"]["per_seed_optimal_load_asr"]) == 2
    assert vc_clamp["porousdiff"]["provenance"]["used_non_ema_debug_weights"] is True



