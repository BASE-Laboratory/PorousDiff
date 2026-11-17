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


def test_ddim_virtual_compression_invertibility() -> None:
    torch.manual_seed(7)
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
        num_train_timesteps=100,
        schedule="zero_snr_scaled_linear",
        prediction_type="v_prediction",
    )

    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[4:12, 4:12, 4:12] = 1
    vol[6:10, 6:10, 6:10] = 2

    # When target_load == source_load and interface_smooth_sigma == 0, deterministic DDIM ODE
    # forward inversion + reverse denoising should closely reconstruct the exact input patch!
    out_map = virtual_compress_volumes(
        model=model,
        alphas_cumprod=sched.alphas_cumprod,
        source_volumes=vol,
        source_load=10.0,
        target_loads=[10.0, 300.0],
        prediction_type="v_prediction",
        num_steps=25,
        invert_strength=0.65,
        guidance_scale=1.4,
        material_id=1,
        voxel_size_um=1.6,
        interface_smooth_sigma=0.0,
    )
    voxel_agreement = float((out_map[10.0] == vol).mean())
    assert voxel_agreement > 0.95
    assert out_map[300.0].shape == (16, 16, 16)
    assert set(np.unique(out_map[300.0])).issubset({0, 1, 2})


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
    assert (opt_dir / "clamping_optimization_curves.csv").is_file()
    assert (opt_dir / "optimal_clamping_pressure.png").is_file()
