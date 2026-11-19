"""Tests for PorousDiff v0.5 OpenImpala v4.6 PDE bridge, Cauchy-Crofton stereology, topology, and Lagrangian slab DDIM."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import openimpala as oi
import pytest
import torch

from porousdiff.metrics import (
    CAUCHY_CROFTON_3D_FACTOR,
    compute_ensemble_distribution_distances,
    euler_characteristic_3d,
    evaluate_multiphase_physics,
    evaluate_volume_physics,
    pore_throat_radius_metrics,
    specific_surface_area,
)
from porousdiff.models import ContinuousLoadConditionedUNet, build_noise_scheduler
from porousdiff.optimization import (
    compute_transport_and_fom_record,
    find_optimal_clamping_pressure,
)
from porousdiff.transport_and_xrd import (
    compute_openimpala_volume_transport,
    has_openimpala_python,
    run_openimpala_rev_convergence,
)
from porousdiff.virtual_compression import (
    compute_virtual_compression_fidelity,
    virtual_compress_lagrangian_slab,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
        }
    )
    return env


def test_cauchy_crofton_ssa_cylinder_sphere_and_openimpala_parity() -> None:
    """Verify Cauchy-Crofton 2/3 correction eliminates voxel staircasing bias and matches OpenImpala."""
    assert has_openimpala_python()
    oi.Session.ensure_active()

    # 1. Analytical 3D sphere inside (48, 48, 48)
    n = 48
    vox = 0.5
    zz, yy, xx = np.ogrid[:n, :n, :n]
    center = (n - 1) / 2.0
    radius_px = 14.0
    sphere = (((zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2) <= radius_px**2).astype(
        np.uint8
    )

    ssa_sp = specific_surface_area(sphere, voxel_size=vox)
    exact_sphere_ssa = (4.0 * math.pi * ((radius_px * vox) ** 2)) / ((n * vox) ** 3)
    raw_rel_err = abs(ssa_sp["ssa_total"] - exact_sphere_ssa) / exact_sphere_ssa
    cc_rel_err = abs(ssa_sp["ssa_total_cc"] - exact_sphere_ssa) / exact_sphere_ssa
    assert raw_rel_err > 0.35  # Raw Manhattan voxel faces overestimate smooth sphere by ~1.5x
    assert cc_rel_err < 0.05  # Cauchy-Crofton 2/3 matches analytical 4*pi*R^2/V within 5%

    oi_ssa_sp = oi.specific_surface_area(sphere, phase_a=0, phase_b=1, voxel_size=vox)
    assert ssa_sp["ssa_total"] == pytest.approx(float(oi_ssa_sp.ssa), rel=1e-5)
    assert ssa_sp["ssa_total_cc"] == pytest.approx(float(oi_ssa_sp.ssa_corrected), rel=1e-5)

    # 2. 3-phase coated cylindrical carbon fiber along Z inside (24, 48, 48)
    d_z = 24
    yy2, xx2 = np.ogrid[:n, :n]
    r2 = (yy2 - center) ** 2 + (xx2 - center) ** 2
    r_carb = 7.0
    r_coat = 11.0
    cyl_slice = np.zeros((n, n), dtype=np.uint8)
    cyl_slice[r2 <= r_coat**2] = 2
    cyl_slice[r2 <= r_carb**2] = 1
    cyl_vol = np.broadcast_to(cyl_slice[np.newaxis, :, :], (d_z, n, n)).copy()

    mp = evaluate_multiphase_physics(cyl_vol, voxel_size=vox)
    oi_res = compute_openimpala_volume_transport(cyl_vol, voxel_size_um=vox, num_phases=3)
    oi_if = oi_res["interfacial"]  # type: ignore[index]

    assert mp["ssa_pore_coating_cc"] == pytest.approx(
        mp["ssa_pore_coating"] * CAUCHY_CROFTON_3D_FACTOR, rel=1e-6
    )
    assert mp["ssa_pore_coating_cc"] == pytest.approx(
        float(oi_if["ssa_pore_coating_cc"]), rel=1e-5
    )
    assert mp["ssa_carbon_coating_cc"] == pytest.approx(
        float(oi_if["ssa_carbon_coating_cc"]), rel=1e-5
    )


def test_euler_characteristic_edt_and_distribution_distances() -> None:
    """Verify exact 3D cubical Euler characteristic, EDT pore radii, and W1/KS ensemble distances."""
    # 1. Single solid block: chi = +1
    block = np.zeros((12, 12, 12), dtype=np.uint8)
    block[3:8, 3:8, 3:8] = 1
    assert euler_characteristic_3d(block, voxel_size=0.5)["euler_characteristic"] == 1.0

    # 2. Closed square ring (1 component, 1 tunnel, 0 cavities): chi = 1 - 1 + 0 = 0
    torus = np.zeros((12, 12, 12), dtype=np.uint8)
    torus[4:7, 2:10, 2:10] = 1
    torus[4:7, 4:8, 4:8] = 0
    assert euler_characteristic_3d(torus, voxel_size=1.0)["euler_characteristic"] == 0.0

    # 3. Closed hollow shell (1 component, 0 tunnels, 1 cavity): chi = 1 - 0 + 1 = +2
    shell = np.zeros((12, 12, 12), dtype=np.uint8)
    shell[2:10, 2:10, 2:10] = 1
    shell[4:8, 4:8, 4:8] = 0
    assert euler_characteristic_3d(shell, voxel_size=1.0)["euler_characteristic"] == 2.0

    # 4. EDT pore radius quantiles (including degenerate all-solid and all-pore edge cases)
    edt_block = pore_throat_radius_metrics(block, voxel_size=1.5)
    assert edt_block["pore_radius_p90_um"] >= edt_block["pore_radius_median_um"] > 0.0
    assert pore_throat_radius_metrics(np.ones((8, 8, 8), dtype=np.uint8))["pore_radius_p90_um"] == 0.0
    edt_empty = pore_throat_radius_metrics(np.zeros((8, 8, 8), dtype=np.uint8), voxel_size=1.0)
    assert math.isfinite(edt_empty["pore_radius_p90_um"]) and edt_empty["pore_radius_p90_um"] > 0.0

    # 5. Ensemble Wasserstein-1 and KS distribution distances
    real_recs = [{"solid_fraction": 0.10 + 0.01 * i, "ssa_total_cc": 0.20} for i in range(8)]
    same_dist = compute_ensemble_distribution_distances(real_recs, real_recs)
    assert same_dist["solid_fraction"]["wasserstein_1"] == pytest.approx(0.0, abs=1e-12)
    assert same_dist["solid_fraction"]["ks_stat"] == pytest.approx(0.0, abs=1e-12)

    shifted_recs = [{"solid_fraction": 0.22, "ssa_total_cc": 0.35} for _ in range(8)]
    diff_dist = compute_ensemble_distribution_distances(real_recs, shifted_recs)
    assert diff_dist["solid_fraction"]["wasserstein_1"] > 0.07
    assert diff_dist["solid_fraction"]["ks_stat"] > 0.8


def test_openimpala_transport_constriction_factor_and_rev_study(tmp_path: Path) -> None:
    """Verify continuum constriction factor beta_Z < 1 on bottlenecked throats and non-percolating safety."""
    # 1. Uniform straight Z-pore channel: beta_Z == 1.0
    uniform = np.ones((20, 16, 16), dtype=np.uint8)
    uniform[:, 3:13, 3:13] = 0  # Constant 10x10 straight pore duct along Z
    phys_u = evaluate_volume_physics(uniform, voxel_size=1.0)
    oi_u = compute_openimpala_volume_transport(
        uniform, voxel_size_um=1.0, num_phases=2, compute_homogenized_tensor=True
    )
    rec_u = compute_transport_and_fom_record(phys_u, oi_u["lookup_summary"])  # type: ignore[arg-type]
    assert phys_u["geodesic_tortuosity_z"] == pytest.approx(1.0, abs=1e-6)
    assert rec_u["constriction_factor_z"] == pytest.approx(1.0, abs=0.03)
    assert oi_u["phases"]["pore_ionic"]["homogenized_d_zz"] > 0.0  # type: ignore[index]

    # 2. Constricted bottlenecked Z-pore channel: wide chamber pinched to a 2x2 throat at mid-Z
    constricted = np.ones((20, 16, 16), dtype=np.uint8)
    constricted[:, 2:14, 2:14] = 0  # Wide 12x12 pore chamber
    constricted[8:12, :, :] = 1
    constricted[8:12, 7:9, 7:9] = 0  # Narrow 2x2 throat bottleneck
    phys_c = evaluate_volume_physics(constricted, voxel_size=1.0)
    oi_c = compute_openimpala_volume_transport(constricted, voxel_size_um=1.0, num_phases=2)
    rec_c = compute_transport_and_fom_record(phys_c, oi_c["lookup_summary"])  # type: ignore[arg-type]
    assert rec_c["constriction_factor_z"] < 0.45

    # 3. Pinched non-percolating Z pore + 3-phase composite solid conduction
    pinched = np.zeros((16, 16, 16), dtype=np.uint8)
    pinched[:, 5:11, 5:11] = 1
    pinched[6:10, 5:11, 5:11] = 2  # Semiconducting coating segment in series with carbon
    pinched_closed_pore = pinched.copy()
    pinched_closed_pore[8, :, :] = 1  # Completely blocks Z pore percolation

    oi_p = compute_openimpala_volume_transport(
        pinched_closed_pore,
        voxel_size_um=1.0,
        num_phases=3,
        include_hetero_solid=True,
        coating_rel_conductivity=0.05,
    )
    phases_p = oi_p["phases"]  # type: ignore[index]
    assert phases_p["pore_ionic"]["d_eff_z"] == 0.0
    assert phases_p["pore_ionic"]["percolates_z"] == 0.0
    assert phases_p["solid_hetero_electronic"]["d_eff_z"] > 0.0

    # 4. REV study wrapper
    rev_out = tmp_path / "rev"
    rev_summary = run_openimpala_rev_convergence(
        uniform, sizes=[10, 16], phase=0, num_samples=2, voxel_size_um=1.0, out_dir=rev_out
    )
    assert (rev_out / "rev_convergence_summary.json").is_file()
    assert len(rev_summary["sizes"]) == 2  # type: ignore[arg-type]


def test_end_to_end_run_openimpala_cli_and_newman_asr(tmp_path: Path) -> None:
    """Verify evaluate_beamtime_multiplier.py --run-openimpala feeds optimize_clamping_pressure.py."""
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    exp_name = "oi_cli_exp"
    version = "v_oi_test"
    patch_sz = 16

    states = [
        ("tio2_low", 50.0, 3),
        ("tio2_mid", 150.0, 5),
        ("tio2_high", 300.0, 7),
    ]
    scans_manifest = []
    for st, kpa, half_w in states:
        vol = np.zeros((2, patch_sz, patch_sz, patch_sz), dtype=np.uint8)
        c = patch_sz // 2
        vol[:, :, c - half_w // 2 : c + half_w // 2 + 1, c - half_w // 2 : c + half_w // 2 + 1] = 1
        vol[:, :, c - half_w // 2, c - half_w // 2 : c + half_w // 2 + 1] = 2

        val_dir = data_root / st / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        np.save(val_dir / "patches.npy", vol)
        tall = np.repeat(vol, 2, axis=1)
        np.save(val_dir / "tall_patches.npy", tall)

        samp_dir = out_root / "results" / exp_name / "samples" / st
        samp_dir.mkdir(parents=True, exist_ok=True)
        np.save(samp_dir / "generated_samples.npy", vol)

        scans_manifest.append(
            {
                "state": st,
                "patch_dir_name": st,
                "material": "tio2_anode",
                "beamline": "DLS_I12",
                "load_mv_per_v": kpa / 10000.0,
                "stress_kpa": kpa,
                "effective_voxel_size_um": 1.0,
                "num_phases": 3,
                "patch_size": patch_sz,
                "is_repeat_or_secondary": False,
            }
        )

    (out_root / "results" / exp_name / "samples" / "sampling_provenance.json").write_text(
        json.dumps({"weight_source": "EMA"})
    )
    ckpt_dir = out_root / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tiny_model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=50.0,
        load_max=300.0,
        use_flash_attention=False,
        num_phases=3,
        num_materials=3,
        ref_voxel_size_um=1.0,
    ).eval()
    torch.save(
        {
            "epoch": 0,
            "model": tiny_model.state_dict(),
            "config": {
                "patch_size": patch_sz,
                "channels": [8, 16],
                "attention_levels": [False, True],
                "num_res_blocks": 1,
                "norm_groups": 4,
                "load_min": 50.0,
                "load_max": 300.0,
                "conditioning_mode": "multiscale_fourier",
                "num_phases": 3,
                "num_materials": 3,
                "ref_voxel_size_um": 1.0,
                "use_stress_kpa": True,
                "timesteps": 100,
                "schedule": "zero_snr_scaled_linear",
                "prediction_type": "v_prediction",
            },
        },
        ckpt_dir / "best_val.pth",
    )
    manifest_path = data_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps({"version": version, "scans": scans_manifest}, indent=2))

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff={exp_name}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_sz),
            "--use-stress-kpa",
            "--include-lagrangian-slab",
            "--virtual-steps",
            "4",
            "--allow-cpu",
            "--max-openimpala-per-method",
            "1",
            "--run-openimpala",
        ],
        check=True,
        env=_subprocess_env(),
    )

    bench_dir = out_root / "results" / exp_name / "beamtime_benchmark"
    assert (bench_dir / "openimpala_transport_parity.csv").is_file()
    report = json.loads((bench_dir / "beamtime_multiplier_report.json").read_text())
    assert "distribution_distances" in report["tio2_mid"]["methods"]["porousdiff"]
    assert "lagrangian_slab" in report["tio2_mid"]["methods"]
    assert "lagrangian_slab_diagnostics" in report["tio2_mid"]

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "optimize_clamping_pressure.py"),
            "--experiments",
            f"porousdiff={exp_name}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_sz),
            "--use-stress-kpa",
            "--dense-sweep-points",
            "3",
            "--dense-samples-per-load",
            "1",
            "--steps",
            "3",
            "--allow-cpu",
            "--bootstrap",
            "20",
        ],
        check=True,
        env=_subprocess_env(),
    )

    opt_dir = out_root / "results" / exp_name / "clamping_optimization"
    opt_json = json.loads((opt_dir / "optimal_clamping_summary.json").read_text())
    mat_opt = opt_json["tio2_anode"]["porousdiff"]
    assert mat_opt["estimator_mode"] == "openimpala"
    assert opt_json["tio2_anode"]["porousdiff_dense"]["estimator_mode"] == "openimpala"
    assert mat_opt["newman_asr"]["cell_architecture"] == "static_swagelok_coin_pouch"
    assert mat_opt["newman_asr"]["kinetic_area_convention"] == "cauchy_crofton_2_3"
    assert all(math.isfinite(pt["constriction_factor_z"]) for pt in mat_opt["discrete_points"])

    # Direct unit check of Cauchy-Crofton default vs raw option
    dummy_rec_cc = compute_transport_and_fom_record(
        {"solid_fraction": 0.2, "porosity": 0.8, "ssa_total": 0.30, "geodesic_tortuosity_z": 1.2}
    )
    dummy_rec_raw = compute_transport_and_fom_record(
        {"solid_fraction": 0.2, "porosity": 0.8, "ssa_total": 0.30, "geodesic_tortuosity_z": 1.2},
        use_cauchy_crofton_ssa=False,
    )
    assert dummy_rec_cc["active_wired_interface"] == pytest.approx(
        dummy_rec_raw["active_wired_interface"] * CAUCHY_CROFTON_3D_FACTOR, rel=1e-6
    )
    res_direct = find_optimal_clamping_pressure({50.0: [dummy_rec_cc], 150.0: [dummy_rec_cc]}, n_bootstrap=0)
    assert res_direct["newman_asr"]["kinetic_area_convention"] == "cauchy_crofton_2_3"


def test_lagrangian_slab_virtual_compression_and_cli(tmp_path: Path) -> None:
    """Verify virtual_compress_lagrangian_slab shrinks Z height, conserves mass, and runs via CLI."""
    torch.manual_seed(0)
    patch_sz = 16
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=50.0,
        load_max=300.0,
        use_flash_attention=False,
        num_phases=3,
        num_materials=3,
        ref_voxel_size_um=1.0,
    ).eval()
    sched = build_noise_scheduler("ddim", 100, "zero_snr_scaled_linear", "v_prediction")

    # Tall uncompressed 3-phase slab (24, 16, 16) with distinct XY pillars
    slab = np.zeros((24, patch_sz, patch_sz), dtype=np.uint8)
    slab[:, 3:7, 3:7] = 1
    slab[:, 3:7, 7] = 2
    slab[:, 10:13, 9:13] = 1

    comp_vol, info = virtual_compress_lagrangian_slab(
        model=model,
        alphas_cumprod=sched.alphas_cumprod,
        source_volume=slab,
        target_load=200.0,
        patch_size=patch_sz,
        compression_ratio=0.75,
        num_steps=6,
        refine_strength=0.35,
        tile_batch_size=2,
        material_id=1,
        seed=7,
    )
    assert comp_vol.shape == (18, patch_sz, patch_sz)
    assert info["compressed_depth"] == 18.0
    assert info["mirror_tiled_fraction"] == 0.0
    assert info["mass_conservation_rel_error"] < 0.02

    fid = compute_virtual_compression_fidelity(slab, comp_vol)
    assert math.isfinite(float(fid["paired_xy_column_corr"]))
    assert fid["is_near_identity_copy"] is False

    # End-to-end CLI check of virtual_compress_scan.py --mode lagrangian_slab
    ckpt_dir = tmp_path / "checkpoints" / "lag_exp"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_val.pth"
    torch.save(
        {
            "epoch": 0,
            "model": model.state_dict(),
            "config": {
                "patch_size": patch_sz,
                "channels": [8, 16],
                "attention_levels": [False, True],
                "num_res_blocks": 1,
                "norm_groups": 4,
                "load_min": 50.0,
                "load_max": 300.0,
                "conditioning_mode": "multiscale_fourier",
                "num_phases": 3,
                "num_materials": 3,
                "ref_voxel_size_um": 1.0,
                "use_stress_kpa": True,
                "timesteps": 100,
                "schedule": "zero_snr_scaled_linear",
                "prediction_type": "v_prediction",
            },
        },
        ckpt_path,
    )
    in_npy = tmp_path / "tall_slab.npy"
    np.save(in_npy, slab)
    vc_out = tmp_path / "vc_lag_out"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "virtual_compress_scan.py"),
            "--input",
            str(in_npy),
            "--experiment",
            "lag_exp",
            "--output-root",
            str(tmp_path),
            "--out-dir",
            str(vc_out),
            "--mode",
            "lagrangian_slab",
            "--source-load",
            "50.0",
            "--target-loads",
            "200.0",
            "--target-compression-ratios",
            "0.75",
            "--material",
            "tio2_anode",
            "--steps",
            "6",
            "--allow-cpu",
        ],
        check=True,
        env=_subprocess_env(),
    )
    vc_summary = json.loads((vc_out / "virtual_compression_summary.json").read_text())
    assert vc_summary["mode"] == "lagrangian_slab"
    assert vc_summary["lagrangian_slab_diagnostics"]["200"]["compressed_depth"] == 18.0


def test_redteam_zero_heldout_porosity_leakage_in_lagrangian_slab(tmp_path: Path) -> None:
    """Prove lagrangian_slab on held-out loads is strictly invariant to held-out real_cache solid_fraction."""
    patch_sz = 12
    version = "v_noleak"
    exp_name = "noleak_exp"
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"

    ckpt_dir = out_root / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=50.0,
        load_max=300.0,
        use_flash_attention=False,
        num_phases=2,
        num_materials=1,
        ref_voxel_size_um=1.0,
    ).eval()
    torch.save(
        {
            "epoch": 0,
            "model": model.state_dict(),
            "config": {
                "patch_size": patch_sz,
                "channels": [8, 16],
                "attention_levels": [False, True],
                "num_res_blocks": 1,
                "norm_groups": 4,
                "load_min": 50.0,
                "load_max": 300.0,
                "conditioning_mode": "multiscale_fourier",
                "num_phases": 2,
                "num_materials": 1,
                "ref_voxel_size_um": 1.0,
                "use_stress_kpa": True,
                "timesteps": 50,
                "schedule": "zero_snr_scaled_linear",
                "prediction_type": "v_prediction",
            },
        },
        ckpt_dir / "best_val.pth",
    )

    rng = np.random.default_rng(0)
    low_vol = (rng.random((1, patch_sz, patch_sz, patch_sz)) < 0.12).astype(np.uint8)
    high_vol = (rng.random((1, patch_sz, patch_sz, patch_sz)) < 0.35).astype(np.uint8)

    for st, v in (("v_low", low_vol), ("v_high", high_vol)):
        vdir = data_root / st / "val"
        vdir.mkdir(parents=True, exist_ok=True)
        np.save(vdir / "patches.npy", v)
        np.save(vdir / "tall_patches.npy", np.repeat(v, 2, axis=1))

    mid_dir = data_root / "v_mid" / "val"
    mid_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": version,
                "patch_size": patch_sz,
                "scans": [
                    {"state": "v_low", "patch_dir_name": "v_low", "material": "virgin_carbon", "beamline": "DLS_I12", "stress_kpa": 50.0, "effective_voxel_size_um": 1.0, "num_phases": 2},
                    {"state": "v_mid", "patch_dir_name": "v_mid", "material": "virgin_carbon", "beamline": "DLS_I12", "stress_kpa": 175.0, "effective_voxel_size_um": 1.0, "num_phases": 2},
                    {"state": "v_high", "patch_dir_name": "v_high", "material": "virgin_carbon", "beamline": "DLS_I12", "stress_kpa": 300.0, "effective_voxel_size_um": 1.0, "num_phases": 2},
                ],
            }
        )
    )

    # Run 1: held-out v_mid has sparse real solid fraction (0.0625)
    mid_sparse = np.zeros((1, patch_sz, patch_sz, patch_sz), dtype=np.uint8)
    mid_sparse[:, :, 3:6, 3:6] = 1
    np.save(mid_dir / "patches.npy", mid_sparse)
    bench_a = tmp_path / "bench_a"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff={exp_name}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--out-dir",
            str(bench_a),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_sz),
            "--use-stress-kpa",
            "--heldout-states",
            "v_mid",
            "--include-lagrangian-slab",
            "--virtual-steps",
            "3",
            "--allow-cpu",
        ],
        check=True,
        env=_subprocess_env(),
    )

    # Run 2: mutate held-out v_mid real XCT to massive solid fraction (0.75!)
    mid_dense = np.ones((1, patch_sz, patch_sz, patch_sz), dtype=np.uint8)
    mid_dense[:, :, :3, :] = 0
    np.save(mid_dir / "patches.npy", mid_dense)
    bench_b = tmp_path / "bench_b"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluation" / "evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff={exp_name}",
            "--data-root",
            str(data_root),
            "--output-root",
            str(out_root),
            "--out-dir",
            str(bench_b),
            "--manifest-json",
            str(manifest_path),
            "--version",
            version,
            "--patch-size",
            str(patch_sz),
            "--use-stress-kpa",
            "--heldout-states",
            "v_mid",
            "--include-lagrangian-slab",
            "--virtual-steps",
            "3",
            "--allow-cpu",
        ],
        check=True,
        env=_subprocess_env(),
    )

    rep_a = json.loads((bench_a / "beamtime_multiplier_report.json").read_text())["v_mid"]
    rep_b = json.loads((bench_b / "beamtime_multiplier_report.json").read_text())["v_mid"]

    # Oracle baseline changes drastically when held-out real scan changes, whereas lagrangian_slab is 100% invariant!
    assert rep_a["methods"]["affine_z_squash"]["uses_heldout_oracle_porosity"] is True
    assert rep_a["methods"]["lagrangian_slab"]["uses_heldout_oracle_porosity"] is False
    assert rep_a["lagrangian_slab_diagnostics"]["target_source"] == "non_heldout_interp"
    sf_lag_a = rep_a["methods"]["lagrangian_slab"]["metrics"]["solid_fraction"]["mean"]
    sf_lag_b = rep_b["methods"]["lagrangian_slab"]["metrics"]["solid_fraction"]["mean"]
    sf_aff_a = rep_a["methods"]["affine_z_squash"]["metrics"]["solid_fraction"]["mean"]
    sf_aff_b = rep_b["methods"]["affine_z_squash"]["metrics"]["solid_fraction"]["mean"]
    assert sf_lag_a == pytest.approx(sf_lag_b, abs=1e-9)
    assert abs(sf_aff_b - sf_aff_a) > 0.30


def test_redteam_per_realization_openimpala_bootstrap_and_static_asr(tmp_path: Path) -> None:
    """Verify load_openimpala_records preserves per-patch variance and static-cell Newman ASR has zero pump penalty."""
    import csv
    from porousdiff.optimization import NEWMAN_REGIMES, load_openimpala_records

    # 1. Static Swagelok/coin/pouch primary regimes have zero convective pumping penalty
    for reg_key in ("kinetics_limited", "balanced", "mass_transport_limited"):
        assert NEWMAN_REGIMES[reg_key]["pump_penalty_ohm_cm2_um2_cm"] == 0.0
    assert NEWMAN_REGIMES["flow_pumped_sensitivity"]["pump_penalty_ohm_cm2_um2_cm"] > 0.0

    # 2. Multi-stem OpenImpala CSV preserves distinct per-realization diffusivities (non-collapsed bootstrap CIs)
    oi_csv = tmp_path / "parity_multi.csv"
    rows = []
    for lv, state, pore_vals, sig_vals in (
        (50.0, "s_low", [0.45, 0.55, 0.62], [0.015, 0.025, 0.035]),
        (175.0, "s_mid", [0.24, 0.31, 0.38], [0.11, 0.14, 0.17]),
        (300.0, "s_high", [0.02, 0.06, 0.11], [0.15, 0.16, 0.18]),
    ):
        for idx, (pv, sv) in enumerate(zip(pore_vals, sig_vals), start=1):
            rows.append(
                {
                    "job_id": f"real__{state}__real_{idx:03d}__pore_ionic",
                    "method": "real",
                    "state": state,
                    "transport_phase": "pore_ionic",
                    "d_eff_z": str(pv),
                }
            )
            rows.append(
                {
                    "job_id": f"real__{state}__real_{idx:03d}__solid_hetero_electronic",
                    "method": "real",
                    "state": state,
                    "transport_phase": "solid_hetero_electronic",
                    "d_eff_z": str(sv),
                }
            )
    with oi_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "method", "state", "transport_phase", "d_eff_z"])
        writer.writeheader()
        writer.writerows(rows)

    rec_map = load_openimpala_records(oi_csv)
    assert len(rec_map[("real", "s_mid")]) == 3
    assert [r["pore_d_eff_z"] for r in rec_map[("real", "s_mid")]] == pytest.approx([0.24, 0.31, 0.38])

    recs_by_load: dict[float, list[dict[str, float]]] = {}
    for lv, st, sf in ((50.0, "s_low", 0.12), (175.0, "s_mid", 0.20), (300.0, "s_high", 0.32)):
        recs_by_load[lv] = [
            compute_transport_and_fom_record(
                {
                    "solid_fraction": sf,
                    "porosity": 1.0 - sf,
                    "ssa_pore_coating_cc": 0.18,
                    "ssa_pore_coating": 0.27,
                    "wired_coating_fraction": 0.9,
                    "geodesic_tortuosity_z": 1.15,
                },
                openimpala_metrics=oi_r,
            )
            for oi_r in rec_map[("real", st)]
        ]
    opt_res = find_optimal_clamping_pressure(recs_by_load, n_bootstrap=120, seed=11)
    ci_lo, ci_hi = opt_res["newman_asr"]["optimal_load_asr_ci95"]
    # Because identical phys_metrics were passed across the 3 patches at each load, any nonzero CI width
    # comes PURELY from preserving per-realization OpenImpala PDE variance!
    assert ci_hi > ci_lo


def test_redteam_load_only_cfg_physics_loss_and_bundle_audit(tmp_path: Path) -> None:
    """Verify factored load_only CFG, cond_mask-gated uncoated-safe physics loss, and Gate 0 bundle audit."""
    from porousdiff.losses import PhysicsInformedDiffusionLoss
    from scripts.evaluation.audit_bundle_readiness import audit_manifest_bundle

    # 1. Factored load_only CFG: zero dead grads, load-invariant at mask=0, material-distinct at mask=0
    torch.manual_seed(4)
    model = ContinuousLoadConditionedUNet(
        channels=(8, 16),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=4,
        load_min=20.0,
        load_max=400.0,
        num_phases=3,
        num_materials=3,
        ref_voxel_size_um=2.0,
        cfg_mode="load_only",
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        opt.zero_grad()
        x = torch.randn(4, 2, 12, 12, 12)
        target = torch.randn_like(x)
        t = torch.tensor([10, 200, 600, 950], dtype=torch.long)
        lds = torch.tensor([20.0, 150.0, 300.0, 400.0], dtype=torch.float32)
        mats = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        voxs = torch.tensor([1.6, 3.2, 3.2, 0.8], dtype=torch.float32)
        msk = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        out = model(x, t, lds, cond_mask=msk, material_ids=mats, voxel_sizes=voxs)
        torch.nn.functional.mse_loss(out, target).backward()
        if step < 2:
            opt.step()

    dead = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
    ]
    assert dead == [], f"Dead parameters in cfg_mode='load_only': {dead}"

    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 2, 12, 12, 12)
        t1 = torch.tensor([300], dtype=torch.long)
        zero_m = torch.zeros(1, dtype=torch.float32)
        vx = torch.tensor([3.2], dtype=torch.float32)
        # At cond_mask == 0, load changes have zero effect, while material_id stays anchored!
        u_low = model(x1, t1, torch.tensor([25.0]), cond_mask=zero_m, material_ids=torch.tensor([1]), voxel_sizes=vx)
        u_high = model(x1, t1, torch.tensor([390.0]), cond_mask=zero_m, material_ids=torch.tensor([1]), voxel_sizes=vx)
        u_virg = model(x1, t1, torch.tensor([25.0]), cond_mask=zero_m, material_ids=torch.tensor([0]), voxel_sizes=vx)
        assert torch.allclose(u_low, u_high, atol=1e-6)
        assert not torch.allclose(u_low, u_virg)

    # 2. PhysicsInformedDiffusionLoss masks conditional terms when cond_mask == 0 and handles uncoated Virgin patches
    sched = build_noise_scheduler("ddpm", 100, "zero_snr_scaled_linear", "v_prediction")
    crit = PhysicsInformedDiffusionLoss(sched.alphas_cumprod, prediction_type="v_prediction")
    # Uncoated 2-phase Virgin patch encoded in 2 channels: c_coating == -1 everywhere
    x0_virg = torch.full((2, 2, 8, 8, 8), -1.0)
    x0_virg[:, 0, 2:5, 2:5, 2:5] = 1.0
    t_b = torch.tensor([20, 20], dtype=torch.long)
    noise_b = torch.randn_like(x0_virg)
    xt_b = sched.add_noise(x0_virg, noise_b, t_b)
    v_targ = sched.get_velocity(x0_virg, noise_b, t_b)
    _, stats_uncond = crit(
        v_targ + 0.2 * torch.randn_like(v_targ),
        v_targ,
        xt_b,
        x0_virg,
        t_b,
        cond_mask=torch.zeros(2),
    )
    assert stats_uncond["loss_loading"] == 0.0
    assert stats_uncond["loss_interface"] == 0.0

    # 3. Gate 0 bundle readiness audit catches non-monotonic solid fraction
    b_root = tmp_path / "audit_bundle"
    for st, sf_fill in (("c_50", 6), ("c_200", 2)):  # Intentional non-monotonic drop!
        for sp in ("train", "val"):
            sdir = b_root / st / sp
            sdir.mkdir(parents=True, exist_ok=True)
            arr = np.zeros((1, 8, 8, 8), dtype=np.uint8)
            arr[:, :, :sf_fill, :sf_fill] = 1
            np.save(sdir / "patches.npy", arr)
    bad_manifest = b_root / "bundle_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "version": "v_aud",
                "patch_size": 8,
                "scans": [
                    {"state": "c_50", "patch_dir_name": "c_50", "material": "virgin_carbon", "beamline": "DLS_I12", "stress_kpa": 50.0, "effective_voxel_size_um": 1.0, "num_phases": 2},
                    {"state": "c_200", "patch_dir_name": "c_200", "material": "virgin_carbon", "beamline": "DLS_I12", "stress_kpa": 200.0, "effective_voxel_size_um": 1.0, "num_phases": 2},
                ],
            }
        )
    )
    rep_bad = audit_manifest_bundle(bad_manifest, data_root=b_root, patch_size=8)
    assert rep_bad["passed"] is False
    assert any("Non-monotonic solid fraction" in e for e in rep_bad["errors"])  # type: ignore[union-attr]

