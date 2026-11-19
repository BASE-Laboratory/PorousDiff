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
