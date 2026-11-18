from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from porousdiff.metrics import (
    connectivity_and_percolation,
    directional_two_point_correlation,
    evaluate_volume_physics,
    export_openimpala_tiff,
    geodesic_pore_tortuosity,
    solid_fraction,
    specific_surface_area,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_physics_metrics_exact_identities(tmp_path: Path) -> None:
    # 1. Straight open pore channel vs baffled tortuous pore channel vs blocked wall
    open_vol = np.zeros((24, 16, 16), dtype=np.uint8)
    assert geodesic_pore_tortuosity(open_vol, axis=0) == pytest.approx(1.0)

    baffled = np.zeros((21, 16, 16), dtype=np.uint8)
    baffled[7, :12, :] = 1
    baffled[14, 4:, :] = 1
    tau_baffled = geodesic_pore_tortuosity(baffled, axis=0)
    assert tau_baffled > 1.15

    blocked = np.zeros((16, 16, 16), dtype=np.uint8)
    blocked[8, :, :] = 1
    assert np.isnan(geodesic_pore_tortuosity(blocked, axis=0))
    conn_blocked = connectivity_and_percolation(blocked)
    assert conn_blocked["pore_percolates_z"] == 0.0
    assert conn_blocked["solid_connected_fraction"] == pytest.approx(1.0)

    # 2. Two-point correlation S_2(0) == solid_fraction
    rng = np.random.default_rng(99)
    rand_vol = (rng.random((24, 24, 24)) < 0.27).astype(np.uint8)
    sf = solid_fraction(rand_vol)
    s2 = directional_two_point_correlation(rand_vol, max_lag=8)
    assert s2["s2_z"][0] == pytest.approx(sf, abs=1e-6)
    assert s2["s2_y"][0] == pytest.approx(sf, abs=1e-6)
    assert s2["s2_x"][0] == pytest.approx(sf, abs=1e-6)

    ssa = specific_surface_area(rand_vol, voxel_size=2.0)
    assert ssa["ssa_total"] > 0.0

    full = evaluate_volume_physics(baffled, voxel_size=1.5)
    assert full["porosity"] == pytest.approx(1.0 - full["solid_fraction"])

    # 3. OpenImpala multi-page TIFF export
    tiff_path = export_openimpala_tiff(baffled, tmp_path / "openimpala.tif")
    reloaded = tifffile.imread(tiff_path)
    assert reloaded.shape == baffled.shape
    assert reloaded.dtype == np.uint8
    assert np.array_equal(reloaded, baffled)


def _stage_mini_xct_datasets(raw_root: Path) -> list[str]:
    """Create 3 realistic 128-slice x 96 x 96 uint16 XCT stacks (using real XCT crops when present)."""
    real_dir = Path("/tmp/compression_data")
    real_map = {
        "0p008mVV": real_dir / "sample_crbon_virgin_1_0p008mVV.npz",
        "0p028mVV": real_dir / "sample_crbon_virgin_1_0p028mVV.npz",
        "0p053mVV": real_dir / "sample_crbon_virgin_1_0p053mVV.npz",
    }
    have_real = all(p.is_file() for p in real_map.values())
    rng = np.random.default_rng(2026)

    yy, xx = np.indices((96, 96))
    in_circle = ((yy - 48.0) ** 2 + (xx - 48.0) ** 2) <= (42.0**2)

    for idx, (state, npz_path) in enumerate(real_map.items()):
        ds_dir = raw_root / state / "rec_16bit_Paganin"
        ds_dir.mkdir(parents=True, exist_ok=True)
        if have_real:
            with np.load(npz_path) as d:
                # Downsample XY 192->96 to run fast on CPU while preserving real synchrotron intensities
                vol = d["crop_3d"][:, ::2, ::2].copy()  # (128, 96, 96)
                if state == "0p028mVV":
                    # z >= 15 in 0p028 crop_3d hits the bullseye seam; bounce-tile clean z < 15
                    clean = vol[:15]
                    pair = np.concatenate([clean[:-1], clean[:0:-1]], axis=0)
                    vol = np.tile(pair, (5, 1, 1))[:128]
        else:
            # Synthetic fallback for standalone CI
            fiber_p = [0.06, 0.19, 0.31][idx]
            vol = (30_000.0 + rng.normal(0.0, 400.0, size=(128, 96, 96))).astype(np.float32)
            vol[rng.random((128, 96, 96)) < fiber_p] += 9500.0
            vol = vol.clip(0, 65535).astype(np.uint16)

        # Surround circular sample punch with gray exterior air (~30,000) like real 2560^2 XCT slices
        for z in range(vol.shape[0]):
            slc = vol[z].copy()
            slc[~in_circle] = 30_100
            tifffile.imwrite(ds_dir / f"slice_{z:04d}.tif", slc)

    return list(real_map.keys())


def test_end_to_end_cli_pipeline_with_heldout_load_interpolation(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    proc_root = tmp_path / "data" / "processed"
    res_root = tmp_path / "results"
    states = _stage_mini_xct_datasets(raw_root)

    # 1. Segment all 3 compression states consistently
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocessing/segment_carbon_xct.py"),
            "--datasets",
            *states,
            "--data-root",
            str(raw_root),
            "--output-root",
            str(proc_root),
            "--version",
            "e2e_v1",
            "--delta-threshold",
            "5500",
            "--sigma",
            "16",
            "--roi-radius-fraction",
            "0.82",
            "--min-interval-length",
            "32",
            "--seam-margin",
            "2",
        ],
        check=True,
    )

    mean_sfs = []
    for st in states:
        seg_dir = proc_root / f"{st}_binary_corrected_e2e_v1"
        valid_meta = json.loads((seg_dir / "valid_slices.json").read_text())
        assert valid_meta["valid_slice_count"] >= 96
        report = json.loads((seg_dir / "segmentation_report.json").read_text())
        mean_sfs.append(report["clean_solid_fraction_mean"])
    assert mean_sfs[0] < mean_sfs[1] < mean_sfs[2]

    # 2. Extract 32^3 Z-stratified patches with buffered train/val split
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocessing/extract_carbon_3d_patches.py"),
            "--datasets",
            *states,
            "--data-root",
            str(proc_root),
            "--output-root",
            str(proc_root),
            "--version",
            "e2e_v1",
            "--patch-size",
            "32",
            "--stride-xy",
            "12",
            "--stride-z",
            "8",
            "--max-patches",
            "20",
            "--roi-radius-fraction",
            "0.82",
            "--no-individual-npy",
        ],
        check=True,
    )

    # 3. Train on outer loads ONLY (0.008 and 0.053), holding out intermediate 0.028 mV/V!
    exp = "e2e_cond_32"
    train_cmd_base = [
        sys.executable,
        str(REPO_ROOT / "scripts/training/train_carbon_conditional_96.py"),
        "--experiment",
        exp,
        "--data-root",
        str(proc_root),
        "--output-root",
        str(res_root),
        "--version",
        "e2e_v1",
        "--states",
        "0p008mVV=0.008",
        "0p053mVV=0.053",
        "--patch-size",
        "32",
        "--channels",
        "16,32,64",
        "--num-res-blocks",
        "1",
        "--norm-groups",
        "8",
        "--batch-size",
        "2",
        "--grad-accum",
        "1",
        "--amp",
        "off",
        "--allow-cpu",
        "--max-steps",
        "3",
        "--max-val-steps",
        "1",
    ]
    subprocess.run([*train_cmd_base, "--epochs", "1"], check=True)
    # Verify exact checkpoint resumption to epoch 2
    subprocess.run([*train_cmd_base, "--epochs", "2", "--resume"], check=True)

    losses_lines = (res_root / "results" / exp / "losses.csv").read_text().strip().splitlines()
    assert len(losses_lines) == 3  # header + 2 epochs
    assert (res_root / "checkpoints" / exp / "best_val.pth").is_file()

    # 4. Sample all 3 loads (including UNSEEN interpolated 0.028 mV/V) + continuous sweep + OpenImpala TIFF
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/sample_carbon_conditional_96.py"),
            "--experiment",
            exp,
            "--output-root",
            str(res_root),
            "--sampler",
            "ddim",
            "--steps",
            "5",
            "--guidance-scale",
            "1.5",
            "--num-samples",
            "2",
            "--batch-size",
            "2",
            "--eval-states",
            "0p008mVV=0.008",
            "0p028mVV=0.028",
            "0p053mVV=0.053",
            "--sweep-loads",
            "0.008,0.028,0.053",
            "--export-openimpala",
            "--allow-cpu",
        ],
        check=True,
    )

    exp_res_dir = res_root / "results" / exp
    assert (exp_res_dir / "samples" / "0p028mVV" / "openimpala" / "sample_001.tif").is_file()
    assert (exp_res_dir / "samples" / "continuous_sweep" / "sweep_0p0280mVV.npy").is_file()
    assert (exp_res_dir / "figures" / "compression_sweep_orthoviews.png").is_file()

    # 5. Run full physics evaluation comparing real vs generated across seen & held-out loads
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/evaluate_carbon_physics.py"),
            "--experiment",
            exp,
            "--data-root",
            str(proc_root),
            "--output-root",
            str(res_root),
            "--version",
            "e2e_v1",
            "--patch-size",
            "32",
            "--states",
            "0p008mVV=0.008",
            "0p028mVV=0.028",
            "0p053mVV=0.053",
            "--heldout-states",
            "0p028mVV",
            "--export-real-openimpala",
            "--num-openimpala",
            "2",
        ],
        check=True,
    )

    eval_dir = exp_res_dir / "physics_validation"
    phys_report = json.loads((eval_dir / "physics_validation_report.json").read_text())
    assert set(phys_report.keys()) == {"0p008mVV", "0p028mVV", "0p053mVV"}
    assert phys_report["0p028mVV"]["is_heldout_interpolation"] is True
    assert phys_report["0p008mVV"]["is_heldout_interpolation"] is False
    assert (eval_dir / "physics_compression_curves.png").is_file()
    assert (eval_dir / "s2_correlation_comparison.png").is_file()
    assert (eval_dir / "openimpala_real" / "0p028mVV" / "real_val_001.tif").is_file()


def test_prepare_local_drive_bundle_cli_and_zip(tmp_path: Path) -> None:
    import zipfile

    drive_root = tmp_path / "dry_samples"
    bundle_dir = tmp_path / "porousdiff_upload_bundle"
    rng = np.random.default_rng(123)
    yy, xx = np.indices((64, 64))
    in_circle = ((yy - 32.0) ** 2 + (xx - 32.0) ** 2) <= (27.0**2)

    folders = [
        ("crbon_virgin_1_0p008mVV_", 0.08),
        ("crbon_virgin_1_0p015mVV_rep2_", 0.14),
        ("crbon_virgin_1_0p053mVV_", 0.30),
    ]
    for folder_name, fiber_p in folders:
        rec_dir = drive_root / folder_name / "rec_16bit_Paganin"
        rec_dir.mkdir(parents=True, exist_ok=True)
        vol = (30_000.0 + rng.normal(0.0, 350.0, size=(80, 64, 64))).astype(np.float32)
        vol[rng.random((80, 64, 64)) < fiber_p] += 9500.0
        vol_u16 = vol.clip(0, 65535).astype(np.uint16)
        for z in range(vol_u16.shape[0]):
            slc = vol_u16[z].copy()
            slc[~in_circle] = 30_100
            tifffile.imwrite(rec_dir / f"slice_{z:04d}.tif", slc)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocessing/prepare_local_drive_bundle.py"),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(bundle_dir),
            "--version",
            "bundle_v1",
            "--sigma",
            "12",
            "--roi-radius-fraction",
            "0.80",
            "--patch-size",
            "24",
            "--stride-xy",
            "12",
            "--stride-z",
            "8",
            "--max-patches",
            "10",
            "--min-interval-length",
            "24",
            "--seam-margin",
            "2",
            "--zip",
        ],
        check=True,
    )

    primary = json.loads((bundle_dir / "primary_states.json").read_text())
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    assert primary == {
        "crbon_virgin_1_0p008mVV": 0.008,
        "crbon_virgin_1_0p053mVV": 0.053,
    }
    assert manifest["repeat_and_secondary_states"] == {
        "crbon_virgin_1_0p015mVV_rep2": 0.015,
    }
    p_low = np.load(bundle_dir / "crbon_virgin_1_0p008mVV_binary_patches24_bundle_v1" / "train" / "patches.npy")
    p_high = np.load(bundle_dir / "crbon_virgin_1_0p053mVV_binary_patches24_bundle_v1" / "train" / "patches.npy")
    assert p_low.shape[1:] == (24, 24, 24)
    assert float(p_low.mean()) < float(p_high.mean())

    zip_path = tmp_path / "porousdiff_upload_bundle.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "primary_states.json" in names
        assert "bundle_manifest.json" in names

