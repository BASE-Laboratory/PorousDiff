from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from porousdiff.baselines import affine_z_compress_to_target, pillar_pore_deletion_compress
from porousdiff.data import decode_channels_to_phases, encode_phases_to_channels
from porousdiff.metrics import evaluate_multiphase_physics, export_openimpala_tiff
from porousdiff.models import ContinuousLoadConditionedUNet
from porousdiff.preprocessing import (
    CalibrationMetadata,
    calibrate_multiphase_thresholds,
    downsample_volume_rev,
    infer_material_name,
    make_circular_mask,
    segment_slice_multiphase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multiphase_segmentation_and_rev_binning() -> None:
    rng = np.random.default_rng(101)
    h, w = 128, 128
    roi = make_circular_mask((h, w), radius_fraction=0.80)

    # Synthetic 3-phase TiO2-on-carbon slice: pore (~30,000), carbon (~38,500), TiO2 coating (~49,500)
    raw = (30_000.0 + rng.normal(0.0, 300.0, size=(h, w))).astype(np.float32)
    u = rng.random((h, w))
    carbon_gt = roi & (u >= 0.80) & (u < 0.93)
    coat_gt = roi & (u >= 0.93)
    raw[carbon_gt] += 8_500.0
    raw[coat_gt] += 19_500.0
    raw_u16 = raw.clip(0, 65535).astype(np.uint16)

    d_carb, d_coat = calibrate_multiphase_thresholds([raw_u16], roi_mask=roi, sigma=16.0)
    assert 3_000.0 < d_carb < 12_000.0
    assert d_carb < d_coat < 18_500.0

    seg3, stats3 = segment_slice_multiphase(
        raw_u16, delta_carbon=d_carb, delta_coating=d_coat, roi_mask=roi, sigma=16.0
    )
    assert set(np.unique(seg3)) == {0, 1, 2}
    assert stats3["carbon_fraction"] == pytest.approx(float(carbon_gt[roi].mean()), abs=0.02)
    assert stats3["coating_fraction"] == pytest.approx(float(coat_gt[roi].mean()), abs=0.02)

    # Virgin 2-phase backward compatibility when delta_coating=None
    seg2, stats2 = segment_slice_multiphase(
        raw_u16, delta_carbon=d_carb, delta_coating=None, roi_mask=roi, sigma=16.0
    )
    assert set(np.unique(seg2)) == {0, 1}
    assert stats2["coating_fraction"] == 0.0

    # Topology- & volume-fraction-preserving 2x REV block binning on 3-phase 3D volume
    vol3 = np.zeros((64, 64, 64), dtype=np.uint8)
    vol3[4:60, 10::8, :] = 1  # Thin carbon fibers
    vol3[4:60, 11::16, ::2] = 2  # Attached active coating deposits
    binned = downsample_volume_rev(vol3, factor=2)
    assert binned.shape == (32, 32, 32)
    assert set(np.unique(binned)) == {0, 1, 2}
    assert float((binned >= 1).mean()) == pytest.approx(float((vol3 >= 1).mean()), abs=0.01)
    assert float((binned == 2).mean()) == pytest.approx(float((vol3 == 2).mean()), abs=0.01)

    calib = CalibrationMetadata(voxel_size_um=1.6, spatial_bin=2, kpa_per_mvv=8500.0)
    assert calib.effective_voxel_size_um == pytest.approx(3.2)
    assert calib.load_to_stress_kpa(0.020) == pytest.approx(170.0)
    assert infer_material_name("TiO2_felt_0p018mVV") == "tio2_anode"
    assert infer_material_name("CuHCF_cathode_0p033mVV") == "cuhcf_cathode"
    assert infer_material_name("crbon_virgin_1_0p008mVV") == "virgin_carbon"


def test_numerical_compression_baselines() -> None:
    rng = np.random.default_rng(7)
    uncomp = np.zeros((32, 24, 24), dtype=np.uint8)
    p = rng.random(uncomp.shape)
    uncomp[(p >= 0.86) & (p < 0.96)] = 1
    uncomp[p >= 0.96] = 2

    target_sf = 0.28
    aff = affine_z_compress_to_target(uncomp, target_solid_fraction=target_sf, output_depth=32)
    pil = pillar_pore_deletion_compress(
        uncomp, target_solid_fraction=target_sf, output_depth=32, rng=rng
    )

    assert aff.shape == (32, 24, 24)
    assert pil.shape == (32, 24, 24)
    assert set(np.unique(aff)).issubset({0, 1, 2})
    assert set(np.unique(pil)).issubset({0, 1, 2})
    assert float((aff >= 1).mean()) == pytest.approx(target_sf, abs=0.025)
    assert float((pil >= 1).mean()) == pytest.approx(target_sf, abs=0.04)
    assert np.any(aff == 2) and np.any(pil == 2)


def test_hierarchical_phase_encoding_and_multielectrode_unet() -> None:
    # 1. Lossless 3-phase hierarchical roundtrip {0, 1, 2} <-> [-1, +1]^2
    vol = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.uint8)
    enc = encode_phases_to_channels(vol, num_phases=3)
    assert enc.shape == (2, 2, 2, 2)
    dec = decode_channels_to_phases(enc, enforce_solid_support=True)
    assert np.array_equal(dec, vol)

    # Verify floating coating in empty pore (c_solid=-1, c_coat=+1) snaps cleanly to pore (0)
    impossible = torch.tensor([[[[-1.0]]], [[[+1.0]]]])  # [2, 1, 1, 1]
    assert int(decode_channels_to_phases(impossible, enforce_solid_support=True).item()) == 0

    # 2. Gradient flow across ALL parameters in MultiModalConditionEmbedder + 3-phase UNet
    torch.manual_seed(12)
    model = ContinuousLoadConditionedUNet(
        channels=(16, 32),
        attention_levels=(False, True),
        num_res_blocks=1,
        norm_num_groups=8,
        load_min=10.0,
        load_max=400.0,
        num_phases=3,
        num_materials=3,
        ref_voxel_size_um=3.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        opt.zero_grad()
        x = torch.randn(4, 2, 12, 12, 12)
        target = torch.randn_like(x)
        t = torch.tensor([15, 220, 610, 990], dtype=torch.long)
        loads = torch.tensor([10.0, 150.0, 320.0, 400.0], dtype=torch.float32)
        mats = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        voxs = torch.tensor([1.6, 3.2, 3.2, 5.0], dtype=torch.float32)
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)

        pred = model(x, t, loads, cond_mask=mask, material_ids=mats, voxel_sizes=voxs)
        assert pred.shape == x.shape
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        if step < 2:
            opt.step()

    dead = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
    ]
    assert dead == [], f"Parameters received zero gradient: {dead}"

    # Verify distinct material predictions & CFG
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, 2, 12, 12, 12)
        t1 = torch.tensor([350], dtype=torch.long)
        lv1 = torch.tensor([120.0], dtype=torch.float32)
        vx1 = torch.tensor([3.2], dtype=torch.float32)
        out_virgin = model.forward_cfg(
            x1, t1, lv1, guidance_scale=1.5, material_ids=torch.tensor([0]), voxel_sizes=vx1
        )
        out_tio2 = model.forward_cfg(
            x1, t1, lv1, guidance_scale=1.5, material_ids=torch.tensor([1]), voxel_sizes=vx1
        )
        assert not torch.allclose(out_virgin, out_tio2)


def test_multiphase_electrochemical_metrics(tmp_path: Path) -> None:
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    # Z-spanning carbon fiber backbone at y=6..9, x=6..9
    vol[:, 6:10, 6:10] = 1
    # Wired active coating slab attached on x=10 across z=4..11, y=6..9 (32 voxels)
    vol[4:12, 6:10, 10] = 2
    # Electrically isolated coating island in pore space (8 voxels)
    vol[7:9, 1:3, 1:3] = 2

    m = evaluate_multiphase_physics(vol, voxel_size=2.0)
    assert m["wired_coating_fraction"] == pytest.approx(32.0 / 40.0)
    # Exact 8 x 4 = 32 carbon-coating contact faces normalized by (16^3 * 2.0)
    assert m["ssa_carbon_coating"] == pytest.approx(32.0 / (4096.0 * 2.0))
    # Exact perimeter of 8 x 4 contact rectangle = 2*(8 + 4) = 24 TPB edges / (16^3 * 2.0^2)
    assert m["tpb_density"] == pytest.approx(24.0 / (4096.0 * 4.0))
    assert m["ssa_pore_coating"] > 0.0

    tif_path = export_openimpala_tiff(vol, tmp_path / "mp.tif", preserve_multiphase=True)
    reloaded = tifffile.imread(tif_path)
    assert set(np.unique(reloaded)) == {0, 1, 2}
    assert np.array_equal(reloaded, vol)


def test_end_to_end_beamtime_multiplier_cli(tmp_path: Path) -> None:
    drive_root = tmp_path / "synchrotron_drive"
    bundle_dir = tmp_path / "multielectrode_bundle"
    out_root = tmp_path / "runs"

    rng = np.random.default_rng(2027)
    yy, xx = np.indices((64, 64))
    in_circle = ((yy - 32.0) ** 2 + (xx - 32.0) ** 2) <= (27.0**2)

    scans = [
        ("crbon_virgin_1_0p008mVV_", 0.07, 0.00),
        ("crbon_virgin_1_0p053mVV_", 0.24, 0.00),
        ("tio2_anode_0p010mVV_", 0.10, 0.04),
        ("tio2_anode_0p028mVV_", 0.16, 0.06),  # Held-out coated load!
        ("tio2_anode_0p048mVV_", 0.24, 0.08),
    ]
    for folder, p_carb, p_coat in scans:
        rec = drive_root / folder / "rec_16bit_Paganin"
        rec.mkdir(parents=True, exist_ok=True)
        vol = (30_000.0 + rng.normal(0.0, 300.0, size=(72, 64, 64))).astype(np.float32)
        u = rng.random((72, 64, 64))
        vol[(u < p_carb)] += 8_500.0
        vol[(u >= p_carb) & (u < p_carb + p_coat)] += 19_500.0
        vol_u16 = vol.clip(0, 65535).astype(np.uint16)
        for z in range(vol_u16.shape[0]):
            slc = vol_u16[z].copy()
            slc[~in_circle] = 30_100
            tifffile.imwrite(rec / f"slice_{z:04d}.tif", slc)

    # 1. Fast --preview-only check
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocessing/prepare_local_drive_bundle.py"),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(bundle_dir),
            "--datasets",
            "tio2_anode_0p028mVV_",
            "--auto-multiphase",
            "--sigma",
            "10",
            "--roi-radius-fraction",
            "0.80",
            "--preview-only",
        ],
        check=True,
    )
    assert (
        bundle_dir
        / "tio2_anode_0p028mVV_binary_patches96_carbon_calibrated_v1"
        / "qa"
        / "multiphase_overlay_diagnostic.png"
    ).is_file()

    # 2. Full 3-phase multi-electrode bundle
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/preprocessing/prepare_local_drive_bundle.py"),
            "--drive-root",
            str(drive_root),
            "--output-dir",
            str(bundle_dir),
            "--version",
            "bm_v1",
            "--auto-multiphase",
            "--delta-threshold",
            "5000",
            "--delta-coating-threshold",
            "13500",
            "--sigma",
            "10",
            "--roi-radius-fraction",
            "0.80",
            "--patch-size",
            "24",
            "--stride-xy",
            "12",
            "--stride-z",
            "8",
            "--max-patches",
            "8",
            "--min-interval-length",
            "24",
            "--seam-margin",
            "2",
            "--voxel-size-um",
            "1.6",
            "--kpa-per-mvv",
            "8000",
        ],
        check=True,
    )
    manifest_path = bundle_dir / "bundle_manifest.json"
    assert manifest_path.is_file()

    # 3. Train Few-Shot Beamtime Multiplier holding out intermediate `tio2_anode_0p028mVV`
    exp = "bm_fewshot_24"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/training/train_carbon_conditional_96.py"),
            "--experiment",
            exp,
            "--data-root",
            str(bundle_dir),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--exclude-states",
            "tio2_anode_0p028mVV",
            "--use-stress-kpa",
            "--balanced-sampling",
            "--version",
            "bm_v1",
            "--patch-size",
            "24",
            "--channels",
            "16,32",
            "--attention-levels",
            "0,1",
            "--num-res-blocks",
            "1",
            "--norm-groups",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--grad-accum",
            "1",
            "--max-steps",
            "3",
            "--max-val-steps",
            "1",
            "--amp",
            "off",
            "--allow-cpu",
        ],
        check=True,
    )

    # 4. Sample all states in manifest (including unseen held-out `tio2_anode_0p028mVV`)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/sample_carbon_conditional_96.py"),
            "--experiment",
            exp,
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--sampler",
            "ddim",
            "--steps",
            "4",
            "--num-samples",
            "2",
            "--batch-size",
            "2",
            "--sweep-loads",
            "",
            "--export-openimpala",
            "--allow-cpu",
        ],
        check=True,
    )

    # 5. Benchmark PorousDiff vs Affine Z-Squash & Pillar Pore-Deletion Baselines
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/evaluate_beamtime_multiplier.py"),
            "--experiments",
            f"porousdiff_fewshot={exp}",
            "--data-root",
            str(bundle_dir),
            "--output-root",
            str(out_root),
            "--manifest-json",
            str(manifest_path),
            "--version",
            "bm_v1",
            "--patch-size",
            "24",
            "--heldout-states",
            "tio2_anode_0p028mVV",
            "--export-real-openimpala",
        ],
        check=True,
    )

    bm_dir = out_root / "results" / exp / "beamtime_benchmark"
    bm_report = json.loads((bm_dir / "beamtime_multiplier_report.json").read_text())
    assert bm_report["tio2_anode_0p028mVV"]["is_heldout"] is True
    assert set(bm_report["tio2_anode_0p028mVV"]["methods"].keys()) == {
        "noop_uncompressed",
        "affine_z_squash",
        "pillar_pore_deletion",
        "porousdiff_fewshot",
    }
    assert (bm_dir / "baseline_vs_porousdiff_table.csv").is_file()
    assert (bm_dir / "beamtime_multiplier_comparison.png").is_file()
    assert (bm_dir / "openimpala_suite" / "openimpala_jobs.json").is_file()
    assert (bm_dir / "openimpala_suite" / "run_openimpala_batch.sh").is_file()


def _write_synthetic_multiphase_stack(
    scan_dir: Path,
    n_slices: int,
    hw: int,
    carbon_prob: float,
    coating_prob: float,
    rng: np.random.Generator,
) -> None:
    import tifffile

    scan_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((hw, hw))
    c = hw / 2.0
    r = 0.44 * hw
    in_circle = ((yy - c) ** 2 + (xx - c) ** 2) <= (r**2)

    vol = (30_000.0 + rng.normal(0.0, 280.0, size=(n_slices, hw, hw))).astype(np.float32)
    u = rng.random((n_slices, hw, hw))
    vol[u < carbon_prob] += 8_500.0
    vol[(u >= carbon_prob) & (u < carbon_prob + coating_prob)] += 20_500.0
    vol_u16 = vol.clip(0, 65535).astype(np.uint16)
    for z_idx in range(n_slices):
        slc = vol_u16[z_idx].copy()
        slc[~in_circle] = 30_050
        tifffile.imwrite(scan_dir / f"slice_{z_idx:04d}.tif", slc)


def test_nested_discovery_group_calibration_append_and_resume(tmp_path: Path) -> None:
    """Verify nested Op001mVV collision safety, shared lowest-load threshold calibration
    (immune to high-load bottom plunger), --resume cache reuse, and --append K11 DIAD merge."""
    rng = np.random.default_rng(77)
    i12_root = tmp_path / "i12_drive"
    diad_root = tmp_path / "diad_drive"
    bundle_dir = tmp_path / "unified_bundle"

    # Nested folders with identical leaf names (`Op001mVV`) under `TiO2/` and `CuHCF/`,
    # plus high-load `TiO2/Op045mVV` containing a bright solid metal plunger in bottom 30% of Z!
    _write_synthetic_multiphase_stack(
        i12_root / "TiO2_Anode" / "Op001mVV",
        n_slices=52,
        hw=56,
        carbon_prob=0.14,
        coating_prob=0.06,
        rng=rng,
    )
    high_load_dir = i12_root / "TiO2_Anode" / "Op045mVV"
    _write_synthetic_multiphase_stack(
        high_load_dir,
        n_slices=52,
        hw=56,
        carbon_prob=0.28,
        coating_prob=0.11,
        rng=rng,
    )
    # Contaminate Z >= 36 (`3N/4 = 39`) of high-load scan with a saturated 62,000-count plunger
    import tifffile

    for z_idx in range(36, 52):
        p = high_load_dir / f"slice_{z_idx:04d}.tif"
        tifffile.imwrite(p, np.full((56, 56), 62_000, dtype=np.uint16))

    _write_synthetic_multiphase_stack(
        i12_root / "CuHCF_Cathode" / "Op001mVV",
        n_slices=52,
        hw=56,
        carbon_prob=0.15,
        coating_prob=0.07,
        rng=rng,
    )

    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/preprocessing/prepare_local_drive_bundle.py"),
        "--output-dir",
        str(bundle_dir),
        "--version",
        "v03_test",
        "--auto-multiphase",
        "--delta-threshold",
        "4500",
        "--sigma",
        "10",
        "--roi-radius-fraction",
        "0.80",
        "--patch-size",
        "16",
        "--stride-xy",
        "12",
        "--stride-z",
        "8",
        "--max-patches",
        "6",
        "--min-interval-length",
        "16",
        "--seam-margin",
        "2",
    ]

    subprocess.run(
        base_cmd
        + [
            "--drive-root",
            str(i12_root),
            "--voxel-size-um",
            "1.6",
            "--kpa-per-mvv",
            "8000",
            "--beamline",
            "DLS_I12",
        ],
        check=True,
    )

    m1 = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    scans_by_state = {sc["state"]: sc for sc in m1["scans"]}
    assert "DLS_I12__tio2_anode__Op001mVV" in scans_by_state
    assert "DLS_I12__tio2_anode__Op045mVV" in scans_by_state
    assert "DLS_I12__cuhcf_cathode__Op001mVV" in scans_by_state

    # Both TiO2 loads MUST share the exact same delta_coating calibrated on Op001mVV (not plunger!)
    t_low = scans_by_state["DLS_I12__tio2_anode__Op001mVV"]
    t_high = scans_by_state["DLS_I12__tio2_anode__Op045mVV"]
    assert t_low["delta_coating"] == t_high["delta_coating"]
    assert 9_000.0 < float(t_high["delta_coating"]) < 19_000.0
    assert float(t_low["strain_z"]) == 0.0
    assert float(t_high["strain_z"]) > 0.15  # Plunger slices gated out -> positive axial compression strain

    # Test --resume (should reuse existing patches instantaneously)
    res_proc = subprocess.run(
        base_cmd
        + [
            "--drive-root",
            str(i12_root),
            "--voxel-size-um",
            "1.6",
            "--kpa-per-mvv",
            "8000",
            "--beamline",
            "DLS_I12",
            "--resume",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Resuming: reusing existing patches" in res_proc.stdout

    # Test --append from a second beamline (K11 DIAD) with Newton load (`50N` on `10 mm` punch) + --auto-calibrate-carbon
    _write_synthetic_multiphase_stack(
        diad_root / "TiO2" / "scan_50N",
        n_slices=52,
        hw=56,
        carbon_prob=0.20,
        coating_prob=0.08,
        rng=rng,
    )
    subprocess.run(
        base_cmd
        + [
            "--drive-root",
            str(diad_root),
            "--voxel-size-um",
            "0.54",
            "--punch-diameter-mm",
            "10.0",
            "--beamline",
            "DLS_K11_DIAD",
            "--auto-calibrate-carbon",
            "--append",
        ],
        check=True,
    )

    m2 = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    merged_states = {sc["state"]: sc for sc in m2["scans"]}
    assert len(merged_states) == 4
    diad_sc = merged_states["DLS_K11_DIAD__tio2_anode__scan_50N"]
    # 50 N over pi * (5 mm)^2 = 636.62 kPa
    assert np.isclose(float(diad_sc["stress_kpa"]), 50.0 / (np.pi * (0.005**2)) * 1e-3, rtol=1e-3)
    assert np.isclose(float(diad_sc["effective_voxel_size_um"]), 0.54)


def test_openimpala_suite_and_diad_xrd_correlation(tmp_path: Path) -> None:
    """Verify OpenImpala output ingestion CLI and K11 DIAD XRD correlation analysis."""
    from porousdiff.metrics import export_openimpala_tiff
    from porousdiff.transport_and_xrd import (
        correlate_xrd_with_microstructure,
        generate_openimpala_job_suite,
    )

    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[:, 4:10, 4:10] = 1
    vol[:, 6:8, 6:8] = 2
    tif_path = tmp_path / "tiffs" / "sample_001.tif"
    export_openimpala_tiff(vol, tif_path, preserve_multiphase=True)

    suite_dir = tmp_path / "openimpala_suite"
    info = generate_openimpala_job_suite(
        [
            {
                "method": "real",
                "state": "diad_s1",
                "material": "tio2_anode",
                "beamline": "DLS_K11_DIAD",
                "load_value": 100.0,
                "tiff_path": tif_path,
                "num_phases": 3,
            },
            {
                "method": "porousdiff",
                "state": "diad_s1",
                "material": "tio2_anode",
                "beamline": "DLS_K11_DIAD",
                "load_value": 100.0,
                "tiff_path": tif_path,
                "num_phases": 3,
            },
        ],
        suite_dir=suite_dir,
    )
    assert info["num_jobs"] == 6  # 2 methods * 3 phases (pore_ionic, carbon_electronic, coating_active)

    # Write mock OpenImpala solver outputs (both JSON and key=value log formats)
    jobs = json.loads((suite_dir / "openimpala_jobs.json").read_text())["jobs"]
    Path(jobs[0]["output_json"]).write_text(
        json.dumps({"d_eff_z": 0.42, "tau_z": 1.65, "volume_fraction": 0.70})
    )
    (suite_dir / "outputs" / f"{jobs[3]['job_id']}.log").write_text(
        "volume_fraction = 0.69\nd_eff_z = 0.41\ntortuosity_z = 1.68\n"
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/parse_openimpala_suite.py"),
            "--suite-dir",
            str(suite_dir),
        ],
        check=True,
    )
    assert (suite_dir / "openimpala_transport_parity.csv").is_file()
    assert (suite_dir / "openimpala_transport_parity.png").is_file()

    # Test DIAD XRD correlation across 4 compression states
    mock_report: dict[str, object] = {}
    xrd_lines = ["state,lattice_strain_pct,fwhm_deg"]
    for idx, lv in enumerate([50.0, 150.0, 300.0, 500.0]):
        st = f"DLS_K11_DIAD__tio2_s{idx}"
        easa_real = 0.30 - 0.04 * idx
        easa_pred = 0.295 - 0.039 * idx
        tpb_real = 0.12 - 0.015 * idx
        tau_real = 1.15 + 0.12 * idx
        metrics_real = {
            "solid_fraction": {"mean": 0.10 + 0.05 * idx},
            "coating_fraction": {"mean": 0.04 + 0.01 * idx},
            "ssa_pore_coating": {"mean": easa_real},
            "tpb_density": {"mean": tpb_real},
            "wired_coating_fraction": {"mean": 0.85 + 0.03 * idx},
            "geodesic_tortuosity_z": {"mean": tau_real},
        }
        metrics_pred = {
            "solid_fraction": {"mean": 0.105 + 0.05 * idx},
            "coating_fraction": {"mean": 0.041 + 0.01 * idx},
            "ssa_pore_coating": {"mean": easa_pred},
            "tpb_density": {"mean": tpb_real - 0.002},
            "wired_coating_fraction": {"mean": 0.84 + 0.03 * idx},
            "geodesic_tortuosity_z": {"mean": tau_real + 0.01},
        }
        mock_report[st] = {
            "material": "tio2_anode",
            "load_value": lv,
            "real": {"metrics": metrics_real},
            "methods": {"porousdiff": {"metrics": metrics_pred}},
        }
        xrd_lines.append(f"tio2_s{idx},{0.05 + 0.08 * idx:.4f},{0.22 + 0.03 * idx:.4f}")

    xrd_csv = tmp_path / "diad_xrd.csv"
    xrd_csv.write_text("\n".join(xrd_lines) + "\n")
    xrd_out = tmp_path / "xrd_out"
    corr = correlate_xrd_with_microstructure(xrd_csv, mock_report, xrd_out)
    assert corr["num_matched_states"] == 4
    r_easa = corr["correlations"]["lattice_strain_pct"]["pred_ssa_pore_coating"]["pearson_r"]
    assert r_easa < -0.99
    assert (xrd_out / "xrd_microstructure_merged.csv").is_file()
    assert (xrd_out / "diad_xrd_microstructure_correlation.png").is_file()

