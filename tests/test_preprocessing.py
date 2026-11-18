from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from porousdiff.preprocessing import (
    build_stratified_patch_plan,
    calibrate_delta_threshold,
    compute_radial_ring_score,
    detect_valid_z_intervals,
    enumerate_cylinder_xy_positions,
    make_circular_mask,
    parse_load_value,
    segment_slice_calibrated,
    split_z_intervals_with_buffer,
)


def test_parse_load_value_variants() -> None:
    assert parse_load_value("crbon_virgin_1_0p048mVV_") == pytest.approx(0.048)
    assert parse_load_value("sample_crbon_virgin_1_0p008mVV") == pytest.approx(0.008)
    assert parse_load_value("Op008mVV") == pytest.approx(0.008)
    assert parse_load_value("Op033mVV") == pytest.approx(0.033)
    with pytest.raises(ValueError):
        parse_load_value("unknown_folder_name")


def test_circular_roi_and_xy_positions_stay_strictly_inside() -> None:
    shape = (256, 256)
    r_frac = 0.70
    mask = make_circular_mask(shape, radius_fraction=r_frac)
    assert mask[128, 128]
    assert not mask[0, 0]
    assert not mask[0, 255]

    patch_size = 32
    positions = enumerate_cylinder_xy_positions(
        shape_hw=shape,
        patch_size=patch_size,
        stride_xy=16,
        radius_fraction=r_frac,
    )
    assert len(positions) > 10
    for y, x in positions:
        assert mask[y : y + patch_size, x : x + patch_size].all()


def test_calibrated_segmentation_rejects_empty_pore_slice_and_preserves_monotonicity() -> None:
    rng = np.random.default_rng(123)
    shape = (192, 192)
    roi = make_circular_mask(shape, radius_fraction=0.80)

    # Empty pore slice with realistic Paganin radial cupping + synchrotron beam offset
    yy, xx = np.indices(shape)
    cupping = 600.0 * (((yy - 96) / 96.0) ** 2 + ((xx - 96) / 96.0) ** 2)
    empty_slice = (30_000.0 + cupping + rng.normal(0.0, 450.0, size=shape)).astype(np.uint16)

    bin_empty, stats_empty = segment_slice_calibrated(
        empty_slice, delta_threshold=5500.0, roi_mask=roi, sigma=24.0
    )
    assert stats_empty["solid_fraction"] < 0.005
    assert bin_empty[~roi].sum() == 0

    # Increasing fiber density across 3 synthetic compression states (with drifting beam baseline)
    fracs = []
    for beam_offset, fiber_prob in [(0.0, 0.06), (1800.0, 0.18), (-1200.0, 0.32)]:
        slc = (30_000.0 + beam_offset + cupping + rng.normal(0.0, 450.0, size=shape)).astype(
            np.float32
        )
        fibers = rng.random(shape) < fiber_prob
        slc[fibers] += 9500.0
        _, st = segment_slice_calibrated(
            slc.clip(0, 65535).astype(np.uint16),
            delta_threshold=5500.0,
            roi_mask=roi,
            sigma=24.0,
        )
        fracs.append(st["solid_fraction"])

    assert fracs[0] < fracs[1] < fracs[2]

    # Auto-calibration smoke check
    auto_thr = calibrate_delta_threshold([slc.clip(0, 65535).astype(np.uint16)], roi_mask=roi, sigma=24.0)
    assert 2000.0 < auto_thr < 8500.0


def test_real_xct_npz_monotonicity_and_ring_rejection_if_present() -> None:
    data_dir = Path("/tmp/compression_data")
    files = [
        data_dir / "sample_crbon_virgin_1_0p008mVV.npz",
        data_dir / "sample_crbon_virgin_1_0p028mVV.npz",
        data_dir / "sample_crbon_virgin_1_0p053mVV.npz",
    ]
    if not all(f.is_file() for f in files):
        pytest.skip("Real XCT sample .npz files not staged in /tmp/compression_data")

    crop_fracs = []
    mid_ring_scores = {}
    for label, path in zip(["0p008", "0p028", "0p053"], files):
        with np.load(path) as d:
            crop = d["crop_3d"]  # (128, 192, 192) uint16 inside sample core
            mid_xy = d["xy_mid"]
        mid_ring_scores[label] = compute_radial_ring_score(mid_xy)
        # In 0p028, z >= 40 intersects the real mid-stack bullseye stitch band; z < 36 is clean felt
        z_limit = 36 if label == "0p028" else crop.shape[0]
        slice_sfs = [
            segment_slice_calibrated(crop[z], delta_threshold=5500.0, sigma=24.0)[1][
                "solid_fraction"
            ]
            for z in range(0, z_limit, 6)
        ]
        crop_fracs.append(float(np.mean(slice_sfs)))

    # Clean mid-XY slices (0p008, 0p053) pass (< 0.05), while 0p028's bullseye mid-XY slice spikes (> 0.20)
    assert mid_ring_scores["0p008"] < 0.05
    assert mid_ring_scores["0p053"] < 0.05
    assert mid_ring_scores["0p028"] > 0.20

    # Solid fraction must strictly increase with mechanical load: 0.008 < 0.028 < 0.053 mV/V
    assert 0.02 < crop_fracs[0] < crop_fracs[1] < crop_fracs[2] < 0.55


def test_ring_artifact_and_plunger_z_interval_gating() -> None:
    rng = np.random.default_rng(7)
    h, w = 256, 256
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - 128.0) ** 2 + (xx - 128.0) ** 2)

    clean_slice = 30_000.0 + rng.normal(0.0, 900.0, size=(h, w))
    bullseye_slice = clean_slice + 2200.0 * np.sin(rr * 0.45)

    assert compute_radial_ring_score(clean_slice) < 0.06
    assert compute_radial_ring_score(bullseye_slice) > 0.25

    # Simulate a 400-slice stack with a bullseye seam at Z=150..165 and plunger at Z=330..399
    n = 400
    ring_scores = np.full(n, 0.02, dtype=np.float32)
    pore_modes = np.full(n, 30_200.0, dtype=np.float32)
    jumps = np.full(n, 80.0, dtype=np.float32)

    ring_scores[150:165] = 0.42
    jumps[150:165] = 350.0
    pore_modes[330:] = 42_000.0
    ring_scores[330:] = 0.65

    valid_mask, intervals = detect_valid_z_intervals(
        ring_scores=ring_scores,
        pore_modes=pore_modes,
        slice_jumps=jumps,
        min_interval_length=64,
        seam_margin=8,
    )
    assert not valid_mask[145:170].any()
    assert not valid_mask[325:].any()
    assert len(intervals) == 2
    assert intervals[0][1] <= 142
    assert intervals[1][0] >= 173 and intervals[1][1] <= 322


def test_buffered_split_and_stratified_patch_plan_cover_all_z() -> None:
    split = split_z_intervals_with_buffer(
        intervals=[(0, 500)],
        patch_size=64,
        val_fraction=0.2,
        buffer_slices=64,
    )
    train_s, train_e = split["train"][0]
    val_s, val_e = split["val"][0]
    assert train_e - train_s >= 64
    assert val_e - val_s >= 64
    # Strictly >= patch_size buffer between train and val slabs
    assert val_s - train_e >= 64

    xy_pos = [(y, x) for y in range(0, 256, 64) for x in range(0, 256, 64)]
    plan = build_stratified_patch_plan(
        z_intervals=[(0, 320)],
        xy_positions=xy_pos,
        patch_size=64,
        stride_z=32,
        max_patches=27,
        rng=np.random.default_rng(42),
    )
    planned_z = [z for z, _ in plan]
    # Every Z window from 0 to 256 (9 windows) must be represented, not just the first 2!
    assert len(planned_z) == 9
    assert min(planned_z) == 0
    assert max(planned_z) == 256
