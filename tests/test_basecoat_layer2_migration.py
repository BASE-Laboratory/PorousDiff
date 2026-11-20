"""Tests for PorousDiff Layer 2 `basecoat` boundary, OME-Zarr export, and OpenImpala v5.1 SI hand-off."""

from __future__ import annotations

from pathlib import Path

import basecoat
from basecoat.layers import check_layer_imports
import numpy as np
import openimpala as oi
import pytest

from porousdiff import export_volume_ome_zarr, load_volume_array, to_basecoat_volume
from porousdiff.metrics import CAUCHY_CROFTON_3D_FACTOR, evaluate_multiphase_physics
from porousdiff_workflows.transport_and_xrd import (
    compute_openimpala_volume_transport,
    run_openimpala_python_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_porousdiff_layer2_import_boundary() -> None:
    """`porousdiff/` must contain zero imports of Layer 3/4 packages (`openimpala`, `porousdiff_workflows`, etc.)."""
    violations = check_layer_imports(REPO_ROOT)
    assert violations == []


def test_basecoat_ome_zarr_multiphase_roundtrip_and_mode_downsampling(tmp_path: Path) -> None:
    """Exporting a 3-phase `{0: pore, 1: carbon, 2: coating}` volume preserves labels, spacing, and mode pyramid."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[:, 4:12, 4:12] = 2  # coating shell
    vol[:, 6:10, 6:10] = 1  # carbon core

    store_path = tmp_path / "electrode_3phase.ome.zarr"
    out = export_volume_ome_zarr(
        vol,
        store_path,
        voxel_size_um=0.65,
        preserve_multiphase=True,
        multiscale_levels=2,
        metadata={"material": "tio2_anode", "stress_kpa": 45.0},
    )
    assert out == store_path and store_path.is_dir()

    vset = basecoat.read_ome_zarr(store_path, level=0, as_set=True)
    assert isinstance(vset, basecoat.VolumeSet)
    assert "phase" in vset.labels
    phase_vol = vset.labels["phase"]
    assert np.array_equal(phase_vol.data, vol)
    assert phase_vol.spacing_in("um") == pytest.approx((0.65, 0.65, 0.65))
    assert phase_vol.spacing_in("m") == pytest.approx((0.65e-6, 0.65e-6, 0.65e-6))
    assert phase_vol.label_names == {0: "pore", 1: "carbon", 2: "coating"}

    # Pyramid level 1 uses block-mode (pure integer labels {0, 1, 2}, never fractional averages)
    vset_l1 = basecoat.read_ome_zarr(store_path, level=1, as_set=True)
    assert isinstance(vset_l1, basecoat.VolumeSet)
    assert set(np.unique(vset_l1.labels["phase"].data)).issubset({0, 1, 2})

    loaded_arr, loaded_vox = load_volume_array(store_path)
    assert np.array_equal(loaded_arr, vol)
    assert loaded_vox == pytest.approx(0.65)


def test_direct_openimpala_v51_handoff_and_si_ssa_calibration(tmp_path: Path) -> None:
    """Both in-memory `to_basecoat_volume` and `.ome.zarr` stores pass calibrated SI metres to OpenImpala v5.1."""
    oi.Session.ensure_active()

    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[:, 5:15, 5:15] = 2
    vol[:, 7:13, 7:13] = 1
    vox_um = 0.8

    bc_vol = to_basecoat_volume(vol, voxel_size_um=vox_um, preserve_multiphase=True)
    store_path = export_volume_ome_zarr(vol, tmp_path / "cyl.ome.zarr", voxel_size_um=vox_um, preserve_multiphase=True)

    # 1. Direct OpenImpala v5.1 tortuosity on both VolumeProtocol and .ome.zarr store
    res_mem = oi.tortuosity(bc_vol, phase=0, direction="z")
    res_zarr = oi.tortuosity(store_path, phase=0, direction="z")
    assert res_mem.effective_diffusivity == pytest.approx(res_zarr.effective_diffusivity, rel=1e-6)
    assert res_mem.tortuosity == pytest.approx(res_zarr.tortuosity, rel=1e-6)

    # 2. SI SSA calibration: OpenImpala returns m^-1 = 1e6 * PorousDiff's um^-1
    mp_um = evaluate_multiphase_physics(vol, voxel_size=vox_um)
    oi_res = compute_openimpala_volume_transport(vol, voxel_size_um=vox_um, num_phases=3)
    interfacial = oi_res["interfacial"]  # type: ignore[index]

    assert float(interfacial["ssa_pore_coating_cc"]) == pytest.approx(
        mp_um["ssa_pore_coating"] * CAUCHY_CROFTON_3D_FACTOR, rel=1e-5
    )
    assert float(interfacial["ssa_pore_coating_cc_m_inv"]) == pytest.approx(
        float(interfacial["ssa_pore_coating_cc"]) * 1e6, rel=1e-5
    )

    # 3. Suite runner consumes `.ome.zarr` entries directly without intermediate TIFFs
    suite_dir = tmp_path / "zarr_suite"
    rows = run_openimpala_python_suite(
        [
            {
                "method": "porousdiff",
                "state": "tio2_45kpa",
                "material": "tio2_anode",
                "load_value": 45.0,
                "zarr_path": store_path,
                "num_phases": 3,
            }
        ],
        suite_dir=suite_dir,
        out_csv=suite_dir / "parity.csv",
    )
    assert len(rows) == 4
    assert (suite_dir / "parity.csv").is_file()
