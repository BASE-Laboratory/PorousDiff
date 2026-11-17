"""Optimal clamping-pressure (`sigma*`) discovery & OpenImpala reaction-transport coupling.

Unifies PorousDiff continuous stress conditioning with OpenImpala continuum transport solvers
(and exact 3D geodesic / Carman-Kozeny microstructural transport surrogates) to identify the
electrochemical cell clamping sweet spot `sigma*` balancing:
  1. Ionic pore-phase transport (`D_eff,Z / D_0` and hydraulic permeability `K_Z`),
  2. Electronic solid-backbone percolation & wiring (`sigma_eff,Z / sigma_0` and wired coating),
  3. Electrochemically active wired catalyst interface (`ssa_pore_coating * wired_fraction` & TPB).

Models total cell conductance via the series resistance law:
  `R_cell(sigma) = w_ion / D_tilde(sigma) + w_elec / sigma_tilde(sigma) + w_kin / A_tilde(sigma)`
so `FoM(sigma) = 1 / R_cell(sigma)` vanishes if either pores pinch closed at over-compression OR
the fibre/coating network loses electrical percolation at under-compression.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


def load_openimpala_lookup(
    csv_path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Group `openimpala_transport_parity.csv` rows by `(method, state)` into mean transport dicts."""
    if not csv_path.is_file():
        return {}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            method = str(row.get("method", "")).strip()
            state = str(row.get("state", "")).strip()
            phase = str(row.get("transport_phase", "")).strip().lower()
            raw_deff = row.get("d_eff_z")
            if not method or not state or raw_deff is None or not str(raw_deff).strip():
                continue
            try:
                val = float(str(raw_deff).strip())
            except ValueError:
                continue
            if not np.isfinite(val):
                continue
            key = (method, state)
            entry = buckets.setdefault(key, {"pore_d_eff_z": [], "solid_sigma_eff_z": []})
            if "pore" in phase or "ion" in phase:
                entry["pore_d_eff_z"].append(val)
            elif "carbon" in phase or "solid" in phase or "elec" in phase:
                entry["solid_sigma_eff_z"].append(val)

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in buckets.items():
        agg: dict[str, float] = {}
        if vals["pore_d_eff_z"]:
            agg["pore_d_eff_z"] = float(np.mean(vals["pore_d_eff_z"]))
        if vals["solid_sigma_eff_z"]:
            agg["solid_sigma_eff_z"] = float(np.mean(vals["solid_sigma_eff_z"]))
        if agg:
            lookup[key] = agg
    return lookup


def compute_transport_and_fom_record(
    phys_metrics: Mapping[str, float],
    openimpala_metrics: Mapping[str, float] | None = None,
    voxel_size_um: float = 1.0,
) -> dict[str, float]:
    """Compute ionic, electronic, hydraulic, and kinetic observables for a single 3D realization."""
    sf = float(phys_metrics.get("solid_fraction", 0.0))
    porosity = float(phys_metrics.get("porosity", max(0.0, 1.0 - sf)))
    coat_frac = float(phys_metrics.get("coating_fraction", 0.0))
    tau_pore_z = float(phys_metrics.get("geodesic_tortuosity_z", float("nan")))
    tau_solid_z = float(phys_metrics.get("geodesic_solid_tortuosity_z", float("nan")))

    # 1. Through-plane ionic effective diffusivity D_eff,Z / D_0
    if openimpala_metrics and "pore_d_eff_z" in openimpala_metrics and np.isfinite(openimpala_metrics["pore_d_eff_z"]):
        d_eff_pore_z = float(max(0.0, openimpala_metrics["pore_d_eff_z"]))
        used_openimpala = 1.0
    elif np.isfinite(tau_pore_z) and tau_pore_z >= 1.0:
        d_eff_pore_z = float(porosity / (tau_pore_z**2))
        used_openimpala = 0.0
    else:
        d_eff_pore_z = 0.0
        used_openimpala = 0.0

    # 2. Through-plane electronic effective conductivity sigma_eff,Z / sigma_0
    is_coated = coat_frac > 1e-6 or float(phys_metrics.get("ssa_pore_coating", 0.0)) > 1e-6
    wired_frac = float(
        phys_metrics.get("wired_coating_fraction", phys_metrics.get("solid_connected_fraction", 1.0))
    )
    solid_conn = float(phys_metrics.get("solid_connected_fraction", 1.0))
    elec_factor = wired_frac if is_coated else solid_conn

    if openimpala_metrics and "solid_sigma_eff_z" in openimpala_metrics and np.isfinite(openimpala_metrics["solid_sigma_eff_z"]):
        sigma_eff_solid_z = float(max(0.0, openimpala_metrics["solid_sigma_eff_z"]))
    elif np.isfinite(tau_solid_z) and tau_solid_z >= 1.0:
        sigma_eff_solid_z = float(sf * elec_factor / (tau_solid_z**2))
    else:
        # Fallback Bruggeman-percolation estimate when thin patches barely miss Z-face contact
        sigma_eff_solid_z = float((sf**1.8) * elec_factor * 0.25)

    # 3. Carman-Kozeny hydraulic permeability proxy (um^2)
    pore_chord_um = float(phys_metrics.get("pore_chord_z", 0.0)) * max(voxel_size_um, 1e-4)
    if np.isfinite(tau_pore_z) and tau_pore_z >= 1.0 and porosity > 1e-4:
        denom = 180.0 * max((1.0 - porosity) ** 2, 1e-5) * (tau_pore_z**2)
        perm_ck_um2 = float((porosity**3) * (pore_chord_um**2) / denom)
    else:
        perm_ck_um2 = 0.0

    # 4. Electrochemically active wired interfacial density (1/um)
    ssa_coat = float(phys_metrics.get("ssa_pore_coating", 0.0))
    tpb = float(phys_metrics.get("tpb_density", 0.0))
    ssa_total = float(phys_metrics.get("ssa_total", 0.0))
    if is_coated:
        active_interface = float(ssa_coat * wired_frac + 0.05 * voxel_size_um * tpb * wired_frac)
    else:
        active_interface = float(ssa_total * solid_conn)

    return {
        "solid_fraction": sf,
        "porosity": porosity,
        "d_eff_pore_z": d_eff_pore_z,
        "sigma_eff_solid_z": sigma_eff_solid_z,
        "permeability_ck_um2": perm_ck_um2,
        "active_wired_interface": active_interface,
        "wired_coating_fraction": wired_frac if is_coated else solid_conn,
        "tpb_density": tpb,
        "z_pinch_contact_density": float(phys_metrics.get("z_pinch_contact_density", 0.0)),
        "coating_pinch_fraction": float(phys_metrics.get("coating_pinch_fraction", 0.0)),
        "used_openimpala": used_openimpala,
    }


def _evaluate_harmonic_fom_curve(
    loads: np.ndarray,
    d_eff: np.ndarray,
    sigma_eff: np.ndarray,
    active_if: np.ndarray,
    grid_loads: np.ndarray,
    weights: tuple[float, float, float],
    norm_scales: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Interpolate transport pillars via PCHIP and evaluate series-conductance FoM on `grid_loads`."""
    w_ion, w_elec, w_kin = weights
    w_sum = max(w_ion + w_elec + w_kin, 1e-8)
    w_ion, w_elec, w_kin = w_ion / w_sum, w_elec / w_sum, w_kin / w_sum

    if norm_scales is None:
        s_ion = float(max(np.max(d_eff), 1e-8))
        s_elec = float(max(np.max(sigma_eff), 1e-8))
        s_kin = float(max(np.max(active_if), 1e-8))
        scales = (s_ion, s_elec, s_kin)
    else:
        scales = norm_scales

    ion_n = np.clip(d_eff / scales[0], 0.0, None)
    elec_n = np.clip(sigma_eff / scales[1], 0.0, None)
    kin_n = np.clip(active_if / scales[2], 0.0, None)

    if loads.size >= 2:
        ion_g = np.clip(PchipInterpolator(loads, ion_n)(grid_loads), 0.0, None)
        elec_g = np.clip(PchipInterpolator(loads, elec_n)(grid_loads), 0.0, None)
        kin_g = np.clip(PchipInterpolator(loads, kin_n)(grid_loads), 0.0, None)
    else:
        ion_g = np.full_like(grid_loads, ion_n[0])
        elec_g = np.full_like(grid_loads, elec_n[0])
        kin_g = np.full_like(grid_loads, kin_n[0])

    eps = 1e-4
    inv_conductance = (
        w_ion / (ion_g + eps)
        + w_elec / (elec_g + eps)
        + w_kin / (kin_g + eps)
    )
    fom_grid = 1.0 / inv_conductance
    return fom_grid, ion_g, elec_g, kin_g, scales


def find_optimal_clamping_pressure(
    records_by_load: Mapping[float, Sequence[Mapping[str, float]]],
    weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    grid_points: int = 201,
    n_bootstrap: int = 200,
    seed: int = 42,
    norm_scales: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    """Identify optimal clamping stress `sigma*` and bootstrap 95% CI from per-load realizations."""
    sorted_loads = sorted(float(k) for k in records_by_load.keys() if records_by_load[k])
    if not sorted_loads:
        raise ValueError("records_by_load must contain at least one non-empty load level")

    loads_arr = np.asarray(sorted_loads, dtype=np.float64)
    mean_d_eff = np.asarray(
        [np.mean([float(r["d_eff_pore_z"]) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )
    mean_sigma = np.asarray(
        [np.mean([float(r["sigma_eff_solid_z"]) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )
    mean_active = np.asarray(
        [np.mean([float(r["active_wired_interface"]) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )
    mean_perm = np.asarray(
        [np.mean([float(r["permeability_ck_um2"]) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )

    if len(sorted_loads) >= 2 and sorted_loads[-1] > sorted_loads[0]:
        grid_loads = np.linspace(sorted_loads[0], sorted_loads[-1], max(11, int(grid_points)))
    else:
        grid_loads = loads_arr.copy()

    fom_grid, ion_g, elec_g, kin_g, scales = _evaluate_harmonic_fom_curve(
        loads_arr,
        mean_d_eff,
        mean_sigma,
        mean_active,
        grid_loads,
        weights=weights,
        norm_scales=norm_scales,
    )
    best_idx = int(np.argmax(fom_grid))
    opt_load = float(grid_loads[best_idx])
    opt_fom = float(fom_grid[best_idx])

    # Non-parametric bootstrap across realizations at each discrete load
    rng = np.random.default_rng(seed)
    boot_opt_loads: list[float] = []
    boot_curves: list[np.ndarray] = []
    can_bootstrap = any(len(records_by_load[lv]) > 1 for lv in sorted_loads) and n_bootstrap > 0

    if can_bootstrap:
        for _ in range(int(n_bootstrap)):
            b_deff = np.empty_like(loads_arr)
            b_sig = np.empty_like(loads_arr)
            b_act = np.empty_like(loads_arr)
            for i, lv in enumerate(sorted_loads):
                recs = records_by_load[lv]
                n_r = len(recs)
                pick = rng.integers(0, n_r, size=n_r)
                b_deff[i] = np.mean([float(recs[j]["d_eff_pore_z"]) for j in pick])
                b_sig[i] = np.mean([float(recs[j]["sigma_eff_solid_z"]) for j in pick])
                b_act[i] = np.mean([float(recs[j]["active_wired_interface"]) for j in pick])
            b_fom, _, _, _, _ = _evaluate_harmonic_fom_curve(
                loads_arr, b_deff, b_sig, b_act, grid_loads, weights=weights, norm_scales=scales
            )
            boot_opt_loads.append(float(grid_loads[int(np.argmax(b_fom))]))
            boot_curves.append(b_fom)

        ci_low, ci_high = (
            float(np.percentile(boot_opt_loads, 2.5)),
            float(np.percentile(boot_opt_loads, 97.5)),
        )
        curve_stack = np.stack(boot_curves, axis=0)
        fom_ci_low = np.percentile(curve_stack, 2.5, axis=0)
        fom_ci_high = np.percentile(curve_stack, 97.5, axis=0)
    else:
        ci_low, ci_high = opt_load, opt_load
        fom_ci_low = fom_grid.copy()
        fom_ci_high = fom_grid.copy()

    discrete_rows: list[dict[str, float]] = []
    discrete_fom, _, _, _, _ = _evaluate_harmonic_fom_curve(
        loads_arr, mean_d_eff, mean_sigma, mean_active, loads_arr, weights=weights, norm_scales=scales
    )
    for i, lv in enumerate(sorted_loads):
        recs = records_by_load[lv]
        discrete_rows.append(
            {
                "load_value": lv,
                "num_samples": float(len(recs)),
                "solid_fraction": float(np.mean([float(r["solid_fraction"]) for r in recs])),
                "porosity": float(np.mean([float(r["porosity"]) for r in recs])),
                "d_eff_pore_z": float(mean_d_eff[i]),
                "sigma_eff_solid_z": float(mean_sigma[i]),
                "permeability_ck_um2": float(mean_perm[i]),
                "active_wired_interface": float(mean_active[i]),
                "wired_coating_fraction": float(
                    np.mean([float(r["wired_coating_fraction"]) for r in recs])
                ),
                "tpb_density": float(np.mean([float(r["tpb_density"]) for r in recs])),
                "coating_pinch_fraction": float(
                    np.mean([float(r["coating_pinch_fraction"]) for r in recs])
                ),
                "fom_harmonic": float(discrete_fom[i]),
                "used_openimpala": float(np.mean([float(r["used_openimpala"]) for r in recs])),
            }
        )

    return {
        "optimal_load": round(opt_load, 6),
        "optimal_load_ci95": [round(ci_low, 6), round(ci_high, 6)],
        "optimal_fom": round(opt_fom, 6),
        "weights": {"ionic": weights[0], "electronic": weights[1], "kinetic": weights[2]},
        "norm_scales": {
            "d_eff_pore_z": scales[0],
            "sigma_eff_solid_z": scales[1],
            "active_wired_interface": scales[2],
        },
        "discrete_points": discrete_rows,
        "continuous_curve": {
            "grid_loads": [float(x) for x in grid_loads],
            "fom_mean": [float(x) for x in fom_grid],
            "fom_ci95_low": [float(x) for x in fom_ci_low],
            "fom_ci95_high": [float(x) for x in fom_ci_high],
            "ionic_norm": [float(x) for x in ion_g],
            "electronic_norm": [float(x) for x in elec_g],
            "kinetic_norm": [float(x) for x in kin_g],
        },
    }
