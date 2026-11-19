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

from porousdiff.metrics import CAUCHY_CROFTON_3D_FACTOR


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
            entry = buckets.setdefault(
                key,
                {"pore_d_eff_z": [], "solid_sigma_eff_z": [], "solid_hetero_sigma_eff_z": []},
            )
            if "pore" in phase or "ion" in phase:
                entry["pore_d_eff_z"].append(val)
            elif "hetero" in phase:
                entry["solid_hetero_sigma_eff_z"].append(val)
            elif "carbon" in phase or "solid" in phase or "elec" in phase:
                entry["solid_sigma_eff_z"].append(val)

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in buckets.items():
        agg: dict[str, float] = {}
        if vals["pore_d_eff_z"]:
            agg["pore_d_eff_z"] = float(np.mean(vals["pore_d_eff_z"]))
        if vals["solid_hetero_sigma_eff_z"]:
            agg["solid_hetero_sigma_eff_z"] = float(np.mean(vals["solid_hetero_sigma_eff_z"]))
        if vals["solid_sigma_eff_z"]:
            agg["solid_sigma_eff_z"] = float(np.mean(vals["solid_sigma_eff_z"]))
        if agg:
            lookup[key] = agg
    return lookup


def compute_transport_and_fom_record(
    phys_metrics: Mapping[str, float],
    openimpala_metrics: Mapping[str, float] | None = None,
    voxel_size_um: float = 1.0,
    strain_z: float | None = None,
    use_cauchy_crofton_ssa: bool = True,
) -> dict[str, float]:
    """Compute ionic, electronic, hydraulic, kinetic, and constriction observables for a 3D realization."""
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

    # Continuum pore-throat constriction factor beta_Z = (D_eff,Z^PDE * tau_geo,Z^2) / porosity
    if used_openimpala > 0.5 and porosity > 1e-6 and np.isfinite(tau_pore_z) and tau_pore_z >= 1.0:
        constriction_factor_z = float(d_eff_pore_z * (tau_pore_z**2) / porosity)
    else:
        constriction_factor_z = float("nan")

    # 2. Through-plane electronic effective conductivity sigma_eff,Z / sigma_0
    is_coated = coat_frac > 1e-6 or float(phys_metrics.get("ssa_pore_coating", 0.0)) > 1e-6
    wired_frac = float(
        phys_metrics.get("wired_coating_fraction", phys_metrics.get("solid_connected_fraction", 1.0))
    )
    solid_conn = float(phys_metrics.get("solid_connected_fraction", 1.0))
    elec_factor = wired_frac if is_coated else solid_conn

    if (
        openimpala_metrics
        and "solid_hetero_sigma_eff_z" in openimpala_metrics
        and np.isfinite(openimpala_metrics["solid_hetero_sigma_eff_z"])
    ):
        sigma_eff_solid_z = float(max(0.0, openimpala_metrics["solid_hetero_sigma_eff_z"]))
    elif (
        openimpala_metrics
        and "solid_sigma_eff_z" in openimpala_metrics
        and np.isfinite(openimpala_metrics["solid_sigma_eff_z"])
    ):
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

    # 4. Electrochemically active wired interfacial density (1/um; Cauchy-Crofton 2/3 default)
    ssa_coat_raw = float(phys_metrics.get("ssa_pore_coating", 0.0))
    ssa_coat_cc = float(
        phys_metrics.get("ssa_pore_coating_cc", ssa_coat_raw * CAUCHY_CROFTON_3D_FACTOR)
    )
    tpb = float(phys_metrics.get("tpb_density", 0.0))
    ssa_total_raw = float(phys_metrics.get("ssa_total", 0.0))
    ssa_total_cc = float(
        phys_metrics.get("ssa_total_cc", ssa_total_raw * CAUCHY_CROFTON_3D_FACTOR)
    )
    if is_coated:
        active_interface_raw = float(ssa_coat_raw * wired_frac + 0.05 * voxel_size_um * tpb * wired_frac)
        active_interface_cc = float(ssa_coat_cc * wired_frac + 0.05 * voxel_size_um * tpb * wired_frac)
    else:
        active_interface_raw = float(ssa_total_raw * solid_conn)
        active_interface_cc = float(ssa_total_cc * solid_conn)
    active_interface = active_interface_cc if use_cauchy_crofton_ssa else active_interface_raw

    raw_sz = strain_z if strain_z is not None else phys_metrics.get("strain_z", float("nan"))
    strain_val = float(raw_sz) if (raw_sz is not None and np.isfinite(raw_sz)) else float("nan")

    return {
        "solid_fraction": sf,
        "porosity": porosity,
        "strain_z": strain_val,
        "d_eff_pore_z": d_eff_pore_z,
        "constriction_factor_z": constriction_factor_z,
        "sigma_eff_solid_z": sigma_eff_solid_z,
        "permeability_ck_um2": perm_ck_um2,
        "active_wired_interface": active_interface,
        "active_wired_interface_cc": active_interface_cc,
        "active_wired_interface_raw": active_interface_raw,
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


def evaluate_newman_asr_curve(
    loads: np.ndarray,
    d_eff: np.ndarray,
    sigma_eff: np.ndarray,
    active_if_um: np.ndarray,
    solid_frac: np.ndarray,
    perm_um2: np.ndarray,
    z_pinch: np.ndarray,
    strain_z: np.ndarray,
    grid_loads: np.ndarray,
    uncompressed_thickness_cm: float = 0.30,
    kappa_bulk_s_cm: float = 0.05,
    sigma_bulk_s_cm: float = 2.0,
    r_ct_ohm_cm2: float = 2.5,
    r_mt_ohm_cm2: float = 0.05,
    r_contact_base_ohm_cm2: float = 0.04,
    pump_penalty_ohm_cm2_um2_cm: float = 1.5e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate dimensional 1D Newman & Tobias porous-electrode ASR (`Ohm * cm^2`) on `grid_loads`.

    Incorporates macroscopic compressed thickness shrinkage `L(sigma) = L_0 * (1 - strain_z)`
    (or uniaxial solid-conservation `L_0 * phi_s,min / phi_s(sigma)` when `strain_z` is absent),
    analytical 1D porous-electrode transmission-line polarization `R_tl(sigma)`, Nernst diffusive
    mass-transport concentration resistance `R_mt(sigma)`, clamping contact resistance `R_contact(sigma)`,
    and flow-through hydraulic pumping penalty `R_pump(sigma)`.
    """
    l0_cm = max(float(uncompressed_thickness_cm), 1e-3)
    ref_sf = float(max(np.min(solid_frac), 1e-3))

    thick_discrete = np.empty_like(loads, dtype=np.float64)
    for i in range(loads.size):
        sz = float(strain_z[i])
        if np.isfinite(sz) and 0.0 <= sz < 0.90:
            thick_discrete[i] = l0_cm * (1.0 - sz)
        else:
            ratio = np.clip(ref_sf / max(float(solid_frac[i]), 1e-3), 0.20, 1.0)
            thick_discrete[i] = l0_cm * ratio

    def _interp(vals: np.ndarray, floor: float) -> np.ndarray:
        if loads.size >= 2:
            return np.clip(PchipInterpolator(loads, vals)(grid_loads), floor, None)
        return np.full_like(grid_loads, max(float(vals[0]), floor), dtype=np.float64)

    d_g = _interp(d_eff, 1e-6)
    s_g = _interp(sigma_eff, 1e-6)
    a_cm_inv = _interp(active_if_um * 1e4, 1e-2)  # 1/um -> 1/cm
    sf_g = _interp(solid_frac, 1e-3)
    k_g = _interp(perm_um2, 1e-5)
    zp_g = _interp(z_pinch, 0.0)
    l_cm = np.clip(_interp(thick_discrete, 0.05 * l0_cm), 0.05 * l0_cm, l0_cm)

    r_ion = 1.0 / np.maximum(kappa_bulk_s_cm * d_g, 1e-6)  # Ohm * cm
    r_elec = 1.0 / np.maximum(sigma_bulk_s_cm * s_g, 1e-6)  # Ohm * cm
    r_kin = r_ct_ohm_cm2 / np.maximum(a_cm_inv, 1e-2)  # Ohm * cm^3

    r_sum = r_ion + r_elec
    nu = l_cm * np.sqrt(r_sum / np.maximum(r_kin, 1e-12))

    # Overflow-free evaluation of Newman & Tobias (1962) 1D transmission-line impedance
    r_tl = np.empty_like(grid_loads, dtype=np.float64)
    small = nu < 1e-3
    if np.any(small):
        r_tl[small] = r_kin[small] / np.maximum(l_cm[small], 1e-6) + l_cm[small] * r_sum[small] / 3.0
    if np.any(~small):
        nu_b = nu[~small]
        ri_b = r_ion[~small]
        re_b = r_elec[~small]
        rk_b = r_kin[~small]
        rs_b = r_sum[~small]
        l_b = l_cm[~small]
        e1 = np.exp(-nu_b)
        e2 = e1 * e1
        denom = np.maximum(1.0 - e2, 1e-15)
        coth_nu = (1.0 + e2) / denom
        csch_nu = 2.0 * e1 / denom
        r_tl[~small] = (l_b * ri_b * re_b) / rs_b + (
            np.sqrt(rk_b) * ((ri_b**2 + re_b**2) * coth_nu + 2.0 * ri_b * re_b * csch_nu)
        ) / (rs_b**1.5)

    r_mt = r_mt_ohm_cm2 / np.maximum(d_g, 1e-6)
    r_contact = r_contact_base_ohm_cm2 / np.maximum(sf_g + 2.0 * zp_g, 0.02)
    r_pump = pump_penalty_ohm_cm2_um2_cm / np.maximum(k_g * l_cm, 1e-4)
    asr_grid = r_tl + r_mt + r_contact + r_pump
    return asr_grid, l_cm


NEWMAN_REGIMES: dict[str, dict[str, float]] = {
    "kinetics_limited": {
        "kappa_bulk_s_cm": 0.08,
        "sigma_bulk_s_cm": 2.0,
        "r_ct_ohm_cm2": 6.0,
        "r_mt_ohm_cm2": 0.01,
        "pump_penalty_ohm_cm2_um2_cm": 5e-4,
    },
    "balanced": {
        "kappa_bulk_s_cm": 0.05,
        "sigma_bulk_s_cm": 2.0,
        "r_ct_ohm_cm2": 2.5,
        "r_mt_ohm_cm2": 0.05,
        "pump_penalty_ohm_cm2_um2_cm": 1.5e-3,
    },
    "mass_transport_limited": {
        "kappa_bulk_s_cm": 0.025,
        "sigma_bulk_s_cm": 2.0,
        "r_ct_ohm_cm2": 0.8,
        "r_mt_ohm_cm2": 0.22,
        "pump_penalty_ohm_cm2_um2_cm": 6e-3,
    },
}


SIMPLEX_WEIGHT_GRID: tuple[tuple[float, float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.35, 0.35, 0.30),
    (0.60, 0.20, 0.20),
    (0.20, 0.60, 0.20),
    (0.20, 0.20, 0.60),
    (0.50, 0.35, 0.15),
    (0.50, 0.15, 0.35),
    (0.35, 0.50, 0.15),
    (0.15, 0.50, 0.35),
    (0.35, 0.15, 0.50),
    (0.15, 0.35, 0.50),
)


def find_optimal_clamping_pressure(
    records_by_load: Mapping[float, Sequence[Mapping[str, float]]],
    weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    grid_points: int = 201,
    n_bootstrap: int = 200,
    seed: int = 42,
    norm_scales: tuple[float, float, float] | None = None,
    uncompressed_thickness_cm: float = 0.30,
    allow_mixed_estimators: bool = False,
) -> dict[str, object]:
    """Identify optimal clamping stress `sigma*` via dimensional 1D Newman ASR and harmonic FoM."""
    sorted_loads = sorted(float(k) for k in records_by_load.keys() if records_by_load[k])
    if not sorted_loads:
        raise ValueError("records_by_load must contain at least one non-empty load level")

    oi_per_load = [
        float(np.mean([float(r.get("used_openimpala", 0.0)) for r in records_by_load[lv]]))
        for lv in sorted_loads
    ]
    n_oi_loads = sum(1 for v in oi_per_load if v > 0.5)
    if 0 < n_oi_loads < len(sorted_loads) and not allow_mixed_estimators:
        raise ValueError(
            f"Refusing to mix OpenImpala ({n_oi_loads}/{len(sorted_loads)} loads) and geodesic surrogate "
            "transport estimates on the same clamping-pressure curve. Run OpenImpala across all loads "
            "or pass allow_mixed_estimators=True."
        )
    estimator_mode = (
        "openimpala"
        if n_oi_loads == len(sorted_loads)
        else ("mixed" if n_oi_loads > 0 else "geodesic_surrogate")
    )

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
    mean_sf = np.asarray(
        [np.mean([float(r.get("solid_fraction", 0.15)) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )
    mean_zpinch = np.asarray(
        [np.mean([float(r.get("z_pinch_contact_density", 0.0)) for r in records_by_load[lv]]) for lv in sorted_loads],
        dtype=np.float64,
    )
    mean_strain = np.empty_like(loads_arr)
    for i, lv in enumerate(sorted_loads):
        vals = [float(r.get("strain_z", float("nan"))) for r in records_by_load[lv]]
        fin = [v for v in vals if np.isfinite(v)]
        mean_strain[i] = float(np.mean(fin)) if fin else float("nan")

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

    weight_sweep_rows: list[dict[str, float]] = []
    weight_opt_loads: list[float] = []
    for w_cand in SIMPLEX_WEIGHT_GRID:
        w_fom_g, _, _, _, _ = _evaluate_harmonic_fom_curve(
            loads_arr,
            mean_d_eff,
            mean_sigma,
            mean_active,
            grid_loads,
            weights=w_cand,
            norm_scales=scales,
        )
        w_idx = int(np.argmax(w_fom_g))
        w_opt = float(grid_loads[w_idx])
        weight_opt_loads.append(w_opt)
        weight_sweep_rows.append(
            {
                "w_ionic": round(w_cand[0], 4),
                "w_electronic": round(w_cand[1], 4),
                "w_kinetic": round(w_cand[2], 4),
                "optimal_load": round(w_opt, 6),
                "optimal_fom": round(float(w_fom_g[w_idx]), 6),
            }
        )

    asr_grid, thick_grid_cm = evaluate_newman_asr_curve(
        loads_arr,
        mean_d_eff,
        mean_sigma,
        mean_active,
        mean_sf,
        mean_perm,
        mean_zpinch,
        mean_strain,
        grid_loads,
        uncompressed_thickness_cm=uncompressed_thickness_cm,
    )
    best_asr_idx = int(np.argmin(asr_grid))
    opt_load_asr = float(grid_loads[best_asr_idx])
    min_asr = float(asr_grid[best_asr_idx])

    regime_sensitivity: dict[str, dict[str, float]] = {}
    for reg_name, reg_params in NEWMAN_REGIMES.items():
        r_asr_g, _ = evaluate_newman_asr_curve(
            loads_arr,
            mean_d_eff,
            mean_sigma,
            mean_active,
            mean_sf,
            mean_perm,
            mean_zpinch,
            mean_strain,
            grid_loads,
            uncompressed_thickness_cm=uncompressed_thickness_cm,
            **reg_params,
        )
        r_idx = int(np.argmin(r_asr_g))
        regime_sensitivity[reg_name] = {
            "optimal_load_asr": round(float(grid_loads[r_idx]), 6),
            "min_asr_ohm_cm2": round(float(r_asr_g[r_idx]), 6),
            **reg_params,
        }

    # Non-parametric bootstrap across realizations at each discrete load
    rng = np.random.default_rng(seed)
    boot_opt_loads: list[float] = []
    boot_curves: list[np.ndarray] = []
    boot_opt_loads_asr: list[float] = []
    boot_asr_curves: list[np.ndarray] = []
    can_bootstrap = any(len(records_by_load[lv]) > 1 for lv in sorted_loads) and n_bootstrap > 0

    if can_bootstrap:
        for _ in range(int(n_bootstrap)):
            b_deff = np.empty_like(loads_arr)
            b_sig = np.empty_like(loads_arr)
            b_act = np.empty_like(loads_arr)
            b_sf = np.empty_like(loads_arr)
            b_perm = np.empty_like(loads_arr)
            b_zp = np.empty_like(loads_arr)
            for i, lv in enumerate(sorted_loads):
                recs = records_by_load[lv]
                n_r = len(recs)
                pick = rng.integers(0, n_r, size=n_r)
                b_deff[i] = np.mean([float(recs[j]["d_eff_pore_z"]) for j in pick])
                b_sig[i] = np.mean([float(recs[j]["sigma_eff_solid_z"]) for j in pick])
                b_act[i] = np.mean([float(recs[j]["active_wired_interface"]) for j in pick])
                b_sf[i] = np.mean([float(recs[j].get("solid_fraction", 0.15)) for j in pick])
                b_perm[i] = np.mean([float(recs[j].get("permeability_ck_um2", 1.0)) for j in pick])
                b_zp[i] = np.mean([float(recs[j].get("z_pinch_contact_density", 0.0)) for j in pick])
            b_fom, _, _, _, _ = _evaluate_harmonic_fom_curve(
                loads_arr, b_deff, b_sig, b_act, grid_loads, weights=weights, norm_scales=scales
            )
            boot_opt_loads.append(float(grid_loads[int(np.argmax(b_fom))]))
            boot_curves.append(b_fom)

            b_asr, _ = evaluate_newman_asr_curve(
                loads_arr,
                b_deff,
                b_sig,
                b_act,
                b_sf,
                b_perm,
                b_zp,
                mean_strain,
                grid_loads,
                uncompressed_thickness_cm=uncompressed_thickness_cm,
            )
            boot_opt_loads_asr.append(float(grid_loads[int(np.argmin(b_asr))]))
            boot_asr_curves.append(b_asr)

        ci_low, ci_high = (
            float(np.percentile(boot_opt_loads, 2.5)),
            float(np.percentile(boot_opt_loads, 97.5)),
        )
        curve_stack = np.stack(boot_curves, axis=0)
        fom_ci_low = np.percentile(curve_stack, 2.5, axis=0)
        fom_ci_high = np.percentile(curve_stack, 97.5, axis=0)

        asr_opt_ci_low, asr_opt_ci_high = (
            float(np.percentile(boot_opt_loads_asr, 2.5)),
            float(np.percentile(boot_opt_loads_asr, 97.5)),
        )
        asr_stack = np.stack(boot_asr_curves, axis=0)
        asr_ci_low = np.percentile(asr_stack, 2.5, axis=0)
        asr_ci_high = np.percentile(asr_stack, 97.5, axis=0)
    else:
        ci_low, ci_high = opt_load, opt_load
        fom_ci_low = fom_grid.copy()
        fom_ci_high = fom_grid.copy()
        asr_opt_ci_low, asr_opt_ci_high = opt_load_asr, opt_load_asr
        asr_ci_low = asr_grid.copy()
        asr_ci_high = asr_grid.copy()

    discrete_rows: list[dict[str, float]] = []
    discrete_fom, _, _, _, _ = _evaluate_harmonic_fom_curve(
        loads_arr, mean_d_eff, mean_sigma, mean_active, loads_arr, weights=weights, norm_scales=scales
    )
    discrete_asr, discrete_thick_cm = evaluate_newman_asr_curve(
        loads_arr,
        mean_d_eff,
        mean_sigma,
        mean_active,
        mean_sf,
        mean_perm,
        mean_zpinch,
        mean_strain,
        loads_arr,
        uncompressed_thickness_cm=uncompressed_thickness_cm,
    )
    for i, lv in enumerate(sorted_loads):
        recs = records_by_load[lv]
        beta_vals = [float(r.get("constriction_factor_z", float("nan"))) for r in recs]
        beta_fin = [v for v in beta_vals if np.isfinite(v)]
        mean_beta_z = float(np.mean(beta_fin)) if beta_fin else float("nan")
        discrete_rows.append(
            {
                "load_value": lv,
                "num_samples": float(len(recs)),
                "solid_fraction": float(mean_sf[i]),
                "porosity": float(np.mean([float(r.get("porosity", 1.0 - mean_sf[i])) for r in recs])),
                "compressed_thickness_cm": float(discrete_thick_cm[i]),
                "d_eff_pore_z": float(mean_d_eff[i]),
                "constriction_factor_z": mean_beta_z,
                "sigma_eff_solid_z": float(mean_sigma[i]),
                "permeability_ck_um2": float(mean_perm[i]),
                "active_wired_interface": float(mean_active[i]),
                "wired_coating_fraction": float(
                    np.mean([float(r.get("wired_coating_fraction", 1.0)) for r in recs])
                ),
                "tpb_density": float(np.mean([float(r.get("tpb_density", 0.0)) for r in recs])),
                "coating_pinch_fraction": float(
                    np.mean([float(r.get("coating_pinch_fraction", 0.0)) for r in recs])
                ),
                "fom_harmonic": float(discrete_fom[i]),
                "asr_ohm_cm2": float(discrete_asr[i]),
                "used_openimpala": float(np.mean([float(r.get("used_openimpala", 0.0)) for r in recs])),
            }
        )

    def _grid_resolution_stats(target_lv: float) -> tuple[float, float]:
        if loads_arr.size <= 1:
            return float(loads_arr[0]), 0.0
        idx_near = int(np.argmin(np.abs(loads_arr - target_lv)))
        gaps: list[float] = []
        if idx_near > 0:
            gaps.append(float(loads_arr[idx_near] - loads_arr[idx_near - 1]))
        if idx_near + 1 < loads_arr.size:
            gaps.append(float(loads_arr[idx_near + 1] - loads_arr[idx_near]))
        return float(loads_arr[idx_near]), 0.5 * max(gaps)

    near_fom_lv, fom_grid_hw = _grid_resolution_stats(opt_load)
    near_asr_lv, asr_grid_hw = _grid_resolution_stats(opt_load_asr)

    return {
        "estimator_mode": estimator_mode,
        "uncertainty_note": (
            "Bootstrap CIs quantify within-scan spatial patch variance; discrete scan spacing bounds sub-grid "
            "localization (see grid_spacing_halfwidth); continuum parameter sensitivity is reported in "
            "weight_sensitivity and newman_asr.regime_sensitivity."
        ),
        "optimal_load": round(opt_load, 6),
        "nearest_discrete_load": round(near_fom_lv, 6),
        "grid_spacing_halfwidth": round(fom_grid_hw, 6),
        "optimal_load_ci95": [round(ci_low, 6), round(ci_high, 6)],
        "optimal_fom": round(opt_fom, 6),
        "weights": {"ionic": weights[0], "electronic": weights[1], "kinetic": weights[2]},
        "weight_sensitivity": {
            "sigma_star_min": round(float(min(weight_opt_loads)), 6),
            "sigma_star_max": round(float(max(weight_opt_loads)), 6),
            "sigma_star_std": round(float(np.std(weight_opt_loads)), 6),
            "grid": weight_sweep_rows,
        },
        "norm_scales": {
            "d_eff_pore_z": scales[0],
            "sigma_eff_solid_z": scales[1],
            "active_wired_interface": scales[2],
        },
        "newman_asr": {
            "parameter_note": (
                "1D Newman-Tobias porous-electrode ASR (not parameter-free: parameterized by bulk conductivities "
                "kappa_0, sigma_0, interfacial kinetics R_ct, mass-transport R_mt, and pumping/contact terms; "
                "see regime_sensitivity)."
            ),
            "kinetic_area_convention": "cauchy_crofton_2_3",
            "uncompressed_thickness_cm": float(uncompressed_thickness_cm),
            "optimal_load_asr": round(opt_load_asr, 6),
            "nearest_discrete_load_asr": round(near_asr_lv, 6),
            "grid_spacing_halfwidth_asr": round(asr_grid_hw, 6),
            "optimal_load_asr_ci95": [round(asr_opt_ci_low, 6), round(asr_opt_ci_high, 6)],
            "min_asr_ohm_cm2": round(min_asr, 6),
            "thickness_cm_grid": [float(x) for x in thick_grid_cm],
            "asr_mean_ohm_cm2": [float(x) for x in asr_grid],
            "asr_ci95_low_ohm_cm2": [float(x) for x in asr_ci_low],
            "asr_ci95_high_ohm_cm2": [float(x) for x in asr_ci_high],
            "regime_sensitivity": regime_sensitivity,
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
