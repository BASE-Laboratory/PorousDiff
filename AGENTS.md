# Agent & Contributor Guidelines (`PorousDiff`)

Read [ECOSYSTEM.md](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The master copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Branch Governance

- **`main`:** Preserved untouched as MSc student Asif Ali's original archive (`0546679`). **Never merge into or overwrite `main` without James Le Houx's explicit instruction.**
- **`feature/paper-ready-conditional`:** Active `v0.5+` mainline maintained by James Le Houx. Open feature pull requests against `--base feature/paper-ready-conditional`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/PorousDiff`
- **Core package (`porousdiff/`):** **Layer 2 — Image & generative tools** (`BSD-3-Clause`).
  - **Allowed ecosystem dependencies:** `basecoat` (Layer 1) **only**.
  - **Banned imports inside `porousdiff/` (enforced by `python -m basecoat.layers` via `[tool.basecoat.layers]`):** `multimodal_benchmark`, `xct_research`, `openimpala`, `sms`, `openlsrx`, `x_fuse`, `braggtrack`, `porousdiff_workflows`, `zarr`, `numcodecs`.
- **Coupled workflow package (`porousdiff_workflows/` & `scripts/evaluation/`):** **Layer 4 — Coupled benchmark & beamline workflow** inside this repository, orchestrating `porousdiff` (Layer 2) + `basecoat` (Layer 1) + `OpenImpala v5.1` (Layer 3) + DLS K11 DIAD XRD clamping-pressure (`sigma*`) optimisation.

## Ownership Boundaries

- **`porousdiff/` owns:** Conditional 3D DDPM/DDIM models with FiLM continuous load/stress conditioning (`porousdiff/models.py`, `porousdiff/losses.py`), Lagrangian slab DDIM virtual compression (`porousdiff/virtual_compression.py`), morphological baselines (`porousdiff/baselines.py`), and 3D Cauchy–Crofton / chord / Euler / geodesic morphology (`porousdiff/metrics.py`).
- **Delegates to `basecoat` (`v0.2`):** Calibrated `Volume` / `VolumeSet` / `AxisSpec` (`unit="um"`), OME-Zarr v0.4 phase-labelled export and reading (`porousdiff.io.export_volume_ome_zarr`, `porousdiff.io.load_volume_array`), and TIFF reading/writing (`basecoat.read_tiff`, `basecoat.write_tiff`).
- **`porousdiff_workflows/` owns:** Coupled `OpenImpala v5.1` continuum PDE transport (`VolumeProtocol` with calibrated SI `spacing_in("m")`), K11 DIAD XRD correlation (`porousdiff_workflows/transport_and_xrd.py`), and series-resistance / Newman ASR clamping optimisation (`porousdiff_workflows/optimization.py`).
- **Deprecated for `v0.6` removal (retained in `v0.5.x` for manuscript reproducibility):** `export_openimpala_tiff` (superseded by `porousdiff.io.export_volume_ome_zarr` consumed natively by `OpenImpala v5.1` and `SMS`) and campaign material tables in `porousdiff/data.py` (`MATERIAL_TO_ID`).
