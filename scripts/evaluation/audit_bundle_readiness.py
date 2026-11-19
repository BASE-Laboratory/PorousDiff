"""Gate 0 pre-flight data, unit, split-leakage, and monotonicity audit for XCT bundles.

Verifies before spending GPU hours on Isambard-AI that a `bundle_manifest.json` + patch root satisfy:
  1. Single unambiguous mechanical load unit (`stress_kpa` present & finite across all scans, or a
     single consistent raw `load_unit` across all scans) and positive `effective_voxel_size_um`.
  2. Existing non-empty `train/patches.npy` and `val/patches.npy` for every non-repeat scan, plus
     mirror-free `val/tall_patches.npy` presence check on each `(material, beamline)` reference state.
  3. Zero spatial `train`/`val` Z-interval overlap when `split_z_intervals` metadata is present in
     `qa/segmentation_report.json`.
  4. Monotonic solid-fraction compaction (`phi_s(sigma_i+1) >= phi_s(sigma_i) - monotonic_slack`)
     within each `(material, beamline)` group to catch mislabeled folders or threshold drift early.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import (
    load_split_patches,
    resolve_patch_split_dir,
)


def audit_manifest_bundle(
    manifest_path: Path,
    data_root: Path | None = None,
    version: str | None = None,
    patch_size: int | None = None,
    include_repeats: bool = False,
    monotonic_slack: float = 0.015,
    require_tall_reference: bool = False,
) -> dict[str, object]:
    """Run Gate 0 readiness checks on `manifest_path` and return a structured audit report."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    eff_root = Path(data_root) if data_root is not None else manifest_path.parent
    eff_version = str(version or manifest.get("version", "carbon_calibrated_v1"))
    raw_scans: Sequence[Mapping[str, object]] = list(manifest.get("scans", []))

    errors: list[str] = []
    warnings: list[str] = []
    if not raw_scans:
        errors.append("Manifest contains zero scans.")
        return {
            "passed": False,
            "num_scans": 0,
            "errors": errors,
            "warnings": warnings,
            "groups": {},
        }

    eff_patch_size = (
        int(patch_size)
        if patch_size is not None
        else int(manifest.get("patch_size", raw_scans[0].get("patch_size", 96)))  # type: ignore[arg-type]
    )
    expected_shape = (eff_patch_size, eff_patch_size, eff_patch_size)

    active_scans = [
        sc
        for sc in raw_scans
        if include_repeats or not bool(sc.get("is_repeat_or_secondary", False))
    ]
    if not active_scans:
        errors.append("No active (non-repeat) scans found in manifest.")

    # 1. Load-unit consistency check
    has_all_kpa = bool(active_scans) and all(
        sc.get("stress_kpa") is not None and math.isfinite(float(sc["stress_kpa"]))  # type: ignore[arg-type]
        for sc in active_scans
    )
    raw_units = {str(sc.get("load_unit", "mv_per_v")) for sc in active_scans}
    if not has_all_kpa and len(raw_units) > 1:
        errors.append(
            f"Mixed uncalibrated load units {sorted(raw_units)} without finite stress_kpa on all scans."
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for sc in active_scans:
        st = str(sc.get("state", "")).strip()
        mat = str(sc.get("material", "virgin_carbon"))
        beam = str(sc.get("beamline", "primary"))
        vox = float(sc.get("effective_voxel_size_um", sc.get("voxel_size_um", 0.0)))  # type: ignore[arg-type]
        if not st:
            errors.append("Encountered scan entry with empty 'state' name.")
            continue
        if not (math.isfinite(vox) and vox > 0.0):
            errors.append(f"[{st}] Invalid effective_voxel_size_um={vox!r} (must be > 0).")

        lv = (
            float(sc["stress_kpa"])  # type: ignore[arg-type]
            if has_all_kpa
            else float(sc.get("load_mv_per_v", sc.get("load_raw_value", float("nan"))))  # type: ignore[arg-type]
        )
        if not math.isfinite(lv):
            errors.append(f"[{st}] Non-finite load/stress value {lv!r}.")

        pdir_hint = str(sc["patch_dir_name"]) if sc.get("patch_dir_name") else None
        train_dir = resolve_patch_split_dir(
            eff_root,
            st,
            eff_patch_size,
            eff_version,
            "train",
            patch_dir_name=pdir_hint,
            fallback_root=manifest_path.parent,
        )
        val_dir = resolve_patch_split_dir(
            eff_root,
            st,
            eff_patch_size,
            eff_version,
            "val",
            patch_dir_name=pdir_hint,
            fallback_root=manifest_path.parent,
        )

        val_sf = float("nan")
        n_train = 0
        n_val = 0
        try:
            tr_vols = load_split_patches(train_dir, expected_shape=expected_shape)
            n_train = int(tr_vols.shape[0])
        except Exception as exc:
            errors.append(f"[{st}] Failed to load train patches ({train_dir}): {exc}")

        try:
            va_vols = load_split_patches(val_dir, expected_shape=expected_shape)
            n_val = int(va_vols.shape[0])
            val_sf = float((va_vols >= 1).mean())
        except Exception as exc:
            errors.append(f"[{st}] Failed to load val patches ({val_dir}): {exc}")

        # Check Z-interval non-overlap if segmentation_report.json exists
        seg_report_path = val_dir.parent / "qa" / "segmentation_report.json"
        if seg_report_path.is_file():
            seg_rep = json.loads(seg_report_path.read_text())
            if str(seg_rep.get("split_mode_used", "z")) == "z":
                split_iv = seg_rep.get("split_z_intervals", {})
                tr_ivs = split_iv.get("train", [])
                va_ivs = split_iv.get("val", [])
                for ts, te in tr_ivs:
                    for vs, ve in va_ivs:
                        if int(ts) < int(ve) and int(vs) < int(te):
                            errors.append(
                                f"[{st}] Train/val Z-interval overlap detected: train=[{ts},{te}) vs val=[{vs},{ve})."
                            )

        has_tall = (val_dir / "tall_patches.npy").is_file()
        rep_sf = sc.get("clean_solid_fraction_mean")
        eff_sf = float(rep_sf) if (rep_sf is not None and math.isfinite(float(rep_sf))) else val_sf  # type: ignore[arg-type]

        grouped.setdefault((mat, beam), []).append(
            {
                "state": st,
                "load_value": lv,
                "solid_fraction": eff_sf,
                "n_train": n_train,
                "n_val": n_val,
                "has_tall_patches": has_tall,
                "num_phases": int(sc.get("num_phases", 2)),  # type: ignore[arg-type]
            }
        )

    groups_report: dict[str, object] = {}
    for (mat, beam), items in sorted(grouped.items()):
        items.sort(key=lambda d: float(d["load_value"]))  # type: ignore[arg-type]
        gkey = f"{mat}__{beam}"
        ref_item = items[0]
        if not bool(ref_item["has_tall_patches"]):
            msg = f"[{gkey}] Reference state {ref_item['state']!r} has no val/tall_patches.npy (baselines will use mirror tiling)."
            if require_tall_reference:
                errors.append(msg)
            else:
                warnings.append(msg)

        monotonic_ok = True
        for i in range(len(items) - 1):
            sf_cur = float(items[i]["solid_fraction"])  # type: ignore[arg-type]
            sf_nxt = float(items[i + 1]["solid_fraction"])  # type: ignore[arg-type]
            lv_cur = float(items[i]["load_value"])  # type: ignore[arg-type]
            lv_nxt = float(items[i + 1]["load_value"])  # type: ignore[arg-type]
            if (
                lv_nxt > lv_cur
                and math.isfinite(sf_cur)
                and math.isfinite(sf_nxt)
                and (sf_nxt + float(monotonic_slack) < sf_cur)
            ):
                monotonic_ok = False
                errors.append(
                    f"[{gkey}] Non-monotonic solid fraction between {items[i]['state']} "
                    f"(load={lv_cur:g}, phi_s={sf_cur:.4f}) and {items[i + 1]['state']} "
                    f"(load={lv_nxt:g}, phi_s={sf_nxt:.4f}; slack={monotonic_slack:.4f})."
                )

        groups_report[gkey] = {
            "material": mat,
            "beamline": beam,
            "num_states": len(items),
            "monotonic_solid_fraction": monotonic_ok,
            "reference_state": ref_item["state"],
            "reference_has_tall_patches": bool(ref_item["has_tall_patches"]),
            "states": items,
        }

    return {
        "passed": len(errors) == 0,
        "load_unit_resolved": "kPa" if has_all_kpa else (next(iter(raw_units)) if raw_units else "unknown"),
        "patch_size": eff_patch_size,
        "num_scans": len(active_scans),
        "errors": errors,
        "warnings": warnings,
        "groups": groups_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate 0 pre-flight bundle readiness checker (units, patches, zero-overlap, phi_s monotonicity)"
    )
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--include-repeats", action="store_true")
    parser.add_argument(
        "--monotonic-slack",
        type=float,
        default=0.015,
        help="Allowed solid-fraction dip tolerance across ascending loads before failing Gate 0",
    )
    parser.add_argument("--require-tall-reference", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    report = audit_manifest_bundle(
        manifest_path=args.manifest_json,
        data_root=args.data_root,
        version=args.version,
        patch_size=args.patch_size,
        include_repeats=args.include_repeats,
        monotonic_slack=args.monotonic_slack,
        require_tall_reference=args.require_tall_reference,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2))

    for w in report["warnings"]:  # type: ignore[attr-defined]
        print(f"WARNING: {w}", flush=True)
    if not report["passed"]:
        for err in report["errors"]:  # type: ignore[attr-defined]
            print(f"ERROR: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"Gate 0 PASSED: {report['num_scans']} scans across {len(report['groups'])} group(s) "  # type: ignore[arg-type]
        f"(unit={report['load_unit_resolved']}, patch_size={report['patch_size']}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
