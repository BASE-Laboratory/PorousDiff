"""Parse completed OpenImpala outputs into transport parity CSVs and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff_workflows.transport_and_xrd import parse_openimpala_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate OpenImpala TortuosityFD outputs across Real, Baselines & PorousDiff"
    )
    parser.add_argument("--suite-dir", type=Path, required=True, help="Path to openimpala_suite folder")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    out_csv = args.out_csv or (args.suite_dir / "openimpala_transport_parity.csv")
    rows = parse_openimpala_results(suite_dir=args.suite_dir, out_csv=out_csv)
    if not rows:
        print(f"No completed OpenImpala output files found in {args.suite_dir / 'outputs'}", flush=True)
        return

    print(f"Parsed {len(rows)} OpenImpala results -> {out_csv}", flush=True)

    # Plot through-plane effective diffusivity D_eff_z (if present) grouped by transport_phase
    phases = sorted({str(r["transport_phase"]) for r in rows})
    if any("d_eff_z" in r for r in rows):
        fig, axes = plt.subplots(1, len(phases), figsize=(5.2 * len(phases), 4.2), squeeze=False)
        for idx, ph in enumerate(phases):
            ax = axes[0, idx]
            sub = [r for r in rows if str(r["transport_phase"]) == ph and "d_eff_z" in r]
            methods = sorted({str(r["method"]) for r in sub})
            for m in methods:
                m_rows = [r for r in sub if str(r["method"]) == m]
                loads = sorted({float(r["load_value"]) for r in m_rows})  # type: ignore[arg-type]
                means = [
                    float(np.mean([float(r["d_eff_z"]) for r in m_rows if float(r["load_value"]) == lv]))  # type: ignore[arg-type]
                    for lv in loads
                ]
                style = "-o" if m == "real" else ("--s" if "squash" in m or "pillar" in m else "-^")
                ax.plot(loads, means, style, lw=1.6, label=m)
            ax.set_title(f"{ph} — D_eff,Z / D_0")
            ax.set_xlabel("Load / Stress")
            ax.set_ylabel("Effective transport factor")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = out_csv.parent / "openimpala_transport_parity.png"
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved OpenImpala parity figure -> {fig_path}", flush=True)


if __name__ == "__main__":
    main()
