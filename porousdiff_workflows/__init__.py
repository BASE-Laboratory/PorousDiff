"""Layer 4 coupled evaluation workflows: PorousDiff + basecoat + OpenImpala v5.1 + K11 DIAD XRD."""

from porousdiff_workflows.optimization import (
    NEWMAN_REGIMES,
    compute_transport_and_fom_record,
    evaluate_newman_asr_curve,
    load_openimpala_lookup,
    load_openimpala_records,
)
from porousdiff_workflows.transport_and_xrd import (
    compute_openimpala_volume_transport,
    correlate_xrd_with_microstructure,
    generate_openimpala_job_suite,
    has_openimpala_python,
    parse_openimpala_results,
    run_openimpala_python_suite,
    run_openimpala_rev_convergence,
    write_openimpala_ini,
)

__all__ = [
    "NEWMAN_REGIMES",
    "compute_openimpala_volume_transport",
    "compute_transport_and_fom_record",
    "correlate_xrd_with_microstructure",
    "evaluate_newman_asr_curve",
    "generate_openimpala_job_suite",
    "has_openimpala_python",
    "load_openimpala_lookup",
    "load_openimpala_records",
    "parse_openimpala_results",
    "run_openimpala_python_suite",
    "run_openimpala_rev_convergence",
    "write_openimpala_ini",
]
