"""Analysis (PLAN §3, §10): survival curves, Cox models, BoN power laws, result tables."""

from magpie.analysis.aggregate import (
    ResultBundle,
    ResultRow,
    build_rows,
    rows_to_markdown,
    write_attempts,
    write_outcomes,
)
from magpie.analysis.survival import (
    ASRResult,
    KMCurve,
    PowerLawFit,
    asr_wilson,
    bootstrap_median_ci,
    fit_bon_power_law,
    kaplan_meier,
    logrank_test,
    median_survival,
)

__all__ = [
    "ASRResult", "KMCurve", "PowerLawFit", "asr_wilson", "bootstrap_median_ci",
    "fit_bon_power_law", "kaplan_meier", "logrank_test", "median_survival",
    "ResultBundle", "ResultRow", "build_rows", "rows_to_markdown",
    "write_attempts", "write_outcomes",
]

from magpie.analysis.cox import cox_margin  # noqa: E402

__all__ += ["cox_margin"]
