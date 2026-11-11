"""Constants and parameter helpers shared by the primitive benchmarks."""

from __future__ import annotations

from benchmarks.cases import MatrixCase
from benchmarks.harness import ParamValue

DENSE_N_CAP = 3000
"""Largest ``n`` at which O(n²)-temporary or GEMM-heavy dense operations are measured.

Above it a dense float64 operand is 72 MB or more per copy and several ops materialize
four to six of them (or run an n³ GEMM), which does not fit the suite's time and memory
budget. The paper's case is n ≈ 3000, so the cap keeps every relevant dense point.
"""

CURRENT_VARIANT = "current"
"""Marks the baseline implementation; later phases record their own variant label."""

STRENGTH_THETA = 0.02
"""Strength threshold at which every stored edge of both benchmark families is strong."""

TEST_VECTOR_COUNT = 8
"""Number of AMG setup test vectors ``k``."""


def case_params(case: MatrixCase, **extra: ParamValue) -> dict[str, ParamValue]:
    """Sweep coordinates identifying a measurement on a matrix case."""
    return {
        "matrix": case.id,
        "n": case.n,
        "nnz": case.nnz,
        "variant": CURRENT_VARIANT,
        **extra,
    }
