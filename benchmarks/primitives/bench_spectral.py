"""Primitive: restarted-Arnoldi estimate of ``ρ(D⁻¹A)``, the Jacobi damping rule."""

from __future__ import annotations

import pytest
import torch

from benchmarks.cases import MatrixCase
from benchmarks.conftest import Record
from benchmarks.harness import Level
from benchmarks.primitives._support import case_params
from torchalg.preconditioners.implementations.amg._jacobi_omega import _seeded_draw
from torchalg.utils.spectral import approximate_spectral_radius

ACCURACY_RTOL = 0.05
"""Arnoldi tolerance is 1e-2 on the Ritz pair; 5% against the exact radius is the bar."""
ACCURACY_CHECK_N = 484


def _exact_radius(case: MatrixCase, dtype: torch.dtype) -> float:
    """``ρ(D⁻¹A)`` from the symmetric similar matrix ``D^{-1/2} A D^{-1/2}``."""
    dense = case.dense(dtype, torch.device("cpu"))
    inv_sqrt = torch.diagonal(dense).rsqrt()
    symmetric = inv_sqrt.unsqueeze(1) * dense * inv_sqrt.unsqueeze(0)
    return float(torch.linalg.eigvalsh(symmetric)[-1])


@pytest.mark.parametrize("format_name", ["dense", "csr"])
def test_spectral_radius(
    format_name: str,
    case: MatrixCase,
    dtype: torch.dtype,
    request: pytest.FixtureRequest,
    record: Record,
) -> None:
    operator = request.getfixturevalue("scaled_dense" if format_name == "dense" else "scaled_csr")
    estimate = approximate_spectral_radius(operator, _seeded_draw)
    if case.n == ACCURACY_CHECK_N:
        exact = _exact_radius(case, dtype)
        assert abs(float(estimate) - exact) <= ACCURACY_RTOL * exact
    record(
        Level.PRIMITIVE,
        "spectral.arnoldi_radius",
        case_params(case, format=format_name),
        lambda: approximate_spectral_radius(operator, _seeded_draw),
    )
