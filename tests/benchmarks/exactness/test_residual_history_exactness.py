"""Benchmark: PCG vs scipy CG vs FCG(inf), step-by-step residual-norm exactness.

Complements ``test_pcg_scipy_equivalence.py`` (final solution/iteration-count
only) and ``test_fcg_pcg_equivalence.py`` (allows small FCG/PCG iteration
slack) with a single, stricter check: for a fixed, unpreconditioned SPD
system, ``pcg``, ``flexible_cg(m_max=-1)``, and the scipy CG oracle all
produce the *same iteration count* and the *same residual norm at every
iteration*, not just at the end. This is the direct numerical consequence of
Notay (2000): PCG and FCG(inf) are identical in exact arithmetic for a fixed
preconditioner, and both implement the same Hestenes-Stiefel two-term
recurrence scipy's ``cg`` does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.benchmarks.exactness.conftest import MATRIX_TYPES, generate_spd_system
from tests.support.scipy_reference.preconditioners.implementations.identity import (
    Identity as SciPyIdentity,
)
from tests.support.scipy_reference.scipy_wrapper import SciPyCGSolver
from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode

HISTORY_RTOL = 1e-10
HISTORY_ATOL = 1e-12


@pytest.mark.benchmark
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_pcg_fcg_scipy_residual_histories_match_step_by_step(
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
) -> None:
    """PCG, FCG(inf), and scipy CG log the identical residual norm at every iteration."""
    matrix_np, rhs_np, _, _ = generate_spd_system(size=100, matrix_type=matrix_type)
    matrix = torch.from_numpy(matrix_np).clone()
    rhs = torch.from_numpy(rhs_np).clone()
    rtol, atol = convergence_tolerances

    _, result_scipy = SciPyCGSolver(preconditioner=SciPyIdentity()).solve(
        matrix_np,
        rhs_np,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL.value,
    )
    _, result_pcg = pcg(
        matrix,
        rhs,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL,
    )
    _, result_fcg = flexible_cg(
        matrix,
        rhs,
        m_max=-1,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL,
    )

    assert result_scipy.converged is True
    assert result_pcg.converged is True
    assert result_fcg.converged is True
    assert result_pcg.iterations == result_scipy.iterations
    assert result_fcg.iterations == result_scipy.iterations

    history_scipy = np.asarray(result_scipy.residual_history_abs)
    history_pcg = np.asarray(result_pcg.residual_history_abs)
    history_fcg = np.asarray(result_fcg.residual_history_abs)

    assert history_scipy.shape == history_pcg.shape == history_fcg.shape

    np.testing.assert_allclose(
        history_pcg,
        history_scipy,
        rtol=HISTORY_RTOL,
        atol=HISTORY_ATOL,
        err_msg="PCG residual history diverges from scipy CG at some iteration",
    )
    np.testing.assert_allclose(
        history_fcg,
        history_scipy,
        rtol=HISTORY_RTOL,
        atol=HISTORY_ATOL,
        err_msg="FCG(inf) residual history diverges from scipy CG at some iteration",
    )
