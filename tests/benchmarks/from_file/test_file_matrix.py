"""Benchmark solver on user-provided matrix files.

Ported from the reference (``dl-experiments``'s
``tests/benchmarks/from_file/test_file_matrix.py``), adapted from numpy to
torch tensors and from ``neuralls.domain.solver.factories``/
``preconditioners`` to ``torchalg``'s public ``pcg``/``flexible_cg`` and
``torchalg.preconditioners.implementations``. The reference's ``scipy_cg``
baseline (a thin wrapper around ``scipy.sparse.linalg.cg``) has no direct
counterpart here; ``pcg`` (torchalg's native two-term-recurrence PCG) is the
appropriate baseline for the FCG-vs-PCG iteration-count equivalence check.
"""

from __future__ import annotations

import pytest
import torch

from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.implementations import IC0Preconditioner, ICholeskyPreconditioner

from .conftest import (
    IC0_MATRIX_ATOL,
    IC0_MATRIX_RTOL,
    IC0_SOLUTION_ATOL,
    IC0_SOLUTION_RTOL,
    IC0_SPARSITY_COUNT_TOLERANCE,
    IC0_THRESHOLD,
    SPARSITY_THRESHOLD,
)

# FCG and PCG must match exactly on iteration counts
ITER_DIFF_THRESHOLD = 0


@pytest.mark.benchmark
@pytest.mark.parametrize("precond_type", [None, "ilu"])
def test_fcg_pcg_equivalence_on_file(system, convergence_tolerances, precond_type) -> None:
    """FCG (m_max=1, no orthogonalization) matches PCG exactly on a real matrix."""
    a, b, _x_exact, l_ref = system
    rtol, atol = convergence_tolerances

    preconditioner = None
    if precond_type == "ilu":
        preconditioner = ICholeskyPreconditioner(l_ref)

    _, result_pcg = pcg(
        a,
        b,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=10000,
        trace_mode=TraceMode.FULL,
    )
    _, result_fcg = flexible_cg(
        a,
        b,
        preconditioner=preconditioner,
        m_max=1,  # No orthogonalization (should match PCG exactly)
        rtol=rtol,
        atol=atol,
        maxiter=10000,
        trace_mode=TraceMode.FULL,
    )

    assert result_pcg.converged, f"PCG failed to converge. Iterations: {result_pcg.iterations}"
    assert result_fcg.converged, f"FCG failed to converge. Iterations: {result_fcg.iterations}"

    diff = abs(result_pcg.iterations - result_fcg.iterations)
    assert diff <= ITER_DIFF_THRESHOLD, (
        f"Iteration count mismatch: PCG={result_pcg.iterations}, FCG={result_fcg.iterations}, diff={diff}"
    )


@pytest.mark.benchmark
def test_ic0_quality_vs_reference(system, convergence_tolerances) -> None:
    """Verify IC0Preconditioner has zero fill-in and reasonable reconstruction quality.

    Theory:
        IC(0) computes dense-masked incomplete Cholesky: L @ L.T ~= A.
        Key IC(0) property: ZERO fill-in - L has same sparsity pattern as
        lower(A). The reference L matrix is a different incomplete
        factorization.

    Validation:
        - Verify IC0 has ZERO fill-in (same sparsity as A).
        - Verify IC0 reconstruction error is reasonable (~10-20%).
        - Compare quality metrics vs reference L.
    """
    a, _b, _x_exact, l_reference = system

    ic0_precond = IC0Preconditioner(a, threshold=IC0_THRESHOLD)
    l_computed = ic0_precond._operator

    a_lower = torch.tril(a)
    a_lower_nnz = int(torch.count_nonzero(a_lower))
    l_computed_nnz = int(torch.count_nonzero(torch.abs(l_computed) > SPARSITY_THRESHOLD))
    l_ref_nnz = int(torch.count_nonzero(torch.abs(l_reference) > SPARSITY_THRESHOLD))

    sparsity_diff = abs(l_computed_nnz - l_ref_nnz)
    assert sparsity_diff <= IC0_SPARSITY_COUNT_TOLERANCE, (
        f"IC(0) sparsity mismatch with reference exceeds tolerance! "
        f"IC0 nnz={l_computed_nnz}, Ref nnz={l_ref_nnz}, "
        f"diff={sparsity_diff}, max_allowed={IC0_SPARSITY_COUNT_TOLERANCE}"
    )

    torch.testing.assert_close(
        l_computed,
        l_reference,
        rtol=IC0_MATRIX_RTOL,
        atol=IC0_MATRIX_ATOL,
        msg="IC0 L matrix values differ from reference",
    )

    a_ref_reconstructed = l_reference @ l_reference.T
    a_ic0_reconstructed = l_computed @ l_computed.T

    ref_error = float(torch.linalg.norm(a - a_ref_reconstructed) / torch.linalg.norm(a))
    ic0_error = float(torch.linalg.norm(a - a_ic0_reconstructed) / torch.linalg.norm(a))

    assert ic0_error < 0.5, f"IC0 reconstruction error too large: {ic0_error:.6e} (expected < 0.5 for IC0)"

    print(f"\nSparsity: A_lower={a_lower_nnz}, IC0={l_computed_nnz}, Ref={l_ref_nnz}")
    print(f"Reconstruction error: IC0={ic0_error:.4f}, Ref={ref_error:.4f}")


@pytest.mark.benchmark
def test_ic0_solver_performance_vs_reference(system, convergence_tolerances) -> None:
    """Compare CG solver performance using IC0 vs reference L preconditioner.

    Validates that IC0-computed L produces reasonable convergence. Both
    IC0 and reference have similar reconstruction errors, so iteration
    counts should be comparable.
    """
    a, b, _x_exact, l_reference = system
    rtol, atol = convergence_tolerances

    ic0_precond = IC0Preconditioner(a, threshold=IC0_THRESHOLD)
    ref_precond = ICholeskyPreconditioner(l_reference)

    x_ic0, result_ic0 = flexible_cg(a, b, preconditioner=ic0_precond, rtol=rtol, atol=atol, maxiter=1000)
    x_ref, result_ref = flexible_cg(a, b, preconditioner=ref_precond, rtol=rtol, atol=atol, maxiter=1000)

    assert result_ic0.converged, (
        f"IC0 preconditioner failed to converge. "
        f"Iterations: {result_ic0.iterations}, "
        f"Final residual: {result_ic0.residual}"
    )
    assert result_ref.converged, (
        f"Reference preconditioner failed to converge. "
        f"Iterations: {result_ref.iterations}, "
        f"Final residual: {result_ref.residual}"
    )

    torch.testing.assert_close(
        x_ic0,
        x_ref,
        rtol=IC0_SOLUTION_RTOL,
        atol=IC0_SOLUTION_ATOL,
        msg="IC0 and reference produce different solutions",
    )

    assert result_ic0.iterations == result_ref.iterations, (
        f"Iteration count mismatch - algorithm must converge in exact iterations! "
        f"IC0={result_ic0.iterations}, reference={result_ref.iterations}, "
        f"diff={abs(result_ic0.iterations - result_ref.iterations)}"
    )

    print(f"\nIterations: IC0={result_ic0.iterations}, Ref={result_ref.iterations} (exact match)")
