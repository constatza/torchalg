"""Residual-history consistency tests for torch-native CG solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable


def test_fcg_pcg_residual_history_length_match(
    diagonal_3x3_torch: torch.Tensor,
    rhs_ones_3_torch: torch.Tensor,
) -> None:
    """FCG and PCG log initial residual plus one value per iteration."""
    _, result_fcg = flexible_cg(
        diagonal_3x3_torch,
        rhs_ones_3_torch,
        m_max=-1,
        rtol=1e-10,
        trace_mode=TraceMode.FULL,
    )
    _, result_pcg = pcg(
        diagonal_3x3_torch,
        rhs_ones_3_torch,
        rtol=1e-10,
        trace_mode=TraceMode.FULL,
    )

    expected_len = result_fcg.iterations + 1
    assert result_fcg.iterations == result_pcg.iterations
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert len(result_fcg.residual_history_abs) == expected_len
    assert len(result_pcg.residual_history_abs) == expected_len
    assert torch.isclose(
        torch.tensor(result_fcg.residual_history_abs[0]),
        torch.tensor(result_pcg.residual_history_abs[0]),
        rtol=1e-12,
    )


def test_first_residual_is_r0_not_preconditioned_residual(
    diagonal_4x4_torch: torch.Tensor,
    rhs_progression_4_torch: torch.Tensor,
) -> None:
    """The first logged residual is ``||b - A @ x0||``, not ``||M^{-1}r0||``."""
    preconditioner = JacobiPreconditioner(diagonal_4x4_torch)

    _, result = flexible_cg(
        diagonal_4x4_torch,
        rhs_progression_4_torch,
        preconditioner=preconditioner,
        m_max=-1,
        rtol=1e-10,
        trace_mode=TraceMode.FULL,
    )

    expected_r0_norm = float(torch.linalg.norm(rhs_progression_4_torch))
    preconditioned_norm = float(torch.linalg.norm(preconditioner.apply(rhs_progression_4_torch)))
    assert result.residual_history_abs is not None
    assert result.residual_history_abs[0] == expected_r0_norm
    assert result.residual_history_abs[0] != preconditioned_norm


def test_fcg_pcg_histories_match_with_jacobi(
    diagonal_4x4_torch: torch.Tensor,
    rhs_progression_4_torch: torch.Tensor,
) -> None:
    """FCG and PCG produce the same residual history for fixed Jacobi preconditioning."""
    fcg_preconditioner = JacobiPreconditioner(diagonal_4x4_torch)
    pcg_preconditioner = JacobiPreconditioner(diagonal_4x4_torch)

    _, result_fcg = flexible_cg(
        diagonal_4x4_torch,
        rhs_progression_4_torch,
        preconditioner=fcg_preconditioner,
        m_max=-1,
        rtol=1e-10,
        trace_mode=TraceMode.FULL,
    )
    _, result_pcg = pcg(
        diagonal_4x4_torch,
        rhs_progression_4_torch,
        preconditioner=pcg_preconditioner,
        rtol=1e-10,
        trace_mode=TraceMode.FULL,
    )

    assert result_fcg.iterations == result_pcg.iterations
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert torch.allclose(
        torch.tensor(result_fcg.residual_history_abs),
        torch.tensor(result_pcg.residual_history_abs),
        rtol=1e-12,
    )


def test_fcg_pcg_histories_match_with_finite_truncation_window(
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """FCG(m_max=3) matches PCG to machine precision for a fixed preconditioner.

    Regression test for the ``PeriodicRestartOrthogonalization`` window-size
    off-by-one fixed in ``strategies/orthogonalization.py`` (see
    ``.claude/plan.md``): unlike every other FCG/PCG-equivalence test in this
    suite, ``m_max=3`` here is small relative to the ~15 iterations this
    system takes to converge, so the periodic-restart cycle
    (``m_i = (i mod m_max) + 1``) actually wraps around multiple times
    within one solve, exercising the finite-``m_max`` branch of
    ``PeriodicRestartOrthogonalization.orthogonalize`` that the fix changed
    (the other tests use ``m_max=-1``, which never reaches that branch, or
    ``m_max=1``, where the fixed and pre-fix formulas coincide).

    Caveat verified directly (not just asserted): this equivalence holds
    bit-for-bit under *both* the fixed formula and the pre-fix off-by-one
    formula, because a fixed SPD preconditioner makes CG's search
    directions automatically globally A-conjugate regardless of which
    historical terms a truncated Gram-Schmidt happens to include - their
    coefficients are ~0 either way. So this test is a genuine regression
    guard on the code path (shape/iteration-count/no-crash), but it cannot
    by itself distinguish the correct restart formula from the bug it
    replaced; that distinction is covered by
    ``test_periodic_restart_window_follows_notay_sawtooth`` in
    ``tests/solver/strategies/test_orthogonalization.py``, which asserts
    the window-size sequence directly.
    """
    n = 30
    matrix = poisson_1d_factory(n)
    rhs = torch.ones(n, dtype=matrix.dtype)
    x0 = torch.zeros(n, dtype=matrix.dtype)

    x_pcg, result_pcg = pcg(
        matrix,
        rhs,
        x0=x0,
        preconditioner=JacobiPreconditioner(matrix),
        rtol=1e-12,
        atol=1e-14,
        trace_mode=TraceMode.FULL,
    )
    x_fcg, result_fcg = flexible_cg(
        matrix,
        rhs,
        x0=x0,
        preconditioner=JacobiPreconditioner(matrix),
        m_max=3,
        rtol=1e-12,
        atol=1e-14,
        trace_mode=TraceMode.FULL,
    )

    assert result_fcg.iterations == result_pcg.iterations
    assert result_fcg.iterations > 3 * 3, "test requires the restart cycle to wrap at least twice"
    assert result_fcg.residual_history_abs is not None
    assert result_pcg.residual_history_abs is not None
    assert torch.allclose(
        torch.tensor(result_fcg.residual_history_abs),
        torch.tensor(result_pcg.residual_history_abs),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(x_fcg, x_pcg, rtol=1e-10, atol=1e-12)
