"""Residual-history consistency tests for torch-native CG solvers."""

from __future__ import annotations

import torch

from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner


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
