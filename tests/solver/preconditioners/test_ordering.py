"""Tests for preconditioner performance ordering.

Tests that compare multiple preconditioners to ensure they follow expected
performance ordering: ILU >= Jacobi >= Identity.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/test_ordering.py``). It exercises the
solver/preconditioner integration boundary now that ``flexible_cg`` and the
concrete preconditioners have landed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchalg.preconditioners.implementations import (
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
)

if TYPE_CHECKING:
    import torch


def test_preconditioner_ordering(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify preconditioner ordering: ILU <= Jacobi <= Identity by iteration count.

    Theory:
        Better approximations to A^{-1} lead to better conditioning and
        fewer iterations:
            ILU approximates A^{-1} better than Jacobi.
            Jacobi approximates A^{-1} better than Identity.

        Expected: iterations_ilu <= iterations_jacobi <= iterations_identity.
    """
    from torchalg import flexible_cg

    A, b, _ = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances
    identity = Identity()
    jacobi = JacobiPreconditioner(A)
    ilu = ILUPreconditioner(A)

    _, result_identity = flexible_cg(
        A,
        b,
        preconditioner=identity,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    _, result_jacobi = flexible_cg(
        A,
        b,
        preconditioner=jacobi,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    _, result_ilu = flexible_cg(
        A,
        b,
        preconditioner=ilu,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    assert result_identity.converged
    assert result_jacobi.converged
    assert result_ilu.converged

    assert result_ilu.iterations <= result_jacobi.iterations
    assert result_jacobi.iterations <= result_identity.iterations
