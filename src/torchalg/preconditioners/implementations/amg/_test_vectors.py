"""Jacobi error-damping helpers shared by POD weighting and snapshot generation.

Lives in ``amg`` (not ``pod``) because ``pod`` depends on ``amg`` and never the
reverse; ``pod.weighting`` re-exports these so its public surface is unchanged.
"""

from __future__ import annotations

import torch

from .smoothers import JacobiSmoother


def apply_jacobi_damping(
    vectors: torch.Tensor,
    matrix: torch.Tensor,
    *,
    omega: float | torch.Tensor | None = None,
    steps: int,
) -> torch.Tensor:
    """Apply ``steps`` weighted-Jacobi error-damping sweeps to a batch of vectors.

    Reuses ``JacobiSmoother`` (``rhs=0``, so ``smooth`` reduces to the pure
    error-propagation map ``v <- v - omega * D^{-1} A v``) rather than
    reimplementing the Jacobi iteration: the smoother is prepared once for
    ``matrix`` (the spectral radius, if ``omega`` is ``None``, is estimated
    there) and all vectors are damped together as the columns of one block. This is what
    ``smoother_persistence_scales`` (weighting) calls, and what any
    snapshot-generation strategy that only needs the final "algebraically
    smooth" probe vector should call. A strategy that needs to select an
    arbitrary window of intermediate sweeps instead of just the final one
    should call ``apply_jacobi_damping_trajectory`` below.

    Args:
        vectors (torch.Tensor): Batch of vectors to damp, shape
            (n_vectors, n_dofs).
        matrix (torch.Tensor): SPD system matrix A, shape (n_dofs, n_dofs).
        omega (float | torch.Tensor | None): Jacobi damping factor; ``None``
            (default) is the relaxation rule ``1 / rho(D^-1 A)`` (estimated
            once, when the smoother is prepared for ``matrix``), matching
            ``JacobiSmoother``.
        steps (int): Number of damping sweeps.

    Returns:
        torch.Tensor: Damped vectors, same shape as ``vectors``.
    """
    prepared = JacobiSmoother(omega=omega).prepare(matrix)
    columns = vectors.T
    return prepared.smooth(torch.zeros_like(columns), columns, steps).T.contiguous()


def apply_jacobi_damping_trajectory(
    vectors: torch.Tensor,
    matrix: torch.Tensor,
    *,
    omega: float | torch.Tensor | None = None,
    steps: int,
) -> torch.Tensor:
    """Apply weighted-Jacobi damping, capturing every intermediate sweep.

    Same error-propagation map as ``apply_jacobi_damping`` (``rhs=0``, so
    ``smooth`` reduces to ``v <- v - omega * D^{-1} A v``), but returns the
    full ``steps + 1``-length trajectory per vector instead of only the
    final iterate - for callers that need to select an arbitrary window of
    sweeps rather than always the last one. Row ``steps`` of the result
    equals ``apply_jacobi_damping``'s own output exactly.

    Exactly ``steps`` sweeps are performed per vector, never more,
    regardless of how much of the returned trajectory a caller ultimately
    keeps: this loops one prepared-smoother sweep ``steps`` times rather than
    reimplementing the sweep or over-running it.

    Args:
        vectors (torch.Tensor): Batch of vectors to damp, shape
            (n_vectors, n_dofs).
        matrix (torch.Tensor): SPD system matrix A, shape (n_dofs, n_dofs).
        omega (float | torch.Tensor | None): Jacobi damping factor; ``None``
            (default) is the relaxation rule ``1 / rho(D^-1 A)`` (estimated
            once, when the smoother is prepared for ``matrix``), matching
            ``JacobiSmoother``.
        steps (int): Number of damping sweeps to run.

    Returns:
        torch.Tensor: Trajectories, shape (n_vectors, steps + 1, n_dofs).
    """
    prepared = JacobiSmoother(omega=omega).prepare(matrix)
    x = vectors.T
    zero_rhs = torch.zeros_like(x)
    history = [x]
    for _ in range(steps):
        x = prepared.smooth(zero_rhs, x, 1)
        history.append(x)
    return torch.stack(history).permute(2, 0, 1).contiguous()
