"""Per-snapshot row-scaling schemes for weighted POD.

Scaling snapshot *rows* before an ordinary Euclidean SVD (see
``compute_pod_basis``'s ``row_scales`` parameter) reproduces a weighted
covariance ``sum_k scale_k^2 e_k e_k^T`` without ever changing the SVD's
inner product - unlike normalizing a snapshot's own magnitude (which merely
rescales that row uniformly and does not preferentially amplify any of its
components), *choosing* the per-row scale as a function of the snapshot (or
of the snapshot and the system matrix) really does change which directions
the resulting basis favors. Every function here returns only that length
``(n_samples,)`` scale vector; applying it is `compute_pod_basis`'s job.

References:
    - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos, V.
      (2022). AI-enhanced iterative solvers for accelerating the solution of
      large-scale parametrized systems. arXiv:2207.02543.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..amg._test_vectors import apply_jacobi_damping, apply_jacobi_damping_trajectory

__all__ = [
    "a_row_norms",
    "apply_jacobi_damping",
    "apply_jacobi_damping_trajectory",
    "l2_row_norms",
    "power_norm_scales",
    "smoother_persistence_scales",
]

_NEAR_ZERO_NORM_TOL = 1e-12
"""Norms below this are clamped before division/power, to avoid inf/nan for
a degenerate (near-zero) snapshot rather than propagating one silently."""


def l2_row_norms(snapshots: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of each snapshot row.

    Args:
        snapshots (torch.Tensor): Snapshot ensemble, shape (n_samples, n_dofs).

    Returns:
        torch.Tensor: Shape (n_samples,), ``||e_k||_2`` per row.
    """
    return torch.linalg.vector_norm(snapshots, dim=1)


def a_row_norms(snapshots: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """A-inner-product norm of each snapshot row, ``sqrt(e_k^T A e_k)``.

    Args:
        snapshots (torch.Tensor): Snapshot ensemble, shape (n_samples, n_dofs).
        matrix (torch.Tensor): SPD system matrix A, shape (n_dofs, n_dofs).

    Returns:
        torch.Tensor: Shape (n_samples,), ``||e_k||_A`` per row. Clamped to
            0 before the square root to absorb tiny negative values from
            floating-point asymmetry on an otherwise-SPD ``matrix``.
    """
    quadratic_form = (snapshots @ matrix * snapshots).sum(dim=1)
    return quadratic_form.clamp_min(0.0).sqrt()


def power_norm_scales(
    snapshots: torch.Tensor,
    *,
    matrix: torch.Tensor | None = None,
    metric: Literal["l2", "a"] = "l2",
    beta: float = 1.0,
) -> torch.Tensor:
    """Row scale ``||e_k||^(-beta)``, interpolating between raw and fully normalized.

    ``beta=0`` is a no-op (every snapshot voting equally regardless of its
    own magnitude); ``beta=1`` is full normalization by the chosen norm.
    Values in between are a bias-variance interpolation between the two.

    Args:
        snapshots (torch.Tensor): Snapshot ensemble, shape (n_samples, n_dofs).
        matrix (torch.Tensor | None): SPD system matrix, required when
            ``metric="a"``.
        metric (Literal["l2", "a"]): Which norm to scale by.
        beta (float): Normalization exponent; 0 disables scaling entirely,
            1 fully normalizes.

    Returns:
        torch.Tensor: Shape (n_samples,), per-row scale factor.

    Raises:
        ValueError: If ``metric="a"`` and ``matrix`` is not given.
    """
    if beta == 0.0:
        return torch.ones(snapshots.shape[0], dtype=snapshots.dtype, device=snapshots.device)

    match metric:
        case "l2":
            norms = l2_row_norms(snapshots)
        case "a":
            if matrix is None:
                raise ValueError("power_norm_scales: metric='a' requires the system matrix.")
            norms = a_row_norms(snapshots, matrix)

    return norms.clamp_min(_NEAR_ZERO_NORM_TOL).pow(-beta)


def smoother_persistence_scales(
    snapshots: torch.Tensor,
    matrix: torch.Tensor,
    *,
    omega: float = 0.67,
    steps: int = 5,
) -> torch.Tensor:
    """Row scale by how well each snapshot survives Jacobi smoothing.

    ``scale_k = ||G^steps e_k|| / ||e_k||`` where ``G = I - omega D^-1 A`` is
    the weighted-Jacobi error-propagation operator. Directions the smoother
    already removes efficiently contribute little to the resulting POD
    basis; directions that persist (the ones an actual multigrid V-cycle
    needs the coarse-grid correction to handle) contribute most - directly
    targeting the smoother the fitted coarsening will run alongside, instead
    of an arbitrary spatial norm.

    Args:
        snapshots (torch.Tensor): Snapshot ensemble, shape (n_samples, n_dofs).
        matrix (torch.Tensor): SPD system matrix A, shape (n_dofs, n_dofs).
        omega (float): Jacobi damping factor, matching
            ``JacobiSmoother``'s default.
        steps (int): Number of damping sweeps defining "persistence".

    Returns:
        torch.Tensor: Shape (n_samples,), per-row scale in [0, 1] (up to
            floating-point slack) for an SPD ``matrix`` and ``omega`` in its
            stable range.
    """
    damped = apply_jacobi_damping(snapshots, matrix, omega=omega, steps=steps)
    original_norms = l2_row_norms(snapshots).clamp_min(_NEAR_ZERO_NORM_TOL)
    damped_norms = l2_row_norms(damped)
    return damped_norms / original_norms
