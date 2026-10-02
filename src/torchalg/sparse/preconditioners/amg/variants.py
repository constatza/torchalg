"""Concrete sparse AMG preconditioner presets.

Sparse-CSR sibling of
``torchalg.preconditioners.implementations.amg.variants`` - same preset
shape (a plain factory function wiring concrete domain objects into
``AMGPreconditioner``, no new class, no behavior beyond constructor wiring -
see ``docs/plan.md``'s "Correction: validation placement, and inheritance
vs. composition for presets"), composed entirely from this package's own
sparse-native kernels/preconditioners plus the shared, format-agnostic
``torchalg.multigrid`` engine (``docs/plan.md``). This function *is* "the
sparse AMG orchestrator" - there is no separate sparse ``AMGPreconditioner``
class, since ``torchalg.multigrid.AMGPreconditioner`` is already
format-agnostic.

The one sparse-specific wiring detail relative to the dense presets: the
cycle's ``coarse_solver`` is ``dense_coarse_solve`` (``coarse_solve.py``)
instead of the bare ``torch.linalg.solve`` default, since the coarsest-level
matrix in a sparse hierarchy is still sparse CSR and ``torch.linalg.solve``
has no sparse-CSR direct-solve path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchalg.multigrid import AMGPreconditioner, VCycle, WCycle

from .coarse_solve import dense_coarse_solve
from .coarsening import AggregationCoarsening
from .smoothers import resolve_jacobi_default

if TYPE_CHECKING:
    import torch

    from torchalg.multigrid.protocols import MultigridSmoother


def vcycle_amg(
    matrix: torch.Tensor,
    theta: float = 0.25,
    omega: float | None = None,
    n_levels: int = 2,
    smoother_omega: float | None = None,
    n_pre: int = 2,
    n_post: int = 2,
    smoother: MultigridSmoother | None = None,
) -> AMGPreconditioner:
    """Build a sparse AMG preconditioner with V-cycle and smoothed aggregation (SA-AMG).

    Sparse-CSR sibling of
    ``torchalg.preconditioners.implementations.amg.variants.vcycle_amg``.
    Every non-coarsest hierarchy level stays sparse CSR throughout setup and
    solve; only the coarsest-level direct solve densifies
    (``dense_coarse_solve``).

    Args:
        matrix (torch.Tensor): System matrix A (n x n), SPD, sparse CSR.
        theta (float): Strength-of-connection threshold theta in (0, 1).
        omega (float | None): Damping of the Jacobi step that smooths the
            tentative prolongator. ``None`` (default) is ``(4/3) /
            rho(D^{-1}A)``, estimated once per level.
        n_levels (int): Number of hierarchy levels; must be at least 2.
        smoother_omega (float | None): Weighted-Jacobi relaxation damping.
            ``None`` (default) is ``1 / rho(D^{-1}A)`` per level.
        n_pre (int): Pre-smoothing steps (symmetric pre/post preserves SPD).
        n_post (int): Post-smoothing steps.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects sparse weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Returns:
        AMGPreconditioner: Wired with sparse smoothed-aggregation coarsening
        and a V-cycle, ready to ``apply()``.
    """
    return AMGPreconditioner(
        matrix=matrix,
        coarsening=AggregationCoarsening(theta=theta, omega=omega),
        cycle=VCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
            coarse_solver=dense_coarse_solve,
        ),
        n_levels=n_levels,
        linear=True,
    )


def wcycle_amg(
    matrix: torch.Tensor,
    theta: float = 0.25,
    omega: float | None = None,
    n_levels: int = 2,
    smoother_omega: float | None = None,
    n_pre: int = 2,
    n_post: int = 2,
    smoother: MultigridSmoother | None = None,
) -> AMGPreconditioner:
    """Build a sparse AMG preconditioner with W-cycle and smoothed aggregation (SA-AMG).

    Sparse-CSR sibling of
    ``torchalg.preconditioners.implementations.amg.variants.wcycle_amg`` -
    same parameters and wiring as ``vcycle_amg`` above, with ``WCycle``
    (two coarse-grid corrections per level, gamma = 2) instead of ``VCycle``.

    Args:
        matrix (torch.Tensor): System matrix A (n x n), SPD, sparse CSR.
        theta (float): Strength-of-connection threshold theta in (0, 1).
        omega (float | None): Prolongation-smoothing damping (same role as
            in ``vcycle_amg``).
        n_levels (int): Number of hierarchy levels; must be at least 2.
        smoother_omega (float | None): Relaxation damping (same role as in
            ``vcycle_amg``).
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects sparse weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Returns:
        AMGPreconditioner: Wired with sparse smoothed-aggregation coarsening
        and a W-cycle, ready to ``apply()``.
    """
    return AMGPreconditioner(
        matrix=matrix,
        coarsening=AggregationCoarsening(theta=theta, omega=omega),
        cycle=WCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
            coarse_solver=dense_coarse_solve,
        ),
        n_levels=n_levels,
        linear=True,
    )
