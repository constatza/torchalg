"""Concrete sparse AMG preconditioner presets.

Sparse-CSR sibling of
``torchalg.preconditioners.implementations.amg.variants`` - same preset
shape (a plain factory function wiring concrete domain objects into
``AMGPreconditioner``, no new class, no behavior beyond constructor wiring,
so composition rather than inheritance), composed entirely from this package's own
sparse-native kernels/preconditioners plus the shared, format-agnostic
``torchalg.multigrid`` engine. This function *is* "the
sparse AMG orchestrator" - there is no separate sparse ``AMGPreconditioner``
class, since ``torchalg.multigrid.AMGPreconditioner`` is already
format-agnostic.

The one sparse-specific wiring detail relative to the dense presets: the
cycle's ``coarse_solver_factory`` is ``dense_lu_coarse_solver`` (``coarse_solve.py``)
instead of the bare ``torch.linalg.solve`` default, since the coarsest-level
matrix in a sparse hierarchy is still sparse CSR and ``torch.linalg.solve``
has no sparse-CSR direct-solve path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchalg.multigrid import AMGPreconditioner, VCycle, WCycle

from .coarse_solve import dense_lu_coarse_solver
from .coarsening import AggregationCoarsening
from .smoothers import resolve_jacobi_default

if TYPE_CHECKING:
    from torchalg.multigrid.protocols import MultigridSmoother


def vcycle_amg(
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
    (``dense_lu_coarse_solver``).

    Cost:
        ``N = n_levels``, non-coarsest levels ``l`` with ``n_l`` rows, ``z_l`` stored
        entries and ``p_l`` prolongator entries, coarsest dimension ``c``,
        ``a = n_pre``, ``b = n_post``, ``k`` right-hand-side columns, default
        weighted-Jacobi smoother with spectral damping (an explicit ``smoother``
        replaces the sweep terms with its own Cost).
        Setup (``AMGPreconditioner.setup``): ``N - 1`` sparse
        ``AggregationCoarsening.build_transfer`` calls (``bt``, its Cost: strength,
        host aggregation, one Arnoldi estimate, ``A P0`` and the Galerkin product), one
        ``JacobiSmoother.prepare`` per level (``z_l + n_l`` flops, the Arnoldi
        estimate already memoized on the level matrix when both dampings are
        spectral), then the densified coarsest LU: ``W = sum_l (W_bt,l + z_l + n_l)
        + 2c^3 / 3``, ``D = sum_l D_bt,l + c`` (the host aggregation and the Arnoldi
        chain of each level, then the elimination), ``L = sum_l (L_bt,l + 19) + 3``
        and ``S = sum_l S_bt,l + 1`` (the LU singularity read).
        Apply, one V-cycle (``VCycle``'s Cost):
        ``W = sum_l [2(a + b) z_l + 3(a + b) n_l + 4 p_l] k + 2c^2 k``,
        ``D = sum_l (a + b + 3)(ceil(log2 m_l) + 4) + 2c`` (``m_l`` the largest row
        count; the dense LU substitutions set the depth when ``c`` is in the
        hundreds), ``L = sum_l [4(a + b) + 2] + 1`` and ``S = 0``.
        Roofline: about 0.15 flop per byte (memory-bound SpMVs).

    Args:
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
        and a V-cycle; call ``.setup(A)`` with the sparse CSR system matrix to build it.
    """
    return AMGPreconditioner(
        coarsening=AggregationCoarsening(theta=theta, omega=omega),
        cycle=VCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
            coarse_solver_factory=dense_lu_coarse_solver,
        ),
        n_levels=n_levels,
        linear=True,
    )


def wcycle_amg(
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

    Cost:
        Notation as in ``vcycle_amg``.
        Setup, identical to ``vcycle_amg``'s: ``W = sum_l (W_bt,l + z_l + n_l) + 2c^3 / 3``, ``D = sum_l D_bt,l + c``,
        ``L = sum_l (L_bt,l + 19) + 3``, ``S = sum_l S_bt,l + 1``.
        Apply, one W-cycle (``WCycle``'s Cost; level ``l`` visited ``2^l`` times, the
        coarsest ``2^(N-1)`` times):
        ``W = sum_l 2^l [(2(a + b) + 2) z_l + (3(a + b) + 2) n_l + 8 p_l] k
        + 2^(N-1) 2c^2 k``,
        ``D = sum_l 2^l (a + b + 5)(ceil(log2 m_l) + 4) + 2^(N-1) 2c``,
        ``L = sum_l 2^l [4(a + b) + 7] + 2^(N-1)`` and ``S = 0``.

    Args:
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
        and a W-cycle; call ``.setup(A)`` with the sparse CSR system matrix to build it.
    """
    return AMGPreconditioner(
        coarsening=AggregationCoarsening(theta=theta, omega=omega),
        cycle=WCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
            coarse_solver_factory=dense_lu_coarse_solver,
        ),
        n_levels=n_levels,
        linear=True,
    )
