"""Adaptive smoothed aggregation (alpha-SA), a torch port of PyAMG's ``adaptive_sa_solver``.

Follows PyAMG 5.3.0's ``pyamg/aggregation/adaptive.py`` step by step
(``adaptive_sa_solver``, ``initial_setup_stage`` = Algorithm 3 and
``general_setup_stage`` = Algorithm 4 of Brezina et al. 2005), on dense
tensors. The pieces it composes are ports of the PyAMG kernels they call:
``fit_candidates`` (tentative prolongator), ``symmetric_strength_of_connection``
and ``standard_aggregation`` (node graph), ``jacobi_prolongation_smoother``
(damping ``omega / rho(D^-1 A)`` with an Arnoldi estimate of ``rho``),
symmetric Gauss-Seidel relaxation, and a V(1,1)-cycle with a pseudo-inverse
coarse solve.

Aggregation note: ``standard_aggregation``'s second pass attaches a node to the
first already-aggregated neighbour *in stored order*. PyAMG's order on coarse
levels is whatever the sparse product leaves (often unsorted) and changes its
result; this port uses ascending column order, i.e. PyAMG's result on a
sorted-index copy of the same graph. Both are valid aggregations.

Coarse levels are block-structured exactly as in PyAMG: a level built from
``k`` candidates has ``k`` dofs per node (one node per aggregate of the level
above), and aggregation/strength act on the node graph.

Not ported (PyAMG options outside the default algorithm): ``improvement_iters``,
``eliminate_local``, ``epsilon``/``pdef`` (only used by a branch PyAMG
disables), non-default *setup* smoothers/strength/aggregation/coarse solvers,
complex and nonsymmetric matrices, and the ``work`` complexity counter. The
public preconditioner's solve-time smoother is independently configurable.

References:
    - Brezina, M., Falgout, R., MacLachlan, S., Manteuffel, T., McCormick,
      S., & Ruge, J. (2005). Adaptive smoothed aggregation (alpha-SA)
      multigrid. SIAM Review, 47(2), 317-346.
    - PyAMG 5.3.0, ``pyamg/aggregation/adaptive.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchalg.multigrid import PreparedFineLevel, PreparedHierarchy
from torchalg.preconditioners.base import PreconditionerContext, PreconditionerNotReadyError
from torchalg.utils.spectral import approximate_spectral_radius
from torchalg.utils.tentative import fit_candidates

from ._aggregation import standard_aggregation
from ._node_strength import node_strength
from ._presets import (
    GS_SETUP_CYCLE,
    GS_SETUP_SMOOTHER,
    PrebuiltCoarsening,
    prebuilt_cycle,
    seeded_draw,
)
from ._prolongation import jacobi_prolongation, make_bridge
from ._relaxation import prepare_symmetric_gauss_seidel
from .amg import AMGPreconditioner
from .hierarchy import MultigridHierarchy, assemble_hierarchy
from .smoothers import resolve_jacobi_default
from .transfer import DenseTransferOperator

if TYPE_CHECKING:
    from typing import Self

    from .protocols import MultigridSmoother, PreparedSmoother

_SOLVE_TOLERANCE = 1e-20
"""Absolute residual tolerance of the trial cycles (``ml.solve(..., tol=1e-20)``)."""


@dataclass
class _Level:
    """Mutable working level (PyAMG's ``MultilevelSolver.Level``) used during setup."""

    A: torch.Tensor
    B: torch.Tensor
    dofs: int = 1
    strength: torch.Tensor | None = None
    aggregate: torch.Tensor | None = None
    T: torch.Tensor | None = None
    P: torch.Tensor | None = None


@dataclass(frozen=True)
class _Predefined:
    """Aggregation and strength kept from an earlier build (PyAMG's ``'predefined'``)."""

    aggregates: tuple[torch.Tensor, ...]
    strengths: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _Params:
    """Setup parameters shared by every stage."""

    theta: float
    omega: float
    candidate_iters: int
    max_levels: int
    max_coarse: int
    draw: Callable[[int], torch.Tensor]

    def spectral_radius(self, matrix: torch.Tensor) -> torch.Tensor:
        """Arnoldi estimate of ``rho`` drawing from the shared random stream.

        Args:
            matrix (torch.Tensor): Square matrix.

        Returns:
            torch.Tensor: Estimated spectral radius, 0-d.
        """
        return approximate_spectral_radius(matrix, self.draw)

    def random_vector(self, like: torch.Tensor, n: int) -> torch.Tensor:
        """Uniform ``[0, 1)`` vector of length ``n`` with ``like``'s dtype/device.

        Args:
            like (torch.Tensor): Tensor providing dtype and device.
            n (int): Length.

        Returns:
            torch.Tensor: Random vector.
        """
        return self.draw(n).to(dtype=like.dtype, device=like.device)


@dataclass(frozen=True)
class AdaptiveSAResult:
    """Result of the adaptive setup.

    Attributes:
        matrices (tuple[torch.Tensor, ...]): Level matrices ``A_l``, finest first.
        prolongations (tuple[torch.Tensor, ...]): ``P_l`` mapping level ``l+1``
            to level ``l`` (``R = P^T``).
        candidates (torch.Tensor): Fine-level near-null-space candidates,
            shape ``(n, num_candidates)``.
        aggregates (tuple[torch.Tensor, ...]): Node aggregation of each
            coarsened level (``-1`` = isolated node).
        strengths (tuple[torch.Tensor, ...]): Boolean node strength graph of
            each coarsened level.
    """

    matrices: tuple[torch.Tensor, ...]
    prolongations: tuple[torch.Tensor, ...]
    candidates: torch.Tensor
    aggregates: tuple[torch.Tensor, ...]
    strengths: tuple[torch.Tensor, ...]

    @property
    def hierarchy(self) -> MultigridHierarchy:
        """The multigrid hierarchy for the engine's cycles.

        Returns:
            MultigridHierarchy: Levels with dense transfer operators.
        """
        return assemble_hierarchy(
            self.matrices,
            [DenseTransferOperator(prolongation) for prolongation in self.prolongations],
        )


def _built(value: torch.Tensor | None) -> torch.Tensor:
    """Return a level attribute that must already have been computed.

    Args:
        value (torch.Tensor | None): ``_Level`` attribute set during setup.

    Returns:
        torch.Tensor: ``value``.

    Raises:
        RuntimeError: If the level has not been built yet.
    """
    if value is None:
        raise RuntimeError("level attribute used before it was built")
    return value


def _relax(relaxation: PreparedSmoother, x: torch.Tensor, params: _Params) -> torch.Tensor:
    """``candidate_iters`` symmetric Gauss-Seidel iterations on ``A x = 0``.

    Args:
        relaxation (PreparedSmoother): Gauss-Seidel bound to the level matrix.
        x (torch.Tensor): Vector to relax.
        params (_Params): Setup parameters.

    Returns:
        torch.Tensor: Relaxed vector.
    """
    return relaxation.smooth(torch.zeros_like(x), x, params.candidate_iters)


def _hierarchy_of(levels: list[_Level]) -> MultigridHierarchy:
    """Engine hierarchy over working levels.

    Args:
        levels (list[_Level]): Working levels, finest first.

    Returns:
        MultigridHierarchy: Hierarchy sharing the levels' tensors.
    """
    return assemble_hierarchy(
        [level.A for level in levels],
        [DenseTransferOperator(_built(level.P)) for level in levels[:-1]],
    )


def _prepared_fine(level: _Level) -> PreparedFineLevel:
    """``level`` bound to the setup cycle's smoother, as ``GS_SETUP_CYCLE.prepare`` binds it.

    Args:
        level (_Level): Working level whose ``A`` and ``P`` are final for the next trial solve.

    Returns:
        PreparedFineLevel: The prepared level.
    """
    return PreparedFineLevel(
        level.A, DenseTransferOperator(_built(level.P)), GS_SETUP_SMOOTHER.prepare(level.A)
    )


def _solve(hierarchy: PreparedHierarchy, x: torch.Tensor, iterations: int) -> torch.Tensor:
    """PyAMG's ``solve(b=0, x0=x, tol=1e-20, maxiter=iterations)`` with V-cycles.

    Uses linearity of the cycle: one cycle from ``x`` on ``A x = 0`` is
    ``x - B (A x)`` with ``B`` the zero-initial-guess cycle.

    Args:
        hierarchy (PreparedHierarchy): Trial hierarchy prepared for ``GS_SETUP_CYCLE``,
            at least two levels.
        x (torch.Tensor): Initial guess.
        iterations (int): Maximum number of cycles.

    Returns:
        torch.Tensor: Iterate after the cycles.
    """
    matrix = hierarchy.levels[0].matrix
    residual = matrix @ x
    for _ in range(iterations):
        x = x - GS_SETUP_CYCLE.apply(hierarchy, residual)
        residual = matrix @ x
        if torch.linalg.norm(residual) < _SOLVE_TOLERANCE:
            break
    return x


def _extend(levels: list[_Level], params: _Params, predefined: _Predefined | None) -> None:
    """Append one coarse level (PyAMG's ``_extend_hierarchy``).

    Args:
        levels (list[_Level]): Working levels; the last one is coarsened.
        params (_Params): Setup parameters.
        predefined (_Predefined | None): Aggregation/strength to reuse.
    """
    level = levels[-1]
    if predefined is None:
        strength = node_strength(level.A, level.dofs, params.theta)
        aggregate = standard_aggregation(strength)
    else:
        strength, aggregate = (
            predefined.strengths[len(levels) - 1],
            predefined.aggregates[len(levels) - 1],
        )
    tentative, coarse_candidates = fit_candidates(aggregate, level.B)
    prolongation = jacobi_prolongation(level.A, tentative, params.omega, params.spectral_radius)
    level.strength, level.aggregate, level.T, level.P = strength, aggregate, tentative, prolongation
    levels.append(
        _Level(
            A=prolongation.T @ level.A @ prolongation,
            B=coarse_candidates,
            dofs=level.B.shape[1],
        )
    )


def _smoothed_aggregation_levels(
    matrix: torch.Tensor,
    candidates: torch.Tensor,
    params: _Params,
    predefined: _Predefined | None,
) -> list[_Level]:
    """Build the SA hierarchy for ``candidates`` (PyAMG's ``smoothed_aggregation_solver``).

    Predefined aggregation fixes the depth (``len + 1`` levels, no coarse-size
    limit), as in PyAMG's ``levelize_strength_or_aggregation``.

    Args:
        matrix (torch.Tensor): Finest matrix.
        candidates (torch.Tensor): Near-null-space candidates ``(n, m)``.
        params (_Params): Setup parameters.
        predefined (_Predefined | None): Aggregation/strength to reuse.

    Returns:
        list[_Level]: Levels, finest first.
    """
    max_levels, max_coarse = (
        (len(predefined.aggregates) + 1, 0)
        if predefined is not None
        else (params.max_levels, params.max_coarse)
    )
    levels = [_Level(A=matrix, B=candidates)]
    while len(levels) < max_levels and levels[-1].A.shape[0] // levels[-1].dofs > max_coarse:
        _extend(levels, params, predefined)
    return levels


def _initial_setup_stage(
    matrix: torch.Tensor, params: _Params
) -> tuple[torch.Tensor, _Predefined | None]:
    """First candidate and aggregation (PyAMG's ``initial_setup_stage``, Algorithm 3).

    Args:
        matrix (torch.Tensor): System matrix.
        params (_Params): Setup parameters.

    Returns:
        tuple[torch.Tensor, _Predefined | None]: The candidate and, when more
            than one level was aggregated, the aggregation to reuse.
    """
    current = matrix
    # Each level is relaxed going down and again coming back up: prepare it once.
    relaxations = [prepare_symmetric_gauss_seidel(matrix)]
    x = _relax(relaxations[0], params.random_vector(matrix, matrix.shape[0]), params)
    iterates, prolongations, matrices = [x], [], [matrix]
    aggregates: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    while matrix.shape[0] > params.max_coarse and params.max_levels > 1:
        strength = node_strength(current, 1, params.theta)
        aggregate = standard_aggregation(strength)
        tentative, coarse = fit_candidates(aggregate, x.unsqueeze(1))
        x = coarse[:, 0]
        prolongation = jacobi_prolongation(current, tentative, params.omega, params.spectral_radius)
        current = prolongation.T @ current @ prolongation
        strengths.append(strength)
        aggregates.append(aggregate)
        prolongations.append(prolongation)
        matrices.append(current)
        if current.shape[0] <= params.max_coarse or len(aggregates) + 1 >= params.max_levels:
            break
        relaxations.append(prepare_symmetric_gauss_seidel(current))
        x = _relax(relaxations[-1], x, params)
        iterates.append(x)
    x = iterates[-1]
    for level in range(len(prolongations) - 2, -1, -1):
        x = _relax(relaxations[level], prolongations[level] @ x, params)
    predefined = _Predefined(tuple(aggregates), tuple(strengths)) if len(aggregates) > 1 else None
    return x, predefined


def _general_setup_stage(levels: list[_Level], params: _Params) -> torch.Tensor:
    """One more candidate from the current hierarchy (PyAMG's ``general_setup_stage``, Algorithm 4).

    Mutates ``levels`` (they are a throw-away hierarchy).

    Args:
        levels (list[_Level]): Hierarchy built from the candidates so far.
        params (_Params): Setup parameters.

    Returns:
        torch.Tensor: The new fine-level candidate (not yet normalized).
    """
    fine = levels[0].A
    x = params.random_vector(fine, fine.shape[0])
    if len(levels) == 1:
        return torch.zeros_like(x)
    # One trial hierarchy per candidate: each sub-hierarchy solve below changes only the
    # level it refits, so the other prepared levels and the coarsest inverse are reused.
    prepared = GS_SETUP_CYCLE.prepare(_hierarchy_of(levels))
    prepared_fine = list(prepared.fine)
    x = _solve(prepared, x, params.candidate_iters)
    for i in range(len(levels) - 2):
        candidates = torch.cat([levels[i].B, x.unsqueeze(1)], dim=1)
        tentative, coarse = fit_candidates(_built(levels[i].aggregate), candidates)
        levels[i].T = tentative
        x = coarse[:, -1]
        prolongation = jacobi_prolongation(
            levels[i].A, tentative, params.omega, params.spectral_radius
        )
        levels[i].P = prolongation
        levels[i + 1].A = prolongation.T @ levels[i].A @ prolongation
        bridge = make_bridge(_built(levels[i + 1].T), levels[i + 1].dofs)
        levels[i + 1].P = jacobi_prolongation(
            levels[i + 1].A, bridge, params.omega, params.spectral_radius
        )
        prepared_fine[i + 1] = _prepared_fine(levels[i + 1])
        x = _solve(
            PreparedHierarchy(tuple(prepared_fine[i + 1 :]), prepared.coarsest),
            x,
            params.candidate_iters,
        )
        levels[i + 1].B = coarse[:, :-1].clone()
        levels[i + 1].T = bridge
        levels[i + 1].dofs = candidates.shape[1]
    for level in reversed(levels[:-2]):
        x = _built(level.P) @ x
        rows = torch.nonzero(x).flatten()
        relaxation = prepare_symmetric_gauss_seidel(level.A, rows=rows)
        x = _relax(relaxation, x, params)
    return x


def adaptive_sa_hierarchy(
    matrix: torch.Tensor,
    *,
    initial_candidates: torch.Tensor | None = None,
    num_candidates: int = 1,
    candidate_iters: int = 5,
    max_levels: int = 10,
    max_coarse: int = 10,
    theta: float = 0.0,
    omega: float = 4.0 / 3.0,
    draw: Callable[[int], torch.Tensor] | None = None,
    seed: int = 0,
) -> AdaptiveSAResult:
    """Adaptive smoothed aggregation setup (PyAMG's ``adaptive_sa_solver``).

    Defaults are PyAMG's: ``theta=0`` (every non-zero coupling is strong),
    ``omega=4/3`` nominal prolongator damping (divided by ``rho(D^-1 A)``),
    ``candidate_iters=5``, ``max_levels=max_coarse=10``, ``num_candidates=1``.

    Cost:
        Once per matrix, dense levels ``n_0 = n > ... > n_{N-1} = c`` (``N`` levels;
        ``c`` grows with the candidate count, coarse nodes times candidates so far),
        ``q`` = ``num_candidates``, ``s`` = ``candidate_iters``, ``V = 8`` bytes per
        ``float64``. Building blocks, per level ``l``: a dense symmetric Gauss-Seidel
        iteration ``W = 6n_l^2 + 2n_l``, ``D = 2n_l + O(log n_l)``, ``L = 6`` (see
        ``PreparedSymmetricGaussSeidel``); a smoothed-aggregation level
        ``W_SA,l = 4n_l^2 n_{l+1} + 2n_l n_{l+1}^2 + 3n_l^2 + W_rho,l`` (node strength,
        host aggregation of depth ``3n_l``, ``fit_candidates``, the prolongator GEMM
        ``D^-1 A T``, the Galerkin product, and an Arnoldi estimate ``W_rho,l`` of
        ``rho(D^-1 A_l)``, ``P`` passes of ``2n_l^2 m + 4n_l m(m + 1)`` flops, never
        memoized here); a GS-smoothed V(1,1) trial cycle
        ``W_cyc = sum_l (12n_l^2 + 4n_l n_{l+1} + 5n_l) + 2c^2``,
        ``D_cyc = sum_l 4n_l + O(N log n)``, ``L_cyc = 14N`` approximately.
        The initial stage relaxes ``s`` iterations per level going down and again
        coming up, with one SA level per step. Each of the ``q - 1`` general stages
        rebuilds the SA hierarchy on the fixed aggregation, prepares it (one GS
        preparation per level, ``S = 2`` each, and the coarse inverse, ``~ c^3`` on the
        SPD path or ``~ 24c^3`` through ``pinv``), runs ``s`` trial cycles on the whole
        hierarchy and ``s`` on each of the ``N - 2`` nested sub-hierarchies (each refit
        costs two more Arnoldi estimates), then relaxes up with an indexed GS per
        level. A final SA rebuild closes the setup.
        In total ``W = 12s sum_l n_l^2 + (q + 1) sum_l W_SA,l
        + (q - 1)[s N W_cyc + 24c^3 + 2N W_rho]`` approximately (quadratic in ``N``
        through the nested sub-hierarchy solves), ``Q`` about ``V`` times the
        dense matrix reads of those blocks (``n_l^2`` per GS triangle product and
        solve, per Galerkin GEMM, per Arnoldi matvec),
        ``D = 4s sum_l n_l + (q - 1) s N D_cyc + (q + 1) sum_l (3n_l + D_rho,l)
        + (q - 1) O(c)`` approximately (the substitution recurrences, the host
        aggregations and the coarse factorizations are all sequential),
        ``L = (q - 1) s N L_cyc + O((q + 1) sum_l L_rho,l)`` and
        ``S = O(q s N)`` (one residual-norm read per trial cycle, the GS
        frozen-row tests, the aggregation, ``fit_candidates`` and Arnoldi reads, one
        ``nonzero`` per indexed GS). The ``n_l^2``-sized dense GS triangles make every
        trial cycle cost ``O(n^2)`` whatever the sparsity of ``A``.
        Work-depth: depth, not flops, limits the setup: every GS iteration is a
        ``2n_l``-step recurrence.

    Args:
        matrix (torch.Tensor): SPD system matrix, shape ``(n, n)``.
        initial_candidates (torch.Tensor | None): Known candidates ``(n, k)``;
            when given, the initial setup stage is skipped.
        num_candidates (int): Total number of candidates (including any
            ``initial_candidates``).
        candidate_iters (int): Relaxation sweeps / cycles per candidate step.
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        theta (float): Strength-of-connection threshold.
        omega (float): Nominal Jacobi prolongator-smoothing damping.
        draw (Callable[[int], torch.Tensor] | None): Uniform ``[0, 1)`` source
            (``n -> tensor of length n``) used for every random vector,
            including the spectral-radius starts; seeded torch generator if None.
        seed (int): Seed of the default source.

    Returns:
        AdaptiveSAResult: Level matrices, prolongations and candidates.

    Raises:
        ValueError: If a computed candidate is identically zero.
    """
    params = _Params(
        theta, omega, candidate_iters, max_levels, max_coarse, draw or seeded_draw(seed)
    )
    predefined: _Predefined | None = None
    if initial_candidates is None:
        x, predefined = _initial_setup_stage(matrix, params)
        candidates = (x / x.abs().max()).unsqueeze(1)
        num_candidates -= 1
    else:
        candidates = initial_candidates.to(device=matrix.device, dtype=matrix.dtype)
        num_candidates -= candidates.shape[1]
        levels = _smoothed_aggregation_levels(matrix, candidates, params, None)
        if len(levels) > 1:
            predefined = _Predefined(
                tuple(_built(level.aggregate) for level in levels[:-1]),
                tuple(_built(level.strength) for level in levels[:-1]),
            )
    for index in range(num_candidates):
        levels = _smoothed_aggregation_levels(matrix, candidates, params, predefined)
        x = _general_setup_stage(levels, params)
        x = x / x.abs().max()
        if not torch.isfinite(x[0]):
            raise ValueError(f"The {index}th adaptive candidate is all 0.")
        candidates = torch.cat([candidates, x.unsqueeze(1)], dim=1)
    levels = _smoothed_aggregation_levels(matrix, candidates, params, predefined)
    return AdaptiveSAResult(
        matrices=tuple(level.A for level in levels),
        prolongations=tuple(_built(level.P) for level in levels[:-1]),
        candidates=candidates,
        aggregates=tuple(_built(level.aggregate) for level in levels[:-1]),
        strengths=tuple(_built(level.strength) for level in levels[:-1]),
    )


class AdaptiveSAPreconditioner(AMGPreconditioner):
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA), PyAMG-faithful.

    Runs ``adaptive_sa_hierarchy`` in ``setup()``, then applies one
    V(1,1)-cycle with weighted-Jacobi smoothing by default and a pseudo-inverse
    coarse solve - a fixed symmetric linear operator, so plain PCG is valid.
    Adaptive candidate construction remains symmetric Gauss-Seidel to preserve
    the alpha-SA setup algorithm; ``smoother`` affects only solve-time cycles.
    Setup costs about ``num_candidates`` hierarchy builds, so it pays off when
    one matrix is reused across many solves (call ``setup()`` once, then
    ``apply()`` many times).

    Cost:
        Setup: ``adaptive_sa_hierarchy`` (see its cost: ``W``, ``D``, ``L`` and ``S``
        there) plus the preparation of the solve-time cycle, one Jacobi ``prepare``
        per non-coarsest level (an Arnoldi estimate ``W_rho,l`` each under the default
        spectral damping) and the coarse inverse (``~ c^3`` SPD, ``~ 24c^3`` ``pinv``):
        ``W = W_adaptive + sum_l W_rho,l + 24c^3`` at most,
        ``D = D_adaptive + sum_l D_rho,l + O(c)``,
        ``L = L_adaptive + sum_l L_rho,l + O(N)`` and
        ``S = S_adaptive + sum_l P_l + 1``. It dominates the method's total cost.
        Apply, one dense V(``n_pre``, ``n_post``) cycle with weighted Jacobi,
        ``nu = n_pre + n_post``, the pseudo-inverse coarse product:
        ``W = sum_l [2 nu n_l^2 + 4n_l n_{l+1} + (3 nu - 1) n_l] + 2c^2``,
        ``Q = V(sum_l [nu n_l^2 + 2n_l n_{l+1} + O(nu n_l)] + c^2 + 2c)``,
        ``D = sum_l (nu + 3)(ceil(log2 n_l) + 4) + ceil(log2 c) + 1`` approximately,
        ``L = sum_l (4 nu + 2) + 1`` and ``S = 0``.
        Roofline: apply is memory-bound (dense matvecs, about 0.25 flop per byte).
        Work-depth: apply has no recurrence (Jacobi, transfers, one coarse product),
        so ``W/D`` is large.

    Args:
        matrix (torch.Tensor): SPD system matrix A (n x n).
        num_candidates (int): Total number of near-null-space candidates.
        candidate_iters (int): Relaxation sweeps / cycles per candidate step.
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        theta (float): Strength threshold (PyAMG's default 0 accepts every coupling).
        omega (float): Nominal prolongator-smoothing damping.
        initial_candidates (torch.Tensor | None): Known near-null vectors ``(n, k)``.
        seed (int): Seed of the random start vectors.
        draw (Callable[[int], torch.Tensor] | None): Explicit uniform ``[0, 1)``
            source overriding ``seed``.
        smoother_omega (float | None): Damping for the default weighted-Jacobi
            solve smoother; ``None`` selects ``1 / rho(D^-1 A)`` per level.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

    Note (DIP):
        Preset/factory leaf class; unlike ``vcycle_amg`` and
        ``pod2g_preconditioner`` (plain factory functions), this one
        overrides ``setup()`` to run its own setup algorithm and then
        ``_commit()`` the realized hierarchy, with genuinely different
        hierarchy-construction logic, so it remains a subclass of
        ``AMGPreconditioner``.

    References:
        - Brezina et al. (2005), SIAM Review 47(2); PyAMG 5.3.0.
    """

    def __init__(
        self,
        num_candidates: int = 1,
        candidate_iters: int = 5,
        max_levels: int = 10,
        max_coarse: int = 10,
        theta: float = 0.0,
        omega: float = 4.0 / 3.0,
        initial_candidates: torch.Tensor | None = None,
        seed: int = 0,
        draw: Callable[[int], torch.Tensor] | None = None,
        smoother_omega: float | None = None,
        smoother: MultigridSmoother | None = None,
        n_pre: int = 1,
        n_post: int = 1,
    ) -> None:
        """Store setup hyperparameters; ``setup()`` runs the adaptive algorithm.

        Args:
            num_candidates (int): Total number of candidates.
            candidate_iters (int): Relaxation sweeps / cycles per candidate step.
            max_levels (int): Maximum number of levels.
            max_coarse (int): Stop coarsening at this many coarse nodes.
            theta (float): Strength threshold.
            omega (float): Nominal prolongator-smoothing damping.
            initial_candidates (torch.Tensor | None): Known near-null vectors.
            seed (int): Seed of the random start vectors.
            draw (Callable[[int], torch.Tensor] | None): Explicit uniform
                ``[0, 1)`` source overriding ``seed``.
            smoother_omega (float | None): Damping for the default
                weighted-Jacobi solve smoother; ``None`` selects the
                per-level spectral rule.
            smoother (MultigridSmoother | None): Explicit solve-time
                smoother, or ``None`` for weighted Jacobi.
            n_pre (int): Solve-time pre-smoothing sweeps.
            n_post (int): Solve-time post-smoothing sweeps.
        """
        self._num_candidates = num_candidates
        self._candidate_iters = candidate_iters
        self._max_levels = max_levels
        self._max_coarse = max_coarse
        self._theta = theta
        self._omega = omega
        self._initial_candidates = initial_candidates
        self._seed = seed
        self._draw = draw
        self._result: AdaptiveSAResult | None = None
        super().__init__(
            coarsening=PrebuiltCoarsening("adaptive SA"),
            cycle=prebuilt_cycle(
                resolve_jacobi_default(smoother, smoother_omega), n_pre=n_pre, n_post=n_post
            ),
            n_levels=2,
            linear=True,
        )

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Run the adaptive setup algorithm and build the resulting hierarchy.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n).
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, now ready for ``apply()``.

        Raises:
            ValueError: If the setup produced a single level (nothing to coarsen).
        """
        result = adaptive_sa_hierarchy(
            matrix,
            initial_candidates=self._initial_candidates,
            num_candidates=self._num_candidates,
            candidate_iters=self._candidate_iters,
            max_levels=self._max_levels,
            max_coarse=self._max_coarse,
            theta=self._theta,
            omega=self._omega,
            seed=self._seed,
            draw=self._draw,
        )
        if len(result.matrices) < 2:
            raise ValueError("adaptive setup produced a single level; lower max_coarse")
        self._commit(self._hierarchy_of(result, matrix))
        self._result = result
        self._n_levels = len(result.matrices)
        return self

    @property
    def result(self) -> AdaptiveSAResult:
        """The realized adaptive-setup hierarchy.

        Returns:
            AdaptiveSAResult: Levels, prolongations, and near-null-space
                candidates from the setup that last ran.
        """
        if self._result is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before the result is available."
            )
        return self._result

    def __str__(self) -> str:
        """Human-readable structural summary.

        Overrides ``AMGPreconditioner.__str__`` — this class stores a
        ``PrebuiltCoarsening`` placeholder as its coarsening strategy (the
        real per-level detail lives in ``self._result``, not something a
        generic coarsening-strategy `str()` could describe), so the base
        class's generic wrapping doesn't apply here.

        Returns:
            str: e.g. ``"alpha-SA(n_levels=3, num_candidates=3,
                coarse_dim=5)"``.
        """
        if self._result is None:
            return "alpha-SA(not set up)"
        return (
            f"alpha-SA(n_levels={len(self._result.matrices)}, "
            f"num_candidates={self._result.candidates.shape[1]}, "
            f"coarse_dim={int(self._result.matrices[-1].shape[0])})"
        )

    @staticmethod
    def _hierarchy_of(result: AdaptiveSAResult, matrix: torch.Tensor) -> MultigridHierarchy:
        """Prebuilt levels of ``result`` moved to the device/dtype of ``matrix``.

        Args:
            result (AdaptiveSAResult): Realized setup hierarchy.
            matrix (torch.Tensor): System matrix being bound.

        Returns:
            MultigridHierarchy: The adaptive-SA hierarchy.
        """
        moved = AdaptiveSAResult(
            matrices=tuple(m.to(matrix) for m in result.matrices),
            prolongations=tuple(p.to(matrix) for p in result.prolongations),
            candidates=result.candidates,
            aggregates=result.aggregates,
            strengths=result.strengths,
        )
        return moved.hierarchy
