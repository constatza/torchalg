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
disables), non-default smoothers/strength/aggregation/coarse solvers, complex
and nonsymmetric matrices, and the ``work`` complexity counter.

References:
    - Brezina, M., Falgout, R., MacLachlan, S., Manteuffel, T., McCormick,
      S., & Ruge, J. (2005). Adaptive smoothed aggregation (alpha-SA)
      multigrid. SIAM Review, 47(2), 317-346.
    - PyAMG 5.3.0, ``pyamg/aggregation/adaptive.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._aggregation import standard_aggregation
from ._node_strength import node_strength
from ._prolongation import jacobi_prolongation, make_bridge
from ._relaxation import symmetric_gauss_seidel
from ._spectral import approximate_spectral_radius
from ._tentative import fit_candidates
from .amg import AMGPreconditioner
from .cycle import VCycle, pseudo_inverse_solve
from .hierarchy import MultigridHierarchy, MultigridLevel
from .smoothers import GaussSeidelSmoother
from .transfer import DenseTransferOperator

_SOLVE_TOLERANCE = 1e-20
"""Absolute residual tolerance of the trial cycles (``ml.solve(..., tol=1e-20)``)."""

_CYCLE = VCycle(GaussSeidelSmoother(), n_pre=1, n_post=1, coarse_solver=pseudo_inverse_solve)
"""PyAMG's default cycle: V(1,1), symmetric Gauss-Seidel, ``pinv`` coarse solve."""


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

    def spectral_radius(self, matrix: torch.Tensor) -> float:
        """Arnoldi estimate of ``rho`` drawing from the shared random stream.

        Args:
            matrix (torch.Tensor): Square matrix.

        Returns:
            float: Estimated spectral radius.
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
        last = len(self.matrices) - 1
        return MultigridHierarchy(
            tuple(
                MultigridLevel(
                    matrix=matrix,
                    transfer=None
                    if index == last
                    else DenseTransferOperator(self.prolongations[index]),
                )
                for index, matrix in enumerate(self.matrices)
            )
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


def _relax(matrix: torch.Tensor, x: torch.Tensor, params: _Params) -> torch.Tensor:
    """``candidate_iters`` symmetric Gauss-Seidel iterations on ``A x = 0``.

    Args:
        matrix (torch.Tensor): Level matrix.
        x (torch.Tensor): Vector to relax.
        params (_Params): Setup parameters.

    Returns:
        torch.Tensor: Relaxed vector.
    """
    return symmetric_gauss_seidel(matrix, x, torch.zeros_like(x), params.candidate_iters)


def _hierarchy_of(levels: list[_Level]) -> MultigridHierarchy:
    """Engine hierarchy over working levels.

    Args:
        levels (list[_Level]): Working levels, finest first.

    Returns:
        MultigridHierarchy: Hierarchy sharing the levels' tensors.
    """
    last = len(levels) - 1
    return MultigridHierarchy(
        tuple(
            MultigridLevel(
                matrix=level.A,
                transfer=None if index == last else DenseTransferOperator(_built(level.P)),
            )
            for index, level in enumerate(levels)
        )
    )


def _solve(levels: list[_Level], x: torch.Tensor, iterations: int) -> torch.Tensor:
    """PyAMG's ``solve(b=0, x0=x, tol=1e-20, maxiter=iterations)`` with V-cycles.

    Uses linearity of the cycle: one cycle from ``x`` on ``A x = 0`` is
    ``x - B (A x)`` with ``B`` the zero-initial-guess cycle.

    Args:
        levels (list[_Level]): Working levels, finest first.
        x (torch.Tensor): Initial guess.
        iterations (int): Maximum number of cycles.

    Returns:
        torch.Tensor: Iterate after the cycles.
    """
    if len(levels) == 1:
        return pseudo_inverse_solve(levels[0].A, torch.zeros_like(x))
    hierarchy, matrix = _hierarchy_of(levels), levels[0].A
    for _ in range(iterations):
        x = x - _CYCLE.apply(hierarchy, matrix @ x)
        if torch.linalg.norm(matrix @ x) < _SOLVE_TOLERANCE:
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
    x = _relax(current, params.random_vector(matrix, matrix.shape[0]), params)
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
        x = _relax(current, x, params)
        iterates.append(x)
    x = iterates[-1]
    for level in range(len(prolongations) - 2, -1, -1):
        x = _relax(matrices[level], prolongations[level] @ x, params)
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
    x = _solve(levels, params.random_vector(fine, fine.shape[0]), params.candidate_iters)
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
        x = _solve(levels[i + 1 :], x, params.candidate_iters)
        levels[i + 1].B = coarse[:, :-1].clone()
        levels[i + 1].T = bridge
        levels[i + 1].dofs = candidates.shape[1]
    for level in reversed(levels[:-2]):
        x = _built(level.P) @ x
        rows = torch.nonzero(x).flatten()
        x = symmetric_gauss_seidel(
            level.A, x, torch.zeros_like(x), params.candidate_iters, rows=rows
        )
    return x


def _seeded_draw(seed: int) -> Callable[[int], torch.Tensor]:
    """Stateful uniform ``[0, 1)`` source seeded for reproducibility.

    Args:
        seed (int): Generator seed.

    Returns:
        Callable[[int], torch.Tensor]: ``n -> float64`` tensor of length ``n``.
    """
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


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
        theta, omega, candidate_iters, max_levels, max_coarse, draw or _seeded_draw(seed)
    )
    predefined: _Predefined | None = None
    if initial_candidates is None:
        x, predefined = _initial_setup_stage(matrix, params)
        candidates = (x / x.abs().max()).unsqueeze(1)
        num_candidates -= 1
    else:
        candidates = initial_candidates.to(matrix.dtype)
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


class _PrebuiltCoarsening:
    """Placeholder ``CoarseningStrategy``: the levels are prebuilt, never built by the engine.

    ``AdaptiveSAPreconditioner`` overrides ``_make_hierarchy``, so the engine
    never asks this strategy for a level.
    """

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Always raises: the hierarchy is prebuilt by ``adaptive_sa_hierarchy``.

        Args:
            A (torch.Tensor): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("adaptive SA levels are prebuilt; the engine must not rebuild them")


class AdaptiveSAPreconditioner(AMGPreconditioner):
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA), PyAMG-faithful.

    Runs ``adaptive_sa_hierarchy`` at construction, then applies one
    V(1,1)-cycle with symmetric Gauss-Seidel smoothing and a pseudo-inverse
    coarse solve - a fixed symmetric linear operator, so plain PCG is valid.
    Setup costs about ``num_candidates`` hierarchy builds, so it pays off when
    one matrix is reused across many solves.

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

    Note (DIP):
        Preset/factory leaf class, same pattern as ``VCycleAMG`` and
        ``POD2GPreconditioner``.

    References:
        - Brezina et al. (2005), SIAM Review 47(2); PyAMG 5.3.0.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        num_candidates: int = 1,
        candidate_iters: int = 5,
        max_levels: int = 10,
        max_coarse: int = 10,
        theta: float = 0.0,
        omega: float = 4.0 / 3.0,
        initial_candidates: torch.Tensor | None = None,
        seed: int = 0,
        draw: Callable[[int], torch.Tensor] | None = None,
    ) -> None:
        """Run the adaptive setup and wrap the resulting hierarchy.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n).
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

        Raises:
            ValueError: If the setup produced a single level (nothing to coarsen).
        """
        result = adaptive_sa_hierarchy(
            matrix,
            initial_candidates=initial_candidates,
            num_candidates=num_candidates,
            candidate_iters=candidate_iters,
            max_levels=max_levels,
            max_coarse=max_coarse,
            theta=theta,
            omega=omega,
            seed=seed,
            draw=draw,
        )
        if len(result.matrices) < 2:
            raise ValueError("adaptive setup produced a single level; lower max_coarse")
        super().__init__(
            matrix=matrix,
            coarsening=_PrebuiltCoarsening(),
            cycle=_CYCLE,
            n_levels=len(result.matrices),
            linear=True,
        )
        self._result = result

    def _make_hierarchy(self) -> MultigridHierarchy:
        """Prebuilt levels moved to the current device/dtype of the system matrix.

        Returns:
            MultigridHierarchy: The adaptive-SA hierarchy.
        """
        moved = AdaptiveSAResult(
            matrices=tuple(m.to(self._matrix) for m in self._result.matrices),
            prolongations=tuple(p.to(self._matrix) for p in self._result.prolongations),
            candidates=self._result.candidates,
            aggregates=self._result.aggregates,
            strengths=self._result.strengths,
        )
        return moved.hierarchy
