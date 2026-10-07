"""Adaptive smoothed aggregation (alpha-SA), sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.adaptive`` (dense; kept unmodified
for comparison - see ``docs/plan.md``'s "Correction: dense and sparse must
be separate implementations, not an internal branch"). Follows the dense
module's port of PyAMG's ``adaptive_sa_solver`` step by step
(``_initial_setup_stage`` = Algorithm 3, ``_general_setup_stage`` =
Algorithm 4 of Brezina et al. 2005) with every format-committing piece
swapped for its sparse sibling:

- ``node_strength``/``standard_aggregation`` -> ``sparse_node_strength``
  (``._node_strength``) / ``sparse_standard_aggregation`` (``.aggregation``).
- ``jacobi_prolongation`` -> ``sparse_smoothed_prolongation``
  (``.aggregation``) composed with ``scaled_by_inverse_diagonal``/
  ``jacobi_spectral_radius``-style damping resolved via the shared
  ``torchalg.utils.spectral.approximate_spectral_radius`` (``_jacobi_
  prolongation``, this module), exactly mirroring the dense ``jacobi_
  prolongation``'s ``P = T - (omega / rho) D^-1 A T`` formula.
- ``make_bridge`` -> ``sparse_make_bridge`` (``._prolongation``).
- ``symmetric_gauss_seidel`` (incl. its indexed ``rows=`` variant) ->
  ``sparse_symmetric_gauss_seidel`` (``._gauss_seidel``).
- dense ``P.T @ A @ P`` Galerkin products -> ``form_sparse_sparse``
  (``torchalg.sparse.kernels.galerkin``).
- ``DenseTransferOperator`` -> ``SparseTransferOperator`` (``.transfer``).
- ``pseudo_inverse_solve`` -> ``dense_pseudo_inverse_solve``
  (``.coarse_solve``) - the one deliberate densification, at whatever
  level's coarsest matrix a trial setup cycle bottoms out at, mirroring the
  same precedent as every other sparse coarsest-level solve in this tree.

``fit_candidates`` (``torchalg.utils.tentative``) is reused directly,
unmodified: it operates on a node-aggregate index tensor and dense
candidate vectors, never on ``A`` or ``P`` themselves, so it is genuinely
format-agnostic (already promoted, per the scoping investigation this
module follows). Its dense ``Q`` output (the tentative prolongator) is
converted to sparse CSR immediately after the call, before any further use,
so prolongation/Galerkin products downstream never see a dense operator.

This module's own candidate-construction algorithm has no greedy/
discrete-selection tie-break analogous to BAMG's ``select_interpolatory_
set`` (see ``bootstrap.py``'s "Known numerical sensitivity" note): alpha-SA
builds its prolongator by orthonormalizing every aggregate's candidate
block (``fit_candidates``, deterministic Gram-Schmidt, no "closest
candidate wins" comparison) and by smoothing it with a fixed Jacobi step -
no step picks among several near-tied discrete choices. Dense-vs-sparse
summation-order differences in the spectral-radius Arnoldi estimate and in
sparse-sparse Galerkin products are still present (the usual float
summation-order sensitivity, not a discrete-choice one), so parity tests
use loose-enough tolerances for those and keep problem sizes modest so the
coarsening loop's level count is identical between trees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchalg.multigrid import AMGPreconditioner, MultigridHierarchy, MultigridLevel
from torchalg.preconditioners.base import PreconditionerContext
from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.utils.spectral import approximate_spectral_radius
from torchalg.utils.tentative import fit_candidates

from ._gauss_seidel import sparse_symmetric_gauss_seidel
from ._jacobi_omega import scaled_by_inverse_diagonal
from ._node_strength import sparse_node_strength
from ._presets import GS_SETUP_CYCLE, PrebuiltCoarsening, prebuilt_cycle, seeded_draw
from ._prolongation import sparse_make_bridge
from .aggregation import sparse_smoothed_prolongation, sparse_standard_aggregation
from .coarse_solve import dense_pseudo_inverse_solve
from .smoothers import resolve_jacobi_default
from .transfer import SparseTransferOperator

if TYPE_CHECKING:
    from typing import Self

    from torchalg.multigrid.protocols import MultigridSmoother

_SOLVE_TOLERANCE = 1e-20
"""Absolute residual tolerance of the trial cycles, matches the dense sibling."""


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
            matrix (torch.Tensor): Square matrix, sparse CSR or dense.

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
    """Result of the sparse adaptive setup.

    Attributes:
        matrices (tuple[torch.Tensor, ...]): Level matrices ``A_l``, sparse
            CSR, finest first.
        prolongations (tuple[torch.Tensor, ...]): Sparse CSR ``P_l`` mapping
            level ``l+1`` to level ``l`` (``R = P^T``).
        candidates (torch.Tensor): Dense fine-level near-null-space
            candidates, shape ``(n, num_candidates)``.
        aggregates (tuple[torch.Tensor, ...]): Node aggregation of each
            coarsened level (``-1`` = isolated node).
        strengths (tuple[torch.Tensor, ...]): Sparse CSR boolean node
            strength graph of each coarsened level.
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
            MultigridHierarchy: Levels with sparse transfer operators.
        """
        last = len(self.matrices) - 1
        return MultigridHierarchy(
            tuple(
                MultigridLevel(
                    matrix=matrix,
                    transfer=None
                    if index == last
                    else SparseTransferOperator(self.prolongations[index]),
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


def _jacobi_prolongation(
    matrix: torch.Tensor, tentative: torch.Tensor, omega_nominal: float, params: _Params
) -> torch.Tensor:
    """Sparse CSR smoothed prolongator ``P = T - (omega / rho(D^-1 A)) D^-1 A T``.

    Sparse sibling of the dense ``_prolongation.jacobi_prolongation``: same
    formula, resolved via ``scaled_by_inverse_diagonal`` (sparse ``D^-1 A``)
    and the shared ``approximate_spectral_radius`` before delegating the
    actual Jacobi step to ``sparse_smoothed_prolongation``.

    Args:
        matrix (torch.Tensor): Sparse CSR fine-grid matrix ``A``.
        tentative (torch.Tensor): Sparse CSR tentative prolongator ``T``.
        omega_nominal (float): Nominal damping (PyAMG's default is 4/3).
        params (_Params): Setup parameters (provides the spectral-radius draw).

    Returns:
        torch.Tensor: Sparse CSR smoothed prolongator, same shape as ``tentative``.
    """
    scaled = scaled_by_inverse_diagonal(matrix)
    rho = params.spectral_radius(scaled)
    return sparse_smoothed_prolongation(matrix, tentative, omega_nominal / rho)


def _relax(matrix: torch.Tensor, x: torch.Tensor, params: _Params) -> torch.Tensor:
    """``candidate_iters`` symmetric Gauss-Seidel iterations on ``A x = 0``, sparse CSR.

    Args:
        matrix (torch.Tensor): Sparse CSR level matrix.
        x (torch.Tensor): Vector to relax.
        params (_Params): Setup parameters.

    Returns:
        torch.Tensor: Relaxed vector.
    """
    return sparse_symmetric_gauss_seidel(matrix, x, torch.zeros_like(x), params.candidate_iters)


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
                transfer=None if index == last else SparseTransferOperator(_built(level.P)),
            )
            for index, level in enumerate(levels)
        )
    )


def _solve(levels: list[_Level], x: torch.Tensor, iterations: int) -> torch.Tensor:
    """PyAMG's ``solve(b=0, x0=x, tol=1e-20, maxiter=iterations)`` with V-cycles, sparse CSR.

    Args:
        levels (list[_Level]): Working levels, finest first.
        x (torch.Tensor): Initial guess.
        iterations (int): Maximum number of cycles.

    Returns:
        torch.Tensor: Iterate after the cycles.
    """
    if len(levels) == 1:
        return dense_pseudo_inverse_solve(levels[0].A, torch.zeros_like(x))
    hierarchy, matrix = _hierarchy_of(levels), levels[0].A
    for _ in range(iterations):
        x = x - GS_SETUP_CYCLE.apply(hierarchy, matrix @ x)
        if torch.linalg.norm(matrix @ x) < _SOLVE_TOLERANCE:
            break
    return x


def _extend(levels: list[_Level], params: _Params, predefined: _Predefined | None) -> None:
    """Append one coarse level (PyAMG's ``_extend_hierarchy``), sparse CSR.

    Args:
        levels (list[_Level]): Working levels; the last one is coarsened.
        params (_Params): Setup parameters.
        predefined (_Predefined | None): Aggregation/strength to reuse.
    """
    level = levels[-1]
    if predefined is None:
        strength = sparse_node_strength(level.A, level.dofs, params.theta)
        aggregate = sparse_standard_aggregation(strength)
    else:
        strength, aggregate = (
            predefined.strengths[len(levels) - 1],
            predefined.aggregates[len(levels) - 1],
        )
    tentative_dense, coarse_candidates = fit_candidates(aggregate, level.B)
    tentative = tentative_dense.to_sparse_csr()
    prolongation = _jacobi_prolongation(level.A, tentative, params.omega, params)
    level.strength, level.aggregate, level.T, level.P = strength, aggregate, tentative, prolongation
    levels.append(
        _Level(
            A=form_sparse_sparse(prolongation, level.A),
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
    """Build the sparse SA hierarchy for ``candidates`` (PyAMG's ``smoothed_aggregation_solver``).

    Args:
        matrix (torch.Tensor): Sparse CSR finest matrix.
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
    """First candidate and aggregation (PyAMG's ``initial_setup_stage``, Algorithm 3), sparse CSR.

    Args:
        matrix (torch.Tensor): Sparse CSR system matrix.
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
        strength = sparse_node_strength(current, 1, params.theta)
        aggregate = sparse_standard_aggregation(strength)
        tentative_dense, coarse = fit_candidates(aggregate, x.unsqueeze(1))
        x = coarse[:, 0]
        tentative = tentative_dense.to_sparse_csr()
        prolongation = _jacobi_prolongation(current, tentative, params.omega, params)
        current = form_sparse_sparse(prolongation, current)
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
        tentative_dense, coarse = fit_candidates(_built(levels[i].aggregate), candidates)
        tentative = tentative_dense.to_sparse_csr()
        levels[i].T = tentative
        x = coarse[:, -1]
        prolongation = _jacobi_prolongation(levels[i].A, tentative, params.omega, params)
        levels[i].P = prolongation
        levels[i + 1].A = form_sparse_sparse(prolongation, levels[i].A)
        bridge = sparse_make_bridge(_built(levels[i + 1].T), levels[i + 1].dofs)
        levels[i + 1].P = _jacobi_prolongation(levels[i + 1].A, bridge, params.omega, params)
        x = _solve(levels[i + 1 :], x, params.candidate_iters)
        levels[i + 1].B = coarse[:, :-1].clone()
        levels[i + 1].T = bridge
        levels[i + 1].dofs = candidates.shape[1]
    for level in reversed(levels[:-2]):
        x = _built(level.P) @ x
        rows = torch.nonzero(x).flatten()
        x = sparse_symmetric_gauss_seidel(
            level.A, x, torch.zeros_like(x), params.candidate_iters, rows=rows
        )
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
    """Adaptive smoothed aggregation setup (PyAMG's ``adaptive_sa_solver``), sparse CSR.

    Sparse-CSR sibling of the dense ``adaptive.adaptive_sa_hierarchy`` -
    same defaults, same algorithm, every format-committing step swapped per
    this module's docstring.

    Args:
        matrix (torch.Tensor): SPD system matrix, sparse CSR, shape ``(n, n)``.
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


class AdaptiveSAPreconditioner(AMGPreconditioner):
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA), sparse-CSR sibling.

    Sparse-CSR counterpart of the dense
    ``preconditioners.implementations.amg.adaptive.AdaptiveSAPreconditioner``
    - same preset shape (runs ``adaptive_sa_hierarchy`` in ``setup()``,
    wraps the resulting fixed hierarchy, overrides ``_make_hierarchy``),
    same defaults, composed entirely from this package's own sparse-native
    kernels/preconditioners plus the shared, format-agnostic
    ``torchalg.multigrid`` engine.

    Args:
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
            ``None`` selects sparse weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

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
        """Store setup hyperparameters; ``setup()`` runs the sparse adaptive algorithm.

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
                smoother, or ``None`` for sparse weighted Jacobi.
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
        super().__init__(
            matrix=torch.empty(0),
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
        """Run the sparse adaptive setup algorithm and build the resulting hierarchy.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n), sparse CSR.
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
        self._result = result
        self._n_levels = len(result.matrices)
        return super().setup(matrix, context)

    @property
    def result(self) -> AdaptiveSAResult:
        """The realized adaptive-setup hierarchy.

        Returns:
            AdaptiveSAResult: Levels, prolongations, and near-null-space
                candidates from the setup that ran at construction.
        """
        return self._result

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"alpha-SA(n_levels=3, num_candidates=3,
                coarse_dim=5)"``.
        """
        return (
            f"alpha-SA(n_levels={len(self._result.matrices)}, "
            f"num_candidates={self._result.candidates.shape[1]}, "
            f"coarse_dim={int(self._result.matrices[-1].shape[0])})"
        )

    def _make_hierarchy(self) -> MultigridHierarchy:
        """Prebuilt levels moved to the current device/dtype of the system matrix.

        Returns:
            MultigridHierarchy: The sparse adaptive-SA hierarchy.
        """
        moved = AdaptiveSAResult(
            matrices=tuple(m.to(self._matrix) for m in self._result.matrices),
            prolongations=tuple(p.to(self._matrix) for p in self._result.prolongations),
            candidates=self._result.candidates,
            aggregates=self._result.aggregates,
            strengths=self._result.strengths,
        )
        return moved.hierarchy
