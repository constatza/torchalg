"""Adaptive smoothed aggregation (alpha-SA), sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.adaptive`` (dense; kept unmodified
for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). Follows the dense
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
  ``PreparedSymmetricGaussSeidel`` (``._gauss_seidel``), prepared once per
  level matrix through a stage-scoped ``GaussSeidelMemo`` (``.smoothers``);
  the trial cycles' sub-hierarchy operators likewise come from a
  stage-scoped ``TransferOperatorMemo`` (``.transfer``).
- dense ``P.T @ A @ P`` Galerkin products -> ``form_sparse_sparse``
  (``torchalg.sparse.kernels.galerkin``).
- ``DenseTransferOperator`` -> ``SparseTransferOperator`` (``.transfer``).
- ``pseudo_inverse_coarse_solver`` -> ``dense_pseudo_inverse_coarse_solver``
  (``.coarse_solve``) - the one deliberate densification, at whatever
  level's coarsest matrix a trial setup cycle bottoms out at, mirroring the
  same precedent as every other sparse coarsest-level solve in this tree.

``fit_candidates_csr`` (``torchalg.utils.tentative``) shares the dense
tree's per-aggregate Gram-Schmidt and builds the tentative prolongator ``Q``
directly as sparse CSR, identical to converting the dense ``Q``, so neither
a dense ``(n, n_coarse)`` intermediate nor any downstream product ever sees
a dense operator.

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

from torchalg.multigrid import (
    AMGPreconditioner,
    MultigridHierarchy,
    PreparedFineLevel,
    PreparedHierarchy,
    assemble_hierarchy,
)
from torchalg.preconditioners.base import PreconditionerContext, PreconditionerNotReadyError
from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.utils.spectral import approximate_spectral_radius
from torchalg.utils.tentative import fit_candidates_csr

from ._jacobi_omega import scaled_by_inverse_diagonal
from ._node_strength import sparse_node_strength
from ._presets import PrebuiltCoarsening, gs_setup_cycle, prebuilt_cycle, seeded_draw
from ._prolongation import sparse_make_bridge
from .aggregation import sparse_smoothed_prolongation, sparse_standard_aggregation
from .smoothers import GaussSeidelMemo, resolve_jacobi_default
from .transfer import SparseTransferOperator, TransferOperatorMemo

if TYPE_CHECKING:
    from typing import Self

    from torchalg.multigrid import VCycle
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
    """Setup parameters and stage-scoped prepared state shared by every stage.

    ``gauss_seidel`` and ``transfers`` live exactly as long as one
    ``adaptive_sa_hierarchy`` call: they prepare each distinct level matrix and
    each prolongation once, however many relaxation sweeps or trial cycles
    revisit it. ``setup_cycle`` is the GS V(1,1) cycle prepared through them.
    """

    theta: float
    omega: float
    candidate_iters: int
    max_levels: int
    max_coarse: int
    draw: Callable[[int], torch.Tensor]
    gauss_seidel: GaussSeidelMemo
    transfers: TransferOperatorMemo
    setup_cycle: VCycle

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
        return assemble_hierarchy(
            self.matrices,
            [SparseTransferOperator(prolongation) for prolongation in self.prolongations],
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
    return params.gauss_seidel.prepare(matrix).smooth(
        torch.zeros_like(x), x, params.candidate_iters
    )


def _hierarchy_of(levels: list[_Level], params: _Params) -> MultigridHierarchy:
    """Engine hierarchy over working levels.

    Args:
        levels (list[_Level]): Working levels, finest first.
        params (_Params): Setup parameters; supply the stage's transfer-operator memo.

    Returns:
        MultigridHierarchy: Hierarchy sharing the levels' tensors.
    """
    return assemble_hierarchy(
        [level.A for level in levels],
        [params.transfers(_built(level.P)) for level in levels[:-1]],
    )


def _prepared_fine(level: _Level, params: _Params) -> PreparedFineLevel:
    """``level`` bound through the stage memos, as ``params.setup_cycle.prepare`` binds it.

    Args:
        level (_Level): Working level whose ``A`` and ``P`` are final for the next trial solve.
        params (_Params): Setup parameters; supply the stage's memos.

    Returns:
        PreparedFineLevel: The prepared level.
    """
    return PreparedFineLevel(
        level.A, params.transfers(_built(level.P)), params.gauss_seidel.prepare(level.A)
    )


def _solve(
    hierarchy: PreparedHierarchy, x: torch.Tensor, iterations: int, params: _Params
) -> torch.Tensor:
    """PyAMG's ``solve(b=0, x0=x, tol=1e-20, maxiter=iterations)`` with V-cycles, sparse CSR.

    Args:
        hierarchy (PreparedHierarchy): Trial hierarchy prepared for ``params.setup_cycle``,
            at least two levels.
        x (torch.Tensor): Initial guess.
        iterations (int): Maximum number of cycles.
        params (_Params): Setup parameters; supply the stage's setup cycle.

    Returns:
        torch.Tensor: Iterate after the cycles.
    """
    cycle = params.setup_cycle
    matrix = hierarchy.levels[0].matrix
    residual = matrix @ x
    for _ in range(iterations):
        x = x - cycle.apply(hierarchy, residual)
        residual = matrix @ x
        if torch.linalg.norm(residual) < _SOLVE_TOLERANCE:
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
    tentative, coarse_candidates = fit_candidates_csr(aggregate, level.B)
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
        tentative, coarse = fit_candidates_csr(aggregate, x.unsqueeze(1))
        x = coarse[:, 0]
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
    x = params.random_vector(fine, fine.shape[0])
    if len(levels) == 1:
        return torch.zeros_like(x)
    # One trial hierarchy per candidate: each sub-hierarchy solve below changes only the
    # level it refits, so the other prepared levels and the coarsest inverse are reused.
    prepared = params.setup_cycle.prepare(_hierarchy_of(levels, params))
    prepared_fine = list(prepared.fine)
    x = _solve(prepared, x, params.candidate_iters, params)
    for i in range(len(levels) - 2):
        candidates = torch.cat([levels[i].B, x.unsqueeze(1)], dim=1)
        tentative, coarse = fit_candidates_csr(_built(levels[i].aggregate), candidates)
        levels[i].T = tentative
        x = coarse[:, -1]
        prolongation = _jacobi_prolongation(levels[i].A, tentative, params.omega, params)
        levels[i].P = prolongation
        levels[i + 1].A = form_sparse_sparse(prolongation, levels[i].A)
        bridge = sparse_make_bridge(_built(levels[i + 1].T), levels[i + 1].dofs)
        levels[i + 1].P = _jacobi_prolongation(levels[i + 1].A, bridge, params.omega, params)
        prepared_fine[i + 1] = _prepared_fine(levels[i + 1], params)
        x = _solve(
            PreparedHierarchy(tuple(prepared_fine[i + 1 :]), prepared.coarsest),
            x,
            params.candidate_iters,
            params,
        )
        levels[i + 1].B = coarse[:, :-1].clone()
        levels[i + 1].T = bridge
        levels[i + 1].dofs = candidates.shape[1]
    for level in reversed(levels[:-2]):
        x = _built(level.P) @ x
        rows = torch.nonzero(x).flatten()
        x = params.gauss_seidel.prepare(level.A).smooth(
            torch.zeros_like(x), x, params.candidate_iters, rows=rows
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

    Cost:
        Once per matrix. ``q`` = ``num_candidates``, ``s`` = ``candidate_iters``,
        ``N`` levels with ``n_l`` rows and ``z_l`` stored entries, ``P_l`` the
        prolongation of level ``l``, ``c`` the order of a dense coarsest block
        (it grows with the candidate count: coarse nodes times candidates so far).
        The first candidate costs ``2s`` symmetric Gauss-Seidel iterations per
        level (down and up), ``W = 2s sum_l (4 z_l + 2 n_l)``, plus one
        aggregation, prolongation smoothing and Galerkin product per level. Each
        of the ``q - 1`` later candidates runs ``s`` GS-smoothed V(1,1) trial
        cycles on the whole hierarchy and ``s`` more on each of its ``N - 2``
        nested sub-hierarchies, so its work is quadratic in ``N``:
        ``W = O(s N sum_l (9 z_l + 4 nnz(P_l)))`` for the cycles, plus one
        smoothed-aggregation rebuild of the hierarchy (per later candidate, and
        a final one) and one coarsest-level inverse per later candidate (the
        nested sub-hierarchies share it), each ``W ~ c^3`` for an SPD block or ``~ 24 c^3`` through
        ``pinv`` (``pseudo_inverse_coarse_solver``). Smoothers are prepared once
        per level matrix and shared across the stage. ``D`` is the sum of the
        stage chains, dominated by the Gauss-Seidel triangular recurrences
        (``lv_f + lv_b`` per iteration, ``s`` iterations, a trial cycle per
        stage), the coarsest factorization (``D = O(c)``) and the host
        aggregation (``3 n_l + nnz`` per level); ``I`` is the aggregation.
        Totals, with ``W_SA``, ``D_SA``, ``L_SA`` one smoothed-aggregation build of
        the hierarchy (strength, aggregation, candidate fit, prolongation smoothing
        and Galerkin product per level) and ``W_V``, ``D_V``, ``L_V`` one GS V(1,1)
        trial cycle: ``W = 2s sum_l (4z_l + 2n_l) + (q + 1) W_SA
        + (q - 1)[s(N - 1)(W_V + 2 z_0) + W_c]``,
        ``D = 2s sum_l (lv_f,l + lv_b,l) + (q + 1) D_SA
        + (q - 1)[s(N - 1) D_V + O(c)]``, ``L = 12 s N + (q + 1) L_SA
        + (q - 1)[s(N - 1)(L_V + 4) + 1]`` (six launches per explicit-product GS
        iteration; four for the residual product and convergence test around each
        trial cycle) and ``S = O(N q s)``: one convergence read per trial cycle, one
        ``nonzero`` per level of each masked back-sweep, and the aggregation and
        candidate-fit reads per level. ``Q`` follows each block (every sparse block
        is memory-bound; the dense inverse has ``W/Q`` of order ``c``).
        Work-depth: ``W/D`` is high in the SpMV, SpGEMM and dense blocks and low
        in the Gauss-Seidel and aggregation chains. The setup is by far the
        largest term of the method's total cost; a smoothed-aggregation setup of
        one level is the unit it multiplies.

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
    gauss_seidel = GaussSeidelMemo()
    params = _Params(
        theta,
        omega,
        candidate_iters,
        max_levels,
        max_coarse,
        draw or seeded_draw(seed),
        gauss_seidel=gauss_seidel,
        transfers=TransferOperatorMemo(),
        setup_cycle=gs_setup_cycle(gauss_seidel),
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
    """Adaptive smoothed-aggregation AMG preconditioner (alpha-SA), sparse-CSR sibling.

    Sparse-CSR counterpart of the dense
    ``preconditioners.implementations.amg.adaptive.AdaptiveSAPreconditioner``
    - same preset shape (runs ``adaptive_sa_hierarchy`` in ``setup()``,
    wraps the resulting fixed hierarchy, overrides ``setup`` to ``_commit`` it),
    same defaults, composed entirely from this package's own sparse-native
    kernels/preconditioners plus the shared, format-agnostic
    ``torchalg.multigrid`` engine.

    Cost:
        Setup: ``adaptive_sa_hierarchy`` (its ``W``, ``D``, ``L`` and ``S``), once per
        matrix, plus the solve-time cycle's preparation (``VCycle``'s Setup: one
        diagonal and one Arnoldi estimate per level for the default Jacobi smoother,
        one coarsest pseudo-inverse); it dominates the method's total cost, so
        ``W = W_aSA + W_prep``, ``D = D_aSA + D_prep``, ``L = L_aSA + L_prep``.
        Apply, one V(``a``, ``b``) cycle with weighted-Jacobi smoothing, ``N`` levels,
        ``V = 8`` bytes per ``float64``, per non-coarsest level ``l``:
        ``2(a + b) z_l + 4 nnz(P_l) + 3(a + b) n_l`` flops (the smoother SpMVs and
        vector passes, plus one prolongation and one restriction) and
        ``4(a + b) + 2`` launches; the coarsest level adds a dense matrix-vector
        product with the cached inverse, ``2c^2`` flops, ``Q = V(c^2 + 2c)``, one
        launch. ``W = sum_l [2(a + b) z_l + 4 nnz(P_l) + 3(a + b) n_l] k + 2c^2 k``,
        ``L = sum_l [4(a + b) + 2] + 1``, ``S = 0``; ``Q`` is the sum of the SpMV and
        vector-pass bytes of those blocks, and
        ``D = sum_l [(a + b)(ceil(log2 m_l) + 4) + 2 (ceil(log2 m_P) + 1)]
        + ceil(log2 c) + 1`` with ``m_l`` the largest row count of level ``l``,
        shallow because Jacobi and transfers have no recurrence. The cycle costs
        about ``(2(a + b) + 4 nnz(P)/z_0) C_op`` fine SpMVs plus the coarse
        product, with operator complexity ``C_op = sum_l z_l / z_0``.
        Roofline: ``W/Q`` is about 0.15 flop per byte (memory-bound).
        Work-depth: ``W/D`` is large (about ``10^4``), the most parallel of the
        methods here: the iteration count and launches, not depth, set its time.

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
        self._commit(self._hierarchy_of(result, matrix))
        self._result = result
        self._n_levels = len(result.matrices)
        return self

    @property
    def result(self) -> AdaptiveSAResult:
        """The realized adaptive-setup hierarchy.

        Returns:
            AdaptiveSAResult: Levels, prolongations, and near-null-space
                candidates from the last ``setup()``.
        """
        if self._result is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before the result is available."
            )
        return self._result

    def __str__(self) -> str:
        """Human-readable structural summary.

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
            MultigridHierarchy: The sparse adaptive-SA hierarchy.
        """
        moved = AdaptiveSAResult(
            matrices=tuple(m.to(matrix) for m in result.matrices),
            prolongations=tuple(p.to(matrix) for p in result.prolongations),
            candidates=result.candidates,
            aggregates=result.aggregates,
            strengths=result.strengths,
        )
        return moved.hierarchy
