"""Bootstrap AMG's outer setup loop, shared by the dense and sparse trees.

Promoted out of
``preconditioners.implementations.amg.bootstrap.{BootstrapSetup,
BootstrapAMGResult,_hierarchy_of}`` (see ``docs/plan.md``'s "Correction:
dense and sparse must be separate implementations" for the general
promotion precedent, and its "Next steps" entry on this specific module for
the design discussion that produced it). Confirmed by direct reading: this
outer loop (``run``, ``_build_levels``, ``_improve_levels``,
``_v_r_after_build``, ``_merge_level_vectors``, plus the module-level
``_random_test_vectors``/``_seeded_test_vectors``/``_relax_columns``/
``_improve_test_vectors`` helpers) only ever operates on plain tensors,
dicts, and injected callables - it never reads a matrix's storage format
directly. Four places originally hardcoded a concrete dense type with no
injection seam; each becomes a constructor field here, the same
dependency-injection pattern already used throughout this codebase
(``relaxation`` on ``hcr_operator``, ``coarse_solver`` on ``VCycle``/
``WCycle``, ``coarsening``/``cycle`` on ``AMGPreconditioner`` itself) - the
caller picks which concrete relaxation/cycle/coarsening-class/transfer-
operator-class to inject *before* any matrix exists, never by reading
``A.is_sparse_csr`` internally, so this is not the "format-follows-input
router" pattern ``docs/plan.md`` already rejected once (that router had to
import both trees to decide; nothing here ever does):

- ``relaxation`` (was: ``GaussSeidelSmoother().smooth`` hardcoded in
  ``run``).
- ``setup_cycle`` (was: module-level ``GS_SETUP_CYCLE`` hardcoded in
  ``_improve_test_vectors``) - the GS V(1,1) cycle used only for setup-time
  test-vector improvement, built from the same injected ``relaxation``.
- ``coarsening_factory`` (was: ``BAMGCoarsening(...)`` hardcoded in
  ``_coarsening_from``) - ``(vectors, relaxation, draw) -> CoarseningStrategy``.
- ``transfer_operator_factory`` (was: ``DenseTransferOperator(...)``
  hardcoded in ``_hierarchy_of``) - ``(prolongation) -> TransferOperator``.

**Not promoted: MGE (``k_e`` / ``_mge.multigrid_eigensolver``).** Multigrid-
eigensolver enrichment has its own un-ported dense-specific internals
(``torch.linalg.cholesky``/``eigh`` directly on a level matrix, not just a
coarsest-level densify-first adapter) - the same class of gap
``docs/plan.md`` already scopes as part of the separately-deferred
``BAMGCoarsening`` sparse follow-on, not something to smuggle in here.
``eigensolver`` is therefore an optional field, ``None`` by default; a
caller passing ``k_e > 0`` with no ``eigensolver`` configured gets a clear
``ValueError`` here, not a confusing dense-only crash three calls deep once
some future sparse preset wires this module up without MGE support.

**Not promoted: ``nu``/``delta``/``theta_ad``/``caliber``/``gamma``/
``use_lsr``.** These are ``BAMGCoarsening``-constructor parameters, not
outer-loop parameters - keeping a second copy of them here (as this
module's own fields, alongside a ``coarsening_factory`` closure that
already captures its own copies to build each ``BAMGCoarsening``) would be
exactly the single-source-of-truth duplication this codebase's own
conventions rule out. They stay only inside whatever closure a caller
passes as ``coarsening_factory``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from .hierarchy import MultigridHierarchy, MultigridLevel
from .protocols import MultigridCycle, TransferOperator

_Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Shape of a ``SmootherBase.smooth``-style relaxation callable: ``(A, rhs, x, steps) -> x``."""


class _BootstrapCoarseningStrategy(Protocol):
    """``CoarseningStrategy`` plus ``BAMGCoarsening``'s own public extensions.

    Narrower than importing the generic ``CoarseningStrategy`` protocol and
    reaching for ``# type: ignore`` at every call site below: this module
    only ever hands a caller-built ``BAMGCoarsening`` (dense or sparse, both
    sharing this exact shape) around internally, so it depends on the
    interface it actually uses (Interface Segregation) rather than the
    broader base protocol a generic ``CoarseningStrategy`` promises.
    """

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, TransferOperator]:
        """Build a coarse grid and the associated transfer operator."""
        ...

    def test_vectors_for(self, matrix: torch.Tensor) -> torch.Tensor:
        """Currently stored test vectors for ``matrix``'s dimension."""
        ...

    def set_test_vectors(self, matrix: torch.Tensor, vectors: torch.Tensor) -> None:
        """Overwrite the stored test vectors for ``matrix``'s dimension."""
        ...

    @property
    def last_prolongation(self) -> torch.Tensor:
        """Raw prolongation tensor from the most recent ``build_transfer`` call."""
        ...


_CoarseningFactory = Callable[
    [torch.Tensor, _Relaxation, Callable[[int], torch.Tensor]], _BootstrapCoarseningStrategy
]
"""``(test_vectors, relaxation, draw) -> _BootstrapCoarseningStrategy`` - builds
a fresh, level-0-seeded ``BAMGCoarsening`` (dense or sparse)."""

_TransferOperatorFactory = Callable[[torch.Tensor], TransferOperator]
"""``(prolongation) -> TransferOperator`` - wraps a raw prolongation tensor
(dense or sparse CSR) as a ``DenseTransferOperator``/``SparseTransferOperator``."""

_Eigensolver = Callable[
    [list[torch.Tensor], list[torch.Tensor], int, _Relaxation, int], dict[int, torch.Tensor]
]
"""Shape of ``_mge.multigrid_eigensolver``: ``(levels, prolongations, k_e,
relaxation, eta) -> per-level eigenvector columns``."""


def _random_test_vectors(
    matrix: torch.Tensor, k: int, draw: Callable[[int], torch.Tensor]
) -> torch.Tensor:
    """``k`` random test vectors on ``matrix``'s grid ([STATUS14] Sec. 4: TVs start random).

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        k (int): Number of test vectors.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.

    Returns:
        torch.Tensor: Test vectors, shape ``(n, k)``.
    """
    columns = [draw(matrix.shape[0]).to(dtype=matrix.dtype, device=matrix.device) for _ in range(k)]
    return torch.stack(columns, dim=1)


def _seeded_test_vectors(
    matrix: torch.Tensor,
    k_r: int,
    draw: Callable[[int], torch.Tensor],
    seed_vectors: torch.Tensor | None,
) -> torch.Tensor:
    """``k_r`` random test vectors, plus known near-null vectors if given ([STATUS14] Table 3).

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        k_r (int): Number of random test vectors to draw.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.
        seed_vectors (torch.Tensor | None): Known near-null vectors, shape
            ``(n, k_seed)``; ``None`` for no seeding (the historical
            behavior).

    Returns:
        torch.Tensor: Test vectors, shape ``(n, k_r + k_seed)`` (or
        ``(n, k_r)`` when ``seed_vectors`` is ``None``).
    """
    random_vectors = _random_test_vectors(matrix, k_r, draw)
    if seed_vectors is None:
        return random_vectors
    return torch.cat(
        [random_vectors, seed_vectors.to(dtype=matrix.dtype, device=matrix.device)], dim=1
    )


def _relax_columns(
    matrix: torch.Tensor, vectors: torch.Tensor, relaxation: _Relaxation, sweeps: int
) -> torch.Tensor:
    """Relax every column of ``vectors`` on ``matrix @ x = 0`` ([STATUS14] eq. 4.1).

    Each test-vector column is independent of every other (``relaxation``
    never couples columns), so all ``k`` columns are relaxed in one batched
    call rather than a Python loop - both the dense and sparse
    ``GaussSeidelSmoother.smooth`` (the ``relaxation`` this is always
    injected with) accept a ``(n, k)`` ``x``/``rhs`` directly.

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``, batched over ``x``'s trailing dimension.
        sweeps (int): Number of sweeps ``eta``.

    Returns:
        torch.Tensor: Relaxed test vectors, shape ``(n, k)``.
    """
    zero_rhs = torch.zeros_like(vectors)
    return relaxation(matrix, zero_rhs, vectors, sweeps)


def _improve_test_vectors(
    matrix: torch.Tensor,
    vectors: torch.Tensor,
    hierarchy: MultigridHierarchy,
    iterations: int,
    setup_cycle: MultigridCycle,
) -> torch.Tensor:
    """Improve ``vectors`` by running ``setup_cycle`` on ``matrix @ x = 0`` ([STATUS14] Sec. 4).

    Uses linearity of the cycle for a zero right-hand side: one cycle from
    ``x`` is ``x - setup_cycle.apply(hierarchy, matrix @ x)``. All ``k``
    columns are improved in one batched pass rather than a Python loop:
    every step inside ``setup_cycle.apply`` is either the (now batched)
    smoother or a plain matrix product - ``matrix @ x``, the transfer
    operators' ``P @ x``/``P.T @ x``, and the coarse solvers
    (``torch.linalg.solve``/``torch.linalg.pinv(...) @ rhs``) - all of which
    already operate columnwise for a ``(n, k)`` right-hand side with no
    further changes.

    Args:
        matrix (torch.Tensor): Finest-level matrix, shape ``(n, n)``.
        vectors (torch.Tensor): Test vectors to improve, shape ``(n, k)``.
        hierarchy (MultigridHierarchy): Current (partial) hierarchy.
        iterations (int): Number of cycles per test vector.
        setup_cycle (MultigridCycle): The setup-time cycle (e.g. dense/sparse
            GS V(1,1)) - injected, not hardcoded.

    Returns:
        torch.Tensor: Improved test vectors, shape ``(n, k)``.
    """
    x = vectors
    for _ in range(iterations):
        x = x - setup_cycle.apply(hierarchy, matrix @ x)
    return x


def _hierarchy_of(
    levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
    prolongations: list[torch.Tensor] | tuple[torch.Tensor, ...],
    transfer_operator_factory: _TransferOperatorFactory,
) -> MultigridHierarchy:
    """Engine hierarchy over plain level-matrix/prolongation lists.

    Args:
        levels (list[torch.Tensor] | tuple[torch.Tensor, ...]): Level
            matrices, finest first.
        prolongations (list[torch.Tensor] | tuple[torch.Tensor, ...]):
            Prolongations, one per level except the coarsest.
        transfer_operator_factory (_TransferOperatorFactory): Wraps a raw
            prolongation tensor as a ``TransferOperator`` - injected, not
            hardcoded.

    Returns:
        MultigridHierarchy: Hierarchy sharing the given tensors.
    """
    last = len(levels) - 1
    return MultigridHierarchy(
        tuple(
            MultigridLevel(
                matrix=matrix,
                transfer=None if index == last else transfer_operator_factory(prolongations[index]),
            )
            for index, matrix in enumerate(levels)
        )
    )


@dataclass(frozen=True)
class BootstrapAMGResult:
    """Result of the Bootstrap AMG setup.

    Attributes:
        matrices (tuple[torch.Tensor, ...]): Level matrices ``A_l``, finest
            first.
        prolongations (tuple[torch.Tensor, ...]): ``P_l`` mapping level
            ``l+1`` to level ``l`` (``R = P^T``).
        candidates (torch.Tensor): Finest-level test vectors after the final
            bootstrap cycle, shape ``(n, k_r)``.
        transfer_operator_factory (_TransferOperatorFactory): Wraps a raw
            prolongation tensor as a ``TransferOperator`` - threaded through
            so ``.hierarchy`` stays a zero-arg property.
    """

    matrices: tuple[torch.Tensor, ...]
    prolongations: tuple[torch.Tensor, ...]
    candidates: torch.Tensor
    transfer_operator_factory: _TransferOperatorFactory

    @property
    def hierarchy(self) -> MultigridHierarchy:
        """The multigrid hierarchy for the engine's cycles.

        Returns:
            MultigridHierarchy: Levels with transfer operators built via
            ``transfer_operator_factory``.
        """
        return _hierarchy_of(self.matrices, self.prolongations, self.transfer_operator_factory)


@dataclass(frozen=True)
class BootstrapSetup:
    """Bootstrap AMG setup: outer loop parameters plus injected format-specific factories.

    Implements ``docs/bootstrap-amg.md`` Sec. 4.1/Sec. 5: builds one
    hierarchy from ``k_r`` random test vectors relaxed ``eta`` sweeps per
    level, coarsening while ``len(levels) < max_levels and
    levels[-1].shape[0] > max_coarse``, then for ``n_bootstrap_cycles``
    cycles replaces plain relaxation with a full ``setup_cycle.apply`` on
    ``A x = 0`` to improve the **finest** level's test vectors and rebuilds
    the whole hierarchy from them. Each coarse level's vectors are
    re-derived from scratch during that rebuild (restriction by ``P^T``
    plus ``eta`` relaxation sweeps in ``_build_levels``), not improved in
    place.

    Args:
        coarsening_factory (_CoarseningFactory): Builds a fresh
            ``BAMGCoarsening`` (dense or sparse) seeded with a level's test
            vectors - the caller's own closure captures whatever
            ``nu``/``delta``/``theta_ad``/``caliber``/``gamma``/``use_lsr``
            it wants.
        transfer_operator_factory (_TransferOperatorFactory): Wraps a raw
            prolongation tensor as a ``TransferOperator``.
        relaxation (_Relaxation): Relaxation callable used by CR and by
            setup-time test-vector relaxation (dense or sparse Gauss-Seidel).
        setup_cycle (MultigridCycle): The setup-time cycle used only for
            ``_improve_levels``' test-vector improvement (e.g. a GS V(1,1)
            built from the same ``relaxation``).
        eta (int): Relaxation sweeps per test vector per level.
        k_r (int): Number of relaxation-derived test vectors.
        k_e (int): Number of multigrid-eigensolver (MGE) eigenvector
            approximations per bootstrap cycle; ``0`` disables MGE (the
            historical default - MGE is an optional enhancement over an
            already-valid relaxation-only baseline, not a requirement).
        eigensolver (_Eigensolver | None): MGE implementation (dense-only
            today - see the module docstring); required if ``k_e > 0``.
        n_bootstrap_cycles (int): Number of bootstrap-cycle passes.
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
    """

    coarsening_factory: _CoarseningFactory
    transfer_operator_factory: _TransferOperatorFactory
    relaxation: _Relaxation
    setup_cycle: MultigridCycle
    eta: int = 4
    k_r: int = 8
    k_e: int = 0
    eigensolver: _Eigensolver | None = None
    n_bootstrap_cycles: int = 2
    max_levels: int = 10
    max_coarse: int = 10

    def __post_init__(self) -> None:
        """Validate that MGE has an implementation whenever it is requested.

        Raises:
            ValueError: If ``k_e > 0`` but ``eigensolver`` is ``None``.
        """
        if self.k_e > 0 and self.eigensolver is None:
            raise ValueError(
                "k_e > 0 requires an eigensolver (MGE has no sparse implementation yet - "
                "see the module docstring)"
            )

    def _coarsening_from(
        self, vectors: torch.Tensor, draw: Callable[[int], torch.Tensor]
    ) -> _BootstrapCoarseningStrategy:
        """Build a fresh coarsening strategy seeded with ``vectors``.

        Args:
            vectors (torch.Tensor): Finest-level test vectors.
            draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
                random-vector source for CR.

        Returns:
            _BootstrapCoarseningStrategy: Fresh coarsening strategy.
        """
        return self.coarsening_factory(vectors, self.relaxation, draw)

    def _improve_levels(
        self,
        levels: list[torch.Tensor],
        prolongations: list[torch.Tensor],
        level_vectors: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Improve every level's relaxation-derived test vectors via its own sub-hierarchy cycle.

        Args:
            levels (list[torch.Tensor]): Level matrices, finest first.
            prolongations (list[torch.Tensor]): Prolongations, one per level
                except the coarsest.
            level_vectors (dict[int, torch.Tensor]): Each level's current
                ``V^r`` test vectors, keyed by matrix dimension (every level
                except the coarsest).

        Returns:
            dict[int, torch.Tensor]: Improved ``V^r`` test vectors keyed by
            each improved level's matrix dimension.
        """
        return {
            levels[index].shape[0]: _improve_test_vectors(
                levels[index],
                level_vectors[levels[index].shape[0]],
                _hierarchy_of(
                    levels[index:], prolongations[index:], self.transfer_operator_factory
                ),
                self.eta,
                self.setup_cycle,
            )
            for index in range(len(levels) - 1)
            if levels[index].shape[0] in level_vectors
        }

    @staticmethod
    def _merge_level_vectors(
        relaxed: dict[int, torch.Tensor], enriched: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Concatenate MGE eigenvector columns onto the relaxation-improved vectors, per level.

        Args:
            relaxed (dict[int, torch.Tensor]): Per-level vectors from
                ``_improve_levels``, keyed by matrix dimension.
            enriched (dict[int, torch.Tensor]): Per-level MGE eigenvector
                approximations, same keys.

        Returns:
            dict[int, torch.Tensor]: Concatenated per-level vectors, same
            keys as both inputs.
        """
        return {
            dimension: torch.cat([vectors, enriched[dimension]], dim=1)
            for dimension, vectors in relaxed.items()
        }

    @staticmethod
    def _v_r_after_build(
        coarsening: _BootstrapCoarseningStrategy, levels: list[torch.Tensor], k_r_total: int
    ) -> dict[int, torch.Tensor]:
        """The ``V^r`` (relaxation-derived) columns of every level's post-``_build_levels`` vectors.

        Args:
            coarsening (_BootstrapCoarseningStrategy): Coarsening
                strategy just returned from a ``_build_levels`` call.
            levels (list[torch.Tensor]): That same call's returned levels.
            k_r_total (int): ``V^r``'s fixed column count (``k_r`` plus any
                ``seed_vectors``), constant for the whole ``run`` call.

        Returns:
            dict[int, torch.Tensor]: ``V^r`` columns per level, keyed by
            matrix dimension (every level except the coarsest).
        """
        return {
            level.shape[0]: coarsening.test_vectors_for(level)[:, :k_r_total]
            for level in levels[:-1]
        }

    def _build_levels(
        self,
        matrix: torch.Tensor,
        coarsening: _BootstrapCoarseningStrategy,
        relaxation: _Relaxation,
        level_vectors: dict[int, torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Coarsen ``matrix`` down to ``max_levels``/``max_coarse``.

        Compatible relaxation can legitimately converge with ``C = empty
        set`` on an already-coarsened level, producing a zero-column
        prolongation and a zero-size coarse matrix. A coarsening pass is
        discarded, and the loop stops with ``current`` standing as the
        final, coarsest level, whenever it is degenerate in either
        direction: ``coarse_matrix.shape[0] == 0`` or
        ``coarse_matrix.shape[0] >= current.shape[0]``.

        Args:
            matrix (torch.Tensor): Finest-level matrix.
            coarsening (_BootstrapCoarseningStrategy): Coarsening
                strategy, already seeded with the finest-level test vectors.
            relaxation (_Relaxation): Relaxation callable applied to the test
                vectors before each level is coarsened.
            level_vectors (dict[int, torch.Tensor] | None): Per-level
                test vectors from ``_improve_levels``, keyed by matrix
                dimension, overriding ``coarsening``'s restriction-derived
                starting vectors for that level before its leading
                ``eta``-sweep relaxation; ``None`` on the very first
                hierarchy build.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Level matrices
            (finest first) and prolongations.
        """
        levels = [matrix]
        prolongations: list[torch.Tensor] = []
        while len(levels) < self.max_levels and levels[-1].shape[0] > self.max_coarse:
            current = levels[-1]
            if level_vectors is not None and current.shape[0] in level_vectors:
                coarsening.set_test_vectors(current, level_vectors[current.shape[0]])
            relaxed = _relax_columns(
                current, coarsening.test_vectors_for(current), relaxation, self.eta
            )
            coarsening.set_test_vectors(current, relaxed)
            coarse_matrix, _ = coarsening.build_transfer(current)
            if coarse_matrix.shape[0] == 0 or coarse_matrix.shape[0] >= current.shape[0]:
                break
            levels.append(coarse_matrix)
            prolongations.append(coarsening.last_prolongation)
        return levels, prolongations

    def run(
        self,
        matrix: torch.Tensor,
        draw: Callable[[int], torch.Tensor],
        seed_vectors: torch.Tensor | None = None,
        test_vector_draw: Callable[[int], torch.Tensor] | None = None,
    ) -> BootstrapAMGResult:
        """Run the Bootstrap AMG setup algorithm.

        Args:
            matrix (torch.Tensor): SPD system matrix, shape ``(n, n)``
                (dense or sparse CSR).
            draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
                random-vector source for compatible relaxation's own
                internal convergence-rate probe (``cr_rate``).
            seed_vectors (torch.Tensor | None): Known near-null vectors,
                shape ``(n, k_seed)``, seeded alongside the ``k_r`` random
                draws; ``None`` for no seeding.
            test_vector_draw (Callable[[int], torch.Tensor] | None):
                Random-vector source for the initial ``k_r`` test vectors
                specifically; ``None`` reuses ``draw`` for this too.

        Returns:
            BootstrapAMGResult: Level matrices, prolongations and the final
            finest-level test vectors.
        """
        vectors = _seeded_test_vectors(
            matrix,
            self.k_r,
            test_vector_draw if test_vector_draw is not None else draw,
            seed_vectors,
        )
        k_r_total = vectors.shape[1]
        coarsening = self._coarsening_from(vectors, draw)
        levels, prolongations = self._build_levels(matrix, coarsening, self.relaxation)
        vectors = coarsening.test_vectors_for(matrix)
        relaxed_level_vectors = self._v_r_after_build(coarsening, levels, k_r_total)

        for _ in range(self.n_bootstrap_cycles):
            if len(levels) < 2:
                break
            relaxed_level_vectors = self._improve_levels(
                levels, prolongations, relaxed_level_vectors
            )
            level_vectors = relaxed_level_vectors
            if self.k_e > 0:
                assert self.eigensolver is not None  # enforced by __post_init__
                enriched = self.eigensolver(
                    levels, prolongations, self.k_e, self.relaxation, self.eta
                )
                level_vectors = self._merge_level_vectors(relaxed_level_vectors, enriched)
            vectors = level_vectors[matrix.shape[0]]
            coarsening = self._coarsening_from(vectors, draw)
            levels, prolongations = self._build_levels(
                matrix, coarsening, self.relaxation, level_vectors=level_vectors
            )
            vectors = coarsening.test_vectors_for(matrix)
            relaxed_level_vectors = self._v_r_after_build(coarsening, levels, k_r_total)

        return BootstrapAMGResult(
            matrices=tuple(levels),
            prolongations=tuple(prolongations),
            candidates=vectors,
            transfer_operator_factory=self.transfer_operator_factory,
        )
