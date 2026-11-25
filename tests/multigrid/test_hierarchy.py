"""Tests for the pure ``build_hierarchy`` function.

Relocated from
``tests/solver/preconditioners/implementations/test_amg_hierarchy.py``
(``TestBuildHierarchy``) alongside the promotion of ``hierarchy.py`` from
``torchalg.preconditioners.implementations.amg`` into ``torchalg.multigrid``
- same convention used for the ``torchalg.utils``
promotions (``tests/solver/utils/``), adapted to this package's own
top-level name since it is a shared engine, not a ``utils`` leaf.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid.coarse_solve import lu_coarse_solver
from torchalg.multigrid.cycle import VCycle, WCycle
from torchalg.multigrid.hierarchy import (
    MultigridCoarsestLevel,
    MultigridFineLevel,
    MultigridHierarchy,
    PreparedCoarsestLevel,
    PreparedFineLevel,
    PreparedHierarchy,
    assemble_hierarchy,
    build_hierarchy,
    prepare_hierarchy,
)
from torchalg.preconditioners.implementations.amg import AggregationCoarsening, JacobiSmoother


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


@pytest.fixture
def coarsening() -> AggregationCoarsening:
    """Default smoothed-aggregation coarsening."""
    return AggregationCoarsening()


class TestBuildHierarchy:
    @pytest.mark.parametrize("n_levels", [2, 3])
    def test_level_count_and_level_kinds(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening, n_levels: int
    ) -> None:
        hierarchy = build_hierarchy(poisson_matrix, coarsening, n_levels)
        assert len(hierarchy.levels) == n_levels
        assert isinstance(hierarchy.coarsest, MultigridCoarsestLevel)
        assert len(hierarchy.fine) == n_levels - 1
        assert all(isinstance(level, MultigridFineLevel) for level in hierarchy.fine)
        assert hierarchy.levels == (*hierarchy.fine, hierarchy.coarsest)

    def test_finest_level_is_input_matrix_and_sizes_shrink(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        hierarchy = build_hierarchy(poisson_matrix, coarsening, 3)
        sizes = [level.matrix.shape[0] for level in hierarchy.levels]
        assert hierarchy.levels[0].matrix is poisson_matrix
        assert sizes == sorted(sizes, reverse=True)
        assert len(set(sizes)) == len(sizes)


class TestPreparedHierarchyTypes:
    """Preparation turns the unprepared level/hierarchy types into the prepared ones."""

    def test_preparation_changes_the_type_not_just_optional_fields(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        built = build_hierarchy(poisson_matrix, coarsening, 3)

        prepared = prepare_hierarchy(built, JacobiSmoother(), lu_coarse_solver)

        assert isinstance(built, MultigridHierarchy)
        assert isinstance(prepared, PreparedHierarchy)
        assert all(isinstance(level, PreparedFineLevel) for level in prepared.levels[:-1])
        assert isinstance(prepared.levels[-1], PreparedCoarsestLevel)

    def test_prepared_levels_carry_their_state_non_optionally(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        prepared = prepare_hierarchy(
            build_hierarchy(poisson_matrix, coarsening, 2), JacobiSmoother(), lu_coarse_solver
        )
        fine, coarsest = prepared.levels

        assert isinstance(fine, PreparedFineLevel)
        assert isinstance(coarsest, PreparedCoarsestLevel)
        rhs = torch.ones(fine.matrix.shape[0], dtype=fine.matrix.dtype)
        assert fine.smoother.residual(rhs, torch.zeros_like(rhs)).shape == rhs.shape
        assert callable(coarsest.coarse_solve)

    def test_preparing_a_prepared_hierarchy_returns_it_unchanged(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        prepared = prepare_hierarchy(
            build_hierarchy(poisson_matrix, coarsening, 3), JacobiSmoother(), lu_coarse_solver
        )

        assert prepare_hierarchy(prepared, JacobiSmoother(), lu_coarse_solver) is prepared


class TestHierarchyStructure:
    """A hierarchy is fine levels plus exactly one coarsest level, by construction."""

    @pytest.fixture
    def built(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> MultigridHierarchy:
        return build_hierarchy(poisson_matrix, coarsening, 3)

    @pytest.fixture
    def prepared(self, built: MultigridHierarchy) -> PreparedHierarchy:
        return prepare_hierarchy(built, JacobiSmoother(), lu_coarse_solver)

    def test_hierarchy_without_a_coarsest_level_cannot_be_constructed(
        self, built: MultigridHierarchy
    ) -> None:
        # Regression: ``MultigridHierarchy(h.levels[:-1])`` used to build, then the
        # cycle raised ``IndexError`` past the last level.
        with pytest.raises(TypeError):
            MultigridHierarchy(built.levels[:-1])  # ty: ignore[missing-argument, invalid-argument-type]

    def test_prepared_hierarchy_without_a_coarsest_level_cannot_be_constructed(
        self, prepared: PreparedHierarchy
    ) -> None:
        with pytest.raises(TypeError):
            PreparedHierarchy(prepared.levels[:-1])  # ty: ignore[missing-argument, invalid-argument-type]

    def test_coarsest_level_in_the_middle_is_rejected_at_construction(
        self, built: MultigridHierarchy
    ) -> None:
        # Regression: a middle level built with ``transfer=None`` prepared as a coarsest
        # level and the cycle silently stopped there.
        middle = MultigridCoarsestLevel(built.fine[0].matrix)
        with pytest.raises(TypeError, match=r"fine\[0\]"):
            MultigridHierarchy(
                (middle, *built.fine[1:]),  # ty: ignore[invalid-argument-type]
                built.coarsest,
            )

    def test_fine_level_as_the_coarsest_is_rejected_at_construction(
        self, built: MultigridHierarchy
    ) -> None:
        with pytest.raises(TypeError, match="coarsest"):
            MultigridHierarchy(built.fine, built.fine[0])  # ty: ignore[invalid-argument-type]

    def test_prepared_level_kinds_are_checked_at_construction(
        self, prepared: PreparedHierarchy
    ) -> None:
        with pytest.raises(TypeError, match=r"fine\[1\]"):
            PreparedHierarchy(
                (prepared.fine[0], prepared.coarsest),  # ty: ignore[invalid-argument-type]
                prepared.coarsest,
            )
        with pytest.raises(TypeError, match="coarsest"):
            PreparedHierarchy(prepared.fine, prepared.fine[0])  # ty: ignore[invalid-argument-type]

    def test_prepared_kinds_follow_the_unprepared_kinds(
        self, built: MultigridHierarchy, prepared: PreparedHierarchy
    ) -> None:
        assert len(prepared.fine) == len(built.fine)
        assert prepared.levels == (*prepared.fine, prepared.coarsest)
        assert all(
            prepared_level.matrix is level.matrix
            for prepared_level, level in zip(prepared.levels, built.levels, strict=True)
        )

    @pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
    @pytest.mark.parametrize("n_levels", [1, 2, 3, 4])
    def test_cycle_returns_a_vector_for_every_depth(
        self,
        cycle_class: type[VCycle | WCycle],
        n_levels: int,
        poisson_matrix: torch.Tensor,
        coarsening: AggregationCoarsening,
    ) -> None:
        cycle = cycle_class(JacobiSmoother())
        prepared = cycle.prepare(build_hierarchy(poisson_matrix, coarsening, n_levels))
        rhs = torch.ones(poisson_matrix.shape[0], dtype=poisson_matrix.dtype)

        assert cycle.apply(prepared, rhs).shape == rhs.shape

    def test_single_level_hierarchy_is_just_the_coarsest_level(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        hierarchy = build_hierarchy(poisson_matrix, coarsening, 1)

        assert hierarchy.fine == ()
        assert hierarchy.coarsest.matrix is poisson_matrix

    def test_assemble_requires_one_transfer_per_fine_level(self, built: MultigridHierarchy) -> None:
        matrices = [level.matrix for level in built.levels]
        transfers = [level.transfer for level in built.fine]

        assembled = assemble_hierarchy(matrices, transfers)
        assert all(
            a.matrix is b.matrix for a, b in zip(assembled.levels, built.levels, strict=True)
        )
        assert all(
            a.transfer is b.transfer for a, b in zip(assembled.fine, built.fine, strict=True)
        )
        with pytest.raises(ValueError, match="transfers"):
            assemble_hierarchy(matrices, transfers[:-1])
