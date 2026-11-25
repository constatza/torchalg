"""Spectral-radius estimates per setup of the Arnoldi-damped AMG presets, dense and sparse trees.

Every estimate is a full Arnoldi run (``2 nnz m + 4 n m (m + 1)`` flops per pass), so these
counts pin how many a setup pays. Bootstrap and BootCMatch estimate each level matrix once, through
the Jacobi smoother's memoized ``jacobi_spectral_radius`` (a hierarchy built later on the same
fine matrix object reuses the finest one). Adaptive SA also estimates inside its prolongator
smoothing, drawing the start vector from the stream shared with its random candidates; two such
estimates of one matrix start from different vectors, differ in the last digits of a 1% tolerance
estimate, and so cannot be merged without changing the hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
import torch

from torchalg.preconditioners.implementations.amg import _jacobi_omega as dense_jacobi
from torchalg.preconditioners.implementations.amg import adaptive as dense_adaptive
from torchalg.preconditioners.implementations.amg.adaptive import (
    AdaptiveSAPreconditioner as DenseAdaptiveSA,
)
from torchalg.preconditioners.implementations.amg.bootcmatch import (
    BootCMatchPreconditioner as DenseBootCMatch,
)
from torchalg.preconditioners.implementations.amg.bootstrap import (
    BootstrapAMGPreconditioner as DenseBootstrap,
)
from torchalg.sparse.preconditioners.amg import _jacobi_omega as sparse_jacobi
from torchalg.sparse.preconditioners.amg import adaptive as sparse_adaptive
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSA,
)
from torchalg.sparse.preconditioners.amg.bootcmatch import (
    BootCMatchPreconditioner as SparseBootCMatch,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrap,
)

if TYPE_CHECKING:
    from collections.abc import Callable

POISSON_SIZE = 300
SEED = 0
ADAPTIVE_PROLONGATOR_ESTIMATES = 18
ADAPTIVE_FINE_ESTIMATES = 4


@dataclass(frozen=True, slots=True)
class Tree:
    """One implementation tree: its estimating modules and preset classes."""

    jacobi: ModuleType
    adaptive: ModuleType
    adaptive_sa: type
    bootstrap: type
    bootcmatch: type
    to_operand: Callable[[torch.Tensor], torch.Tensor]


TREES = {
    "dense": Tree(
        dense_jacobi, dense_adaptive, DenseAdaptiveSA, DenseBootstrap, DenseBootCMatch, lambda a: a
    ),
    "sparse": Tree(
        sparse_jacobi,
        sparse_adaptive,
        SparseAdaptiveSA,
        SparseBootstrap,
        SparseBootCMatch,
        lambda a: a.to_sparse_csr(),
    ),
}


@pytest.fixture(params=list(TREES.values()), ids=list(TREES))
def tree(request: pytest.FixtureRequest) -> Tree:
    return request.param


@pytest.fixture
def operand(tree: Tree, poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return tree.to_operand(poisson_1d_factory(POISSON_SIZE).to(torch.float64))


@dataclass(slots=True)
class Estimates:
    """Matrix sizes of every estimate, by the module that asked for it."""

    smoother: list[int]
    prolongator: list[int]
    prolongator_values: list[float]


@pytest.fixture
def estimates(tree: Tree, monkeypatch: pytest.MonkeyPatch) -> Estimates:
    recorded = Estimates(smoother=[], prolongator=[], prolongator_values=[])
    for module, sizes in ((tree.jacobi, recorded.smoother), (tree.adaptive, recorded.prolongator)):
        original = module.approximate_spectral_radius

        def counting(
            matrix: torch.Tensor,
            *args: Any,
            _original: Callable[..., torch.Tensor] = original,
            _sizes: list[int] = sizes,
            **kwargs: Any,
        ) -> torch.Tensor:
            _sizes.append(matrix.shape[0])
            estimate = _original(matrix, *args, **kwargs)
            if _sizes is recorded.prolongator:
                recorded.prolongator_values.append(float(estimate))
            return estimate

        monkeypatch.setattr(module, "approximate_spectral_radius", counting)
    return recorded


def _level_sizes(preconditioner: Any) -> list[int]:
    return [int(level.matrix.shape[0]) for level in preconditioner._hierarchy.levels]


def test_bootstrap_estimates_each_non_coarsest_level_once(
    tree: Tree, operand: torch.Tensor, estimates: Estimates
) -> None:
    preconditioner = tree.bootstrap(seed=SEED).setup(operand)

    assert estimates.smoother == _level_sizes(preconditioner)[:-1]
    assert estimates.prolongator == []


def test_bootcmatch_estimates_the_finest_matrix_once_across_hierarchies(
    tree: Tree, operand: torch.Tensor, estimates: Estimates
) -> None:
    preconditioner = tree.bootcmatch(
        seed=SEED, max_levels=4, rho_desired=0.01, max_hierarchies=3
    ).setup(operand)

    hierarchies = preconditioner.hierarchies
    assert len(hierarchies) == 3
    expected = [POISSON_SIZE] + [
        int(level.matrix.shape[0]) for h in hierarchies for level in h.levels[1:-1]
    ]
    assert estimates.smoother == expected
    assert estimates.smoother.count(POISSON_SIZE) == 1
    assert estimates.prolongator == []


def test_adaptive_sa_fine_estimates_differ_by_start_vector(
    tree: Tree, operand: torch.Tensor, estimates: Estimates
) -> None:
    tree.adaptive_sa(seed=SEED, num_candidates=2).setup(operand)

    assert len(estimates.prolongator) == ADAPTIVE_PROLONGATOR_ESTIMATES
    fine = [
        value
        for size, value in zip(estimates.prolongator, estimates.prolongator_values, strict=True)
        if size == POISSON_SIZE
    ]
    assert len(fine) == ADAPTIVE_FINE_ESTIMATES
    assert len(set(fine)) == len(fine)
    assert estimates.smoother == [POISSON_SIZE, 200, 68, 24]
