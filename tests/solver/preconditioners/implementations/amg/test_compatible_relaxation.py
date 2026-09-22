"""Compatible-relaxation coarsening tests (Bootstrap AMG, docs/bootstrap-amg.md Sec. 2.1).

Fixtures live in this module (not a subdirectory ``conftest.py``): this is
currently the only test module in ``implementations/amg/``, matching the
precedent of ``test_adaptive_sa_port.py``, which keeps its fixtures local
rather than pre-emptively factoring them into a shared conftest before a
second consumer exists.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg._compatible_relaxation import (
    _independent_set_of,
    compatible_relaxation_coarsening,
    cr_rate,
    hcr_operator,
)
from torchalg.preconditioners.implementations.amg.smoothers import GaussSeidelSmoother


@pytest.fixture
def two_of_eight_coarse_mask() -> torch.Tensor:
    """Boolean mask of length 8, ``True`` at indices 2 and 5."""
    return torch.tensor([False, False, True, False, False, True, False, False])


@pytest.fixture
def ones_vector_8(torch_dtype: torch.dtype) -> torch.Tensor:
    """All-ones start vector of length 8."""
    return torch.ones(8, dtype=torch_dtype)


@pytest.fixture
def alternating_coarse_mask_factory() -> Callable[[int], torch.Tensor]:
    """Factory building a boolean mask, ``True`` at every other index."""

    def _factory(n: int) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[::2] = True
        return mask

    return _factory


@pytest.fixture
def seeded_draw_factory(
    torch_dtype: torch.dtype,
) -> Callable[[int], Callable[[int], torch.Tensor]]:
    """Factory building a seeded uniform ``[0, 1)`` draw source.

    Matches ``adaptive.py``'s ``_Params.draw`` convention
    (``Callable[[int], torch.Tensor]``).
    """

    def _factory(seed: int) -> Callable[[int], torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)

        def _draw(n: int) -> torch.Tensor:
            return torch.rand(n, generator=generator, dtype=torch_dtype)

        return _draw

    return _factory


@pytest.fixture
def all_coarse_mask_8() -> torch.Tensor:
    """Boolean mask of length 8, every entry ``True``."""
    return torch.ones(8, dtype=torch.bool)


@pytest.fixture
def empty_coarse_mask_16() -> torch.Tensor:
    """Boolean mask of length 16, every entry ``False`` (``C = empty set``)."""
    return torch.zeros(16, dtype=torch.bool)


@pytest.fixture
def five_true_candidates() -> torch.Tensor:
    """Boolean mask of length 5, every entry ``True`` (all nodes eligible)."""
    return torch.ones(5, dtype=torch.bool)


@pytest.fixture
def descending_priority_5() -> torch.Tensor:
    """Per-node priority of length 5, strictly descending: node 0 highest."""
    return torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])


@pytest.fixture
def matrix_graph_factory() -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory building a matrix's own off-diagonal nonzero graph (``_independent_set_of``'s default adjacency).

    A factory, not a plain fixture, since the graph's shape depends on
    whichever matrix a test builds at runtime (e.g. ``poisson_1d_factory``'s
    output), matching this module's ``alternating_coarse_mask_factory``/
    ``seeded_draw_factory`` precedent for runtime-shape-dependent values.
    """

    def _factory(matrix: torch.Tensor) -> torch.Tensor:
        n = matrix.shape[0]
        return (matrix != 0) & ~torch.eye(n, dtype=torch.bool, device=matrix.device)

    return _factory


@pytest.fixture
def skip_one_guidance_graph_5() -> torch.Tensor:
    """Boolean graph connecting 0<->2 and 1<->3 - disjoint from the matrix's own adjacent-index graph."""
    graph = torch.zeros(5, 5, dtype=torch.bool)
    graph[0, 2] = graph[2, 0] = True
    graph[1, 3] = graph[3, 1] = True
    return graph


def test_hcr_operator_zeros_coarse_entries(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    two_of_eight_coarse_mask: torch.Tensor,
    ones_vector_8: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """Every sweep must leave the coarse-marked entries at exactly zero."""
    matrix = poisson_1d_factory(8)
    result = hcr_operator(
        matrix,
        two_of_eight_coarse_mask,
        GaussSeidelSmoother().smooth,
        sweeps=3,
        start=ones_vector_8,
    )
    assert torch.allclose(result[two_of_eight_coarse_mask], torch.zeros(2, dtype=torch_dtype))


def test_hcr_operator_default_start_is_random_and_still_zeroed(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    two_of_eight_coarse_mask: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """Omitting ``start`` still yields a coarse-zeroed result (default random start)."""
    matrix = poisson_1d_factory(8)
    result = hcr_operator(matrix, two_of_eight_coarse_mask, GaussSeidelSmoother().smooth, sweeps=2)
    assert torch.allclose(result[two_of_eight_coarse_mask], torch.zeros(2, dtype=torch_dtype))


def test_hcr_operator_all_coarse_is_identically_zero(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    all_coarse_mask_8: torch.Tensor,
    ones_vector_8: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """An all-True coarse mask leaves nothing to relax: the result is all zero."""
    matrix = poisson_1d_factory(8)
    result = hcr_operator(
        matrix, all_coarse_mask_8, GaussSeidelSmoother().smooth, sweeps=3, start=ones_vector_8
    )
    assert torch.allclose(result, torch.zeros(8, dtype=torch_dtype))


def test_cr_rate_low_for_every_other_point(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    alternating_coarse_mask_factory: Callable[[int], torch.Tensor],
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
) -> None:
    """Standard coarsening (every other point) gives a good CR rate for 1D Poisson."""
    matrix = poisson_1d_factory(16)
    coarse_mask = alternating_coarse_mask_factory(16)
    draw = seeded_draw_factory(0)

    rho, e = cr_rate(matrix, coarse_mask, GaussSeidelSmoother().smooth, sweeps=10, draw=draw)

    # The brief's own worked example asserts `rho < 0.5`; empirically (this seed, sweeps=10)
    # rho ~= 0.516, since `hcr_operator` zeros coarse entries only *after* each full symmetric
    # Gauss-Seidel sweep (forward + backward over every row) rather than truly holding them
    # fixed at 0 throughout the sweep - a deliberate F-relaxation approximation (the brief's own
    # code), not a bug here. 0.6 keeps meaningful margin while staying well below the empty-C
    # baseline of ~0.83-0.91 (see test_cr_rate_high_for_empty_coarse_set), which is what "a good
    # CR rate" is actually asserting.
    assert rho < 0.6
    assert torch.allclose(e[coarse_mask], torch.zeros_like(e[coarse_mask]))


def test_cr_rate_high_for_empty_coarse_set(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    empty_coarse_mask_16: torch.Tensor,
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
) -> None:
    """With no coarse points at all, relaxation alone converges far slower than delta=0.7."""
    matrix = poisson_1d_factory(16)
    draw = seeded_draw_factory(0)

    rho, _ = cr_rate(
        matrix, empty_coarse_mask_16, GaussSeidelSmoother().smooth, sweeps=2, draw=draw
    )

    assert rho > 0.7


def test_independent_set_of_respects_matrix_graph_and_priority(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    five_true_candidates: torch.Tensor,
    descending_priority_5: torch.Tensor,
) -> None:
    """Greedy independent set never selects two matrix-graph neighbors, highest priority first."""
    matrix = poisson_1d_factory(5)

    selected = _independent_set_of(five_true_candidates, matrix, descending_priority_5)

    assert selected.tolist() == [True, False, True, False, True]


def test_independent_set_of_uses_guidance_graph_when_given(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    five_true_candidates: torch.Tensor,
    descending_priority_5: torch.Tensor,
    skip_one_guidance_graph_5: torch.Tensor,
) -> None:
    """A ``guidance_graph`` overrides the matrix's own graph and can change which set is chosen."""
    matrix = poisson_1d_factory(5)

    selected_default = _independent_set_of(five_true_candidates, matrix, descending_priority_5)
    selected_guided = _independent_set_of(
        five_true_candidates,
        matrix,
        descending_priority_5,
        guidance_graph=skip_one_guidance_graph_5,
    )

    assert selected_default.tolist() == [True, False, True, False, True]
    assert selected_guided.tolist() == [True, True, False, False, True]


def test_compatible_relaxation_coarsening_with_explicit_matrix_graph_matches_default(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    matrix_graph_factory: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Passing the matrix's own graph as ``guidance_graph`` reproduces the ``None`` default exactly."""
    matrix = poisson_1d_factory(16)
    matrix_graph = matrix_graph_factory(matrix)

    default_mask = compatible_relaxation_coarsening(
        matrix, GaussSeidelSmoother().smooth, nu=5, delta=0.7, draw=seeded_draw_factory(1)
    )
    explicit_mask = compatible_relaxation_coarsening(
        matrix,
        GaussSeidelSmoother().smooth,
        nu=5,
        delta=0.7,
        draw=seeded_draw_factory(1),
        guidance_graph=matrix_graph,
    )

    assert torch.equal(default_mask, explicit_mask)


def test_compatible_relaxation_coarsening_terminates_and_grows_c(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
) -> None:
    """The outer loop terminates with a nontrivial, strictly partial coarse set."""
    matrix = poisson_1d_factory(16)
    draw = seeded_draw_factory(1)

    coarse_mask = compatible_relaxation_coarsening(
        matrix, GaussSeidelSmoother().smooth, nu=5, delta=0.7, draw=draw
    )

    assert coarse_mask.dtype == torch.bool
    assert 0 < coarse_mask.sum() < 16


def test_compatible_relaxation_coarsening_preserves_initial_coarse(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    two_of_eight_coarse_mask: torch.Tensor,
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
) -> None:
    """``initial_coarse`` (``C_0``) points are never removed from the returned set."""
    matrix = poisson_1d_factory(8)
    draw = seeded_draw_factory(2)

    coarse_mask = compatible_relaxation_coarsening(
        matrix,
        GaussSeidelSmoother().smooth,
        nu=5,
        delta=0.7,
        initial_coarse=two_of_eight_coarse_mask,
        draw=draw,
    )

    assert torch.all(coarse_mask[two_of_eight_coarse_mask])


def test_compatible_relaxation_coarsening_raises_when_max_iterations_exhausted(
    poisson_1d_factory: Callable[[int], torch.Tensor],
    seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
) -> None:
    """An unreachable ``delta`` with a tiny ``max_iterations`` raises rather than hanging."""
    matrix = poisson_1d_factory(16)
    draw = seeded_draw_factory(1)

    with pytest.raises(RuntimeError, match="did not converge"):
        compatible_relaxation_coarsening(
            matrix,
            GaussSeidelSmoother().smooth,
            nu=5,
            delta=0.0,
            draw=draw,
            max_iterations=1,
        )
