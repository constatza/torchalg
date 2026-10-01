"""Tests for ``torchalg.sparse.preconditioners.amg._compatible_relaxation``.

Dense-as-oracle, this project's standing rule: every function is checked
against its dense sibling
(``preconditioners.implementations.amg._compatible_relaxation``) given the
same matrix (one dense, one its sparse CSR twin), the same coarse
mask/priority/seeded draw, and the sparse ``GaussSeidelSmoother`` in place of
the dense one (already parity-tested elsewhere,
``tests/solver/preconditioners/implementations/amg/test_smoothers.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg._compatible_relaxation import (
    _independent_set_of as dense_independent_set_of,
)
from torchalg.preconditioners.implementations.amg._compatible_relaxation import (
    compatible_relaxation_coarsening as dense_compatible_relaxation_coarsening,
)
from torchalg.preconditioners.implementations.amg._compatible_relaxation import (
    cr_rate as dense_cr_rate,
)
from torchalg.preconditioners.implementations.amg._compatible_relaxation import (
    hcr_operator as dense_hcr_operator,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother as DenseGaussSeidelSmoother,
)
from torchalg.sparse.preconditioners.amg._compatible_relaxation import (
    _independent_set_of,
    compatible_relaxation_coarsening,
    cr_rate,
    hcr_operator,
)
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother


@pytest.fixture
def two_of_eight_coarse_mask() -> torch.Tensor:
    return torch.tensor([False, False, True, False, False, True, False, False])


@pytest.fixture
def ones_vector_8(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(8, dtype=torch_dtype)


@pytest.fixture
def seeded_draw_factory(
    torch_dtype: torch.dtype,
) -> Callable[[int], Callable[[int], torch.Tensor]]:
    def _factory(seed: int) -> Callable[[int], torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)

        def _draw(n: int) -> torch.Tensor:
            return torch.rand(n, generator=generator, dtype=torch_dtype)

        return _draw

    return _factory


@pytest.fixture
def five_true_candidates() -> torch.Tensor:
    return torch.ones(5, dtype=torch.bool)


@pytest.fixture
def descending_priority_5() -> torch.Tensor:
    return torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])


@pytest.fixture
def skip_one_guidance_graph_5() -> torch.Tensor:
    """Boolean graph connecting 0<->2 and 1<->3 - disjoint from the matrix's own adjacent-index graph."""
    graph = torch.zeros(5, 5, dtype=torch.bool)
    graph[0, 2] = graph[2, 0] = True
    graph[1, 3] = graph[3, 1] = True
    return graph


class TestHcrOperator:
    def test_matches_dense(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        two_of_eight_coarse_mask: torch.Tensor,
        ones_vector_8: torch.Tensor,
    ) -> None:
        dense = poisson_1d_factory(8)
        sparse = dense.to_sparse_csr()

        expected = dense_hcr_operator(
            dense,
            two_of_eight_coarse_mask,
            DenseGaussSeidelSmoother().smooth,
            sweeps=3,
            start=ones_vector_8,
        )
        actual = hcr_operator(
            sparse,
            two_of_eight_coarse_mask,
            GaussSeidelSmoother().smooth,
            sweeps=3,
            start=ones_vector_8,
        )
        torch.testing.assert_close(actual, expected)

    def test_zeros_coarse_entries(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        two_of_eight_coarse_mask: torch.Tensor,
        ones_vector_8: torch.Tensor,
        torch_dtype: torch.dtype,
    ) -> None:
        sparse = poisson_1d_factory(8).to_sparse_csr()
        result = hcr_operator(
            sparse,
            two_of_eight_coarse_mask,
            GaussSeidelSmoother().smooth,
            sweeps=3,
            start=ones_vector_8,
        )
        assert torch.allclose(result[two_of_eight_coarse_mask], torch.zeros(2, dtype=torch_dtype))


class TestCrRate:
    def test_matches_dense(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        dense = poisson_1d_factory(16)
        sparse = dense.to_sparse_csr()
        coarse_mask = torch.zeros(16, dtype=torch.bool)
        coarse_mask[::2] = True

        rho_expected, e_expected = dense_cr_rate(
            dense,
            coarse_mask,
            DenseGaussSeidelSmoother().smooth,
            sweeps=5,
            draw=seeded_draw_factory(0),
        )
        rho_actual, e_actual = cr_rate(
            sparse, coarse_mask, GaussSeidelSmoother().smooth, sweeps=5, draw=seeded_draw_factory(0)
        )
        assert rho_actual == pytest.approx(rho_expected)
        torch.testing.assert_close(e_actual, e_expected)

    def test_bounded_in_zero_one(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse = poisson_1d_factory(16).to_sparse_csr()
        coarse_mask = torch.zeros(16, dtype=torch.bool)
        coarse_mask[::2] = True

        rho, _ = cr_rate(
            sparse, coarse_mask, GaussSeidelSmoother().smooth, sweeps=5, draw=seeded_draw_factory(0)
        )
        assert 0.0 <= rho < 1.0


class TestIndependentSetOf:
    def test_matches_dense_default_adjacency(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        five_true_candidates: torch.Tensor,
        descending_priority_5: torch.Tensor,
    ) -> None:
        dense = poisson_1d_factory(5)
        sparse = dense.to_sparse_csr()

        expected = dense_independent_set_of(five_true_candidates, dense, descending_priority_5)
        actual = _independent_set_of(five_true_candidates, sparse, descending_priority_5)
        assert actual.tolist() == expected.tolist()

    def test_matches_dense_with_guidance_graph(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        five_true_candidates: torch.Tensor,
        descending_priority_5: torch.Tensor,
        skip_one_guidance_graph_5: torch.Tensor,
    ) -> None:
        dense = poisson_1d_factory(5)
        sparse = dense.to_sparse_csr()
        sparse_guidance = skip_one_guidance_graph_5.to_sparse_csr()

        expected = dense_independent_set_of(
            five_true_candidates,
            dense,
            descending_priority_5,
            guidance_graph=skip_one_guidance_graph_5,
        )
        actual = _independent_set_of(
            five_true_candidates, sparse, descending_priority_5, guidance_graph=sparse_guidance
        )
        assert actual.tolist() == expected.tolist()


class TestCompatibleRelaxationCoarsening:
    def test_matches_dense(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        dense = poisson_1d_factory(16)
        sparse = dense.to_sparse_csr()

        expected = dense_compatible_relaxation_coarsening(
            dense, DenseGaussSeidelSmoother().smooth, nu=5, delta=0.7, draw=seeded_draw_factory(1)
        )
        actual = compatible_relaxation_coarsening(
            sparse, GaussSeidelSmoother().smooth, nu=5, delta=0.7, draw=seeded_draw_factory(1)
        )
        assert torch.equal(actual, expected)

    def test_terminates_and_grows_c(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse = poisson_1d_factory(16).to_sparse_csr()
        coarse_mask = compatible_relaxation_coarsening(
            sparse, GaussSeidelSmoother().smooth, nu=5, delta=0.7, draw=seeded_draw_factory(1)
        )
        assert coarse_mask.dtype == torch.bool
        assert 0 < coarse_mask.sum() < 16

    def test_preserves_initial_coarse(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        two_of_eight_coarse_mask: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse = poisson_1d_factory(8).to_sparse_csr()
        coarse_mask = compatible_relaxation_coarsening(
            sparse,
            GaussSeidelSmoother().smooth,
            nu=5,
            delta=0.7,
            initial_coarse=two_of_eight_coarse_mask,
            draw=seeded_draw_factory(2),
        )
        assert torch.all(coarse_mask[two_of_eight_coarse_mask])

    def test_raises_when_max_iterations_exhausted(
        self,
        poisson_1d_factory: Callable[[int], torch.Tensor],
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse = poisson_1d_factory(16).to_sparse_csr()
        with pytest.raises(RuntimeError, match="did not converge"):
            compatible_relaxation_coarsening(
                sparse,
                GaussSeidelSmoother().smooth,
                nu=5,
                delta=0.0,
                draw=seeded_draw_factory(1),
                max_iterations=1,
            )
