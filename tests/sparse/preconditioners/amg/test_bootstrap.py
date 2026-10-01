"""Tests for ``torchalg.sparse.preconditioners.amg.bootstrap.BAMGCoarsening``.

Dense-as-oracle, this project's standing rule: the sparse sibling is
checked against the dense ``BAMGCoarsening`` given the same matrix (one
dense, one its sparse CSR twin), the same seeded test vectors, the same
seeded CR draw, and the sparse ``GaussSeidelSmoother`` in place of the dense
one (already parity-tested elsewhere) - full "two classes, same result"
parity at the whole-algorithm level, matching
``tests/sparse/preconditioners/amg/test_coarsening.py``'s existing pattern
for ``AggregationCoarsening``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg import pcg
from torchalg.preconditioners.implementations.amg.bootstrap import (
    BAMGCoarsening as DenseBAMGCoarsening,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother as DenseGaussSeidelSmoother,
)
from torchalg.sparse.preconditioners.amg.bootstrap import BAMGCoarsening
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother


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


class TestBAMGCoarseningBuildTransfer:
    """``caliber`` is deliberately kept below the test-vector count ``k`` in every
    case below (``caliber=4 < k=6``) - see the sparse ``BAMGCoarsening`` module
    docstring's "Known numerical sensitivity" note: once ``caliber >= k``, the
    LS fit becomes exact (zero residual) after the first ``k`` points, so
    ``select_interpolatory_set``'s choice of any further point is decided by
    pure floating-point noise (residuals at the ~1e-30 level, confirmed by
    direct measurement) - dense and sparse (or even dense-vs-dense on
    different hardware) can legitimately pick a different, equally-valid
    point there, which is not a correctness bug but does break bit-level
    parity. Confirmed empirically: 8 random seeds with ``caliber=4 >= k=3``
    all diverged (by up to ~1.5 in ``A_coarse`` after the Galerkin product
    amplifies several per-row tie-flips); the same 8 seeds with
    ``caliber=3 < k=6`` all matched to ~1e-14."""

    def test_matches_dense_on_poisson_1d(
        self,
        poisson_1d_large_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        torch.manual_seed(0)
        n = poisson_1d_large_dense.shape[0]
        test_vectors = torch.randn(n, 6, dtype=poisson_1d_large_dense.dtype)
        sparse_matrix = poisson_1d_large_dense.to_sparse_csr()

        dense_coarsening = DenseBAMGCoarsening(
            test_vectors.clone(),
            DenseGaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(1),
        )
        sparse_coarsening = BAMGCoarsening(
            test_vectors.clone(),
            GaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(1),
        )

        dense_coarse, dense_transfer = dense_coarsening.build_transfer(poisson_1d_large_dense)
        sparse_coarse, sparse_transfer = sparse_coarsening.build_transfer(sparse_matrix)

        torch.testing.assert_close(sparse_coarse.to_dense(), dense_coarse)
        torch.testing.assert_close(
            sparse_coarsening.last_prolongation.to_dense(), dense_coarsening.last_prolongation
        )

        coarse_vector = torch.randn(dense_coarse.shape[0], dtype=poisson_1d_large_dense.dtype)
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )
        fine_vector = torch.randn(n, dtype=poisson_1d_large_dense.dtype)
        torch.testing.assert_close(
            sparse_transfer.restrict(fine_vector), dense_transfer.restrict(fine_vector)
        )

    def test_default_parameter_level_matches_dense_values(
        self,
        poisson_1d_large_dense: torch.Tensor,
        bamg_default_test_vectors: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        """Default benchmark-level BAMG setup preserves dense coarse and P values."""
        dense_coarsening = DenseBAMGCoarsening(
            bamg_default_test_vectors.clone(),
            DenseGaussSeidelSmoother().smooth,
            draw=seeded_draw_factory(0),
        )
        sparse_coarsening = BAMGCoarsening(
            bamg_default_test_vectors.clone(),
            GaussSeidelSmoother().smooth,
            draw=seeded_draw_factory(0),
        )

        dense_coarse, _ = dense_coarsening.build_transfer(poisson_1d_large_dense)
        sparse_coarse, _ = sparse_coarsening.build_transfer(poisson_1d_large_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_coarse.to_dense(), dense_coarse)
        torch.testing.assert_close(
            sparse_coarsening.last_prolongation.to_dense(), dense_coarsening.last_prolongation
        )

    def test_matches_dense_across_two_levels(
        self,
        poisson_1d_large_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        """Second ``build_transfer`` call (composite T_l != I) still matches dense."""
        torch.manual_seed(2)
        n = poisson_1d_large_dense.shape[0]
        test_vectors = torch.randn(n, 6, dtype=poisson_1d_large_dense.dtype)
        sparse_matrix = poisson_1d_large_dense.to_sparse_csr()

        dense_coarsening = DenseBAMGCoarsening(
            test_vectors.clone(),
            DenseGaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(3),
        )
        sparse_coarsening = BAMGCoarsening(
            test_vectors.clone(),
            GaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(3),
        )

        dense_level1, _ = dense_coarsening.build_transfer(poisson_1d_large_dense)
        sparse_level1, _ = sparse_coarsening.build_transfer(sparse_matrix)
        torch.testing.assert_close(sparse_level1.to_dense(), dense_level1)

        dense_level2, _ = dense_coarsening.build_transfer(dense_level1)
        sparse_level2, _ = sparse_coarsening.build_transfer(sparse_level1)
        torch.testing.assert_close(sparse_level2.to_dense(), dense_level2)

    def test_matches_dense_at_larger_scale(
        self,
        poisson_1d_xlarge_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        """A 300-dof system to exercise the vectorized padded-candidate gathering.

        ``poisson_1d_large_dense`` (64 dofs) is small enough that every
        fine row's coarse-neighbor candidate set tends to land at the same
        size; at 300 dofs, fine rows plausibly see several distinct
        candidate-set sizes (including the all-coarse-points fallback for
        rows whose LS-ring neighborhood has no coarse neighbor yet) - this
        is exactly the padding/masking edge case the vectorized gather in
        ``BAMGCoarsening._prolongation`` could get subtly wrong without
        tripping the smaller fixtures above.
        """
        torch.manual_seed(6)
        n = poisson_1d_xlarge_dense.shape[0]
        test_vectors = torch.randn(n, 6, dtype=poisson_1d_xlarge_dense.dtype)
        sparse_matrix = poisson_1d_xlarge_dense.to_sparse_csr()

        dense_coarsening = DenseBAMGCoarsening(
            test_vectors.clone(),
            DenseGaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(7),
        )
        sparse_coarsening = BAMGCoarsening(
            test_vectors.clone(),
            GaussSeidelSmoother().smooth,
            nu=3,
            delta=0.7,
            caliber=4,
            draw=seeded_draw_factory(7),
        )

        dense_coarse, dense_transfer = dense_coarsening.build_transfer(poisson_1d_xlarge_dense)
        sparse_coarse, sparse_transfer = sparse_coarsening.build_transfer(sparse_matrix)

        torch.testing.assert_close(sparse_coarse.to_dense(), dense_coarse)
        torch.testing.assert_close(
            sparse_coarsening.last_prolongation.to_dense(), dense_coarsening.last_prolongation
        )

        coarse_vector = torch.randn(dense_coarse.shape[0], dtype=poisson_1d_xlarge_dense.dtype)
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )
        fine_vector = torch.randn(n, dtype=poisson_1d_xlarge_dense.dtype)
        torch.testing.assert_close(
            sparse_transfer.restrict(fine_vector), dense_transfer.restrict(fine_vector)
        )

    def test_test_vectors_for_and_set_test_vectors(self, poisson_1d_csr: torch.Tensor) -> None:
        torch.manual_seed(4)
        n = poisson_1d_csr.shape[0]
        test_vectors = torch.randn(n, 2, dtype=poisson_1d_csr.values().dtype)
        coarsening = BAMGCoarsening(
            test_vectors, GaussSeidelSmoother().smooth, draw=lambda m: torch.rand(m)
        )
        assert torch.equal(coarsening.test_vectors_for(poisson_1d_csr), test_vectors)

        new_vectors = torch.randn(n, 2, dtype=poisson_1d_csr.values().dtype)
        coarsening.set_test_vectors(poisson_1d_csr, new_vectors)
        assert torch.equal(coarsening.test_vectors_for(poisson_1d_csr), new_vectors)

    def test_last_prolongation_raises_before_build_transfer(
        self, poisson_1d_csr: torch.Tensor
    ) -> None:
        n = poisson_1d_csr.shape[0]
        coarsening = BAMGCoarsening(
            torch.randn(n, 2, dtype=poisson_1d_csr.values().dtype),
            GaussSeidelSmoother().smooth,
            draw=lambda m: torch.rand(m),
        )
        with pytest.raises(RuntimeError, match="build_transfer has not been called"):
            _ = coarsening.last_prolongation


class TestBootstrapAMGPreconditioner:
    """``BootstrapAMGPreconditioner``, dense-as-oracle end to end.

    ``caliber`` stays below the test-vector count ``k`` throughout - see
    ``BAMGCoarsening``'s "Known numerical sensitivity" module docstring note
    (``caliber >= k`` makes the greedy LS candidate choice a floating-point
    near-tie, breaking dense-vs-sparse bit-parity without being a
    correctness bug).
    """

    def test_matches_dense_hierarchy_and_apply(
        self,
        poisson_1d_large_dense: torch.Tensor,
    ) -> None:
        from torchalg.preconditioners.implementations.amg.bootstrap import (
            BootstrapAMGPreconditioner as DenseBootstrapAMGPreconditioner,
        )
        from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner

        sparse_matrix = poisson_1d_large_dense.to_sparse_csr()

        dense_precond = DenseBootstrapAMGPreconditioner(
            poisson_1d_large_dense,
            k_r=6,
            eta=2,
            n_bootstrap_cycles=1,
            nu=3,
            delta=0.7,
            caliber=3,
            max_levels=3,
            max_coarse=4,
            seed=5,
        )
        sparse_precond = BootstrapAMGPreconditioner(
            sparse_matrix,
            k_r=6,
            eta=2,
            n_bootstrap_cycles=1,
            nu=3,
            delta=0.7,
            caliber=3,
            max_levels=3,
            max_coarse=4,
            seed=5,
        )

        assert len(sparse_precond.result.matrices) == len(dense_precond.result.matrices)
        for sparse_level, dense_level in zip(
            sparse_precond.result.matrices, dense_precond.result.matrices, strict=True
        ):
            torch.testing.assert_close(sparse_level.to_dense(), dense_level)
        for sparse_p, dense_p in zip(
            sparse_precond.result.prolongations, dense_precond.result.prolongations, strict=True
        ):
            torch.testing.assert_close(sparse_p.to_dense(), dense_p)

        torch.manual_seed(0)
        residual = torch.randn(poisson_1d_large_dense.shape[0], dtype=poisson_1d_large_dense.dtype)
        torch.testing.assert_close(
            sparse_precond.apply(residual), dense_precond.apply(residual), atol=1e-8, rtol=1e-6
        )

    def test_pcg_matches_dense(
        self,
        poisson_1d_large_dense: torch.Tensor,
        poisson_1d_large_rhs: torch.Tensor,
    ) -> None:
        """The full Bootstrap-AMG-preconditioned PCG solve matches its dense sibling."""
        from torchalg.preconditioners.implementations.amg.bootstrap import (
            BootstrapAMGPreconditioner as DenseBootstrapAMGPreconditioner,
        )
        from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner

        dense_preconditioner = DenseBootstrapAMGPreconditioner(
            poisson_1d_large_dense,
            k_r=6,
            eta=2,
            n_bootstrap_cycles=1,
            nu=3,
            delta=0.7,
            caliber=3,
            max_levels=3,
            max_coarse=4,
            seed=5,
        )
        sparse_preconditioner = BootstrapAMGPreconditioner(
            poisson_1d_large_dense.to_sparse_csr(),
            k_r=6,
            eta=2,
            n_bootstrap_cycles=1,
            nu=3,
            delta=0.7,
            caliber=3,
            max_levels=3,
            max_coarse=4,
            seed=5,
        )

        dense_solution, dense_info = pcg(
            poisson_1d_large_dense,
            poisson_1d_large_rhs,
            preconditioner=dense_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )
        sparse_solution, sparse_info = pcg(
            poisson_1d_large_dense.to_sparse_csr(),
            poisson_1d_large_rhs,
            preconditioner=sparse_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )

        assert dense_info.converged and sparse_info.converged
        torch.testing.assert_close(sparse_solution, dense_solution, rtol=1e-5, atol=1e-6)

    def test_raises_on_single_level(self, poisson_1d_csr: torch.Tensor) -> None:
        from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner

        with pytest.raises(ValueError, match="single level"):
            BootstrapAMGPreconditioner(poisson_1d_csr, max_coarse=poisson_1d_csr.shape[0])
