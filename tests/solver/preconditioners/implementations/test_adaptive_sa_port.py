"""Adaptive smoothed aggregation, checked kernel-by-kernel against PyAMG's implementation.

Every kernel of the port (relaxation, spectral-radius estimate, candidate
fitting, node strength, prolongator smoothing) and finally the complete
``adaptive_sa_solver`` are compared with PyAMG on identical inputs. Randomness
is replayed from NumPy's stream (``replay_draw``), so the runs are fully
deterministic and *identical* draws are consumed in identical order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pyamg
import pytest
import scipy.sparse as sp
import torch
from pyamg.aggregation.aggregate import standard_aggregation as pyamg_aggregation
from pyamg.aggregation.smooth import jacobi_prolongation_smoother as pyamg_jacobi_smoother
from pyamg.aggregation.tentative import fit_candidates as pyamg_fit_candidates
from pyamg.relaxation.relaxation import gauss_seidel as pyamg_gauss_seidel
from pyamg.relaxation.relaxation import gauss_seidel_indexed as pyamg_gauss_seidel_indexed
from pyamg.strength import symmetric_strength_of_connection as pyamg_strength
from pyamg.util.linalg import approximate_spectral_radius as pyamg_spectral_radius

from tests.support.pyamg_reference import (
    aggregation_operator,
    pinned_specs,
    replay_draw,
    to_csr,
)
from torchalg.preconditioners.implementations.amg._node_strength import node_strength
from torchalg.preconditioners.implementations.amg._presets import PRESET_CYCLE
from torchalg.preconditioners.implementations.amg._prolongation import (
    jacobi_prolongation,
    make_bridge,
)
from torchalg.preconditioners.implementations.amg._relaxation import symmetric_gauss_seidel
from torchalg.preconditioners.implementations.amg._spectral import approximate_spectral_radius
from torchalg.preconditioners.implementations.amg._tentative import fit_candidates
from torchalg.preconditioners.implementations.amg.adaptive import adaptive_sa_hierarchy

ATOL = 1e-11


@pytest.fixture
def aniso_matrix(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """12x12 anisotropic (eps=0.05) 2D Poisson matrix, 144 unknowns."""
    return anisotropic_2d_factory(12, 0.05)


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


@pytest.fixture
def random_pair(aniso_matrix: torch.Tensor, torch_dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Seeded random ``(x, b)`` pair for relaxation tests."""
    generator = torch.Generator().manual_seed(21)
    n = aniso_matrix.shape[0]
    return (
        torch.randn(n, generator=generator, dtype=torch_dtype),
        torch.randn(n, generator=generator, dtype=torch_dtype),
    )


@pytest.fixture
def matrix_with_dead_row(aniso_matrix: torch.Tensor) -> torch.Tensor:
    """Matrix whose row/column 5 is entirely zero (zero diagonal, as after dropped candidates)."""
    matrix = aniso_matrix.clone()
    matrix[5, :] = 0.0
    matrix[:, 5] = 0.0
    return matrix


@pytest.fixture
def dependent_vectors(aggregate_map: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """(13, 3) block whose third column duplicates the first: triggers PyAMG's tolerance drop."""
    generator = torch.Generator().manual_seed(22)
    base = torch.randn(aggregate_map.shape[0], 2, generator=generator, dtype=torch_dtype)
    return torch.cat([base, base[:, :1]], dim=1)


@pytest.fixture
def block_vectors(aggregate_map: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """(13*3, 2) block: 3 dofs per node, 2 candidates."""
    generator = torch.Generator().manual_seed(23)
    return torch.randn(aggregate_map.shape[0] * 3, 2, generator=generator, dtype=torch_dtype)


class TestSymmetricGaussSeidel:
    @pytest.mark.parametrize("iterations", [1, 3])
    def test_matches_pyamg(
        self, aniso_matrix: torch.Tensor, random_pair: tuple[torch.Tensor, ...], iterations: int
    ) -> None:
        x, b = random_pair
        expected = x.numpy().copy()
        pyamg_gauss_seidel(
            to_csr(aniso_matrix), expected, b.numpy(), iterations=iterations, sweep="symmetric"
        )
        ours = symmetric_gauss_seidel(aniso_matrix, x, b, iterations)
        np.testing.assert_allclose(ours.numpy(), expected, atol=ATOL)

    def test_zero_diagonal_rows_are_skipped_like_pyamg(
        self, matrix_with_dead_row: torch.Tensor, random_pair: tuple[torch.Tensor, ...]
    ) -> None:
        x, b = random_pair
        expected = x.numpy().copy()
        pyamg_gauss_seidel(to_csr(matrix_with_dead_row), expected, b.numpy(), sweep="symmetric")
        ours = symmetric_gauss_seidel(matrix_with_dead_row, x, b, 1)
        np.testing.assert_allclose(ours.numpy(), expected, atol=ATOL)
        assert ours[5] == x[5]

    def test_indexed_variant_matches_pyamg(
        self, aniso_matrix: torch.Tensor, random_pair: tuple[torch.Tensor, ...]
    ) -> None:
        x, b = random_pair
        rows = torch.tensor([2, 3, 7, 40, 41, 100])
        expected = x.numpy().copy()
        pyamg_gauss_seidel_indexed(
            to_csr(aniso_matrix), expected, b.numpy(), rows.numpy(), iterations=2, sweep="symmetric"
        )
        ours = symmetric_gauss_seidel(aniso_matrix, x, b, 2, rows=rows)
        np.testing.assert_allclose(ours.numpy(), expected, atol=ATOL)


class TestApproximateSpectralRadius:
    def test_matches_pyamg_with_restarts(self, aniso_matrix: torch.Tensor) -> None:
        d_inv_a = aniso_matrix / torch.diagonal(aniso_matrix).unsqueeze(1)
        start = replay_draw(3)(aniso_matrix.shape[0])
        expected = pyamg_spectral_radius(to_csr(d_inv_a), initial_guess=start.numpy().copy())
        ours = approximate_spectral_radius(d_inv_a, lambda n: start, initial_guess=start)
        assert ours == pytest.approx(expected, abs=1e-10)


class TestFitCandidates:
    @pytest.mark.parametrize("fixture", ["dependent_vectors", "block_vectors"])
    def test_matches_pyamg(
        self, aggregate_map: torch.Tensor, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        vectors = request.getfixturevalue(fixture)
        q_ref, r_ref = pyamg_fit_candidates(aggregation_operator(aggregate_map), vectors.numpy())
        q, r = fit_candidates(aggregate_map, vectors)
        np.testing.assert_allclose(q.numpy(), q_ref.toarray(), atol=ATOL)
        np.testing.assert_allclose(r.numpy(), r_ref, atol=ATOL)

    def test_isolated_node_row_stays_zero(
        self, aggregate_map: torch.Tensor, dependent_vectors: torch.Tensor
    ) -> None:
        q, _ = fit_candidates(aggregate_map, dependent_vectors)
        assert torch.count_nonzero(q[-1]) == 0


class TestNodeStrength:
    @pytest.mark.parametrize("theta", [0.0, 0.25])
    def test_scalar_matches_pyamg(self, aniso_matrix: torch.Tensor, theta: float) -> None:
        reference = pyamg_strength(to_csr(aniso_matrix), theta).toarray() != 0
        np.fill_diagonal(reference, False)
        assert np.array_equal(node_strength(aniso_matrix, 1, theta).numpy(), reference)

    @pytest.mark.parametrize("theta", [0.0, 0.3])
    def test_block_matches_pyamg(self, torch_dtype: torch.dtype, theta: float) -> None:
        generator = torch.Generator().manual_seed(24)
        raw = torch.randn(30, 30, generator=generator, dtype=torch_dtype)
        matrix = raw @ raw.T + 30 * torch.eye(30, dtype=torch_dtype)
        matrix[torch.abs(matrix) < 4.0] = 0.0
        matrix.fill_diagonal_(30.0)
        reference = pyamg_strength(sp.bsr_array(matrix.numpy(), blocksize=(3, 3)), theta).toarray()
        reference = reference != 0
        np.fill_diagonal(reference, False)
        assert np.array_equal(node_strength(matrix, 3, theta).numpy(), reference)


class TestProlongationSmoothing:
    def test_jacobi_smoother_matches_pyamg_including_estimated_rho(
        self, aniso_matrix: torch.Tensor, aggregate_map: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        n = aniso_matrix.shape[0]
        aggregate = torch.arange(n) // 4
        tentative, coarse = fit_candidates(aggregate, torch.ones(n, 1, dtype=torch_dtype))
        np.random.seed(5)
        expected = pyamg_jacobi_smoother(
            to_csr(aniso_matrix),
            sp.csr_matrix(tentative.numpy()).tobsr(blocksize=(1, 1)),
            to_csr(aniso_matrix),
            coarse.numpy(),
            omega=4.0 / 3.0,
        ).toarray()
        ours = jacobi_prolongation(
            aniso_matrix,
            tentative,
            4.0 / 3.0,
            lambda m: approximate_spectral_radius(m, replay_draw(5)),
        )
        np.testing.assert_allclose(ours.numpy(), expected, atol=ATOL)

    def test_bridge_appends_one_zero_dof_per_node(self, torch_dtype: torch.dtype) -> None:
        tentative = torch.arange(1.0, 13.0, dtype=torch_dtype).reshape(6, 2)  # 3 nodes x 2 dofs
        bridged = make_bridge(tentative, 2)
        assert bridged.shape == (9, 2)
        assert torch.equal(bridged[[2, 5, 8]], torch.zeros(3, 2, dtype=torch_dtype))
        assert torch.equal(bridged[[0, 1, 3, 4, 6, 7]], tentative)


CASES = [
    pytest.param(
        "poisson_matrix",
        {"num_candidates": 1, "max_levels": 3, "max_coarse": 5},
        0.0,
        id="poisson-1cand",
    ),
    pytest.param(
        "aniso_matrix",
        {"num_candidates": 3, "max_levels": 3, "max_coarse": 5},
        0.0,
        id="aniso-3cand-theta0-pinned-aggregation",
    ),
    pytest.param(
        "aniso_matrix",
        {"num_candidates": 3, "max_levels": 4, "max_coarse": 5},
        0.25,
        id="aniso-3cand-theta.25",
    ),
    pytest.param(
        "aniso_matrix",
        {"num_candidates": 2, "max_levels": 3, "max_coarse": 5, "candidate_iters": 3},
        0.25,
        id="aniso-2cand-3iters",
    ),
]


class TestAdaptiveSolverAgainstPyAMG:
    @pytest.mark.parametrize("fixture,options,theta", CASES)
    def test_hierarchy_candidates_and_cycle_match_pyamg(
        self,
        request: pytest.FixtureRequest,
        fixture: str,
        options: dict[str, Any],
        theta: float,
        torch_dtype: torch.dtype,
    ) -> None:
        matrix = request.getfixturevalue(fixture)
        result = adaptive_sa_hierarchy(matrix, theta=theta, draw=replay_draw(11), **options)
        pin = request.node.callspec.id.endswith("pinned-aggregation")
        # PyAMG's standard aggregation depends on the stored column order of its sparse
        # coarse matrices (see adaptive.py), so where that bites the aggregation is pinned.
        specs = (
            pinned_specs(result.aggregates, result.strengths)
            if pin
            else {"strength": ("symmetric", {"theta": theta})}
        )
        np.random.seed(11)
        reference, _ = pyamg.aggregation.adaptive_sa_solver(
            to_csr(matrix), **specs, **({**options, "max_coarse": 0} if pin else options)
        )

        assert len(result.matrices) == len(reference.levels)
        for level, (ours, ref) in enumerate(zip(result.matrices, reference.levels, strict=True)):
            np.testing.assert_allclose(
                ours.numpy(), ref.A.toarray(), atol=1e-9, err_msg=f"A[{level}]"
            )
        for level, ours in enumerate(result.prolongations):
            np.testing.assert_allclose(
                ours.numpy(), reference.levels[level].P.toarray(), atol=1e-9, err_msg=f"P[{level}]"
            )
        np.testing.assert_allclose(result.candidates.numpy(), reference.levels[0].B, atol=1e-9)

        residual = torch.randn(
            matrix.shape[0], generator=torch.Generator().manual_seed(31), dtype=torch_dtype
        )
        oracle = reference.aspreconditioner(cycle="V").matvec(residual.numpy())
        ours = PRESET_CYCLE.apply(result.hierarchy, residual)
        np.testing.assert_allclose(ours.numpy(), oracle, atol=1e-8)

    def test_aggregation_matches_pyamg_on_sorted_graph(self, aniso_matrix: torch.Tensor) -> None:
        """Level-0 strength graph and a coarse-like graph: ours = PyAMG with sorted indices."""
        result = adaptive_sa_hierarchy(
            aniso_matrix, theta=0.0, draw=replay_draw(11), max_levels=3, max_coarse=5
        )
        for strength, aggregate in zip(result.strengths, result.aggregates, strict=True):
            graph = sp.csr_matrix(strength.numpy().astype(float) + np.eye(len(strength)))
            graph.sort_indices()
            expected = pyamg_aggregation(graph)[0].toarray()
            expected_index = np.where(expected.any(1), expected.argmax(1), -1)
            assert np.array_equal(aggregate.numpy(), expected_index)

    def test_user_supplied_candidates_match_pyamg(
        self, aniso_matrix: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        initial = torch.ones(aniso_matrix.shape[0], 1, dtype=torch_dtype)
        np.random.seed(12)
        reference, _ = pyamg.aggregation.adaptive_sa_solver(
            to_csr(aniso_matrix),
            initial_candidates=initial.numpy(),
            num_candidates=3,
            max_levels=3,
            max_coarse=5,
            strength=("symmetric", {"theta": 0.25}),
        )
        result = adaptive_sa_hierarchy(
            aniso_matrix,
            initial_candidates=initial,
            num_candidates=3,
            max_levels=3,
            max_coarse=5,
            theta=0.25,
            draw=replay_draw(12),
        )
        np.testing.assert_allclose(result.candidates.numpy(), reference.levels[0].B, atol=1e-9)
        for ours, ref in zip(result.prolongations, reference.levels, strict=False):
            np.testing.assert_allclose(ours.numpy(), ref.P.toarray(), atol=1e-9)
