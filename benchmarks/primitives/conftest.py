"""Seeded input fixtures for the primitive benchmarks.

Every tensor is generated on CPU from an explicit seed and then moved to the benchmark
device, so all devices see identical data. Fixtures that need a dense ``n × n`` operand
skip above :data:`~benchmarks.primitives._support.DENSE_N_CAP`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from benchmarks.cases import BENCHMARK_SEED, MatrixCase
from benchmarks.primitives._support import DENSE_N_CAP, STRENGTH_THETA, TEST_VECTOR_COUNT
from torchalg.preconditioners.implementations.amg._aggregation import strength_of_connection
from torchalg.preconditioners.implementations.amg._algebraic_distance import algebraic_distance
from torchalg.preconditioners.implementations.amg._jacobi_omega import (
    scaled_by_inverse_diagonal as dense_scaled_by_inverse_diagonal,
)
from torchalg.sparse.kernels.algebraic_distance import sparse_algebraic_distance
from torchalg.sparse.kernels.strength import sparse_strength_of_connection
from torchalg.sparse.preconditioners.amg._jacobi_omega import (
    scaled_by_inverse_diagonal as sparse_scaled_by_inverse_diagonal,
)
from torchalg.strategies.orthogonalization import DEFAULT_M_MAX
from torchalg.utils.ls_interpolation import batched_select_interpolatory_set

COARSE_SIZE = 315
"""Coarsest-level size of the paper's hierarchies (dense coarse solve operand)."""

COARSE_SHIFT = 1.0
"""Added to the Gram matrix ``BBᵀ/n`` so the coarse fixture is comfortably SPD."""

AGGREGATE_SIZE = 4
"""Fine nodes per aggregate in the piecewise-constant prolongation fixture."""

FINE_FRACTION = 0.7
"""Fraction of nodes marked fine (``F``) in the strength-graph fixture."""

LS_ROWS = 20_000
LS_MAX_CANDIDATES = 10
LS_CALIBER = 4
"""Realistic shape of the bootstrap LS selection stage (m rows, candidates per row, caliber)."""


def _generator(offset: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(BENCHMARK_SEED + offset)


@pytest.fixture
def dense_case(case: MatrixCase) -> MatrixCase:
    """The case, skipping sizes whose dense form is too large for the operation."""
    if case.n > DENSE_N_CAP:
        pytest.skip(f"dense n={case.n} exceeds DENSE_N_CAP={DENSE_N_CAP}")
    return case


@pytest.fixture
def capped_dense_matrix(dense_case: MatrixCase, request: pytest.FixtureRequest) -> torch.Tensor:
    """``dense_matrix`` that is only materialized when the size passes the cap."""
    del dense_case
    return request.getfixturevalue("dense_matrix")


@pytest.fixture
def zero_guess(rhs: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(rhs)


@pytest.fixture
def coarse_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Dense SPD matrix of the paper's coarse size."""
    factor = torch.randn(COARSE_SIZE, COARSE_SIZE, dtype=dtype, generator=_generator(1))
    gram = factor @ factor.T / COARSE_SIZE
    return (gram + COARSE_SHIFT * torch.eye(COARSE_SIZE, dtype=dtype)).to(device)


@pytest.fixture
def coarse_csr(coarse_matrix: torch.Tensor) -> torch.Tensor:
    return coarse_matrix.to_sparse_csr()


@pytest.fixture
def coarse_rhs(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.randn(COARSE_SIZE, dtype=dtype, generator=_generator(2)).to(device)


@pytest.fixture
def prolongation_csr(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Piecewise-constant prolongation ``n × n/4`` with seeded positive entries."""
    n_coarse = case.n // AGGREGATE_SIZE
    rows = torch.arange(case.n)
    cols = (rows // AGGREGATE_SIZE).clamp(max=n_coarse - 1)
    values = 0.5 + torch.rand(case.n, dtype=dtype, generator=_generator(3))
    coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, size=(case.n, n_coarse))
    return coo.to_sparse_csr().to(device)


@pytest.fixture
def prolongation_dense(dense_case: MatrixCase, request: pytest.FixtureRequest) -> torch.Tensor:
    del dense_case
    return request.getfixturevalue("prolongation_csr").to_dense()


@pytest.fixture
def test_vectors(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Seeded test vectors ``V``, shape ``(n, k)``."""
    shape = (case.n, TEST_VECTOR_COUNT)
    return torch.randn(shape, dtype=dtype, generator=_generator(4)).to(device)


@pytest.fixture
def fine_mask(case: MatrixCase, device: torch.device) -> torch.Tensor:
    draw = torch.rand(case.n, generator=_generator(5))
    return (draw < FINE_FRACTION).to(device)


@pytest.fixture
def smooth_vector(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Positive smooth-vector surrogate for the BootCMatch weight kernels."""
    return torch.rand(case.n, dtype=dtype, generator=_generator(6)).to(device)


@pytest.fixture
def strength_dense(capped_dense_matrix: torch.Tensor) -> torch.Tensor:
    return strength_of_connection(capped_dense_matrix, STRENGTH_THETA)


@pytest.fixture
def strength_csr(csr_matrix_: torch.Tensor) -> torch.Tensor:
    return sparse_strength_of_connection(csr_matrix_, STRENGTH_THETA)


@pytest.fixture
def distance_dense(capped_dense_matrix: torch.Tensor, test_vectors: torch.Tensor) -> torch.Tensor:
    return algebraic_distance(test_vectors, capped_dense_matrix, 1)


@pytest.fixture
def distance_csr(csr_matrix_: torch.Tensor, test_vectors: torch.Tensor) -> torch.Tensor:
    return sparse_algebraic_distance(test_vectors, csr_matrix_, 1)


@pytest.fixture
def scaled_dense(dense_matrix: torch.Tensor) -> torch.Tensor:
    """Dense ``D⁻¹A``, the operator whose spectral radius sets the Jacobi damping."""
    return dense_scaled_by_inverse_diagonal(dense_matrix)


@pytest.fixture
def scaled_csr(csr_matrix_: torch.Tensor) -> torch.Tensor:
    return sparse_scaled_by_inverse_diagonal(csr_matrix_)


@dataclass(frozen=True, slots=True)
class LsProblem:
    """Inputs of the batched LS interpolation selection stage."""

    padded_candidates: torch.Tensor
    candidate_mask: torch.Tensor
    test_vectors: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class LsSelection:
    """Output of the selection stage, the input of the final LS fit."""

    chosen: torch.Tensor
    chosen_mask: torch.Tensor


@pytest.fixture
def ls_problem(dtype: torch.dtype, device: torch.device) -> LsProblem:
    """Synthetic m ≈ 20000 rows with up to 10 candidates and k = 8 test vectors."""
    generator = _generator(7)
    vectors = torch.randn(LS_ROWS, TEST_VECTOR_COUNT, dtype=dtype, generator=generator)
    candidates = torch.randint(0, LS_ROWS, (LS_ROWS, LS_MAX_CANDIDATES), generator=generator)
    counts = torch.randint(LS_CALIBER, LS_MAX_CANDIDATES + 1, (LS_ROWS,), generator=generator)
    mask = torch.arange(LS_MAX_CANDIDATES) < counts.unsqueeze(1)
    weights = 0.5 + torch.rand(TEST_VECTOR_COUNT, dtype=dtype, generator=generator)
    return LsProblem(
        padded_candidates=candidates.to(device),
        candidate_mask=mask.to(device),
        test_vectors=vectors.to(device),
        targets=torch.arange(LS_ROWS, device=device),
        weights=weights.to(device),
    )


@pytest.fixture
def ls_selection(ls_problem: LsProblem) -> LsSelection:
    chosen, chosen_mask = batched_select_interpolatory_set(
        ls_problem.padded_candidates,
        ls_problem.candidate_mask,
        ls_problem.test_vectors,
        ls_problem.targets,
        ls_problem.weights,
        LS_CALIBER,
    )
    return LsSelection(chosen=chosen, chosen_mask=chosen_mask)


@pytest.fixture
def orthogonalization_history(
    case: MatrixCase, dtype: torch.dtype, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``DEFAULT_M_MAX`` stored FCG direction pairs ``(d_j, q_j)``."""
    generator = _generator(8)
    directions = [
        torch.randn(case.n, dtype=dtype, generator=generator).to(device)
        for _ in range(DEFAULT_M_MAX)
    ]
    images = [
        torch.randn(case.n, dtype=dtype, generator=generator).to(device)
        for _ in range(DEFAULT_M_MAX)
    ]
    return directions, images
