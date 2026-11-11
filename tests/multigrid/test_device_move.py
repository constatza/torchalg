"""Set up once, then move: every AMG-family preconditioner follows ``.to()`` without re-setup.

The prepared hierarchy (level matrices, transfers, prepared smoothers, coarse factorization)
moves with ``utils.device.moved_to``; the moved preconditioner stays ready and reproduces
the original application. Dtype moves run everywhere; device moves need a GPU. Sparse CSR
products abort on ROCm (rocSPARSE fault on gfx1030), so the sparse GPU cases run on CUDA
only.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from torchalg.preconditioners.base import MatrixSetupPreconditioner
from torchalg.preconditioners.implementations import (
    AdaptiveSAPreconditioner,
    BootstrapAMGPreconditioner,
)
from torchalg.preconditioners.implementations.amg import vcycle_amg
from torchalg.preconditioners.implementations.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootcmatch import (
    BootCMatchPreconditioner as SparseBootCMatchPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg

GRID_SIDE = 12
ANISOTROPY = 1.0
STRENGTH_THETA = 0.0
CAST_RTOL = 1e-4
CAST_ATOL = 1e-5

type Builder = Callable[[torch.Tensor], MatrixSetupPreconditioner]

DENSE_BUILDERS: dict[str, Builder] = {
    "aggregation": lambda matrix: vcycle_amg(matrix, theta=STRENGTH_THETA),
    "adaptive_sa": lambda _matrix: AdaptiveSAPreconditioner(),
    "bootstrap": lambda _matrix: BootstrapAMGPreconditioner(),
    "bootcmatch": lambda _matrix: BootCMatchPreconditioner(),
}
SPARSE_BUILDERS: dict[str, Builder] = {
    "aggregation": lambda matrix: sparse_vcycle_amg(matrix, theta=STRENGTH_THETA),
    "adaptive_sa": lambda _matrix: SparseAdaptiveSAPreconditioner(),
    "bootstrap": lambda _matrix: SparseBootstrapAMGPreconditioner(),
    "bootcmatch": lambda _matrix: SparseBootCMatchPreconditioner(),
}
CASES = [("dense", name) for name in DENSE_BUILDERS] + [("csr", name) for name in SPARSE_BUILDERS]


@pytest.fixture
def poisson_2d(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense 144x144 isotropic 2-D Poisson matrix."""
    return anisotropic_2d_factory(GRID_SIDE, ANISOTROPY)


@pytest.fixture(params=CASES, ids=[f"{tree}-{name}" for tree, name in CASES])
def prepared(
    request: pytest.FixtureRequest, poisson_2d: torch.Tensor
) -> tuple[str, MatrixSetupPreconditioner]:
    """A set-up preconditioner (tree tag, object) on the CPU in float64."""
    tree, name = request.param
    matrix = poisson_2d if tree == "dense" else poisson_2d.to_sparse_csr()
    builders = DENSE_BUILDERS if tree == "dense" else SPARSE_BUILDERS
    return tree, builders[name](matrix).setup(matrix)


@pytest.fixture
def residual(poisson_2d: torch.Tensor) -> torch.Tensor:
    """Smooth float64 residual for the Poisson fixture."""
    return torch.linspace(1.0, 2.0, poisson_2d.shape[0], dtype=poisson_2d.dtype)


def test_dtype_move_keeps_the_prepared_preconditioner(
    prepared: tuple[str, MatrixSetupPreconditioner], residual: torch.Tensor
) -> None:
    _, preconditioner = prepared
    expected = preconditioner.apply(residual)

    assert isinstance(preconditioner, nn.Module)
    moved = preconditioner.to(dtype=torch.float32)
    result = moved.apply(residual.to(torch.float32))

    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected.to(torch.float32), rtol=CAST_RTOL, atol=CAST_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_device_move_keeps_the_prepared_preconditioner(
    prepared: tuple[str, MatrixSetupPreconditioner], residual: torch.Tensor
) -> None:
    tree, preconditioner = prepared
    if tree == "csr" and torch.version.hip is not None:
        pytest.skip("rocSPARSE CSR products fault on ROCm")
    expected = preconditioner.apply(residual)

    assert isinstance(preconditioner, nn.Module)
    moved = preconditioner.to("cuda")
    first = moved.apply(residual.cuda())
    second = moved.apply(residual.cuda())

    torch.testing.assert_close(first.cpu(), expected)
    torch.testing.assert_close(second.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_adaptive_sa_accepts_host_candidates_for_a_gpu_matrix(poisson_2d: torch.Tensor) -> None:
    """Known near-null candidates given on the host follow the matrix to its device."""
    candidates = torch.ones(poisson_2d.shape[0], 1, dtype=poisson_2d.dtype)
    matrix = poisson_2d.cuda()

    preconditioner = AdaptiveSAPreconditioner(initial_candidates=candidates).setup(matrix)

    assert preconditioner.apply(torch.ones_like(matrix[0])).device.type == "cuda"
