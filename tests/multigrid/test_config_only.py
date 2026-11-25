"""AMG preconditioners bind configuration at construction and the matrix in ``setup``.

Every AMG-family preconditioner (dense and sparse) is constructible without a matrix, is not
ready until ``setup(matrix)``, builds its hierarchy once per ``setup``, and a ``setup`` that
fails keeps the previous matrix, hierarchy and result untouched.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid import AMGPreconditioner, VCycle
from torchalg.preconditioners.base import MatrixSetupPreconditioner, PreconditionerNotReadyError
from torchalg.preconditioners.implementations import (
    AdaptiveSAPreconditioner,
    BootstrapAMGPreconditioner,
    pod2g_preconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    DenseTransferOperator,
    JacobiSmoother,
    vcycle_amg,
    wcycle_amg,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy, compute_pod_basis
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_lu_coarse_solver
from torchalg.sparse.preconditioners.amg.smoothers import (
    resolve_jacobi_default as sparse_resolve_jacobi_default,
)
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg
from torchalg.sparse.preconditioners.amg.variants import wcycle_amg as sparse_wcycle_amg

GRID_SIDE = 12
STRENGTH_THETA = 0.0
POD_RANK = 4
POD_SNAPSHOTS = 6
SEED = 3

type Config = Callable[[torch.Tensor], MatrixSetupPreconditioner]


def _dense_pod(snapshots: torch.Tensor) -> MatrixSetupPreconditioner:
    return pod2g_preconditioner(snapshots, rank=POD_RANK)


def _sparse_pod(snapshots: torch.Tensor) -> MatrixSetupPreconditioner:
    return AMGPreconditioner(
        coarsening=PODCoarseningStrategy(compute_pod_basis(snapshots, POD_RANK)),
        cycle=VCycle(
            sparse_resolve_jacobi_default(None, None),
            n_pre=2,
            n_post=2,
            coarse_solver_factory=dense_lu_coarse_solver,
        ),
        n_levels=2,
        linear=True,
    )


# name -> (tree, builder taking the POD snapshots; only the POD builders read them)
BUILDERS: dict[str, tuple[str, Config]] = {
    "dense-vcycle": ("dense", lambda _s: vcycle_amg(theta=STRENGTH_THETA)),
    "dense-wcycle": ("dense", lambda _s: wcycle_amg(theta=STRENGTH_THETA)),
    "dense-adaptive_sa": ("dense", lambda _s: AdaptiveSAPreconditioner(seed=SEED)),
    "dense-bootstrap": ("dense", lambda _s: BootstrapAMGPreconditioner(seed=SEED)),
    "dense-pod": ("dense", _dense_pod),
    "csr-vcycle": ("csr", lambda _s: sparse_vcycle_amg(theta=STRENGTH_THETA)),
    "csr-wcycle": ("csr", lambda _s: sparse_wcycle_amg(theta=STRENGTH_THETA)),
    "csr-adaptive_sa": ("csr", lambda _s: SparseAdaptiveSAPreconditioner(seed=SEED)),
    "csr-bootstrap": ("csr", lambda _s: SparseBootstrapAMGPreconditioner(seed=SEED)),
    "csr-pod": ("csr", _sparse_pod),
}


@pytest.fixture
def poisson_2d(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense isotropic 2-D Poisson matrix."""
    return anisotropic_2d_factory(GRID_SIDE, 1.0)


@pytest.fixture
def snapshots(poisson_2d: torch.Tensor) -> torch.Tensor:
    """Solution snapshots of the Poisson system for seeded random right-hand sides."""
    generator = torch.Generator().manual_seed(SEED)
    rhs = torch.randn(
        poisson_2d.shape[0], POD_SNAPSHOTS, dtype=poisson_2d.dtype, generator=generator
    )
    return torch.linalg.solve(poisson_2d, rhs).T.contiguous()


@pytest.fixture(params=list(BUILDERS))
def fresh(
    request: pytest.FixtureRequest, poisson_2d: torch.Tensor, snapshots: torch.Tensor
) -> tuple[MatrixSetupPreconditioner, torch.Tensor]:
    """An unprepared preconditioner built from configuration alone, and its system matrix."""
    tree, build = BUILDERS[request.param]
    matrix = poisson_2d if tree == "dense" else poisson_2d.to_sparse_csr()
    return build(snapshots), matrix


def _rhs(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linspace(1.0, 2.0, matrix.shape[0], dtype=matrix.dtype)


def test_constructed_without_a_matrix_is_not_ready(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor],
) -> None:
    preconditioner, matrix = fresh

    assert not preconditioner.is_ready
    with pytest.raises(PreconditionerNotReadyError):
        preconditioner.apply(_rhs(matrix))


def test_setup_binds_the_matrix_and_makes_it_ready(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor],
) -> None:
    preconditioner, matrix = fresh

    assert preconditioner.setup(matrix) is preconditioner

    assert preconditioner.is_ready
    assert isinstance(preconditioner, AMGPreconditioner)
    assert preconditioner.matrix is matrix
    assert preconditioner.apply(_rhs(matrix)).shape == (matrix.shape[0],)


def test_matrix_is_unavailable_before_setup(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor],
) -> None:
    preconditioner, _ = fresh
    assert isinstance(preconditioner, AMGPreconditioner)

    with pytest.raises(PreconditionerNotReadyError):
        _ = preconditioner.matrix


def test_setup_prepares_the_hierarchy_exactly_once(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    preconditioner, matrix = fresh
    assert isinstance(preconditioner, AMGPreconditioner)
    prepare = preconditioner._cycle.prepare
    calls: list[object] = []
    monkeypatch.setattr(
        preconditioner._cycle,
        "prepare",
        lambda hierarchy: calls.append(hierarchy) or prepare(hierarchy),
    )

    preconditioner.setup(matrix)
    preconditioner.apply(_rhs(matrix))

    assert len(calls) == 1


def test_failed_setup_keeps_the_previous_state(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor], monkeypatch: pytest.MonkeyPatch
) -> None:
    preconditioner, matrix = fresh
    assert isinstance(preconditioner, AMGPreconditioner)
    preconditioner.setup(matrix)
    rhs = _rhs(matrix)
    before = preconditioner.apply(rhs)
    hierarchy = preconditioner._hierarchy
    n_levels = preconditioner.n_levels
    result = getattr(preconditioner, "_result", None)

    def fail(_hierarchy: object) -> None:
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(preconditioner._cycle, "prepare", fail)
    with pytest.raises(RuntimeError, match="prepare failed"):
        preconditioner.setup(2.0 * matrix)

    assert preconditioner.is_ready
    assert preconditioner._hierarchy is hierarchy
    assert preconditioner.matrix is matrix
    assert preconditioner.n_levels == n_levels
    assert getattr(preconditioner, "_result", None) is result
    assert torch.equal(preconditioner.apply(rhs), before)


def test_dtype_move_keeps_the_hierarchy_and_matrix(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor],
) -> None:
    preconditioner, matrix = fresh
    assert isinstance(preconditioner, AMGPreconditioner)
    preconditioner.setup(matrix)

    moved = preconditioner.to(dtype=torch.float32)

    assert moved.is_ready
    assert moved.matrix.dtype == torch.float32
    assert moved.apply(_rhs(matrix).to(torch.float32)).dtype == torch.float32


def test_dtype_move_before_setup_stays_unprepared(
    fresh: tuple[MatrixSetupPreconditioner, torch.Tensor],
) -> None:
    preconditioner, _ = fresh
    assert isinstance(preconditioner, AMGPreconditioner)

    assert not preconditioner.to(dtype=torch.float32).is_ready


class _RecordingCoarsening:
    """Aggregation-like strategy that records the inputs bound to it before ``setup``."""

    extra_input_names = ("positions",)

    def __init__(self, inner: AggregationCoarsening) -> None:
        self.inner = inner
        self.bound: dict[str, torch.Tensor] = {}

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        self.bound = inputs

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        return self.inner.build_transfer(A)


def test_bind_inputs_works_before_setup(poisson_2d: torch.Tensor) -> None:
    coarsening = _RecordingCoarsening(AggregationCoarsening(theta=STRENGTH_THETA))
    preconditioner = AMGPreconditioner(
        coarsening=coarsening, cycle=VCycle(JacobiSmoother()), n_levels=2
    )
    positions = torch.zeros(poisson_2d.shape[0], dtype=poisson_2d.dtype)

    preconditioner.bind_inputs(positions=positions)

    assert coarsening.bound == {"positions": positions}
    assert not preconditioner.is_ready
    assert preconditioner.setup(poisson_2d).is_ready
