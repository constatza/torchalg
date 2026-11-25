"""Cost-based CPU/GPU placement policy (``torchalg.placement``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.placement import (
    DENSE_GPU_MIN_N,
    SPMV_GPU_MIN_NNZ,
    PlacementProblem,
    choose_device,
    placement_problem,
)
from torchalg.preconditioners.base import Identity, Placement

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tests.conftest import ToTorch

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


@pytest.fixture
def tridiagonal_spd_small_torch(tridiagonal_spd_small: NDArray, to_torch: ToTorch) -> torch.Tensor:
    """Small SPD tridiagonal matrix as a float64 dense tensor."""
    return to_torch(tridiagonal_spd_small, dtype=torch.float64)


@pytest.fixture
def cuda_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a CUDA device is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def cuda_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no CUDA device is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.usefixtures("cuda_absent")
@pytest.mark.parametrize("kind", list(Placement), ids=lambda kind: kind.name)
def test_cpu_when_no_cuda(kind: Placement) -> None:
    """Without a GPU every kind lands on the CPU."""
    problem = PlacementProblem(n=10**7, nnz=10**9, kind=kind)
    assert choose_device(problem) == CPU


@pytest.mark.usefixtures("cuda_visible")
@pytest.mark.parametrize("n", [10, 10**5, 10**7])
def test_sequential_sweep_stays_on_cpu(n: int) -> None:
    """Triangular-solve preconditioners are launch-latency bound on a GPU at any size."""
    problem = PlacementProblem(n=n, nnz=50 * n, kind=Placement.SEQUENTIAL_SWEEP)
    assert choose_device(problem) == CPU


@pytest.mark.usefixtures("cuda_visible")
def test_spmv_bound_switches_at_nnz_threshold() -> None:
    """SpMV-bound work pays for the GPU latency floor only above the nnz threshold."""
    below = PlacementProblem(n=1000, nnz=SPMV_GPU_MIN_NNZ - 1, kind=Placement.SPMV_BOUND)
    above = PlacementProblem(n=1000, nnz=SPMV_GPU_MIN_NNZ, kind=Placement.SPMV_BOUND)
    assert choose_device(below) == CPU
    assert choose_device(above) == CUDA


@pytest.mark.usefixtures("cuda_visible")
def test_dense_bound_switches_at_row_threshold() -> None:
    """Dense products win on the GPU once the matrix has enough rows."""
    below = PlacementProblem(n=DENSE_GPU_MIN_N - 1, nnz=0, kind=Placement.DENSE_BOUND)
    above = PlacementProblem(n=DENSE_GPU_MIN_N, nnz=0, kind=Placement.DENSE_BOUND)
    assert choose_device(below) == CPU
    assert choose_device(above) == CUDA


def test_problem_from_dense_matrix_is_dense_bound(
    tridiagonal_spd_small_torch: torch.Tensor,
) -> None:
    """A strided system matrix with an SpMV-class preconditioner is classified dense."""
    problem = placement_problem(tridiagonal_spd_small_torch, Identity())
    n = tridiagonal_spd_small_torch.shape[0]
    assert problem == PlacementProblem(n=n, nnz=n * n, kind=Placement.DENSE_BOUND)


def test_problem_from_csr_matrix_counts_stored_entries(
    tridiagonal_spd_small_torch: torch.Tensor,
) -> None:
    """A CSR system matrix contributes its stored entry count and stays SpMV bound."""
    csr = tridiagonal_spd_small_torch.to_sparse_csr()
    problem = placement_problem(csr, Identity())
    assert problem == PlacementProblem(n=csr.shape[0], nnz=csr._nnz(), kind=Placement.SPMV_BOUND)


def test_problem_keeps_sequential_preconditioner_kind(
    tridiagonal_spd_small_torch: torch.Tensor,
) -> None:
    """A preconditioner declaring sequential sweeps wins over the matrix format."""
    ic0 = _SweepPreconditioner()
    problem = placement_problem(tridiagonal_spd_small_torch.to_sparse_csr(), ic0)
    assert problem.kind is Placement.SEQUENTIAL_SWEEP


class _SweepPreconditioner(Identity):
    """Identity that declares a triangular-solve placement."""

    placement = Placement.SEQUENTIAL_SWEEP
