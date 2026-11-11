"""Setup/apply contract of the sparse AMG presets: ``setup()`` does all one-off work.

After ``setup()`` returns, ``apply()`` must not estimate a spectral radius,
build a level schedule, or otherwise do matrix-dependent one-off work; and the
stage-scoped preparation inside the setup algorithms must prepare each distinct
level matrix and prolongation at most once.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
import torch

from torchalg.multigrid import AMGPreconditioner, VCycle
from torchalg.preconditioners.base import MatrixSetupPreconditioner
from torchalg.sparse.kernels import triangular as triangular_kernel
from torchalg.sparse.preconditioners.amg import _gauss_seidel
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner,
    adaptive_sa_hierarchy,
)
from torchalg.sparse.preconditioners.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_lu_coarse_solver
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother, JacobiSmoother
from torchalg.sparse.preconditioners.amg.transfer import SparseTransferOperator
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg, wcycle_amg
from torchalg.sparse.preconditioners.pod.coarsening import PODCoarseningStrategy

_SPECTRAL_NAME = "approximate_spectral_radius"
_POD_RANK = 5


@pytest.fixture
def forbid_spectral_radius(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Make every bound ``approximate_spectral_radius`` in ``torchalg`` raise when called."""

    def forbidden(*_: object, **__: object) -> torch.Tensor:
        raise AssertionError("spectral radius estimated outside setup")

    def arm() -> None:
        for name, module in list(sys.modules.items()):
            is_torchalg = name.startswith("torchalg") and isinstance(module, ModuleType)
            if is_torchalg and hasattr(module, _SPECTRAL_NAME):
                monkeypatch.setattr(module, _SPECTRAL_NAME, forbidden)

    return arm


@pytest.fixture
def pod_snapshots_large(poisson_1d_large_dense: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Solution snapshots spanning the 64-dof Poisson solution space, shape (8, 64)."""
    generator = torch.Generator().manual_seed(test_seed)
    n = poisson_1d_large_dense.shape[0]
    rhs_batch = torch.randn(8, n, dtype=poisson_1d_large_dense.dtype, generator=generator)
    return torch.linalg.solve(poisson_1d_large_dense, rhs_batch.T).T


def _pod2g(matrix: torch.Tensor, snapshots: torch.Tensor) -> AMGPreconditioner:
    coarsening = PODCoarseningStrategy(rank=_POD_RANK)
    coarsening.fit(snapshots)
    return AMGPreconditioner(
        matrix=matrix,
        coarsening=coarsening,
        cycle=VCycle(JacobiSmoother(), coarse_solver_factory=dense_lu_coarse_solver),
        n_levels=2,
        linear=True,
    )


@pytest.fixture(
    params=["vcycle_amg", "wcycle_amg", "adaptive_sa", "bootstrap", "bootcmatch", "pod2g"]
)
def preconditioner_factory(
    request: pytest.FixtureRequest, pod_snapshots_large: torch.Tensor
) -> Callable[[torch.Tensor], MatrixSetupPreconditioner]:
    """Unsetup preconditioner builders for every sparse preset that smooths with Jacobi."""
    factories: dict[str, Callable[[torch.Tensor], MatrixSetupPreconditioner]] = {
        "vcycle_amg": lambda matrix: vcycle_amg(matrix, n_levels=3),
        "wcycle_amg": lambda matrix: wcycle_amg(matrix, n_levels=3),
        "adaptive_sa": lambda matrix: AdaptiveSAPreconditioner(
            num_candidates=2, candidate_iters=2, max_levels=3, max_coarse=4, seed=0
        ),
        "bootstrap": lambda matrix: BootstrapAMGPreconditioner(
            k_r=6, eta=2, n_bootstrap_cycles=1, nu=3, caliber=3, max_levels=3, max_coarse=4, seed=5
        ),
        "bootcmatch": lambda matrix: BootCMatchPreconditioner(
            seed=3, max_levels=4, max_coarse=4, k_max=4, max_hierarchies=3
        ),
        "pod2g": lambda matrix: _pod2g(matrix, pod_snapshots_large),
    }
    return factories[request.param]


class TestApplyAfterSetupDoesNoSpectralEstimate:
    def test_apply_works_with_the_estimate_disabled(
        self,
        preconditioner_factory: Callable[[torch.Tensor], MatrixSetupPreconditioner],
        poisson_1d_large_csr: torch.Tensor,
        poisson_1d_large_rhs: torch.Tensor,
        forbid_spectral_radius: Callable[[], None],
    ) -> None:
        preconditioner = preconditioner_factory(poisson_1d_large_csr)
        preconditioner.setup(poisson_1d_large_csr)
        expected = preconditioner.apply(poisson_1d_large_rhs)

        forbid_spectral_radius()
        actual = preconditioner.apply(poisson_1d_large_rhs)

        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestGaussSeidelSolveWorkIsPrepared:
    def test_apply_builds_no_level_schedule(
        self,
        poisson_1d_large_csr: torch.Tensor,
        poisson_1d_large_rhs: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        original = triangular_kernel.level_schedule

        def counting(*args: Any, **kwargs: Any):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(triangular_kernel, "level_schedule", counting)
        preconditioner = vcycle_amg(
            poisson_1d_large_csr, n_levels=3, smoother=GaussSeidelSmoother()
        )
        preconditioner.setup(poisson_1d_large_csr)
        built_in_setup = len(calls)
        assert built_in_setup == 2 * 2  # forward + backward on each of the 2 smoothed levels

        for _ in range(3):
            preconditioner.apply(poisson_1d_large_rhs)
        assert len(calls) == built_in_setup


class TestSetupStagesPrepareEachMatrixOnce:
    @pytest.fixture
    def built_matrices(self, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
        """Matrices handed to ``PreparedSymmetricGaussSeidel.build``, in call order."""
        matrices: list[torch.Tensor] = []
        original = _gauss_seidel.PreparedSymmetricGaussSeidel.build.__func__  # type: ignore[attr-defined]

        def recording(cls: type[_gauss_seidel.PreparedSymmetricGaussSeidel], matrix: torch.Tensor):
            matrices.append(matrix)
            return original(cls, matrix)

        monkeypatch.setattr(
            _gauss_seidel.PreparedSymmetricGaussSeidel, "build", classmethod(recording)
        )
        return matrices

    @pytest.fixture
    def built_operators(self, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
        """Prolongations handed to ``SparseTransferOperator``, in call order."""
        prolongations: list[torch.Tensor] = []
        original = SparseTransferOperator.__init__

        def recording(self: SparseTransferOperator, P: torch.Tensor) -> None:
            prolongations.append(P)
            original(self, P)

        monkeypatch.setattr(SparseTransferOperator, "__init__", recording)
        return prolongations

    def test_adaptive_hierarchy_prepares_each_matrix_once(
        self,
        poisson_1d_large_csr: torch.Tensor,
        built_matrices: list[torch.Tensor],
        built_operators: list[torch.Tensor],
    ) -> None:
        adaptive_sa_hierarchy(
            poisson_1d_large_csr,
            num_candidates=3,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
        )
        assert built_matrices
        assert len({id(matrix) for matrix in built_matrices}) == len(built_matrices)
        assert built_operators
        assert len({id(p) for p in built_operators}) == len(built_operators)

    def test_bootstrap_setup_prepares_each_matrix_once(
        self,
        poisson_1d_large_csr: torch.Tensor,
        built_matrices: list[torch.Tensor],
        built_operators: list[torch.Tensor],
    ) -> None:
        BootstrapAMGPreconditioner(
            k_r=6, eta=2, n_bootstrap_cycles=2, nu=3, caliber=3, max_levels=3, max_coarse=4, seed=5
        ).setup(poisson_1d_large_csr)
        assert built_matrices
        assert len({id(matrix) for matrix in built_matrices}) == len(built_matrices)

    def test_setup_runs_do_not_share_prepared_state(
        self,
        poisson_1d_large_csr: torch.Tensor,
        built_matrices: list[torch.Tensor],
    ) -> None:
        """The finest matrix is prepared once per ``setup()``, never cached across runs."""
        preconditioner = BootstrapAMGPreconditioner(
            k_r=4, eta=1, n_bootstrap_cycles=1, nu=3, caliber=3, max_levels=3, max_coarse=4, seed=5
        )
        preconditioner.setup(poisson_1d_large_csr)
        first_run = sum(matrix is poisson_1d_large_csr for matrix in built_matrices)
        preconditioner.setup(poisson_1d_large_csr)
        second_run = sum(matrix is poisson_1d_large_csr for matrix in built_matrices) - first_run
        assert first_run == second_run == 1
