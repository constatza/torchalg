"""``setup()`` is the only place a spectral radius is estimated; ``apply()`` never estimates.

Every preset that smooths with weighted Jacobi estimates ``rho(D^-1 A)`` per level while it is
set up. Once ``setup()`` returns, the estimator is replaced by one that raises, so any lazy
first-use estimate left in ``apply()`` fails loudly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.base import MatrixSetupPreconditioner
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    BootstrapAMGPreconditioner,
    vcycle_amg,
    wcycle_amg,
)
from torchalg.preconditioners.implementations.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.preconditioners.implementations.pod import pod2g_preconditioner
from torchalg.utils import spectral

POISSON_SIZE = 64
POD_RANK = 3
SNAPSHOT_COUNT = 5
APPLICATIONS = 3

type PresetBuilder = Callable[[torch.Tensor], MatrixSetupPreconditioner]


def _vcycle(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    return vcycle_amg(matrix, n_levels=3)


def _wcycle(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    return wcycle_amg(matrix, n_levels=3)


def _adaptive(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    return AdaptiveSAPreconditioner(num_candidates=2, max_levels=3, max_coarse=6)


def _bootstrap(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    return BootstrapAMGPreconditioner(k_r=4, eta=2, n_bootstrap_cycles=1, seed=5, max_coarse=6)


def _bootcmatch(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    return BootCMatchPreconditioner(
        seed=3, max_levels=4, max_coarse=4, k_max=3, rho_desired=0.8, max_hierarchies=2
    )


def _pod(matrix: torch.Tensor) -> MatrixSetupPreconditioner:
    snapshots = torch.randn(
        SNAPSHOT_COUNT,
        POISSON_SIZE,
        generator=torch.Generator().manual_seed(4),
        dtype=matrix.dtype,
    )
    return pod2g_preconditioner(matrix, snapshots, rank=POD_RANK)


PRESETS: dict[str, PresetBuilder] = {
    "vcycle_amg": _vcycle,
    "wcycle_amg": _wcycle,
    "adaptive_sa": _adaptive,
    "bootstrap_amg": _bootstrap,
    "bootcmatch": _bootcmatch,
    "pod2g": _pod,
}


@pytest.fixture
def poisson(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return poisson_1d_factory(POISSON_SIZE)


@pytest.fixture
def residual(poisson: torch.Tensor) -> torch.Tensor:
    return torch.ones(POISSON_SIZE, dtype=poisson.dtype)


@pytest.fixture
def forbid_spectral_estimates(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    """Callable that makes every import site of ``approximate_spectral_radius`` raise."""

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("spectral radius estimated outside setup()")

    def install() -> None:
        original = spectral.approximate_spectral_radius
        for module in list(sys.modules.values()):
            if getattr(module, "approximate_spectral_radius", None) is original:
                monkeypatch.setattr(module, "approximate_spectral_radius", forbidden)

    return install


@pytest.mark.parametrize("builder", PRESETS.values(), ids=PRESETS.keys())
def test_apply_never_estimates_a_spectral_radius(
    builder: PresetBuilder,
    poisson: torch.Tensor,
    residual: torch.Tensor,
    forbid_spectral_estimates: Callable[[], None],
) -> None:
    preconditioner = builder(poisson)
    preconditioner.setup(poisson)

    forbid_spectral_estimates()
    results = [preconditioner.apply(residual) for _ in range(APPLICATIONS)]

    assert all(torch.isfinite(result).all() for result in results)
    assert all(torch.equal(result, results[0]) for result in results)


def test_the_guard_reaches_the_estimator_import_sites(
    poisson: torch.Tensor,
    forbid_spectral_estimates: Callable[[], None],
) -> None:
    """A preset that estimates during setup trips the guard (so the guard is not vacuous)."""
    forbid_spectral_estimates()

    with pytest.raises(AssertionError, match="outside setup"):
        vcycle_amg(poisson, n_levels=3).setup(poisson)
