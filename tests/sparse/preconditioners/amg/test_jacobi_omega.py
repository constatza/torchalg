"""Tests for ``torchalg.sparse.preconditioners.amg._jacobi_omega``.

The 2 sparse test cases from
``tests/solver/preconditioners/implementations/test_jacobi_omega.py``,
moved here per ``docs/plan.md``'s migration notes (the pre-existing dense
cases stay at that path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from torchalg.sparse.preconditioners.amg import _jacobi_omega as jacobi_omega_module
from torchalg.sparse.preconditioners.amg._jacobi_omega import (
    PROLONGATION_NOMINAL,
    RELAXATION_NOMINAL,
    jacobi_omega,
    jacobi_spectral_radius,
)
from torchalg.utils.spectral import approximate_spectral_radius

_SEED = 0


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


@pytest.fixture
def seeded_draw() -> Callable[[int], torch.Tensor]:
    """The fixed-seed start-vector source the sparse estimate uses."""
    generator = torch.Generator().manual_seed(_SEED)
    return lambda size: torch.rand(size, generator=generator, dtype=torch.float64)


class TestSparseJacobiSpectralRadius:
    """``jacobi_spectral_radius``/``scaled_by_inverse_diagonal`` accept sparse CSR input."""

    def test_matches_dense_reference(
        self, poisson_matrix: torch.Tensor, seeded_draw: Callable[[int], torch.Tensor]
    ) -> None:
        """The Arnoldi estimate equals the one on the dense ``D^-1 A``."""
        scaled = poisson_matrix / torch.diagonal(poisson_matrix).unsqueeze(1)
        dense_rho = approximate_spectral_radius(scaled, seeded_draw)
        sparse_rho = jacobi_spectral_radius(poisson_matrix.to_sparse_csr())
        torch.testing.assert_close(sparse_rho, dense_rho)

    def test_scaled_by_inverse_diagonal_matches_dense_reference(
        self, poisson_matrix: torch.Tensor
    ) -> None:
        """``D^-1 A`` densified from the sparse path equals the dense path exactly."""
        dense = poisson_matrix / torch.diagonal(poisson_matrix).unsqueeze(1)
        sparse = jacobi_omega_module.scaled_by_inverse_diagonal(poisson_matrix.to_sparse_csr())
        torch.testing.assert_close(sparse.to_dense(), dense)


class TestJacobiOmega:
    """``jacobi_omega`` is a pure function of the matrix: nothing is cached across calls."""

    def test_nominal_over_rho(self, poisson_matrix: torch.Tensor) -> None:
        csr = poisson_matrix.to_sparse_csr()
        rho = jacobi_spectral_radius(csr)
        torch.testing.assert_close(jacobi_omega(csr, RELAXATION_NOMINAL, None), 1.0 / rho)
        torch.testing.assert_close(
            jacobi_omega(csr, PROLONGATION_NOMINAL, None), PROLONGATION_NOMINAL / rho
        )

    def test_override_wins_without_estimating(
        self, poisson_matrix: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*_: object, **__: object) -> None:
            raise AssertionError("override must not estimate rho")

        monkeypatch.setattr(jacobi_omega_module, "approximate_spectral_radius", forbidden)
        omega = jacobi_omega(poisson_matrix.to_sparse_csr(), RELAXATION_NOMINAL, 0.5)
        torch.testing.assert_close(omega, torch.tensor(0.5, dtype=poisson_matrix.dtype))

    def test_every_call_estimates_again(
        self, poisson_matrix: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No hidden id()-keyed cache: repeated calls on one matrix each pay for the estimate."""
        calls: list[int] = []
        original = jacobi_omega_module.approximate_spectral_radius

        def counting(*args: Any, **kwargs: Any) -> torch.Tensor:
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(jacobi_omega_module, "approximate_spectral_radius", counting)
        csr = poisson_matrix.to_sparse_csr()
        jacobi_spectral_radius(csr)
        jacobi_spectral_radius(csr)
        assert len(calls) == 2

    def test_module_holds_no_global_cache(self) -> None:
        assert not hasattr(jacobi_omega_module, "_cache")
