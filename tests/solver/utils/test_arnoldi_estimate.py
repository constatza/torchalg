"""``approximate_spectral_radius``: equivalence with the pre-change estimate and host-sync budget."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.support.arnoldi_reference import reference_spectral_radius
from torchalg.utils import spectral
from torchalg.utils.spectral import (
    ORTHOGONALIZER_BY_DEVICE,
    _classical_gram_schmidt,
    approximate_spectral_radius,
)

SCALAR_CONVERSIONS = ("__bool__", "__float__", "__int__", "item")
"""Tensor methods that force a device-to-host sync when the tensor lives on a device."""


def _seeded_draw(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


KRYLOV_DIMENSION = 15
MAX_DISPATCHES_PER_KRYLOV_STEP = 30
"""Block (classical) Gram-Schmidt dispatches a constant number of operators per Krylov step;
the per-vector modified loop dispatches several per basis vector (1124 at 15 steps)."""


class _LaunchCounter(TorchDispatchMode):
    """Counts every ATen operator dispatched while active."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


class TestArnoldiEquivalence:
    """The CPU estimate is the frozen reference; the block variant agrees to rounding."""

    def test_cpu_default_is_bit_identical_to_reference(
        self, arnoldi_matrix: torch.Tensor, test_seed: int
    ) -> None:
        """The CPU keeps modified Gram-Schmidt, so the estimate does not change at all."""
        new = approximate_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        old = reference_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        assert torch.equal(new, old)

    def test_classical_variant_matches_reference(
        self, arnoldi_matrix: torch.Tensor, test_seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The block variant used on a GPU gives the same estimate to rounding."""
        monkeypatch.setattr(
            spectral, "ORTHOGONALIZER_BY_DEVICE", MappingProxyType({"cpu": _classical_gram_schmidt})
        )
        new = approximate_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        old = reference_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        torch.testing.assert_close(new, old, rtol=1e-12, atol=0.0)

    def test_registry_maps_cuda_to_the_block_variant(self) -> None:
        assert ORTHOGONALIZER_BY_DEVICE["cuda"] is _classical_gram_schmidt
        assert "cpu" not in ORTHOGONALIZER_BY_DEVICE


class TestArnoldiSyncBudget:
    """One restart-free pass converts at most two tensors to Python scalars."""

    def test_scalar_conversions_per_pass(
        self, arnoldi_matrix: torch.Tensor, test_seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-step breakdown checks must not convert a tensor each Krylov step."""
        calls: list[str] = []
        for name in SCALAR_CONVERSIONS:
            original = getattr(torch.Tensor, name)

            def counted(self, *args, _original=original, _name=name):
                calls.append(_name)
                return _original(self, *args)

            monkeypatch.setattr(torch.Tensor, name, counted)
        approximate_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed), restart=0)
        assert len(calls) <= 2


class TestArnoldiLaunchBudget:
    """Orthogonalization launches are independent of the basis size."""

    def test_launches_per_pass_do_not_grow_with_the_basis(
        self, arnoldi_matrix: torch.Tensor, test_seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            spectral, "ORTHOGONALIZER_BY_DEVICE", MappingProxyType({"cpu": _classical_gram_schmidt})
        )
        with _LaunchCounter() as counter:
            approximate_spectral_radius(
                arnoldi_matrix, _seeded_draw(test_seed), restart=0, maxiter=KRYLOV_DIMENSION
            )
        assert counter.count <= MAX_DISPATCHES_PER_KRYLOV_STEP * KRYLOV_DIMENSION
