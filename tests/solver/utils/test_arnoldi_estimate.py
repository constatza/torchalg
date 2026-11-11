"""``approximate_spectral_radius``: equivalence with the pre-change estimate and host-sync budget."""

from __future__ import annotations

import pytest
import torch

from tests.support.arnoldi_reference import reference_spectral_radius
from torchalg.utils.spectral import approximate_spectral_radius

SCALAR_CONVERSIONS = ("__bool__", "__float__", "__int__", "item")
"""Tensor methods that force a device-to-host sync when the tensor lives on a device."""


def _seeded_draw(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


class TestArnoldiEquivalence:
    """The restructured estimate equals the frozen reference."""

    def test_matches_reference(self, arnoldi_matrix: torch.Tensor, test_seed: int) -> None:
        """Same random stream, same estimate to rounding."""
        new = approximate_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        old = reference_spectral_radius(arnoldi_matrix, _seeded_draw(test_seed))
        torch.testing.assert_close(new, old, rtol=1e-12, atol=0.0)


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
