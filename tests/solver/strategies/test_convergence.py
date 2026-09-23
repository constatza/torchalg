"""Tests for ``torchalg.strategies.convergence``.

Includes a parity check against the frozen numpy reference snapshot
(``tests/support/scipy_reference/strategies/convergence.py``, Stage 0):
this criterion is what the scipy-equivalence benchmarks ultimately depend
on being mathematically identical to, so the formula gets a direct
cross-check here, not just isolated unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchalg.strategies.convergence import CombinedToleranceCriterion
from torchalg.strategies.norms import energy_norm, euclidean_norm


class TestCombinedToleranceCriterion:
    """Tests for ``CombinedToleranceCriterion``."""

    def test_default_norm_is_euclidean(self) -> None:
        """The default-constructed criterion uses ``euclidean_norm``."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.norm is euclidean_norm

    def test_threshold_is_rtol_scaled_when_it_dominates(self) -> None:
        """``threshold == rtol * ||b||`` when that exceeds ``atol``."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.threshold(rhs_norm=1.0) == pytest.approx(1e-6)

    def test_threshold_is_atol_floor_when_it_dominates(self) -> None:
        """``threshold == atol`` when ``rtol * ||b||`` is smaller (tiny RHS)."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.threshold(rhs_norm=1e-12) == pytest.approx(1e-14)

    def test_converged_residual_satisfies_criterion(
        self,
        converged_residual: torch.Tensor,
    ) -> None:
        """A residual well within tolerance is reported as converged."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.has_converged(converged_residual, rhs_norm=1.0) is True

    def test_unconverged_residual_fails_criterion(
        self,
        unconverged_residual: torch.Tensor,
    ) -> None:
        """A residual well outside tolerance is reported as not converged."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.has_converged(unconverged_residual, rhs_norm=1.0) is False

    def test_nan_residual_never_converges(self, nan_residual: torch.Tensor) -> None:
        """A non-finite residual norm is never reported as converged."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.has_converged(nan_residual, rhs_norm=1.0) is False

    def test_nan_rhs_norm_never_converges(
        self,
        converged_residual: torch.Tensor,
    ) -> None:
        """A non-finite RHS norm is never reported as converged, even with a tiny residual."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        assert criterion.has_converged(converged_residual, rhs_norm=float("nan")) is False

    def test_injected_energy_norm_changes_convergence_decision(
        self,
        energy_norm_diag_entries: torch.Tensor,
        ones_vector_2: torch.Tensor,
    ) -> None:
        """Injecting ``energy_norm`` (not the default L2 norm) changes the magnitude used."""
        criterion = CombinedToleranceCriterion(
            rtol=1e-6, atol=1e-14, norm=energy_norm(energy_norm_diag_entries)
        )
        # ||[1, 1]||_A = sqrt(10) ~= 3.16, well above rtol*||b|| for rhs_norm=1.
        assert criterion.has_converged(ones_vector_2, rhs_norm=1.0) is False

    def test_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        import dataclasses

        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        with pytest.raises(dataclasses.FrozenInstanceError):
            criterion.rtol = 1e-3  # ty: ignore[invalid-assignment]


class TestHasConvergedFromNorm:
    """Tests for ``has_converged_from_norm`` with tensor inputs."""

    def test_converged_with_tensor_residual_norm(self) -> None:
        """Tensor residual norm within tolerance reports converged."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        residual_norm = torch.tensor(1e-8)
        rhs_norm = torch.tensor(1.0)
        assert criterion.has_converged_from_norm(residual_norm, rhs_norm) is True

    def test_not_converged_with_tensor_residual_norm(self) -> None:
        """Tensor residual norm outside tolerance reports not converged."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        residual_norm = torch.tensor(1e-4)
        rhs_norm = torch.tensor(1.0)
        assert criterion.has_converged_from_norm(residual_norm, rhs_norm) is False

    def test_nan_residual_with_tensors_never_converges(self) -> None:
        """Non-finite tensor residual norm never converges."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        residual_norm = torch.tensor(float("nan"))
        rhs_norm = torch.tensor(1.0)
        assert criterion.has_converged_from_norm(residual_norm, rhs_norm) is False

    def test_nan_rhs_norm_with_tensors_never_converges(self) -> None:
        """Non-finite tensor rhs_norm never converges."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        residual_norm = torch.tensor(1e-8)
        rhs_norm = torch.tensor(float("nan"))
        assert criterion.has_converged_from_norm(residual_norm, rhs_norm) is False

    def test_mixed_tensor_and_float_inputs(self) -> None:
        """Mixed tensor and float inputs work correctly."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        # tensor residual, float rhs
        assert criterion.has_converged_from_norm(torch.tensor(1e-8), 1.0) is True
        # float residual, tensor rhs
        assert criterion.has_converged_from_norm(1e-8, torch.tensor(1.0)) is True

    def test_mixed_inputs_do_not_downcast_to_torchs_float32_default(self) -> None:
        """A mixed float/tensor call must not silently lose precision.

        ``torch.as_tensor(a_python_float)`` with no explicit dtype falls back
        to torch's ambient default dtype (float32) unless the caller matches
        it to the other, genuinely-tensor operand's dtype — this repo never
        sets a global default dtype, so a naive conversion would blur exactly
        the boundary this comparison depends on. ``1e-10`` relative
        difference is far below float32's ~1.19e-7 relative precision (would
        round away to equality, reporting ``True``) but easily representable
        in float64 (correctly ``False``).
        """
        criterion = CombinedToleranceCriterion(rtol=1.0, atol=0.0)
        rhs_norm = torch.tensor(1.0, dtype=torch.float64)
        residual_norm = 1.0 * (1.0 + 1e-10)  # strictly above threshold=1.0 in float64

        assert criterion.has_converged_from_norm(residual_norm, rhs_norm) is False

    def test_exactly_one_bool_sync_with_tensor_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly one bool() call when both inputs are tensors."""
        criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        residual_norm = torch.tensor(1e-8)
        rhs_norm = torch.tensor(1.0)

        bool_call_count = 0
        original_bool = torch.Tensor.__bool__

        def counting_bool(self: torch.Tensor) -> bool:  # type: ignore[no-untyped-def]
            nonlocal bool_call_count
            bool_call_count += 1
            return original_bool(self)  # type: ignore[return-value]

        monkeypatch.setattr(torch.Tensor, "__bool__", counting_bool)

        result = criterion.has_converged_from_norm(residual_norm, rhs_norm)
        assert result is True
        assert bool_call_count == 1


class TestConvergenceParityWithScipyReference:
    """Cross-checks the torch criterion against the frozen numpy reference.

    ``tests/support/scipy_reference/strategies/convergence.py`` is the
    Stage 0 frozen snapshot of the reference's numpy implementation. The
    torch port must be mathematically identical - same formula, only the
    tensor type differs - since scipy-equivalence benchmarks later in the
    migration depend on this criterion being exactly right.
    """

    @pytest.mark.parametrize(
        ("residual_values", "rhs_norm"),
        [
            ([1e-8, 0.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1e-8, 0.0], 1e-12),
            ([0.5, 0.5, 0.5], 2.0),
        ],
    )
    def test_has_converged_matches_reference(
        self,
        residual_values: list[float],
        rhs_norm: float,
        torch_dtype: torch.dtype,
    ) -> None:
        """The torch and numpy criteria agree on convergence for the same inputs."""
        from tests.support.scipy_reference.strategies.convergence import (
            CombinedToleranceCriterion as ReferenceCriterion,
        )

        torch_criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        reference_criterion = ReferenceCriterion(rtol=1e-6, atol=1e-14)

        torch_residual = torch.tensor(residual_values, dtype=torch_dtype)
        numpy_residual = np.asarray(residual_values, dtype=np.float64)

        assert torch_criterion.has_converged(
            torch_residual, rhs_norm
        ) == reference_criterion.has_converged(numpy_residual, rhs_norm)

    def test_threshold_matches_reference(self) -> None:
        """The torch and numpy criteria compute the identical threshold formula."""
        from tests.support.scipy_reference.strategies.convergence import (
            CombinedToleranceCriterion as ReferenceCriterion,
        )

        torch_criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        reference_criterion = ReferenceCriterion(rtol=1e-6, atol=1e-14)

        for rhs_norm in (1.0, 1e-12, 1e6):
            assert torch_criterion.threshold(rhs_norm) == reference_criterion.threshold(rhs_norm)
