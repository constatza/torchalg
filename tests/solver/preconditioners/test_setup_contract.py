"""Regression tests for the uniform two-phase ``setup()``/``apply()`` contract.

Covers the base-level guarantee every ``Preconditioner`` shares (see
``torchalg.preconditioners.base.Preconditioner``'s docstring): ``apply()``
raises ``PreconditionerNotReadyError`` before ``setup()`` has run, and a
built hierarchy/operator is not rebuilt by a later ``apply()`` call once
``setup()`` has run once.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.multigrid import AMGPreconditioner
from torchalg.preconditioners.base import PreconditionerNotReadyError
from torchalg.preconditioners.implementations.amg import AggregationCoarsening, JacobiSmoother
from torchalg.preconditioners.implementations.amg.cycle import VCycle
from torchalg.preconditioners.implementations.callable import CallablePreconditioner
from torchalg.preconditioners.implementations.ic0 import IC0Preconditioner
from torchalg.preconditioners.implementations.identity import Identity
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner
from torchalg.preconditioners.implementations.scheduled import ScheduledPreconditioner


def test_jacobi_apply_before_setup_raises(well_conditioned_matrix: torch.Tensor) -> None:
    """``LinearPreconditioner``-derived ``apply()`` requires ``setup()`` first."""
    precond = JacobiPreconditioner()
    with pytest.raises(PreconditionerNotReadyError):
        precond.apply(
            torch.ones(well_conditioned_matrix.shape[0], dtype=well_conditioned_matrix.dtype)
        )


def test_jacobi_setup_builds_operator_once(well_conditioned_matrix: torch.Tensor) -> None:
    """Calling ``apply()`` twice after one ``setup()`` does not recompute the operator."""
    precond = JacobiPreconditioner()
    precond.setup(well_conditioned_matrix)
    operator = precond.inv_diag

    residual = torch.ones(well_conditioned_matrix.shape[0], dtype=well_conditioned_matrix.dtype)
    precond.apply(residual)
    precond.apply(residual)

    assert precond.inv_diag is operator


def test_amg_apply_before_setup_raises(poisson_1d: torch.Tensor) -> None:
    """``AMGPreconditioner.apply()`` requires ``setup()`` first."""
    precond = AMGPreconditioner(
        coarsening=AggregationCoarsening(),
        cycle=VCycle(smoother=JacobiSmoother()),
    )
    with pytest.raises(PreconditionerNotReadyError):
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))


def test_amg_setup_builds_hierarchy_once(poisson_1d: torch.Tensor) -> None:
    """Calling ``apply()`` twice after one ``setup()`` does not rebuild the hierarchy."""
    precond = AMGPreconditioner(
        coarsening=AggregationCoarsening(),
        cycle=VCycle(smoother=JacobiSmoother()),
    )
    precond.setup(poisson_1d)
    hierarchy = precond._hierarchy

    residual = torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype)
    precond.apply(residual)
    precond.apply(residual)

    assert precond._hierarchy is hierarchy


def test_setup_returns_self_for_chaining(well_conditioned_matrix: torch.Tensor) -> None:
    """``setup()`` returns ``self``, matching ``nn.Module.to()``'s chaining convention."""
    precond = JacobiPreconditioner()
    assert precond.setup(well_conditioned_matrix) is precond


class TestReadinessIsDerivedFromState:
    """``is_ready`` reads the prepared state; no flag is set by ``setup()``."""

    def test_matrix_setup_preconditioner_is_unready_until_setup(
        self, well_conditioned_matrix: torch.Tensor
    ) -> None:
        precond = JacobiPreconditioner()
        assert not precond.is_ready

        precond.setup(well_conditioned_matrix)

        assert precond.is_ready

    def test_amg_readiness_follows_its_prepared_hierarchy(self, poisson_1d: torch.Tensor) -> None:
        precond = AMGPreconditioner(
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        assert not precond.is_ready

        precond.setup(poisson_1d)

        assert precond.is_ready

    def test_static_preconditioners_are_ready_at_construction(self) -> None:
        assert Identity().is_ready
        assert CallablePreconditioner(lambda residual: residual).is_ready

    def test_failed_setup_keeps_the_previous_prepared_state(
        self, well_conditioned_matrix: torch.Tensor, non_spd_matrix_2x2_torch: torch.Tensor
    ) -> None:
        precond = IC0Preconditioner().setup(well_conditioned_matrix)
        factor = precond._operator
        residual = torch.ones(well_conditioned_matrix.shape[0], dtype=well_conditioned_matrix.dtype)
        expected = precond.apply(residual)

        with pytest.raises(ValueError, match="pivot"):
            precond.setup(non_spd_matrix_2x2_torch)

        assert precond.is_ready
        assert precond._operator is factor
        assert torch.equal(precond.apply(residual), expected)

    def test_failed_first_setup_leaves_the_preconditioner_unready(
        self, non_spd_matrix_2x2_torch: torch.Tensor
    ) -> None:
        precond = IC0Preconditioner()

        with pytest.raises(ValueError, match="pivot"):
            precond.setup(non_spd_matrix_2x2_torch)

        assert not precond.is_ready

    def test_scheduled_readiness_follows_both_components(
        self, well_conditioned_matrix: torch.Tensor
    ) -> None:
        scheduled = ScheduledPreconditioner(primary=JacobiPreconditioner())
        assert not scheduled.is_ready

        scheduled.setup(well_conditioned_matrix)

        assert scheduled.is_ready
