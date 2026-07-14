"""Tests for ``torchalg.models.config``."""

from __future__ import annotations

import dataclasses

import pytest

from torchalg.models.config import SolverConfig, SolverParams


class TestSolverConfig:
    """Tests for ``SolverConfig``."""

    def test_required_fields_are_set(self) -> None:
        """Constructing with only required fields preserves their values."""
        config = SolverConfig(
            algorithm="preconditioned_cg",
            rtol=1e-6,
            atol=1e-14,
            maxiter=1000,
            trace_mode="minimal",
        )
        assert config.algorithm == "preconditioned_cg"
        assert config.trace_mode == "minimal"

    def test_optional_fields_default(self) -> None:
        """Unset optional fields take their documented defaults."""
        config = SolverConfig(
            algorithm="flexible_cg",
            rtol=1e-6,
            atol=1e-14,
            maxiter=1000,
            trace_mode="disabled",
        )
        assert config.m_max is None
        assert config.preconditioner is None
        assert config.extra_params == {}

    def test_each_config_gets_its_own_extra_params_dict(self) -> None:
        """``extra_params`` uses ``default_factory``, not a shared mutable default."""
        first = SolverConfig(algorithm="a", rtol=1e-6, atol=1e-14, maxiter=10, trace_mode="minimal")
        second = SolverConfig(
            algorithm="b", rtol=1e-6, atol=1e-14, maxiter=10, trace_mode="minimal"
        )
        assert first.extra_params is not second.extra_params

    def test_extra_params_are_immutable(self) -> None:
        """Nested extra params cannot mutate a frozen config after construction."""
        config = SolverConfig(
            algorithm="a",
            rtol=1e-6,
            atol=1e-14,
            maxiter=10,
            trace_mode="minimal",
            extra_params={"restart": 8},
        )

        with pytest.raises(TypeError):
            config.extra_params["restart"] = 4  # ty: ignore[invalid-assignment]

    def test_extra_params_are_copied_from_input(self) -> None:
        """External mutation of the source mapping does not affect the config."""
        extra_params = {"restart": 8}
        config = SolverConfig(
            algorithm="a",
            rtol=1e-6,
            atol=1e-14,
            maxiter=10,
            trace_mode="minimal",
            extra_params=extra_params,
        )

        extra_params["restart"] = 4

        assert config.extra_params["restart"] == 8

    def test_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        config = SolverConfig(
            algorithm="a", rtol=1e-6, atol=1e-14, maxiter=10, trace_mode="minimal"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.maxiter = 20  # ty: ignore[invalid-assignment]


class TestSolverParams:
    """Tests for ``SolverParams``."""

    def test_fields_are_preserved(self) -> None:
        """Constructed fields are accessible without transformation."""
        params = SolverParams(
            rtol=1e-6,
            atol=1e-14,
            max_iterations=500,
            stopping_criterion="residual_norm",
            m_max=10,
            breakdown_tol=1e-14,
        )
        assert params.max_iterations == 500
        assert params.stopping_criterion == "residual_norm"

    def test_breakdown_tol_accepts_none(self) -> None:
        """``breakdown_tol`` can be explicitly disabled via ``None``."""
        params = SolverParams(
            rtol=1e-6,
            atol=1e-14,
            max_iterations=500,
            stopping_criterion="fixed_iterations",
            m_max=10,
            breakdown_tol=None,
        )
        assert params.breakdown_tol is None

    def test_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        params = SolverParams(
            rtol=1e-6,
            atol=1e-14,
            max_iterations=500,
            stopping_criterion="residual_norm",
            m_max=10,
            breakdown_tol=1e-14,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            params.max_iterations = 100  # ty: ignore[invalid-assignment]
