"""Spectral damping rules for weighted Jacobi: relaxation ``1/rho`` and prolongation ``(4/3)/rho``.

``rho = rho(D^-1 A)`` is estimated once per matrix object (seeded Arnoldi,
cached) and shared by the relaxation and the prolongator smoothing of the
same level. A float still fixes omega explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    JacobiSmoother,
    VCycleAMG,
    WCycleAMG,
)
from torchalg.preconditioners.implementations.amg import _jacobi_omega as jacobi_omega_module
from torchalg.preconditioners.implementations.amg._jacobi_omega import (
    PROLONGATION_NOMINAL,
    RELAXATION_NOMINAL,
    jacobi_omega,
    jacobi_spectral_radius,
)
from torchalg.preconditioners.implementations.pod import POD2GPreconditioner
from torchalg.preconditioners.implementations.pod.weighting import (
    apply_jacobi_damping,
    smoother_persistence_scales,
)


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


@pytest.fixture
def exact_radius(poisson_matrix: torch.Tensor) -> float:
    """Exact rho(D^-1 A) of the Poisson matrix."""
    scaled = poisson_matrix / torch.diagonal(poisson_matrix).unsqueeze(1)
    return float(torch.linalg.eigvals(scaled).abs().max())


@pytest.fixture
def inference_poisson_matrix(poisson_matrix: torch.Tensor) -> torch.Tensor:
    """Poisson matrix created under inference mode, without a version counter."""
    with torch.inference_mode():
        return poisson_matrix.clone()


@pytest.fixture
def high_radius_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """SPD ``0.1 I + 0.9 ones``: rho(D^-1 A) ~ 9, far above 3, so a fixed omega=0.67 diverges."""
    n = 10
    return 0.1 * torch.eye(n, dtype=torch_dtype) + 0.9 * torch.ones(n, n, dtype=torch_dtype)


@pytest.fixture
def start_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random vector, length 64."""
    return torch.randn(64, generator=torch.Generator().manual_seed(3), dtype=torch_dtype)


@pytest.fixture
def estimator_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts calls of the Arnoldi estimator used by the omega rules."""
    calls: list[int] = []
    original = jacobi_omega_module.approximate_spectral_radius

    def counting(matrix: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        calls.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(jacobi_omega_module, "approximate_spectral_radius", counting)
    return calls


class TestSpectralRadius:
    def test_estimate_is_close_to_exact(
        self, poisson_matrix: torch.Tensor, exact_radius: float
    ) -> None:
        assert jacobi_spectral_radius(poisson_matrix) == pytest.approx(exact_radius, rel=0.02)

    def test_is_deterministic_across_matrix_copies(self, poisson_matrix: torch.Tensor) -> None:
        assert jacobi_spectral_radius(poisson_matrix.clone()) == jacobi_spectral_radius(
            poisson_matrix.clone()
        )

    def test_is_cached_per_matrix_object(
        self, poisson_matrix: torch.Tensor, estimator_calls: list[int]
    ) -> None:
        fresh = poisson_matrix.clone()
        jacobi_spectral_radius(fresh)
        jacobi_spectral_radius(fresh)
        assert len(estimator_calls) == 1

    def test_is_cached_for_inference_tensor(
        self, inference_poisson_matrix: torch.Tensor, estimator_calls: list[int]
    ) -> None:
        jacobi_spectral_radius(inference_poisson_matrix)
        jacobi_spectral_radius(inference_poisson_matrix)
        assert len(estimator_calls) == 1


class TestRules:
    def test_explicit_value_wins(self, poisson_matrix: torch.Tensor) -> None:
        assert jacobi_omega(poisson_matrix, RELAXATION_NOMINAL, 0.5) == 0.5

    def test_nominals_are_pyamgs(self) -> None:
        assert (RELAXATION_NOMINAL, PROLONGATION_NOMINAL) == (1.0, 4.0 / 3.0)

    def test_none_divides_nominal_by_rho(self, poisson_matrix: torch.Tensor) -> None:
        rho = jacobi_spectral_radius(poisson_matrix)
        assert jacobi_omega(poisson_matrix, 4.0 / 3.0, None) == pytest.approx(4.0 / 3.0 / rho)


class TestJacobiSmoother:
    def test_default_is_the_relaxation_rule(
        self, poisson_matrix: torch.Tensor, start_vector: torch.Tensor
    ) -> None:
        rho = jacobi_spectral_radius(poisson_matrix)
        zero = torch.zeros_like(start_vector)
        expected = JacobiSmoother(omega=1.0 / rho).smooth(poisson_matrix, zero, start_vector, 3)
        assert torch.allclose(
            JacobiSmoother().smooth(poisson_matrix, zero, start_vector, 3), expected
        )

    def test_default_converges_where_a_fixed_omega_diverges(
        self, high_radius_matrix: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        x = torch.ones(10, dtype=torch_dtype)
        zero = torch.zeros(10, dtype=torch_dtype)
        fixed = JacobiSmoother(omega=0.67).smooth(high_radius_matrix, zero, x, 30)
        spectral = JacobiSmoother().smooth(high_radius_matrix, zero, x, 30)
        assert torch.linalg.norm(fixed) > 1e3
        assert torch.linalg.norm(spectral) < torch.linalg.norm(x)


class TestProlongationSmoothing:
    def test_default_is_the_prolongation_rule(self, poisson_matrix: torch.Tensor) -> None:
        rho = jacobi_spectral_radius(poisson_matrix)
        _, expected = AggregationCoarsening(omega=(4.0 / 3.0) / rho).build_transfer(poisson_matrix)
        _, ours = AggregationCoarsening().build_transfer(poisson_matrix)
        residual = torch.ones(ours.restrict(torch.ones(64, dtype=poisson_matrix.dtype)).shape[0])
        assert torch.allclose(
            ours.prolongate(residual.to(poisson_matrix.dtype)),
            expected.prolongate(residual.to(poisson_matrix.dtype)),
        )


class TestPresets:
    @pytest.mark.parametrize("preset", [VCycleAMG, WCycleAMG])
    def test_split_parameters_replace_the_shared_omega(
        self, preset: type, poisson_matrix: torch.Tensor
    ) -> None:
        with pytest.raises(TypeError):
            preset(poisson_matrix, omega=0.67)
        explicit = preset(poisson_matrix, smoother_omega=0.67, prolongation_omega=0.67)
        default = preset(poisson_matrix)
        residual = torch.ones(64, dtype=poisson_matrix.dtype)
        assert not torch.allclose(explicit.apply(residual), default.apply(residual))

    def test_estimation_is_shared_and_not_repeated_per_apply(
        self, poisson_matrix: torch.Tensor, estimator_calls: list[int]
    ) -> None:
        precond = VCycleAMG(poisson_matrix.clone(), n_levels=3)
        residual = torch.ones(64, dtype=poisson_matrix.dtype)
        precond.apply(residual)
        after_first = len(estimator_calls)
        for _ in range(3):
            precond.apply(residual)
        assert len(estimator_calls) == after_first
        assert after_first <= 3

    def test_pod_preset_uses_smoother_omega(
        self, poisson_matrix: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        snapshots = torch.randn(
            5, 64, generator=torch.Generator().manual_seed(4), dtype=torch_dtype
        )
        with pytest.raises(TypeError):
            POD2GPreconditioner(poisson_matrix, snapshots, rank=3, omega=0.67)  # ty: ignore[unknown-argument]
        residual = torch.ones(64, dtype=torch_dtype)
        explicit = POD2GPreconditioner(poisson_matrix, snapshots, rank=3, smoother_omega=0.67)
        default = POD2GPreconditioner(poisson_matrix, snapshots, rank=3)
        assert not torch.allclose(explicit.apply(residual), default.apply(residual))


class TestPodWeighting:
    def test_damping_default_is_the_relaxation_rule(
        self, poisson_matrix: torch.Tensor, start_vector: torch.Tensor
    ) -> None:
        rho = jacobi_spectral_radius(poisson_matrix)
        vectors = start_vector.unsqueeze(0)
        assert torch.allclose(
            apply_jacobi_damping(vectors, poisson_matrix, steps=4),
            apply_jacobi_damping(vectors, poisson_matrix, omega=1.0 / rho, steps=4),
        )

    def test_persistence_scales_default_is_the_relaxation_rule(
        self, poisson_matrix: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        rho = jacobi_spectral_radius(poisson_matrix)
        snapshots = torch.randn(
            6, 64, generator=torch.Generator().manual_seed(5), dtype=torch_dtype
        )
        assert torch.allclose(
            smoother_persistence_scales(snapshots, poisson_matrix, steps=3),
            smoother_persistence_scales(snapshots, poisson_matrix, omega=1.0 / rho, steps=3),
        )
