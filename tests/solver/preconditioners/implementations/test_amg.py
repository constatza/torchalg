"""Tests for AMG preconditioner components.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/implementations/test_amg.py``), adapted from
numpy to torch tensors, using the shared ``poisson_1d``/``poisson_rhs``/
``small_spd_amg``/``zero_iteration_context`` fixtures
(``tests/solver/preconditioners/conftest.py``,
``tests/conftest.py::poisson_1d_factory``) instead of inline construction.

Deviations from the reference:
    - ``TestAMGConfigParsing`` (TOML/Pydantic config parsing) is not ported:
      the TOML-driven preconditioner factory is an explicit scope trim (see
      ``docs/plan.md``'s "Scope trim" section) - not part of this migration.
    - ``TestNeuralPreconditionerFallback`` is not ported here: it exercises
      ``NeuralPreconditioner``/``ExtraInputPredictorPort``, which land in
      Stage 7 (``ports.py``/``neural.py``), not Stage 5.
    - Tests requiring a solver (``flexible_cg``) are active now that the
      solver stage has landed.
    - ``TestNeuralStubs`` is new (the reference has no test for these two
      stubs at all): a small regression suite confirming
      ``NeuralCoarseningStrategy``/``NeuralTransferOperator`` are ported
      as-is and stay a deliberate, tracked gap (see ``docs/plan.md``'s
      architectural-corrections section) rather than silently rotting.
    - Class/fixture names follow the Stage 5 renames documented in
      ``implementations/amg/__init__.py``: ``SparseAggregationCoarsening`` ->
      ``AggregationCoarsening``, ``SparseTransferOperator`` ->
      ``DenseTransferOperator``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.base import PreconditionerContext
from torchalg.preconditioners.implementations import AMGPreconditioner, Identity
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    DenseTransferOperator,
    JacobiSmoother,
    MultigridHierarchy,
    NeuralCoarseningStrategy,
    NeuralTransferOperator,
    SmootherBase,
    TargetDimensionCoarsening,
    VCycle,
    VCycleAMG,
    WCycle,
    WCycleAMG,
)
from torchalg.preconditioners.ports import ExtraInputPredictorPort

# ---------------------------------------------------------------------------
# Fixtures (AMG-specific: transfer, presets)
# ---------------------------------------------------------------------------


@pytest.fixture
def amg_preconditioner(poisson_1d: torch.Tensor) -> AMGPreconditioner:
    """Classical AMGPreconditioner on the 1D Poisson matrix.

    Args:
        poisson_1d: 20x20 Poisson matrix fixture.

    Returns:
        AMGPreconditioner instance with VCycle + JacobiSmoother.
    """
    return AMGPreconditioner(
        poisson_1d,
        coarsening=AggregationCoarsening(),
        cycle=VCycle(smoother=JacobiSmoother()),
        n_levels=2,
    )


@pytest.fixture
def dense_transfer(torch_dtype: torch.dtype) -> DenseTransferOperator:
    """2:1 aggregation transfer operator (8 fine -> 4 coarse).

    Returns:
        DenseTransferOperator with piecewise-constant dense P.
    """
    rows = torch.arange(8)
    cols = rows // 2
    P = torch.zeros(8, 4, dtype=torch_dtype)
    P[rows, cols] = 1.0
    return DenseTransferOperator(P)


@pytest.fixture(
    params=[
        pytest.param("VCycleAMG", id="VCycleAMG"),
        pytest.param("WCycleAMG", id="WCycleAMG"),
    ]
)
def amg_preset_class(request: pytest.FixtureRequest) -> type[VCycleAMG] | type[WCycleAMG]:
    """Parametrize tests over both preset AMG variant classes.

    Args:
        request: Pytest fixture request carrying the class name.

    Returns:
        The preset AMG class (VCycleAMG or WCycleAMG). Typed as the union of
        the two concrete presets (not the ``AMGPreconditioner`` base) so
        ``ty`` resolves each preset's own ``__init__`` override (which needs
        only ``matrix``/``n_levels``/...) instead of the base class's
        ``__init__`` (which additionally requires ``coarsening``/``cycle``).
    """
    return {"VCycleAMG": VCycleAMG, "WCycleAMG": WCycleAMG}[request.param]


# ---------------------------------------------------------------------------
# JacobiSmoother
# ---------------------------------------------------------------------------


class TestJacobiSmoother:
    def test_reduces_residual(self, small_spd_amg: torch.Tensor) -> None:
        """Smoothing should reduce the residual norm."""
        smoother = JacobiSmoother(omega=0.67)
        rhs = torch.ones(6, dtype=small_spd_amg.dtype)
        x = torch.zeros(6, dtype=small_spd_amg.dtype)
        x_smooth = smoother.smooth(small_spd_amg, rhs, x, steps=5)
        r_before = torch.linalg.norm(rhs - small_spd_amg @ x)
        r_after = torch.linalg.norm(rhs - small_spd_amg @ x_smooth)
        assert r_after < r_before

    def test_output_shape(self, small_spd_amg: torch.Tensor) -> None:
        """Output shape must match input shape."""
        smoother = JacobiSmoother()
        rhs = torch.ones(6, dtype=small_spd_amg.dtype)
        x = smoother.smooth(small_spd_amg, rhs, torch.zeros(6, dtype=small_spd_amg.dtype), steps=1)
        assert x.shape == (6,)

    def test_zero_steps_returns_copy(self, small_spd_amg: torch.Tensor) -> None:
        """Zero smoothing steps should return the initial iterate unchanged."""
        smoother = JacobiSmoother()
        x0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=small_spd_amg.dtype)
        rhs = torch.ones(6, dtype=small_spd_amg.dtype)
        x = smoother.smooth(small_spd_amg, rhs, x0, steps=0)
        torch.testing.assert_close(x, x0)


# ---------------------------------------------------------------------------
# DenseTransferOperator
# ---------------------------------------------------------------------------


class TestDenseTransferOperator:
    def test_prolongate_shape(self, dense_transfer: DenseTransferOperator) -> None:
        """Prolongate maps coarse (4) -> fine (8)."""
        coarse = torch.ones(4, dtype=torch.float64)
        fine = dense_transfer.prolongate(coarse)
        assert fine.shape == (8,)

    def test_restrict_shape(self, dense_transfer: DenseTransferOperator) -> None:
        """Restrict maps fine (8) -> coarse (4)."""
        fine = torch.ones(8, dtype=torch.float64)
        coarse = dense_transfer.restrict(fine)
        assert coarse.shape == (4,)

    def test_restrict_prolongate_consistency(self, dense_transfer: DenseTransferOperator) -> None:
        """Restriction should be Galerkin: R = P^T, so restrict(ones) sums pairs."""
        fine = torch.ones(8, dtype=torch.float64)
        coarse = dense_transfer.restrict(fine)
        # Each coarse dof aggregates 2 fine dofs -> sum = 2
        torch.testing.assert_close(coarse, torch.full((4,), 2.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# AggregationCoarsening
# ---------------------------------------------------------------------------


class TestAggregationCoarsening:
    def test_returns_smaller_matrix(self, poisson_1d: torch.Tensor) -> None:
        """Coarse matrix must have fewer dofs than fine matrix."""
        a_c, _ = AggregationCoarsening().build_transfer(poisson_1d)
        assert a_c.shape[0] < poisson_1d.shape[0]

    def test_coarse_matrix_spd(self, poisson_1d: torch.Tensor) -> None:
        """Galerkin coarse matrix (P^T A P) must be SPD for SPD fine matrix."""
        a_c, _ = AggregationCoarsening().build_transfer(poisson_1d)
        eigenvalues = torch.linalg.eigvalsh(a_c)
        assert torch.all(eigenvalues > 0)

    def test_transfer_shapes_consistent(self, poisson_1d: torch.Tensor) -> None:
        """Prolongate output (fine) and restrict output (coarse) sizes must match."""
        a_c, transfer = AggregationCoarsening().build_transfer(poisson_1d)
        n_fine = poisson_1d.shape[0]
        n_coarse = a_c.shape[0]
        assert transfer.prolongate(torch.ones(n_coarse, dtype=poisson_1d.dtype)).shape == (n_fine,)
        assert transfer.restrict(torch.ones(n_fine, dtype=poisson_1d.dtype)).shape == (n_coarse,)


# ---------------------------------------------------------------------------
# TargetDimensionCoarsening
# ---------------------------------------------------------------------------


class TestTargetDimensionCoarsening:
    """`poisson_1d` (20x20, uniform 1D stencil) has two theta plateaus:
    c=10 for theta in roughly [0.01, 0.5], c=20 (fully degenerate, no
    coarsening) for theta >= 0.6 — used directly as the two targets below.
    """

    def test_returns_closest_achievable_coarse_dimension(self, poisson_1d: torch.Tensor) -> None:
        """Target within the low plateau resolves to that plateau's realized dimension."""
        coarsening = TargetDimensionCoarsening(
            target_coarse_dim=10, theta_min=0.01, theta_max=0.99, step=0.01, omega=0.67
        )
        a_c, _ = coarsening.build_transfer(poisson_1d)
        assert a_c.shape[0] == 10

    def test_respects_theta_bounds_even_when_target_is_unreachable(
        self, poisson_1d: torch.Tensor
    ) -> None:
        """A target only reachable outside [theta_min, theta_max] still stays in-bounds."""
        coarsening = TargetDimensionCoarsening(
            target_coarse_dim=20, theta_min=0.01, theta_max=0.5, step=0.01, omega=0.67
        )
        a_c, _ = coarsening.build_transfer(poisson_1d)
        assert a_c.shape[0] == 10
        assert coarsening._theta is not None
        assert 0.01 <= coarsening._theta <= 0.5

    def test_caches_realized_theta_and_dimension_after_build(
        self, poisson_1d: torch.Tensor
    ) -> None:
        """Winning theta/dimension are readable afterward without rebuilding."""
        coarsening = TargetDimensionCoarsening(
            target_coarse_dim=20, theta_min=0.01, theta_max=0.99, step=0.01, omega=0.67
        )
        assert coarsening._theta is None
        assert coarsening._realized_coarse_dim is None
        a_c, _ = coarsening.build_transfer(poisson_1d)
        assert coarsening._realized_coarse_dim == a_c.shape[0] == 20

    def test_transfer_shapes_consistent(self, poisson_1d: torch.Tensor) -> None:
        """Prolongate output (fine) and restrict output (coarse) sizes must match."""
        coarsening = TargetDimensionCoarsening(
            target_coarse_dim=10, theta_min=0.01, theta_max=0.99, step=0.01, omega=0.67
        )
        a_c, transfer = coarsening.build_transfer(poisson_1d)
        n_fine = poisson_1d.shape[0]
        n_coarse = a_c.shape[0]
        assert transfer.prolongate(torch.ones(n_coarse, dtype=poisson_1d.dtype)).shape == (n_fine,)
        assert transfer.restrict(torch.ones(n_fine, dtype=poisson_1d.dtype)).shape == (n_coarse,)


# ---------------------------------------------------------------------------
# VCycle
# ---------------------------------------------------------------------------


class TestVCycle:
    def test_output_shape(self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor) -> None:
        """V-cycle output must match rhs shape."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_output_dtype_matches_input(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """V-cycle output dtype must match the working dtype (no float64 hardcode)."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        z = precond.apply(poisson_rhs)
        assert z.dtype == poisson_1d.dtype


# ---------------------------------------------------------------------------
# AMGPreconditioner
# ---------------------------------------------------------------------------


class TestAMGPreconditioner:
    def test_apply_returns_correct_shape(
        self, amg_preconditioner: AMGPreconditioner, poisson_rhs: torch.Tensor
    ) -> None:
        """apply() output must match residual shape."""
        z = amg_preconditioner.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_accepts_context(
        self,
        amg_preconditioner: AMGPreconditioner,
        poisson_rhs: torch.Tensor,
        zero_iteration_context: PreconditionerContext,
    ) -> None:
        """apply() must accept PreconditionerContext without error."""
        z = amg_preconditioner.apply(poisson_rhs, zero_iteration_context)
        assert z.shape == poisson_rhs.shape

    def test_lazy_hierarchy_build(self, poisson_1d: torch.Tensor) -> None:
        """Hierarchy must be None before first apply, populated after."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        assert precond._hierarchy is None
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
        assert precond._hierarchy is not None

    def test_linear_requires_flexible_cg_false(self, poisson_1d: torch.Tensor) -> None:
        """Classical AMG (linear=True) must not request FCG."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            linear=True,
        )
        assert precond.requires_flexible_cg is False

    def test_nonlinear_requires_flexible_cg_true(self, poisson_1d: torch.Tensor) -> None:
        """AMG with linear=False must request FCG (neural AMG case)."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            linear=False,
        )
        assert precond.requires_flexible_cg is True

    def test_extra_input_names_empty_for_classical(
        self, amg_preconditioner: AMGPreconditioner
    ) -> None:
        """Classical AMG needs no extra inputs."""
        assert amg_preconditioner.extra_input_names == ()

    def test_bind_inputs_noop_for_classical(
        self, amg_preconditioner: AMGPreconditioner, poisson_rhs: torch.Tensor
    ) -> None:
        """bind_inputs on classical AMG must not raise and must still work."""
        amg_preconditioner.bind_inputs(positions=torch.zeros(20, dtype=poisson_rhs.dtype))
        z = amg_preconditioner.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_bind_inputs_invalidates_hierarchy(self, poisson_1d: torch.Tensor) -> None:
        """bind_inputs must invalidate the cached hierarchy."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
        assert precond._hierarchy is not None
        precond.bind_inputs()
        assert precond._hierarchy is None

    def test_hierarchy_has_correct_n_levels(self, poisson_1d: torch.Tensor) -> None:
        """Built hierarchy must have exactly n_levels MultigridLevel entries."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
        assert precond._hierarchy is not None
        assert len(precond._hierarchy.levels) == 2

    def test_rejects_single_level_hierarchy(self, poisson_1d: torch.Tensor) -> None:
        """AMG must have at least one coarse level; n_levels=1 is a direct solve."""
        with pytest.raises(ValueError, match="n_levels"):
            AMGPreconditioner(
                poisson_1d,
                coarsening=AggregationCoarsening(),
                cycle=VCycle(smoother=JacobiSmoother()),
                n_levels=1,
            )

    def test_coarsest_level_has_no_transfer(self, poisson_1d: torch.Tensor) -> None:
        """The coarsest level must have transfer=None."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
        assert precond._hierarchy is not None
        assert precond._hierarchy.levels[-1].transfer is None

    def test_hierarchy_invalidated_after_device_dtype_move(self, poisson_1d: torch.Tensor) -> None:
        """Moving the module via ``.to()`` must invalidate the cached hierarchy.

        Otherwise a stale-device/dtype hierarchy could be reused after
        ``.to()`` - see ``amg.py``'s ``_apply`` override docstring for the
        full design reasoning.
        """
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
        assert precond._hierarchy is not None

        moved = precond.to(dtype=torch.float32)
        assert moved._hierarchy is None
        z = moved.apply(torch.ones(poisson_1d.shape[0], dtype=torch.float32))
        assert z.dtype == torch.float32


# ---------------------------------------------------------------------------
# Interface: context always passed (no ContextualPreconditioner split)
# ---------------------------------------------------------------------------


class TestUniformContextInterface:
    def test_all_preconditioners_accept_context(self, poisson_1d: torch.Tensor) -> None:
        """Every preconditioner must accept a PreconditionerContext without error."""
        ctx = PreconditionerContext(iteration=5, residual_norm=0.1, rhs_norm=1.0)
        r = torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype)
        for precond in [
            Identity(),
            JacobiPreconditioner(poisson_1d),
            AMGPreconditioner(
                poisson_1d,
                coarsening=AggregationCoarsening(),
                cycle=VCycle(smoother=JacobiSmoother()),
            ),
        ]:
            z = precond.apply(r, ctx)
            assert z.shape == r.shape


# ---------------------------------------------------------------------------
# SmootherBase hierarchy
# ---------------------------------------------------------------------------


class TestSmootherBase:
    def test_jacobi_smoother_is_smoother_base(self) -> None:
        """JacobiSmoother must be a SmootherBase instance (nominative hierarchy)."""
        assert isinstance(JacobiSmoother(), SmootherBase)

    def test_jacobi_smoother_satisfies_multigrid_smoother_protocol(
        self, small_spd_amg: torch.Tensor
    ) -> None:
        """JacobiSmoother must satisfy the MultigridSmoother protocol (structural typing)."""
        smoother = JacobiSmoother()
        rhs = torch.ones(6, dtype=small_spd_amg.dtype)
        x0 = torch.zeros(6, dtype=small_spd_amg.dtype)
        result = smoother.smooth(small_spd_amg, rhs, x0, steps=1)
        assert result.shape == x0.shape


# ---------------------------------------------------------------------------
# WCycle
# ---------------------------------------------------------------------------


class TestWCycle:
    def test_output_shape(self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor) -> None:
        """W-cycle output must match rhs shape."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=WCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_output_dtype_matches_input(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """W-cycle output dtype must match the working dtype (no float64 hardcode)."""
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=WCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.dtype == poisson_1d.dtype

    def test_wcycle_residual_not_larger_than_vcycle(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """W-cycle must not produce a larger residual than V-cycle on the same rhs.

        Theory: W-cycle does strictly more coarse-grid work (gamma=2 vs
        gamma=1), so its residual after one cycle should be <= that of
        V-cycle for SPD problems (Briggs et al. 2000, Section 3.3).
        """
        smoother = JacobiSmoother()
        coarsening = AggregationCoarsening()

        v_precond = AMGPreconditioner(
            poisson_1d, coarsening=coarsening, cycle=VCycle(smoother), n_levels=2
        )
        w_precond = AMGPreconditioner(
            poisson_1d, coarsening=coarsening, cycle=WCycle(smoother), n_levels=2
        )

        e_v = v_precond.apply(poisson_rhs)
        e_w = w_precond.apply(poisson_rhs)

        res_v = torch.linalg.norm(poisson_rhs - poisson_1d @ e_v)
        res_w = torch.linalg.norm(poisson_rhs - poisson_1d @ e_w)
        assert res_w <= res_v + 1e-10  # W-cycle never worse than V-cycle


# ---------------------------------------------------------------------------
# Preset classes: VCycleAMG and WCycleAMG
# ---------------------------------------------------------------------------


class TestAMGPresets:
    def test_apply_returns_correct_shape(
        self,
        amg_preset_class: type[VCycleAMG] | type[WCycleAMG],
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
    ) -> None:
        """apply() output must match residual shape for both preset classes."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_returns_expected_dtype(
        self,
        amg_preset_class: type[VCycleAMG] | type[WCycleAMG],
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
    ) -> None:
        """apply() output dtype must match the working dtype for both preset classes."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        assert precond.apply(poisson_rhs).dtype == poisson_1d.dtype

    def test_fcg_converges(
        self,
        amg_preset_class: type[VCycleAMG] | type[WCycleAMG],
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
    ) -> None:
        """FCG preconditioned with either preset must converge on 1D Poisson."""
        from torchalg import flexible_cg

        precond = amg_preset_class(poisson_1d, n_levels=2)
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"{amg_preset_class.__name__}+FCG did not converge: {info}"
        torch.testing.assert_close(poisson_1d @ x, poisson_rhs, rtol=1e-6, atol=1e-6)

    def test_spd_preservation(
        self, amg_preset_class: type[VCycleAMG] | type[WCycleAMG], poisson_1d: torch.Tensor
    ) -> None:
        """Preset AMG preconditioner must be SPD when applied to an SPD matrix.

        Theory: V/W-cycle with symmetric Jacobi smoother (equal pre/post steps)
        and Galerkin coarsening produces an SPD preconditioner M when A is SPD
        (Vanek, Mandel & Brezina 1996, Theorem 4.1).

        Verification: assemble M by applying the preconditioner to each canonical
        basis vector, then check symmetry and positive-definiteness via eigenvalues.
        """
        n = poisson_1d.shape[0]
        precond = amg_preset_class(poisson_1d, n_levels=2)
        identity = torch.eye(n, dtype=poisson_1d.dtype)
        m = torch.stack([precond.apply(identity[:, i]) for i in range(n)], dim=1)

        # Symmetry
        torch.testing.assert_close(
            m, m.T, atol=1e-12, rtol=0.0, msg="Preconditioner matrix is not symmetric"
        )

        # Positive definiteness
        eigenvalues = torch.linalg.eigvalsh(m)
        assert torch.all(eigenvalues > 0), (
            f"Non-positive eigenvalue(s): {eigenvalues[eigenvalues <= 0]}"
        )

    def test_requires_flexible_cg_false(
        self, amg_preset_class: type[VCycleAMG] | type[WCycleAMG], poisson_1d: torch.Tensor
    ) -> None:
        """Both presets are linear preconditioners and must not request FCG."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        assert precond.requires_flexible_cg is False

    def test_rejects_single_level_hierarchy(
        self, amg_preset_class: type[VCycleAMG] | type[WCycleAMG], poisson_1d: torch.Tensor
    ) -> None:
        """Preset AMG variants must reject n_levels=1 for two-grid clarity."""
        with pytest.raises(ValueError, match="n_levels"):
            amg_preset_class(poisson_1d, n_levels=1)

    def test_wcycle_iters_not_worse_than_vcycle(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """WCycleAMG must need <= iterations than VCycleAMG on SPD Poisson.

        Theory: W-cycle's extra coarse-grid correction never increases the
        iteration count for SPD problems (Briggs et al. 2000, Section 3.3).
        """
        from torchalg import flexible_cg

        _, info_v = flexible_cg(
            poisson_1d,
            poisson_rhs,
            preconditioner=VCycleAMG(poisson_1d, n_levels=2),
            rtol=1e-8,
        )
        _, info_w = flexible_cg(
            poisson_1d,
            poisson_rhs,
            preconditioner=WCycleAMG(poisson_1d, n_levels=2),
            rtol=1e-8,
        )
        assert info_w.iterations <= info_v.iterations


# ---------------------------------------------------------------------------
# Integration: AMG inside FCG solver
# ---------------------------------------------------------------------------


class TestAMGWithFCGSolver:
    def test_amg_preconditioned_fcg_converges(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """AMG-preconditioned FCG must converge on 1D Poisson."""
        from torchalg import flexible_cg

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"FCG+AMG did not converge: {info}"
        torch.testing.assert_close(poisson_1d @ x, poisson_rhs, rtol=1e-6, atol=1e-6)

    def test_amg_fewer_iters_than_unpreconditioned(
        self, poisson_1d: torch.Tensor, poisson_rhs: torch.Tensor
    ) -> None:
        """AMG preconditioning must reduce iteration count vs. no preconditioning."""
        from torchalg import flexible_cg

        _, info_unprecond = flexible_cg(poisson_1d, poisson_rhs, rtol=1e-8)
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=AggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        _, info_amg = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info_amg.iterations < info_unprecond.iterations


# ---------------------------------------------------------------------------
# Neural stubs: deliberate, tracked gaps (not ported from the reference -
# the reference has no test at all for these two stubs; new for this port).
# ---------------------------------------------------------------------------


class TestNeuralStubs:
    """Regression tests for the deliberately-unimplemented neural AMG extension points.

    ``NeuralCoarseningStrategy``/``NeuralTransferOperator`` are ported as-is,
    matching the reference (see ``docs/plan.md``'s architectural-corrections
    section) - genuine OCP extension points, not dead code. These tests keep
    this a deliberate, tracked gap: if a future change accidentally "fixes"
    ``build_transfer`` without anyone noticing, this test starts failing
    instead of the gap silently rotting.
    """

    def test_neural_coarsening_build_transfer_raises_not_implemented(
        self, poisson_1d: torch.Tensor
    ) -> None:
        """``NeuralCoarseningStrategy.build_transfer`` must still raise ``NotImplementedError``."""
        strategy = NeuralCoarseningStrategy(prolongator=_RecordingPredictor(), restrictor=None)
        with pytest.raises(NotImplementedError):
            strategy.build_transfer(poisson_1d)

    def test_neural_transfer_operator_forwards_to_predictors(self) -> None:
        """``NeuralTransferOperator`` is functional glue, not itself a stub.

        It must forward ``prolongate``/``restrict`` calls to the injected
        predictor objects unchanged - this is the "stub-adjacent" half of
        the pair: not yet reachable in practice (nothing produces a real
        neural P/R today), but not raising ``NotImplementedError`` itself
        either, matching the reference exactly.
        """

        prolongator = _RecordingPredictor()
        restrictor = _RecordingPredictor()
        transfer = NeuralTransferOperator(prolongator=prolongator, restrictor=restrictor)

        coarse = torch.ones(3, dtype=torch.float64)
        fine = torch.ones(6, dtype=torch.float64)
        torch.testing.assert_close(transfer.prolongate(coarse), coarse * 2)
        torch.testing.assert_close(transfer.restrict(fine), fine * 2)
        assert prolongator.calls == [coarse]
        assert restrictor.calls == [fine]


class _RecordingPredictor(ExtraInputPredictorPort):
    """Predictor test double that doubles inputs and records calls."""

    def __init__(self) -> None:
        """Initialize call records."""
        self.calls: list[torch.Tensor] = []

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Declare no required extra inputs."""
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Record the residual and return a doubled tensor."""
        self.calls.append(residual)
        return residual * 2

    def cleanup(self) -> None:
        """No-op cleanup."""


# ---------------------------------------------------------------------------
# MultigridHierarchy sanity (frozen dataclass import check)
# ---------------------------------------------------------------------------


def test_multigrid_hierarchy_is_frozen(poisson_1d: torch.Tensor) -> None:
    """MultigridHierarchy instances must be immutable (frozen dataclass)."""
    precond = AMGPreconditioner(
        poisson_1d,
        coarsening=AggregationCoarsening(),
        cycle=VCycle(smoother=JacobiSmoother()),
    )
    precond.apply(torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype))
    hierarchy = precond._hierarchy
    assert isinstance(hierarchy, MultigridHierarchy)
    with pytest.raises(AttributeError):
        hierarchy.levels = ()  # ty: ignore[invalid-assignment]
