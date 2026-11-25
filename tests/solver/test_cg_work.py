"""Work accounting and equivalence of the CG loop.

Pins what one ``k``-iteration solve costs (preconditioner applies, matvecs,
curvature dot products) and proves the restructured loop reproduces an
independent textbook reference: PCG and FCG to rounding (the fused ``addcmul``
updates may contract a multiply-add, so the last bit can differ), FCG against the
previous per-pair Gram-Schmidt implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.support.dispatch_counter import DispatchCounter
from torchalg import flexible_cg, pcg
from torchalg.base import DEFAULT_RTOL
from torchalg.strategies.orthogonalization import DEFAULT_M_MAX
from torchalg.utils.numerics import stable_dot_product

ITERATIONS = 6
SYSTEM_SIZE = 48
NO_TOLERANCE = 0.0
PCG_ROUNDING_RTOL = 1e-14
"""Allowed relative deviation of PCG residual histories: a few ulp from fused multiply-adds."""
FCG_ROUNDING_RTOL = 1e-12
"""Allowed relative deviation of FCG residual histories from the sequential reference.

Observed: below 1e-15 for ten iterations at any window and for 25 iterations at the
default window. Past that, truncated FCG on this spectrum amplifies a single-ulp
perturbation by about 3e7 (a 2e-16 relative change of ``b`` alone moves the m=3
history by 7e-9), so longer comparisons would test the conditioning, not the code.
"""


@dataclass(slots=True)
class CallCounter:
    """Counts invocations of a wrapped tensor function."""

    function: Callable[[torch.Tensor], torch.Tensor]
    calls: int = 0

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.function(vector)


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    """Seeded SPD matrix with a spread spectrum (slow enough that 6 iterations never converge)."""
    generator = torch.Generator().manual_seed(7)
    factor = torch.randn(SYSTEM_SIZE, SYSTEM_SIZE, generator=generator, dtype=torch.float64)
    scale = torch.logspace(0, 2, SYSTEM_SIZE, dtype=torch.float64)
    return factor @ factor.T / SYSTEM_SIZE + torch.diag(scale)


@pytest.fixture
def rhs(spd_matrix: torch.Tensor) -> torch.Tensor:
    """Seeded right-hand side."""
    generator = torch.Generator().manual_seed(8)
    return torch.randn(spd_matrix.shape[0], generator=generator, dtype=torch.float64)


@pytest.fixture
def initial_guess(spd_matrix: torch.Tensor) -> torch.Tensor:
    """Seeded nonzero initial guess."""
    generator = torch.Generator().manual_seed(9)
    return torch.randn(spd_matrix.shape[0], generator=generator, dtype=torch.float64)


@pytest.fixture
def counting_matvec(spd_matrix: torch.Tensor) -> CallCounter:
    """``v -> A v`` that counts calls."""
    return CallCounter(lambda vector: spd_matrix @ vector)


@pytest.fixture
def counting_jacobi(spd_matrix: torch.Tensor) -> CallCounter:
    """Diagonal preconditioner ``r -> r / diag(A)`` that counts applies."""
    diagonal = spd_matrix.diagonal().clone()
    return CallCounter(lambda residual: residual / diagonal)


def _reference_pcg_residual_norms(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    inverse_diagonal: torch.Tensor,
    iterations: int,
) -> list[float]:
    """Textbook Jacobi-preconditioned CG from a zero guess; returns ``||r_0..r_k||``."""
    r = rhs.clone()
    norms = [float(torch.linalg.norm(r))]
    direction = torch.zeros_like(rhs)
    rw_prev = torch.ones((), dtype=rhs.dtype)
    u = torch.zeros_like(rhs)
    for k in range(iterations):
        w = r * inverse_diagonal
        rw = torch.dot(r, w)
        direction = w.clone() if k == 0 else w + (rw / rw_prev) * direction
        q = matrix @ direction
        alpha = rw / torch.dot(direction, q)
        u = u + alpha * direction
        r = r - alpha * q
        rw_prev = rw
        norms.append(float(torch.linalg.norm(r)))
    return norms


def _reference_fcg_residual_norms(
    matrix: torch.Tensor, rhs: torch.Tensor, m_max: int, iterations: int
) -> list[float]:
    """Previous FCG: tuple history, per-pair ``stable_dot_product`` classical Gram-Schmidt."""
    r = rhs.clone()
    u = torch.zeros_like(rhs)
    norms = [float(torch.linalg.norm(r))]
    directions: list[torch.Tensor] = []
    products: list[torch.Tensor] = []
    for k in range(iterations):
        w = r.clone()
        rw = stable_dot_product(r, w)
        window = min((k % m_max) + 1, len(directions))
        direction = w.clone()
        if window:
            for d_j, q_j in zip(directions[-window:], products[-window:], strict=True):
                coeff = stable_dot_product(w, q_j) / stable_dot_product(d_j, q_j)
                direction = direction - coeff * d_j
        q = matrix @ direction
        alpha = rw / stable_dot_product(direction, q)
        u = u + alpha * direction
        r = r - alpha * q
        directions = (directions + [direction])[-m_max:]
        products = (products + [q])[-m_max:]
        norms.append(float(torch.linalg.norm(r)))
    return norms


class TestWorkPerSolve:
    """A ``k``-iteration solve does the minimum number of applies and matvecs."""

    def test_zero_guess_costs_k_applies_and_k_matvecs(
        self, rhs: torch.Tensor, counting_matvec: CallCounter, counting_jacobi: CallCounter
    ) -> None:
        """No ``M^-1`` or ``A`` product is computed twice from the same vector."""
        _, result = pcg(
            counting_matvec,
            rhs,
            preconditioner=counting_jacobi,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        assert result.iterations == ITERATIONS
        assert counting_jacobi.calls == ITERATIONS
        assert counting_matvec.calls == ITERATIONS

    def test_nonzero_guess_needs_one_extra_matvec_for_the_initial_residual(
        self,
        rhs: torch.Tensor,
        initial_guess: torch.Tensor,
        counting_matvec: CallCounter,
        counting_jacobi: CallCounter,
    ) -> None:
        """``r_0 = b - A x_0`` is the only additional product."""
        pcg(
            counting_matvec,
            rhs,
            initial_guess,
            preconditioner=counting_jacobi,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        assert counting_jacobi.calls == ITERATIONS
        assert counting_matvec.calls == ITERATIONS + 1

    def test_flexible_cg_has_the_same_work_per_iteration(
        self, rhs: torch.Tensor, counting_matvec: CallCounter, counting_jacobi: CallCounter
    ) -> None:
        """FCG shares the loop: same applies and matvecs as PCG."""
        flexible_cg(
            counting_matvec,
            rhs,
            preconditioner=counting_jacobi,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        assert counting_jacobi.calls == ITERATIONS
        assert counting_matvec.calls == ITERATIONS

    def test_each_iteration_computes_two_dot_products(
        self, rhs: torch.Tensor, counting_matvec: CallCounter
    ) -> None:
        """``(r, w)`` once (shared with the direction strategy) and ``(d, q)`` once."""
        counter = DispatchCounter()
        with counter:
            pcg(
                counting_matvec,
                rhs,
                maxiter=ITERATIONS,
                rtol=NO_TOLERANCE,
                atol=NO_TOLERANCE,
            )

        assert counter.ops["aten.dot.default"] == 2 * ITERATIONS


class TestEquivalence:
    """The restructured loop reproduces independent references."""

    def test_pcg_residual_history_matches_textbook_pcg(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """Same operations in the same order, fused updates aside: equal to rounding."""
        inverse_diagonal = 1.0 / spd_matrix.diagonal()
        _, result = pcg(
            spd_matrix,
            rhs,
            preconditioner=lambda residual: residual * inverse_diagonal,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        expected = _reference_pcg_residual_norms(spd_matrix, rhs, inverse_diagonal, ITERATIONS)
        assert result.residual_history_abs is not None
        for observed, reference in zip(result.residual_history_abs, expected, strict=True):
            assert math.isclose(observed, reference, rel_tol=PCG_ROUNDING_RTOL, abs_tol=0.0)

    @pytest.mark.parametrize(
        ("m_max", "iterations"), [(1, 10), (3, 10), (DEFAULT_M_MAX, 10), (DEFAULT_M_MAX, 25)]
    )
    def test_fcg_residual_history_matches_previous_gram_schmidt(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor, m_max: int, iterations: int
    ) -> None:
        """Buffered two-matvec orthogonalization equals the per-pair loop to rounding."""
        _, result = flexible_cg(
            spd_matrix,
            rhs,
            m_max=m_max,
            maxiter=iterations,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        expected = _reference_fcg_residual_norms(spd_matrix, rhs, m_max, iterations)
        assert result.residual_history_abs is not None
        for observed, reference in zip(result.residual_history_abs, expected, strict=True):
            assert math.isclose(observed, reference, rel_tol=FCG_ROUNDING_RTOL, abs_tol=1e-13)

    def test_differentiable_fcg_backpropagates_through_the_buffered_history(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """Autograd must not see the in-place buffer writes; d sum(x)/db = A^-1 1."""
        tracked_rhs = rhs.clone().requires_grad_(True)
        solution, result = flexible_cg(
            spd_matrix,
            tracked_rhs,
            m_max=4,
            maxiter=3 * SYSTEM_SIZE,
            rtol=1e-12,
            atol=1e-14,
            differentiable=True,
        )
        assert result.converged

        solution.sum().backward()

        assert tracked_rhs.grad is not None
        expected = torch.linalg.solve(spd_matrix, torch.ones_like(rhs))
        torch.testing.assert_close(tracked_rhs.grad, expected, rtol=1e-6, atol=1e-8)


MARGINAL_ITERATIONS = 4
PCG_SYNCS_PER_ITERATION = 1
"""One packed read of the ``((r, w), (d, q))`` finiteness flag and the stopping test."""
FCG_SYNCS_PER_ITERATION = 2
"""PCG's one plus the orthogonalization projection finiteness read."""
PCG_LAUNCHES_PER_ITERATION_BOUND = 26
FCG_LAUNCHES_PER_ITERATION_BOUND = 55

OVERFLOW_SCALE = 1e155
"""Large enough that a squared entry (1e310) overflows float64 while the dot stays finite."""

OVERFLOW_RTOL = 1e-12


@dataclass(frozen=True, slots=True)
class OverflowSystem:
    """A matrix-free system whose plain dot products overflow but whose true values do not."""

    matvec: Callable[[torch.Tensor], torch.Tensor]
    preconditioner: Callable[[torch.Tensor], torch.Tensor]
    rhs: torch.Tensor
    iterations: int


@pytest.fixture(params=["curvature", "residual_and_curvature"])
def overflow_system(request: pytest.FixtureRequest) -> OverflowSystem:
    """Cancelling large terms: ``(d, q)`` alone, or ``(r, w)`` as well, overflows in plain float64."""
    ones = torch.ones(4, dtype=torch.float64)
    near_one = 1.0 - 1e-10
    if request.param == "curvature":
        signs = torch.tensor([1.0, -1.0, 1.0, -near_one], dtype=torch.float64)
        return OverflowSystem(
            matvec=lambda vector: signs * vector,
            preconditioner=lambda residual: residual * 1e55,
            rhs=ones * 1e100,
            iterations=3,
        )
    signs = torch.tensor([1.0, -1.0, 1.0, -near_one], dtype=torch.float64)
    return OverflowSystem(
        matvec=lambda vector: 1e-10 * vector,
        preconditioner=lambda residual: signs * residual,
        rhs=ones * OVERFLOW_SCALE,
        iterations=2,
    )


def _reference_pcg_with_stable_dots(
    system: OverflowSystem, iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Textbook PCG whose every dot product is the overflow-safe ``stable_dot_product``."""
    r = system.rhs.clone()
    u = torch.zeros_like(r)
    direction = torch.zeros_like(r)
    rw_prev = torch.ones((), dtype=r.dtype)
    for k in range(iterations):
        w = system.preconditioner(r)
        rw = stable_dot_product(r, w)
        # Fused like the solver: this fixture cancels ~1e9 in ``w + beta d``, so the
        # exact product of the fused form and the rounded product of the unfused one differ.
        direction = w.clone() if k == 0 else torch.addcmul(w, rw / rw_prev, direction)
        q = system.matvec(direction)
        alpha = rw / stable_dot_product(direction, q)
        u = u + alpha * direction
        r = r - alpha * q
        rw_prev = rw
    return u, r


class TestNonFinitePath:
    """Overflowing plain dot products still take the rescaled path, with the same results."""

    def test_fixture_overflows_in_plain_arithmetic(self, overflow_system: OverflowSystem) -> None:
        """Guard: the plain ``(r, w)`` or ``(d, q)`` is non-finite, so the fallback is exercised."""
        residual = overflow_system.rhs
        w = overflow_system.preconditioner(residual)
        q = overflow_system.matvec(w)
        plain = torch.stack([torch.dot(residual, w), torch.dot(w, q)])

        assert not bool(torch.isfinite(plain).all())
        assert bool(torch.isfinite(stable_dot_product(w, q)))

    def test_pcg_matches_the_stable_dot_reference(self, overflow_system: OverflowSystem) -> None:
        """Solution and residual after a few iterations equal the all-stable-dot textbook loop."""
        solution, result = pcg(
            overflow_system.matvec,
            overflow_system.rhs,
            preconditioner=overflow_system.preconditioner,
            maxiter=overflow_system.iterations,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )
        expected_u, expected_r = _reference_pcg_with_stable_dots(
            overflow_system, overflow_system.iterations
        )

        assert result.iterations == overflow_system.iterations
        assert bool(torch.isfinite(solution).all())
        torch.testing.assert_close(solution, expected_u, rtol=OVERFLOW_RTOL, atol=0.0)
        assert math.isclose(
            result.residual_abs, float(torch.linalg.norm(expected_r)), rel_tol=OVERFLOW_RTOL
        )

    def test_nan_system_still_reports_breakdown(self) -> None:
        """A NaN right-hand side is rejected at the boundary; a NaN matvec ends in breakdown."""
        rhs = torch.ones(4, dtype=torch.float64)

        _, result = pcg(
            lambda vector: vector * float("nan"),
            rhs,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        assert not result.converged
        assert result.breakdown


class TestStructuralCost:
    """Host syncs and kernel launches per iteration, measured as marginal cost of more iterations."""

    @staticmethod
    def _run(
        solver: Callable[..., tuple[torch.Tensor, object]],
        counter: DispatchCounter,
        rhs: torch.Tensor,
        matrix: torch.Tensor,
        iterations: int,
    ) -> DispatchCounter:
        inverse_diagonal = 1.0 / matrix.diagonal()
        with counter:
            solver(
                matrix,
                rhs,
                preconditioner=lambda residual: residual * inverse_diagonal,
                maxiter=iterations,
                rtol=NO_TOLERANCE,
                atol=NO_TOLERANCE,
            )
        return counter

    @pytest.mark.parametrize(
        ("solver", "syncs_per_iteration", "launches_bound"),
        [
            (pcg, PCG_SYNCS_PER_ITERATION, PCG_LAUNCHES_PER_ITERATION_BOUND),
            (flexible_cg, FCG_SYNCS_PER_ITERATION, FCG_LAUNCHES_PER_ITERATION_BOUND),
        ],
        ids=["pcg", "fcg"],
    )
    def test_marginal_iteration_cost(
        self,
        solver: Callable[..., tuple[torch.Tensor, object]],
        syncs_per_iteration: int,
        launches_bound: int,
        spd_matrix: torch.Tensor,
        rhs: torch.Tensor,
    ) -> None:
        """Each extra iteration costs the documented syncs and at most the launch bound."""
        short = self._run(solver, DispatchCounter(), rhs, spd_matrix, ITERATIONS)
        long = self._run(
            solver, DispatchCounter(), rhs, spd_matrix, ITERATIONS + MARGINAL_ITERATIONS
        )

        assert (long.syncs - short.syncs) == syncs_per_iteration * MARGINAL_ITERATIONS
        assert (long.launches - short.launches) <= launches_bound * MARGINAL_ITERATIONS

    def test_differentiable_pcg_backpropagates(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """The fused updates stay on the autograd tape; d sum(x)/db = A^-1 1."""
        tracked_rhs = rhs.clone().requires_grad_(True)
        solution, result = pcg(
            spd_matrix,
            tracked_rhs,
            maxiter=3 * SYSTEM_SIZE,
            rtol=1e-12,
            atol=1e-14,
            differentiable=True,
        )
        assert result.converged

        solution.sum().backward()

        assert tracked_rhs.grad is not None
        expected = torch.linalg.solve(spd_matrix, torch.ones_like(rhs))
        torch.testing.assert_close(tracked_rhs.grad, expected, rtol=1e-6, atol=1e-8)


class IndexWidthRecorder(TorchDispatchMode):
    """Records the CSR index dtype of every sparse operand of ``aten.matmul``."""

    def __init__(self) -> None:
        super().__init__()
        self.index_dtypes: list[torch.dtype] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        for operand in args:
            if (
                "matmul" in str(func)
                and isinstance(operand, torch.Tensor)
                and operand.layout == torch.sparse_csr
            ):
                self.index_dtypes.append(operand.col_indices().dtype)
        return func(*args, **(kwargs or {}))


@pytest.fixture
def spd_csr(spd_matrix: torch.Tensor) -> torch.Tensor:
    """The SPD test matrix as a CSR tensor with PyTorch's default int64 indices."""
    csr = spd_matrix.to_sparse_csr()
    assert csr.col_indices().dtype == torch.int64
    return csr


class TestSystemMatrixIndexWidth:
    """A CSR system matrix is applied through int32 indices; the caller's tensor is untouched."""

    def test_matches_the_int64_index_solve_to_rounding(
        self, spd_csr: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """Same SpMV values through narrower indices; a matvec callable keeps int64 indices."""
        wide_u, wide_result = pcg(
            lambda vector: spd_csr @ vector,
            rhs,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )
        narrow_u, narrow_result = pcg(
            spd_csr, rhs, maxiter=ITERATIONS, rtol=NO_TOLERANCE, atol=NO_TOLERANCE
        )

        assert narrow_result.iterations == wide_result.iterations
        torch.testing.assert_close(narrow_u, wide_u, rtol=1e-12, atol=0.0)

    def test_spmv_sees_int32_indices_and_the_callers_matrix_keeps_int64(
        self, spd_csr: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """The wrapped matvec narrows; the user's object is neither replaced nor mutated."""
        recorder = IndexWidthRecorder()
        with recorder:
            pcg(spd_csr, rhs, maxiter=ITERATIONS, rtol=NO_TOLERANCE, atol=NO_TOLERANCE)

        assert recorder.index_dtypes
        assert set(recorder.index_dtypes) == {torch.int32}
        assert spd_csr.crow_indices().dtype == spd_csr.col_indices().dtype == torch.int64

    def test_coo_system_matrix_is_left_alone(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """Only CSR is narrowed; a COO matrix reaches the matvec unchanged."""
        coo = spd_matrix.to_sparse_coo()

        solution, result = pcg(coo, rhs, maxiter=SYSTEM_SIZE * 3)

        assert result.converged
        residual = torch.linalg.norm(spd_matrix @ solution - rhs)
        assert residual <= 10 * DEFAULT_RTOL * torch.linalg.norm(rhs)

    def test_differentiable_csr_system_keeps_its_graph(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """A CSR matrix that requires grad is not rebuilt, so gradients still reach it."""
        values = spd_matrix.to_sparse_csr().values().clone().requires_grad_(True)
        csr = spd_matrix.to_sparse_csr()
        tracked = torch.sparse_csr_tensor(
            csr.crow_indices(),
            csr.col_indices(),
            values,
            csr.shape,
            check_invariants=False,
        )

        solution, _ = pcg(tracked, rhs, maxiter=3 * SYSTEM_SIZE, differentiable=True)
        solution.sum().backward()

        assert values.grad is not None
        assert bool(torch.isfinite(values.grad).all())
