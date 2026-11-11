"""Work accounting and equivalence of the CG loop.

Pins what one ``k``-iteration solve costs (preconditioner applies, matvecs,
curvature dot products) and proves the restructured loop reproduces an
independent textbook reference: PCG bit for bit, FCG to rounding against the
previous per-pair Gram-Schmidt implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

import torchalg.conjugate_gradient as cg_module
from torchalg import flexible_cg, pcg
from torchalg.strategies.orthogonalization import DEFAULT_M_MAX
from torchalg.utils.numerics import stable_dot_product

ITERATIONS = 6
SYSTEM_SIZE = 48
NO_TOLERANCE = 0.0
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


@pytest.fixture
def counting_dot(monkeypatch: pytest.MonkeyPatch) -> CallCounter:
    """Counts ``stable_dot_product`` calls made by the CG loop."""
    counter = CallCounter(lambda vector: vector)  # replaced below; only ``calls`` is used

    def counted(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        counter.calls += 1
        return stable_dot_product(a, b)

    monkeypatch.setattr(cg_module, "stable_dot_product", counted)
    return counter


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
        self, rhs: torch.Tensor, counting_matvec: CallCounter, counting_dot: CallCounter
    ) -> None:
        """``(r, w)`` once (shared with the direction strategy) and ``(d, q)`` once."""
        pcg(
            counting_matvec,
            rhs,
            maxiter=ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        )

        assert counting_dot.calls == 2 * ITERATIONS


class TestEquivalence:
    """The restructured loop reproduces independent references."""

    def test_pcg_residual_history_is_bit_identical_to_textbook_pcg(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        """Same operations in the same order: histories are equal, not merely close."""
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
        assert result.residual_history_abs == tuple(expected)

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
