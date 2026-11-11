"""Orthogonalization strategies for CG-family Krylov solvers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from torchalg.models.history import DirectionWindow
from torchalg.utils.numerics import is_finite, stable_dot_product

DEFAULT_M_MAX = 20
REORTHOG_ZERO_NORM_TOL = 1e-14


@dataclass(frozen=True, slots=True)
class OrthogonalizationReport:
    """Diagnostic report returned by an orthogonalization step."""

    coefficients: torch.Tensor
    """Orthogonalization coefficients, oldest stored direction first (1-d
    tensor of length ``k``; no production consumer needs these as floats -
    only tests inspect them)."""

    breakdown: torch.Tensor = field(default_factory=lambda: torch.tensor(False))
    """0-d bool tensor: whether the orthogonalized vector became near-zero.
    Deferred to a tensor rather than resolved to bool here, since nothing
    reads this per-iteration — it's batched into a single extraction at the
    very end of the solve (see CGState.ortho_breakdown_at)."""


def _no_breakdown(reference: torch.Tensor) -> torch.Tensor:
    """A 0-d ``False`` tensor matching ``reference``'s device.

    ``OrthogonalizationReport.breakdown``'s dataclass default is a bare
    ``torch.tensor(False)`` (always CPU) — fine as a type-level default, but
    the empty-history early returns below must not actually use it: combining
    a CPU flag with GPU-computed ones from ``_build_report`` in the solver's
    breakdown accumulator would fail on any GPU solve (device mismatch).
    """
    return torch.zeros((), dtype=torch.bool, device=reference.device)


def _empty_report(vector: torch.Tensor) -> OrthogonalizationReport:
    """Report for an orthogonalization against an empty window."""
    return OrthogonalizationReport(
        coefficients=torch.empty(0, dtype=vector.dtype, device=vector.device),
        breakdown=_no_breakdown(vector),
    )


class OrthogonalizationStrategy(ABC):
    """Interface for search-direction orthogonalization."""

    @property
    @abstractmethod
    def window_size(self) -> int | None:
        """Return the required direction-history window size."""

    @abstractmethod
    def orthogonalize(
        self,
        vector: torch.Tensor,
        window: DirectionWindow,
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize ``vector`` against the stored directions/products."""


class PeriodicRestartOrthogonalization(OrthogonalizationStrategy):
    """FCG(m) orthogonalization with Notay (2000) periodic restart.

    Implements Algorithm 2.1 of Notay, Y. (2000), "Flexible Conjugate
    Gradients", SIAM J. Sci. Comput. 22(4), 1444-1460: a plain classical
    Gram-Schmidt summation over the truncated window, with no term-skipping.

    The window size cycles as ``m_i = (i mod m_max) + 1`` for (0-indexed)
    call count ``i`` — a clean sawtooth of period ``m_max``:
    ``1, 2, ..., m_max, 1, 2, ...``. This is the same periodic-restart
    truncation PETSc documents and implements for ``KSPFCG``/
    ``KSPPIPEFCG``/``KSPPIPEGCR`` (``truncation_type=notay``; see
    ``src/ksp/ksp/impls/fcg/fcg.c``, ``mi = ((i - 1) % mmax) + 1`` for
    1-indexed ``i``), which was used to verify this formula since the
    primary source was not directly available while auditing this module.
    """

    def __init__(self, m_max: float) -> None:
        """Initialize periodic-restart orthogonalization."""
        if not (m_max >= 1 or math.isinf(m_max)):
            raise ValueError(f"m_max must be >= 1 or inf, got {m_max}")

        self.m_max = m_max
        self._iteration_count = 0

    @property
    def window_size(self) -> int | None:
        """Return configured maximum window size, or ``None`` for FCG(inf)."""
        if math.isinf(self.m_max):
            return None
        return int(self.m_max)

    def orthogonalize(
        self,
        vector: torch.Tensor,
        window: DirectionWindow,
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize using FCG periodic restart."""
        n_history = len(window)

        if n_history == 0:
            self._iteration_count += 1
            return vector.clone(), _empty_report(vector)

        if math.isinf(self.m_max):
            m_i = n_history
        else:
            m_i = (self._iteration_count % int(self.m_max)) + 1
            m_i = min(m_i, n_history)

        result, report = _orthogonalize_classical(vector, window.tail(m_i))
        self._iteration_count += 1
        return result, report


class TruncatedGramSchmidt(OrthogonalizationStrategy):
    """Truncated Gram-Schmidt using a sliding direction-history window.

    Classical Gram-Schmidt (Golub & Van Loan, "Matrix Computations", §5.2;
    Saad, "Iterative Methods for Sparse Linear Systems", incomplete
    orthogonalization) restricted to the last ``window_size`` directions -
    a plain summation, no term-skipping.
    """

    def __init__(self, window_size: int) -> None:
        """Initialize truncated Gram-Schmidt."""
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size

    @property
    def window_size(self) -> int:
        """Return configured sliding-window size."""
        return self._window_size

    def orthogonalize(
        self,
        vector: torch.Tensor,
        window: DirectionWindow,
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize against the last ``window_size`` directions."""
        if len(window) == 0:
            return vector.clone(), _empty_report(vector)

        return _orthogonalize_classical(vector, window.tail(self._window_size))


class ModifiedGramSchmidt(OrthogonalizationStrategy):
    """Modified Gram-Schmidt over all stored directions.

    Björck, Å. (1994), "Numerics of Gram-Schmidt Orthogonalization", Linear
    Algebra Appl. 197-198; Golub & Van Loan, "Matrix Computations", §5.2.8:
    MGS differs from classical Gram-Schmidt only in using the
    already-updated vector for each numerator - still a plain summation,
    no term-skipping.
    """

    @property
    def window_size(self) -> None:
        """Return ``None`` for unlimited history."""
        return None

    def orthogonalize(
        self,
        vector: torch.Tensor,
        window: DirectionWindow,
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize using the updated vector for each numerator.

        Inherently sequential: each numerator uses the vector already
        updated by the previous directions, so this cannot be collapsed into
        matrix-vector products without changing the algorithm.
        """
        if len(window) == 0:
            return vector.clone(), _empty_report(vector)

        result = vector
        coefficients: list[torch.Tensor] = []
        for d_j, q_j, curvature_j in zip(window.d, window.q, window.curvature, strict=True):
            coeff = stable_dot_product(result, q_j) / curvature_j
            coefficients.append(coeff)
            result = result - coeff * d_j

        return result, _build_report(vector, result, torch.stack(coefficients))


def create_fcg_orthogonalization(
    m_max: float = DEFAULT_M_MAX,
) -> OrthogonalizationStrategy:
    """Create Notay FCG(m) orthogonalization from a public ``m_max`` value."""
    if m_max == 0:
        raise ValueError("m_max cannot be 0. Use m_max=1 for minimal orthogonalization.")
    if m_max < -1 and not math.isinf(m_max):
        raise ValueError(f"m_max must be >= -1 or inf, got {m_max}")
    if m_max == -1:
        m_max = math.inf
    return PeriodicRestartOrthogonalization(m_max=float(m_max))


def _projections(vector: torch.Tensor, window: DirectionWindow) -> torch.Tensor:
    """Return ``[(vector, q_j)]`` for every stored pair, shape ``(k,)``.

    One matrix-vector product; if any entry is non-finite (overflow), the
    per-pair ``stable_dot_product`` loop recomputes them, so overflow
    handling matches the sequential formulation.
    """
    projections = window.q @ vector
    if is_finite(projections):
        return projections
    return torch.stack([stable_dot_product(vector, q_j) for q_j in window.q])


def _orthogonalize_classical(
    vector: torch.Tensor,
    window: DirectionWindow,
) -> tuple[torch.Tensor, OrthogonalizationReport]:
    """Apply classical A-conjugacy Gram-Schmidt against ``window``.

    Classical Gram-Schmidt takes every numerator ``(vector, q_j)`` from the
    original ``vector``, so the coefficients are independent of each other
    and the sequential sum ``vector - sum_j c_j d_j`` is one matrix-vector
    product ``D^T c`` (the same formula, summed in a different order). The
    denominators ``(d_j, q_j)`` are the curvatures stored with the window.
    """
    coefficients = _projections(vector, window) / window.curvature
    result = torch.addmv(vector, window.d.mT, coefficients, alpha=-1)
    return result, _build_report(vector, result, coefficients)


def _build_report(
    vector: torch.Tensor,
    result: torch.Tensor,
    coefficients: torch.Tensor,
) -> OrthogonalizationReport:
    """Build an orthogonalization report with breakdown classification.

    Both `breakdown` and `coefficients` stay tensors — nothing here needs a
    Python scalar. `breakdown`'s combined comparison uses one tensor
    expression instead of two separate float() conversions; `coefficients`
    aren't converted at all, since nothing downstream reads them as floats.
    """
    result_norm = torch.linalg.norm(result)
    vector_norm = torch.linalg.norm(vector)
    breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * torch.clamp_min(vector_norm, 1.0)
    return OrthogonalizationReport(coefficients=coefficients, breakdown=breakdown)
