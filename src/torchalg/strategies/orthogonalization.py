"""Orthogonalization strategies for CG-family Krylov solvers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from torchalg.utils.numerics import stable_dot_product

DEFAULT_M_MAX = 20
REORTHOG_ZERO_NORM_TOL = 1e-14


@dataclass(frozen=True, slots=True)
class OrthogonalizationReport:
    """Diagnostic report returned by an orthogonalization step."""

    coefficients: tuple[float, ...]
    """Orthogonalization coefficients used in order."""

    breakdown: bool = False
    """Whether the orthogonalized vector became near-zero."""


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
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize ``vector`` against stored directions/products."""


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
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize using FCG periodic restart."""
        result = vector.clone()
        n_history = len(d_vectors)

        if n_history == 0:
            self._iteration_count += 1
            return result, OrthogonalizationReport(coefficients=())

        if math.isinf(self.m_max):
            m_i = n_history
        else:
            m_i = (self._iteration_count % int(self.m_max)) + 1
            m_i = min(m_i, n_history)

        start_idx = n_history - m_i
        result, report = _orthogonalize_classical(
            vector=vector,
            result=result,
            d_vectors=d_vectors[start_idx:],
            q_vectors=q_vectors[start_idx:],
        )
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
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize against the last ``window_size`` directions."""
        result = vector.clone()
        m = min(len(d_vectors), self._window_size)
        if m == 0:
            return result, OrthogonalizationReport(coefficients=())

        return _orthogonalize_classical(
            vector=vector,
            result=result,
            d_vectors=d_vectors[-m:],
            q_vectors=q_vectors[-m:],
        )


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
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize using the updated vector for each numerator."""
        result = vector.clone()
        if len(d_vectors) == 0:
            return result, OrthogonalizationReport(coefficients=())

        coefficients: list[float] = []
        for d_j, q_j in zip(d_vectors, q_vectors, strict=True):
            numerator = stable_dot_product(result, q_j)
            denominator = stable_dot_product(d_j, q_j)
            coeff = numerator / denominator
            coefficients.append(coeff)
            result = result - coeff * d_j

        return result, _build_report(vector, result, coefficients)


def create_fcg_orthogonalization(
    m_max: float | int = DEFAULT_M_MAX,
) -> OrthogonalizationStrategy:
    """Create Notay FCG(m) orthogonalization from a public ``m_max`` value."""
    if m_max == 0:
        raise ValueError("m_max cannot be 0. Use m_max=1 for minimal orthogonalization.")
    if m_max < -1 and not math.isinf(m_max):
        raise ValueError(f"m_max must be >= -1 or inf, got {m_max}")
    if m_max == -1:
        m_max = math.inf
    return PeriodicRestartOrthogonalization(m_max=float(m_max))


def _orthogonalize_classical(
    *,
    vector: torch.Tensor,
    result: torch.Tensor,
    d_vectors: Sequence[torch.Tensor],
    q_vectors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, OrthogonalizationReport]:
    """Apply classical A-conjugacy Gram-Schmidt to selected history."""
    coefficients: list[float] = []

    for d_j, q_j in zip(d_vectors, q_vectors, strict=True):
        numerator = stable_dot_product(vector, q_j)
        denominator = stable_dot_product(d_j, q_j)
        coeff = numerator / denominator
        coefficients.append(coeff)
        result = result - coeff * d_j

    return result, _build_report(vector, result, coefficients)


def _build_report(
    vector: torch.Tensor,
    result: torch.Tensor,
    coefficients: list[float],
) -> OrthogonalizationReport:
    """Build an orthogonalization report with breakdown classification."""
    result_norm = float(torch.linalg.norm(result))
    vector_norm = float(torch.linalg.norm(vector))
    breakdown = result_norm < REORTHOG_ZERO_NORM_TOL * max(vector_norm, 1.0)
    return OrthogonalizationReport(
        coefficients=tuple(coefficients),
        breakdown=breakdown,
    )
