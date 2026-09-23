"""Spectral damping rules for weighted Jacobi, sharing one ``rho(D^-1 A)`` estimate per level.

Two roles use the same operator ``D^-1 A`` with different nominal damping,
divided by its spectral radius ``rho`` (PyAMG's rules):

- **relaxation** (smoothing the error every cycle): ``omega = 1 / rho``;
- **prolongation smoothing** (smoothing the interpolation once, at setup,
  Vanek, Mandel & Brezina 1996): ``omega = (4/3) / rho``.

``rho`` is estimated with the restarted-Arnoldi routine of ``_spectral.py``,
started from a fixed seed so it is deterministic, and cached per matrix
object, so the smoother and the prolongator smoothing of the same level - and
every ``apply`` - share a single estimate. Matrices are treated as immutable:
the entry lives only as long as that exact tensor object. An explicit float
always overrides the rule. Unlike a fixed 0.67, the rule keeps the iteration
convergent when ``rho`` is well above 2 (``omega`` must stay below ``2 / rho``).

References:
    - PyAMG 5.3.0, ``relaxation/smoothing.py`` (``rho_D_inv_A``) and
      ``aggregation/smooth.py`` (``jacobi_prolongation_smoother``).
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Computing, 56(3), 179-196.
"""

from __future__ import annotations

import weakref

import torch

from ._spectral import approximate_spectral_radius

RELAXATION_NOMINAL = 1.0
"""Nominal Jacobi damping for relaxation: ``omega = 1 / rho``."""

PROLONGATION_NOMINAL = 4.0 / 3.0
"""Nominal Jacobi damping for prolongator smoothing: ``omega = (4/3) / rho``."""

_SEED = 0
_cache: dict[int, tuple[weakref.ReferenceType[torch.Tensor], float]] = {}


def scaled_by_inverse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """``D^-1 A`` with ``D^-1`` zero where the diagonal is zero (PyAMG's ``get_diagonal(inv=True)``).

    Args:
        matrix (torch.Tensor): Square matrix ``A``.

    Returns:
        torch.Tensor: Row-scaled matrix.
    """
    diagonal = torch.diagonal(matrix)
    inverse = torch.where(diagonal != 0, 1.0 / diagonal, torch.zeros_like(diagonal))
    return inverse.unsqueeze(1) * matrix


def _seeded_draw(size: int) -> torch.Tensor:
    """Fixed-seed uniform ``[0, 1)`` start vector for the Arnoldi estimate.

    Args:
        size (int): Vector length.

    Returns:
        torch.Tensor: The same vector on every call.
    """
    generator = torch.Generator().manual_seed(_SEED)
    return torch.rand(size, generator=generator, dtype=torch.float64)


def jacobi_spectral_radius(matrix: torch.Tensor) -> float:
    """Cached, deterministic Arnoldi estimate of ``rho(D^-1 A)``.

    Args:
        matrix (torch.Tensor): Square matrix ``A``.

    Returns:
        float: Estimated spectral radius of ``D^-1 A``.
    """
    key = id(matrix)
    entry = _cache.get(key)
    if entry is not None and entry[0]() is matrix:
        return entry[1]
    rho = approximate_spectral_radius(scaled_by_inverse_diagonal(matrix), _seeded_draw)
    _cache[key] = (weakref.ref(matrix, lambda _: _cache.pop(key, None)), rho)
    return rho


def jacobi_omega(matrix: torch.Tensor, nominal: float, override: float | None) -> float:
    """Resolve a Jacobi damping factor: the explicit value, else ``nominal / rho``.

    Args:
        matrix (torch.Tensor): Matrix the damping is applied to.
        nominal (float): ``RELAXATION_NOMINAL`` or ``PROLONGATION_NOMINAL``.
        override (float | None): Explicit omega; ``None`` selects the spectral rule.

    Returns:
        float: Damping factor.
    """
    return override if override is not None else nominal / jacobi_spectral_radius(matrix)
