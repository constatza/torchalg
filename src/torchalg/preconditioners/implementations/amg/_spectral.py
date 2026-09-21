"""Approximate spectral radius by restarted Arnoldi, port of PyAMG's routine.

Used by the Jacobi prolongator smoother, which damps with
``omega / rho(D^{-1} A)``. The estimate is the largest-magnitude Ritz value
of a short Arnoldi run (``maxiter`` steps), restarted from the corresponding
Ritz vector until the Ritz-pair error is below ``tol`` (relative) or
``restart`` restarts have been used. It is an *estimate* with about 1 %
tolerance, started from a random vector, exactly as in PyAMG.

References:
    - PyAMG 5.3.0, ``pyamg/util/linalg.py``
      (``approximate_spectral_radius``, ``_approximate_eigenvalues``).
    - Bai, Demmel, Dongarra, Ruhe & van der Vorst (2000). Templates for the
      Solution of Algebraic Eigenvalue Problems. SIAM.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _breakdown_tolerance(dtype: torch.dtype) -> float:
    """PyAMG's ``set_tol``: a precision-based Arnoldi breakdown threshold.

    Args:
        dtype (torch.dtype): Real or complex floating dtype.

    Returns:
        float: ``1e3 * eps`` for single, ``1e6 * eps`` for double precision.
    """
    real = torch.empty((), dtype=dtype).real.dtype
    return (1e3 if real == torch.float32 else 1e6) * torch.finfo(real).eps


def _arnoldi(
    matrix: torch.Tensor, start: torch.Tensor, maxiter: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], bool]:
    """Arnoldi projection of ``matrix`` and its Ritz eigen-decomposition.

    Args:
        matrix (torch.Tensor): Square matrix.
        start (torch.Tensor): Starting vector, shape ``(n,)``.
        maxiter (int): Maximum Krylov dimension (clamped to ``n``).

    Returns:
        tuple: ``(ritz_vectors, ritz_values, H, V, breakdown)``.
    """
    steps = min(matrix.shape[0], maxiter)
    basis = [start / torch.linalg.norm(start)]
    dtype = torch.promote_types(basis[0].dtype, matrix.dtype)
    matrix = matrix.to(dtype)
    hessenberg = torch.zeros(steps + 1, steps, dtype=dtype, device=matrix.device)
    threshold = _breakdown_tolerance(dtype)
    breakdown = False
    for j in range(steps):
        w = matrix @ basis[-1]
        for i, v in enumerate(basis):
            hessenberg[i, j] = torch.vdot(v, w)
            w = w - hessenberg[i, j] * v
        hessenberg[j + 1, j] = torch.linalg.norm(w)
        if hessenberg[j + 1, j].real < threshold:
            breakdown = True
            if hessenberg[j + 1, j] != 0.0:
                w = w / hessenberg[j + 1, j]
            basis.append(w)
            break
        basis.append(w / hessenberg[j + 1, j])
    values, vectors = torch.linalg.eig(hessenberg[: j + 1, : j + 1])
    return vectors, values, hessenberg, basis, breakdown


def approximate_spectral_radius(
    matrix: torch.Tensor,
    draw: Callable[[int], torch.Tensor],
    *,
    tol: float = 0.01,
    maxiter: int = 15,
    restart: int = 5,
    initial_guess: torch.Tensor | None = None,
) -> float:
    """Estimate ``rho(matrix)`` with restarted Arnoldi.

    Args:
        matrix (torch.Tensor): Square matrix (need not be symmetric).
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random
            source, ``n -> tensor of length n``; called once unless
            ``initial_guess`` is given.
        tol (float): Relative Ritz-pair error at which to stop restarting.
        maxiter (int): Krylov dimension per Arnoldi pass.
        restart (int): Maximum number of restarts.
        initial_guess (torch.Tensor | None): Starting vector, shape ``(n,)``.

    Returns:
        float: Largest-magnitude Ritz value.
    """
    start = (draw(matrix.shape[0]) if initial_guess is None else initial_guess).to(
        dtype=matrix.dtype, device=matrix.device
    )
    for _ in range(restart + 1):
        vectors, values, hessenberg, basis, breakdown = _arnoldi(matrix, start, maxiter)
        count = values.shape[0]
        top = int(torch.argmax(values.abs()))
        error = hessenberg[count, count - 1] * vectors[-1, top]
        start = torch.stack(basis[:-1], dim=1).to(vectors.dtype) @ vectors[:, top]
        if float(error.abs() / values[top].abs()) < tol or breakdown:
            break
    return float(values[top].abs())
