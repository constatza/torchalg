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

from collections.abc import Callable, Mapping
from types import MappingProxyType

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


Orthogonalizer = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
"""``(span, w) -> (coefficients, w_orthogonal)``: orthogonalize ``w`` against the rows of
``span`` and return the Hessenberg column ``coefficients`` of length ``len(span)``."""


def _modified_gram_schmidt(
    span: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonalize ``w`` against the rows of ``span`` one vector at a time (PyAMG's loop).

    Cost:
        ``k`` = ``len(span)``. W = ``4 n k``; D = ``k (ceil(log2 n) + 1)`` (each
        projection depends on the previous one); L = ``O(k)`` launches (one dot and one
        axpy per basis vector). Bit-identical to PyAMG's reference loop.
    """
    coefficients = []
    for vector in span:
        coefficient = torch.vdot(vector, w)
        w = w - coefficient * vector
        coefficients.append(coefficient)
    return torch.stack(coefficients), w


def _classical_gram_schmidt(
    span: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonalize ``w`` against the rows of ``span`` with two block projections.

    Classical Gram-Schmidt with one reorthogonalization ("twice is enough", Giraud et
    al. 2005): as orthogonal as modified Gram-Schmidt to rounding, but four products
    per step instead of one dot and one axpy per basis vector.

    Cost:
        ``k`` = ``len(span)``. W = ``8 n k``; D = ``4 (ceil(log2 n) + 1)``; L = 4
        launches independent of ``k``.
    """
    coefficients = span.conj() @ w
    w = w - span.mT @ coefficients
    correction = span.conj() @ w
    return coefficients + correction, w - span.mT @ correction


# UNMEASURED ON A GPU: the "cuda" entry is registered on trust (fewer launches). Remove it if
# the block variant is not faster there; plans/cost-audit-fixes.md tracks the measurement.
ORTHOGONALIZER_BY_DEVICE: Mapping[str, Orthogonalizer] = MappingProxyType(
    {"cuda": _classical_gram_schmidt}
)
"""The one place that maps a device type to its Gram-Schmidt variant; unlisted types
(the CPU) get ``_modified_gram_schmidt``, which keeps the estimate bit-identical to
PyAMG's. The ``"cuda"`` entry is registered because a launch-bound device pays per
kernel and the block form issues a constant number of them (2 249 against 879 ATen
dispatches per estimate on a CPU count); it is unmeasured on a GPU and must be
removed if it is not faster there. On a CPU it was no faster and shifted the estimate
by about ``1e-10`` on tiny coarse matrices, which moved a preconditioner application
by up to ``2e-10``."""


def _arnoldi(
    matrix: torch.Tensor,
    start: torch.Tensor,
    maxiter: int,
    orthogonalize: Orthogonalizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Arnoldi projection of ``matrix`` and its Ritz eigen-decomposition.

    The Krylov loop never reads a device value back: breakdown is detected
    afterwards from one host copy of the Hessenberg matrix, and the steps
    taken past a breakdown are discarded. The arithmetic is unchanged, so
    the result equals the step-by-step check of PyAMG's routine; the loop
    costs one device-to-host transfer per pass instead of one per step. The
    small eigenproblem is solved on the host, which keeps it independent of
    device ``linalg.eig`` support.

    Each new vector is orthogonalized against the basis by ``orthogonalize``, by default
    the variant registered for the matrix's device type (``ORTHOGONALIZER_BY_DEVICE``).

    Cost:
        ``m`` = ``min(n, maxiter)`` Krylov steps. W = ``m * (2 nnz + 8 n m)``
        flops (one SpMV, four ``n x k`` matvecs per step, ``k <= m``);
        D = ``O(m log n)``; L = ``O(m)`` launches (the previous per-vector loop
        launched ``O(m^2)``); S = 1 (the Hessenberg copy). The sum over steps
        is exactly ``2 nnz m + 4 n m (m + 1)``, the bound above counts the
        full basis at every step.

    Args:
        matrix (torch.Tensor): Square matrix.
        start (torch.Tensor): Starting vector, shape ``(n,)``.
        maxiter (int): Maximum Krylov dimension (clamped to ``n``).
        orthogonalize (Orthogonalizer | None): Gram-Schmidt variant; ``None`` selects it
            from ``ORTHOGONALIZER_BY_DEVICE`` by the matrix's device type.

    Returns:
        tuple: ``(ritz_vectors, ritz_values, H, V, breakdown)``; the Ritz
            data and ``H`` live on the host, ``V`` (basis vectors as rows,
            ``count + 1`` of them) on the matrix's device.
    """
    orthogonalize = orthogonalize or ORTHOGONALIZER_BY_DEVICE.get(
        matrix.device.type, _modified_gram_schmidt
    )
    steps = min(matrix.shape[0], maxiter)
    first = start / torch.linalg.norm(start)
    dtype = torch.promote_types(first.dtype, matrix.dtype)
    matrix = matrix.to(dtype)
    basis = torch.zeros(steps + 1, matrix.shape[0], dtype=dtype, device=matrix.device)
    basis[0] = first
    hessenberg = torch.zeros(steps + 1, steps, dtype=dtype, device=matrix.device)
    for j in range(steps):
        w = matrix @ basis[j]
        coefficients, w = orthogonalize(basis[: j + 1], w)
        hessenberg[: j + 1, j] = coefficients
        norm = torch.linalg.norm(w)
        hessenberg[j + 1, j] = norm
        # A zero norm leaves ``w`` unscaled, as the early-exit form did.
        basis[j + 1] = w / torch.where(norm == 0.0, torch.ones_like(norm), norm)
    hessenberg = hessenberg.cpu()
    threshold = _breakdown_tolerance(dtype)
    below = (hessenberg.diagonal(-1).real < threshold).tolist()
    broken_at = next((j for j, flag in enumerate(below) if flag), None)
    breakdown = broken_at is not None
    count = steps if broken_at is None else broken_at + 1
    basis = basis[: count + 1]
    values, vectors = torch.linalg.eig(hessenberg[:count, :count])
    return vectors, values, hessenberg, basis, breakdown


def approximate_spectral_radius(
    matrix: torch.Tensor,
    draw: Callable[[int], torch.Tensor],
    *,
    tol: float = 0.01,
    maxiter: int = 15,
    restart: int = 5,
    initial_guess: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate ``rho(matrix)`` with restarted Arnoldi.

    Cost:
        ``P`` passes of ``m = min(n, maxiter)`` steps, ``1 <= P <= restart + 1``
        (data dependent: restarting stops once the Ritz-pair error is below
        ``tol``). With ``z = nnz``, ``V = 8`` bytes per ``float64``, per pass:
        ``W = 2zm + 4nm(m + 1) + O(nm)`` (``m`` SpMVs and the Gram-Schmidt
        products of each step against the growing basis; the CPU's modified
        variant spends ``2 n k`` per step as ``k`` dots and axpys, the block
        variant registered for a GPU four products and the same order of work), plus a host
        ``m x m`` eigenproblem of about ``25 m^3`` flops,
        ``Q = m`` SpMV bytes ``+ 2Vnm(m + 1)`` (the basis is re-read by each
        product, a standard no-reuse upper bound),
        ``D = m (ceil(log2 n) + 1) * O(1)`` (block variant; the modified variant adds a factor
        of the basis size), ``L = O(m)`` block against ``O(m^2)`` modified, and ``S = 1`` (the
        Hessenberg copy), so the call costs ``P`` times these with ``P``
        usually 1 to 2. A call is independent: two calls on one matrix cost
        twice, so callers memoize per matrix (``jacobi_spectral_radius``).
        Roofline: ``W/Q`` is below 0.2 flop per byte (memory-bound).
        Work-depth: ``W/D`` is large in the basis products; the ``m``-step
        recurrence and the host eigensolve bound the depth.

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
        torch.Tensor: Largest-magnitude Ritz value, 0-d, ``matrix``'s
            real dtype and device.
    """
    start = (draw(matrix.shape[0]) if initial_guess is None else initial_guess).to(
        dtype=matrix.dtype, device=matrix.device
    )
    for _ in range(restart + 1):
        vectors, values, hessenberg, basis, breakdown = _arnoldi(matrix, start, maxiter)
        count = values.shape[0]
        top = int(torch.argmax(values.abs()))
        error = hessenberg[count, count - 1] * vectors[-1, top]
        ritz_vector = vectors[:, top].to(matrix.device)
        start = basis[:-1].mT.to(vectors.dtype) @ ritz_vector
        if float(error.abs() / values[top].abs()) < tol or breakdown:
            break
    return values[top].abs().to(matrix.device)
