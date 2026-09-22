"""Post-hoc analysis of recorded solver iteration history.

Deliberately separate from ``solve()``/``IterationHistory``: both functions
here are explicit, opt-in calls the caller makes on their own schedule, never
something a solve automatically runs. ``energy_norm_history`` needs ``A``
back on a device, in bounded chunks the caller sizes; the Golub-Meurant bound
needs a look-ahead ``delay`` only the caller can choose sensibly. See
``docs/bug-full-trace-history-exhausts-gpu-memory.md`` for why this stays out
of the solve's own device-memory budget.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from torchalg.utils.energy import energy_dot


def energy_norm_history(
    A: torch.Tensor,
    x_exact: torch.Tensor,
    solution_vectors: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute ``||u_k - x_exact||_A`` for every recorded iterate.

    Streams ``solution_vectors`` (a CPU tensor from a ``FULL``-trace
    ``SolverResult``) to ``A``'s device in bounded chunks, so peak device
    memory is ``O(chunk_size * n)`` regardless of how many iterations were
    recorded - never the whole history at once.

    Args:
        A (torch.Tensor): SPD system matrix, shape ``(n, n)``.
        x_exact (torch.Tensor): Known exact solution, shape ``(n,)``.
        solution_vectors (torch.Tensor): Recorded iterates, shape
            ``(k, n)`` (typically ``SolverResult.solution_vectors``).
        chunk_size (int): Number of rows moved to ``A.device`` at a time.

    Returns:
        torch.Tensor: ``||u_k - x_exact||_A`` per row, shape ``(k,)``, on
            the CPU.

    Example:
        >>> import torch
        >>> A = torch.diag(torch.tensor([2.0, 8.0]))
        >>> x_exact = torch.zeros(2)
        >>> solution_vectors = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        >>> energy_norm_history(A, x_exact, solution_vectors)
        tensor([3.1623, 0.0000])
    """
    device = A.device
    x_exact_device = x_exact.to(device)
    chunks = []
    for start in range(0, solution_vectors.shape[0], chunk_size):
        error = solution_vectors[start : start + chunk_size].to(device) - x_exact_device
        chunks.append(energy_dot(error, error, A).clamp_min(0.0).sqrt().cpu())
    return torch.cat(chunks)


def golub_meurant_error_bound(
    energy_decrements: Sequence[float],
    delay: int = 10,
) -> tuple[float, ...]:
    """Estimate a lower bound on ``||e_k||_A`` from CG's own decrements.

    No ground truth required: sums a forward window of the exact
    per-iteration decrements ``alpha_j * rho_j`` (Golub & Meurant 1994;
    Strakoš & Tichý 2002) — ``sum_{j=k}^{k+delay-1} alpha_j * rho_j`` is a
    lower bound on ``||e_k||_A^2`` for any ``delay``, growing tighter as
    ``delay`` increases (up to the iterations actually available).

    Args:
        energy_decrements (Sequence[float]): Raw per-iteration
            ``alpha_j * rho_j``, e.g. ``SolverResult.energy_decrements``.
        delay (int): Forward look-ahead window size; larger gives a tighter
            bound but needs that many further iterations to have run.

    Returns:
        tuple[float, ...]: Lower-bound estimate of ``||e_k||_A`` for each
            recorded iteration ``k``.

    Example:
        >>> golub_meurant_error_bound([1.0, 0.5, 0.25, 0.125], delay=2)
        (1.224744871391589, 0.8660254037844386, 0.6123724356957945, 0.35355339059327373)
    """
    decrements = list(energy_decrements)
    n = len(decrements)
    return tuple(math.sqrt(max(sum(decrements[k : k + delay]), 0.0)) for k in range(n))
