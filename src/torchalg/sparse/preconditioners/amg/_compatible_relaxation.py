"""Compatible-relaxation (CR) coarsening kernels for Bootstrap AMG, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._compatible_relaxation`` (dense; kept
unmodified for comparison - see ``docs/plan.md``'s "Correction: dense and
sparse must be separate implementations, not an internal branch").

``hcr_operator``/``cr_rate``/``compatible_relaxation_coarsening``'s *outer*
logic is format-agnostic in the dense module (confirmed by reading: it only
ever indexes a dense ``(n,)`` error/priority vector by boolean mask, calls
the injected ``relaxation`` callable, and computes vector norms - never
touches ``matrix``'s own storage format directly). They are duplicated here
rather than imported, because the one genuinely format-specific piece,
``_independent_set_of``'s adjacency-row access, is called from *inside*
``compatible_relaxation_coarsening``'s own body with no existing injection
seam - and ``torchalg.sparse`` may never import from
``torchalg.preconditioners.implementations`` regardless of how small the
format-agnostic surface is (the enforced ``tach.toml`` boundary). Splitting
``_independent_set_of`` out via a new injectable parameter on the dense
``compatible_relaxation_coarsening`` was considered and rejected: it would
add a new seam to already-shipped, well-tested dense code for a single
caller (this module), where duplicating ~250 lines of logic - small,
self-contained, with its own direct parity tests below - carries zero risk
to the dense side. This mirrors this project's own precedent of duplicating
small, entangled boilerplate (e.g. ``resolve_jacobi_default``) rather than
promoting it when promotion would need a new seam for one consumer.

``_independent_set_of``'s only real change from the dense version: a sparse
CSR ``guidance_graph``/``matrix`` graph can't be row-indexed with
``adjacency[i]`` the way a dense boolean ``(n, n)`` tensor can - row ``i``'s
neighbor list is read directly from CSR's own ``crow_indices()``/
``col_indices()`` slice instead, which is *simpler* than the dense boolean-
row-AND it replaces, not more complex. The default adjacency (when no
``guidance_graph`` is given) uses ``kernels.depth_neighborhood
.sparse_depth_neighborhood(matrix, 1)`` instead of dense ``depth_neighborhood``.
``hcr_operator``/``cr_rate`` are otherwise textually identical to the dense
module - still inherently CPU-sequential in ``_independent_set_of``'s greedy
MIS loop (dense or sparse), since each pick invalidates neighbors for every
later pick; the sparse version's value is avoiding an ``O(n^2)`` dense
adjacency, not a GPU-parallel win (there isn't one available here, sparse or
dense).

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11]: eq. 3.3 (F-relaxation operator), eq. 3.4 (asymptotic
      rate estimate), Remark 4.2 (candidate-set measure sigma_i), Algorithm 1
      (the CR coarsening outer loop).
    - Brandt, Brannick, Kahl, Livshits (2014). Bootstrap Algebraic
      Multigrid: status report, open problems, and outlook. arXiv:1406.1819.
      Cited as [STATUS14]: Algorithm 2.1, verbatim with [AD11] Algorithm 1.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from torchalg.sparse.kernels.depth_neighborhood import sparse_depth_neighborhood

_Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Shape of a ``SmootherBase.smooth``-style relaxation callable: ``(A, rhs, x, steps) -> x``."""

# TODO(parallel-algorithm): Evaluate a separate heavy-edge-matching coarsening
# strategy for a tensor-parallel Bootstrap-AMG setup path. It must not be
# presented as a parallel implementation of compatible relaxation: matching
# changes the outer coarsening algorithm and requires independent quality and
# convergence benchmarks. Ganesan and Shah, "SParSH-AMG: A library for hybrid
# CPU-GPU algebraic multigrid and preconditioned iterative methods,"
# arXiv:2007.00056, https://arxiv.org/abs/2007.00056.


def hcr_operator(
    matrix: torch.Tensor,
    coarse_mask: torch.Tensor,
    relaxation: _Relaxation,
    sweeps: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """F-relaxation form of compatible relaxation on ``A x = 0`` ([AD11] eq. 3.3).

    Args:
        matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
            ``(n, n)``.
        coarse_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``C``-points.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x`` (e.g. the sparse ``GaussSeidelSmoother().smooth``).
        sweeps (int): Number of relaxation sweeps.
        start (torch.Tensor | None): Initial iterate, shape ``(n,)``; a
            uniform ``[0, 1)`` random vector on ``matrix``'s dtype/device if
            ``None``.

    Returns:
        torch.Tensor: The relaxed error vector ``e``, shape ``(n,)``, exactly
        zero at every coarse-marked entry.
    """
    n = matrix.shape[0]
    zero_rhs = torch.zeros(n, dtype=matrix.dtype, device=matrix.device)
    x = start if start is not None else torch.rand(n, dtype=matrix.dtype, device=matrix.device)
    x = x.clone()
    x[coarse_mask] = 0.0
    for _ in range(sweeps):
        x = relaxation(matrix, zero_rhs, x, 1)
        x[coarse_mask] = 0.0
    return x


def cr_rate(
    matrix: torch.Tensor,
    coarse_mask: torch.Tensor,
    relaxation: _Relaxation,
    sweeps: int,
    draw: Callable[[int], torch.Tensor],
) -> tuple[float, torch.Tensor]:
    """Estimate the HCR asymptotic convergence rate rho_f ([AD11] eq. 3.4).

    Args:
        matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
            ``(n, n)``.
        coarse_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``C``-points.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``.
        sweeps (int): Number of CR sweeps ``nu``.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.

    Returns:
        tuple[float, torch.Tensor]: ``(rho_f, e)``, the estimated asymptotic
        rate and the final relaxed error vector (coarse entries exactly
        zero).
    """
    start = draw(matrix.shape[0]).to(dtype=matrix.dtype, device=matrix.device).clone()
    start[coarse_mask] = 0.0
    initial_norm = torch.linalg.norm(start)
    e = hcr_operator(matrix, coarse_mask, relaxation, sweeps, start=start)
    if initial_norm == 0.0:
        return 0.0, e
    final_norm = torch.linalg.norm(e)
    rho_f = (final_norm / initial_norm).item() ** (1.0 / sweeps)
    return rho_f, e


def _independent_set_of(
    candidates: torch.Tensor,
    matrix: torch.Tensor,
    priority: torch.Tensor,
    guidance_graph: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy independent set over ``candidates``, ordered by descending ``priority``.

    Visits nodes in descending ``priority`` order; a still-eligible node is
    added to the set and its graph neighbors are removed from further
    consideration. The adjacency guiding that removal is ``guidance_graph``
    when given (e.g. the algebraic-distance-guided strength graph ``M_d``,
    [AD11] eq. 4.4, which ``BAMGCoarsening`` wires in), else
    ``sparse_depth_neighborhood(matrix, 1)`` - the sparse sibling of the
    dense version's plain-matrix-graph default.

    Only row ``i`` of the adjacency is consulted when removing neighbors, so
    the graph is treated as undirected. Callers passing a directional graph
    (such as the algebraic-distance strength graph, where ``r_ij != r_ji``)
    must symmetrize it themselves.

    Args:
        candidates (torch.Tensor): Boolean mask, shape ``(n,)``, the
            eligible node set ``Z``.
        matrix (torch.Tensor): Sparse CSR graph source; ignored in favor of
            ``guidance_graph`` when the latter is given.
        priority (torch.Tensor): Per-node priority (``sigma_i``), shape
            ``(n,)``; higher is visited first.
        guidance_graph (torch.Tensor | None): Sparse CSR boolean adjacency,
            shape ``(n, n)``, overriding ``matrix``'s own graph as the
            independent-set guide; ``None`` keeps the default matrix-graph
            behavior.

    Returns:
        torch.Tensor: Boolean mask, shape ``(n,)``, the selected independent
        subset of ``candidates``.
    """
    n = matrix.shape[0]
    device = matrix.device
    adjacency = (
        guidance_graph if guidance_graph is not None else sparse_depth_neighborhood(matrix, 1)
    )
    # Greedy MIS is inherently sequential, same as the dense sibling - CPU-
    # resident for the same reason (avoid one host<->device sync per
    # candidate node). CSR's own crow/col slice gives row i's neighbor list
    # directly - simpler than the dense boolean-row-AND it replaces, not
    # more complex.
    adjacency_cpu = adjacency.cpu()
    crow = adjacency_cpu.crow_indices()
    col = adjacency_cpu.col_indices()
    order = torch.argsort(priority, descending=True).cpu().tolist()
    eligible = candidates.cpu().clone()
    selected = torch.zeros(n, dtype=torch.bool)
    for i in order:
        if not eligible[i]:
            continue
        selected[i] = True
        eligible[i] = False
        neighbors = col[crow[i] : crow[i + 1]]
        eligible[neighbors] = False
    return selected.to(device)


def compatible_relaxation_coarsening(
    matrix: torch.Tensor,
    relaxation: _Relaxation,
    *,
    nu: int = 5,
    delta: float = 0.7,
    initial_coarse: torch.Tensor | None = None,
    draw: Callable[[int], torch.Tensor],
    max_iterations: int = 100,
    guidance_graph: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatible-relaxation coarsening outer loop (Algorithm 2.1, [AD11]/[STATUS14]).

    Args:
        matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
            ``(n, n)``.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``.
        nu (int): CR sweeps performed per stage (paper default ``nu=5``).
        delta (float): CR stopping tolerance; coarsening halts once
            ``rho_f <= delta`` (paper default ``0.7``).
        initial_coarse (torch.Tensor | None): ``C_0``, boolean mask, shape
            ``(n,)``, ``True`` at initial ``C``-points; ``None`` is the empty
            set (all ``False``).
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.
        max_iterations (int): Defensive cap (not part of Algorithm 2.1) on
            the number of outer-loop stages before giving up.
        guidance_graph (torch.Tensor | None): Sparse CSR boolean adjacency,
            shape ``(n, n)``, overriding ``matrix``'s own graph as the
            independent-set guide (e.g. the algebraic-distance strength
            graph ``M_d``); ``None`` keeps the default matrix-graph
            behavior.

    Returns:
        torch.Tensor: Boolean mask, shape ``(n,)``, ``True`` at the final
        ``C``-points.

    Raises:
        RuntimeError: If ``rho_f`` has not reached ``delta`` after
            ``max_iterations`` stages.
    """
    n = matrix.shape[0]
    coarse_mask = (
        initial_coarse.clone()
        if initial_coarse is not None
        else torch.zeros(n, dtype=torch.bool, device=matrix.device)
    )
    rho_f, e = cr_rate(matrix, coarse_mask, relaxation, nu, draw)
    iterations = 0
    while rho_f > delta:
        if iterations >= max_iterations:
            raise RuntimeError(
                f"compatible_relaxation_coarsening did not converge: "
                f"rho_f={rho_f!r} still above delta={delta!r} after "
                f"max_iterations={max_iterations!r} stages"
            )
        sigma = e.abs() / e.abs().max()
        tol = 1.0 - rho_f
        candidates = ~coarse_mask & (sigma > tol)
        coarse_mask = coarse_mask | _independent_set_of(
            candidates, matrix, sigma, guidance_graph=guidance_graph
        )
        rho_f, e = cr_rate(matrix, coarse_mask, relaxation, nu, draw)
        iterations += 1
    return coarse_mask
