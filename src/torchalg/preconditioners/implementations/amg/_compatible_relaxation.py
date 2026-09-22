"""Compatible-relaxation (CR) coarsening kernels for Bootstrap AMG.

Pure functions with no coarsening-strategy state, following the module
style of ``_aggregation.py``: the HCR relaxation operator, its
convergence-rate estimate, and the CR coarsening outer loop, each a
transcription of one piece of ``docs/bootstrap-amg.md`` Sec. 2.1.

``docs/bootstrap-amg.md`` Sec. 2.1 is the authoritative spec (transcribed
directly from the primary papers, not a secondary summary); every equation
number cited below refers to it.

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

_Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Shape of a ``SmootherBase.smooth``-style relaxation callable: ``(A, rhs, x, steps) -> x``."""


def hcr_operator(
    matrix: torch.Tensor,
    coarse_mask: torch.Tensor,
    relaxation: _Relaxation,
    sweeps: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """F-relaxation form of compatible relaxation on ``A x = 0`` ([AD11] eq. 3.3).

    Applies ``relaxation`` one sweep at a time, projecting the coarse-marked
    entries to zero after every sweep so the coarse variables stay exactly
    fixed at zero throughout - the effect of ``E_ff = I - M_ff^{-1} B_ff``
    without requiring a separate F-only relaxation routine.

    Args:
        matrix (torch.Tensor): System matrix ``A``, shape ``(n, n)``.
        coarse_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``C``-points.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x`` (e.g. ``GaussSeidelSmoother().smooth``).
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

    Runs ``hcr_operator`` for ``sweeps`` sweeps from a random start drawn via
    ``draw``, then estimates ``rho_f ~= (||e^sweeps|| / ||e^0||)^(1/sweeps)``
    - the eq. 3.4 root form, evaluated with the actual sweep count (small,
    typically ``nu=5``) rather than assuming ``nu`` large enough for the
    single-step ratio to have settled.

    Args:
        matrix (torch.Tensor): System matrix ``A``, shape ``(n, n)``.
        coarse_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``C``-points.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``.
        sweeps (int): Number of CR sweeps ``nu``.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n`` (``adaptive.py``'s
            ``_Params.draw`` convention).

    Returns:
        tuple[float, torch.Tensor]: ``(rho_f, e)``, the estimated asymptotic
        rate and the final relaxed error vector (coarse entries exactly
        zero).
    """
    start = draw(matrix.shape[0]).to(dtype=matrix.dtype, device=matrix.device).clone()
    start[coarse_mask] = 0.0
    initial_norm = torch.linalg.norm(start)
    e = hcr_operator(matrix, coarse_mask, relaxation, sweeps, start=start)
    final_norm = torch.linalg.norm(e)
    rho_f = (final_norm / initial_norm).item() ** (1.0 / sweeps)
    return rho_f, e


def _independent_set_of(
    candidates: torch.Tensor,
    matrix: torch.Tensor,
    priority: torch.Tensor,
) -> torch.Tensor:
    """Greedy independent set over ``candidates``, ordered by descending ``priority``.

    Visits nodes in descending ``priority`` order; a still-eligible node is
    added to the set and its graph neighbors (``matrix``'s nonzero
    off-diagonal pattern) are removed from further consideration - a
    placeholder for the algebraic-distance-guided strength graph ``M_d``
    ([AD11] eq. 4.4), which Task 4's ``BAMGCoarsening`` wires in in place of
    the plain matrix graph used here.

    Args:
        candidates (torch.Tensor): Boolean mask, shape ``(n,)``, the
            eligible node set ``Z``.
        matrix (torch.Tensor): Graph source; a nonzero off-diagonal entry
            ``(i, j)`` marks an edge, shape ``(n, n)``.
        priority (torch.Tensor): Per-node priority (``sigma_i``), shape
            ``(n,)``; higher is visited first.

    Returns:
        torch.Tensor: Boolean mask, shape ``(n,)``, the selected independent
        subset of ``candidates``.
    """
    n = matrix.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    adjacency = (matrix != 0) & off_diagonal
    eligible = candidates.clone()
    selected = torch.zeros(n, dtype=torch.bool, device=matrix.device)
    for i in torch.argsort(priority, descending=True).tolist():
        if not eligible[i]:
            continue
        selected[i] = True
        eligible[i] = False
        eligible &= ~adjacency[i]
    return selected


def compatible_relaxation_coarsening(
    matrix: torch.Tensor,
    relaxation: _Relaxation,
    *,
    nu: int = 5,
    delta: float = 0.7,
    initial_coarse: torch.Tensor | None = None,
    draw: Callable[[int], torch.Tensor],
    max_iterations: int = 100,
) -> torch.Tensor:
    """Compatible-relaxation coarsening outer loop (Algorithm 2.1, [AD11]/[STATUS14]).

    Grows the coarse set ``C`` from ``initial_coarse`` until the HCR
    convergence rate ``rho_f`` drops to ``delta`` or below: each iteration
    marks candidate points ``i`` whose relative error component ``sigma_i =
    |e_i| / ||e||_inf`` exceeds ``tol = 1 - rho_f`` ([AD11] Remark 4.2), then
    adds a greedy independent set of those candidates (guided by ``matrix``'s
    graph; see ``_independent_set_of``) to ``C``.

    Algorithm 2.1 itself has no iteration bound - its ``while rho_f > delta``
    is a mathematical description, not production code. ``max_iterations``
    is a defensive cap this port adds on top of it: without one, a candidate
    set ``Z`` that plateaus (stays empty) before ``rho_f`` reaches ``delta``
    would spin forever re-running ``cr_rate`` on an unchanged ``C``.

    Args:
        matrix (torch.Tensor): System matrix ``A``, shape ``(n, n)``.
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
        coarse_mask = coarse_mask | _independent_set_of(candidates, matrix, sigma)
        rho_f, e = cr_rate(matrix, coarse_mask, relaxation, nu, draw)
        iterations += 1
    return coarse_mask
