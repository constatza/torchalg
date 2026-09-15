"""POD basis construction (snapshot method, via thin SVD)."""

from __future__ import annotations

import torch


def _resolve_rank(singular_values: torch.Tensor, rank: float, max_rank: int) -> int:
    """Resolve a mode count from a fixed count or an energy-capture threshold.

    Args:
        singular_values (torch.Tensor): Singular values, descending (from
            ``torch.linalg.svd``).
        rank (int | float): Fixed mode count (int) or minimum cumulative
            captured energy (float in (0, 1], e.g. 0.9999 - matches
            Nikolopoulos et al. 2022, §4.1: "r=8 ... over 99.99% of the
            variance").
        max_rank (int): Upper bound (``min(n_samples, n_dofs)``).

    Returns:
        int: Resolved mode count, clamped to ``max_rank``.

    Raises:
        ValueError: If ``rank`` is a float outside (0, 1], or an int
            exceeding ``max_rank``.
    """
    if isinstance(rank, float):
        if not (0.0 < rank <= 1.0):
            raise ValueError(f"rank as an energy threshold must be in (0, 1], got {rank}")
        energy = singular_values.pow(2)
        cumulative = torch.cumsum(energy, dim=0) / energy.sum()
        resolved = int((cumulative < rank).sum().item()) + 1
        return min(resolved, max_rank)

    if rank > max_rank:
        raise ValueError(f"rank ({rank}) cannot exceed min(n_samples, n_dofs) ({max_rank})")
    return rank


def compute_pod_basis(
    snapshots: torch.Tensor,
    rank: float,
    dtype: torch.dtype | None = None,
    row_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a truncated POD basis from a snapshot ensemble (snapshot method).

    Implements the reduced-basis construction of Nikolopoulos et al. (2022),
    §3.3, eq. 26-27, via the thin SVD of the snapshot matrix rather than
    eigendecomposing a Gram matrix: forming UUt or UtU squares the condition
    number of U, which hurts precision precisely when POD is most useful
    (highly correlated/redundant snapshots). ``torch.linalg.svd`` computes the
    same POD modes without ever forming that Gram matrix, at the same
    asymptotic cost as the snapshot method for n_samples << n_dofs (the
    paper's own justification for avoiding the full n_dofs x n_dofs
    eigenproblem still holds), and returns singular values already sorted
    descending and non-negative - no manual sort or clamp needed.

    Unlike the reference (``neuralls.domain.solver``), this port has no
    numpy in/out boundary: ``torch.linalg.svd`` was always an internal
    implementation detail in the reference too (it converted numpy -> torch
    -> numpy purely to cross the array-library boundary shared by the rest
    of that codebase), so removing the boundary is a pure simplification,
    not a behavior change. Two consequences of dropping that boundary:

    - **dtype**: still preserves the caller's input dtype by default (the
      reference's deliberate choice, called out explicitly in its own
      docstring) - unlike the rest of ``torchalg``'s solver stack, which
      defaults to ``float64`` throughout. ``dtype`` remains an explicit
      opt-in override, not a forced upcast.
    - **device**: the reference hardcoded ``device="cpu"`` because crossing
      the numpy boundary required picking *some* device to land the
      intermediate torch tensor on. With that boundary gone, there is
      nothing left to hardcode: ``snapshots`` is already a ``torch.Tensor``
      wherever the caller put it, and the SVD (and the returned basis)
      simply run on that same device - preserved implicitly, exactly like
      dtype, with no separate ``device`` parameter needed.

    Args:
        snapshots (torch.Tensor): Solution snapshot ensemble, shape
            (n_samples, n_dofs) - one row per high-fidelity solution vector.
        rank (int | float): Either a fixed number of POD modes to retain
            (int, largest singular values first), or a minimum fraction of
            cumulative captured energy to retain (float in (0, 1] - e.g.
            0.9999 retains the smallest r with sum(sigma_1^2..sigma_r^2) /
            sum(sigma_i^2) >= rank, matching how the paper itself selects r).
        dtype (torch.dtype | None): Target dtype for the SVD. Defaults to
            ``snapshots.dtype`` - the caller's own precision is preserved
            unless an explicit override is given.
        row_scales (torch.Tensor | None): Optional per-snapshot scale,
            shape (n_samples,). When given, each row is multiplied by its
            scale before the SVD: ``snapshots' = row_scales[:, None] *
            snapshots``, so ``snapshots'^T snapshots' = sum_k
            row_scales_k^2 e_k e_k^T`` - a weighted covariance - without
            changing the SVD's (Euclidean) inner product. ``None`` (default)
            reproduces the unweighted basis exactly. See
            ``preconditioners.implementations.pod.weighting`` for schemes
            that compute this vector (raw/L2/A-normalized,
            smoother-persistence).

    Returns:
        torch.Tensor: POD basis Phi_r, shape (n_dofs, r), in the resolved
            dtype, on the same device as ``snapshots``.

    Raises:
        ValueError: If ``rank`` is a float outside (0, 1], an int exceeding
            ``min(n_samples, n_dofs)``, or ``row_scales`` has a length other
            than ``n_samples``.

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos,
          V. (2022). AI-enhanced iterative solvers for accelerating the
          solution of large-scale parametrized systems. arXiv:2207.02543.
    """
    n_samples, n_dofs = snapshots.shape
    max_rank = min(n_samples, n_dofs)

    snapshots_t = snapshots if dtype is None else snapshots.to(dtype=dtype)

    if row_scales is not None:
        if row_scales.shape != (n_samples,):
            raise ValueError(
                f"row_scales must have shape ({n_samples},), got {tuple(row_scales.shape)}"
            )
        snapshots_t = snapshots_t * row_scales.to(dtype=snapshots_t.dtype).unsqueeze(-1)

    _, s, vh = torch.linalg.svd(snapshots_t, full_matrices=False)  # vh: (max_rank, n_dofs)
    resolved_rank = _resolve_rank(s, rank, max_rank)
    basis = vh[:resolved_rank].T  # (n_dofs, r), columns orthonormal by construction

    return basis
