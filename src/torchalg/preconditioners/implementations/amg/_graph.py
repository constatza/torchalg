"""Device-resident graph kernels shared by AMG setup algorithms."""

from __future__ import annotations

import torch


def depth_neighborhood(matrix: torch.Tensor, depth: int) -> torch.Tensor:
    """Boolean off-diagonal adjacency of ``matrix`` raised to ``depth``.

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d``, ``>= 1``.

    Returns:
        torch.Tensor: Boolean neighborhood matrix, shape ``(n, n)``.
    """
    n = matrix.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    adjacency = (matrix != 0) & off_diagonal
    reachable = adjacency
    for _ in range(depth - 1):
        reachable = (reachable.to(torch.float32) @ adjacency.to(torch.float32)) > 0
    return reachable & off_diagonal
