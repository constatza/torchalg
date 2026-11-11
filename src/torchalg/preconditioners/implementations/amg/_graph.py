"""Device-resident graph kernels shared by AMG setup algorithms."""

from __future__ import annotations

from collections.abc import Callable

import torch

type NeighborhoodOf = Callable[[torch.Tensor, int], torch.Tensor]
"""``(matrix, depth) -> boolean neighborhood``: ``depth_neighborhood`` or a cache of it."""


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


class NeighborhoodCache:
    """``depth_neighborhood`` memoized per matrix object and depth, for one setup run.

    Every bootstrap pass revisits the finest matrix, whose neighborhoods depend
    on ``A`` alone, so they are evaluated once per run instead of once per pass.
    Holds the matrices (and their ``(n, n)`` boolean results) alive, so it must
    not outlive the run, and a cached matrix must not be mutated in place.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._entries: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, matrix: torch.Tensor, depth: int) -> torch.Tensor:
        """Return the neighborhood of ``matrix`` at ``depth``, evaluating it on first use.

        Args:
            matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
            depth (int): Search depth ``d``, ``>= 1``.

        Returns:
            torch.Tensor: Boolean neighborhood matrix, shared by later calls.
        """
        key = (id(matrix), depth)
        entry = self._entries.get(key)
        if entry is None:
            # ``matrix`` is stored beside its result so its id cannot be recycled.
            entry = (matrix, depth_neighborhood(matrix, depth))
            self._entries[key] = entry
        return entry[1]
