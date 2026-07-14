"""History tracking for iterative solvers.

Ported from ``neuralls.domain.solver.models.history`` (see ``docs/plan.md``)
with ``NDArray`` fields translated to ``torch.Tensor`` and the numpy copy
inside ``DirectionHistory.add`` translated to ``torch.Tensor.clone()``. This
module defines immutable history dataclasses for tracking algorithm state:

- ``DirectionHistory``: Search directions and matrix-vector products (for FCG).
- ``ResidualHistory``: Residual norms (absolute and relative).

All classes use frozen, slotted dataclasses for immutability, following
functional programming principles.

Theory:
    Krylov methods like FCG require maintaining a window of search
    directions for orthogonalization. Using immutable tuples ensures no
    accidental mutation and simplifies reasoning about algorithm
    correctness.

Design:
    - Immutable (``frozen=True``) for thread safety and predictable behavior.
    - Tuple-based storage (immutable sequences).
    - Factory methods for creating empty instances.
    - Update methods return new instances (functional style).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DirectionHistory:
    """Immutable history of search directions and matrix-vector products.

    Maintains a sliding window of search directions (``d_i``) and their
    corresponding matrix-vector products (``q_i = A d_i``) for algorithms
    like Flexible Conjugate Gradient that require orthogonalization against
    previous directions.

    When ``max_size`` is reached, older directions are dropped to maintain
    the window size.

    For full orthogonalization (FCG(inf)), set ``max_size`` to a large value
    (e.g. the reference's ``DEFAULT_FCG_HISTORY_LIMIT=200``) or the expected
    maximum iteration count.

    Theory (Notay 2000):
        FCG orthogonalization formula::

            d_i = w_i - sum_{j=i-m}^{i-1} [(w_i, q_j) / (d_j, q_j)] d_j

        This requires storing the last ``m`` directions and products, where
        ``m`` is the orthogonalization window size.

    Attributes:
        d_vectors (tuple[torch.Tensor, ...]): Previous search directions
            ``[d_{i-m}, ..., d_{i-1}]`` (Notay 2000). Immutable tuple ensures
            no accidental modification.
        q_vectors (tuple[torch.Tensor, ...]): Previous matrix-vector
            products ``[q_{i-m}, ..., q_{i-1}]``. Each ``q_j = A @ d_j``.
            Used in orthogonalization coefficients.
        max_size (int): Maximum window size (effectively unlimited if
            large). Vectors are truncated to keep only the last ``max_size``.
        total_updates (int): Total number of directions added (monotonic,
            not truncated). Used for debugging and statistics.

    Example:
        >>> history = DirectionHistory.empty(max_size=10)
        >>> history = history.add(d, q)
        >>> history = history.add(d2, q2)
        >>> len(history.d_vectors)  # 2
    """

    d_vectors: tuple[torch.Tensor, ...] = ()
    """Previous search directions (immutable tuple) (Notay 2000)."""

    q_vectors: tuple[torch.Tensor, ...] = ()
    """Previous matrix-vector products q_j = A @ d_j (immutable tuple)."""

    max_size: int = 10
    """Maximum window size for truncation."""

    total_updates: int = 0
    """Total number of directions ever added (monotonic)."""

    @classmethod
    def empty(cls, max_size: int = 10) -> DirectionHistory:
        """Create empty history with specified maximum size.

        Args:
            max_size (int): Maximum number of directions to store. Defaults
                to ``10``.

        Returns:
            DirectionHistory: New history with no vectors.

        Example:
            >>> history = DirectionHistory.empty(max_size=20)
            >>> history.d_vectors
            ()
        """
        return cls(d_vectors=(), q_vectors=(), max_size=max_size, total_updates=0)

    def add(self, d: torch.Tensor, q: torch.Tensor) -> DirectionHistory:
        """Return new history with vectors added (immutable update).

        Args:
            d (torch.Tensor): New search direction to add (Notay 2000).
            q (torch.Tensor): New matrix-vector product (``A @ d``) to add.

        Returns:
            DirectionHistory: New history with the vector added and old
                vectors possibly truncated.

        Theory:
            Truncation maintains only the last ``max_size`` vectors to:
            1. Bound memory usage at ``O(max_size * n)``.
            2. Limit orthogonalization cost to ``O(max_size * n)`` per
               iteration.
            3. Prevent numerical error accumulation.

        Example:
            >>> history = DirectionHistory.empty(max_size=2)
            >>> history = history.add(d1, q1)
            >>> history = history.add(d2, q2)
            >>> history = history.add(d3, q3)  # Drops d1, q1
            >>> len(history.d_vectors)  # 2
        """
        # Clone so the stored history never aliases a tensor the caller
        # might later mutate in place (mirrors the reference's
        # `np.asarray(d, copy=True)`).
        new_d = self.d_vectors + (d.clone(),)
        new_q = self.q_vectors + (q.clone(),)

        if len(new_d) > self.max_size:
            new_d = new_d[-self.max_size :]
        if len(new_q) > self.max_size:
            new_q = new_q[-self.max_size :]

        return DirectionHistory(
            d_vectors=new_d,
            q_vectors=new_q,
            max_size=self.max_size,
            total_updates=self.total_updates + 1,
        )

    def __len__(self) -> int:
        """Return number of directions stored (not ``total_updates``).

        Returns:
            int: Current window size (``len(d_vectors)``).
        """
        return len(self.d_vectors)


@dataclass(frozen=True, slots=True)
class ResidualHistory:
    """Immutable history of residual norms.

    Tracks absolute and relative residual norms across iterations for
    convergence monitoring and post-analysis.

    Attributes:
        norms_abs (tuple[float, ...]): Absolute residual norms
            ``||r_k||_2``.
        norms_rel (tuple[float, ...]): Relative residual norms
            ``||r_k||_2 / ||b||_2``.

    Example:
        >>> history = ResidualHistory.empty()
        >>> history = history.add(norm_abs=1.0, norm_rel=0.1)
        >>> history = history.add(norm_abs=0.5, norm_rel=0.05)
        >>> history.norms_abs
        (1.0, 0.5)
    """

    norms_abs: tuple[float, ...] = ()
    """Absolute residual norms ||r_k||_2 (immutable tuple)."""

    norms_rel: tuple[float, ...] = ()
    """Relative residual norms ||r_k||_2 / ||b||_2 (immutable tuple)."""

    @classmethod
    def empty(cls) -> ResidualHistory:
        """Create empty residual history.

        Returns:
            ResidualHistory: New history with no norms.

        Example:
            >>> history = ResidualHistory.empty()
            >>> history.norms_abs
            ()
        """
        return cls(norms_abs=(), norms_rel=())

    def add(self, norm_abs: float, norm_rel: float) -> ResidualHistory:
        """Return new history with norms added (immutable update).

        Args:
            norm_abs (float): Absolute residual norm ``||r_k||_2``.
            norm_rel (float): Relative residual norm
                ``||r_k||_2 / ||b||_2``.

        Returns:
            ResidualHistory: New history with norms appended.

        Example:
            >>> history = ResidualHistory.empty()
            >>> history = history.add(1.0, 0.1)
            >>> history.norms_abs[-1]
            1.0
        """
        return ResidualHistory(
            norms_abs=self.norms_abs + (float(norm_abs),),
            norms_rel=self.norms_rel + (float(norm_rel),),
        )

    def __len__(self) -> int:
        """Return number of residual norms stored.

        Returns:
            int: Length of ``norms_abs`` (same as ``norms_rel``).
        """
        return len(self.norms_abs)
