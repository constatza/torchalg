"""History tracking for iterative solvers.

Ported from ``neuralls.domain.solver.models.history`` (see ``docs/plan.md``)
with ``NDArray`` fields translated to ``torch.Tensor``. This module defines:

- ``DirectionHistory``: sliding window of search directions, matrix-vector
  products and their curvatures (for FCG), held in preallocated ``(m, n)``
  buffers; ``DirectionWindow`` is the contiguous view orthogonalization
  consumes.
- ``ResidualHistory``: Residual norms (absolute and relative); immutable.

Theory:
    Krylov methods like FCG require maintaining a window of search
    directions for orthogonalization. Keeping the window in contiguous
    buffers lets one matrix-vector product replace a Python loop over the
    stored pairs.

Design:
    - ``DirectionHistory`` is an append-only accumulator owned by one solve
      (mutation is what makes it allocation-free per iteration);
      ``ResidualHistory`` stays frozen with tuple storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DirectionWindow:
    """Contiguous view of the stored ``(d_j, q_j)`` pairs, oldest first.

    Row ``j`` of ``d`` and ``q`` is the ``j``-th stored direction and its
    product ``q_j = A d_j``. ``curvature[j] = (d_j, q_j)`` is the A-norm
    squared of that direction, computed once when the pair was stored.
    The tensors are views into the history's buffers (valid until the next
    ``DirectionHistory.append``), except when autograd is tracking the
    stored pairs, where they are detached copies of the buffers.

    Attributes:
        d: Search directions, shape ``(k, n)``.
        q: Matrix-vector products ``A d_j``, shape ``(k, n)``.
        curvature: ``(d_j, q_j)``, shape ``(k,)``.
    """

    d: torch.Tensor
    q: torch.Tensor
    curvature: torch.Tensor

    @classmethod
    def empty(cls, like: torch.Tensor) -> DirectionWindow:
        """Return a zero-pair window whose dtype, device and width match ``like``."""
        n = like.shape[-1]
        return cls(
            d=torch.empty((0, n), dtype=like.dtype, device=like.device),
            q=torch.empty((0, n), dtype=like.dtype, device=like.device),
            curvature=torch.empty(0, dtype=like.dtype, device=like.device),
        )

    def __len__(self) -> int:
        """Return the number of stored pairs ``k``."""
        return self.d.shape[0]

    def tail(self, count: int) -> DirectionWindow:
        """Return the newest ``count`` pairs (all of them if fewer are stored)."""
        start = max(len(self) - count, 0)
        return DirectionWindow(
            d=self.d[start:],
            q=self.q[start:],
            curvature=self.curvature[start:],
        )


_INITIAL_ROWS = 8
"""Rows allocated by the first ``append``; the buffers then double up to capacity."""


class DirectionHistory:
    """Sliding window of search directions and matrix-vector products.

    Stores the last ``max_size`` pairs ``(d_j, q_j = A d_j)`` that Flexible
    Conjugate Gradient (Notay 2000) orthogonalizes against, together with
    each pair's curvature ``(d_j, q_j)`` so it is computed once instead of
    on every later iteration.

    Storage is a pair of contiguous ``(capacity, n)`` buffers, allocated on
    the first ``append`` and grown geometrically up to ``max_size`` plus a
    slack of ``max_size // 2`` rows. New pairs are written after the newest
    row; once the buffer is full, the live window is shifted to the front
    (one ``(max_size, n)`` copy every ``max_size // 2`` appends) so the
    window is always one contiguous slice that a single matrix-vector
    product can consume. Memory is therefore ``O(1.5 * max_size * n)``
    regardless of the iteration count, and nothing is allocated per
    iteration beyond the stored copies.

    Unlike the frozen solver state, this object is an append-only
    accumulator owned by exactly one solve: ``append`` mutates it in
    place, and the ``DirectionWindow`` views from earlier calls must not be
    used after a later ``append``. The buffers copy their inputs, so stored
    pairs never alias tensors the caller keeps.

    Theory (Notay 2000):
        FCG orthogonalization formula::

            d_i = w_i - sum_{j=i-m}^{i-1} [(w_i, q_j) / (d_j, q_j)] d_j

        This requires storing the last ``m`` directions and products, where
        ``m`` is the orthogonalization window size.

    Attributes:
        max_size: Maximum window size; older pairs are dropped beyond it.
        total_updates: Total number of pairs ever appended (monotonic, not
            truncated).

    Example:
        >>> history = DirectionHistory.empty(max_size=10)
        >>> history.append(d, q, curvature)
        >>> history.append(d2, q2, curvature2)
        >>> len(history)  # 2
    """

    __slots__ = (
        "_curvature",
        "_d",
        "_end",
        "_q",
        "_start",
        "_tracks_grad",
        "max_size",
        "total_updates",
    )

    def __init__(self, max_size: int = 10) -> None:
        """Create an empty history keeping at most ``max_size`` pairs."""
        self.max_size = max_size
        self.total_updates = 0
        self._d: torch.Tensor | None = None
        self._q: torch.Tensor | None = None
        self._curvature: torch.Tensor | None = None
        self._start = 0
        self._end = 0
        self._tracks_grad = False

    @classmethod
    def empty(cls, max_size: int = 10) -> DirectionHistory:
        """Create an empty history with the given maximum window size."""
        return cls(max_size=max_size)

    @property
    def _full_capacity(self) -> int:
        """Rows needed for the full window plus compaction slack."""
        return self.max_size + max(1, self.max_size // 2)

    def append(self, d: torch.Tensor, q: torch.Tensor, curvature: torch.Tensor) -> None:
        """Store ``(d, q)`` as the newest pair, dropping the oldest beyond ``max_size``.

        Args:
            d: New search direction, shape ``(n,)``.
            q: Its matrix-vector product ``A d``, shape ``(n,)``.
            curvature: ``(d, q)`` as a 0-d tensor, computed by the caller
                (the CG step already has it).
        """
        if self.max_size < 1:
            raise ValueError(f"cannot append to a history with max_size={self.max_size} < 1")
        if self._d is None:
            self._allocate(min(_INITIAL_ROWS, self._full_capacity), like=d)
        elif self._end == self._buffer_rows:
            self._make_room()
        assert self._d is not None
        assert self._q is not None
        assert self._curvature is not None
        row = self._end
        self._d[row].copy_(d)
        self._q[row].copy_(q)
        self._curvature[row].copy_(curvature)
        self._tracks_grad = self._tracks_grad or (
            torch.is_grad_enabled()
            and (d.requires_grad or q.requires_grad or curvature.requires_grad)
        )
        self._end += 1
        self.total_updates += 1
        self._start = max(self._start, self._end - self.max_size)

    def window(self, like: torch.Tensor) -> DirectionWindow:
        """Return the live window (oldest first) as contiguous views.

        ``like`` supplies the dtype/device/width of an empty window when
        nothing has been stored yet.

        When autograd tracks the stored pairs the views are cloned: a later
        in-place ``append`` into the same buffer would otherwise invalidate
        tensors saved for backward.
        """
        if self._d is None or self._q is None or self._curvature is None:
            return DirectionWindow.empty(like)
        live = slice(self._start, self._end)
        window = DirectionWindow(
            d=self._d[live],
            q=self._q[live],
            curvature=self._curvature[live],
        )
        if not self._tracks_grad:
            return window
        return DirectionWindow(
            d=window.d.clone(),
            q=window.q.clone(),
            curvature=window.curvature.clone(),
        )

    def __len__(self) -> int:
        """Return the number of pairs currently stored (not ``total_updates``)."""
        return self._end - self._start

    @property
    def _buffer_rows(self) -> int:
        """Allocated row count (0 before the first append)."""
        return 0 if self._d is None else self._d.shape[0]

    def _allocate(self, rows: int, *, like: torch.Tensor) -> None:
        """Allocate fresh buffers of ``rows`` rows, copying any live window to the front."""
        n = like.shape[-1]
        old = None if self._d is None else self._live_buffers()
        new_d = torch.empty((rows, n), dtype=like.dtype, device=like.device)
        new_q = torch.empty((rows, n), dtype=like.dtype, device=like.device)
        new_curvature = torch.empty(rows, dtype=like.dtype, device=like.device)
        live = len(self)
        if old is not None:
            for new_buffer, old_buffer in zip((new_d, new_q, new_curvature), old, strict=True):
                new_buffer[:live] = old_buffer
        self._d, self._q = new_d, new_q
        self._curvature = new_curvature
        self._start, self._end = 0, live

    def _live_buffers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the live rows of the three buffers, for relocation."""
        assert self._d is not None
        assert self._q is not None
        assert self._curvature is not None
        live = slice(self._start, self._end)
        return self._d[live], self._q[live], self._curvature[live]

    def _make_room(self) -> None:
        """Grow the buffers, or shift the live window to the front once at capacity."""
        assert self._d is not None
        rows = self._buffer_rows
        if rows < self._full_capacity:
            self._allocate(min(2 * rows, self._full_capacity), like=self._d[0])
            return
        live = len(self)
        for buffer in (self._d, self._q, self._curvature):
            assert buffer is not None
            # Clone first: the source and destination slices overlap.
            buffer[:live] = buffer[self._start : self._end].clone()
        self._start, self._end = 0, live


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

    def add(
        self, norm_abs: torch.Tensor | float, norm_rel: torch.Tensor | float
    ) -> ResidualHistory:
        """Return new history with norms added (immutable update).

        Args:
            norm_abs (torch.Tensor | float): Absolute residual norm
                ``||r_k||_2``. May be a 0-d tensor; cast to float here.
            norm_rel (torch.Tensor | float): Relative residual norm
                ``||r_k||_2 / ||b||_2``. May be a 0-d tensor; cast to float
                here.

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
