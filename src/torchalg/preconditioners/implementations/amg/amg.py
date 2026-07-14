"""AMG preconditioner.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.amg``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``.

Buffer/hierarchy design (see ``docs/plan.md``'s Stage 5 entry):
    ``AMGPreconditioner`` is the first AMG-tier class holding real internal
    tensor state, but unlike Jacobi/ILU/IC0/ICholesky (Stages 3-4) that
    state isn't a single fixed-shape tensor computed once at construction -
    it's a *lazily built, variable-length* list of per-level matrices whose
    shapes depend on the coarsening strategy and aren't known until the
    first ``apply()`` call (and are invalidated/rebuilt whenever
    ``bind_inputs`` is called, e.g. for neural coarsening).

    Registering one ``nn.Module`` buffer per hierarchy level with indexed
    names (``f"level_{i}_matrix"``) would require deregistering and
    re-registering a variable number of buffers on every rebuild - more
    machinery than the problem needs, and nothing else in this codebase
    does that. Instead:

    - The finest-level matrix ``A`` (fixed shape, known at construction -
      exactly the case ``register_buffer`` is designed for) is registered
      as a buffer, so it participates in ``.to(device/dtype)`` propagation
      like every other stateful preconditioner in this package.
    - The built ``MultigridHierarchy`` (a plain frozen-dataclass tree of
      tensors *derived* from that buffer) is kept as a plain, non-buffer
      attribute, since it is a cache, not owned state - it's already
      invalidated by ``bind_inputs`` per the reference's own design.
    - ``_apply`` (the ``nn.Module`` hook that ``.to()``/``.cuda()``/
      ``.double()`` etc. all funnel through) is overridden to also
      invalidate that cache after the buffer moves, so a stale-device
      hierarchy can never be reused - the next ``apply()`` rebuilds it
      from the now-moved ``A`` buffer automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ...base import BindableInputs, Preconditioner, PreconditionerContext
from .hierarchy import MultigridHierarchy, MultigridLevel

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from .protocols import CoarseningStrategy, MultigridCycle


class AMGPreconditioner(Preconditioner, nn.Module):
    """Algebraic Multigrid preconditioner.

    Builds a multigrid hierarchy from a system matrix and applies a multigrid
    cycle as the preconditioner. Structurally implements ``BindableInputs``
    (always safe to call ``bind_inputs``/``extra_input_names``); when the
    coarsening strategy has no extra inputs, these are no-ops.

    Hierarchy construction is **lazy**: it happens on the first ``apply()``
    call. For classical coarsening (no ``bind_inputs`` needed), this is
    effectively immediate. For neural coarsening, ``bind_inputs`` must be
    called before the first ``apply()``.

    Args:
        matrix (torch.Tensor): System matrix A (n x n).
        coarsening (CoarseningStrategy): Strategy that builds each coarse
            level.
        cycle (MultigridCycle): Multigrid cycle to apply as preconditioner.
        n_levels (int): Total number of levels; must be at least 2
            (2 = one coarse grid).
        linear (bool): Whether the preconditioner is linear (False for
            neural AMG). Controls ``requires_flexible_cg``.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        coarsening: CoarseningStrategy,
        cycle: MultigridCycle,
        n_levels: int = 2,
        linear: bool = True,
    ) -> None:
        """Store the coarsening/cycle strategies and register the finest-level matrix.

        Args:
            matrix (torch.Tensor): System matrix A (n x n).
            coarsening (CoarseningStrategy): Strategy that builds each
                coarse level.
            cycle (MultigridCycle): Multigrid cycle to apply as
                preconditioner.
            n_levels (int): Total number of levels; must be at least 2
                (2 = one coarse grid).
            linear (bool): Whether the preconditioner is linear (False for
                neural AMG). Controls ``requires_flexible_cg``.
        """
        nn.Module.__init__(self)
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2 for AMG coarsening.")
        self._matrix: torch.Tensor
        self.register_buffer("_matrix", matrix)
        self._coarsening = coarsening
        self._cycle = cycle
        self._n_levels = n_levels
        self._linear = linear
        self._hierarchy: MultigridHierarchy | None = None

    def _apply(self, fn: Callable, recurse: bool = True) -> Self:
        """Invalidate the cached hierarchy whenever the buffer moves.

        ``nn.Module.to()``/``.cuda()``/``.double()``/etc. all call this
        internally to move buffers/parameters. The cached hierarchy's
        tensors are derived from ``_matrix`` but are not themselves
        buffers, so without this override a device/dtype change after the
        first ``apply()`` would silently leave a stale-device hierarchy in
        place. Overriding ``_apply`` guarantees the next ``apply()`` call
        rebuilds the hierarchy from the freshly-moved ``_matrix`` buffer.

        Args:
            fn (Callable): Conversion function applied to each
                parameter/buffer (see ``nn.Module._apply``).
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Self: This module, per ``nn.Module._apply``'s contract.
        """
        result = super()._apply(fn, recurse=recurse)
        self._hierarchy = None
        return result

    # BindableInputs structural implementation ---------------------------------

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Extra input names required by the coarsening strategy, if any.

        Returns:
            tuple[str, ...]: Names from the coarsening strategy when it
                implements ``BindableInputs``, else an empty tuple.
        """
        if isinstance(self._coarsening, BindableInputs):
            return self._coarsening.extra_input_names
        return ()

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Forward extra domain inputs to the coarsening strategy.

        Also invalidates the cached hierarchy so it is rebuilt on the next
        ``apply()`` call with the newly bound inputs.

        Args:
            **inputs (torch.Tensor): Named tensors (e.g., positions, theta).
        """
        if isinstance(self._coarsening, BindableInputs):
            self._coarsening.bind_inputs(**inputs)
        self._hierarchy = None

    # Preconditioner -----------------------------------------------------------

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether Flexible CG is required.

        Returns:
            bool: ``True`` for neural AMG (non-linear cycle); ``False`` for
                classical AMG.
        """
        return not self._linear

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply one V-cycle (or configured cycle) to the residual.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Ignored (AMG cycle is
                stateless w.r.t. solver iteration).

        Returns:
            torch.Tensor: Approximate solution to Az = r from one multigrid
                cycle.
        """
        if self._hierarchy is None:
            self._hierarchy = self._build_hierarchy(self._matrix)
        return self._cycle.apply(self._hierarchy, residual)

    # Private ------------------------------------------------------------------

    def _build_hierarchy(self, matrix: torch.Tensor) -> MultigridHierarchy:
        """Iteratively apply coarsening to build all levels.

        Args:
            matrix (torch.Tensor): Finest-level system matrix.

        Returns:
            MultigridHierarchy: Frozen hierarchy of levels from fine to
                coarse.
        """
        levels: list[MultigridLevel] = []
        current_matrix = matrix
        for _ in range(self._n_levels - 1):
            coarse_matrix, transfer = self._coarsening.build_transfer(current_matrix)
            levels.append(MultigridLevel(matrix=current_matrix, transfer=transfer))
            current_matrix = coarse_matrix
        levels.append(MultigridLevel(matrix=current_matrix, transfer=None))
        return MultigridHierarchy(levels=tuple(levels))
