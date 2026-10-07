"""AMG preconditioner.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.amg``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``.
Promoted from ``torchalg.preconditioners.implementations.amg.amg`` into this
shared, format-agnostic package (see ``docs/plan.md``) - nothing in this
class commits to dense or sparse storage; see ``torchalg.multigrid``'s
package docstring.

Buffer/hierarchy design (see ``docs/plan.md``'s Stage 5 entry):
    ``AMGPreconditioner`` is the first AMG-tier class holding real internal
    tensor state, but unlike Jacobi/ILU/IC0/ICholesky (Stages 3-4) that
    state isn't a single fixed-shape tensor built once by ``setup()`` - it's
    a *variable-length* list of per-level matrices whose shapes depend on
    the coarsening strategy and aren't known until ``setup()`` runs (and
    must be rebuilt by calling ``setup()`` again whenever ``bind_inputs``
    changes an extra input, e.g. for neural coarsening).

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
      attribute, since it is a cache, not owned state - rebuilt by the next
      explicit ``setup()`` call, not invalidated/lazily-rebuilt behind the
      caller's back.
    - ``_apply`` (the ``nn.Module`` hook that ``.to()``/``.cuda()``/
      ``.double()`` etc. all funnel through) is overridden to also drop that
      cache after the buffer moves, so a stale-device hierarchy can never be
      reused - callers must call ``setup()`` again after moving a
      preconditioner to rebuild the hierarchy on the now-moved ``A`` buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchalg.preconditioners.base import (
    BindableInputs,
    Preconditioner,
    PreconditionerContext,
    PreconditionerNotReadyError,
)

from .hierarchy import MultigridHierarchy, build_hierarchy

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

    Hierarchy construction is **explicit**: it happens in ``setup(matrix)``,
    which must be called once before the first ``apply()`` call (``apply()``
    raises ``PreconditionerNotReadyError`` otherwise). For neural
    coarsening, ``bind_inputs`` must be called before ``setup()`` so the
    bound inputs are available when the hierarchy is built.

    Args:
        matrix (torch.Tensor): System matrix A (n x n).
        coarsening (CoarseningStrategy): Strategy that builds each coarse
            level.
        cycle (MultigridCycle): Multigrid cycle to apply as preconditioner.
        n_levels (int): Total number of levels; must be at least 2
            (2 = one coarse grid).
        linear (bool): Whether the preconditioner is linear (False for
            neural AMG). Controls ``requires_flexible_cg``.

    Complexity: this engine is format-agnostic (see ``torchalg.multigrid``'s
    package docstring) - the complexity below holds **only when a dense**
    ``CoarseningStrategy`` **is injected** (the dense ``torchalg
    .preconditioners.implementations.amg`` presets). It is **not** the
    textbook ``O(n)`` per V-cycle that multigrid theory gives for a *sparse*
    hierarchy with a bounded coarsening ratio, because every level is then a
    dense ``torch.Tensor``, never a sparse structure, regardless of the fine
    matrix's true sparsity, so the actual cost is dominated by dense linear
    algebra:
        - ``VCycle._vcycle`` (``cycle.py``) computes ``rhs - matrix @ x`` once
          per pre-smooth and once per post-smooth at every level — a dense
          ``(level_size, level_size)`` matvec, ``O(level_size^2)``. Summed
          across levels this is a geometric-ish series dominated by the
          finest level, i.e. ``O(n^2)`` overall per V-cycle (``WCycle``
          doubles the coarse-grid-correction work per level, same finest-level
          dominance).
        - The coarsest-level solve (``torch.linalg.solve`` or
          ``pseudo_inverse_solve``) is a dense direct solve,
          ``O(coarsest_size^3)`` — small in absolute terms when the hierarchy
          bottoms out at a small coarse dimension, but still cubic, not the
          near-free coarsest solve a sparse/iterative coarse solver would give.
    A neural-transfer or neural-coarsening variant adds whatever its network's
    forward-pass cost is on top of this — see the coarsening/transfer class's
    own complexity note.

    When a sparse ``CoarseningStrategy`` is injected instead (e.g.
    ``torchalg.sparse.preconditioners.amg``'s ``AggregationCoarsening``),
    every non-coarsest level's matrix stays sparse CSR, so each pre-/post-
    smooth matvec is ``O(nnz)`` per level rather than ``O(level_size^2)`` -
    only the coarsest-level direct solve still densifies (see
    ``torchalg.sparse.preconditioners.amg.coarse_solve.dense_coarse_solve``).
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
        """Drop the cached hierarchy only when the buffer's device/dtype actually changes.

        ``nn.Module.to()``/``.cuda()``/``.double()``/etc. all call this
        internally, including internally from ``IterativeSolverBase
        ._place_on_device`` on every ``solve()`` call (even when the
        preconditioner is already on the resolved device - ``nn.Module.to()``
        provides no cheap way to ask "would this actually move anything?"
        up front). The cached hierarchy's tensors are derived from
        ``_matrix`` but are not themselves buffers, so a real device/dtype
        change after ``setup()`` would silently leave a stale-device
        hierarchy in place if left unhandled - but invalidating
        unconditionally on every ``_apply`` call would also wipe readiness
        on every same-device ``solve()`` call, breaking the ordinary
        "``setup()`` once, ``solve()`` many times" flow. Comparing
        ``_matrix``'s device/dtype before and after distinguishes a real
        move (invalidate; caller must call ``setup()`` again) from a no-op
        one (readiness and the cached hierarchy are left untouched).

        Args:
            fn (Callable): Conversion function applied to each
                parameter/buffer (see ``nn.Module._apply``).
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Self: This module, per ``nn.Module._apply``'s contract.
        """
        before_device, before_dtype = self._matrix.device, self._matrix.dtype
        result = super()._apply(fn, recurse=recurse)
        if self._matrix.device != before_device or self._matrix.dtype != before_dtype:
            self._hierarchy = None
            self._is_ready = False
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

        Binding new inputs changes what the next ``setup()`` call builds;
        it does not itself rebuild or invalidate anything - call ``setup()``
        again after ``bind_inputs()`` to rebuild the hierarchy with the
        newly bound inputs.

        Args:
            **inputs (torch.Tensor): Named tensors (e.g., positions, theta).
        """
        if isinstance(self._coarsening, BindableInputs):
            self._coarsening.bind_inputs(**inputs)

    # Preconditioner -----------------------------------------------------------

    @property
    def coarsening(self) -> CoarseningStrategy:
        """The coarsening strategy building each coarse level.

        Returns:
            CoarseningStrategy: The strategy passed at construction.
        """
        return self._coarsening

    @property
    def matrix(self) -> torch.Tensor:
        """The finest-level system matrix A.

        Returns:
            torch.Tensor: The matrix passed at construction (n x n).
        """
        return self._matrix

    @property
    def n_levels(self) -> int:
        """Total number of multigrid levels.

        Returns:
            int: The level count passed at construction.
        """
        return self._n_levels

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether Flexible CG is required.

        Returns:
            bool: ``True`` for neural AMG (non-linear cycle); ``False`` for
                classical AMG.
        """
        return not self._linear

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Build the multigrid hierarchy from ``matrix``, unconditionally.

        Rebinds this preconditioner to ``matrix`` (replacing the ``_matrix``
        buffer) and builds a fresh hierarchy, discarding any previously
        built one. Call again - e.g. after ``bind_inputs()`` changes an
        extra input, or to rebind to a new/changed matrix - to rebuild.

        Args:
            matrix (torch.Tensor): System matrix A (n x n).
            context (PreconditionerContext | None): Ignored (AMG hierarchy
                construction is stateless w.r.t. solver iteration).

        Returns:
            Self: This preconditioner, now ready for ``apply()``.
        """
        self._matrix = matrix
        self._hierarchy = self._make_hierarchy()
        self._mark_ready()
        return self

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

        Raises:
            PreconditionerNotReadyError: If ``setup()`` has not been called
                (also raised by the base-class guard before this method
                runs; this second check only protects against ``_hierarchy``
                itself being absent, e.g. a future subclass bypassing the
                guard).
        """
        if self._hierarchy is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before apply()."
            )
        return self._cycle.apply(self._hierarchy, residual)

    def _make_hierarchy(self) -> MultigridHierarchy:
        """Build the hierarchy from the current ``_matrix`` buffer.

        Called by ``setup()``, unconditionally, every time it runs.
        Subclasses whose levels are computed elsewhere override this.

        Returns:
            MultigridHierarchy: Levels built by the coarsening strategy.
        """
        return build_hierarchy(self._matrix, self._coarsening, self._n_levels)

    def __str__(self) -> str:
        """Human-readable structural summary.

        The generic engine formats every structural coarsening strategy
        uniformly. Feature-specific presets may override this presentation.

        Returns:
            str: e.g. ``"AMG(n_levels=2,
                AggregationCoarsening(theta=0.25, omega=0.67))"``.
        """
        return f"AMG(n_levels={self._n_levels}, {self._coarsening})"

    # Private ------------------------------------------------------------------
