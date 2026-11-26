"""AMG preconditioner.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.amg``
with ``NDArray`` translated to ``torch.Tensor``.
Promoted from ``torchalg.preconditioners.implementations.amg.amg`` into this
shared, format-agnostic package - nothing in this
class commits to dense or sparse storage; see ``torchalg.multigrid``'s
package docstring.

Lifecycle and state design:
    The constructor takes configuration only (coarsening strategy, cycle, level
    count); ``setup(matrix)`` is the one mutating verb that binds a matrix and
    builds the prepared hierarchy. The prepared ``PreparedHierarchy`` is the
    whole prepared state - its finest level already carries the system matrix, so
    matrix and hierarchy cannot disagree and a failed ``setup`` leaves both
    untouched (the new hierarchy is built completely before it replaces the old
    one).

    The hierarchy is a plain frozen-dataclass tree of tensors, not registered
    buffers: the number of levels and their shapes depend on the coarsening
    strategy and are unknown until ``setup()`` runs, and re-registering a
    variable number of indexed buffers on every rebuild is more machinery than the
    problem needs. ``_apply`` (the ``nn.Module`` hook that ``.to()``/``.cuda()``/
    ``.double()`` etc. all funnel through) is overridden to move the hierarchy
    (``utils.device.moved_to``), so a preconditioner set up on the CPU stays ready
    after ``.to("cuda")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchalg.preconditioners.base import (
    BindableInputs,
    MatrixSetupPreconditioner,
    PreconditionerContext,
    PreconditionerNotReadyError,
)
from torchalg.utils.device import moved_to

from .hierarchy import MultigridHierarchy, PreparedHierarchy, build_hierarchy

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from .protocols import CoarseningStrategy, MultigridCycle


class AMGPreconditioner(MatrixSetupPreconditioner, nn.Module):
    """Algebraic Multigrid preconditioner.

    Builds a multigrid hierarchy from a system matrix and applies a multigrid
    cycle as the preconditioner. Structurally implements ``BindableInputs``
    (always safe to call ``bind_inputs``/``extra_input_names``); when the
    coarsening strategy has no extra inputs, these are no-ops.

    Construction takes configuration only. Hierarchy construction is
    **explicit**: it happens in ``setup(matrix)``, which must be called once
    before the first ``apply()`` call (``apply()`` raises
    ``PreconditionerNotReadyError`` otherwise). For neural coarsening,
    ``bind_inputs`` may be called before ``setup()`` (it only forwards to the
    coarsening strategy) so the bound inputs are available when the hierarchy
    is built.

    Args:
        coarsening (CoarseningStrategy): Strategy that builds each coarse
            level.
        cycle (MultigridCycle): Multigrid cycle to apply as preconditioner.
        n_levels (int): Total number of levels; must be at least 2
            (2 = one coarse grid).
        linear (bool): Whether the preconditioner is linear (False for
            neural AMG). Controls ``requires_flexible_cg``.

    Cost:
        Format-agnostic: the formulas hold for any injected coarsening and cycle,
        with ``z_l`` the stored entries of level ``l`` - ``n_l^2`` when a dense
        ``CoarseningStrategy`` keeps every level a dense ``torch.Tensor`` (the dense
        ``torchalg.preconditioners.implementations.amg`` presets, so a level matvec
        is ``2 n_l^2 k`` regardless of the fine matrix's sparsity, not the textbook
        ``O(n)`` per cycle), and ``nnz(A_l)`` when a sparse one keeps the
        non-coarsest levels sparse CSR (only the coarsest solve densifies; see
        ``torchalg.sparse.preconditioners.amg.coarse_solve``). ``N = n_levels``.
        Setup: ``N - 1`` ``coarsening.build_transfer`` calls in sequence (each coarse
        matrix is the next call's input) and one ``cycle.prepare``:
        ``W = sum_{l < N-1} W_bt,l + W_prep``, ``D = sum_l D_bt,l + D_prep``,
        ``L = sum_l L_bt,l + L_prep`` and ``S = sum_l S_bt,l + S_prep``, with the
        ``bt`` terms the coarsening's ``build_transfer`` Cost and the ``prep`` terms
        the cycle's Setup Cost (smoother preparation per level plus the coarse
        factorization, ``2c^3 / 3`` flops for the default LU). Presets that run their
        own setup algorithm (adaptive, bootstrap, BootCMatch) document theirs.
        Apply: one ``cycle.apply`` and nothing else (``W = W_cycle``,
        ``D = D_cycle``, ``L = L_cycle``, ``S = 0``); for a ``VCycle`` with weighted
        Jacobi, in ``VCycle``'s notation, ``W = sum_l [2(a + b) z_l + 3(a + b) n_l + 4 p_l] k + 2c^2 k`` and
        ``L = sum_l [4(a + b) + 2] + 1`` (``VCycle``'s Cost). A neural coarsening or
        transfer adds its network's forward pass to the setup and, for a neural
        transfer, to every restriction and prolongation.
        Roofline: the apply is memory-bound on sparse and dense levels alike at
        ``k = 1``; the dense coarsest factorization (``W/Q = c / 24``) is the one
        compute-bound kernel of a setup.

    Attributes:
        placement (Placement): Taken from ``cycle``, i.e. from its smoother: Jacobi
            smoothing is ``SPMV_BOUND``, Gauss-Seidel ``SEQUENTIAL_SWEEP``.
    """

    def __init__(
        self,
        coarsening: CoarseningStrategy,
        cycle: MultigridCycle,
        n_levels: int = 2,
        linear: bool = True,
    ) -> None:
        """Store the coarsening and cycle strategies; no matrix is bound until ``setup()``.

        Args:
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
        self._coarsening = coarsening
        self._cycle = cycle
        self.placement = cycle.placement
        self._n_levels = n_levels
        self._linear = linear
        self._hierarchy: PreparedHierarchy | None = None

    def _apply(self, fn: Callable, recurse: bool = True) -> Self:
        """Move the prepared hierarchy on a real device/dtype change.

        ``nn.Module.to()``/``.cuda()``/``.double()`` all call this, including
        ``IterativeSolverBase._place_on_device`` on every ``solve()``. The hierarchy is
        not made of buffers, so ``fn`` is probed on an empty tensor of the finest matrix's
        device and dtype (no copy of the matrix) and, when that changes either, the whole
        hierarchy (level matrices, transfers, prepared smoothers, the coarse factorization)
        is moved with ``utils.device.moved_to``. Readiness is kept: setup on the CPU, then
        move and apply on a GPU. A dtype change casts the prepared state (as ``nn.Module``
        casts its weights); it is not a re-setup in the new precision. Before ``setup()``
        there is nothing to move.

        Args:
            fn (Callable): Conversion function applied to each
                parameter/buffer (see ``nn.Module._apply``).
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Self: This module, per ``nn.Module._apply``'s contract.
        """
        result = super()._apply(fn, recurse=recurse)
        if self._hierarchy is not None:
            fine = self._hierarchy.levels[0].matrix
            probe = fn(torch.empty(0, dtype=fine.dtype, device=fine.device))
            if probe.device != fine.device or probe.dtype != fine.dtype:
                self._hierarchy = moved_to(self._hierarchy, probe.device, probe.dtype)
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
        """The finest-level system matrix A bound by ``setup()``.

        Returns:
            torch.Tensor: The matrix of the finest prepared level (n x n).

        Raises:
            PreconditionerNotReadyError: If ``setup()`` has not been called.
        """
        if self._hierarchy is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before the matrix is available."
            )
        return self._hierarchy.levels[0].matrix

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
        """Bind ``matrix`` and build the multigrid hierarchy from it, unconditionally.

        Builds a fresh hierarchy, discarding any previously built one. Call
        again - e.g. after ``bind_inputs()`` changes an extra input, or to
        rebind to a new/changed matrix - to rebuild. If the build fails, the
        previous matrix and hierarchy are kept.

        Args:
            matrix (torch.Tensor): System matrix A (n x n).
            context (PreconditionerContext | None): Ignored (AMG hierarchy
                construction is stateless w.r.t. solver iteration).

        Returns:
            Self: This preconditioner, now ready for ``apply()``.
        """
        self._commit(self._make_hierarchy(matrix))
        return self

    def _commit(self, hierarchy: MultigridHierarchy) -> None:
        """Prepare ``hierarchy`` completely, then replace the bound state with it.

        Preparation can fail (spectral estimates, factorizations); nothing is
        assigned until it has succeeded, so a failure keeps the previous state.

        Args:
            hierarchy (MultigridHierarchy): Unprepared hierarchy of the matrix being bound.
        """
        self._hierarchy = self._cycle.prepare(hierarchy)

    @property
    def is_ready(self) -> bool:
        """Whether ``setup()`` has built the prepared hierarchy.

        Returns:
            bool: ``True`` once ``_hierarchy`` holds a ``PreparedHierarchy``.
        """
        return self._hierarchy is not None

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
                runs; this second check narrows ``_hierarchy`` for the type
                checker and covers a subclass bypassing the guard).
        """
        if self._hierarchy is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before apply()."
            )
        return self._cycle.apply(self._hierarchy, residual)

    def _make_hierarchy(self, matrix: torch.Tensor) -> MultigridHierarchy:
        """Build the unprepared hierarchy of ``matrix``.

        Called by ``setup()``, unconditionally, every time it runs. Presets
        whose levels come from their own setup algorithm override ``setup()``
        and ``_commit()`` the realized hierarchy instead.

        Args:
            matrix (torch.Tensor): Finest-level system matrix being bound.

        Returns:
            MultigridHierarchy: Levels built by the coarsening strategy.
        """
        return build_hierarchy(matrix, self._coarsening, self._n_levels)

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
