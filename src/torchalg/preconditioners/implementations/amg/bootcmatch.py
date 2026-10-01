"""BootCMatch preconditioner, dense sibling.

Wires ``torchalg.multigrid.bootcmatch_setup.BootCMatchSetup`` (Component E,
the TOMS Section 5 outer bootstrap loop) with this tree's own
``BootCMatchCoarsening`` (dense, ``bootcmatch_coarsening.py``) to build a
composite of independent multigrid hierarchies at construction time, then
applies their multiplicative composition as a fixed linear preconditioner -
``z = B_0(r) + B_1(r - A z_0) + ...``, like multiplicative Schwarz.

Deliberately does **not** subclass ``torchalg.multigrid.AMGPreconditioner``:
that class hard-assumes exactly one ``(hierarchy, cycle)`` pair (its
``apply`` is literally ``self._cycle.apply(self._hierarchy, residual)``,
singular). Forcing a composite of several hierarchies into its
single-hierarchy constructor would need a meaningless dummy argument.
Instead this class implements ``Preconditioner`` + ``nn.Module`` directly,
the established pattern for any preconditioner owning precomputed tensor
state (``JacobiPreconditioner``, ``BootstrapAMGPreconditioner``, etc.):
``register_buffer`` for ``.to(device/dtype)`` propagation, never
``register_parameter`` since none of this is learnable.

Device/dtype moves: unlike ``AMGPreconditioner`` (whose single hierarchy is
*lazily* built on first ``apply()`` and so needs an ``_apply`` override to
invalidate a stale-device cache), every hierarchy here is built once at
construction and stored eagerly - there is no lazy rebuild path to protect.
A ``.to()`` call after construction moves the registered ``_matrix`` buffer
but intentionally leaves the already-built hierarchies' tensors on their
original device: rebuilding the whole composite bootstrap loop on every
device move would be surprising, expensive (it is multiple multigrid
setups, not a cheap recompute), and inconsistent with how every other
prebuilt-hierarchy preset (``BootstrapAMGPreconditioner``,
``AdaptiveSAPreconditioner``) handles this - they override ``_make_hierarchy``
to move their *single* stored hierarchy's tensors via ``.to(self._matrix)``
on every ``apply()``. This class follows that same precedent: ``apply()``
moves each stored hierarchy's level matrices to the current buffer's
device/dtype before using it, so a ``.to()`` move is still honored, just
applied lazily inside ``apply()`` rather than via an ``_apply`` override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchalg.multigrid import MultigridHierarchy, MultigridLevel, VCycle
from torchalg.multigrid.bootcmatch_setup import BootCMatchSetup
from torchalg.preconditioners.base import Preconditioner, PreconditionerContext

from ._presets import prebuilt_cycle, seeded_draw
from .bootcmatch_coarsening import BootCMatchCoarsening
from .smoothers import resolve_jacobi_default

if TYPE_CHECKING:
    from collections.abc import Callable

    from .protocols import MultigridSmoother

__all__ = ["BootCMatchPreconditioner"]


def _move_hierarchy(hierarchy: MultigridHierarchy, like: torch.Tensor) -> MultigridHierarchy:
    """Move every level matrix in ``hierarchy`` to ``like``'s device/dtype.

    Transfer operators carry their own tensors (the prolongation) and are
    rebuilt fresh at construction time per hierarchy, so only the level
    matrices themselves need moving here; the transfer operators' own
    tensors are moved by the same ``DenseTransferOperator`` wrapping
    convention used elsewhere in this tree (plain tensor attribute, moved
    via ``.to()`` at the call site that owns it).

    Args:
        hierarchy (MultigridHierarchy): Hierarchy to move.
        like (torch.Tensor): Tensor whose device/dtype to match.

    Returns:
        MultigridHierarchy: A new hierarchy sharing ``like``'s device/dtype.
    """
    return MultigridHierarchy(
        tuple(
            MultigridLevel(matrix=level.matrix.to(like), transfer=level.transfer)
            for level in hierarchy.levels
        )
    )


class BootCMatchPreconditioner(Preconditioner, nn.Module):
    """BootCMatch preconditioner (TOMS Sections 4-5), dense.

    Runs ``BootCMatchSetup.run`` at construction to build a composite of
    independent multigrid hierarchies, then applies their multiplicative
    composition as the preconditioner. A fixed linear composite of fixed
    linear cycles, so plain PCG is valid (same reasoning as
    ``BootstrapAMGPreconditioner``).

    Args:
        matrix (torch.Tensor): SPD system matrix A (n x n).
        seed (int): Seed of the default random-vector source.
        draw (Callable[[int], torch.Tensor] | None): Explicit uniform
            ``[0, 1)`` source overriding ``seed``.
        max_levels (int): Maximum number of levels per hierarchy.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        k_max (int): Power-iteration sweeps per outer-loop round.
        rho_desired (float): Target asymptotic convergence rate.
        max_hierarchies (int): Hard cap on the number of hierarchies in the
            composite.
        smoother (MultigridSmoother | None): Explicit solve-time smoother;
            ``None`` selects weighted Jacobi.
        smoother_omega (float | None): Damping for the default
            weighted-Jacobi solve smoother; ``None`` selects
            ``1 / rho(D^-1 A)`` per level.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

    References:
        - D'Ambra, Filippone, Vassilevski (2018). BootCMatch: a Software
          Package for Bootstrap AMG Based on Graph Weighted Matching.
          ACM TOMS 44(4). Sections 4-5.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        seed: int = 0,
        draw: Callable[[int], torch.Tensor] | None = None,
        max_levels: int = 10,
        max_coarse: int = 10,
        k_max: int = 10,
        rho_desired: float = 0.7,
        max_hierarchies: int = 5,
        smoother: MultigridSmoother | None = None,
        smoother_omega: float | None = None,
        n_pre: int = 1,
        n_post: int = 1,
    ) -> None:
        """Run the BootCMatch bootstrap setup and store its composite hierarchies.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n).
            seed (int): Seed of the default random-vector source.
            draw (Callable[[int], torch.Tensor] | None): Explicit uniform
                ``[0, 1)`` source overriding ``seed``.
            max_levels (int): Maximum number of levels per hierarchy.
            max_coarse (int): Stop coarsening at this many coarse nodes.
            k_max (int): Power-iteration sweeps per outer-loop round.
            rho_desired (float): Target asymptotic convergence rate.
            max_hierarchies (int): Hard cap on the number of hierarchies in
                the composite.
            smoother (MultigridSmoother | None): Explicit solve-time
                smoother, or ``None`` for weighted Jacobi.
            smoother_omega (float | None): Damping for the default
                weighted-Jacobi solve smoother.
            n_pre (int): Solve-time pre-smoothing sweeps.
            n_post (int): Solve-time post-smoothing sweeps.
        """
        nn.Module.__init__(self)
        cycle: VCycle = prebuilt_cycle(
            resolve_jacobi_default(smoother, smoother_omega), n_pre=n_pre, n_post=n_post
        )
        setup = BootCMatchSetup(
            coarsening_factory=BootCMatchCoarsening,
            cycle=cycle,
            draw=draw if draw is not None else seeded_draw(seed),
            max_levels=max_levels,
            max_coarse=max_coarse,
            k_max=k_max,
            rho_desired=rho_desired,
            max_hierarchies=max_hierarchies,
        )
        self._hierarchies: tuple[MultigridHierarchy, ...] = setup.run(matrix)
        self._cycle = cycle
        self._matrix: torch.Tensor
        self.register_buffer("_matrix", matrix)

    @property
    def hierarchies(self) -> tuple[MultigridHierarchy, ...]:
        """The composite's hierarchies, in application order.

        Returns:
            tuple[MultigridHierarchy, ...]: ``B_0, B_1, ..., B_{r-1}``.
        """
        return self._hierarchies

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether Flexible CG is required.

        Returns:
            bool: ``False`` - a fixed linear composite of fixed linear
                cycles, same as ``BootstrapAMGPreconditioner``.
        """
        return False

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply the multiplicative composite correction.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Ignored (the composite
                is stateless w.r.t. solver iteration).

        Returns:
            torch.Tensor: Approximate solution to ``A z = r`` from sweeping
            every hierarchy's cycle once, multiplicative-Schwarz style.
        """
        x = torch.zeros_like(residual)
        for hierarchy in self._hierarchies:
            moved = _move_hierarchy(hierarchy, self._matrix)
            x = x + self._cycle.apply(moved, residual - self._matrix @ x)
        return x

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"BootCMatch(n_hierarchies=2)"``.
        """
        return f"BootCMatch(n_hierarchies={len(self._hierarchies)})"
