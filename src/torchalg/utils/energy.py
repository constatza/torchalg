"""Shared A-weighted ("energy") inner-product primitive.

Extracted from two independent reimplementations of the same math:
``strategies/norms.py::energy_norm`` (single vector, branching on whether
``A`` was given as a full matrix or diagonal entries) and
``preconditioners/implementations/pod/weighting.py::energy_row_norms`` (a batch of
snapshot rows against a full matrix). Both compute the A-inner-product of a
vector with itself, ``x^T A x`` - the "energy" terminology already used
throughout this codebase (``energy_norm``, ``||e_k||_A``), not the
unqualified ``a_*`` prefix an earlier draft of this module used. This is the
one place that shape polymorphism lives, in the dependency-floor
``torchalg.utils`` package (see ``tach.toml``) so every higher-level caller -
convergence norms, POD weighting, post-hoc monitoring analysis - can depend
on it without any of them depending on each other.
"""

from __future__ import annotations

import torch


def energy_dot(x: torch.Tensor, y: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Compute the A-weighted inner product ``x^T A y``.

    Polymorphic over a single vector pair or a row-batch pair: passing the
    same tensor for ``x`` and ``y`` gives the energy quadratic form
    ``v^T A v``, the argument of the energy norm ``||v||_A``.

    Args:
        x (torch.Tensor): Either a single vector, shape ``(n,)``, or a batch
            of row vectors, shape ``(k, n)``.
        y (torch.Tensor): Same shape as ``x``.
        A (torch.Tensor): SPD operator, either full matrix shape ``(n, n)``
            or its diagonal entries, shape ``(n,)``.

    Returns:
        torch.Tensor: ``x^T A y`` for a single vector pair (0-d tensor), or
            one value per row for a batch pair (shape ``(k,)``).

    Example:
        >>> import torch
        >>> v = torch.tensor([1.0, 1.0])
        >>> energy_dot(v, v, torch.tensor([2.0, 8.0]))
        tensor(10.)
    """
    Ay = A * y if A.ndim == 1 else y @ A
    return (x * Ay).sum(dim=-1)
