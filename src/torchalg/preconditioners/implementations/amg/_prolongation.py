"""Jacobi prolongator smoothing and the alpha-SA bridging prolongator.

References:
    - PyAMG 5.3.0, ``pyamg/aggregation/smooth.py``
      (``jacobi_prolongation_smoother``, ``weighting='diagonal'``,
      ``degree=1``) and ``pyamg/aggregation/adaptive.py`` (``make_bridge``).
    - Brezina, Falgout, MacLachlan, Manteuffel, McCormick & Ruge (2005).
      Adaptive smoothed aggregation (alpha-SA) multigrid. SIAM Review, 47(2).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def jacobi_prolongation(
    matrix: torch.Tensor,
    tentative: torch.Tensor,
    omega: float,
    spectral_radius: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    """Smooth ``tentative`` once: ``P = T - (omega / rho(D^-1 S)) D^-1 S T``.

    ``D^-1`` has zeros where the diagonal of ``S`` is zero (PyAMG's
    ``get_diagonal(inv=True)``).

    Args:
        matrix (torch.Tensor): System matrix ``S = A`` on this level.
        tentative (torch.Tensor): Tentative prolongator ``T``.
        omega (float): Nominal damping (PyAMG's default is 4/3).
        spectral_radius (Callable[[torch.Tensor], float]): Estimator of
            ``rho`` applied to ``D^-1 S``.

    Returns:
        torch.Tensor: Smoothed prolongator, same shape as ``tentative``.
    """
    diagonal = torch.diagonal(matrix)
    inverse = torch.where(diagonal != 0, 1.0 / diagonal, torch.zeros_like(diagonal))
    scaled = inverse.unsqueeze(1) * matrix
    scaled = (omega / spectral_radius(scaled)) * scaled
    return tentative - scaled @ tentative


def make_bridge(tentative: torch.Tensor, dofs_per_node: int) -> torch.Tensor:
    """Extend ``T`` by one always-zero dof per node (the new candidate's dof).

    Args:
        tentative (torch.Tensor): ``T``, shape ``(n_nodes * dofs_per_node, n_coarse)``.
        dofs_per_node (int): Current dofs per node ``K``.

    Returns:
        torch.Tensor: Bridge of shape ``(n_nodes * (K + 1), n_coarse)`` whose
            new (last) dof rows are zero.
    """
    n_nodes = tentative.shape[0] // dofs_per_node
    bridge = torch.zeros(
        n_nodes,
        dofs_per_node + 1,
        tentative.shape[1],
        dtype=tentative.dtype,
        device=tentative.device,
    )
    bridge[:, :-1, :] = tentative.reshape(n_nodes, dofs_per_node, tentative.shape[1])
    return bridge.reshape(n_nodes * (dofs_per_node + 1), tentative.shape[1])
