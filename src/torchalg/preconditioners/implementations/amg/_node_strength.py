"""Symmetric strength of connection on a (possibly block) node graph.

Port of PyAMG's ``symmetric_strength_of_connection``: ``j`` is strongly
connected to ``i`` when ``|a_ij|^2 >= theta^2 |a_ii| |a_jj|`` (stored,
i.e. non-zero, entries only). For a matrix with several dofs per node
(PyAMG's BSR levels) the node matrix holds the Frobenius norm of each block.
With ``theta = 0`` every non-zero coupling is strong - PyAMG's default.

References:
    - PyAMG 5.3.0, ``pyamg/strength.py`` and
      ``pyamg/amg_core/smoothed_aggregation.h``
      (``symmetric_strength_of_connection``).
"""

from __future__ import annotations

import torch


def node_strength(matrix: torch.Tensor, dofs_per_node: int, theta: float) -> torch.Tensor:
    """Boolean strong-connection graph between nodes (diagonal excluded).

    Args:
        matrix (torch.Tensor): Dense matrix, shape ``(n_dofs, n_dofs)``.
        dofs_per_node (int): Block size (1 for a scalar matrix).
        theta (float): Strength threshold, at least 0.

    Returns:
        torch.Tensor: Boolean ``(n_nodes, n_nodes)`` strength matrix.

    Raises:
        ValueError: If ``theta`` is negative.
    """
    if theta < 0:
        raise ValueError("expected a positive theta")
    n_nodes = matrix.shape[0] // dofs_per_node
    node = (
        matrix.abs()
        if dofs_per_node == 1
        else torch.sqrt(
            (matrix.reshape(n_nodes, dofs_per_node, n_nodes, dofs_per_node) ** 2).sum((1, 3))
        )
    )
    diagonal = torch.diagonal(node)
    strong = (node**2 >= theta**2 * diagonal.unsqueeze(1) * diagonal.unsqueeze(0)) & (node != 0)
    strong.fill_diagonal_(False)
    return strong
