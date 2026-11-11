"""Data type shared by the selection-stage fixtures and tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SelectionProblem:
    """Inputs of ``batched_select_interpolatory_set`` for one test scenario."""

    padded_candidates: torch.Tensor
    candidate_mask: torch.Tensor
    test_vectors: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


SELECTION_CALIBER = 4
"""Caliber used by the selection-stage tests."""
