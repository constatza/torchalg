"""Immutable input context for iteration-aware collaborators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class IterationContext:
    """Iteration-specific solver values exposed to adaptive helpers."""

    iteration: int
    residual: torch.Tensor
    solution: torch.Tensor
    matrix: torch.Tensor
    rhs: torch.Tensor
