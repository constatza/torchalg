"""Counts products ``matrix @ vector`` for one specific matrix object."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode


class MatvecCounter(TorchFunctionMode):
    """Counts ``matrix @ vector`` products whose left operand is exactly ``matrix``."""

    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self._matrix = matrix
        self.count = 0

    def __torch_function__(
        self, func: Callable[..., Any], types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
    ) -> Any:
        left = args[0] if args else None
        if (
            getattr(func, "__name__", "") in ("__matmul__", "matmul")
            and left is self._matrix
            and args[1].ndim == 1
        ):
            self.count += 1
        return func(*args, **(kwargs or {}))
