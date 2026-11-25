"""Structural cost counter: every dispatched ATen op and the host synchronizations among them."""

from __future__ import annotations

from collections import Counter
from typing import Any

from torch.utils._python_dispatch import TorchDispatchMode

HOST_SYNC_OPS = frozenset({"aten.item.default", "aten.is_nonzero.default"})
"""Dispatch names of the ops that block on a device value (``.item()``, ``bool(tensor)``)."""


class DispatchCounter(TorchDispatchMode):
    """Counts every dispatched ATen op while active; ``syncs`` counts host synchronizations."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: Counter[str] = Counter()

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.ops[str(func)] += 1
        return func(*args, **(kwargs or {}))

    @property
    def launches(self) -> int:
        """Total dispatched ops."""
        return sum(self.ops.values())

    @property
    def syncs(self) -> int:
        """Dispatched ops that force a host read."""
        return sum(self.ops[name] for name in HOST_SYNC_OPS)
