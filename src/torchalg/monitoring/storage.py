"""Immutable storage classes for solver monitoring histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _to_cpu_copy(vector: torch.Tensor) -> torch.Tensor:
    """Detach and move ``vector`` to a fresh CPU tensor.

    ``.to("cpu")`` is a no-op view (not a copy) when ``vector`` is already on
    CPU, so the trailing ``.clone()`` only actually allocates in that case -
    it guarantees the stored copy is immune to the caller mutating its own
    tensor afterward, regardless of the solve device.
    """
    return vector.detach().to("cpu").clone()


@dataclass(frozen=True, slots=True)
class ScalarHistory:
    """Immutable history of scalar values over iterations."""

    values: tuple[float, ...] = ()
    """Scalar values over iterations."""

    @classmethod
    def empty(cls) -> ScalarHistory:
        """Create an empty scalar history."""
        return cls(values=())

    def add(self, value: torch.Tensor | float) -> ScalarHistory:
        """Return a new history with ``value`` appended (cast to float here)."""
        return ScalarHistory(values=self.values + (float(value),))

    def prepend(self, value: torch.Tensor | float) -> ScalarHistory:
        """Return a new history with ``value`` prepended (cast to float here)."""
        return ScalarHistory(values=(float(value),) + self.values)

    def to_list(self) -> list[float]:
        """Return values as a list."""
        return list(self.values)

    def __len__(self) -> int:
        """Return the number of stored values."""
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        """Return the scalar value at ``index``."""
        return self.values[index]


@dataclass(frozen=True, slots=True)
class VectorHistory:
    """Immutable history of tensor values over iterations.

    Vectors are always stored on the CPU, regardless of the device they were
    computed on: ``add``/``prepend`` move each vector to host memory
    immediately, one at a time, rather than accumulating on the solve device
    and transferring at the end - the latter would not bound peak device
    memory at all, since the accumulation itself is what exhausts it (see
    ``docs/bug-full-trace-history-exhausts-gpu-memory.md``).
    """

    vectors: tuple[torch.Tensor, ...] = ()
    """Tensor values over iterations, always CPU-resident."""

    @classmethod
    def empty(cls) -> VectorHistory:
        """Create an empty vector history."""
        return cls(vectors=())

    def add(self, vector: torch.Tensor) -> VectorHistory:
        """Return a new history with ``vector`` appended, moved to CPU."""
        return VectorHistory(vectors=(*self.vectors, _to_cpu_copy(vector)))

    def prepend(self, vector: torch.Tensor) -> VectorHistory:
        """Return a new history with ``vector`` prepended, moved to CPU."""
        return VectorHistory(vectors=(_to_cpu_copy(vector), *self.vectors))

    def to_tensor(self) -> torch.Tensor:
        """Stack vectors into a tensor with iteration as the first dimension."""
        if not self.vectors:
            return torch.empty((0,), dtype=torch.float64)
        return torch.stack(self.vectors)

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return len(self.vectors)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a clone of the vector at ``index``."""
        return self.vectors[index].clone()
