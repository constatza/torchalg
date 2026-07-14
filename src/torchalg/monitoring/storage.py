"""Immutable storage classes for solver monitoring histories."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ScalarHistory:
    """Immutable history of scalar values over iterations."""

    values: tuple[float, ...] = ()
    """Scalar values over iterations."""

    @classmethod
    def empty(cls) -> ScalarHistory:
        """Create an empty scalar history."""
        return cls(values=())

    def add(self, value: float) -> ScalarHistory:
        """Return a new history with ``value`` appended."""
        return ScalarHistory(values=self.values + (float(value),))

    def prepend(self, value: float) -> ScalarHistory:
        """Return a new history with ``value`` prepended."""
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
    """Immutable history of tensor values over iterations."""

    vectors: tuple[torch.Tensor, ...] = ()
    """Tensor values over iterations."""

    @classmethod
    def empty(cls) -> VectorHistory:
        """Create an empty vector history."""
        return cls(vectors=())

    def add(self, vector: torch.Tensor) -> VectorHistory:
        """Return a new history with ``vector`` appended."""
        return VectorHistory(vectors=self.vectors + (vector.clone(),))

    def prepend(self, vector: torch.Tensor) -> VectorHistory:
        """Return a new history with ``vector`` prepended."""
        return VectorHistory(vectors=(vector.clone(),) + self.vectors)

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
