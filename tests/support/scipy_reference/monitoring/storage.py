"""Storage classes for solver monitoring histories.

This module defines immutable history dataclasses for continuous monitoring:
- ScalarHistory: Scalar values tracked over iterations (e.g., residual norms)
- VectorHistory: Vector values tracked over iterations (e.g., residuals, solutions)

All classes use frozen dataclasses for immutability, following functional programming principles
and matching the pattern established by DirectionHistory.

Design:
    - Immutable (frozen=True) for thread safety and predictable behavior
    - Tuple-based storage (immutable sequences)
    - Factory methods for creating empty instances
    - Update methods return new instances (functional style)
    - Common interface: empty(), add(), prepend(), __len__(), __getitem__()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ScalarHistory:
    """Immutable history of scalar values over iterations.

    Tracks scalar measurements (e.g., residual norms, step lengths) across iterations
    for convergence monitoring and post-analysis.

    Follows same pattern as DirectionHistory - frozen dataclass with functional updates.

    Attributes:
        values: Tuple of scalar values, one per iteration.

    Example:
        >>> history = ScalarHistory.empty()
        >>> history = history.add(1.0)
        >>> history = history.add(0.5)
        >>> history.values
        (1.0, 0.5)
        >>> history.to_list()
        [1.0, 0.5]
    """

    values: tuple[float, ...] = ()
    """Scalar values over iterations (immutable tuple)."""

    @classmethod
    def empty(cls) -> ScalarHistory:
        """Create empty scalar history.

        Returns:
            New ScalarHistory with no values.

        Example:
            >>> history = ScalarHistory.empty()
            >>> len(history)
            0
        """
        return cls(values=())

    def add(self, value: float) -> ScalarHistory:
        """Return new history with value appended (immutable update).

        Args:
            value: Scalar value to append.

        Returns:
            New ScalarHistory with value appended.

        Example:
            >>> history = ScalarHistory.empty()
            >>> history = history.add(1.0)
            >>> history.values[-1]
            1.0
        """
        return ScalarHistory(values=self.values + (float(value),))

    def prepend(self, value: float) -> ScalarHistory:
        """Return new history with value prepended (immutable update).

        Args:
            value: Scalar value to prepend.

        Returns:
            New ScalarHistory with value prepended.

        Example:
            >>> history = ScalarHistory(values=(2.0, 3.0))
            >>> history = history.prepend(1.0)
            >>> history.values
            (1.0, 2.0, 3.0)
        """
        return ScalarHistory(values=(float(value),) + self.values)

    def to_list(self) -> list[float]:
        """Convert values to list.

        Returns:
            List of scalar values.

        Example:
            >>> history = ScalarHistory(values=(1.0, 2.0))
            >>> history.to_list()
            [1.0, 2.0]
        """
        return list(self.values)

    def __len__(self) -> int:
        """Return number of values stored.

        Returns:
            Length of values tuple.

        Example:
            >>> history = ScalarHistory(values=(1.0, 2.0))
            >>> len(history)
            2
        """
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        """Get value at index.

        Args:
            index: Index of value to retrieve.

        Returns:
            Scalar value at index.

        Example:
            >>> history = ScalarHistory(values=(1.0, 2.0, 3.0))
            >>> history[0]
            1.0
            >>> history[-1]
            3.0
        """
        return self.values[index]


@dataclass(frozen=True)
class VectorHistory:
    """Immutable history of vector values over iterations.

    Tracks vector measurements (e.g., residuals, solutions) across iterations
    for debugging and post-analysis.

    Follows same pattern as DirectionHistory - frozen dataclass with functional updates.

    Attributes:
        vectors: Tuple of numpy arrays, one per iteration.

    Example:
        >>> import numpy as np
        >>> history = VectorHistory.empty()
        >>> history = history.add(np.array([1.0, 2.0]))
        >>> history = history.add(np.array([3.0, 4.0]))
        >>> len(history)
        2
        >>> history.to_array().shape
        (2, 2)
    """

    vectors: tuple[NDArray, ...] = ()
    """Vector values over iterations (immutable tuple of numpy arrays)."""

    @classmethod
    def empty(cls) -> VectorHistory:
        """Create empty vector history.

        Returns:
            New VectorHistory with no vectors.

        Example:
            >>> history = VectorHistory.empty()
            >>> len(history)
            0
        """
        return cls(vectors=())

    def add(self, vector: NDArray) -> VectorHistory:
        """Return new history with vector appended (immutable update).

        Args:
            vector: Numpy array to append (copied for immutability).

        Returns:
            New VectorHistory with vector appended.

        Example:
            >>> import numpy as np
            >>> history = VectorHistory.empty()
            >>> history = history.add(np.array([1.0, 2.0]))
            >>> len(history)
            1
        """
        import numpy as np

        return VectorHistory(vectors=self.vectors + (np.asarray(vector, copy=True),))

    def prepend(self, vector: NDArray) -> VectorHistory:
        """Return new history with vector prepended (immutable update).

        Args:
            vector: Numpy array to prepend (copied for immutability).

        Returns:
            New VectorHistory with vector prepended.

        Example:
            >>> import numpy as np
            >>> history = VectorHistory.empty()
            >>> history = history.add(np.array([2.0, 3.0]))
            >>> history = history.prepend(np.array([1.0, 1.0]))
            >>> history.to_array()[0, 0]
            1.0
        """
        import numpy as np

        return VectorHistory(vectors=(np.asarray(vector, copy=True),) + self.vectors)

    def to_array(self) -> NDArray:
        """Stack vectors into 2D array (iterations × dimension).

        Returns:
            2D numpy array where each row is a vector (iteration).
            Returns empty (0,) array if no vectors stored.

        Example:
            >>> import numpy as np
            >>> history = VectorHistory.empty()
            >>> history = history.add(np.array([1.0, 2.0]))
            >>> history = history.add(np.array([3.0, 4.0]))
            >>> arr = history.to_array()
            >>> arr.shape
            (2, 2)
            >>> arr[0, 0]
            1.0
        """
        import numpy as np

        if not self.vectors:
            return np.empty((0,), dtype=np.float64)
        return np.stack(self.vectors)

    def __len__(self) -> int:
        """Return number of vectors stored.

        Returns:
            Length of vectors tuple.

        Example:
            >>> import numpy as np
            >>> history = VectorHistory.empty()
            >>> history = history.add(np.array([1.0]))
            >>> len(history)
            1
        """
        return len(self.vectors)

    def __getitem__(self, index: int) -> NDArray:
        """Get vector at index (returns copy for safety).

        Args:
            index: Index of vector to retrieve.

        Returns:
            Copy of numpy array at index.

        Example:
            >>> import numpy as np
            >>> history = VectorHistory.empty()
            >>> history = history.add(np.array([1.0, 2.0]))
            >>> vec = history[0]
            >>> vec[0]
            1.0
        """
        return self.vectors[index].copy()  # Return copy for safety
