"""Automatic device selection for iterative solvers.

Design:
    - Pure function (no state, no config).
    - Environment-only decision: honors the standard ``CUDA_VISIBLE_DEVICES``
      variable transitively via ``torch.cuda.is_available()``, so forcing
      CPU-only execution needs no torchalg-specific flag.
"""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return the CUDA device if available, else CPU.

    Returns:
        torch.device: ``cuda`` when ``torch.cuda.is_available()`` is
            ``True``, otherwise ``cpu``.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
