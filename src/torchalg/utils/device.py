"""Automatic device selection for iterative solvers.

Design:
    - Pure function (no state, no config).
    - Environment-only decision: honors the standard ``CUDA_VISIBLE_DEVICES``
      variable transitively via ``torch.cuda.is_available()``, so forcing
      CPU-only execution needs no torchalg-specific flag.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import cast

import torch
from torch import nn


def resolve_device() -> torch.device:
    """Return the CUDA device if available, else CPU.

    Returns:
        torch.device: ``cuda`` when ``torch.cuda.is_available()`` is
            ``True``, otherwise ``cpu``.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def moved_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move ``tensor`` to ``device``, casting it to ``dtype`` only if it is floating point.

    Index, mask and integer tensors keep their dtype, as ``nn.Module.to(dtype=...)`` does
    for buffers; sparse CSR tensors keep their index width.

    Args:
        tensor (torch.Tensor): Dense or sparse tensor.
        device (torch.device): Target device.
        dtype (torch.dtype): Target floating dtype.

    Returns:
        torch.Tensor: The moved tensor (``tensor`` itself if nothing changes).
    """
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def moved_to[T](obj: T, device: torch.device, dtype: torch.dtype) -> T:
    """Return ``obj`` with every tensor it holds on ``device`` (floating ones cast to ``dtype``).

    Walks prepared state: frozen dataclasses (rebuilt with ``dataclasses.replace``),
    tuples, tensors, ``nn.Module`` instances (moved in place with ``.to``), and objects
    exposing their own ``moved_to(device, dtype)``. Scalars, strings and ``None`` are
    returned unchanged. Any other object raises, so state that cannot move is never left
    silently on the old device.

    Args:
        obj (T): Prepared state to move.
        device (torch.device): Target device.
        dtype (torch.dtype): Target floating dtype.

    Returns:
        T: The moved state.

    Raises:
        TypeError: If ``obj`` (or something it holds) has no known way to move.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, torch.dtype, torch.device)):
        return obj
    if isinstance(obj, torch.Tensor):
        return cast("T", moved_tensor(obj, device, dtype))
    if isinstance(obj, nn.Module):
        return cast("T", obj.to(device=device, dtype=dtype))
    if isinstance(obj, tuple):
        return cast("T", type(obj)(moved_to(item, device, dtype) for item in obj))
    own = getattr(obj, "moved_to", None)
    if callable(own):
        return own(device, dtype)
    if is_dataclass(obj) and not isinstance(obj, type):
        changes = {
            field.name: moved_to(getattr(obj, field.name), device, dtype) for field in fields(obj)
        }
        return cast("T", replace(obj, **changes))
    raise TypeError(f"cannot move {type(obj).__name__} between devices")
