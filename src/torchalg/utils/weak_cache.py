"""Memoization keyed on an object's lifetime, not on a size bound."""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from weakref import finalize

import torch


def weak_memo[T, R](function: Callable[..., R]) -> Callable[..., R]:
    """Memoize ``function(subject, *key)`` per ``subject`` object and ``key``.

    Entries are keyed on ``id(subject)`` and dropped by a finalizer when ``subject``
    is garbage collected, so the cache never keeps a subject alive. An
    ``lru_cache`` would hold strong references to every matrix it has seen until
    1024 later entries evict them. Identity keys avoid ``==`` on subjects, which
    for a ``torch.Tensor`` is elementwise and not usable for lookup.

    Args:
        function: Called as ``function(subject, *key)``; ``key`` parts must be hashable.

    Returns:
        Callable[..., R]: Memoized ``function``.
    """
    entries: dict[int, dict[tuple[Hashable, ...], R]] = {}

    def memoized(subject: T, *key: Hashable) -> R:
        subject_id = id(subject)
        per_subject = entries.get(subject_id)
        if per_subject is None:
            per_subject = entries[subject_id] = {}
            finalize(subject, entries.pop, subject_id, None)
        if key not in per_subject:
            per_subject[key] = function(subject, *key)
        return per_subject[key]

    return memoized


# Hidden identity cache by design: cycle protocols hand ``prepare``/``build_transfer`` only the
# matrix, so object identity is the sole channel to share one estimate between their two callers.
def tensor_memo[R](function: Callable[[torch.Tensor], R]) -> Callable[[torch.Tensor], R]:
    """Memoize a pure function of one tensor per tensor object, until it is edited in place.

    ``weak_memo`` keyed on the tensor's version counter, which every in-place write
    bumps, so a stale result is never served for a mutated matrix. Inference tensors
    carry no version counter, so they bypass the cache and are recomputed on every
    call. Entries of superseded versions stay until the tensor is collected.

    Args:
        function (Callable[[torch.Tensor], R]): Pure function of the tensor's values.

    Returns:
        Callable[[torch.Tensor], R]: Memoized ``function``.
    """
    memoized = weak_memo(lambda tensor, _version: function(tensor))

    @functools.wraps(function)
    def call(tensor: torch.Tensor) -> R:
        if tensor.is_inference():
            return function(tensor)
        return memoized(tensor, tensor._version)

    return call
