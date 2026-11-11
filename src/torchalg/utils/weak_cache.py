"""Memoization keyed on an object's lifetime, not on a size bound."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from weakref import finalize


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
