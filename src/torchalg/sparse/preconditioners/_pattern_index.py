"""Index helpers shared by the IC(0) and ILU(0) pattern analyses.

Both analyses store their structural arrays at the narrowest index width that fits
(``structural_index``), resolve ``(row, col)`` queries against the stored entries with one
batched search (``build_lookup``/``lookup``) and rebuild ``crow_indices`` from per-entry
rows (``row_pointer``). They live here, not in either analysis, so neither depends on the
other.
"""

from __future__ import annotations

import torch

from torchalg.utils.csr_index import INT32_INDEX_LIMIT


def structural_index(index: torch.Tensor, extent: int) -> torch.Tensor:
    """``index`` as int32 when every value and count up to ``extent`` fits, else unchanged.

    Only for arrays that merely sit in the pattern (they are gathered once per
    factorization, not per elimination step): the per-step positions stay int64
    because int64 advanced-index gathers measured 10-40% faster than int32 ones.
    """
    return index.to(torch.int32) if extent <= INT32_INDEX_LIMIT else index


def build_lookup(row_index: torch.Tensor, col_index: torch.Tensor, n: int) -> torch.Tensor:
    """Global sorted key ``row * n + col`` for every stored pattern entry.

    CSR storage is already row-major with ascending columns per row, so
    ``row * n + col`` is already strictly increasing in storage order - no
    sort needed, just the key computation. Since the factorization's
    sparsity pattern is fixed throughout (zero fill-in), this is computed
    once upfront and reused by every ``lookup`` call.

    Args:
        row_index (torch.Tensor): int64, one row index per stored entry
            (storage order).
        col_index (torch.Tensor): int64, one column index per stored entry
            (storage order), same shape as ``row_index``.
        n (int): Matrix dimension.

    Returns:
        torch.Tensor: int64, sorted ascending, one key per stored entry.
    """
    return row_index * n + col_index


def lookup(
    keys: torch.Tensor,
    values: torch.Tensor,
    query_rows: torch.Tensor,
    query_cols: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve a batch of ``(row, col)`` queries against the global sorted keys.

    A single ``torch.searchsorted`` call resolves an entire batch of queries
    at once, regardless of how many distinct rows they touch - exactly what
    per-level, per-degree-position batching needs (a batch's queries
    routinely span every row active in a level).

    Args:
        keys (torch.Tensor): int64, sorted ascending, from ``build_lookup``.
        values (torch.Tensor): Values aligned with ``keys`` (same storage
            order as the pattern ``keys`` was built from).
        query_rows (torch.Tensor): int64, rows to look up.
        query_cols (torch.Tensor): int64, columns to look up, same shape as
            ``query_rows``.
        n (int): Matrix dimension (must match the ``n`` used to build
            ``keys``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(found_mask, gathered_values)``,
            both shaped like ``query_rows``; ``gathered_values`` is 0
            wherever ``found_mask`` is ``False``.
    """
    query_keys = query_rows * n + query_cols
    positions = torch.searchsorted(keys, query_keys)
    clamped = positions.clamp(max=keys.numel() - 1)
    found = (positions < keys.numel()) & (keys[clamped] == query_keys)
    at_clamped = values[clamped]
    return found, torch.where(found, at_clamped, torch.zeros_like(at_clamped))


def row_pointer(row_of_entry: torch.Tensor, n: int) -> torch.Tensor:
    """``crow_indices`` for entries sorted by ``row_of_entry``."""
    crow = torch.zeros(n + 1, dtype=torch.int64, device=row_of_entry.device)
    crow[1:] = torch.cumsum(torch.bincount(row_of_entry, minlength=n), 0)
    return crow
