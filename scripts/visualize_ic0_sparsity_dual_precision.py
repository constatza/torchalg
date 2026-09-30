#!/usr/bin/env python3
"""Visualization script for IC(0)'s sparsity filter in both float32 and float64 precision.

Generates two separate figures comparing IC(0)'s sparsity pattern across precisions on
the rectangular-high-condition matrix (840×840, 1.96% sparse). Each figure shows:
- Raw matrix (original)
- The matrix's own values restricted to IC(0)'s filter at three threshold levels (raw)
- The same, on the normalized matrix

Plots ``ic0_sparsity_mask``'s filter applied to the matrix's own values (A masked to
the filter), not the fully-factorized IC(0) Cholesky factor L. L's values are the
result of elimination (division by pivots, rank-1 corrections) and have nothing to do
with sparsity - plotting them here would show factorization artifacts, not the
non-filling property this figure exists to demonstrate. Since no elimination runs,
there is no pivot/breakdown concept to report either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from torchalg.preconditioners.implementations._masked_factorization import ic0_sparsity_mask

_ZOOM_WINDOW_SIZE = 50
"""Side length (in matrix entries) of the square window shown in each inset."""

_ZOOM_OFFSET = 12
"""How far (in matrix entries) the window is shifted off the main diagonal.

Must be < _ZOOM_WINDOW_SIZE / 2 for the diagonal to still pass through the
window at all - see ``add_zoom_inset`` for why a full-window shift lost it
entirely."""

# Viridis: perceptually-uniform, colorblind-safe, and - the property that
# matters here - has no white/near-white color anywhere in its range (its
# low end is dark purple). A custom ramp with a light or gray midpoint kept
# blending small-but-real entries into the #fcfcfb background used for true
# zeros (set_bad); viridis can't do that at either end.
_MAGNITUDE_CMAP = colormaps["viridis"].copy()
_MAGNITUDE_CMAP.set_bad("#fcfcfb")


def load_rectangular_high_condition_matrix() -> torch.Tensor:
    """Load the rectangular-high-condition test matrix from shared data directory.

    Returns:
        torch.Tensor: Dense matrix, float64.
    """
    matrix_path = Path("/data/shared/mgroup-clusters/rectangular-high-condition.txt")
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    data = np.loadtxt(matrix_path)
    return torch.from_numpy(data).float()


def normalize_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Normalize matrix by Frobenius norm.

    Args:
        matrix: Input matrix.

    Returns:
        Tuple of (normalized_matrix, scale_factor).
    """
    scale = torch.norm(matrix, p="fro").item()
    if scale <= 0:
        scale = 1.0
    normalized = matrix / scale
    return normalized, scale


def apply_ic0_filter(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Restrict a matrix to IC(0)'s sparsity filter, keeping its own values.

    Uses the same ``ic0_sparsity_mask`` helper ``dense_ic0`` calls internally
    - so this is exactly the pattern IC(0) would factorize within - but
    stops there: no elimination, no pivots, no factorization. Surviving
    entries keep A's own numeric value.

    Args:
        matrix: SPD system matrix.
        threshold: Drop tolerance.

    Returns:
        torch.Tensor: ``matrix``'s values, zeroed outside the IC(0) filter.
    """
    mask = ic0_sparsity_mask(matrix, threshold)
    return torch.where(mask, torch.tril(matrix), torch.zeros_like(matrix))


def log_abs_image(matrix: torch.Tensor) -> np.ndarray:
    """log10(|value|) image for a sequential colormap (for plt.imshow).

    Sign is dropped entirely - this shows magnitude only. Exact zeros become
    NaN (rather than a very negative number) so they paint as the colormap's
    "bad" color (chart surface) instead of falsely reading as "smallest
    magnitude entry".

    Args:
        matrix: Matrix to visualize.

    Returns:
        np.ndarray: log10(|value|), float64, with NaN where the entry is zero.
    """
    abs_vals = matrix.abs().cpu().numpy().astype(np.float64)
    with np.errstate(divide="ignore"):
        log_vals = np.log10(abs_vals)
    log_vals[abs_vals == 0] = np.nan
    return log_vals


def shared_log_norm(*matrices: torch.Tensor) -> Normalize:
    """Build one Normalize (on log10(|value|)) shared across matrices.

    Sharing the scale across a row's subplots makes color directly
    comparable between the original matrix and its filtered versions.

    Args:
        *matrices: Matrices sharing this norm's color scale.

    Returns:
        Normalize: Linear norm over the log10-magnitude range.
    """
    all_abs = torch.cat([m.abs().flatten() for m in matrices])
    nonzero = all_abs[all_abs > 0]
    if nonzero.numel() == 0:
        return Normalize(vmin=-10.0, vmax=0.0)
    log_vals = torch.log10(nonzero)
    vmin, vmax = log_vals.min().item(), log_vals.max().item()
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def add_zoom_inset(ax: plt.Axes, image: np.ndarray, cmap, norm: Normalize) -> plt.Axes:
    """Add a magnified inset offset into the lower-triangular band.

    Uses ``axes_grid1``'s ``inset_axes``/``mark_inset`` - matplotlib's lens
    API. ``inset_axes`` (rather than ``zoomed_inset_axes``) is used so the
    inset box's on-figure size is set directly (``width``/``height`` as a
    percentage of the parent axes), independent of the zoom-region's data
    extent - "bigger" means a bigger box, not just a higher magnification
    ratio.

    A window centered exactly ON the diagonal wastes half its area: the
    strictly-upper-triangular half of any such window is zero by
    construction (this matrix is lower-triangular), regardless of where
    along the diagonal it's centered. But shifting the window by a full
    window's width (tried first) overshoots: the diagonal then only
    touches one corner of the window, in practice vanishing entirely. This
    matrix's band is clustered, not a thin tridiagonal - each row has
    separate nonzero blocks roughly at column offsets -``_ZOOM_WINDOW_SIZE``,
    0, and +``_ZOOM_WINDOW_SIZE`` - so a partial shift, by ``_ZOOM_OFFSET``
    (chosen < half the window width so the diagonal still cuts through the
    window), keeps the main diagonal band visible in one part of the crop
    while reaching toward the neighboring off-diagonal cluster in the
    other, instead of an all-or-nothing choice between the two.

    The inset itself is placed in the upper-right, which is empty
    background for a lower-triangular matrix, so it never covers real
    data. ``loc1=2, loc2=4`` connects the rect's upper-left/lower-right
    corners to the inset's - the matplotlib-gallery-standard pairing for
    an upper-right inset, chosen to avoid the crossed "bowtie" connectors
    a same-side pairing (e.g. ``loc1=1, loc2=3``) produces here.

    Args:
        ax: Parent axes already showing the full matrix.
        image: Full image array, same one passed to the parent's ``imshow``.
        cmap: Colormap to reuse, so inset colors match the parent exactly.
        norm: Norm to reuse, so inset colors match the parent exactly.

    Returns:
        plt.Axes: The inset axes.
    """
    n = image.shape[0]
    center = n // 2
    half = _ZOOM_WINDOW_SIZE // 2
    row_start = center - half + _ZOOM_OFFSET
    row_end = row_start + _ZOOM_WINDOW_SIZE
    col_start = center - half - _ZOOM_OFFSET
    col_end = col_start + _ZOOM_WINDOW_SIZE

    axins = inset_axes(ax, width="55%", height="55%", loc="upper right", borderpad=0.6)
    axins.imshow(image, cmap=cmap, norm=norm, aspect="auto")
    axins.set_xlim(col_start - 0.5, col_end - 0.5)
    axins.set_ylim(row_end - 0.5, row_start - 0.5)  # inverted: lower row index at top
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("0.3")
        spine.set_linewidth(0.8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3", lw=0.6)
    return axins


def nnz_stats(matrix: torch.Tensor) -> tuple[int, float]:
    """Count non-zeros and compute fill percentage.

    Args:
        matrix: Input matrix.

    Returns:
        Tuple of (nnz, fill_percent).
    """
    nnz = (matrix.abs() > 0).sum().item()
    fill = 100.0 * nnz / matrix.numel()
    return nnz, fill


def visualize_ic0_precision(matrix: torch.Tensor, dtype: torch.dtype, output_suffix: str) -> bool:
    """Visualize IC(0)'s sparsity filter for a specific precision.

    Args:
        matrix: Input matrix (will be converted to specified dtype).
        dtype: Target precision (torch.float32 or torch.float64).
        output_suffix: Suffix for output filename.

    Returns:
        bool: True if the non-filling property held for every threshold.
    """
    print(f"\n{'=' * 80}")
    print(f"Processing {dtype} precision")
    print(f"{'=' * 80}")

    # Convert to target precision
    matrix = matrix.to(dtype)
    print(f"  Shape: {matrix.shape}, dtype: {matrix.dtype}")
    nnz_orig, fill_orig = nnz_stats(matrix)
    print(f"  Sparsity: {nnz_orig} non-zeros ({fill_orig:.2f}% fill)")

    eps = torch.finfo(dtype).eps

    # Same thresholds for both precisions: 0, 1e-14, 1e-7
    thresholds = [0.0, 1e-14, 1e-7]
    threshold_labels = ["t=0", "t=1e-14", "t=1e-7"]

    print(f"\nApplying IC(0) filter ({dtype})...")
    print(f"  Machine epsilon: {eps:.2e}")
    print(f"  Threshold values: {thresholds}")

    # Filtered raw matrix at each threshold - just A masked to IC(0)'s
    # pattern, no factorization, so there's nothing that can break down.
    filtered_raw = []
    for threshold in thresholds:
        print(f"  Applying filter (raw, threshold={threshold:.2e})...")
        filtered = apply_ic0_filter(matrix, threshold)
        filtered_raw.append(filtered)
        nnz_filtered, fill_filtered = nnz_stats(filtered)
        print(f"    nnz = {nnz_filtered} ({fill_filtered:.2f}%)")

    # Filtered normalized matrix at each threshold
    print(f"\nNormalizing matrix (Frobenius norm, {dtype})...")
    normalized, scale = normalize_matrix(matrix)
    print(f"  Scale factor (||A||_F): {scale:.4e}")
    nnz_norm, fill_norm = nnz_stats(normalized)
    print(f"  After normalization: {nnz_norm} non-zeros ({fill_norm:.2f}% fill)")

    filtered_norm = []
    for threshold in thresholds:
        print(f"  Applying filter (normalized, threshold={threshold:.2e})...")
        filtered = apply_ic0_filter(normalized, threshold)
        filtered_norm.append(filtered)
        nnz_filtered, fill_filtered = nnz_stats(filtered)
        print(f"    nnz = {nnz_filtered} ({fill_filtered:.2f}%)")

    # Create figure: 2 rows (raw/normalized) x 4 cols (original + 3 filtered)
    fig_height = 11
    fig_width = 16
    fig, axes = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.35, wspace=0.3, right=0.90, top=0.88)  # room for titles/colorbars

    # Compare lower triangles only (fair comparison with the filtered versions)
    matrix_lower = torch.tril(matrix)
    nnz_orig, fill_orig = nnz_stats(matrix_lower)

    # Separate color scale per row (raw vs. normalized) - each row's panels
    # share one scale so the original and its three filtered versions are
    # directly comparable, and each row uses its own full contrast range
    # rather than being squeezed by the other row's very different scale.
    normalized_lower = torch.tril(normalized)
    raw_norm = shared_log_norm(matrix_lower, *filtered_raw)
    norm_norm = shared_log_norm(normalized_lower, *filtered_norm)

    # Row 0: Raw matrix
    # Col 0: Original (lower triangle only)
    ax = axes[0, 0]
    image = log_abs_image(matrix_lower)
    im = ax.imshow(image, cmap=_MAGNITUDE_CMAP, norm=raw_norm, aspect="auto")
    add_zoom_inset(ax, image, _MAGNITUDE_CMAP, raw_norm)
    ax.set_title(f"Original (A_lower) | nnz={nnz_orig}", fontsize=9, weight="bold")
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row", fontsize=8)
    ax.grid(False)

    # Cols 1-3: IC(0) filter at different thresholds
    for col, (filtered, label, threshold_val) in enumerate(
        zip(filtered_raw, threshold_labels, thresholds), start=1
    ):
        ax = axes[0, col]
        image = log_abs_image(filtered)
        im = ax.imshow(image, cmap=_MAGNITUDE_CMAP, norm=raw_norm, aspect="auto")
        add_zoom_inset(ax, image, _MAGNITUDE_CMAP, raw_norm)
        nnz_filtered, fill_filtered = nnz_stats(filtered)
        fillin = nnz_filtered - nnz_orig  # Should be ≤ 0 (reduction, not addition)
        ax.set_title(f"IC(0) filter [{label}] | nnz={nnz_filtered}", fontsize=9, weight="bold")
        ax.text(0.5, -0.15, f"Fill-in: {fillin}", ha="center", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.grid(False)
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, pad=0.01, label="log10(|value|)")

    # Row 1: Normalized matrix (lower triangle only)
    # Col 0: Normalized original (lower triangle only)
    nnz_norm, fill_norm = nnz_stats(normalized_lower)
    ax = axes[1, 0]
    image = log_abs_image(normalized_lower)
    im = ax.imshow(image, cmap=_MAGNITUDE_CMAP, norm=norm_norm, aspect="auto")
    add_zoom_inset(ax, image, _MAGNITUDE_CMAP, norm_norm)
    ax.set_title(f"Normalized (A_lower/||·||_F) | nnz={nnz_norm}", fontsize=9, weight="bold")
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row", fontsize=8)
    ax.grid(False)

    # Cols 1-3: IC(0) filter on normalized matrix at different thresholds
    for col, (filtered, label, threshold_val) in enumerate(
        zip(filtered_norm, threshold_labels, thresholds), start=1
    ):
        ax = axes[1, col]
        image = log_abs_image(filtered)
        im = ax.imshow(image, cmap=_MAGNITUDE_CMAP, norm=norm_norm, aspect="auto")
        add_zoom_inset(ax, image, _MAGNITUDE_CMAP, norm_norm)
        nnz_filtered, fill_filtered = nnz_stats(filtered)
        fillin = nnz_filtered - nnz_norm  # Should be ≤ 0 (non-filling property)
        ax.set_title(f"IC(0) filter Norm [{label}] | nnz={nnz_filtered}", fontsize=9, weight="bold")
        ax.text(0.5, -0.15, f"Fill-in: {fillin}", ha="center", transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.grid(False)

    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.8, pad=0.01, label="log10(|value|)")

    # Only figure-wide facts here - anything that varies per panel (threshold,
    # scale, dtype label, nnz) already appears in that panel's own title/text.
    dtype_name = "float32" if dtype == torch.float32 else "float64"
    plt.suptitle(
        f"IC(0) Filter Analysis: {dtype_name} (eps={eps:.2e}) | rectangular-high-condition (840×840, {fill_orig:.2f}% sparse)\n"
        f"Normalization: A_norm = A / ||A||_F (scale={scale:.4e}) | Non-filling: nnz(filtered) ≤ nnz(A)",
        fontsize=11,
        weight="bold",
        y=0.995,
    )

    # Save figure to ~/Pictures/
    output_dir = Path.home() / "Pictures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ic0_sparsity_patterns_{output_suffix}.png"
    print(f"\nSaving figure to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("  Done!")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SPARSITY SUMMARY ({dtype})")
    print("=" * 80)
    print(f"{'Matrix Type':<25} {'Threshold':<20} {'nnz':<10} {'Fill %':<10}")
    print("-" * 80)

    print(f"{'Original':<25} {'-':<20} {nnz_orig:<10} {fill_orig:<10.2f}")

    for label, filtered in zip(threshold_labels, filtered_raw):
        nnz, fill = nnz_stats(filtered)
        print(f"{'IC(0) filter Raw':<25} {label:<20} {nnz:<10} {fill:<10.2f}")

    print()
    print(f"{'Normalized':<25} {'-':<20} {nnz_norm:<10} {fill_norm:<10.2f}")

    for label, filtered in zip(threshold_labels, filtered_norm):
        nnz, fill = nnz_stats(filtered)
        print(f"{'IC(0) filter Norm':<25} {label:<20} {nnz:<10} {fill:<10.2f}")

    # Verify non-filling property
    print("\n" + "=" * 80)
    print(f"NON-FILLING VERIFICATION ({dtype})")
    print("=" * 80)
    all_valid = True
    for label, filtered in zip(threshold_labels, filtered_raw):
        nnz_filtered, _ = nnz_stats(filtered)
        is_valid = nnz_filtered <= nnz_orig
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  Raw {label}: nnz={nnz_filtered} ≤ nnz(A)={nnz_orig} ... {status}")
        all_valid = all_valid and is_valid

    for label, filtered in zip(threshold_labels, filtered_norm):
        nnz_filtered, _ = nnz_stats(filtered)
        is_valid = nnz_filtered <= nnz_norm
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"  Norm {label}: nnz={nnz_filtered} ≤ nnz(A_norm)={nnz_norm} ... {status}")
        all_valid = all_valid and is_valid

    if all_valid:
        print(f"\n✓ IC(0) filter is non-filling in {dtype} at every threshold.")
    else:
        print(f"\n✗ VIOLATION in {dtype}: filter added entries outside original pattern!")

    return all_valid


def main() -> None:
    """Load matrix and visualize IC(0)'s filter at both precisions."""
    print("Loading rectangular-high-condition matrix...")
    matrix = load_rectangular_high_condition_matrix()
    print(f"Loaded as {matrix.dtype}, shape {matrix.shape}")

    # Visualize for both precisions
    success_f32 = visualize_ic0_precision(matrix, torch.float32, "float32")
    success_f64 = visualize_ic0_precision(matrix, torch.float64, "float64")

    print("\n" + "=" * 80)
    print("DUAL-PRECISION SUMMARY")
    print("=" * 80)
    print(f"float32: {'✓ PASS' if success_f32 else '✗ FAIL'}")
    print(f"float64: {'✓ PASS' if success_f64 else '✗ FAIL'}")

    if not (success_f32 and success_f64):
        sys.exit(1)


if __name__ == "__main__":
    main()
