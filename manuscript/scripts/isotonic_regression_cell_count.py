#!/usr/bin/env python
"""
Generate Figure 3e confidence interval plot for isotonic regression fits.
Shows log2(clone size) vs number of cells with bootstrap-derived confidence bands.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import re

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

from isotonic_utils import isotonic_with_ci, prepare_clone_counts_df

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)

COMBINED_LABEL = "Progenitor + Progenitor Cycling"
COMBINED_CELL_TYPES = ("Progenitor", "Progenitor Cycling")
DEFAULT_CELL_TYPES = list(COMBINED_CELL_TYPES) + [COMBINED_LABEL]
PLOT_CELL_TYPE_COLORS = cell_type_colors.copy()
PLOT_CELL_TYPE_COLORS.setdefault(COMBINED_LABEL, "black")


def _sanitize_cell_type(cell_type: str) -> str:
    """Return a safe suffix for count columns (handles spaces, punctuation, plus sign)."""
    return re.sub(r'[^0-9a-zA-Z]+', '_', cell_type).strip('_').lower()


def _count_column(cell_type: str) -> str:
    """Map a cell type label to its corresponding count column."""
    return f"{_sanitize_cell_type(cell_type)}_count"


def _add_combined_counts(clone_df):
    """Add combined Progenitor + Progenitor Cycling counts."""
    base_cols = [_count_column(ct) for ct in COMBINED_CELL_TYPES]
    missing = [col for col in base_cols if col not in clone_df.columns]
    if missing:
        raise ValueError(f"Missing required count columns: {missing}")

    combined_col = _count_column(COMBINED_LABEL)
    clone_df[combined_col] = clone_df[base_cols].sum(axis=1)
    return clone_df


def plot_isotonic_confidence(
    clone_df,
    perturbation='NT',
    n_bootstrap=500,
    cell_types=None,
    color_map=None,
):
    """Plot isotonic regression with bootstrap confidence intervals for selected cell types."""

    if cell_types is None:
        cell_types = DEFAULT_CELL_TYPES
    if color_map is None:
        color_map = PLOT_CELL_TYPE_COLORS

    fig, ax = plt.subplots(figsize=(12 * 0.7 * 0.8, 6 * 0.8))

    subset = clone_df[clone_df['perturbation'] == perturbation].copy()
    if len(subset) == 0:
        print(f"No data found for perturbation: {perturbation}")
        return fig, ax

    subset['log2_size'] = np.log2(subset['num_cells'])
    subset = subset.sort_values('log2_size')

    x = subset['log2_size'].to_numpy()
    if x.size == 0:
        print(f"No valid clone sizes for perturbation: {perturbation}")
        return fig, ax

    x_min = max(x.min(), 0)
    x_max = min(x.max(), 8)
    if x_max <= x_min:
        x_max = x_min + 1
    x_grid = np.linspace(x_min, x_max, 200)

    max_y = 0

    for cell_type in cell_types:
        col_name = _count_column(cell_type)
        if col_name not in subset.columns:
            print(f"Skipping {cell_type} (column missing).")
            continue

        y = subset[col_name].to_numpy()
        if np.all(y == 0):
            continue

        color = color_map.get(cell_type, 'black')

        ci_result = isotonic_with_ci(
            x=x,
            y=y,
            x_grid=x_grid,
            n_bootstrap=n_bootstrap,
            random_state=42,
        )

        ax.plot(
            ci_result.x_grid,
            ci_result.fit,
            color=color,
            linewidth=3,
            label=cell_type,
        )

        if ci_result.bootstrap_samples > 0 and np.all(np.isfinite(ci_result.ci_lower)):
            ax.fill_between(
                ci_result.x_grid,
                ci_result.ci_lower,
                ci_result.ci_upper,
                color=color,
                alpha=0.25,
            )
            max_y = max(max_y, float(np.nanmax(ci_result.ci_upper)))
        else:
            max_y = max(max_y, float(np.nanmax(ci_result.fit)))

    ax.set_xlabel('log2(clone size)')
    ax.set_ylabel('Average number of cells')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max_y * 1.2 if max_y > 0 else 1)
    # Legend exported separately via Figure 3e legend script

    return fig, ax


def main():
    output_dir = "../manuscript/plots/Figure3"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    adata = get_adata(
        'gene_10',
        use_cells_only_one_construct=True,
        remove_NG=True,
        get_raw_scaled=True,
    )

    print("Processing clone data...")
    clone_df = prepare_clone_counts_df(adata, cell_type_orders_gene10)
    print(f"Created clone DataFrame with {len(clone_df)} clones")
    clone_df = _add_combined_counts(clone_df)

    print("Creating visualization...")
    fig, ax = plot_isotonic_confidence(
        clone_df,
        perturbation='NT',
        cell_types=DEFAULT_CELL_TYPES,
        color_map=PLOT_CELL_TYPE_COLORS,
    )
    plt.tight_layout()

    png_path = os.path.join(output_dir, "e.png")
    svg_path = os.path.join(output_dir, "e.svg")

    print("Saving figures...")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)
    print("Figure 3e confidence interval plot saved successfully!")


if __name__ == "__main__":
    main()
