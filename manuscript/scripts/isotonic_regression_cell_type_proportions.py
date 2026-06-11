#!/usr/bin/env python
"""
Generate Figure 3d isotonic regression plot with confidence intervals.
Shows log2(clone size) vs cell type proportion with bootstrap-derived bands.
"""

import os
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import (
    cell_type_colors as base_cell_type_colors,
    cell_type_orders_gene10,
    get_adata,
)

from isotonic_utils import isotonic_with_ci, prepare_clone_proportions_df

plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

COMBINED_LABEL = "Progenitor + Progenitor Cycling"
COMBINED_CELL_TYPES = ("Progenitor", "Progenitor Cycling")
PLOT_CELL_TYPE_COLORS = base_cell_type_colors.copy()
PLOT_CELL_TYPE_COLORS[COMBINED_LABEL] = 'black'


def _add_combined_cell_type(clone_df):
    """Append combined progenitor proportion derived from raw adata cell types."""
    missing = [ct for ct in COMBINED_CELL_TYPES if ct not in clone_df.columns]
    if missing:
        raise ValueError(f"Missing required cell type columns: {missing}")

    combined = (
        clone_df[list(COMBINED_CELL_TYPES)]
        .fillna(0)
        .sum(axis=1)
        .clip(upper=1)
    )
    clone_df[COMBINED_LABEL] = combined
    return clone_df


def plot_isotonic_confidence(
    clone_df,
    perturbation='NT',
    n_bootstrap=500,
    cell_types: Sequence[str] | None = None,
    color_map: Mapping[str, str] | None = None,
):
    """Plot isotonic regression with confidence intervals for cell type proportions."""

    if cell_types is None:
        cell_types = cell_type_orders_gene10
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

    for cell_type in cell_types:
        if cell_type not in subset.columns:
            continue

        y = subset[cell_type].to_numpy()
        if np.allclose(y, 0):
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
            linewidth=2,
            label=cell_type,
        )

        if ci_result.bootstrap_samples > 0 and np.all(np.isfinite(ci_result.ci_lower)):
            ax.fill_between(
                ci_result.x_grid,
                ci_result.ci_lower,
                ci_result.ci_upper,
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel('log2(clone size)')
    ax.set_ylabel('Cell type proportion (CTP)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    # No legend for Figure 3d

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
    clone_df = prepare_clone_proportions_df(adata, cell_type_orders_gene10)
    print(f"Created clone DataFrame with {len(clone_df)} clones")
    clone_df = _add_combined_cell_type(clone_df)
    plot_cell_types = list(cell_type_orders_gene10) + [COMBINED_LABEL]

    print("Creating visualization...")
    fig, ax = plot_isotonic_confidence(
        clone_df,
        perturbation='NT',
        cell_types=plot_cell_types,
        color_map=PLOT_CELL_TYPE_COLORS,
    )
    plt.tight_layout()

    png_path = os.path.join(output_dir, "d.png")
    svg_path = os.path.join(output_dir, "d.svg")

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
    print("Figure 3d confidence interval plot saved successfully!")


if __name__ == "__main__":
    main()
