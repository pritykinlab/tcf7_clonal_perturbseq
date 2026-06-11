#!/usr/bin/env python
"""
Generate Figure 3f isotonic regression plot with log2 observed/expected confidence intervals.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append('../scripts/utils')
from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

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


def plot_isotonic_confidence(clone_df, adata, perturbation='NT', n_bootstrap=500):
    """Plot log2 observed/expected isotonic regression with confidence intervals."""

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

    adata_subset = adata[adata.obs['perturbation'] == perturbation]
    global_props = {
        cell_type: (adata_subset.obs['cell_type'] == cell_type).mean()
        for cell_type in cell_type_orders_gene10
    }

    pseudocount = 1e-10

    for cell_type in cell_type_orders_gene10:
        if cell_type not in subset.columns:
            continue

        global_prop = global_props.get(cell_type, 0)
        if global_prop == 0:
            continue

        y = subset[cell_type].to_numpy()
        if np.allclose(y, 0):
            continue

        transform = lambda pred, gp=global_prop, pc=pseudocount: np.log2((pred + pc) / (gp + pc))

        ci_result = isotonic_with_ci(
            x=x,
            y=y,
            x_grid=x_grid,
            n_bootstrap=n_bootstrap,
            random_state=42,
            transform=transform,
        )

        ax.plot(
            ci_result.x_grid,
            ci_result.fit,
            color=cell_type_colors[cell_type],
            linewidth=2,
            label=cell_type,
        )

        if ci_result.bootstrap_samples > 0 and np.all(np.isfinite(ci_result.ci_lower)):
            ax.fill_between(
                ci_result.x_grid,
                ci_result.ci_lower,
                ci_result.ci_upper,
                color=cell_type_colors[cell_type],
                alpha=0.2,
            )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('log2(clone size)')
    ax.set_ylabel('Normalized cell type proportion')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-4, 4)
    # No legend for Figure 3f

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

    print("Creating visualization...")
    fig, ax = plot_isotonic_confidence(clone_df, adata, perturbation='NT')
    plt.tight_layout()

    png_path = os.path.join(output_dir, "f.png")
    svg_path = os.path.join(output_dir, "f.svg")

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
    print("Figure 3f confidence interval plot saved successfully!")


if __name__ == "__main__":
    main()
