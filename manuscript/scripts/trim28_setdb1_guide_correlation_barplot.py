#!/usr/bin/env python
"""
Barplots of per-guide correlations (Pearson and Spearman) between Trim28 vs NT
and each Setdb1 guide vs NT within the trim28_setdb1 experiment (legacy internal Sup_Figure_5_9).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata
import DE_testing
from importlib import reload

plt.rcParams["axes.titlesize"] = 28
plt.rcParams["axes.labelsize"] = 28
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["lines.markersize"] = 6.5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (12, 6.5)

GUIDES = [
    "Setdb1-1", "Setdb1-2", "Setdb1-3", "Setdb1-4", "Setdb1-5",
    "Setdb1-6", "Setdb1-7", "Setdb1-8", "Setdb1-9", "Setdb1-10"
]


def run_de_vs_nt(adata, mask):
    """Run DE (mask vs NT) using Wilcoxon test."""
    group1_cells = adata.obs_names[mask]
    nt_cells = adata.obs_names[adata.obs['perturbation'] == 'NT']
    if len(group1_cells) == 0 or len(nt_cells) == 0:
        raise ValueError("Empty group for DE testing.")

    results_df = DE_testing.wilcoxon_DE_test(adata, group1_cells, nt_cells)
    return results_df.set_index('Gene')


def safe_corr(x, y, corr_fn):
    """Compute correlation; return nan if arrays are constant or too short."""
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return corr_fn(x, y)[0]


def collect_correlations(trim28_results, guide_results):
    """Compute Pearson and Spearman on all shared genes."""
    shared = trim28_results.index.intersection(guide_results.index)
    x = trim28_results.loc[shared, 'Log2FC'].values
    y = guide_results.loc[shared, 'Log2FC'].values
    pearson_val = safe_corr(x, y, pearsonr)
    spearman_val = safe_corr(x, y, spearmanr)
    return pearson_val, spearman_val, len(shared)


def plot_bar(correlations, key, output_png, output_svg, ylabel):
    guides = [c['guide'] for c in correlations if not np.isnan(c[key])]
    values = [c[key] for c in correlations if not np.isnan(c[key])]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(guides, values, color='#4C6EF5')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Setdb1 guides')
    ax.set_ylim(min(values + [0]) - 0.1, max(values + [0]) + 0.1)
    plt.xticks(rotation=45, ha='right')

    # Annotate bar heights
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha='center', va='bottom', fontsize=14)

    fig.tight_layout()

    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_svg, format='svg', bbox_inches='tight')

    png_size = os.path.getsize(output_png) / 1024 / 1024  # MB
    svg_size = os.path.getsize(output_svg) / 1024  # KB
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)


def main():
    output_dir = "../manuscript/plots/Sup_Figure_5_9_old"
    guide_dir = os.path.join(output_dir, "8_trim28_setdb1")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(guide_dir, exist_ok=True)

    print("Loading trim28_setdb1 dataset...")
    adata = get_adata('trim28_setdb1', use_cells_only_one_construct=True,
                      remove_NG=True, remove_lymph_node=True)

    print("Running DE for Trim28 vs NT...")
    trim28_results = run_de_vs_nt(adata, adata.obs['perturbation'] == 'Trim28')

    correlations = []

    for guide in GUIDES:
        print(f"\nProcessing {guide}...")
        guide_mask = (adata.obs['guide'] == guide) & (adata.obs['perturbation'] == 'Setdb1')
        if guide_mask.sum() == 0:
            print("  No cells; skipping.")
            continue

        guide_results = run_de_vs_nt(adata, guide_mask)
        pearson_val, spearman_val, n_shared = collect_correlations(trim28_results, guide_results)

        correlations.append({
            'guide': guide,
            'pearson': pearson_val,
            'spearman': spearman_val,
            'n_shared': n_shared
        })
        print(f"  Shared genes: {n_shared}, Pearson: {pearson_val:.3f} Spearman: {spearman_val:.3f}")

    # Barplots
    pearson_png = os.path.join(output_dir, "9_pearson.png")
    pearson_svg = os.path.join(output_dir, "9_pearson.svg")
    spearman_png = os.path.join(output_dir, "9_spearman.png")
    spearman_svg = os.path.join(output_dir, "9_spearman.svg")

    plot_bar(correlations, 'pearson', pearson_png, pearson_svg,
             ylabel="Pearson correlation (all genes)")
    plot_bar(correlations, 'spearman', spearman_png, spearman_svg,
             ylabel="Spearman correlation (all genes)")

    print("\nlegacy internal Sup_Figure_5_9 barplots completed successfully!")
    print("Summary:")
    for c in correlations:
        print(f"- {c['guide']}: Pearson {c['pearson']:.3f}, Spearman {c['spearman']:.3f}, shared genes {c['n_shared']}")


if __name__ == "__main__":
    reload(DE_testing)
    main()
