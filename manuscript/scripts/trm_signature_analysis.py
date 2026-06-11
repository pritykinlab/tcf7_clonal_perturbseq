#!/usr/bin/env python
"""
Generate TRM-like signature UMAP and violin plots (Figure 6c)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_colors, cell_type_orders_multiome

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def plot_trm_like_violin(adata, score_key='score', groupby='cell_type', order=None):
    """
    Plots a violin plot of the TRM-like signature score (adata.obs[score_key])
    grouped by cell type (adata.obs[groupby]) in a specified order.
    """
    # Make a simple DataFrame from obs
    df = pd.DataFrame({
        'cell_type': adata.obs[groupby],
        'TRM_like_score': adata.obs[score_key]
    })

    # Initialize the figure - match UMAP height
    fig, ax = plt.subplots(figsize=(5, 7))  # Changed from 6 to 7 to match UMAP

    # Create color palette for cell types
    from figure_format import cell_type_colors
    colors = [cell_type_colors[ct] for ct in order if ct in cell_type_colors]

    # Create violin plot with hue to avoid FutureWarning
    sns.violinplot(
        data=df,
        x='cell_type',
        y='TRM_like_score',
        order=order,
        hue='cell_type',
        hue_order=order,
        palette=colors,
        inner='box',
        cut=0,
        density_norm='width',
        linewidth=1,
        ax=ax,
        legend=False  # Don't show legend since hue is same as x
    )

    # Labeling - larger ylabel, consistent xlabel size
    ax.set_xlabel("")  # No xlabel for violin plot
    ax.set_ylabel("TRM-like signature score", fontsize=20)  # Match other labels

    # Fix x-tick label rotation and positioning
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)  # Ensure ytick size

    sns.despine()
    plt.tight_layout()

    return fig

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Load multiome RNA data
    print("Loading multiome RNA data...")
    adata = get_adata('multiome_rna')

    # Load TRM-like signature
    print("Loading TRM-like signature...")
    TRM_like_signature_file = "../data/gene10/TRM_like_signature.txt"

    with open(TRM_like_signature_file, 'r') as file:
        TRM_like_signature = [line.strip() for line in file]

    print(f"TRM-like signature genes: {len(TRM_like_signature)}")

    # Calculate TRM-like score
    print("Calculating TRM-like signature scores...")
    sc.tl.score_genes(adata, TRM_like_signature, use_raw=False)

    # Create UMAP with TRM-like score
    print("Creating TRM-like signature UMAP...")

    fig, ax = plt.subplots(figsize=(8,7))

    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']
    scores = adata.obs['score']

    # Calculate vmax as 99.9th percentile
    vmax = np.percentile(scores, 99.9)

    # Create scatter plot with rasterization for SVG size control
    scatter = ax.scatter(umap_coords[:, 0],
                        umap_coords[:, 1],
                        c=scores,
                        cmap='Reds',
                        s=10,
                        vmax=vmax,
                        vmin=0,
                        rasterized=True,  # Rasterize points for smaller SVG
                        edgecolors='none',
                        alpha=0.9)

    # Add colorbar with intermediate height and matching text size
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.045, aspect=15)  # Intermediate: aspect=15 (between 10 and 20)
    cbar.set_label('TRM-like signature score', fontsize=20)  # Match xlabel size
    cbar.ax.tick_params(labelsize=18)  # Colorbar tick labels

    # No title for UMAP, just axes labels
    ax.set_xlabel('UMAP1', fontsize=20)
    ax.set_ylabel('UMAP2', fontsize=20)

    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add frame
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    plt.tight_layout()

    # Save figures
    png_path = os.path.join(output_dir, "c_umap.png")
    svg_path = os.path.join(output_dir, "c_umap.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"WARNING: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"WARNING: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ UMAP file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close(fig)

    # Create violin plot
    print("Creating TRM-like signature violin plot...")
    fig_violin = plot_trm_like_violin(
        adata=adata,
        score_key='score',
        groupby='cell_type',
        order=cell_type_orders_multiome
    )

    png_path_violin = os.path.join(output_dir, "c_violin.png")
    svg_path_violin = os.path.join(output_dir, "c_violin.svg")

    fig_violin.savefig(png_path_violin, dpi=300, bbox_inches='tight')
    fig_violin.savefig(svg_path_violin, format='svg', bbox_inches='tight')

    # Validate file sizes for violin plot
    png_size_violin = os.path.getsize(png_path_violin) / 1024 / 1024  # MB
    svg_size_violin = os.path.getsize(svg_path_violin) / 1024  # KB

    if png_size_violin > 5:
        print(f"WARNING: Violin PNG {png_size_violin:.1f}MB exceeds 5MB limit!")
    if svg_size_violin > 500:
        print(f"WARNING: Violin SVG {svg_size_violin:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ Violin file sizes OK: PNG {png_size_violin:.2f}MB, SVG {svg_size_violin:.1f}KB")

    plt.close(fig_violin)

    print("\nFigure 6c completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {adata.n_obs:,}")
    print(f"TRM-like signature genes: {len(TRM_like_signature)}")
    print(f"Mean TRM-like score: {adata.obs['score'].mean():.3f}")
    print(f"Max TRM-like score: {adata.obs['score'].max():.3f}")
    print(f"TRM-like cells with highest score: {(adata.obs['cell_type'] == 'TRM-like').sum():,}")

if __name__ == "__main__":
    main()
