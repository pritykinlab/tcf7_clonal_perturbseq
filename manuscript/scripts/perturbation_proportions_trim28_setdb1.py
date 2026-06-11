#!/usr/bin/env python
"""
Generate perturbation proportions stacked bar charts for the trim28_setdb1 dataset.
- internal perturbation-proportion panel: perturbation proportions (NG excluded)
- Figure 5g: main figure (NG excluded; Setdb1+Trim28 shown in cornflowerblue)
- Figure S11 source no-guide panel: same as Figure 5g, but including no-guide cells
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, perturbation_colors, cell_type_orders_trim28_setdb1

# Set figure parameters (consistent with other panels)
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)

FIGURE_DIR = "../manuscript/plots/Figure5"
SUPPLEMENTARY_DIR = "../manuscript/plots/Sup_Figure_5_3_old"
SUPPLEMENTARY_FIGURE_5_1_DIR = "../manuscript/plots/Sup_Figure_11_shared_source"


def _save_legend(order, get_color, output_path_base: str):
    """Create and save a standalone legend using the provided order."""
    display_labels = {"No Guide": "no guide"}
    fig_legend, ax_legend = plt.subplots(figsize=(3, 3))
    for pert in order:
        ax_legend.bar(0, 0, color=get_color(pert), label=display_labels.get(pert, pert))
    handles, labels = ax_legend.get_legend_handles_labels()
    ax_legend.set_visible(False)
    fig_legend.legend(handles, labels,
                      loc='center', frameon=True,
                      fontsize=15, title_fontsize=15)
    png_path_legend = f"{output_path_base}_legend.png"
    svg_path_legend = f"{output_path_base}_legend.svg"
    fig_legend.savefig(png_path_legend, dpi=300, bbox_inches='tight')
    fig_legend.savefig(svg_path_legend, format='svg', bbox_inches='tight')
    plt.close(fig_legend)
    return png_path_legend, svg_path_legend


def _save_barplot(proportions_df, order_reversed, get_color, output_path_base: str):
    """Plot stacked bar chart with reversed legend order and save."""
    fig, ax = plt.subplots(figsize=(3.25, 6.5))
    proportions_df[order_reversed].plot(
        kind='bar',
        stacked=True,
        color=[get_color(x) for x in order_reversed],
        ax=ax,
        width=0.8
    )
    ax.set_xlabel('')
    ax.set_ylabel('proportion of cell type', fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=16)
    ax.legend().set_visible(False)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    png_path = f"{output_path_base}.png"
    svg_path = f"{output_path_base}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return png_path, svg_path


def _validate_file_sizes(png_path: str, svg_path: str, label: str):
    """Report whether PNG/SVG outputs meet manuscript size requirements."""
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: {label} PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: {label} SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ {label} file sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main():
    # Create output directories
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(SUPPLEMENTARY_DIR, exist_ok=True)
    os.makedirs(SUPPLEMENTARY_FIGURE_5_1_DIR, exist_ok=True)

    # Load data
    print("Loading trim28_setdb1 dataset...")
    adata = get_adata('trim28_setdb1', use_cells_only_one_construct=False, remove_NG=False)

    # Adjust color scale for readability
    perturbation_colors2 = perturbation_colors.copy()
    perturbation_colors2.update({
        'NG': (150 / 255, 150 / 255, 150 / 255),
        'NT': (200 / 255, 200 / 255, 200 / 255),
        'No Guide': (105 / 255, 105 / 255, 105 / 255),
        'Setdb1+Trim28': 'cornflowerblue'
    })

    # Helper function to fetch colors
    def get_color(name):
        return perturbation_colors2.get(name, (0, 0, 0))  # Black as default

    # Process perturbations, handle multiple and missing entries
    print("Processing perturbations...")
    adata.obs['perturbation_processed'] = adata.obs['perturbation'].astype(str)
    empty_mask = adata.obs['perturbation_processed'].eq('') | adata.obs['perturbation_processed'].eq('nan')
    adata.obs.loc[empty_mask, 'perturbation_processed'] = 'No Guide'
    if empty_mask.any():
        print(f"Retained {int(empty_mask.sum())} cells with empty perturbation labels as 'No Guide'.")

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'cell_type': adata.obs['cell_type'],
        'perturbation': adata.obs['perturbation_processed']
    })

    # Calculate counts and normalize to proportions
    counts = df.groupby(['cell_type', 'perturbation'], observed=False).size().unstack(fill_value=0)
    proportions = counts.divide(counts.sum(axis=1), axis=0)

    # Define custom legend order
    order = ['NT'] + (['No Guide'] if 'No Guide' in proportions.columns else []) + [
        x for x in proportions.columns
        if x not in ['NT', 'No Guide', 'Setdb1+Trim28']
    ] + (['Setdb1+Trim28'] if 'Setdb1+Trim28' in proportions.columns else [])

    # Reorder proportions to match cell type order
    proportions_ordered = proportions.loc[cell_type_orders_trim28_setdb1]

    # =================================================================
    # Create standalone legend with original order for legacy supplementary plot
    # =================================================================
    legacy_order = [pert for pert in order if pert != 'No Guide']
    legacy_counts = counts.drop(columns=['No Guide'], errors='ignore')
    legacy_proportions = legacy_counts.divide(
        legacy_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)
    legacy_proportions_ordered = legacy_proportions.loc[cell_type_orders_trim28_setdb1]

    print("Creating standalone legend with original order (legacy supplementary)...")
    sup_legend_png, sup_legend_svg = _save_legend(
        legacy_order, get_color, os.path.join(SUPPLEMENTARY_DIR, "3")
    )

    # =================================================================
    # Legacy supplementary plot without No Guide
    # =================================================================
    print("Creating legacy supplementary visualization with reversed legend order...")
    legacy_order_reversed = legacy_order[::-1]
    print("Saving legacy supplementary panel...")
    sup_png, sup_svg = _save_barplot(
        legacy_proportions_ordered,
        legacy_order_reversed,
        get_color,
        os.path.join(SUPPLEMENTARY_DIR, "3"),
    )
    _validate_file_sizes(sup_png, sup_svg, "Legacy supplementary figure")
    _validate_file_sizes(sup_legend_png, sup_legend_svg, "Legacy supplementary legend")

    # =================================================================
    # source no-guide panel with No Guide included
    # =================================================================
    print("Preparing source no-guide panel with No Guide included...")
    order_with_no_guide = [pert for pert in order if pert in counts.columns]
    order_with_no_guide_reversed = order_with_no_guide[::-1]

    sup_5_1_legend_png, sup_5_1_legend_svg = _save_legend(
        order_with_no_guide,
        get_color,
        os.path.join(SUPPLEMENTARY_FIGURE_5_1_DIR, "a"),
    )
    sup_5_1_png, sup_5_1_svg = _save_barplot(
        proportions_ordered,
        order_with_no_guide_reversed,
        get_color,
        os.path.join(SUPPLEMENTARY_FIGURE_5_1_DIR, "a"),
    )
    _validate_file_sizes(sup_5_1_png, sup_5_1_svg, "source no-guide panel")
    _validate_file_sizes(sup_5_1_legend_png, sup_5_1_legend_svg, "source no-guide panel legend")

    # =================================================================
    # Main Figure 5g (No Guide excluded)
    # =================================================================
    print("Preparing main Figure 5g...")
    counts_main = legacy_counts
    proportions_main = counts_main.divide(counts_main.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    order_main = [pert for pert in legacy_order if pert in counts_main.columns]
    order_main_reversed = order_main[::-1]
    proportions_main_ordered = proportions_main.loc[cell_type_orders_trim28_setdb1]

    print("Creating standalone legend for main figure...")
    main_legend_png, main_legend_svg = _save_legend(order_main, get_color, os.path.join(FIGURE_DIR, "g"))

    print("Saving main figure panel with reversed legend order...")
    main_png, main_svg = _save_barplot(proportions_main_ordered, order_main_reversed, get_color, os.path.join(FIGURE_DIR, "g"))

    _validate_file_sizes(main_png, main_svg, "Main figure")
    _validate_file_sizes(main_legend_png, main_legend_svg, "Main legend")

    print("internal perturbation-proportion panel, source no-guide panel, and Figure 5g completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total cells: {len(adata)}")
    print(f"Cell types: {proportions.index.tolist()}")
    print(f"Perturbations found (inclusive order): {order_with_no_guide_reversed}")
    print(f"Perturbations used for Figure 5g: {order_main_reversed}")

    # Print cell counts per perturbation
    perturbation_counts = adata.obs['perturbation_processed'].value_counts()
    print(f"\nCells per perturbation:")
    for pert in order_with_no_guide_reversed:
        if pert in perturbation_counts.index:
            print(f"  {pert}: {perturbation_counts[pert]}")

if __name__ == "__main__":
    main()
