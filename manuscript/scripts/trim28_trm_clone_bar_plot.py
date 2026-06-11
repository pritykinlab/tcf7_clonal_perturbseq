#!/usr/bin/env python
"""
Generate bar plots for Figure 5k and alternate variants (v2-v5).

Figure 5k: Dual y-axis bar plot comparing non-TRM-like cell counts by cell type
between Trim28 TRM+ clones and NT clones (size > 10).

Figure 5k_v2: Three-bar plot showing pooled non-TRM cell type proportions (size > 10).
Figure 5k_v3: Three-bar plot showing cell type proportions with no clone-size cutoff.
Figure 5k_v4: Three-bar plot showing cell type proportions with minimum clone size >= 2.
Figure 5k_v5: Five-bar plot showing cell type proportions across five clone groups.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, NullLocator

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, cell_type_orders_gene10

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure5"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------
    # 1)  Load the AnnData exactly as in your earlier analyses
    # --------------------------------------------------------------
    print("Loading data...")
    adata = get_adata(
        'trim28_setdb1',
        use_cells_only_one_construct=False,
        remove_NG=True
    )

    # --------------------------------------------------------------
    # 2) Identify clone groups
    #    Trim28 TRM+ clones:
    #      – perturbation == 'Trim28'
    #      – clone size  > 10 cells
    #      – ≥ 5% of cells are TRM‑like
    #    NT clones:
    #      – perturbation == 'NT'
    #      – clone size > 10 cells
    # --------------------------------------------------------------
    print("Identifying Trim28 TRM+ clones and NT clones...")
    obs = adata.obs.copy()
    obs[['clone', 'cell_type', 'perturbation']] = obs[
        ['clone', 'cell_type', 'perturbation']
    ].astype(str)

    trim28_cells = obs[obs['perturbation'] == 'Trim28']
    nt_cells = obs[obs['perturbation'] == 'NT']

    clone_tbl = (
        trim28_cells
        .groupby('clone')
        .agg(size      = ('cell_type', 'size'),                  # total cells
             trm_count = ('cell_type', lambda x: (x == 'TRM-like').sum()))
    )
    clone_tbl['trm_frac'] = clone_tbl['trm_count'] / clone_tbl['size']

    high_trm_clones = clone_tbl.query('(size > 10) & (trm_count >= 1)').index
    trim28_large_clones = clone_tbl.query('size > 10').index
    trim28_non_trm_clones = trim28_large_clones.difference(high_trm_clones)

    nt_clone_tbl = (
        nt_cells
        .groupby('clone')
        .agg(size=('cell_type', 'size'))
    )
    nt_large_clones = nt_clone_tbl.query('size > 10').index

    print(f"Found {len(high_trm_clones)} Trim28 TRM+ clones")
    print(f"Found {len(nt_large_clones)} NT clones with size > 10")

    def define_clone_groups(min_clone_size=None, trm_min_count=1):
        if min_clone_size is None:
            trim28_size_mask = clone_tbl['size'] >= 1
            nt_size_mask = nt_clone_tbl['size'] >= 1
        else:
            trim28_size_mask = clone_tbl['size'] >= min_clone_size
            nt_size_mask = nt_clone_tbl['size'] >= min_clone_size

        eligible_trim28 = clone_tbl.index[trim28_size_mask]
        trm_mask = (clone_tbl['trm_count'] >= trm_min_count) & trim28_size_mask
        high_trm = clone_tbl.index[trm_mask]
        non_trm = eligible_trim28.difference(high_trm)

        nt_clones = nt_clone_tbl.index[nt_size_mask]
        return high_trm, non_trm, nt_clones

    # --------------------------------------------------------------
    # 3) Collect NON‑TRM‑like cells for each clone group (Figure 5k)
    # --------------------------------------------------------------
    print("Collecting non-TRM-like cells for Figure 5k...")
    trim28_mask = (
        (obs['perturbation'] == 'Trim28')
        & (obs['clone'].isin(high_trm_clones))
        & (obs['cell_type'] != 'TRM-like')
    )
    nt_mask = (
        (obs['perturbation'] == 'NT')
        & (obs['clone'].isin(nt_large_clones))
        & (obs['cell_type'] != 'TRM-like')
    )

    trim28_counts = obs.loc[trim28_mask, 'cell_type'].value_counts()
    nt_counts = obs.loc[nt_mask, 'cell_type'].value_counts()

    # order by your usual gene‑10 list but drop TRM‑like
    ct_order = [ct for ct in cell_type_orders_gene10 if ct != 'TRM-like']
    trim28_counts = trim28_counts.reindex(ct_order).fillna(0).astype(int)
    nt_counts = nt_counts.reindex(ct_order).fillna(0).astype(int)

    # --------------------------------------------------------------
    # 4) Figure 5k: Dual y-axis bar plot
    # --------------------------------------------------------------
    print("Creating Figure 5k (dual y-axis bar plot)...")
    x = np.arange(len(ct_order))
    width = 0.40

    fig, ax = plt.subplots(figsize=(6, 6))
    ax2 = ax.twinx()

    trim28_bars = ax.bar(
        x - width / 2,
        trim28_counts.values,
        width=width,
        color='#d95f02',
        label='Trim28 TRM+'
    )
    nt_bars = ax2.bar(
        x + width / 2,
        nt_counts.values,
        width=width,
        color='#7570b3',
        label='NT'
    )

    trim28_top_pad = max(trim28_counts.max(), 1) * 0.10
    nt_top_pad = max(nt_counts.max(), 1) * 0.10
    ax.set_ylim(0, trim28_counts.max() + trim28_top_pad)
    ax2.set_ylim(0, nt_counts.max() + nt_top_pad)

    ax.set_ylabel('Trim28 TRM+ cell count')
    ax2.set_ylabel('NT cell count (clones > 10)')

    ax.set_xticks(x)
    ax.set_xticklabels(ct_order, rotation=90, ha='center')

    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax2.yaxis.set_minor_locator(NullLocator())

    ax.set_title(
        f'Trim28 TRM+ clones (n={len(high_trm_clones)}) vs NT clones (n={len(nt_large_clones)})'
    )

    ax.legend([trim28_bars, nt_bars], ['Trim28 TRM+', 'NT'], frameon=False, loc='upper right')

    fig.tight_layout()

    # Save outputs
    print("Saving Figure 5k...")
    png_path = os.path.join(output_dir, "k.png")
    svg_path = os.path.join(output_dir, "k.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()
    print("Figure 5k completed successfully!")

    # --------------------------------------------------------------
    # 5) Figure 5k_v2: Cell type proportions by clone group
    # --------------------------------------------------------------
    print("Creating Figure 5k_v2 (cell type proportions)...")
    trim28_high_cells = trim28_cells[trim28_cells['clone'].isin(high_trm_clones)]
    trim28_non_trm_cells = trim28_cells[trim28_cells['clone'].isin(trim28_non_trm_clones)]
    nt_large_cells = nt_cells[nt_cells['clone'].isin(nt_large_clones)]

    def cell_type_proportions(df, order):
        if df.empty:
            return pd.Series(0, index=order, dtype=float)
        counts = df['cell_type'].value_counts().reindex(order).fillna(0)
        return counts / counts.sum()

    ct_order_v2 = [ct for ct in cell_type_orders_gene10 if ct != 'TRM-like']
    proportions = pd.DataFrame(
        {
            'Trim28 TRM+': cell_type_proportions(trim28_high_cells, ct_order_v2),
            'NT >10': cell_type_proportions(nt_large_cells, ct_order_v2),
            'Trim28 non-TRM': cell_type_proportions(trim28_non_trm_cells, ct_order_v2)
        }
    )
    trim28_high_counts = (
        trim28_high_cells.loc[trim28_high_cells['cell_type'] != 'TRM-like', 'cell_type']
        .value_counts()
        .reindex(ct_order_v2)
        .fillna(0)
        .astype(int)
    )
    nt_large_counts = (
        nt_large_cells.loc[nt_large_cells['cell_type'] != 'TRM-like', 'cell_type']
        .value_counts()
        .reindex(ct_order_v2)
        .fillna(0)
        .astype(int)
    )
    trim28_non_trm_counts = (
        trim28_non_trm_cells.loc[trim28_non_trm_cells['cell_type'] != 'TRM-like', 'cell_type']
        .value_counts()
        .reindex(ct_order_v2)
        .fillna(0)
        .astype(int)
    )

    fig_v2, ax_v2 = plt.subplots(figsize=(6.5, 6))
    x = np.arange(len(ct_order_v2))
    width = 0.26

    ax_v2.bar(x - width, proportions['Trim28 TRM+'].values, width=width, color='#d95f02', label='Trim28 TRM+')
    ax_v2.bar(x, proportions['NT >10'].values, width=width, color='#7570b3', label='NT >10')
    ax_v2.bar(x + width, proportions['Trim28 non-TRM'].values, width=width, color='#1b9e77', label='Trim28 non-TRM')

    ax_v2.set_ylabel('Cell type proportion')
    ax_v2.set_ylim(0, min(1.0, proportions.values.max() + 0.15))
    ax_v2.set_xticks(x)
    ax_v2.set_xticklabels(ct_order_v2, rotation=90, ha='center')
    ax_v2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_v2.legend(frameon=False, loc='upper right')
    y_offset = 0.01
    for idx, ct in enumerate(ct_order_v2):
        ax_v2.text(
            idx - width,
            proportions.loc[ct, 'Trim28 TRM+'] + y_offset,
            f"n={trim28_high_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#d95f02',
            fontsize=16
        )
        ax_v2.text(
            idx,
            proportions.loc[ct, 'NT >10'] + y_offset,
            f"n={nt_large_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#7570b3',
            fontsize=16
        )
        ax_v2.text(
            idx + width,
            proportions.loc[ct, 'Trim28 non-TRM'] + y_offset,
            f"n={trim28_non_trm_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#1b9e77',
            fontsize=16
        )

    fig_v2.tight_layout()

    print("Saving Figure 5k_v2...")
    png_path_v2 = os.path.join(output_dir, "k_v2.png")
    svg_path_v2 = os.path.join(output_dir, "k_v2.svg")

    fig_v2.savefig(png_path_v2, dpi=300, bbox_inches='tight')
    fig_v2.savefig(svg_path_v2, format='svg', bbox_inches='tight')

    png_size_v2 = os.path.getsize(png_path_v2) / 1024 / 1024  # MB
    svg_size_v2 = os.path.getsize(svg_path_v2) / 1024  # KB

    if png_size_v2 > 5:
        print(f"ERROR: PNG {png_size_v2:.1f}MB exceeds 5MB limit!")
    if svg_size_v2 > 500:
        print(f"ERROR: SVG {svg_size_v2:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v2:.2f}MB, SVG {svg_size_v2:.1f}KB")

    plt.close()
    print("Figure 5k_v2 completed successfully!")

    # --------------------------------------------------------------
    # 6) Figure 5k_v3: Cell type proportions (no clone-size cutoff, include TRM-like)
    # --------------------------------------------------------------
    print("Creating Figure 5k_v3 (no clone-size cutoff, include TRM-like)...")
    v3_high_trm_clones, v3_non_trm_clones, v3_nt_clones = define_clone_groups(
        min_clone_size=None,
        trm_min_count=1
    )

    v3_trim28_high_cells = trim28_cells[trim28_cells['clone'].isin(v3_high_trm_clones)]
    v3_trim28_non_trm_cells = trim28_cells[trim28_cells['clone'].isin(v3_non_trm_clones)]
    v3_nt_cells = nt_cells[nt_cells['clone'].isin(v3_nt_clones)]

    ct_order_v3 = cell_type_orders_gene10
    proportions_v3 = pd.DataFrame(
        {
            'Trim28 TRM+': cell_type_proportions(v3_trim28_high_cells, ct_order_v3),
            'NT all': cell_type_proportions(v3_nt_cells, ct_order_v3),
            'Trim28 non-TRM': cell_type_proportions(v3_trim28_non_trm_cells, ct_order_v3)
        }
    )
    v3_trim28_high_counts = (
        v3_trim28_high_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v3)
        .fillna(0)
        .astype(int)
    )
    v3_nt_counts = (
        v3_nt_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v3)
        .fillna(0)
        .astype(int)
    )
    v3_trim28_non_trm_counts = (
        v3_trim28_non_trm_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v3)
        .fillna(0)
        .astype(int)
    )

    fig_v3, ax_v3 = plt.subplots(figsize=(6.5, 6))
    x = np.arange(len(ct_order_v3))
    width = 0.26

    ax_v3.bar(x - width, proportions_v3['Trim28 TRM+'].values, width=width, color='#d95f02', label='Trim28 TRM+')
    ax_v3.bar(x, proportions_v3['NT all'].values, width=width, color='#7570b3', label='NT all')
    ax_v3.bar(x + width, proportions_v3['Trim28 non-TRM'].values, width=width, color='#1b9e77', label='Trim28 non-TRM')

    ax_v3.set_ylabel('Cell type proportion')
    ax_v3.set_ylim(0, min(1.0, proportions_v3.values.max() + 0.15))
    ax_v3.set_xticks(x)
    ax_v3.set_xticklabels(ct_order_v3, rotation=90, ha='center')
    ax_v3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_v3.legend(frameon=False, loc='upper right')
    y_offset = 0.01
    for idx, ct in enumerate(ct_order_v3):
        ax_v3.text(
            idx - width,
            proportions_v3.loc[ct, 'Trim28 TRM+'] + y_offset,
            f"n={v3_trim28_high_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#d95f02',
            fontsize=16
        )
        ax_v3.text(
            idx,
            proportions_v3.loc[ct, 'NT all'] + y_offset,
            f"n={v3_nt_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#7570b3',
            fontsize=16
        )
        ax_v3.text(
            idx + width,
            proportions_v3.loc[ct, 'Trim28 non-TRM'] + y_offset,
            f"n={v3_trim28_non_trm_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#1b9e77',
            fontsize=16
        )

    fig_v3.tight_layout()

    print("Saving Figure 5k_v3...")
    png_path_v3 = os.path.join(output_dir, "k_v3.png")
    svg_path_v3 = os.path.join(output_dir, "k_v3.svg")

    fig_v3.savefig(png_path_v3, dpi=300, bbox_inches='tight')
    fig_v3.savefig(svg_path_v3, format='svg', bbox_inches='tight')

    png_size_v3 = os.path.getsize(png_path_v3) / 1024 / 1024  # MB
    svg_size_v3 = os.path.getsize(svg_path_v3) / 1024  # KB

    if png_size_v3 > 5:
        print(f"ERROR: PNG {png_size_v3:.1f}MB exceeds 5MB limit!")
    if svg_size_v3 > 500:
        print(f"ERROR: SVG {svg_size_v3:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v3:.2f}MB, SVG {svg_size_v3:.1f}KB")

    plt.close()
    print("Figure 5k_v3 completed successfully!")

    # --------------------------------------------------------------
    # 7) Figure 5k_v4: Cell type proportions (min clone size >= 2, include TRM-like)
    # --------------------------------------------------------------
    print("Creating Figure 5k_v4 (min clone size >= 2, include TRM-like)...")
    v4_high_trm_clones, v4_non_trm_clones, v4_nt_clones = define_clone_groups(
        min_clone_size=2,
        trm_min_count=3
    )

    v4_trim28_high_cells = trim28_cells[trim28_cells['clone'].isin(v4_high_trm_clones)]
    v4_trim28_non_trm_cells = trim28_cells[trim28_cells['clone'].isin(v4_non_trm_clones)]
    v4_nt_cells = nt_cells[nt_cells['clone'].isin(v4_nt_clones)]

    ct_order_v4 = cell_type_orders_gene10
    proportions_v4 = pd.DataFrame(
        {
            'Trim28 TRM+': cell_type_proportions(v4_trim28_high_cells, ct_order_v4),
            'NT >=2': cell_type_proportions(v4_nt_cells, ct_order_v4),
            'Trim28 non-TRM': cell_type_proportions(v4_trim28_non_trm_cells, ct_order_v4)
        }
    )
    v4_trim28_high_counts = (
        v4_trim28_high_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v4)
        .fillna(0)
        .astype(int)
    )
    v4_nt_counts = (
        v4_nt_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v4)
        .fillna(0)
        .astype(int)
    )
    v4_trim28_non_trm_counts = (
        v4_trim28_non_trm_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v4)
        .fillna(0)
        .astype(int)
    )

    fig_v4, ax_v4 = plt.subplots(figsize=(6.5, 6))
    x = np.arange(len(ct_order_v4))
    width = 0.26

    ax_v4.bar(x - width, proportions_v4['Trim28 TRM+'].values, width=width, color='#d95f02', label='Trim28 TRM+')
    ax_v4.bar(x, proportions_v4['NT >=2'].values, width=width, color='#7570b3', label='NT >=2')
    ax_v4.bar(x + width, proportions_v4['Trim28 non-TRM'].values, width=width, color='#1b9e77', label='Trim28 non-TRM')

    ax_v4.set_ylabel('Cell type proportion')
    ax_v4.set_ylim(0, min(1.0, proportions_v4.values.max() + 0.15))
    ax_v4.set_xticks(x)
    ax_v4.set_xticklabels(ct_order_v4, rotation=90, ha='center')
    ax_v4.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_v4.legend(frameon=False, loc='upper right')
    y_offset = 0.01
    for idx, ct in enumerate(ct_order_v4):
        ax_v4.text(
            idx - width,
            proportions_v4.loc[ct, 'Trim28 TRM+'] + y_offset,
            f"n={v4_trim28_high_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#d95f02',
            fontsize=16
        )
        ax_v4.text(
            idx,
            proportions_v4.loc[ct, 'NT >=2'] + y_offset,
            f"n={v4_nt_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#7570b3',
            fontsize=16
        )
        ax_v4.text(
            idx + width,
            proportions_v4.loc[ct, 'Trim28 non-TRM'] + y_offset,
            f"n={v4_trim28_non_trm_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color='#1b9e77',
            fontsize=16
        )

    fig_v4.tight_layout()

    print("Saving Figure 5k_v4...")
    png_path_v4 = os.path.join(output_dir, "k_v4.png")
    svg_path_v4 = os.path.join(output_dir, "k_v4.svg")

    fig_v4.savefig(png_path_v4, dpi=300, bbox_inches='tight')
    fig_v4.savefig(svg_path_v4, format='svg', bbox_inches='tight')

    png_size_v4 = os.path.getsize(png_path_v4) / 1024 / 1024  # MB
    svg_size_v4 = os.path.getsize(svg_path_v4) / 1024  # KB

    if png_size_v4 > 5:
        print(f"ERROR: PNG {png_size_v4:.1f}MB exceeds 5MB limit!")
    if svg_size_v4 > 500:
        print(f"ERROR: SVG {svg_size_v4:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v4:.2f}MB, SVG {svg_size_v4:.1f}KB")

    plt.close()
    print("Figure 5k_v4 completed successfully!")

    # --------------------------------------------------------------
    # 8) Figure 5k_v5: Cell type proportions (five clone groups)
    # --------------------------------------------------------------
    print("Creating Figure 5k_v5 (five clone groups)...")
    v5_trm_min_count = 3

    # Trim28 TRM+ clones (>=3 TRM-like cells, no size filter beyond that)
    v5_high_trm_clones, v5_non_trm_all, _ = define_clone_groups(
        min_clone_size=None,
        trm_min_count=v5_trm_min_count
    )

    # NT clones: all vs >=3
    nt_all_clones = nt_clone_tbl.index
    nt_ge2_clones = nt_clone_tbl.index[nt_clone_tbl['size'] >= 3]

    # Trim28 non-TRM clones: all vs >=3 total cells
    trim28_non_trm_all = v5_non_trm_all
    trim28_non_trm_ge3 = clone_tbl.index[
        (clone_tbl['size'] >= 3)
        & (~clone_tbl.index.isin(v5_high_trm_clones))
    ]

    v5_trim28_high_cells = trim28_cells[trim28_cells['clone'].isin(v5_high_trm_clones)]
    v5_nt_all_cells = nt_cells[nt_cells['clone'].isin(nt_all_clones)]
    v5_nt_ge2_cells = nt_cells[nt_cells['clone'].isin(nt_ge2_clones)]
    v5_trim28_non_trm_all_cells = trim28_cells[trim28_cells['clone'].isin(trim28_non_trm_all)]
    v5_trim28_non_trm_ge3_cells = trim28_cells[trim28_cells['clone'].isin(trim28_non_trm_ge3)]

    ct_order_v5 = cell_type_orders_gene10
    proportions_v5 = pd.DataFrame(
        {
            'Trim28 TRM+ (>= 3 TRM-like cells)': cell_type_proportions(v5_trim28_high_cells, ct_order_v5),
            'NT all': cell_type_proportions(v5_nt_all_cells, ct_order_v5),
            'NT >= 3 cells': cell_type_proportions(v5_nt_ge2_cells, ct_order_v5),
            'Trim28 non-TRM all': cell_type_proportions(v5_trim28_non_trm_all_cells, ct_order_v5),
            'Trim28 non-TRM >= 3 cells': cell_type_proportions(v5_trim28_non_trm_ge3_cells, ct_order_v5)
        }
    )

    v5_trim28_high_counts = (
        v5_trim28_high_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v5)
        .fillna(0)
        .astype(int)
    )
    v5_nt_all_counts = (
        v5_nt_all_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v5)
        .fillna(0)
        .astype(int)
    )
    v5_nt_ge2_counts = (
        v5_nt_ge2_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v5)
        .fillna(0)
        .astype(int)
    )
    v5_trim28_non_trm_all_counts = (
        v5_trim28_non_trm_all_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v5)
        .fillna(0)
        .astype(int)
    )
    v5_trim28_non_trm_ge3_counts = (
        v5_trim28_non_trm_ge3_cells['cell_type']
        .value_counts()
        .reindex(ct_order_v5)
        .fillna(0)
        .astype(int)
    )

    fig_v5, ax_v5 = plt.subplots(figsize=(7.5, 6))
    x = np.arange(len(ct_order_v5))
    width = 0.16

    colors = {
        'Trim28 TRM+ (>= 3 TRM-like cells)': '#d95f02',
        'NT all': '#7570b3',
        'NT >= 3 cells': '#a6a0d5',
        'Trim28 non-TRM all': '#1b9e77',
        'Trim28 non-TRM >= 3 cells': '#66c2a5'
    }

    labels_v5 = [
        'Trim28 TRM+ (>= 3 TRM-like cells)',
        'NT all',
        'NT >= 3 cells',
        'Trim28 non-TRM all',
        'Trim28 non-TRM >= 3 cells'
    ]

    for idx, label in enumerate(labels_v5):
        ax_v5.bar(
            x + (idx - 2) * width,
            proportions_v5[label].values,
            width=width,
            color=colors[label],
            label=label
        )

    ax_v5.set_ylabel('Cell type proportion')
    ax_v5.set_ylim(0, min(1.0, proportions_v5.values.max() + 0.18))
    ax_v5.set_xticks(x)
    ax_v5.set_xticklabels(ct_order_v5, rotation=90, ha='center')
    ax_v5.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_v5.legend(frameon=False, loc='upper right')

    y_offset = 0.01
    for idx, ct in enumerate(ct_order_v5):
        ax_v5.text(
            idx - 2 * width,
            proportions_v5.loc[ct, labels_v5[0]] + y_offset,
            f"n={v5_trim28_high_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color=colors[labels_v5[0]],
            fontsize=14
        )
        ax_v5.text(
            idx - width,
            proportions_v5.loc[ct, labels_v5[1]] + y_offset,
            f"n={v5_nt_all_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color=colors[labels_v5[1]],
            fontsize=14
        )
        ax_v5.text(
            idx,
            proportions_v5.loc[ct, labels_v5[2]] + y_offset,
            f"n={v5_nt_ge2_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color=colors[labels_v5[2]],
            fontsize=14
        )
        ax_v5.text(
            idx + width,
            proportions_v5.loc[ct, labels_v5[3]] + y_offset,
            f"n={v5_trim28_non_trm_all_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color=colors[labels_v5[3]],
            fontsize=14
        )
        ax_v5.text(
            idx + 2 * width,
            proportions_v5.loc[ct, labels_v5[4]] + y_offset,
            f"n={v5_trim28_non_trm_ge3_counts.loc[ct]}",
            rotation=90,
            ha='center',
            va='bottom',
            color=colors[labels_v5[4]],
            fontsize=14
        )

    fig_v5.tight_layout()

    print("Saving Figure 5k_v5...")
    png_path_v5 = os.path.join(output_dir, "k_v5.png")
    svg_path_v5 = os.path.join(output_dir, "k_v5.svg")

    fig_v5.savefig(png_path_v5, dpi=300, bbox_inches='tight')
    fig_v5.savefig(svg_path_v5, format='svg', bbox_inches='tight')

    png_size_v5 = os.path.getsize(png_path_v5) / 1024 / 1024  # MB
    svg_size_v5 = os.path.getsize(svg_path_v5) / 1024  # KB

    if png_size_v5 > 5:
        print(f"ERROR: PNG {png_size_v5:.1f}MB exceeds 5MB limit!")
    if svg_size_v5 > 500:
        print(f"ERROR: SVG {svg_size_v5:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size_v5:.2f}MB, SVG {svg_size_v5:.1f}KB")

    plt.close()
    print("Figure 5k_v5 completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Trim28 TRM+ clones: {len(high_trm_clones)}")
    print(f"NT clones (size > 10): {len(nt_large_clones)}")
    print(f"Trim28 non-TRM clones (size > 10): {len(trim28_non_trm_clones)}")
    print(f"Trim28 TRM+ clones (no cutoff): {len(v3_high_trm_clones)}")
    print(f"NT clones (no cutoff): {len(v3_nt_clones)}")
    print(f"Trim28 non-TRM clones (no cutoff): {len(v3_non_trm_clones)}")
    print(f"Trim28 TRM+ clones (>=2): {len(v4_high_trm_clones)}")
    print(f"NT clones (>=2): {len(v4_nt_clones)}")
    print(f"Trim28 non-TRM clones (>=2): {len(v4_non_trm_clones)}")
    print(f"Trim28 TRM+ clones (>=3 TRM-like cells): {len(v5_high_trm_clones)}")
    print(f"NT clones (no cutoff): {len(nt_all_clones)}")
    print(f"NT clones (>=2): {len(nt_ge2_clones)}")
    print(f"Trim28 non-TRM clones (no cutoff): {len(trim28_non_trm_all)}")
    print(f"Trim28 non-TRM clones (>=3 cells): {len(trim28_non_trm_ge3)}")

if __name__ == "__main__":
    main()
