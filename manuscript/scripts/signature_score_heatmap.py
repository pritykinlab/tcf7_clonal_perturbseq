#!/usr/bin/env python
"""
Generate signature score heatmap (Figure 2e)
"""

import os
import sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms
from scipy.stats import mannwhitneyu

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_orders_gene10
import misc
import signature_scoring_updated as signature_scoring
from signature_scoring_updated import (
    get_trm_signatures,
    get_yuri_signatures,
    get_miller_signatures,
    get_burn2025_signatures,
    get_park2025_signatures
)
from GO import GOParser

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text


GO_DISPLAY_LABELS = {
    "GOBP_THYMOCYTE_MIGRATION": "Thymocyte migration",
    "GOBP_DENDRITIC_CELL_CHEMOTAXIS": "Dendritic cell chemotaxis",
    "GOBP_POSITIVE_REGULATION_OF_RESPONSE_TO_TUMOR_CELL": "Pos. reg. response to tumor cell",
    "GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_INTEGRIN": "Pos. reg. cell-cell adhesion by integrin",
    "GOCC_MHC_CLASS_I_PROTEIN_COMPLEX": "MHC-I complex",
    "GOCC_CYTOLYTIC_GRANULE": "Cytolytic granule",
    "GOMF_CCR1_CHEMOKINE_RECEPTOR_BINDING": "CCR1 chemokine binding",
    "GOBP_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_CADHERIN": "Reg. cell-cell adhesion by cadherin",
    "GOMF_CELL_ADHESION_MEDIATOR_ACTIVITY": "Cell adhesion mediator activity",
}


def draw_right_bracket(ax, start: int, end: int, label: str) -> None:
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_inner = -0.02
    x_outer = -0.06
    y0 = start
    y1 = end
    y_mid = (y0 + y1) / 2
    ax.plot([x_inner, x_outer], [y0, y0], transform=trans, color="black", lw=0.8, clip_on=False)
    ax.plot([x_outer, x_outer], [y0, y1], transform=trans, color="black", lw=0.8, clip_on=False)
    ax.plot([x_inner, x_outer], [y1, y1], transform=trans, color="black", lw=0.8, clip_on=False)
    ax.text(
        -0.12,
        y_mid,
        label,
        transform=trans,
        va="center",
        ha="center",
        fontsize=24,
        rotation=90,
    )


def main():
    # Create output directory if it doesn't exist
    output_dir = "../manuscript/plots/Figure2"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=False,
                      get_norm_log_zscore_control=True)

    print("Computing signatures...")

    # Define GO terms
    go_parser = GOParser()
    GO_TERMS_TO_USE = [
        "GOBP_THYMOCYTE_MIGRATION",
        "GOBP_DENDRITIC_CELL_CHEMOTAXIS",
        "GOBP_POSITIVE_REGULATION_OF_RESPONSE_TO_TUMOR_CELL",
        "GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_INTEGRIN",
        "GOCC_MHC_CLASS_I_PROTEIN_COMPLEX",
        "GOCC_CYTOLYTIC_GRANULE",
        "GOMF_CCR1_CHEMOKINE_RECEPTOR_BINDING",
        "GOBP_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_CADHERIN",
        "GOMF_CELL_ADHESION_MEDIATOR_ACTIVITY"
    ]

    go_dict = {}
    for term in GO_TERMS_TO_USE:
        genes = go_parser.get_genes_for_term(term)
        go_dict[term] = genes

    # Get other signatures
    yuri_sigs = get_yuri_signatures({
        'func_dysfunc': 'func vs dysfunc',
        'dysfunc_prog': 'dysfunc vs prog'
    })

    miller_sigs = get_miller_signatures()
    trm_sigs = get_trm_signatures()
    burn2025_sigs = get_burn2025_signatures()
    park2025_sigs = get_park2025_signatures()

    # Cell cycle genes
    s_genes, g2m_genes = misc.get_cell_cycle_genes()

    # Build ordered dictionary of all signatures
    all_terms_in_order = OrderedDict()

    # Add GO terms
    for term, g_list in go_dict.items():
        all_terms_in_order[term] = g_list

    # Add Yuri/Pritykin signatures
    all_terms_in_order["Progenitor Pritykin et al., 2021"] = yuri_sigs["Up prog vs dysfunc"]
    all_terms_in_order["Dysfunction Pritykin et al., 2021"] = yuri_sigs["Up dysfunc vs prog"]

    # Add Miller signatures
    all_terms_in_order["Progenitor Miller et al., 2019"] = miller_sigs["Up Tumor Prog vs Tumor Terminal"]
    all_terms_in_order["Terminal Miller et al., 2019"] = miller_sigs["Up Tumor Terminal vs Tumor Prog"]

    # Add TRM signatures
    for name, gene_list in trm_sigs.items():
        all_terms_in_order[name] = gene_list

    # Add Burn2025 signatures (after TRM Crowl et al., 2022)
    for name, gene_list in burn2025_sigs.items():
        all_terms_in_order[name] = gene_list

    # Add Park2025 signatures
    for name, gene_list in park2025_sigs.items():
        all_terms_in_order[name] = gene_list

    # Add cell cycle
    all_terms_in_order["S phase"] = s_genes
    all_terms_in_order["G2/M phase"] = g2m_genes

    # Score each signature
    for term_name, gene_list in all_terms_in_order.items():
        valid = [g for g in gene_list if g in adata.var_names]
        if len(valid) == 0:
            adata.obs[term_name] = 0
        else:
            misc.my_score_genes(adata, valid, term_name, zscore_layer='norm_log_zscore_control')

    # Average by cell type
    row_labels = list(all_terms_in_order.keys())
    score_df = adata.obs.groupby("cell_type")[row_labels].mean()
    score_df_t = score_df.T

    # Reorder columns by cell type order
    col_order = [ct for ct in cell_type_orders_gene10 if ct in score_df_t.columns]
    score_df_t = score_df_t[col_order]
    display_index = [GO_DISPLAY_LABELS.get(label, label) for label in score_df_t.index]
    plot_df = score_df_t.copy()
    plot_df.index = display_index

    # Format numerical values for annotation
    # Round to 2 decimal places and convert to string
    annot_matrix = score_df_t.round(2)

    print("Creating heatmap...")

    # Create the heatmap with custom colorbar positioning
    fig, ax = plt.subplots(figsize=(6, 12))

    # Create colorbar axes on the left side with more offset
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([-0.08, 0.4, 0.05, 0.2])

    g = sns.heatmap(
        plot_df,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot_matrix,
        fmt='.2f',
        annot_kws={'fontsize': 12},
        ax=ax,
        cbar_ax=cbar_ax,
        yticklabels=True
    )

    # Add colorbar label at the top
    cbar_ax.set_title('Mean\nSignature\nScore', fontsize=18, pad=10)

    ax.set_xlabel("")  # Remove x-axis label
    ax.set_ylabel("")  # Remove y-axis label

    # Make x-axis labels vertical
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

    # Move y-tick labels to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Mark the GO-term block as a grouped section.
    draw_right_bracket(ax, start=0, end=len(GO_TERMS_TO_USE), label="GO terms")

    # Save the figure
    print("Saving figures...")
    fig.savefig(os.path.join(output_dir, "e.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "e.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Figure 2e completed successfully!")

if __name__ == "__main__":
    main()
