#!/usr/bin/env python
"""
Generate the multiome Trim28-versus-Control cell-type enrichment heatmap
for Figure S13d.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

sys.path.append('../scripts/utils')
sys.path.append('../manuscript/scripts')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

from figure_format import get_adata, cell_type_orders_multiome  # noqa: E402
from proportion_heatmap_utils import truncate_colormap, validate_file_sizes  # noqa: E402


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_13"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "d.png")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "d.svg")
CASE_LABEL = "Trim28"
CONTROL_LABEL = "Control"
P_VALUE_THRESHOLD = 0.05


def fisher_celltype_multiome(adata, case=CASE_LABEL, control=CONTROL_LABEL):
    """Compute per-cell-type Fisher exact tests for case versus control."""
    obs = adata.obs.copy()
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["condition"] = obs["condition"].astype(str)

    case_mask = obs["condition"] == case
    control_mask = obs["condition"] == control

    n_case = int(case_mask.sum())
    n_control = int(control_mask.sum())
    if n_case == 0 or n_control == 0:
        raise ValueError(
            f"Expected both {case!r} and {control!r} cells, got "
            f"{n_case} case and {n_control} control cells."
        )

    rows = []
    for cell_type in cell_type_orders_multiome:
        a = int((case_mask & (obs["cell_type"] == cell_type)).sum())
        c = int((control_mask & (obs["cell_type"] == cell_type)).sum())
        b = n_case - a
        d = n_control - c

        table = np.array([[a, b], [c, d]], dtype=int)
        _, p_value = fisher_exact(table, alternative="two-sided")
        odds_ratio_pc, _ = fisher_exact(table + 1)

        rows.append(
            {
                "cell_type": cell_type,
                "case_in_cell_type": a,
                "case_not_cell_type": b,
                "control_in_cell_type": c,
                "control_not_cell_type": d,
                "log2_odds_ratio": float(np.log2(odds_ratio_pc)),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading multiome RNA data...")
    adata = get_adata("multiome_rna")

    print("Running Fisher exact tests...")
    results = fisher_celltype_multiome(adata)
    results["p_adjusted"] = multipletests(results["p_value"], method="fdr_bh")[1]
    results["cell_type"] = pd.Categorical(
        results["cell_type"],
        categories=cell_type_orders_multiome,
        ordered=True,
    )
    results = results.sort_values("cell_type")

    stat_tbl = results.set_index("cell_type")["log2_odds_ratio"].to_frame(CASE_LABEL)
    p_tbl = results.set_index("cell_type")["p_adjusted"].to_frame(CASE_LABEL)

    fig, ax = plt.subplots(figsize=(3.8, 7.2))
    cmap = truncate_colormap(plt.cm.PuOr, 0.30, 0.70)

    sns.heatmap(
        stat_tbl,
        mask=p_tbl >= P_VALUE_THRESHOLD,
        cmap=cmap,
        center=0,
        vmin=-4,
        vmax=4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "log2 Trim28/control odds ratio"},
        rasterized=True,
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("log2 Trim28/control odds ratio", fontsize=12)

    ax.set_title("Change in cell type\nproportion", fontsize=18, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=12, rotation=90)
    ax.tick_params(axis="y", labelsize=12, rotation=0)

    plt.tight_layout()

    print("Saving figures...")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(OUTPUT_PNG, OUTPUT_SVG)
    print(
        results[
            ["cell_type", "log2_odds_ratio", "p_value", "p_adjusted"]
        ].to_string(index=False)
    )
    print("Figure S13d completed successfully!")


if __name__ == "__main__":
    main()
