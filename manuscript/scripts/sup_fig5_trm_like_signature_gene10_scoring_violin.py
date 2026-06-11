#!/usr/bin/env python
"""
Figure S11f companion source: TRM-like signature scoring in trim28_setdb1 using a cross-experiment signature.

Uses the TRM-like signature defined in Table S5.2.

Outputs:
  - manuscript/plots/Sup_Figure_5_10_old/10.png, 10.svg
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

# Add utils to path
sys.path.append("../scripts/utils")

from figure_format import get_adata, perturbation_colors  # noqa: E402
import misc  # noqa: E402

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text

# Style settings
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.grid"] = False

OUT_DIR = "../manuscript/plots/Sup_Figure_5_10_old"
SIGNATURE_TABLE = "../manuscript/supplementary_tables/5_2.tsv"


def _validate_file_sizes(png_path: str, svg_path: str) -> None:
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB
    if png_size > 5:
        print(f"WARNING: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"WARNING: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def _load_signature_genes(signature_table: str) -> list[str]:
    if not os.path.exists(signature_table):
        raise FileNotFoundError(
            f"Signature table not found: {signature_table}. "
            "Generate Table S5.2 before running this script."
        )
    table = pd.read_csv(signature_table, sep="\t")
    gene_col = None
    for candidate in ("Gene", "gene", "GENE"):
        if candidate in table.columns:
            gene_col = candidate
            break
    if gene_col is None:
        raise ValueError(
            f"Expected a 'Gene' column in signature table: {signature_table}. "
            f"Columns: {list(table.columns)}"
        )
    genes = table[gene_col].astype(str).str.strip().tolist()
    genes = [g for g in genes if g]
    genes_unique = list(dict.fromkeys(genes))
    print(f"Loaded signature genes from Table S5.2: {len(genes_unique)} genes")
    return genes_unique


def _perturbation_palette_like_figure5e(order: list[str]) -> dict[str, tuple[float, float, float]]:
    colors = perturbation_colors.copy()
    colors.update(
        {
            "NT": (200 / 255, 200 / 255, 200 / 255),
            "Setdb1+Trim28": "cornflowerblue",
        }
    )
    return {k: colors.get(k, (0, 0, 0)) for k in order}


def _p_one_sided_setdb1_gt_nt(df: pd.DataFrame) -> float | None:
    mask_nt = df["perturbation"] == "NT"
    mask_setdb1 = df["perturbation"] == "Setdb1"
    if mask_nt.sum() == 0 or mask_setdb1.sum() == 0:
        return None
    _, p_value = mannwhitneyu(df.loc[mask_setdb1, "score"], df.loc[mask_nt, "score"], alternative="greater")
    return float(p_value)


def _plot_violin(df: pd.DataFrame, order: list[str], palette: dict[str, tuple[float, float, float]], title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.25))

    sns.violinplot(
        data=df,
        x="perturbation",
        y="score",
        order=order,
        hue="perturbation",
        hue_order=order,
        palette=palette,
        inner="box",
        cut=0,
        density_norm="width",
        linewidth=1,
        legend=False,
        ax=ax,
    )

    n_before = len(ax.collections)
    sns.stripplot(
        data=df,
        x="perturbation",
        y="score",
        order=order,
        color="black",
        alpha=0.15,
        size=1.4,
        jitter=0.25,
        ax=ax,
    )
    for coll in ax.collections[n_before:]:
        coll.set_rasterized(True)

    ax.set_xlabel("")
    ax.set_ylabel("TRM-like signature score")
    ax.set_title(title, pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score TRM-like signature (Table S5.2) on trim28_setdb1 and plot by perturbation."
    )
    parser.add_argument(
        "--signature-table",
        default=SIGNATURE_TABLE,
        help="Path to Table S5.2 with TRM-like signature genes.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Output directory for Sup_Figure_5_10_old/10 plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load trim28_setdb1 data for scoring (all cells; control-based z-scores)
    print("Loading trim28_setdb1 dataset for scoring...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=False,
        remove_NG=False,
        remove_lymph_node=True,
        get_raw_scaled=False,
        get_norm_log_zscore_control=True,
    )

    # Drop empty perturbation labels (keeps colors/order consistent with Figure 5e)
    perturbation = adata.obs["perturbation"].astype(str)
    empty_mask = perturbation.eq("") | perturbation.eq("nan")
    if empty_mask.any():
        adata = adata[~empty_mask].copy()
        print(f"Filtered {int(empty_mask.sum())} cells with empty perturbation labels.")

    perturbation_order = ["NT", "Trim28", "Setdb1", "Setdb1+Trim28"]
    present = set(adata.obs["perturbation"].astype(str).unique())
    perturbation_order = [p for p in perturbation_order if p in present]
    palette = _perturbation_palette_like_figure5e(perturbation_order)

    genes = _load_signature_genes(args.signature_table)
    valid = [g for g in genes if g in adata.var_names]
    if len(valid) == 0:
        raise ValueError("No signature genes present in trim28_setdb1 var_names from Table S5.2.")
    if len(valid) < len(genes):
        print(f"Signature genes present in trim28_setdb1: {len(valid)}/{len(genes)}")

    score_name = "trm_like_sig_table_s5_2"
    print(f"Scoring signature on trim28_setdb1: {len(valid)}/{len(genes)} genes present.")
    misc.my_score_genes(adata, valid, score_name=score_name, zscore_layer="norm_log_zscore_control")

    df = pd.DataFrame(
        {
            "perturbation": adata.obs["perturbation"].astype(str),
            "score": adata.obs[score_name].astype(float),
        }
    )

    p_setdb1 = _p_one_sided_setdb1_gt_nt(df)
    title = (
        f"TRM-like signature (Table S5.2); Setdb1>NT p={p_setdb1:.2g}"
        if p_setdb1 is not None
        else "TRM-like signature (Table S5.2)"
    )

    fig = _plot_violin(df, perturbation_order, palette, title=title)
    png_path = os.path.join(args.output_dir, "10.png")
    svg_path = os.path.join(args.output_dir, "10.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    _validate_file_sizes(png_path, svg_path)
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
