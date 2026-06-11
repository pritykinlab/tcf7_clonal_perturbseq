#!/usr/bin/env python
"""
legacy internal Sup_Figure_5_13: Heatmap of guide minus NT for TRM-like signature genes.

Rows are guides, columns are signature genes (Table S5.2).
Genes are ordered by sum(abs) across guides.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add utils to path
sys.path.append("../scripts/utils")

from figure_format import get_adata  # noqa: E402

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text

OUT_DIR = "../manuscript/plots/Sup_Figure_5_13_old"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Setdb1-1 minus NT heatmap for TRM-like signature genes."
    )
    parser.add_argument(
        "--signature-table",
        default=SIGNATURE_TABLE,
        help="Path to Table S5.2 with TRM-like signature genes.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Output directory for Sup_Figure_5_13_old/13 plots.",
    )
    parser.add_argument(
        "--include-nt-guides",
        action="store_true",
        help="Include NT guides (NO_SITE) as rows in the heatmap.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trim28_setdb1 dataset (single-construct only) for heatmap...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=True,
        remove_NG=False,
        remove_lymph_node=True,
        get_raw_scaled=False,
        get_norm_log_zscore_control=True,
    )

    if "guide" not in adata.obs.columns:
        raise ValueError("Expected `guide` column in trim28_setdb1 .obs.")

    signature_genes = _load_signature_genes(args.signature_table)
    valid_genes = [g for g in signature_genes if g in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("No signature genes present in trim28_setdb1 var_names from Table S5.2.")
    if len(valid_genes) < len(signature_genes):
        print(f"Signature genes present in trim28_setdb1: {len(valid_genes)}/{len(signature_genes)}")

    nt_mask = adata.obs["perturbation"].astype(str) == "NT"
    if nt_mask.sum() == 0:
        raise ValueError("No NT cells found for comparison.")

    layer = adata.layers.get("norm_log_zscore_control")
    if layer is None:
        raise ValueError("Expected norm_log_zscore_control layer for z-scored expression.")

    gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]
    nt_vals = layer[nt_mask.values, :][:, gene_indices]

    try:
        import scipy.sparse as sp
        if sp.issparse(nt_vals):
            nt_vals = nt_vals.toarray()
    except Exception:
        pass

    nt_mean = np.asarray(nt_vals).mean(axis=0)

    guide_series = adata.obs["guide"].astype(str)
    guide_list = sorted(guide_series.unique().tolist())

    def _order_by_numeric_suffix(items: list[str], prefix: str) -> list[str]:
        pattern = re.compile(rf"^{re.escape(prefix)}[-_]?(\d+)$", flags=re.IGNORECASE)
        parsed: list[tuple[int, str]] = []
        rest: list[str] = []
        for item in items:
            match = pattern.match(item)
            if match:
                parsed.append((int(match.group(1)), item))
            else:
                rest.append(item)
        return [x[1] for x in sorted(parsed)] + sorted(rest)

    def _guide_order(guides: list[str]) -> list[str]:
        guides = [str(g) for g in guides]
        no_site = _order_by_numeric_suffix([g for g in guides if g.startswith("NO_SITE")], "NO_SITE")
        intergenic = sorted([g for g in guides if "INTERGENIC" in g])
        setdb1 = _order_by_numeric_suffix([g for g in guides if g.lower().startswith("setdb1")], "Setdb1")
        trim28 = _order_by_numeric_suffix([g for g in guides if g.lower().startswith("trim28")], "Trim28")
        used = set(no_site + intergenic + setdb1 + trim28)
        other = sorted([g for g in guides if g not in used])
        ordered = no_site + intergenic + setdb1 + trim28 + other
        if not args.include_nt_guides:
            ordered = [g for g in ordered if not g.startswith("NO_SITE")]
        return ordered

    guide_order = _guide_order(guide_list)
    if not guide_order:
        raise ValueError("No guides available for heatmap after filtering.")

    diff_rows = []
    for guide in guide_order:
        guide_mask = guide_series == guide
        if guide_mask.sum() == 0:
            continue
        guide_vals = layer[guide_mask.values, :][:, gene_indices]
        try:
            import scipy.sparse as sp
            if sp.issparse(guide_vals):
                guide_vals = guide_vals.toarray()
        except Exception:
            pass
        guide_mean = np.asarray(guide_vals).mean(axis=0)
        diff_rows.append(guide_mean - nt_mean)

    data = pd.DataFrame(
        diff_rows,
        index=[f"{g} - NT" for g in guide_order],
        columns=valid_genes,
    )

    fig_width = max(6.0, 0.35 * len(guide_order))
    fig_height = max(6.0, 0.08 * len(valid_genes))

    print("Ordering genes by sum(abs) across Setdb1 guides and rendering heatmap...")
    setdb1_rows = data.index.str.startswith("Setdb1")
    setdb1_no_first = setdb1_rows & ~data.index.str.startswith("Setdb1-1")
    if setdb1_no_first.any():
        gene_order = data.loc[setdb1_no_first].abs().sum(axis=0).sort_values(ascending=False).index.tolist()
    elif setdb1_rows.any():
        gene_order = data.loc[setdb1_rows].abs().sum(axis=0).sort_values(ascending=False).index.tolist()
    else:
        gene_order = data.abs().sum(axis=0).sort_values(ascending=False).index.tolist()
    data = data.loc[:, gene_order]
    data_plot = data.T

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        data_plot,
        cmap="RdBu_r",
        center=0,
        linewidths=0.0,
        cbar_kws={"label": "Mean z-score (guide minus NT)"},
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=2.5)
    ax.set_yticks(np.arange(len(data_plot.index)) + 0.5)
    ax.set_yticklabels(data_plot.index, rotation=0)
    if ax.collections:
        ax.collections[0].set_rasterized(True)

    png_path = os.path.join(args.output_dir, "13.png")
    svg_path = os.path.join(args.output_dir, "13.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    _validate_file_sizes(png_path, svg_path)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
