#!/usr/bin/env python
"""
legacy internal TRM-like signature dotplot by guide (legacy internal Sup_Figure_5_12).

Rows are guides (ordered), columns are signature genes from Table S5.2.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add utils to path
sys.path.append("../scripts/utils")
sys.path.append("../manuscript/scripts")

from figure_format import get_adata  # noqa: E402
from dotplot_utils import create_proper_dotplot  # noqa: E402

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text

# Style settings
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.grid"] = False

OUT_DIR = "../manuscript/plots/Sup_Figure_5_12_old"
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
    return no_site + intergenic + setdb1 + trim28 + other


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create TRM-like signature dotplot by guide (Table S5.2 genes)."
    )
    parser.add_argument(
        "--signature-table",
        default=SIGNATURE_TABLE,
        help="Path to Table S5.2 with TRM-like signature genes.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Output directory for Sup_Figure_5_12_old/12 plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trim28_setdb1 dataset (single-construct only) for guide dotplot...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=True,
        remove_NG=False,
        remove_lymph_node=True,
        get_raw_scaled=False,
        get_norm_log_zscore_control=True,
    )

    if "guide" not in adata.obs.columns:
        raise ValueError("Expected `guide` column in trim28_setdb1 .obs for guide-level dotplot.")

    guide_series = adata.obs["guide"].astype(str)
    empty_mask = guide_series.eq("") | guide_series.eq("nan")
    if empty_mask.any():
        adata = adata[~empty_mask].copy()
        print(f"Filtered {int(empty_mask.sum())} cells with empty guide labels.")

    signature_genes = _load_signature_genes(args.signature_table)
    valid_genes = [g for g in signature_genes if g in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("No signature genes present in trim28_setdb1 var_names from Table S5.2.")
    if len(valid_genes) < len(signature_genes):
        print(f"Signature genes present in trim28_setdb1: {len(valid_genes)}/{len(signature_genes)}")

    guide_order = _guide_order(sorted(adata.obs["guide"].astype(str).unique().tolist()))
    fig_width = max(12.0, 0.12 * len(valid_genes))
    fig_height = max(4.0, 0.25 * len(guide_order))

    print("Creating dotplot...")
    dp = create_proper_dotplot(
        adata,
        groupby="guide",
        var_names=valid_genes,
        categories_order=guide_order,
        figsize=(fig_width, fig_height),
        cmap="Reds",
        vmin=-1,
        vmax=1,
    )
    dp.add_totals(color="maroon")
    dp.show()

    fig = plt.gcf()
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)

    png_path = os.path.join(args.output_dir, "12.png")
    svg_path = os.path.join(args.output_dir, "12.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=70)
    plt.close(fig)

    _validate_file_sizes(png_path, svg_path)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
