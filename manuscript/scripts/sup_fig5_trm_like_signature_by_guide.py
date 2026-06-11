#!/usr/bin/env python
"""
Figure S11f source panel: TRM-like signature scoring by guide in trim28_setdb1.

Uses the TRM-like signature defined in Table S5.2.

Outputs:
  - manuscript/plots/Sup_Figure_11_shared_source/e.png, e.svg
"""

from __future__ import annotations

import argparse
import os
import re
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
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.grid"] = False

OUT_DIR = "../manuscript/plots/Sup_Figure_11_shared_source"
SIGNATURE_TABLE = "../manuscript/supplementary_tables/5_2.tsv"
MIN_CELLS_PER_GUIDE = 10
DEFAULT_PNG = os.path.join(OUT_DIR, "e.png")
DEFAULT_SVG = os.path.join(OUT_DIR, "e.svg")


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


def _star_from_p(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _plot_violin(
    df: pd.DataFrame,
    order: list[str],
    png_path: str,
    svg_path: str,
    title: str | None,
    guide_to_p: dict[str, float | None],
    guide_counts: dict[str, int],
) -> None:
    width = max(8.0, 0.35 * len(order))
    fig, ax = plt.subplots(figsize=(width, 4.5))

    palette = {g: df.loc[df["guide"] == g, "color"].iloc[0] for g in order if g in df["guide"].values}
    sns.violinplot(
        data=df,
        x="guide",
        y="score",
        order=order,
        hue="guide",
        hue_order=order,
        palette=palette,
        inner="box",
        cut=0,
        density_norm="width",
        linewidth=1,
        legend=False,
        dodge=False,
        ax=ax,
    )

    n_before = len(ax.collections)
    sns.stripplot(
        data=df,
        x="guide",
        y="score",
        order=order,
        color="black",
        alpha=0.18,
        size=1.4,
        jitter=0.25,
        ax=ax,
    )
    for coll in ax.collections[n_before:]:
        coll.set_rasterized(True)

    ax.set_xlabel("")
    ax.set_ylabel("TRM-like signature score")
    if title:
        ax.set_title(title, pad=10)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xticks(range(len(order)))
    xtick_labels = [f"{g} (n={guide_counts.get(g, 0)})" for g in order]
    ax.set_xticklabels(xtick_labels)
    sns.despine(ax=ax)

    score_range = df["score"].max() - df["score"].min()
    if score_range == 0:
        score_range = 1.0
    y_offset = 0.05 * score_range
    for idx, guide in enumerate(order):
        star = _star_from_p(guide_to_p.get(guide))
        if not star:
            continue
        guide_scores = df.loc[df["guide"] == guide, "score"]
        if guide_scores.empty:
            continue
        y_pos = guide_scores.max() + y_offset
        ax.text(idx, y_pos, star, ha="center", va="bottom", fontsize=14)

    plt.tight_layout()

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    _validate_file_sizes(png_path, svg_path)
    plt.close(fig)


def _guide_color_map(guides: list[str], guide_to_perturbation: dict[str, str]) -> dict[str, tuple[float, float, float]]:
    colors = perturbation_colors.copy()
    colors["NT"] = (200 / 255, 200 / 255, 200 / 255)
    guide_colors: dict[str, tuple[float, float, float]] = {}
    for guide in guides:
        perturbation = guide_to_perturbation.get(guide, "NT")
        guide_colors[guide] = colors.get(perturbation, (0.5, 0.5, 0.5))
    return guide_colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score TRM-like signature (Table S5.2) on trim28_setdb1 and plot by guide."
    )
    parser.add_argument(
        "--signature-table",
        default=SIGNATURE_TABLE,
        help="Path to Table S5.2 with TRM-like signature genes.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Output directory for the Figure S11f source panel.",
    )
    parser.add_argument(
        "--png-path",
        default=DEFAULT_PNG,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--svg-path",
        default=DEFAULT_SVG,
        help="Output SVG path.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.png_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.svg_path), exist_ok=True)

    print("Loading trim28_setdb1 dataset (single-construct only) for guide plot...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=True,
        remove_NG=False,
        remove_lymph_node=True,
        get_raw_scaled=False,
        get_norm_log_zscore_control=True,
    )

    if "guide" not in adata.obs.columns:
        raise ValueError("Expected `guide` column in trim28_setdb1 .obs for guide-level violin plot.")

    guide_series = adata.obs["guide"].astype(str)
    empty_mask = guide_series.eq("") | guide_series.eq("nan")
    if empty_mask.any():
        adata = adata[~empty_mask].copy()
        print(f"Filtered {int(empty_mask.sum())} cells with empty guide labels.")

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
            "guide": adata.obs["guide"].astype(str),
            "perturbation": adata.obs["perturbation"].astype(str),
            "score": adata.obs[score_name].astype(float),
        }
    )
    if df["guide"].nunique() == 0:
        raise ValueError("No guide labels available for plotting after filtering.")

    guide_counts = df["guide"].value_counts()
    keep_guides = guide_counts[guide_counts > MIN_CELLS_PER_GUIDE].index.tolist()
    removed_guides = guide_counts[guide_counts <= MIN_CELLS_PER_GUIDE].index.tolist()
    if removed_guides:
        removed_guides_sorted = sorted(removed_guides)
        print(
            f"Removing guides with <= {MIN_CELLS_PER_GUIDE} cells "
            f"(n={len(removed_guides_sorted)}): {', '.join(removed_guides_sorted)}"
        )
    df = df[df["guide"].isin(keep_guides)].copy()
    if df["guide"].nunique() == 0:
        raise ValueError(
            f"All guides removed after filtering (<= {MIN_CELLS_PER_GUIDE} cells)."
        )

    guide_to_perturbation = (
        df.groupby("guide")["perturbation"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )
    guide_order = _guide_order(sorted(df["guide"].unique().tolist()))
    guide_counts = df["guide"].value_counts().to_dict()
    guide_color_map = _guide_color_map(guide_order, guide_to_perturbation)
    df["color"] = df["guide"].map(guide_color_map)

    nt_mask = df["perturbation"] == "NT"
    if nt_mask.sum() == 0:
        raise ValueError("No NT cells available for guide-level significance testing.")

    guide_to_p: dict[str, float | None] = {}
    for guide in guide_order:
        perturbation = guide_to_perturbation.get(guide, "NT")
        if perturbation == "NT":
            guide_to_p[guide] = None
            continue
        guide_scores = df.loc[df["guide"] == guide, "score"]
        nt_scores = df.loc[nt_mask, "score"]
        if guide_scores.empty or nt_scores.empty:
            guide_to_p[guide] = None
            continue
        _, p_value = mannwhitneyu(guide_scores, nt_scores, alternative="greater")
        guide_to_p[guide] = float(p_value)

    title = None

    _plot_violin(df, guide_order, args.png_path, args.svg_path, title, guide_to_p, guide_counts)

    print("Per-guide one-sided Mann-Whitney U (guide > NT combined):")
    for guide in guide_order:
        perturbation = guide_to_perturbation.get(guide, "NT")
        if perturbation == "NT":
            continue
        p_value = guide_to_p.get(guide)
        star = _star_from_p(p_value)
        if p_value is None:
            print(f"- {guide} ({perturbation}): p=NA")
        else:
            print(f"- {guide} ({perturbation}): p={p_value:.3g} {star}")

    setdb1_guides = [g for g in guide_order if guide_to_perturbation.get(g) == "Setdb1"]
    significant_setdb1 = [g for g in setdb1_guides if guide_to_p.get(g) is not None and guide_to_p[g] < 0.05]
    print(
        f"Setdb1 guides with > {MIN_CELLS_PER_GUIDE} cells and one-sided p < 0.05: "
        f"{len(significant_setdb1)}/{len(setdb1_guides)}"
    )
    if significant_setdb1:
        print(f"Significant Setdb1 guides: {', '.join(significant_setdb1)}")

    print(f"Wrote: {args.png_path}")
    print(f"Wrote: {args.svg_path}")
    if removed_guides:
        removed_guides_sorted = sorted(removed_guides)
        print(
            f"Removed guides (<= {MIN_CELLS_PER_GUIDE} cells): {', '.join(removed_guides_sorted)}"
        )
    else:
        print(f"No guides removed (<= {MIN_CELLS_PER_GUIDE} cells).")


if __name__ == "__main__":
    main()
