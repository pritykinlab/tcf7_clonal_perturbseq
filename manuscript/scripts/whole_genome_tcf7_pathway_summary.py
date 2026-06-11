#!/usr/bin/env python
"""
Generate the Figure 1g whole-genome Tcf7 screen pathway summary panel.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ANALYSIS_DIR = (
    REPO_ROOT
    / "Figure1_data_handoff"
    / "08_codex_handoff_2026_05_01"
    / "analysis_outputs"
    / "whole_genome_pathway_analysis_2026-03-13"
)
SELECTED_TERMS = (
    CANONICAL_ANALYSIS_DIR
    / "figure1_pathway_summary"
    / "whole_genome_tcf7_pathway_summary_selected_terms.tsv"
)
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"

TARGET_HIGH_COLOR = "#c83f4f"
TARGET_LOW_COLOR = "#2f6fa7"
AXIS_TEXT_SCALE = 2.0

SOURCE_LIBRARY_LABELS = {
    "GO_Biological_Process_2025": "Enrichr GO Biological Process 2025",
    "Reactome_2022": "Enrichr Reactome 2022",
    "KEGG_2019_Mouse": "Enrichr KEGG 2019 Mouse",
    "WikiPathways_2024_Mouse": "Enrichr WikiPathways 2024 Mouse",
}


def source_library_from_file(source_file: str) -> str:
    for library_name, label in SOURCE_LIBRARY_LABELS.items():
        if library_name in source_file:
            return label
    raise ValueError(f"Could not infer pathway source library from {source_file}")


def make_display_label(row: pd.Series) -> str:
    compact_label = " ".join(str(row["compact_label"]).replace("\r\n", "\n").replace("\r", "\n").splitlines())
    if compact_label == "JAK-STAT negative feedback":
        compact_label = "JAK-STAT negative\nfeedback"
    return f"{compact_label} ({int(row['overlap_count'])})"


def pathway_enrichment_table() -> pd.DataFrame:
    required_columns = [
        "side",
        "order_idx",
        "display_label",
        "compact_label",
        "source_file",
        "term",
        "fdr",
        "combined_score",
        "overlap_count",
        "genes",
        "top3_display",
    ]
    df = pd.read_csv(SELECTED_TERMS, sep="\t", dtype={"fdr": str, "combined_score": str})
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {SELECTED_TERMS}: {missing}")

    df = df.copy()
    df["source_fdr_text"] = df["fdr"].astype(str)
    df["source_combined_score_text"] = df["combined_score"].astype(str)
    df["order_idx"] = pd.to_numeric(df["order_idx"], errors="raise").astype(int)
    df["fdr"] = pd.to_numeric(df["fdr"], errors="raise")
    df["combined_score"] = pd.to_numeric(df["combined_score"], errors="raise")
    df["overlap_count"] = pd.to_numeric(df["overlap_count"], errors="raise").astype(int)
    if (df["fdr"] <= 0).any():
        bad_terms = df.loc[df["fdr"] <= 0, "term"].tolist()
        raise ValueError(f"Cannot plot non-positive FDR values for: {bad_terms}")

    df["panel"] = df["side"].map({"up": "high", "down": "low"})
    if df["panel"].isna().any():
        bad_sides = sorted(df.loc[df["panel"].isna(), "side"].astype(str).unique())
        raise ValueError(f"Unexpected pathway side values: {bad_sides}")

    df["display_order"] = df["order_idx"]
    df["display_label"] = df.apply(make_display_label, axis=1)
    df["display_genes"] = df["top3_display"]
    df["enriched_population"] = df["side"].map(
        {"up": "Tcf7-high sorted population", "down": "Tcf7-low sorted population"}
    )
    df["selected_terms_file"] = SELECTED_TERMS.relative_to(REPO_ROOT).as_posix()
    df["source_file_name"] = df["source_file"]
    df["source_file"] = df["source_file"].map(
        lambda filename: (CANONICAL_ANALYSIS_DIR / str(filename)).relative_to(REPO_ROOT).as_posix()
    )
    df["source_tool"] = "gseapy.enrichr"
    df["source_kind"] = "gseapy.enrichr cached Enrichr output"
    df["source_organism"] = "mouse"
    df["source_library"] = df["source_file_name"].map(source_library_from_file)
    df["source_term"] = df["term"]
    df["source_fdr"] = df["fdr"]
    df["source_neg_log10_fdr"] = df["source_fdr"].map(lambda fdr: -math.log10(fdr))
    df["source_combined_score"] = df["combined_score"]
    df["source_overlap_count"] = df["overlap_count"]
    df["source_genes"] = df["genes"]
    df["plot_score"] = df["source_neg_log10_fdr"]
    df["plot_score_label"] = "-log10(FDR)"
    df["plot_score_note"] = (
        "Plotted value is exactly -log10(fdr) from "
        "whole_genome_tcf7_pathway_summary_selected_terms.tsv."
    )
    df["is_plotted"] = True

    panel_order = pd.CategoricalDtype(categories=["high", "low"], ordered=True)
    df["panel"] = df["panel"].astype(panel_order)
    df = df.sort_values(["panel", "display_order"]).reset_index(drop=True)
    df["panel"] = df["panel"].astype(str)

    ordered_columns = [
        "panel",
        "side",
        "enriched_population",
        "display_order",
        "display_label",
        "display_genes",
        "source_tool",
        "source_kind",
        "source_organism",
        "source_library",
        "source_file",
        "selected_terms_file",
        "source_term",
        "source_fdr_text",
        "source_fdr",
        "source_neg_log10_fdr",
        "source_combined_score_text",
        "source_combined_score",
        "source_overlap_count",
        "source_genes",
        "plot_score",
        "plot_score_label",
        "plot_score_note",
        "is_plotted",
    ]
    return df[ordered_columns]


def plot_pathway_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    color: str,
    title: str,
    show_xlabel: bool,
    x_max: float,
) -> None:
    row_spacing = 1.55
    y_positions = [row_spacing * index for index in range(len(df) - 1, -1, -1)]
    label_x = -0.02

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.9, row_spacing * (len(df) - 1) + 0.9)
    ax.set_xticks(range(0, int(math.ceil(x_max)) + 1, 2))
    ax.set_yticks([])
    ax.grid(axis="x", color="#e6e6e6", linewidth=1.0)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for y, (_, row) in zip(y_positions, df.iterrows()):
        score = row["plot_score"]
        display_label = row["display_label"]
        has_wrapped_label = "\n" in display_label
        label_y = y + (0.48 if has_wrapped_label else 0.34)
        gene_y = y - (0.50 if has_wrapped_label else 0.38)
        ax.hlines(y, 0, score, color=color, alpha=0.55, linewidth=1.3, zorder=1)
        ax.scatter(
            score,
            y,
            s=90,
            color=color,
            edgecolors=color,
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            label_x,
            label_y,
            display_label,
            ha="right",
            va="center",
            fontsize=6.8 * AXIS_TEXT_SCALE,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            label_x,
            gene_y,
            row["display_genes"],
            ha="right",
            va="center",
            fontsize=5.9 * AXIS_TEXT_SCALE,
            color="#555555",
            clip_on=False,
        )

    ax.set_title(title, fontsize=9.4 * AXIS_TEXT_SCALE, color=color, pad=0, y=1.02)
    ax.tick_params(axis="x", labelsize=7.2 * AXIS_TEXT_SCALE, width=0.8, length=4)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("-log10(FDR)", fontsize=9.2 * AXIS_TEXT_SCALE, labelpad=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def save_pathway_source_table(df: pd.DataFrame) -> Path:
    output = OUTPUT_DIR / "g_pathway_sources.tsv"
    export_df = df.copy()
    for column in ["display_label", "display_genes", "source_term", "source_genes", "plot_score_note"]:
        export_df[column] = export_df[column].astype(str).str.replace("\n", "\\n", regex=False)
    export_df.to_csv(output, sep="\t", index=False, float_format="%.17g")
    return output


def strip_trailing_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: {png_path.name} {png_size:.2f}MB, {svg_path.name} {svg_size:.1f}KB")


def save_pathway_panel() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enrichment = pathway_enrichment_table()
    source_table = save_pathway_source_table(enrichment)
    plotted = enrichment[enrichment["is_plotted"]].copy()
    if plotted.empty:
        raise ValueError("No pathways were selected for plotting.")
    x_max = max(4.0, math.ceil(plotted["plot_score"].max() + 0.5))

    fig, axes = plt.subplots(nrows=2, figsize=(5.6, 7.56), sharex=True)
    fig.subplots_adjust(left=0.50, right=0.985, bottom=0.13, top=0.82, hspace=0.45)

    plot_pathway_axis(
        axes[0],
        plotted[plotted["panel"] == "high"],
        TARGET_HIGH_COLOR,
        "Enriched pathways\nin Tcf7 high cells",
        show_xlabel=False,
        x_max=x_max,
    )
    plot_pathway_axis(
        axes[1],
        plotted[plotted["panel"] == "low"],
        TARGET_LOW_COLOR,
        "Enriched pathways\nin Tcf7 low cells",
        show_xlabel=True,
        x_max=x_max,
    )

    png_path = OUTPUT_DIR / "g.png"
    svg_path = OUTPUT_DIR / "g.svg"
    fig.set_dpi(300)
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path, format="svg", dpi=300)
    strip_trailing_whitespace(svg_path)
    plt.close(fig)
    validate_file_sizes(png_path, svg_path)
    print(f"Saved pathway source table: {source_table.relative_to(REPO_ROOT)}")


def main() -> None:
    save_pathway_panel()
    print("Figure 1g pathway summary completed successfully!")


if __name__ == "__main__":
    main()
