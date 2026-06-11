#!/usr/bin/env python
"""
Create the Figure 1f whole-genome in vitro Tcf7 screen volcano plot.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
SCREEN_DATA = (
    REPO_ROOT
    / "raw_data"
    / "CRISPR_sort_screens"
    / "2_26_2021_BRIE_run1"
    / "hypergeometric"
    / "tcf7.txt"
)
PATHWAY_DIR = SCREEN_DATA.parent / "pathways"
STRING_HTML = PATHWAY_DIR / "STRING_ functional protein association networks.html"
GO_TCF7_ENRICHED = PATHWAY_DIR / "GO_tcf7_enriched.csv"
GO_TCF7_DEPLETED = PATHWAY_DIR / "GO_tcf7_depleted.csv"
KEGG_GSEA = PATHWAY_DIR / "tcf7gseaKEGG.csv"
GO_SCREEN_ENRICHED = (
    REPO_ROOT
    / "data"
    / "screen"
    / "pathway_results"
    / "GO_enriched_D7tcf7hi-low_100_21-03-01-18-00.txt.csv"
)
GO_SCREEN_DEPLETED = (
    REPO_ROOT
    / "data"
    / "screen"
    / "pathway_results"
    / "GO_depleted_D7tcf7hi-low_100_21-03-01-18-00.txt.csv"
)
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"

P_THRESHOLD = 0.01
NEG_LOG10_P_THRESHOLD = 2.0
LFC_THRESHOLD = 1.0
REQUIRED_GUIDES_PER_TARGET_GENE = 4

TARGET_HIGH_COLOR = "#c83f4f"
TARGET_LOW_COLOR = "#2f6fa7"
NON_TARGETING_COLOR = "#1a9850"
INTERGENIC_COLOR = "#6a5acd"
NONSIGNIFICANT_COLOR = "#d5d7da"
TOP_LABELS_PER_DIRECTION = 20
LABEL_X_OFFSET_DATA = 0.34
LABEL_MIN_Y_SEPARATION = 0.38
LABEL_MIN_Y = 3.15
LABEL_MAX_Y = 9.65
LABEL_LINE_COLOR = "#4d4d4d"
LABEL_TEXT_GAP_DATA = 0.025
LEFT_PANEL_TEXT_SCALE = 2.0
LEFT_PANEL_AXIS_LABEL_EXTRA_SCALE = 2.0
LEFT_PANEL_X_TICK_EXTRA_SCALE = 2.0
LEFT_PANEL_GENE_LABEL_EXTRA_SCALE = 1.5
LEFT_PANEL_WIDTH_INCHES = 10.08
LEFT_PANEL_HEIGHT_INCHES = 13.6
LEFT_PANEL_DIRECTION_ARROW_Y = 0.965
LEFT_PANEL_DIRECTION_LABEL_Y = 0.995
PATHWAY_PANELS = [
    {
        "panel": "high",
        "label": "Type II IFN (7)",
        "genes": "Stat1, Jak2, Ifngr2",
        "source": {"kind": "go_screen", "file": GO_SCREEN_ENRICHED, "term": "GO:0060333"},
    },
    {
        "panel": "high",
        "label": "IL-2 family / JAK-STAT (8)",
        "genes": "Stat1, Stat3, Jak2",
        "source": {"kind": "string", "category": "RCTM", "term": "MMU-9020558"},
    },
    {
        "panel": "high",
        "label": "T cell activation (7)",
        "genes": "Jak2, Smarce1, Lck",
        "source": {"kind": "go_screen", "file": GO_SCREEN_ENRICHED, "term": "GO:0046634"},
    },
    {
        "panel": "high",
        "label": "HIF-1 signaling (11)",
        "genes": "Arnt, Hif1a, Ldha",
        "source": {"kind": "kegg_gsea", "term": "mmu04066"},
    },
    {
        "panel": "high",
        "label": "Glycolysis (8)",
        "genes": "Ldha, Pkm, Gpi1",
        "source": {"kind": "kegg_gsea", "term": "mmu00010"},
    },
    {
        "panel": "low",
        "label": "JAK-STAT negative feedback (6)",
        "genes": "Ptpn2, Ptprc, Socs3",
        "source": {"kind": "go_clusterprofiler", "file": GO_TCF7_DEPLETED, "term": "GO:0046426"},
    },
    {
        "panel": "low",
        "label": "RNA Pol II tx (28)",
        "genes": "Pten, Foxo1, Tcf7",
        "source": {"kind": "string", "category": "KEGG", "term": "mmu03020"},
    },
    {
        "panel": "low",
        "label": "Neddylation (12)",
        "genes": "Ube2m, Socs3, Vhl",
        "source": {"kind": "string", "category": "Process", "term": "GO:0045116"},
    },
    {
        "panel": "low",
        "label": "Ubiquitin proteolysis (10)",
        "genes": "Ube2m, Socs3, Vhl",
        "source": {"kind": "kegg_gsea", "term": "mmu04120"},
    },
    {
        "panel": "low",
        "label": "Protein dephosphorylation (6)",
        "genes": "Pten, Ptpn2, Ptprc",
        "source": {"kind": "go_screen", "file": GO_SCREEN_DEPLETED, "term": "GO:0006470"},
    },
]

def load_screen() -> pd.DataFrame:
    df = pd.read_csv(SCREEN_DATA, sep="\t")
    required = ["gene_symbol", "LFC", "pvalue", "perturbations"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {SCREEN_DATA}: {missing}")

    df = df.copy()
    df["x"] = df["LFC"]
    df["y"] = df["pvalue"]
    df["category"] = df["gene_symbol"].map(categorize_gene)
    df = df[(df["category"] != "target") | (df["perturbations"] == REQUIRED_GUIDES_PER_TARGET_GENE)]
    df["is_significant_target"] = (
        (df["category"] == "target")
        & (df["y"] >= NEG_LOG10_P_THRESHOLD)
        & (df["x"].abs() >= LFC_THRESHOLD)
    )
    df["direction"] = "nonsignificant"
    df.loc[df["is_significant_target"] & (df["x"] > 0), "direction"] = "high"
    df.loc[df["is_significant_target"] & (df["x"] < 0), "direction"] = "low"
    return df


def categorize_gene(gene: str) -> str:
    if "INTERGENIC" in gene:
        return "intergenic"
    if gene.startswith("NO_SITE") or gene.startswith("INACTIVE") or gene.startswith("POTENTIALLY_ACTIVE"):
        return "non_targeting"
    return "target"


def plot_left_panel(ax: plt.Axes) -> None:
    df = load_screen()

    background = df[(df["category"] == "target") & (~df["is_significant_target"])]
    high = df[df["direction"] == "high"]
    low = df[df["direction"] == "low"]
    non_targeting = df[df["category"] == "non_targeting"]
    intergenic = df[df["category"] == "intergenic"]

    ax.scatter(
        background["x"],
        background["y"],
        s=5,
        c=NONSIGNIFICANT_COLOR,
        edgecolors="none",
        alpha=0.52,
        zorder=1,
        rasterized=True,
    )
    ax.scatter(
        high["x"],
        high["y"],
        s=8,
        c=TARGET_HIGH_COLOR,
        edgecolors=TARGET_HIGH_COLOR,
        linewidths=0.35,
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        low["x"],
        low["y"],
        s=8,
        c=TARGET_LOW_COLOR,
        edgecolors=TARGET_LOW_COLOR,
        linewidths=0.35,
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        non_targeting["x"],
        non_targeting["y"],
        s=20,
        facecolors="white",
        edgecolors=NON_TARGETING_COLOR,
        linewidths=0.75,
        zorder=4,
        rasterized=True,
    )
    ax.scatter(
        intergenic["x"],
        intergenic["y"],
        s=24,
        marker="D",
        facecolors="white",
        edgecolors=INTERGENIC_COLOR,
        linewidths=0.75,
        zorder=5,
        rasterized=True,
    )

    ax.axhline(NEG_LOG10_P_THRESHOLD, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(-LFC_THRESHOLD, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(LFC_THRESHOLD, color="black", linestyle="--", linewidth=0.8, zorder=0)

    label_hits(ax, df)

    ax.set_title(
        "Whole-genome in vitro Tcf7 screen",
        fontsize=8.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_AXIS_LABEL_EXTRA_SCALE,
        pad=9,
    )
    ax.set_xlabel(
        "log2 FC",
        fontsize=8.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_AXIS_LABEL_EXTRA_SCALE,
    )
    ax.set_ylabel(
        "-log10(p-value)",
        fontsize=8.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_AXIS_LABEL_EXTRA_SCALE,
    )
    ax.set_xlim(-13.0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.set_xticks(range(-12, 11, 4))
    ax.set_yticks(range(0, 11, 2))
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.45)
    ax.grid(axis="x", visible=False)
    ax.tick_params(
        axis="x",
        labelsize=7.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_X_TICK_EXTRA_SCALE,
        width=0.7,
        length=3,
    )
    ax.tick_params(
        axis="y",
        labelsize=7.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_X_TICK_EXTRA_SCALE,
        width=0.7,
        length=3,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    add_tcf7_direction_arrows(ax)


def label_hits(ax: plt.Axes, df: pd.DataFrame) -> None:
    high_hits = (
        df[df["direction"] == "high"]
        .sort_values(["y", "x"], ascending=[False, False])
        .head(TOP_LABELS_PER_DIRECTION)
    )
    low_hits = (
        df[df["direction"] == "low"]
        .sort_values(["y", "x"], ascending=[False, True])
        .head(TOP_LABELS_PER_DIRECTION)
    )

    label_records = direction_label_records(high_hits, "high") + direction_label_records(low_hits, "low")
    for record in label_records:
        ax.text(
            record["x_label"],
            record["y_label"],
            record["gene"],
            fontsize=6.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_GENE_LABEL_EXTRA_SCALE,
            ha=record["horizontal_alignment"],
            va="center",
            zorder=6,
            clip_on=False,
        )

    for record in label_records:
        point = record["point"]
        line_end_x = record["x_label"] + (
            LABEL_TEXT_GAP_DATA if record["horizontal_alignment"] == "right" else -LABEL_TEXT_GAP_DATA
        )
        ax.plot(
            [point["x"], line_end_x],
            [point["y"], record["y_label"]],
            color=LABEL_LINE_COLOR,
            linewidth=0.35,
            zorder=5,
            clip_on=False,
        )


def direction_label_records(hits: pd.DataFrame, direction: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    alignments = label_alignments_for_hits(hits, direction)
    for rank, (_, point) in enumerate(hits.iterrows()):
        horizontal_alignment = alignments[rank]
        if horizontal_alignment == "left":
            x_label = point["x"] + LABEL_X_OFFSET_DATA
        else:
            x_label = point["x"] - LABEL_X_OFFSET_DATA

        y_label = point["y"]
        records.append(
            {
                "gene": point["gene_symbol"],
                "point": point,
                "horizontal_alignment": horizontal_alignment,
                "x_label": x_label,
                "y_label": y_label,
            }
        )
    spread_label_y_positions(records)
    return records


def label_alignments_for_hits(hits: pd.DataFrame, direction: str) -> list[str]:
    alignments: list[str] = []
    hit_rows = [row for _, row in hits.iterrows()]
    for index in range(0, len(hit_rows), 2):
        pair = hit_rows[index : index + 2]
        if len(pair) == 1:
            alignments.append(outward_alignment(direction))
            continue

        if direction == "high":
            first_is_more_outward = pair[0]["x"] > pair[1]["x"]
        else:
            first_is_more_outward = pair[0]["x"] < pair[1]["x"]
        if first_is_more_outward:
            alignments.extend([outward_alignment(direction), inward_alignment(direction)])
        else:
            alignments.extend([inward_alignment(direction), outward_alignment(direction)])
    return alignments


def outward_alignment(direction: str) -> str:
    return "left" if direction == "high" else "right"


def inward_alignment(direction: str) -> str:
    return "right" if direction == "high" else "left"


def spread_label_y_positions(records: list[dict[str, object]]) -> None:
    for horizontal_alignment in ("left", "right"):
        side_records = [
            record
            for record in records
            if record["horizontal_alignment"] == horizontal_alignment
        ]
        side_records.sort(key=lambda record: float(record["point"]["y"]), reverse=True)
        y_positions: list[float] = []
        previous_y = LABEL_MAX_Y + LABEL_MIN_Y_SEPARATION
        for record in side_records:
            desired_y = min(max(float(record["point"]["y"]), LABEL_MIN_Y), LABEL_MAX_Y)
            y_label = min(desired_y, previous_y - LABEL_MIN_Y_SEPARATION)
            y_positions.append(y_label)
            previous_y = y_label

        if y_positions and y_positions[-1] < LABEL_MIN_Y:
            shift = LABEL_MIN_Y - y_positions[-1]
            y_positions = [y + shift for y in y_positions]

        for record, y_label in zip(side_records, y_positions):
            record["y_label"] = y_label


def add_tcf7_direction_arrows(ax: plt.Axes) -> None:
    arrow_kwargs = {
        "arrowstyle": "-|>",
        "linewidth": 1.5,
        "mutation_scale": 16,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(0.055, LEFT_PANEL_DIRECTION_ARROW_Y),
        xytext=(0.315, LEFT_PANEL_DIRECTION_ARROW_Y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_kwargs, "color": TARGET_LOW_COLOR},
        annotation_clip=False,
        zorder=8,
    )
    ax.text(
        0.185,
        LEFT_PANEL_DIRECTION_LABEL_Y,
        "Tcf7 Low",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_GENE_LABEL_EXTRA_SCALE,
        color=TARGET_LOW_COLOR,
        clip_on=False,
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(0.945, LEFT_PANEL_DIRECTION_ARROW_Y),
        xytext=(0.685, LEFT_PANEL_DIRECTION_ARROW_Y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_kwargs, "color": TARGET_HIGH_COLOR},
        annotation_clip=False,
        zorder=8,
    )
    ax.text(
        0.815,
        LEFT_PANEL_DIRECTION_LABEL_Y,
        "Tcf7 High",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5 * LEFT_PANEL_TEXT_SCALE * LEFT_PANEL_GENE_LABEL_EXTRA_SCALE,
        color=TARGET_HIGH_COLOR,
        clip_on=False,
        zorder=8,
    )


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(value), flags=re.IGNORECASE)
    if match is None:
        return math.nan
    return float(match.group(0))


def parse_ratio(value: object) -> tuple[int, int]:
    numerator, denominator = str(value).split("/", maxsplit=1)
    return int(numerator), int(denominator)


def log2_odds_from_ratios(gene_ratio: object, background_ratio: object) -> float:
    selected_in_term, selected_total = parse_ratio(gene_ratio)
    background_in_term, background_total = parse_ratio(background_ratio)
    selected_not_in_term = selected_total - selected_in_term
    background_not_selected_in_term = background_in_term - selected_in_term
    background_not_selected_not_in_term = (
        background_total
        - selected_in_term
        - selected_not_in_term
        - background_not_selected_in_term
    )
    if (
        selected_in_term <= 0
        or selected_not_in_term <= 0
        or background_not_selected_in_term <= 0
        or background_not_selected_not_in_term <= 0
    ):
        return math.nan
    odds_ratio = (
        selected_in_term
        * background_not_selected_not_in_term
        / (selected_not_in_term * background_not_selected_in_term)
    )
    return math.log2(odds_ratio)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    valid = pd.to_numeric(pvalues, errors="coerce").dropna().sort_values()
    ranks = pd.Series(range(1, len(valid) + 1), index=valid.index, dtype=float)
    adjusted = (valid * len(valid) / ranks).clip(upper=1.0)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]
    return adjusted.reindex(pvalues.index)


def load_string_rows() -> dict[tuple[str, str], dict[str, object]]:
    html = STRING_HTML.read_text(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    rows: dict[tuple[str, str], dict[str, object]] = {}

    for div in soup.find_all("div", class_="global_enrichment_table_content_container"):
        row_id = div.get("id", "")
        if ":" not in row_id:
            continue
        category = row_id.split(":", maxsplit=1)[0]
        children = div.find_all("div", recursive=False)
        if len(children) < 6:
            continue

        term = children[0].get_text(" ", strip=True)
        fdr_text = children[5].get_text(" ", strip=True)
        fdr = parse_float(fdr_text)
        score = parse_float(children[2].get_text(" ", strip=True))
        rows[(category, term)] = {
            "source_file": STRING_HTML.relative_to(REPO_ROOT).as_posix(),
            "source_kind": "STRING saved HTML",
            "source_category": category,
            "source_term": term,
            "source_description": children[1].get_text(" ", strip=True),
            "source_direction": children[3].get_text(" ", strip=True),
            "source_pathway_size": children[4].get_text(" ", strip=True),
            "source_score": score,
            "source_score_label": "enrichment score",
            "source_pvalue": math.nan,
            "source_fdr": fdr,
            "source_neg_log10_fdr": -math.log10(fdr) if fdr > 0 else math.nan,
            "source_genes": "",
        }
    return rows


def load_go_csv_record(path: Path, term: str, kind: str) -> dict[str, object]:
    df = pd.read_csv(path)
    term_column = "GO_term" if "GO_term" in df.columns else "ID"
    if "log2_odds_ratio" in df.columns and "p.adjust" not in df.columns:
        df["computed_bh_fdr"] = benjamini_hochberg(df["pvalue"])
    row = df.loc[df[term_column].astype(str).eq(term)]
    if row.empty:
        raise ValueError(f"Could not find {term} in {path}")
    record = row.iloc[0]

    if "log2_odds_ratio" in record.index:
        score = float(record["log2_odds_ratio"])
        score_label = "log2_odds_ratio"
        pvalue = float(record["pvalue"])
        fdr = float(record["computed_bh_fdr"])
        neg_log10_fdr = -math.log10(fdr) if fdr > 0 else math.nan
        size = record.get("overlap", "")
        genes = record.get("overlapping_genes", "")
    else:
        score = log2_odds_from_ratios(record["GeneRatio"], record["BgRatio"])
        score_label = "log2_odds_ratio"
        pvalue = float(record["pvalue"])
        fdr = float(record["p.adjust"])
        neg_log10_fdr = -math.log10(fdr) if fdr > 0 else math.nan
        size = record.get("Count", "")
        genes = str(record.get("geneID", "")).replace("/", ",")

    return {
        "source_file": path.relative_to(REPO_ROOT).as_posix(),
        "source_kind": kind,
        "source_category": "GO",
        "source_term": term,
        "source_description": record["GO_name"] if "GO_name" in record.index else record["Description"],
        "source_direction": "",
        "source_pathway_size": size,
        "source_score": score,
        "source_score_label": score_label,
        "source_pvalue": pvalue,
        "source_fdr": fdr,
        "source_neg_log10_fdr": neg_log10_fdr,
        "source_genes": genes,
    }


def load_kegg_gsea_records() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with KEGG_GSEA.open() as handle:
        next(handle)
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",", maxsplit=10)
            if len(parts) < 10:
                continue
            term, kegg_id, description, set_size, enrichment_score, nes, pvalue, p_adjust, qvalue, rank = parts[:10]
            fdr = float(p_adjust)
            records[term] = {
                "source_file": KEGG_GSEA.relative_to(REPO_ROOT).as_posix(),
                "source_kind": "KEGG GSEA",
                "source_category": "KEGG",
                "source_term": kegg_id,
                "source_description": description,
                "source_direction": "top of input" if float(nes) > 0 else "bottom of input",
                "source_pathway_size": set_size,
                "source_score": float(nes),
                "source_score_label": "NES",
                "source_pvalue": float(pvalue),
                "source_fdr": fdr,
                "source_neg_log10_fdr": -math.log10(fdr) if fdr > 0 else math.nan,
                "source_genes": "",
            }
    return records


def source_record_for(
    row: dict[str, object],
    string_rows: dict[tuple[str, str], dict[str, object]],
    kegg_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    source = row["source"]
    if source["kind"] == "string":
        key = (source["category"], source["term"])
        if key not in string_rows:
            raise ValueError(f"Could not find STRING row {key}")
        return string_rows[key].copy()
    if source["kind"] == "kegg_gsea":
        term = source["term"]
        if term not in kegg_rows:
            raise ValueError(f"Could not find KEGG GSEA row {term}")
        return kegg_rows[term].copy()
    if source["kind"] == "go_screen":
        return load_go_csv_record(source["file"], source["term"], "screen GO Fisher")
    if source["kind"] == "go_clusterprofiler":
        return load_go_csv_record(source["file"], source["term"], "clusterProfiler GO")
    raise ValueError(f"Unknown source kind: {source['kind']}")


def display_gene_screen_summary(genes: str, screen_by_gene: pd.DataFrame, column: str) -> str:
    values = []
    for gene in [gene.strip() for gene in genes.split(",")]:
        if not gene:
            continue
        if gene not in screen_by_gene.index:
            values.append(f"{gene}=NA")
            continue
        values.append(f"{gene}={screen_by_gene.loc[gene, column]:.4g}")
    return "; ".join(values)


def pathway_enrichment_table() -> pd.DataFrame:
    string_rows = load_string_rows()
    kegg_rows = load_kegg_gsea_records()
    screen_by_gene = load_screen().set_index("gene_symbol")
    records = []

    for display_order, row in enumerate(PATHWAY_PANELS):
        record = {
            "panel": row["panel"],
            "display_order": display_order,
            "display_label": row["label"],
            "display_genes": row["genes"],
            "display_gene_average_lfc_log2": display_gene_screen_summary(
                row["genes"],
                screen_by_gene,
                "x",
            ),
            "display_gene_source_lfc": display_gene_screen_summary(
                row["genes"],
                screen_by_gene,
                "LFC",
            ),
            "plot_score_note": (
                "legacy Figure 1g pathway selection; plotted value is -log10(FDR) "
                "from the saved pathway output. For screen GO Fisher rows, FDR was "
                "computed by Benjamini-Hochberg correction across the saved GO result file."
            ),
            "is_plotted": True,
        }
        record.update(source_record_for(row, string_rows, kegg_rows))
        if not math.isfinite(record["source_neg_log10_fdr"]):
            raise ValueError(f"Missing finite FDR for {row['label']}")
        record["plot_score"] = record["source_neg_log10_fdr"]
        record["plot_score_label"] = "-log10(FDR)"
        records.append(record)

    return pd.DataFrame(records)


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
    label_x = -0.18

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.9, row_spacing * (len(df) - 1) + 0.9)
    ax.set_xticks(range(0, int(math.ceil(x_max)) + 1, 2))
    ax.set_yticks([])
    ax.grid(axis="x", color="#e6e6e6", linewidth=1.0)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    for y, (_, row) in zip(y_positions, df.iterrows()):
        score = row["plot_score"]
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
            y + 0.34,
            row["display_label"],
            ha="right",
            va="center",
            fontsize=6.8,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            label_x,
            y - 0.38,
            row["display_genes"],
            ha="right",
            va="center",
            fontsize=5.9,
            color="#555555",
            clip_on=False,
        )

    ax.set_title(title, fontsize=9.4, color=color, pad=0, y=1.02)
    ax.tick_params(axis="x", labelsize=7.2, width=0.8, length=4)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("-log10(FDR)", fontsize=9.2, labelpad=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def save_pathway_source_table(df: pd.DataFrame) -> None:
    output = OUTPUT_DIR / "g_pathway_sources.tsv"
    df.to_csv(output, sep="\t", index=False, float_format="%.6g")


def save_right_panel() -> None:
    enrichment = pathway_enrichment_table()
    save_pathway_source_table(enrichment)
    plotted = enrichment[enrichment["is_plotted"]].copy()
    if plotted.empty:
        raise ValueError("No pathways were selected for plotting.")
    x_max = max(4.0, math.ceil(plotted["plot_score"].max() + 0.5))

    fig, axes = plt.subplots(nrows=2, figsize=(4.65, 5.15), sharex=True)
    fig.subplots_adjust(left=0.47, right=0.98, bottom=0.10, top=0.90, hspace=0.58)

    plot_pathway_axis(
        axes[0],
        plotted[plotted["panel"] == "high"],
        TARGET_HIGH_COLOR,
        "Enriched pathways (# hits)\nin Tcf7 high cells",
        show_xlabel=False,
        x_max=x_max,
    )
    plot_pathway_axis(
        axes[1],
        plotted[plotted["panel"] == "low"],
        TARGET_LOW_COLOR,
        "Enriched pathways (# hits)\nin Tcf7 low cells",
        show_xlabel=True,
        x_max=x_max,
    )
    save_figure(fig, "g")


def save_left_panel() -> None:
    fig, ax = plt.subplots(figsize=(LEFT_PANEL_WIDTH_INCHES, LEFT_PANEL_HEIGHT_INCHES))
    fig.subplots_adjust(left=0.21, right=0.99, bottom=0.18, top=0.91)
    plot_left_panel(ax)
    save_figure(fig, "f", transparent=True, tight=True)


def save_figure(fig: plt.Figure, stem: str, transparent: bool = False, tight: bool = False) -> None:
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.set_dpi(300)
    if transparent:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.patch.set_alpha(0)
    bbox_kwargs = {"bbox_inches": "tight", "pad_inches": 0.25} if tight else {}
    fig.savefig(png_path, dpi=300, transparent=transparent, **bbox_kwargs)
    fig.savefig(svg_path, format="svg", dpi=300, transparent=transparent, **bbox_kwargs)
    strip_trailing_whitespace(svg_path)
    plt.close(fig)
    validate_file_sizes(png_path, svg_path)


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_left_panel()
    print("Figure 1f output completed successfully!")


if __name__ == "__main__":
    main()
