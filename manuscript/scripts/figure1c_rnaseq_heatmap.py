#!/usr/bin/env python
"""
Generate Figure 1c from the gcloud Tcf7 paper RNA-seq data.

This is a Python refactor of the original R notebook workflow:
1. select the notebook's T cell activation/differentiation gene list,
2. collapse workflow-generated DESeq2-normalized expression values by cleaned sample group,
3. keep the six main groups shown in Figure 1c,
4. row-scale expression values, and
5. draw a pheatmap-style clustered heatmap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"

DEFAULT_TCF7_GCLOUD_RNASEQ_DIR = (
    REPO_ROOT / "gcloud" / "Tcf7_paper_data" / "RNAseq" / "tcellcombo"
)
DEFAULT_COUNTS = MANUSCRIPT_DIR / "derived" / "rnaseq_deseq2" / "deseq2_normalized_counts.tsv"
DEFAULT_METADATA = DEFAULT_TCF7_GCLOUD_RNASEQ_DIR / "meta_2_29_24.csv"
DEFAULT_OUTPUT_DIR = MANUSCRIPT_DIR / "plots" / "Figure1"

PANEL_WIDTH_IN = 4.68
PANEL_HEIGHT_IN = 9.15
DPI = 300

MAIN_GROUPS = [
    "Naive",
    "Lenti",
    "Activated",
    "D7 Control",
    "D7 Lenti",
    "D7 Activated",
]

MAIN_GROUP_DISPLAY_LABELS = {
    "Naive": "Naive",
    "Lenti": "Lenti naive",
    "Activated": "Activated",
    "D7 Control": "Control",
    "D7 Lenti": "Lenti naive",
    "D7 Activated": "Lenti activated",
}

REFERENCE_ROW_ORDER = [
    "Id2",
    "Gzma",
    "Ccr2",
    "Il15ra",
    "Gzmk",
    "Klrg1",
    "Itgam",
    "Cx3cr1",
    "Havcr2",
    "Klrk1",
    "Klrc1",
    "Entpd1",
    "Mki67",
    "Tbx21",
    "Tigit",
    "Cd44",
    "Prdm1",
    "Irf3",
    "Stat1",
    "Il7r",
    "Cd69",
    "Tgfbr1",
    "Ikzf1",
    "Cd3e",
    "Tgfbr2",
    "Smad2",
    "Irf7",
    "Runx1",
    "Eomes",
    "Socs1",
    "Lrrc32",
    "Foxp1",
    "Bcl9",
    "Ccr7",
    "Tcf7",
    "Il6ra",
    "Foxo1",
    "Sell",
    "Smad1",
    "Sox4",
    "Tgfb1",
    "Lap3",
    "Irf4",
    "Il2ra",
    "Ifng",
    "Stat3",
    "Kif2c",
    "Cdc25c",
    "Cdc6",
    "Cdk1",
    "Il2rb",
    "Gzmb",
    "Lag3",
]

NOTEBOOK_CD8_GENES = [
    "Foxo1",
    "Foxp1",
    "Irf3",
    "Sox4",
    "Tcf7",
    "Sell",
    "Il7r",
    "Ccr7",
    "Bcl9",
    "Eomes",
    "Id2",
    "Tbx21",
    "Socs1",
    "Prdm1",
    "Runx1",
    "Ikzf1",
    "Irf4",
    "Stat3",
    "Gzma",
    "Gzmb",
    "Gzmk",
    "Perf1",
    "Ifng",
    "Il2ra",
    "Tnfa",
    "Ccr2",
    "Havcr2",
    "Klrg1",
    "Entpd1",
    "Lag3",
    "Tigit",
    "Itgam",
    "Cx3cr1",
    "Mki67",
    "Kif2c",
    "Kif4",
    "Cdc6",
    "Cdk1",
    "Cdc25c",
    "Klrk1",
    "Klrc1",
    "Cd44",
    "Cd69",
    "Il2rb",
    "Il2rg",
    "Il15ra",
    "Lrrc32",
    "Trbv16",
    "Tgfbr1",
    "Tgfbr2",
    "Il6ra",
    "Cd3e",
    "Irf7",
    "Stat1",
    "Lap3",
    "Smad1",
    "Smad2",
    "Tgfb1",
]

GENE_GROUPS = {
    "Bcl9": "naive/memory-associated regulators",
    "Ccr7": "naive/memory-associated regulators",
    "Eomes": "naive/memory-associated regulators",
    "Foxo1": "naive/memory-associated regulators",
    "Foxp1": "naive/memory-associated regulators",
    "Id2": "naive/memory-associated regulators",
    "Il7r": "naive/memory-associated regulators",
    "Sell": "naive/memory-associated regulators",
    "Tcf7": "naive/memory-associated regulators",
    "Ccr2": "effector/cytotoxic markers",
    "Cx3cr1": "effector/cytotoxic markers",
    "Gzma": "effector/cytotoxic markers",
    "Gzmb": "effector/cytotoxic markers",
    "Gzmk": "effector/cytotoxic markers",
    "Ifng": "effector/cytotoxic markers",
    "Klrg1": "effector/cytotoxic markers",
    "Perf1": "effector/cytotoxic markers",
    "Tnfa": "effector/cytotoxic markers",
    "Cd3e": "activation/checkpoint/surface markers",
    "Cd44": "activation/checkpoint/surface markers",
    "Cd69": "activation/checkpoint/surface markers",
    "Entpd1": "activation/checkpoint/surface markers",
    "Havcr2": "activation/checkpoint/surface markers",
    "Il2ra": "activation/checkpoint/surface markers",
    "Il2rb": "activation/checkpoint/surface markers",
    "Il2rg": "activation/checkpoint/surface markers",
    "Il15ra": "activation/checkpoint/surface markers",
    "Itgam": "activation/checkpoint/surface markers",
    "Klrc1": "activation/checkpoint/surface markers",
    "Klrk1": "activation/checkpoint/surface markers",
    "Lag3": "activation/checkpoint/surface markers",
    "Tigit": "activation/checkpoint/surface markers",
    "Trbv16": "activation/checkpoint/surface markers",
    "Cdc25c": "cell-cycle genes",
    "Cdc6": "cell-cycle genes",
    "Cdk1": "cell-cycle genes",
    "Kif2c": "cell-cycle genes",
    "Kif4": "cell-cycle genes",
    "Mki67": "cell-cycle genes",
    "Il6ra": "cytokine/TGF-beta/IFN signaling",
    "Lap3": "cytokine/TGF-beta/IFN signaling",
    "Lrrc32": "cytokine/TGF-beta/IFN signaling",
    "Smad1": "cytokine/TGF-beta/IFN signaling",
    "Smad2": "cytokine/TGF-beta/IFN signaling",
    "Socs1": "cytokine/TGF-beta/IFN signaling",
    "Stat1": "cytokine/TGF-beta/IFN signaling",
    "Stat3": "cytokine/TGF-beta/IFN signaling",
    "Tgfb1": "cytokine/TGF-beta/IFN signaling",
    "Tgfbr1": "cytokine/TGF-beta/IFN signaling",
    "Tgfbr2": "cytokine/TGF-beta/IFN signaling",
    "Ikzf1": "transcriptional regulators",
    "Irf3": "transcriptional regulators",
    "Irf4": "transcriptional regulators",
    "Irf7": "transcriptional regulators",
    "Prdm1": "transcriptional regulators",
    "Runx1": "transcriptional regulators",
    "Sox4": "transcriptional regulators",
    "Tbx21": "transcriptional regulators",
}

NOTEBOOK_GENE_NOTE = (
    "Original R notebook cell 53 defined these as T cell activation/"
    "differentiation genes differentially expressed between LCMV control and "
    "naive cells."
)

BLUE_WHITE_RED_DIVERGING = [
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#F7F7F7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-stem", default="c")
    parser.add_argument("--row-order", choices=["cluster", "reference"], default="cluster")
    parser.add_argument("--vmin", type=float, default=-2.0)
    parser.add_argument("--vmax", type=float, default=2.0)
    parser.add_argument("--colorbar-label", default="Mean normalized expression row z-score")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Figure 1c data file is missing: {path}\n"
            "Expected input is the workflow-generated DESeq2-normalized matrix "
            "`manuscript/derived/rnaseq_deseq2/deseq2_normalized_counts.tsv`."
        )

    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def read_expression(path: Path) -> pd.DataFrame:
    data = read_table(path)
    data.columns = [str(col).strip() for col in data.columns]

    index_candidates = ["gene_name", "Gene", "gene", "Unnamed: 0", ""]
    index_col = next((col for col in index_candidates if col in data.columns), data.columns[0])
    data[index_col] = data[index_col].astype(str).str.strip()
    data = data.set_index(index_col)

    drop_cols = [col for col in ["gene_id", "gene"] if col in data.columns and col != index_col]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")
    data = data.groupby(data.index).mean()
    return data


def read_metadata(path: Path) -> pd.DataFrame:
    metadata = read_table(path)
    metadata.columns = [str(col).strip() for col in metadata.columns]
    if "sampletype" not in metadata.columns:
        raise ValueError(f"Metadata table must include a `sampletype` column: {path}")

    id_candidates = ["X", "sample", "sample_id", "Sample", "Unnamed: 0", ""]
    id_col = next((col for col in id_candidates if col in metadata.columns), metadata.columns[0])
    if id_col == "sampletype":
        raise ValueError(f"Could not identify sample IDs in metadata table: {path}")

    metadata[id_col] = metadata[id_col].astype(str).str.strip()
    metadata["sampletype"] = metadata["sampletype"].astype(str).str.strip()
    return metadata.set_index(id_col)


def clean_sampletype(sampletype: str) -> str | None:
    if sampletype == "Naive":
        return "Naive"
    if sampletype in {"Thy1.1pos", "Thy1.1neg"}:
        return "Lenti"
    if sampletype == "Tcellsstim":
        return "Activated"
    if sampletype in {"D8_naive_control", "D8_activated_control"}:
        return "D7 Control"
    if sampletype in {"D8_naive_Thy1.1pos", "D8_naive_Thy1.1neg"}:
        return "D7 Lenti"
    if sampletype in {"D8_activated_Thy1.1pos", "D8_activated_Thy1.1neg"}:
        return "D7 Activated"
    if sampletype in {
        "D30_naive_control_CM",
        "D30_fresh_Thy1.1_CM",
        "D30_fresh_control_CM",
        "D30_activated_control_CM",
    }:
        return "D30 Control CM"
    if sampletype in {
        "D30_naive_control_EM",
        "D30_fresh_Thy1.1_EM",
        "D30_fresh_control_EM",
        "D30_activated_control_EM",
    }:
        return "D30 Control EM"
    if sampletype == "D30_naive_Thy1.1_CM":
        return "D30 Lenti CM"
    if sampletype == "D30_naive_Thy1.1_EM":
        return "D30 Lenti EM"
    if sampletype == "D30_activated_Thy1.1_CM":
        return "D30 Activated CM"
    if sampletype == "D30_activated_Thy1.1_EM":
        return "D30 Activated EM"
    return None


def prepare_heatmap_data(expression: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_genes = [gene for gene in NOTEBOOK_CD8_GENES if gene in expression.index]
    if not available_genes:
        raise ValueError("None of the Figure 1c notebook genes were found in the normalized-count matrix.")

    metadata = metadata.copy()
    metadata["sampletype_clean"] = metadata["sampletype"].map(clean_sampletype)
    unmapped = sorted(metadata.loc[metadata["sampletype_clean"].isna(), "sampletype"].unique())
    if unmapped:
        raise ValueError(f"Unmapped sampletype values in metadata: {unmapped}")

    sample_cols = [sample for sample in expression.columns if sample in metadata.index]
    if not sample_cols:
        raise ValueError("No expression sample columns matched metadata sample IDs.")

    expression = expression.loc[available_genes, sample_cols]
    metadata = metadata.loc[sample_cols]

    collapsed = pd.DataFrame(index=expression.index)
    for group in MAIN_GROUPS:
        group_samples = metadata.index[metadata["sampletype_clean"] == group].tolist()
        if not group_samples:
            raise ValueError(f"No samples found for required Figure 1c group: {group}")
        collapsed[group] = expression[group_samples].mean(axis=1)

    row_means = collapsed.mean(axis=1)
    row_sds = collapsed.std(axis=1, ddof=1).replace(0, np.nan)
    row_scaled = collapsed.sub(row_means, axis=0).div(row_sds, axis=0)
    row_scaled = row_scaled.dropna(axis=0, how="any")
    return collapsed.loc[row_scaled.index], row_scaled


def order_rows(row_scaled: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, np.ndarray]:
    if strategy == "reference":
        ordered = [gene for gene in REFERENCE_ROW_ORDER if gene in row_scaled.index]
        remaining = [gene for gene in row_scaled.index if gene not in ordered]
        row_scaled = row_scaled.loc[ordered + remaining]

    row_linkage = linkage(row_scaled.values, method="complete", metric="euclidean")
    if strategy == "cluster":
        row_scaled = row_scaled.iloc[leaves_list(row_linkage)]
    return row_scaled, row_linkage


def write_gene_provenance(path: Path, row_scaled: pd.DataFrame) -> None:
    displayed_order = {gene: idx + 1 for idx, gene in enumerate(row_scaled.index)}
    records = []
    for idx, gene in enumerate(NOTEBOOK_CD8_GENES, start=1):
        records.append(
            {
                "gene": gene,
                "notebook_gene_list_order": idx,
                "shown_in_panel": gene in displayed_order,
                "display_order": displayed_order.get(gene, ""),
                "curation_group": GENE_GROUPS.get(gene, "other"),
                "selection_note": NOTEBOOK_GENE_NOTE,
            }
        )

    pd.DataFrame.from_records(records).to_csv(path, sep="\t", index=False)


def make_pheatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("blue_white_red_diverging", BLUE_WHITE_RED_DIVERGING, N=100)


def draw_heatmap(
    row_scaled: pd.DataFrame,
    row_linkage: np.ndarray,
    png_path: Path,
    svg_path: Path,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    fig = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_HEIGHT_IN), dpi=DPI, facecolor="white")
    gene_label_fontsize = 8.4
    colorbar_fontsize = gene_label_fontsize * 1.25

    dendro_ax = fig.add_axes([0.035, 0.16, 0.125, 0.76])
    heatmap_ax = fig.add_axes([0.165, 0.16, 0.51, 0.76])
    row_label_ax = fig.add_axes([0.682, 0.16, 0.105, 0.76])
    colorbar_ax = fig.add_axes([0.755, 0.35, 0.02, 0.38])

    dendrogram(
        row_linkage,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
        link_color_func=lambda _: "black",
        ax=dendro_ax,
    )
    dendro_ax.invert_yaxis()
    dendro_ax.axis("off")
    for line in dendro_ax.lines:
        line.set_linewidth(0.45)
    for collection in dendro_ax.collections:
        collection.set_linewidth(0.45)

    image = heatmap_ax.imshow(
        row_scaled.values,
        aspect="auto",
        interpolation="nearest",
        cmap=make_pheatmap_cmap(),
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    heatmap_ax.set_xticks(range(len(row_scaled.columns)))
    heatmap_ax.set_xticklabels(
        [MAIN_GROUP_DISPLAY_LABELS.get(group, group) for group in row_scaled.columns],
        rotation=270,
        fontsize=14,
        ha="left",
        va="center",
        rotation_mode="anchor",
    )
    heatmap_ax.tick_params(axis="x", length=0, pad=4)
    heatmap_ax.set_yticks([])
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)

    row_label_ax.set_xlim(0, 1)
    row_label_ax.set_ylim(len(row_scaled.index), 0)
    row_label_ax.axis("off")
    for row_idx, gene in enumerate(row_scaled.index):
        row_label_ax.text(0, row_idx + 0.5, gene, fontsize=gene_label_fontsize, ha="left", va="center")

    cbar = fig.colorbar(image, cax=colorbar_ax, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.tick_params(labelsize=colorbar_fontsize, length=0)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=colorbar_fontsize, labelpad=10)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")
    plt.close(fig)


def report_file_sizes(paths: list[Path]) -> None:
    for path in paths:
        size_mb = os.path.getsize(path) / 1024 / 1024
        if path.suffix.lower() == ".png" and size_mb > 5:
            print(f"ERROR: {path} is {size_mb:.2f}MB; PNG exceeds 5MB limit")
        elif path.suffix.lower() == ".svg" and size_mb * 1024 > 500:
            print(f"ERROR: {path} is {size_mb * 1024:.1f}KB; SVG exceeds 500KB limit")
        else:
            unit = "KB" if path.suffix.lower() == ".svg" else "MB"
            value = size_mb * 1024 if unit == "KB" else size_mb
            print(f"File size OK: {path} ({value:.1f}{unit})")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    expression = read_expression(args.normalized_counts)
    metadata = read_metadata(args.metadata)
    collapsed, row_scaled = prepare_heatmap_data(expression, metadata)
    row_scaled, row_linkage = order_rows(row_scaled, args.row_order)
    collapsed = collapsed.loc[row_scaled.index]

    png_path = output_dir / f"{args.panel_stem}.png"
    svg_path = output_dir / f"{args.panel_stem}.svg"
    gene_table_path = output_dir / f"{args.panel_stem}_gene_provenance.tsv"
    collapsed_path = output_dir / f"{args.panel_stem}_collapsed_group_means.tsv"
    scaled_path = output_dir / f"{args.panel_stem}_row_scaled_expression.tsv"

    draw_heatmap(
        row_scaled,
        row_linkage,
        png_path,
        svg_path,
        args.vmin,
        args.vmax,
        args.colorbar_label,
    )
    write_gene_provenance(gene_table_path, row_scaled)
    collapsed.to_csv(collapsed_path, sep="\t")
    row_scaled.to_csv(scaled_path, sep="\t")
    report_file_sizes([png_path, svg_path])
    print(f"Saved gene provenance: {gene_table_path}")
    print(f"Saved collapsed group means: {collapsed_path}")
    print(f"Saved row-scaled expression: {scaled_path}")


if __name__ == "__main__":
    main()
