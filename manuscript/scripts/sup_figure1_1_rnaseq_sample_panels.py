#!/usr/bin/env python
"""Generate Figure S1a-b RNA-seq sample panels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from figure1c_rnaseq_heatmap import (
    DEFAULT_COUNTS,
    DEFAULT_METADATA,
    MAIN_GROUPS,
    MAIN_GROUP_DISPLAY_LABELS,
    NOTEBOOK_CD8_GENES,
    REFERENCE_ROW_ORDER,
    clean_sampletype,
    make_pheatmap_cmap,
    read_expression,
    read_metadata,
)


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_1"
DEFAULT_PCA_COORDINATES = (
    REPO_ROOT / "manuscript" / "derived" / "rnaseq_deseq2" / "deseq2_plotpca_sup_figure1_1a_samples.tsv"
)
DEFAULT_MEMORY_PCA_COORDINATES = (
    REPO_ROOT
    / "manuscript"
    / "derived"
    / "rnaseq_deseq2"
    / "deseq2_plotpca_sup_figure1_1a_memory_samples.tsv"
)
DPI = 300

EXPERIMENTAL_MEMORY_GROUPS = [
    "D30 naive control CM",
    "D30 naive control EM",
    "D30 naive Thy1.1 CM",
    "D30 naive Thy1.1 EM",
    "D30 activated control CM",
    "D30 activated control EM",
    "D30 activated Thy1.1 CM",
    "D30 activated Thy1.1 EM",
    "D30 fresh control CM",
    "D30 fresh control EM",
    "D30 fresh Thy1.1 CM",
    "D30 fresh Thy1.1 EM",
]
ALL_SAMPLE_GROUPS = MAIN_GROUPS + EXPERIMENTAL_MEMORY_GROUPS
PCA_GROUPS = MAIN_GROUPS

MEMORY_CONDITION_ORDER = ["Control", "D30 naive", "Activated", "Fresh"]
MEMORY_CONDITION_SHAPES = {
    "Control": "o",
    "D30 naive": "s",
    "Activated": "^",
    "Fresh": "D",
}
MEMORY_CONDITION_LABELS = {
    "Control": "Control",
    "D30 naive": "Lenti",
    "Activated": "Activated",
    "Fresh": "Memory",
}
MEMORY_CONDITION_LABEL_COLORS = {
    "Control": "#555555",
    "D30 naive": "#2F6DA3",
    "Activated": "#2F8F4E",
    "Fresh": "#B36B00",
}
B_EXPERIMENT_V3_COMPARISONS = [
    ("Within group", "Control", "Control", "Control-Control"),
    ("Within group", "Fresh", "Fresh", "Fresh-Fresh"),
    ("Within group", "D30 naive", "D30 naive", "Lenti-Lenti"),
    ("Within group", "Activated", "Activated", "Activated-Activated"),
    ("Control vs group", "Control", "Fresh", "Control-Fresh"),
    ("Control vs group", "Control", "D30 naive", "Control-Lenti"),
    ("Control vs group", "Control", "Activated", "Control-Activated"),
]
B_EXPERIMENT_V4_CONDITION_ORDER = ["Fresh", "D30 naive", "Activated"]
B_EXPERIMENT_V4_CONDITION_LABELS = {
    "Fresh": "Fresh",
    "D30 naive": "Lenti",
    "Activated": "Activated",
}
B_EXPERIMENT_V4_SOURCE_COMPARISONS = [
    ("Within control", "control", "control"),
    ("Within Thy1.1", "Thy1.1", "Thy1.1"),
    ("Control vs Thy1.1", "control", "Thy1.1"),
]
B_EXPERIMENT_V4_COMPARISON_COLORS = {
    "Within control": "#8A8A8A",
    "Within Thy1.1": "#6A3D9A",
    "Control vs Thy1.1": "#CC7A00",
}
B_EXPERIMENT_V5_CONDITION_ORDER = ["Fresh", "D30 naive", "Activated"]
B_EXPERIMENT_V5_CONDITION_LABELS = {
    "Fresh": "Naive",
    "D30 naive": "Lenti naive",
    "Activated": "Lenti activated",
}
B_EXPERIMENT_V5_SUBSET_PANELS = [
    ("CM", "Central memory"),
    ("EM", "effector memory"),
]
FIGURE7B_NEUTRAL_AXIS_COLOR = "#202020"
FIGURE7B_NEUTRAL_BOX_FACE = "#B1B1B1"
C_EXPERIMENTAL_COMPARISONS = [
    {
        "panel_key": "in_vitro",
        "panel_order": 1,
        "panel_label": "In vitro",
        "comparison_label": "Naive vs Lenti naive",
        "x_label": "Lenti naive",
        "treatment_group": "Lenti",
        "control_group": "Naive",
    },
    {
        "panel_key": "in_vitro",
        "panel_order": 1,
        "panel_label": "In vitro",
        "comparison_label": "Naive vs Activated",
        "x_label": "Activated",
        "treatment_group": "Activated",
        "control_group": "Naive",
    },
    {
        "panel_key": "d7_lcmv",
        "panel_order": 2,
        "panel_label": "D7 LCMV in vivo",
        "comparison_label": "Control vs Lenti naive",
        "x_label": "Lenti naive",
        "treatment_group": "D7 Lenti",
        "control_group": "D7 Control",
    },
    {
        "panel_key": "d7_lcmv",
        "panel_order": 2,
        "panel_label": "D7 LCMV in vivo",
        "comparison_label": "Control vs Lenti activated",
        "x_label": "Lenti activated",
        "treatment_group": "D7 Activated",
        "control_group": "D7 Control",
    },
]
B_EXPERIMENT_SUBSET_PANELS = [
    ("both", "CM and EM", None),
    ("cm", "CM", "CM"),
    ("em", "EM", "EM"),
]
MEMORY_SOURCE_COLORS = {
    "control": "#8A8A8A",
    "Thy1.1": "#6A3D9A",
}
MEMORY_SUBSET_COLORS = {
    "CM": "#009E73",
    "EM": "#D55E00",
}
EXPERIMENTAL_MEMORY_GENES = [
    "Sell",
    "Cxcr5",
    "Lef1",
    "Id3",
    "Bach2",
    "Bcl2",
    "Ccr7",
    "Klf2",
    "Slamf6",
    "Cd27",
    "Tcf7",
    "Eomes",
    "Il7r",
    "Traf1",
    "S1pr1",
    "Cxcr3",
    "Nkg7",
    "Ccl5",
    "Klrg1",
    "S1pr5",
    "Cx3cr1",
    "Gzma",
    "Gzmb",
    "Fasl",
    "Klrd1",
    "Zeb2",
    "Tbx21",
    "Prdm1",
    "Id2",
    "Bhlhe40",
    "Zbtb32",
]

EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS = {
    "D30_naive_control_CM": "D30 naive control CM",
    "D30_naive_control_EM": "D30 naive control EM",
    "D30_naive_Thy1.1_CM": "D30 naive Thy1.1 CM",
    "D30_naive_Thy1.1_EM": "D30 naive Thy1.1 EM",
    "D30_activated_control_CM": "D30 activated control CM",
    "D30_activated_control_EM": "D30 activated control EM",
    "D30_activated_Thy1.1_CM": "D30 activated Thy1.1 CM",
    "D30_activated_Thy1.1_EM": "D30 activated Thy1.1 EM",
    "D30_fresh_control_CM": "D30 fresh control CM",
    "D30_fresh_control_EM": "D30 fresh control EM",
    "D30_fresh_Thy1.1_CM": "D30 fresh Thy1.1 CM",
    "D30_fresh_Thy1.1_EM": "D30 fresh Thy1.1 EM",
}

GROUP_COLORS = {
    "Naive": "#4C78A8",
    "Lenti": "#F58518",
    "Activated": "#54A24B",
    "D7 Control": "#B279A2",
    "D7 Lenti": "#72B7B2",
    "D7 Activated": "#E45756",
    "D8 Control": "#B279A2",
    "D8 Lenti": "#72B7B2",
    "D8 Activated": "#E45756",
    "D30 Control CM": "#2F4B7C",
    "D30 Control EM": "#7A5195",
    "D30 Lenti CM": "#00876C",
    "D30 Lenti EM": "#89C079",
    "D30 Activated CM": "#BC5090",
    "D30 Activated EM": "#FFA600",
    "D30 naive control CM": "#2F4B7C",
    "D30 naive control EM": "#5B8DB8",
    "D30 naive Thy1.1 CM": "#006D77",
    "D30 naive Thy1.1 EM": "#83C5BE",
    "D30 activated control CM": "#BC5090",
    "D30 activated control EM": "#EF5675",
    "D30 activated Thy1.1 CM": "#7A5195",
    "D30 activated Thy1.1 EM": "#FFA600",
    "D30 fresh control CM": "#00876C",
    "D30 fresh control EM": "#89C079",
    "D30 fresh Thy1.1 CM": "#8F6A00",
    "D30 fresh Thy1.1 EM": "#D6A01D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--pca-coordinates", type=Path, default=DEFAULT_PCA_COORDINATES)
    parser.add_argument("--memory-pca-coordinates", type=Path, default=DEFAULT_MEMORY_PCA_COORDINATES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vmin", type=float, default=-2.0)
    parser.add_argument("--vmax", type=float, default=2.0)
    return parser.parse_args()


def load_sample_matrix(
    counts_path: Path,
    metadata_path: Path,
    group_order: list[str],
    group_context: str,
    group_mapper=clean_sampletype,
    label_builder=None,
    row_order: str = "reference",
    extra_genes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression = read_expression(counts_path)
    metadata = read_metadata(metadata_path)
    metadata = metadata.copy()
    metadata["sample_group"] = metadata["sampletype"].map(group_mapper)
    unmapped = sorted(metadata.loc[metadata["sample_group"].isna(), "sampletype"].unique())
    if unmapped:
        raise ValueError(f"Unmapped sampletype values in {group_context} metadata: {unmapped}")

    extra_genes = extra_genes or []
    missing_extra_genes = [gene for gene in extra_genes if gene not in expression.index]
    if missing_extra_genes:
        raise ValueError(f"Requested extra genes are missing from {group_context} expression matrix: {missing_extra_genes}")
    candidate_genes = list(dict.fromkeys(NOTEBOOK_CD8_GENES + extra_genes))
    available_genes = [gene for gene in candidate_genes if gene in expression.index]
    ordered_genes = [gene for gene in REFERENCE_ROW_ORDER if gene in available_genes]
    ordered_genes.extend(gene for gene in available_genes if gene not in ordered_genes)

    sample_records: list[dict[str, object]] = []
    sample_order: list[str] = []
    for group in group_order:
        group_samples = [
            sample
            for sample in expression.columns
            if (
                sample in metadata.index
                and metadata.loc[sample, "sample_group"] == group
            )
        ]
        if not group_samples:
            raise ValueError(f"No samples found for required {group_context} group: {group}")

        for replicate_idx, sample in enumerate(group_samples, start=1):
            sample_order.append(sample)
            sampletype = metadata.loc[sample, "sampletype"]
            record = {
                "sample": sample,
                "sampletype": sampletype,
                "sample_group": group,
                "display_order": len(sample_order),
            }
            if label_builder is not None:
                record["x_axis_label"] = label_builder(sample=sample, sampletype=sampletype, group=group, replicate_idx=replicate_idx)
            sample_records.append(record)

    matrix = expression.loc[ordered_genes, sample_order]
    row_means = matrix.mean(axis=1)
    row_sds = matrix.std(axis=1, ddof=1).replace(0, np.nan)
    row_scaled = matrix.sub(row_means, axis=0).div(row_sds, axis=0).dropna(axis=0, how="any")
    if row_order == "cluster":
        row_linkage = linkage(row_scaled.values, method="complete", metric="euclidean")
        row_scaled = row_scaled.iloc[leaves_list(row_linkage)]
    elif row_order != "reference":
        raise ValueError(f"Unsupported row_order for {group_context}: {row_order}")
    normalized = matrix.loc[row_scaled.index]
    sample_metadata = pd.DataFrame.from_records(sample_records)
    return normalized, row_scaled, sample_metadata


def load_figure1c_sample_matrix(
    counts_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_sample_matrix(
        counts_path,
        metadata_path,
        MAIN_GROUPS,
        "Figure 1c",
        label_builder=figure1c_x_axis_label,
    )


def load_all_sample_matrix(
    counts_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_sample_matrix(
        counts_path,
        metadata_path,
        ALL_SAMPLE_GROUPS,
        "Figure S1b experimental",
        group_mapper=experimental_sample_group,
        label_builder=experimental_x_axis_label,
        row_order="cluster",
        extra_genes=EXPERIMENTAL_MEMORY_GENES,
    )


def experimental_sample_group(sampletype: str) -> str | None:
    if sampletype in EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS:
        return EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS[sampletype]
    return clean_sampletype(sampletype)


def experimental_x_axis_label(sample: str, sampletype: str, group: str, replicate_idx: int) -> str:
    if sampletype in EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS:
        compact_group = group.replace(" CM", "CM").replace(" EM", "EM")
        return f"{compact_group} {replicate_idx}"
    return sample


def figure1c_group_display_label(group: str) -> str:
    return MAIN_GROUP_DISPLAY_LABELS.get(group, group)


def figure1c_x_axis_label(sample: str, sampletype: str, group: str, replicate_idx: int) -> str:
    thy1_label_by_sampletype = {
        "Thy1.1neg": "Thy1.1-",
        "Thy1.1pos": "Thy1.1+",
        "D8_naive_Thy1.1neg": "Thy1.1-",
        "D8_naive_Thy1.1pos": "Thy1.1+",
        "D8_activated_Thy1.1neg": "Thy1.1-",
        "D8_activated_Thy1.1pos": "Thy1.1+",
    }
    display_group = figure1c_group_display_label(group)
    thy1_label = thy1_label_by_sampletype.get(sampletype)
    if thy1_label is None:
        return display_group
    return f"{display_group} {thy1_label}"


def memory_input_condition(sample_group: str) -> str:
    if " control " in sample_group:
        return "Control"
    if sample_group.startswith("D30 naive Thy1.1 "):
        return "D30 naive"
    if sample_group.startswith("D30 activated Thy1.1 "):
        return "Activated"
    if sample_group.startswith("D30 fresh Thy1.1 "):
        return "Fresh"
    raise ValueError(f"Cannot assign memory input condition for sample group: {sample_group}")


def memory_source(sample_group: str) -> str:
    if " Thy1.1 " in sample_group:
        return "Thy1.1"
    if " control " in sample_group:
        return "control"
    raise ValueError(f"Cannot assign memory source for sample group: {sample_group}")


def memory_subset(sample_group: str) -> str:
    if sample_group.endswith(" CM"):
        return "CM"
    if sample_group.endswith(" EM"):
        return "EM"
    raise ValueError(f"Cannot assign CM/EM subset for sample group: {sample_group}")


def memory_input_context(sample_group: str) -> str:
    if sample_group.startswith("D30 naive "):
        return "D30 naive"
    if sample_group.startswith("D30 activated "):
        return "Activated"
    if sample_group.startswith("D30 fresh "):
        return "Fresh"
    raise ValueError(f"Cannot assign memory input context for sample group: {sample_group}")


def memory_mouse_id(sample: str) -> int:
    try:
        sample_number = int(str(sample))
    except ValueError as exc:
        raise ValueError(f"Memory sample ID is not numeric and cannot be assigned to a mouse block: {sample}") from exc
    if sample_number < 1:
        raise ValueError(f"Memory sample ID must be positive to assign mouse block: {sample}")
    return (sample_number - 1) // 4 + 1


def memory_mouse_position(sample: str) -> int:
    sample_number = int(str(sample))
    return (sample_number - 1) % 4 + 1


def subset_memory_sample_metadata(
    sample_metadata: pd.DataFrame,
    subset_value: str | None,
) -> pd.DataFrame:
    if subset_value is None:
        return sample_metadata.copy()

    subset_metadata = sample_metadata[sample_metadata["memory_subset"] == subset_value].copy()
    if subset_metadata.empty:
        raise ValueError(f"No memory samples found for subset: {subset_value}")
    return subset_metadata


def b_experiment_output_stem(base_stem: str, subset_key: str) -> str:
    if subset_key == "both":
        return base_stem
    return f"{base_stem}_{subset_key}"


def tag_subset_tables(
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    subset_key: str,
    subset_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_df = pairwise_df.copy()
    summary_df = summary_df.copy()
    pairwise_df.insert(0, "subset", subset_key)
    pairwise_df.insert(1, "subset_label", subset_label)
    summary_df.insert(0, "subset", subset_key)
    summary_df.insert(1, "subset_label", subset_label)
    return pairwise_df, summary_df


def build_pca_sample_metadata(metadata_path: Path, group_order: list[str]) -> pd.DataFrame:
    metadata = read_metadata(metadata_path)
    metadata = metadata.copy()
    metadata["sample_group"] = metadata["sampletype"].map(experimental_sample_group)
    unmapped = sorted(metadata.loc[metadata["sample_group"].isna(), "sampletype"].unique())
    if unmapped:
        raise ValueError(f"Unmapped sampletype values in PCA metadata: {unmapped}")

    sample_records: list[dict[str, object]] = []
    for group in group_order:
        group_samples = metadata.index[metadata["sample_group"] == group].tolist()
        if not group_samples:
            raise ValueError(f"No samples found for required Figure S1a group: {group}")

        for sample in group_samples:
            sample_records.append(
                {
                    "sample": sample,
                    "sampletype": metadata.loc[sample, "sampletype"],
                    "sample_group": group,
                    "display_order": len(sample_records) + 1,
                }
            )

    return pd.DataFrame.from_records(sample_records)


def load_memory_expression_and_metadata(
    counts_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = read_expression(counts_path)
    metadata = read_metadata(metadata_path)
    metadata = metadata.copy()
    metadata["sample_group"] = metadata["sampletype"].map(experimental_sample_group)

    sample_records: list[dict[str, object]] = []
    sample_order: list[str] = []
    for group in EXPERIMENTAL_MEMORY_GROUPS:
        group_samples = [
            sample
            for sample in expression.columns
            if sample in metadata.index and metadata.loc[sample, "sample_group"] == group
        ]
        if not group_samples:
            raise ValueError(f"No samples found for required memory correlation group: {group}")

        for replicate_idx, sample in enumerate(group_samples, start=1):
            input_condition = memory_input_condition(group)
            source = memory_source(group)
            subset = memory_subset(group)
            sample_order.append(sample)
            sample_records.append(
                {
                    "sample": sample,
                    "sampletype": metadata.loc[sample, "sampletype"],
                    "sample_group": group,
                    "input_condition": input_condition,
                    "label_category": MEMORY_CONDITION_LABELS[input_condition],
                    "label_color": MEMORY_CONDITION_LABEL_COLORS[input_condition],
                    "source_category": source,
                    "source_color": MEMORY_SOURCE_COLORS[source],
                    "memory_subset": subset,
                    "memory_subset_color": MEMORY_SUBSET_COLORS[subset],
                    "display_label": experimental_x_axis_label(
                        sample=sample,
                        sampletype=metadata.loc[sample, "sampletype"],
                        group=group,
                        replicate_idx=replicate_idx,
                    ),
                    "display_order": len(sample_order),
                }
            )

    return expression.loc[:, sample_order], pd.DataFrame.from_records(sample_records)


def load_deseq2_pca_coordinates(
    pca_coordinates_path: Path,
    pca_sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    pca_df = pd.read_csv(pca_coordinates_path, sep="\t")
    required_columns = {
        "sample",
        "sampletype",
        "sample_group",
        "PC1",
        "PC2",
        "PC1_variance_percent",
        "PC2_variance_percent",
    }
    missing = sorted(required_columns - set(pca_df.columns))
    if missing:
        raise ValueError(f"DESeq2 PCA coordinate table is missing columns: {missing}")

    pca_df["sample"] = pca_df["sample"].astype(str)
    expected_samples = pca_sample_metadata["sample"].tolist()
    pca_df = pca_df.set_index("sample")
    missing_samples = [sample for sample in expected_samples if sample not in pca_df.index]
    if missing_samples:
        raise ValueError(f"DESeq2 PCA coordinate table is missing samples: {missing_samples}")

    pca_df = pca_df.loc[expected_samples].reset_index()
    pca_df["display_order"] = pca_sample_metadata["display_order"].to_numpy()
    expected_groups = pca_sample_metadata["sample_group"].to_numpy()
    observed_groups = pca_df["sample_group"].to_numpy()
    if not np.array_equal(expected_groups, observed_groups):
        raise ValueError("DESeq2 PCA coordinate table sample groups do not match the metadata-derived PCA groups.")

    explained = np.array(
        [
            pca_df["PC1_variance_percent"].iloc[0] / 100,
            pca_df["PC2_variance_percent"].iloc[0] / 100,
        ]
    )
    return pca_df, explained


def plot_pca(
    pca_df: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    output_stem: str = "a",
    group_order: list[str] | None = None,
    figsize: tuple[float, float] = (7.5, 6.2),
    legend_fontsize: float = 19.0,
    point_size: float = 136,
    shape_column: str | None = None,
    shape_map: dict[str, str] | None = None,
    shape_order: list[str] | None = None,
    legend_bbox: tuple[float, float] = (1.02, 0.5),
    shape_legend_bbox: tuple[float, float] = (1.02, 0.02),
) -> None:
    if group_order is None:
        group_order = PCA_GROUPS
    if (shape_column is None) != (shape_map is None):
        raise ValueError("shape_column and shape_map must be provided together.")

    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    label_fontsize = 30
    tick_fontsize = 24

    for group in group_order:
        group_df = pca_df[pca_df["sample_group"] == group]
        if group_df.empty:
            continue
        marker = "o"
        if shape_column is not None and shape_map is not None:
            group_shapes = group_df[shape_column].dropna().unique()
            if len(group_shapes) != 1:
                raise ValueError(f"Expected one {shape_column} value for PCA group {group}; found {group_shapes}")
            marker = shape_map[group_shapes[0]]

        ax.scatter(
            group_df["PC1"],
            group_df["PC2"],
            s=point_size,
            color=GROUP_COLORS[group],
            marker=marker,
            edgecolor="black",
            linewidth=0.3,
            label=figure1c_group_display_label(group),
            zorder=3,
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=label_fontsize)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)", fontsize=label_fontsize)
    group_legend = ax.legend(
        frameon=False,
        fontsize=legend_fontsize,
        loc="center left",
        bbox_to_anchor=legend_bbox,
        borderaxespad=0,
        handletextpad=0.4,
        markerscale=1.0,
    )
    legend_artists = [group_legend]
    if shape_map is not None:
        ax.add_artist(group_legend)
        shape_order = shape_order or list(shape_map)
        shape_handles = [
            Line2D(
                [0],
                [0],
                marker=shape_map[label],
                linestyle="None",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=12.0,
                label=label,
            )
            for label in shape_order
        ]
        shape_legend = ax.legend(
            handles=shape_handles,
            frameon=False,
            fontsize=legend_fontsize,
            loc="lower left",
            bbox_to_anchor=shape_legend_bbox,
            borderaxespad=0,
            handletextpad=0.6,
            markerscale=2.0,
        )
        legend_artists.append(shape_legend)
    ax.tick_params(labelsize=tick_fontsize)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=DPI,
        bbox_inches="tight",
        bbox_extra_artists=tuple(legend_artists),
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{output_stem}.svg",
        format="svg",
        bbox_inches="tight",
        bbox_extra_artists=tuple(legend_artists),
        facecolor="white",
    )
    plt.close(fig)


def cluster_correlation_matrix(correlation: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if correlation.shape[0] < 2:
        return correlation, np.empty((0, 4)), correlation.index.tolist()

    clipped = correlation.clip(lower=-1.0, upper=1.0)
    distance = 1.0 - clipped
    np.fill_diagonal(distance.values, 0.0)
    condensed_distance = squareform(distance.to_numpy(), checks=False)
    sample_linkage = linkage(condensed_distance, method="average")
    ordered_labels = correlation.index[leaves_list(sample_linkage)].tolist()
    return correlation.loc[ordered_labels, ordered_labels], sample_linkage, ordered_labels


def compute_memory_sample_correlations(
    expression: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    panel_specs = {
        "all": ("CM and EM", sample_metadata),
        "cm": ("CM", sample_metadata[sample_metadata["memory_subset"] == "CM"]),
        "em": ("EM", sample_metadata[sample_metadata["memory_subset"] == "EM"]),
    }
    log_expression = np.log2(expression + 1.0)
    log_expression = log_expression.loc[log_expression.var(axis=1, ddof=1) > 0]
    results: dict[str, dict[str, object]] = {}
    for key, (title, panel_metadata) in panel_specs.items():
        samples = panel_metadata["sample"].tolist()
        correlation = log_expression.loc[:, samples].corr(method="pearson")
        ordered_correlation, sample_linkage, ordered_samples = cluster_correlation_matrix(correlation)
        results[key] = {
            "title": title,
            "metadata": panel_metadata,
            "correlation": correlation,
            "ordered_correlation": ordered_correlation,
            "linkage": sample_linkage,
            "ordered_samples": ordered_samples,
        }
    return results


def compute_b_experiment_v3_pairwise_correlations(
    correlation_results: dict[str, dict[str, object]],
    sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = correlation_results["all"]["correlation"].copy()
    correlation.index = correlation.index.astype(str)
    correlation.columns = correlation.columns.astype(str)

    metadata = sample_metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata_by_sample = metadata.set_index("sample")
    condition_to_samples = {
        condition: metadata.loc[metadata["input_condition"] == condition, "sample"].tolist()
        for condition in MEMORY_CONDITION_ORDER
    }

    pairwise_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for comparison_order, (comparison_type, condition_a, condition_b, comparison_label) in enumerate(
        B_EXPERIMENT_V3_COMPARISONS,
        start=1,
    ):
        samples_a = condition_to_samples[condition_a]
        samples_b = condition_to_samples[condition_b]
        if not samples_a or not samples_b:
            raise ValueError(
                "Missing samples for b_experiment_v3 comparison: "
                f"{condition_a} vs {condition_b}"
            )

        if condition_a == condition_b:
            sample_pairs = [
                (samples_a[i], samples_a[j])
                for i in range(len(samples_a))
                for j in range(i + 1, len(samples_a))
            ]
        else:
            sample_pairs = [
                (sample_a, sample_b)
                for sample_a in samples_a
                for sample_b in samples_b
            ]
        if not sample_pairs:
            raise ValueError(f"No pairwise correlations available for b_experiment_v3 comparison: {comparison_label}")

        values: list[float] = []
        for sample_a, sample_b in sample_pairs:
            sample_a_record = metadata_by_sample.loc[sample_a]
            sample_b_record = metadata_by_sample.loc[sample_b]
            value = float(correlation.loc[sample_a, sample_b])
            values.append(value)
            pairwise_records.append(
                {
                    "comparison_order": comparison_order,
                    "comparison_type": comparison_type,
                    "comparison_label": comparison_label,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "sample_a": sample_a,
                    "sample_a_group": sample_a_record["sample_group"],
                    "sample_a_memory_subset": sample_a_record["memory_subset"],
                    "sample_b": sample_b,
                    "sample_b_group": sample_b_record["sample_group"],
                    "sample_b_memory_subset": sample_b_record["memory_subset"],
                    "pearson_correlation": value,
                }
            )

        flat_values = np.array(values)
        summary_records.append(
            {
                "comparison_order": comparison_order,
                "comparison_type": comparison_type,
                "comparison_label": comparison_label,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_condition_a_samples": len(samples_a),
                "n_condition_b_samples": len(samples_b),
                "n_pairwise_correlations": flat_values.size,
                "mean_pearson_correlation": float(np.mean(flat_values)),
                "sd_pairwise_correlation": float(np.std(flat_values, ddof=1)),
                "sem_pairwise_correlation": float(np.std(flat_values, ddof=1) / np.sqrt(flat_values.size)),
                "min_pairwise_correlation": float(np.min(flat_values)),
                "max_pairwise_correlation": float(np.max(flat_values)),
            }
        )

    return pd.DataFrame.from_records(pairwise_records), pd.DataFrame.from_records(summary_records)


def compute_b_experiment_v4_pairwise_correlations(
    correlation_results: dict[str, dict[str, object]],
    sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = correlation_results["all"]["correlation"].copy()
    correlation.index = correlation.index.astype(str)
    correlation.columns = correlation.columns.astype(str)

    metadata = sample_metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata["input_context"] = metadata["sample_group"].map(memory_input_context)
    metadata_by_sample = metadata.set_index("sample")

    pairwise_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    comparison_order = 0
    for condition_order, condition in enumerate(B_EXPERIMENT_V4_CONDITION_ORDER, start=1):
        condition_label = B_EXPERIMENT_V4_CONDITION_LABELS[condition]
        condition_metadata = metadata[metadata["input_context"] == condition]
        for source_comparison_order, (source_comparison, source_a, source_b) in enumerate(
            B_EXPERIMENT_V4_SOURCE_COMPARISONS,
            start=1,
        ):
            comparison_order += 1
            samples_a = condition_metadata.loc[
                condition_metadata["source_category"] == source_a,
                "sample",
            ].tolist()
            samples_b = condition_metadata.loc[
                condition_metadata["source_category"] == source_b,
                "sample",
            ].tolist()
            if not samples_a or not samples_b:
                raise ValueError(
                    "Missing samples for b_experiment_v4 comparison: "
                    f"{condition_label} {source_comparison}"
                )

            if source_a == source_b:
                sample_pairs = [
                    (samples_a[i], samples_a[j])
                    for i in range(len(samples_a))
                    for j in range(i + 1, len(samples_a))
                ]
            else:
                sample_pairs = [
                    (sample_a, sample_b)
                    for sample_a in samples_a
                    for sample_b in samples_b
                ]
            if not sample_pairs:
                raise ValueError(
                    "No pairwise correlations available for b_experiment_v4 comparison: "
                    f"{condition_label} {source_comparison}"
                )

            correlations: list[float] = []
            for sample_a, sample_b in sample_pairs:
                sample_a_record = metadata_by_sample.loc[sample_a]
                sample_b_record = metadata_by_sample.loc[sample_b]
                correlation_value = float(correlation.loc[sample_a, sample_b])
                correlations.append(correlation_value)
                pairwise_records.append(
                    {
                        "comparison_order": comparison_order,
                        "condition_order": condition_order,
                        "condition": condition,
                        "condition_label": condition_label,
                        "source_comparison_order": source_comparison_order,
                        "source_comparison": source_comparison,
                        "source_a": source_a,
                        "source_b": source_b,
                        "sample_a": sample_a,
                        "sample_a_group": sample_a_record["sample_group"],
                        "sample_a_memory_subset": sample_a_record["memory_subset"],
                        "sample_b": sample_b,
                        "sample_b_group": sample_b_record["sample_group"],
                        "sample_b_memory_subset": sample_b_record["memory_subset"],
                        "pearson_correlation": correlation_value,
                    }
                )

            correlation_values = np.array(correlations)
            summary_records.append(
                {
                    "comparison_order": comparison_order,
                    "condition_order": condition_order,
                    "condition": condition,
                    "condition_label": condition_label,
                    "source_comparison_order": source_comparison_order,
                    "source_comparison": source_comparison,
                    "source_a": source_a,
                    "source_b": source_b,
                    "n_source_a_samples": len(samples_a),
                    "n_source_b_samples": len(samples_b),
                    "n_pairwise_correlations": correlation_values.size,
                    "mean_pearson_correlation": float(np.mean(correlation_values)),
                    "sd_pairwise_correlation": float(np.std(correlation_values, ddof=1)),
                    "sem_pairwise_correlation": float(np.std(correlation_values, ddof=1) / np.sqrt(correlation_values.size)),
                    "min_pearson_correlation": float(np.min(correlation_values)),
                    "max_pearson_correlation": float(np.max(correlation_values)),
                }
            )

    return pd.DataFrame.from_records(pairwise_records), pd.DataFrame.from_records(summary_records)


def compute_b_experimental_v5_pairwise_correlations(
    correlation_results: dict[str, dict[str, object]],
    sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = correlation_results["all"]["correlation"].copy()
    correlation.index = correlation.index.astype(str)
    correlation.columns = correlation.columns.astype(str)

    metadata = sample_metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata["input_context"] = metadata["sample_group"].map(memory_input_context)
    metadata_by_sample = metadata.set_index("sample")

    pairwise_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    comparison_order = 0
    for subset_order, (subset, subset_label) in enumerate(B_EXPERIMENT_V5_SUBSET_PANELS, start=1):
        subset_metadata = metadata[metadata["memory_subset"] == subset]
        for condition_order, condition in enumerate(B_EXPERIMENT_V5_CONDITION_ORDER, start=1):
            comparison_order += 1
            condition_label = B_EXPERIMENT_V5_CONDITION_LABELS[condition]
            condition_metadata = subset_metadata[subset_metadata["input_context"] == condition]
            control_samples = condition_metadata.loc[
                condition_metadata["source_category"] == "control",
                "sample",
            ].tolist()
            thy11_samples = condition_metadata.loc[
                condition_metadata["source_category"] == "Thy1.1",
                "sample",
            ].tolist()
            if not control_samples or not thy11_samples:
                raise ValueError(
                    "Missing samples for b_experimental_v5 comparison: "
                    f"{subset_label} {condition_label} control vs Thy1.1"
                )

            correlations: list[float] = []
            for control_sample in control_samples:
                for thy11_sample in thy11_samples:
                    control_sample = str(control_sample)
                    thy11_sample = str(thy11_sample)
                    control_record = metadata_by_sample.loc[control_sample]
                    thy11_record = metadata_by_sample.loc[thy11_sample]
                    correlation_value = float(correlation.loc[control_sample, thy11_sample])
                    correlations.append(correlation_value)
                    pairwise_records.append(
                        {
                            "comparison_order": comparison_order,
                            "subset_order": subset_order,
                            "subset": subset,
                            "subset_label": subset_label,
                            "condition_order": condition_order,
                            "condition": condition,
                            "condition_label": condition_label,
                            "source_comparison": "Control vs Thy1.1",
                            "source_a": "control",
                            "source_b": "Thy1.1",
                            "sample_a": control_sample,
                            "sample_a_group": control_record["sample_group"],
                            "sample_a_memory_subset": control_record["memory_subset"],
                            "sample_b": thy11_sample,
                            "sample_b_group": thy11_record["sample_group"],
                            "sample_b_memory_subset": thy11_record["memory_subset"],
                            "pearson_correlation": correlation_value,
                        }
                    )

            correlation_values = np.array(correlations)
            summary_records.append(
                {
                    "comparison_order": comparison_order,
                    "subset_order": subset_order,
                    "subset": subset,
                    "subset_label": subset_label,
                    "condition_order": condition_order,
                    "condition": condition,
                    "condition_label": condition_label,
                    "source_comparison": "Control vs Thy1.1",
                    "source_a": "control",
                    "source_b": "Thy1.1",
                    "n_control_samples": len(control_samples),
                    "n_thy1_1_samples": len(thy11_samples),
                    "n_pairwise_correlations": correlation_values.size,
                    "mean_pearson_correlation": float(np.mean(correlation_values)),
                    "sd_pairwise_correlation": float(np.std(correlation_values, ddof=1)),
                    "sem_pairwise_correlation": float(
                        np.std(correlation_values, ddof=1) / np.sqrt(correlation_values.size)
                    ),
                    "min_pearson_correlation": float(np.min(correlation_values)),
                    "max_pearson_correlation": float(np.max(correlation_values)),
                }
            )

    return pd.DataFrame.from_records(pairwise_records), pd.DataFrame.from_records(summary_records)


def adjust_pvalues_bh(p_values: list[float]) -> np.ndarray:
    p_values_array = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values_array.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values_array)
    if not valid.any():
        return adjusted

    valid_p_values = p_values_array[valid]
    order = np.argsort(valid_p_values)
    ranked = valid_p_values[order] * valid_p_values.size / np.arange(1, valid_p_values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    valid_adjusted = np.empty_like(valid_p_values)
    valid_adjusted[order] = np.minimum(ranked, 1.0)
    adjusted[valid] = valid_adjusted
    return adjusted


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 1e-4:
        return "P < 0.0001"
    if p_value < 0.001:
        return f"P = {p_value:.1e}"
    return f"P = {p_value:.3f}"


def compute_b_experimental_v5_naive_activated_stats(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for subset_order, (subset, subset_label) in enumerate(B_EXPERIMENT_V5_SUBSET_PANELS, start=1):
        subset_pairwise = pairwise_df[pairwise_df["subset"] == subset]
        naive = subset_pairwise.loc[
            subset_pairwise["condition"] == "D30 naive",
            "pearson_correlation",
        ].to_numpy()
        activated = subset_pairwise.loc[
            subset_pairwise["condition"] == "Activated",
            "pearson_correlation",
        ].to_numpy()
        if naive.size == 0 or activated.size == 0:
            raise ValueError(f"Missing Naive or Activated values for b_experimental_v5 {subset_label}")

        test = mannwhitneyu(naive, activated, alternative="two-sided", method="auto")
        records.append(
            {
                "comparison_order": subset_order,
                "subset_order": subset_order,
                "subset": subset,
                "subset_label": subset_label,
                "group_a": "D30 naive",
                "group_a_label": B_EXPERIMENT_V5_CONDITION_LABELS["D30 naive"],
                "group_b": "Activated",
                "group_b_label": B_EXPERIMENT_V5_CONDITION_LABELS["Activated"],
                "n_group_a_pairwise_correlations": naive.size,
                "n_group_b_pairwise_correlations": activated.size,
                "mean_group_a_pearson_correlation": float(np.mean(naive)),
                "mean_group_b_pearson_correlation": float(np.mean(activated)),
                "median_group_a_pearson_correlation": float(np.median(naive)),
                "median_group_b_pearson_correlation": float(np.median(activated)),
                "test": "Mann-Whitney U",
                "alternative": "two-sided",
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )

    stats_df = pd.DataFrame.from_records(records)
    stats_df["p_adjusted_bh"] = adjust_pvalues_bh(stats_df["p_value"].tolist())
    stats_df["plot_label"] = stats_df["p_value"].map(significance_label)
    return stats_df


def compute_c_experimental_pairwise_correlations(
    counts_path: Path,
    sample_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = read_expression(counts_path)
    metadata = sample_metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata_by_sample = metadata.set_index("sample")
    required_group_set = {
        group
        for comparison in C_EXPERIMENTAL_COMPARISONS
        for group in (comparison["treatment_group"], comparison["control_group"])
    }
    required_groups = [group for group in MAIN_GROUPS if group in required_group_set]
    group_to_samples = {
        group: metadata.loc[metadata["sample_group"] == group, "sample"].tolist()
        for group in required_groups
    }
    missing_groups = [group for group, samples in group_to_samples.items() if not samples]
    if missing_groups:
        raise ValueError(f"Missing samples for c_experimental groups: {missing_groups}")

    samples = list(dict.fromkeys(sample for group in required_groups for sample in group_to_samples[group]))
    missing_samples = [sample for sample in samples if sample not in expression.columns]
    if missing_samples:
        raise ValueError(f"Missing c_experimental samples from normalized counts: {missing_samples}")

    log_expression = np.log2(expression.loc[:, samples] + 1.0)
    log_expression = log_expression.loc[log_expression.var(axis=1, ddof=1) > 0]
    correlation = log_expression.corr(method="pearson")

    pairwise_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    comparison_order_by_panel: dict[str, int] = {}
    for comparison_order, comparison in enumerate(C_EXPERIMENTAL_COMPARISONS, start=1):
        panel_key = comparison["panel_key"]
        comparison_order_by_panel[panel_key] = comparison_order_by_panel.get(panel_key, 0) + 1
        comparison_order_within_panel = comparison_order_by_panel[panel_key]
        panel_order = comparison["panel_order"]
        panel_label = comparison["panel_label"]
        comparison_label = comparison["comparison_label"]
        treatment_group = comparison["treatment_group"]
        control_group = comparison["control_group"]
        x_label = comparison["x_label"]
        treatment_group_label = figure1c_group_display_label(treatment_group)
        control_group_label = figure1c_group_display_label(control_group)
        treatment_samples = group_to_samples[treatment_group]
        control_samples = group_to_samples[control_group]
        sample_pairs = [
            (treatment_sample, control_sample)
            for treatment_sample in treatment_samples
            for control_sample in control_samples
        ]
        correlations: list[float] = []
        for treatment_sample, control_sample in sample_pairs:
            treatment_record = metadata_by_sample.loc[treatment_sample]
            control_record = metadata_by_sample.loc[control_sample]
            correlation_value = float(correlation.loc[treatment_sample, control_sample])
            correlations.append(correlation_value)
            pairwise_records.append(
                {
                    "comparison_order": comparison_order,
                    "panel_order": panel_order,
                    "panel_key": panel_key,
                    "panel_label": panel_label,
                    "comparison_order_within_panel": comparison_order_within_panel,
                    "comparison_label": comparison_label,
                    "x_label": x_label,
                    "treatment_group": treatment_group,
                    "treatment_group_label": treatment_group_label,
                    "control_group": control_group,
                    "control_group_label": control_group_label,
                    "sample_a": treatment_sample,
                    "sample_a_group": treatment_record["sample_group"],
                    "sample_b": control_sample,
                    "sample_b_group": control_record["sample_group"],
                    "pearson_correlation": correlation_value,
                }
            )

        correlation_values = np.array(correlations)
        summary_records.append(
            {
                "comparison_order": comparison_order,
                "panel_order": panel_order,
                "panel_key": panel_key,
                "panel_label": panel_label,
                "comparison_order_within_panel": comparison_order_within_panel,
                "comparison_label": comparison_label,
                "x_label": x_label,
                "treatment_group": treatment_group,
                "treatment_group_label": treatment_group_label,
                "control_group": control_group,
                "control_group_label": control_group_label,
                "n_treatment_samples": len(treatment_samples),
                "n_control_samples": len(control_samples),
                "n_pairwise_correlations": correlation_values.size,
                "mean_pearson_correlation": float(np.mean(correlation_values)),
                "sd_pairwise_correlation": float(np.std(correlation_values, ddof=1)),
                "sem_pairwise_correlation": float(
                    np.std(correlation_values, ddof=1) / np.sqrt(correlation_values.size)
                ),
                "min_pearson_correlation": float(np.min(correlation_values)),
                "max_pearson_correlation": float(np.max(correlation_values)),
            }
        )

    return pd.DataFrame.from_records(pairwise_records), pd.DataFrame.from_records(summary_records)


def compute_c_experimental_stats(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    comparison_metadata = (
        pairwise_df[
            [
                "panel_order",
                "panel_key",
                "panel_label",
                "comparison_order",
                "comparison_order_within_panel",
                "comparison_label",
                "x_label",
            ]
        ]
        .drop_duplicates()
        .sort_values(["panel_order", "comparison_order_within_panel"])
    )
    for panel_key, panel_metadata in comparison_metadata.groupby("panel_key", sort=False):
        panel_metadata = panel_metadata.sort_values("comparison_order_within_panel")
        if panel_metadata.shape[0] != 2:
            raise ValueError(
                "Expected exactly two comparisons for c_experimental panel "
                f"{panel_key}, found {panel_metadata.shape[0]}"
            )

        group_a_record = panel_metadata.iloc[0]
        group_b_record = panel_metadata.iloc[1]
        group_a = pairwise_df.loc[
            pairwise_df["comparison_label"] == group_a_record["comparison_label"],
            "pearson_correlation",
        ].to_numpy()
        group_b = pairwise_df.loc[
            pairwise_df["comparison_label"] == group_b_record["comparison_label"],
            "pearson_correlation",
        ].to_numpy()
        if group_a.size == 0 or group_b.size == 0:
            raise ValueError(f"Missing pairwise correlations for c_experimental panel {panel_key}")

        test = mannwhitneyu(group_a, group_b, alternative="two-sided", method="auto")
        records.append(
            {
                "comparison_order": int(group_a_record["panel_order"]),
                "panel_order": int(group_a_record["panel_order"]),
                "panel_key": panel_key,
                "panel_label": group_a_record["panel_label"],
                "group_a": group_a_record["comparison_label"],
                "group_a_label": group_a_record["x_label"],
                "group_b": group_b_record["comparison_label"],
                "group_b_label": group_b_record["x_label"],
                "n_group_a_pairwise_correlations": group_a.size,
                "n_group_b_pairwise_correlations": group_b.size,
                "mean_group_a_pearson_correlation": float(np.mean(group_a)),
                "mean_group_b_pearson_correlation": float(np.mean(group_b)),
                "median_group_a_pearson_correlation": float(np.median(group_a)),
                "median_group_b_pearson_correlation": float(np.median(group_b)),
                "test": "Mann-Whitney U",
                "alternative": "two-sided",
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )

    stats_df = pd.DataFrame.from_records(records)
    stats_df["p_adjusted_bh"] = adjust_pvalues_bh(stats_df["p_value"].tolist())
    stats_df["plot_label"] = stats_df["p_value"].map(significance_label)
    return stats_df


def memory_annotation_color_table(sample_metadata: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    metadata_by_sample = sample_metadata.set_index("sample").loc[samples]
    return pd.DataFrame(
        {
            "Input": metadata_by_sample["label_color"],
            "Source": metadata_by_sample["source_color"],
            "Subset": metadata_by_sample["memory_subset_color"],
        },
        index=samples,
    )


def memory_annotation_legend_handles() -> list[Patch]:
    input_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"input: {MEMORY_CONDITION_LABELS[condition].lower()}")
        for condition, color in MEMORY_CONDITION_LABEL_COLORS.items()
    ]
    source_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"Source: {source}")
        for source, color in MEMORY_SOURCE_COLORS.items()
    ]
    subset_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"Subset: {subset}")
        for subset, color in MEMORY_SUBSET_COLORS.items()
    ]
    return input_handles + source_handles + subset_handles


def clear_clustermap_annotation_axis_labels(grid: sns.matrix.ClusterGrid) -> None:
    for axis_name in ["ax_row_colors", "ax_col_colors"]:
        axis = getattr(grid, axis_name, None)
        if axis is None:
            continue
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.tick_params(
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelleft=False,
            labelright=False,
            labelbottom=False,
            labeltop=False,
        )
        for text in axis.texts:
            text.set_visible(False)
        for spine in axis.spines.values():
            spine.set_visible(False)


def plot_single_memory_correlation_clustermap(
    result: dict[str, object],
    sample_metadata: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
    vmin: float,
) -> None:
    correlation = result["correlation"]
    sample_linkage = result["linkage"]
    samples = correlation.index.tolist()
    n_samples = len(samples)
    label_lookup = sample_metadata.set_index("sample")["display_label"].to_dict()
    display_labels = [label_lookup[sample] for sample in samples]

    labeled_correlation = correlation.copy()
    labeled_correlation.index = display_labels
    labeled_correlation.columns = display_labels
    annotation_colors = memory_annotation_color_table(sample_metadata, samples)
    annotation_colors.index = display_labels

    figsize = (7.2, 7.8) if n_samples > 40 else (6.2, 6.8)
    label_fontsize = 3.2 if n_samples > 40 else 5.0
    grid = sns.clustermap(
        labeled_correlation,
        row_linkage=sample_linkage,
        col_linkage=sample_linkage,
        row_colors=annotation_colors,
        col_colors=annotation_colors,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=1.0,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.12, 0.12),
        colors_ratio=0.035,
        cbar_pos=(0.04, 0.94, 0.16, 0.020),
        cbar_kws={
            "label": "Pearson correlation",
            "ticks": [vmin, (vmin + 1.0) / 2.0, 1.0],
            "orientation": "horizontal",
        },
        tree_kws={"linewidths": 0.8, "colors": "black"},
        rasterized=True,
    )
    for collection in grid.ax_heatmap.collections:
        collection.set_rasterized(True)
    clear_clustermap_annotation_axis_labels(grid)
    grid.fig.suptitle(str(result["title"]), fontsize=12, y=0.995)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="both", length=0, pad=1)
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(),
        rotation=90,
        fontsize=label_fontsize,
        ha="center",
        va="top",
        color="black",
    )
    grid.ax_heatmap.set_yticklabels(
        grid.ax_heatmap.get_yticklabels(),
        fontsize=label_fontsize,
        color="black",
    )
    for spine in grid.ax_heatmap.spines.values():
        spine.set_visible(False)
    grid.fig.legend(
        handles=memory_annotation_legend_handles(),
        frameon=False,
        fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.01),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    grid.fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    grid.fig.savefig(output_dir / f"{output_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(grid.fig)


def plot_memory_correlation_clustermaps(
    correlation_results: dict[str, dict[str, object]],
    sample_metadata: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "a_experimental_v2",
) -> None:
    panel_outputs = [
        ("all", f"{output_stem}_left"),
        ("cm", f"{output_stem}_middle"),
        ("em", f"{output_stem}_right"),
    ]
    panel_correlations = [
        correlation_results[key]["ordered_correlation"]
        for key, _ in panel_outputs
    ]
    global_min = min(float(correlation.min().min()) for correlation in panel_correlations)
    vmin = max(0.5, np.floor(global_min * 20.0) / 20.0)
    for key, output_name in panel_outputs:
        plot_single_memory_correlation_clustermap(
            correlation_results[key],
            sample_metadata,
            output_dir,
            output_stem=output_name,
            vmin=vmin,
        )


def save_memory_correlation_outputs(
    correlation_results: dict[str, dict[str, object]],
    sample_metadata: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "a_experimental_v2",
) -> None:
    sample_metadata.to_csv(output_dir / f"{output_stem}_sample_metadata.tsv", sep="\t", index=False)
    order_records: list[dict[str, object]] = []
    for key, result in correlation_results.items():
        ordered_correlation = result["ordered_correlation"]
        ordered_correlation.to_csv(output_dir / f"{output_stem}_correlation_{key}.tsv", sep="\t")
        for cluster_order, sample in enumerate(result["ordered_samples"], start=1):
            sample_record = sample_metadata.set_index("sample").loc[sample].to_dict()
            order_records.append(
                {
                    "panel": key,
                    "panel_title": result["title"],
                    "cluster_order": cluster_order,
                    "sample": sample,
                    **sample_record,
                }
            )
    pd.DataFrame.from_records(order_records).to_csv(
        output_dir / f"{output_stem}_cluster_order.tsv",
        sep="\t",
        index=False,
    )


def plot_b_experiment_v3_correlation_boxplots(
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "b_experiment_v3",
    subset_label: str | None = None,
) -> None:
    summary_df = summary_df.sort_values("comparison_order").reset_index(drop=True)
    pairwise_df = pairwise_df.copy()
    x_positions = np.arange(summary_df.shape[0])
    x_lookup = dict(zip(summary_df["comparison_label"], x_positions))
    bar_colors = [
        MEMORY_CONDITION_LABEL_COLORS[condition]
        for condition in summary_df["condition_b"]
    ]

    y_min = max(0.0, np.floor((pairwise_df["pearson_correlation"].min() - 0.02) * 20.0) / 20.0)
    y_max = min(1.0, np.ceil((pairwise_df["pearson_correlation"].max() + 0.01) * 20.0) / 20.0)
    if y_max - y_min < 0.1:
        y_min = max(0.0, y_max - 0.1)

    fig, ax = plt.subplots(figsize=(6.1, 3.3), dpi=DPI)
    plot_values = [
        pairwise_df.loc[pairwise_df["comparison_label"] == label, "pearson_correlation"].to_numpy()
        for label in summary_df["comparison_label"]
    ]
    boxplot = ax.boxplot(
        plot_values,
        positions=x_positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        zorder=2,
    )
    for patch, color in zip(boxplot["boxes"], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)

    rng = np.random.default_rng(0)
    for comparison_label, group_df in pairwise_df.groupby("comparison_label", sort=False):
        x = x_lookup[comparison_label]
        jitter = rng.normal(0.0, 0.055, size=group_df.shape[0])
        ax.scatter(
            np.full(group_df.shape[0], x) + jitter,
            group_df["pearson_correlation"],
            s=4.5,
            color="black",
            alpha=0.16,
            linewidth=0,
            rasterized=True,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.replace("-", "\n") for label in summary_df["comparison_label"]], fontsize=8)
    ax.axvline(3.5, color="#A8A8A8", linewidth=0.7, linestyle="--", zorder=1)
    ax.text(1.5, 1.02, "Within group", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ax.text(5.0, 1.02, "Control vs group", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
    ylabel = "pairwise Pearson correlation" if subset_label is None else f"{subset_label}\npairwise Pearson correlation"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_b_experiment_v4_correlation_boxplots(
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "b_experiment_v4",
    subset_label: str | None = None,
) -> None:
    summary_df = summary_df.sort_values("comparison_order").reset_index(drop=True)
    pairwise_df = pairwise_df.copy()
    x_positions = (
        (summary_df["condition_order"].to_numpy() - 1) * 4
        + summary_df["source_comparison_order"].to_numpy()
        - 1
    )
    comparison_to_x = dict(zip(summary_df["comparison_order"], x_positions))
    box_colors = [
        B_EXPERIMENT_V4_COMPARISON_COLORS[source_comparison]
        for source_comparison in summary_df["source_comparison"]
    ]

    y_min = max(0.0, np.floor((pairwise_df["pearson_correlation"].min() - 0.02) * 20.0) / 20.0)
    y_max = min(1.0, np.ceil((pairwise_df["pearson_correlation"].max() + 0.01) * 20.0) / 20.0)
    if y_max - y_min < 0.1:
        y_min = max(0.0, y_max - 0.1)
    fig, ax = plt.subplots(figsize=(7.3, 3.4), dpi=DPI)
    plot_values = [
        pairwise_df.loc[pairwise_df["comparison_order"] == comparison_order, "pearson_correlation"].to_numpy()
        for comparison_order in summary_df["comparison_order"]
    ]
    boxplot = ax.boxplot(
        plot_values,
        positions=x_positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        zorder=2,
    )
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)

    rng = np.random.default_rng(1)
    for comparison_order, group_df in pairwise_df.groupby("comparison_order", sort=False):
        x = comparison_to_x[comparison_order]
        jitter = rng.normal(0.0, 0.055, size=group_df.shape[0])
        ax.scatter(
            np.full(group_df.shape[0], x) + jitter,
            group_df["pearson_correlation"],
            s=4.5,
            color="black",
            alpha=0.16,
            linewidth=0,
            rasterized=True,
            zorder=3,
        )

    source_tick_labels = {
        "Within control": "Control",
        "Within Thy1.1": "Thy1.1",
        "Control vs Thy1.1": "Control vs\nThy1.1",
    }
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [source_tick_labels[label] for label in summary_df["source_comparison"]],
        fontsize=7.4,
    )
    for condition_order, condition_label in (
        summary_df[["condition_order", "condition_label"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        center = (condition_order - 1) * 4 + 1
        ax.text(center, 1.02, condition_label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
    for separator in [2.95, 6.95]:
        ax.axvline(separator, color="#A8A8A8", linewidth=0.7, linestyle="--", zorder=1)

    ax.set_xlim(-0.85, x_positions[-1] + 0.85)
    ylabel = "pairwise Pearson correlation" if subset_label is None else f"{subset_label}\npairwise Pearson correlation"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_b_experimental_v5_correlation_boxplots(
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "d",
) -> None:
    suptitle_fontsize = 18
    title_fontsize = 18
    tick_fontsize = 14
    label_fontsize = 16
    annotation_fontsize = 16
    point_size = 20
    jitter_width = 0.16

    summary_df = summary_df.sort_values(["subset_order", "condition_order"]).reset_index(drop=True)
    pairwise_df = pairwise_df.copy()

    y_min = max(0.0, np.floor((pairwise_df["pearson_correlation"].min() - 0.02) * 20.0) / 20.0)
    y_max = min(1.0, np.ceil((pairwise_df["pearson_correlation"].max() + 0.01) * 20.0) / 20.0)
    if y_max - y_min < 0.1:
        y_min = max(0.0, y_max - 0.1)
    y_ticks = np.linspace(y_min, y_max, 5)
    y_range = y_max - y_min
    annotation_y = y_max - 0.12 * y_range
    annotation_tick = 0.025 * y_range
    annotation_text_y = annotation_y + annotation_tick + 0.012 * y_range

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=DPI, sharey=True)
    rng = np.random.default_rng(5)
    x_positions = np.arange(len(B_EXPERIMENT_V5_CONDITION_ORDER)) * 0.70
    for ax, (subset, subset_title) in zip(axes, B_EXPERIMENT_V5_SUBSET_PANELS):
        subset_summary = summary_df[summary_df["subset"] == subset].sort_values("condition_order")
        subset_pairwise = pairwise_df[pairwise_df["subset"] == subset]
        plot_values = [
            subset_pairwise.loc[
                subset_pairwise["comparison_order"] == comparison_order,
                "pearson_correlation",
            ].to_numpy()
            for comparison_order in subset_summary["comparison_order"]
        ]
        boxplot = ax.boxplot(
            plot_values,
            positions=x_positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.2},
            whiskerprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            capprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            boxprops={"edgecolor": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            zorder=2,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(FIGURE7B_NEUTRAL_BOX_FACE)
            patch.set_alpha(1.0)

        comparison_to_x = dict(zip(subset_summary["comparison_order"], x_positions))
        for comparison_order, group_df in subset_pairwise.groupby("comparison_order", sort=False):
            x = comparison_to_x[comparison_order]
            jitter = rng.uniform(-jitter_width / 2, jitter_width / 2, size=group_df.shape[0])
            ax.scatter(
                np.full(group_df.shape[0], x) + jitter,
                group_df["pearson_correlation"],
                s=point_size,
                color=FIGURE7B_NEUTRAL_AXIS_COLOR,
                alpha=0.88,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )

        stats_row = stats_df.loc[stats_df["subset"] == subset]
        if stats_row.empty:
            raise ValueError(f"Missing Naive vs Activated statistics for b_experimental_v5 {subset_title}")
        stats_row = stats_row.iloc[0]
        naive_x = x_positions[B_EXPERIMENT_V5_CONDITION_ORDER.index("D30 naive")]
        activated_x = x_positions[B_EXPERIMENT_V5_CONDITION_ORDER.index("Activated")]
        ax.plot(
            [naive_x, naive_x, activated_x, activated_x],
            [annotation_y, annotation_y + annotation_tick, annotation_y + annotation_tick, annotation_y],
            color=FIGURE7B_NEUTRAL_AXIS_COLOR,
            linewidth=1.0,
            clip_on=False,
            zorder=4,
        )
        ax.text(
            (naive_x + activated_x) / 2,
            annotation_text_y,
            str(stats_row["plot_label"]),
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            clip_on=False,
        )

        ax.set_title(subset_title, fontsize=title_fontsize, pad=18)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
        [label.replace("Lenti ", "Lenti\n") for label in subset_summary["condition_label"]],
            fontsize=tick_fontsize,
        )
        ax.set_xlim(x_positions[0] - 0.36, x_positions[-1] + 0.36)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="both", labelsize=tick_fontsize, width=0.8, length=3, color="#202020")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.8)

    axes[0].set_ylabel(
        "Pairwise transcriptional\nPearson correlation\nwith naive cotransferred cells",
        fontsize=label_fontsize,
        labelpad=12,
    )
    fig.suptitle("D30 LCMV in vivo", fontsize=suptitle_fontsize, y=0.99)
    fig.subplots_adjust(left=0.25, bottom=0.25, right=0.98, top=0.82, wspace=0.38)
    fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.svg", format="svg", facecolor="white")
    plt.close(fig)


def plot_c_experimental_correlation_boxplots(
    pairwise_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    output_stem: str = "c",
) -> None:
    title_fontsize = 18
    tick_fontsize = 14
    label_fontsize = 16
    annotation_fontsize = 16
    point_size = 20
    jitter_width = 0.16

    summary_df = summary_df.sort_values(["panel_order", "comparison_order_within_panel"]).reset_index(drop=True)
    pairwise_df = pairwise_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=DPI, sharey=False)
    rng = np.random.default_rng(6)

    panel_specs = (
        summary_df[["panel_order", "panel_key", "panel_label"]]
        .drop_duplicates()
        .sort_values("panel_order")
    )
    for ax, panel in zip(axes, panel_specs.itertuples(index=False)):
        panel_summary = summary_df[summary_df["panel_key"] == panel.panel_key].sort_values(
            "comparison_order_within_panel"
        )
        panel_pairwise = pairwise_df[pairwise_df["panel_key"] == panel.panel_key]
        x_positions = np.arange(panel_summary.shape[0]) * 0.70
        comparison_to_x = dict(zip(panel_summary["comparison_label"], x_positions))
        plot_values = [
            panel_pairwise.loc[
                panel_pairwise["comparison_label"] == comparison_label,
                "pearson_correlation",
            ].to_numpy()
            for comparison_label in panel_summary["comparison_label"]
        ]
        panel_values = panel_pairwise["pearson_correlation"].to_numpy()
        panel_range = panel_values.max() - panel_values.min()
        panel_padding = max(0.005, panel_range * 0.35)
        y_min = max(0.0, np.floor((panel_values.min() - panel_padding) * 1000.0) / 1000.0)
        y_max = min(1.0, np.ceil((panel_values.max() + panel_padding) * 1000.0) / 1000.0)
        if y_max - y_min < 0.02:
            midpoint = (y_min + y_max) / 2
            y_min = max(0.0, midpoint - 0.01)
            y_max = min(1.0, midpoint + 0.01)
        y_ticks = np.linspace(y_min, y_max, 5)
        y_range = y_max - y_min
        annotation_y = y_max - 0.12 * y_range
        annotation_tick = 0.025 * y_range
        annotation_text_y = annotation_y + annotation_tick + 0.012 * y_range

        boxplot = ax.boxplot(
            plot_values,
            positions=x_positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.2},
            whiskerprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            capprops={"color": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            boxprops={"edgecolor": FIGURE7B_NEUTRAL_AXIS_COLOR, "linewidth": 1.0},
            zorder=2,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(FIGURE7B_NEUTRAL_BOX_FACE)
            patch.set_alpha(1.0)

        for comparison_label, group_df in panel_pairwise.groupby("comparison_label", sort=False):
            x = comparison_to_x[comparison_label]
            jitter = rng.uniform(-jitter_width / 2, jitter_width / 2, size=group_df.shape[0])
            ax.scatter(
                np.full(group_df.shape[0], x) + jitter,
                group_df["pearson_correlation"],
                s=point_size,
                color=FIGURE7B_NEUTRAL_AXIS_COLOR,
                alpha=0.88,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )

        stats_row = stats_df.loc[stats_df["panel_key"] == panel.panel_key]
        if stats_row.empty:
            raise ValueError(f"Missing significance statistics for c_experimental {panel.panel_label}")
        stats_row = stats_row.iloc[0]
        group_a_x = x_positions[0]
        group_b_x = x_positions[1]
        ax.plot(
            [group_a_x, group_a_x, group_b_x, group_b_x],
            [annotation_y, annotation_y + annotation_tick, annotation_y + annotation_tick, annotation_y],
            color=FIGURE7B_NEUTRAL_AXIS_COLOR,
            linewidth=1.0,
            clip_on=False,
            zorder=4,
        )
        ax.text(
            (group_a_x + group_b_x) / 2,
            annotation_text_y,
            str(stats_row["plot_label"]),
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            clip_on=False,
        )

        ax.set_title(panel.panel_label, fontsize=title_fontsize, pad=6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [label.replace("Lenti ", "Lenti\n") for label in panel_summary["x_label"]],
            fontsize=tick_fontsize,
        )
        ax.set_xlim(x_positions[0] - 0.36, x_positions[-1] + 0.36)
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="both", labelsize=tick_fontsize, width=0.8, length=3, color="#202020")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.8)

    axes[0].set_ylabel(
        "Pairwise transcriptional\nPearson correlation",
        fontsize=label_fontsize,
        labelpad=12,
    )
    fig.subplots_adjust(left=0.20, bottom=0.25, right=0.98, top=0.84, wspace=0.42)
    fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.svg", format="svg", facecolor="white")
    plt.close(fig)


def plot_memory_pca_panels(
    pca_df: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    output_stem: str = "a_experimental",
    group_order: list[str] | None = None,
) -> None:
    if group_order is None:
        group_order = EXPERIMENTAL_MEMORY_GROUPS

    panel_specs = [
        ("CM and EM", pca_df),
        ("CM", pca_df[pca_df["memory_subset"] == "CM"]),
        ("EM", pca_df[pca_df["memory_subset"] == "EM"]),
    ]
    x_padding = max(1.0, (pca_df["PC1"].max() - pca_df["PC1"].min()) * 0.06)
    y_padding = max(1.0, (pca_df["PC2"].max() - pca_df["PC2"].min()) * 0.08)
    xlim = (pca_df["PC1"].min() - x_padding, pca_df["PC1"].max() + x_padding)
    ylim = (pca_df["PC2"].min() - y_padding, pca_df["PC2"].max() + y_padding)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), dpi=DPI, sharex=True, sharey=True)
    point_size = 33
    title_fontsize = 12
    tick_fontsize = 9

    for ax, (title, panel_df) in zip(axes, panel_specs):
        for group in group_order:
            group_df = panel_df[panel_df["sample_group"] == group]
            if group_df.empty:
                continue

            input_condition = memory_input_condition(group)
            ax.scatter(
                group_df["PC1"],
                group_df["PC2"],
                s=point_size,
                color=GROUP_COLORS[group],
                marker=MEMORY_CONDITION_SHAPES[input_condition],
                edgecolor="black",
                linewidth=0.3,
                zorder=3,
            )

        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=tick_fontsize)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)", fontsize=12)
    axes[1].set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=12)

    group_handles = [
        Line2D(
            [0],
            [0],
            marker=MEMORY_CONDITION_SHAPES[memory_input_condition(group)],
            linestyle="None",
            color="black",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5.7,
            label=group,
        )
        for group in group_order
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=MEMORY_CONDITION_SHAPES[label],
            linestyle="None",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=5.7,
            label=MEMORY_CONDITION_LABELS[label],
        )
        for label in MEMORY_CONDITION_ORDER
    ]

    group_legend = axes[-1].legend(
        handles=group_handles,
        frameon=False,
        fontsize=7.2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        handletextpad=0.5,
    )
    axes[-1].add_artist(group_legend)
    shape_legend = axes[-1].legend(
        handles=shape_handles,
        frameon=False,
        fontsize=7.6,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.085, right=0.79, bottom=0.22, top=0.88, wspace=0.18)
    legend_artists = (group_legend, shape_legend)
    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=DPI,
        bbox_inches="tight",
        bbox_extra_artists=legend_artists,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{output_stem}.svg",
        format="svg",
        bbox_inches="tight",
        bbox_extra_artists=legend_artists,
        facecolor="white",
    )
    plt.close(fig)


def plot_heatmap(
    row_scaled: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    output_dir: Path,
    vmin: float,
    vmax: float,
    output_stem: str = "b",
    group_order: list[str] | None = None,
    figsize: tuple[float, float] = (6.1, 9.6),
    axes: dict[str, list[float]] | None = None,
    x_label_fontsize: float = 7,
    y_label_fontsize: float = 7,
    legend_fontsize: float = 8,
    colorbar_fontsize: float = 9,
) -> None:
    if group_order is None:
        group_order = MAIN_GROUPS
    if axes is None:
        axes = {
            "strip": [0.12, 0.865, 0.68, 0.025],
            "heatmap": [0.12, 0.25, 0.68, 0.61],
            "legend": [0.865, 0.61, 0.30, 0.25],
            "colorbar": [0.885, 0.35, 0.035, 0.20],
        }

    samples = sample_metadata["sample"].tolist()
    sample_labels = (
        sample_metadata["x_axis_label"].tolist()
        if "x_axis_label" in sample_metadata.columns
        else samples
    )
    groups = sample_metadata["sample_group"].tolist()
    group_to_index = {group: idx for idx, group in enumerate(group_order)}
    missing_groups = sorted(set(groups) - set(group_to_index))
    if missing_groups:
        raise ValueError(f"Heatmap group order is missing groups: {missing_groups}")
    group_indices = np.array([group_to_index[group] for group in groups], dtype=int)[None, :]

    fig = plt.figure(figsize=figsize, dpi=DPI, facecolor="white")
    strip_ax = fig.add_axes(axes["strip"])
    heatmap_ax = fig.add_axes(axes["heatmap"])
    legend_ax = fig.add_axes(axes["legend"])
    cbar_ax = fig.add_axes(axes["colorbar"])

    image = heatmap_ax.imshow(
        row_scaled.loc[:, samples].values,
        aspect="auto",
        interpolation="nearest",
        cmap=make_pheatmap_cmap(),
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    heatmap_ax.set_xticks(np.arange(len(samples)))
    heatmap_ax.set_xticklabels(sample_labels, rotation=90, fontsize=x_label_fontsize, ha="center", va="top")
    heatmap_ax.set_yticks(np.arange(row_scaled.shape[0]))
    heatmap_ax.set_yticklabels(row_scaled.index, fontsize=y_label_fontsize)
    heatmap_ax.yaxis.tick_right()
    heatmap_ax.tick_params(axis="both", length=0)
    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)

    group_boundaries = np.cumsum(sample_metadata["sample_group"].value_counts(sort=False).to_numpy())
    for boundary in group_boundaries[:-1]:
        heatmap_ax.axvline(boundary - 0.5, color="white", linewidth=1.0)
        strip_ax.axvline(boundary - 0.5, color="white", linewidth=1.0)

    strip_cmap = ListedColormap([GROUP_COLORS[group] for group in group_order])
    strip_ax.imshow(group_indices, aspect="auto", cmap=strip_cmap, vmin=0, vmax=len(group_order) - 1)
    strip_ax.set_xlim(-0.5, len(samples) - 0.5)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])
    for spine in strip_ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, cax=cbar_ax, ticks=[-2, -1, 0, 1, 2])
    cbar.ax.tick_params(labelsize=colorbar_fontsize, length=0)
    cbar.set_label("Normalized expression (row z-score)", fontsize=colorbar_fontsize)

    legend_ax.axis("off")
    handles = [
        Patch(facecolor=GROUP_COLORS[group], edgecolor="none", label=figure1c_group_display_label(group))
        for group in group_order
    ]
    legend_ax.legend(handles=handles, frameon=False, fontsize=legend_fontsize, loc="upper left", borderaxespad=0)

    fig.savefig(output_dir / f"{output_stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalized, row_scaled, sample_metadata = load_figure1c_sample_matrix(
        args.normalized_counts,
        args.metadata,
    )
    c_experimental_pairwise_df, c_experimental_summary_df = compute_c_experimental_pairwise_correlations(
        args.normalized_counts,
        sample_metadata,
    )
    c_experimental_stats_df = compute_c_experimental_stats(c_experimental_pairwise_df)
    experimental_normalized, experimental_row_scaled, experimental_sample_metadata = load_all_sample_matrix(
        args.normalized_counts,
        args.metadata,
    )
    memory_expression, memory_sample_metadata = load_memory_expression_and_metadata(
        args.normalized_counts,
        args.metadata,
    )
    memory_correlation_results = compute_memory_sample_correlations(memory_expression, memory_sample_metadata)
    b_experimental_v5_pairwise_df, b_experimental_v5_summary_df = compute_b_experimental_v5_pairwise_correlations(
        memory_correlation_results,
        memory_sample_metadata,
    )
    b_experimental_v5_stats_df = compute_b_experimental_v5_naive_activated_stats(b_experimental_v5_pairwise_df)
    b_experiment_v3_outputs: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}
    b_experiment_v4_outputs: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}
    for subset_key, subset_label, subset_value in B_EXPERIMENT_SUBSET_PANELS:
        subset_metadata = subset_memory_sample_metadata(memory_sample_metadata, subset_value)
        v3_pairwise_df, v3_summary_df = compute_b_experiment_v3_pairwise_correlations(
            memory_correlation_results,
            subset_metadata,
        )
        v3_pairwise_df, v3_summary_df = tag_subset_tables(
            v3_pairwise_df,
            v3_summary_df,
            subset_key,
            subset_label,
        )
        b_experiment_v3_outputs[subset_key] = (subset_label, v3_pairwise_df, v3_summary_df)

        v4_pairwise_df, v4_summary_df = compute_b_experiment_v4_pairwise_correlations(
            memory_correlation_results,
            subset_metadata,
        )
        v4_pairwise_df, v4_summary_df = tag_subset_tables(
            v4_pairwise_df,
            v4_summary_df,
            subset_key,
            subset_label,
        )
        b_experiment_v4_outputs[subset_key] = (subset_label, v4_pairwise_df, v4_summary_df)
    pca_sample_metadata = build_pca_sample_metadata(args.metadata, PCA_GROUPS)
    pca_df, explained = load_deseq2_pca_coordinates(args.pca_coordinates, pca_sample_metadata)
    memory_pca_sample_metadata = build_pca_sample_metadata(args.metadata, EXPERIMENTAL_MEMORY_GROUPS)
    memory_pca_df, memory_explained = load_deseq2_pca_coordinates(
        args.memory_pca_coordinates,
        memory_pca_sample_metadata,
    )
    memory_pca_df = memory_pca_df.copy()
    memory_pca_df["input_condition"] = memory_pca_df["sample_group"].map(memory_input_condition)
    memory_pca_df["memory_subset"] = memory_pca_df["sample_group"].map(memory_subset)

    plot_pca(pca_df, explained, args.output_dir)
    plot_memory_pca_panels(
        memory_pca_df,
        memory_explained,
        args.output_dir,
        group_order=EXPERIMENTAL_MEMORY_GROUPS,
    )
    plot_memory_correlation_clustermaps(memory_correlation_results, memory_sample_metadata, args.output_dir)
    for subset_key, (subset_label, pairwise_df, summary_df) in b_experiment_v3_outputs.items():
        plot_b_experiment_v3_correlation_boxplots(
            pairwise_df,
            summary_df,
            args.output_dir,
            output_stem=b_experiment_output_stem("b_experiment_v3", subset_key),
            subset_label=subset_label,
        )
    for subset_key, (subset_label, pairwise_df, summary_df) in b_experiment_v4_outputs.items():
        plot_b_experiment_v4_correlation_boxplots(
            pairwise_df,
            summary_df,
            args.output_dir,
            output_stem=b_experiment_output_stem("b_experiment_v4", subset_key),
            subset_label=subset_label,
        )
    plot_b_experimental_v5_correlation_boxplots(
        b_experimental_v5_pairwise_df,
        b_experimental_v5_summary_df,
        b_experimental_v5_stats_df,
        args.output_dir,
    )
    plot_c_experimental_correlation_boxplots(
        c_experimental_pairwise_df,
        c_experimental_summary_df,
        c_experimental_stats_df,
        args.output_dir,
    )
    plot_heatmap(row_scaled, sample_metadata, args.output_dir, args.vmin, args.vmax)
    plot_heatmap(
        experimental_row_scaled,
        experimental_sample_metadata,
        args.output_dir,
        args.vmin,
        args.vmax,
        output_stem="b_experimental",
        group_order=ALL_SAMPLE_GROUPS,
        figsize=(13.2, 9.8),
        axes={
            "strip": [0.055, 0.86, 0.74, 0.025],
            "heatmap": [0.055, 0.24, 0.74, 0.615],
            "legend": [0.84, 0.56, 0.16, 0.33],
            "colorbar": [0.87, 0.29, 0.020, 0.20],
        },
        x_label_fontsize=4.5,
        y_label_fontsize=6.2,
        legend_fontsize=7.0,
        colorbar_fontsize=8.0,
    )

    pca_df.to_csv(args.output_dir / "a_pca_coordinates.tsv", sep="\t", index=False)
    memory_pca_df.to_csv(args.output_dir / "a_experimental_pca_coordinates.tsv", sep="\t", index=False)
    save_memory_correlation_outputs(memory_correlation_results, memory_sample_metadata, args.output_dir)
    for subset_key, (_, pairwise_df, summary_df) in b_experiment_v3_outputs.items():
        output_stem = b_experiment_output_stem("b_experiment_v3", subset_key)
        pairwise_df.to_csv(
            args.output_dir / f"{output_stem}_pairwise_correlations.tsv",
            sep="\t",
            index=False,
        )
        summary_df.to_csv(
            args.output_dir / f"{output_stem}_summary.tsv",
            sep="\t",
            index=False,
        )
    for subset_key, (_, pairwise_df, summary_df) in b_experiment_v4_outputs.items():
        output_stem = b_experiment_output_stem("b_experiment_v4", subset_key)
        pairwise_df.to_csv(
            args.output_dir / f"{output_stem}_pairwise_correlations.tsv",
            sep="\t",
            index=False,
        )
        summary_df.to_csv(
            args.output_dir / f"{output_stem}_summary.tsv",
            sep="\t",
            index=False,
        )
    b_experimental_v5_pairwise_df.to_csv(
        args.output_dir / "d_pairwise_correlations.tsv",
        sep="\t",
        index=False,
    )
    b_experimental_v5_summary_df.to_csv(
        args.output_dir / "d_summary.tsv",
        sep="\t",
        index=False,
    )
    b_experimental_v5_stats_df.to_csv(
        args.output_dir / "d_stats.tsv",
        sep="\t",
        index=False,
    )
    c_experimental_pairwise_df.to_csv(
        args.output_dir / "c_pairwise_correlations.tsv",
        sep="\t",
        index=False,
    )
    c_experimental_summary_df.to_csv(
        args.output_dir / "c_summary.tsv",
        sep="\t",
        index=False,
    )
    c_experimental_stats_df.to_csv(
        args.output_dir / "c_stats.tsv",
        sep="\t",
        index=False,
    )
    sample_metadata.to_csv(args.output_dir / "b_sample_metadata.tsv", sep="\t", index=False)
    normalized.to_csv(args.output_dir / "b_normalized_expression.tsv", sep="\t")
    row_scaled.to_csv(args.output_dir / "b_row_scaled_expression.tsv", sep="\t")
    experimental_sample_metadata.to_csv(args.output_dir / "b_experimental_sample_metadata.tsv", sep="\t", index=False)
    experimental_normalized.to_csv(args.output_dir / "b_experimental_normalized_expression.tsv", sep="\t")
    experimental_row_scaled.to_csv(args.output_dir / "b_experimental_row_scaled_expression.tsv", sep="\t")

    report_file_sizes(
        [
            args.output_dir / "a.png",
            args.output_dir / "a.svg",
            args.output_dir / "a_experimental.png",
            args.output_dir / "a_experimental.svg",
            args.output_dir / "a_experimental_v2_left.png",
            args.output_dir / "a_experimental_v2_left.svg",
            args.output_dir / "a_experimental_v2_middle.png",
            args.output_dir / "a_experimental_v2_middle.svg",
            args.output_dir / "a_experimental_v2_right.png",
            args.output_dir / "a_experimental_v2_right.svg",
            args.output_dir / "b_experiment_v3.png",
            args.output_dir / "b_experiment_v3.svg",
            args.output_dir / "b_experiment_v3_cm.png",
            args.output_dir / "b_experiment_v3_cm.svg",
            args.output_dir / "b_experiment_v3_em.png",
            args.output_dir / "b_experiment_v3_em.svg",
            args.output_dir / "b_experiment_v4.png",
            args.output_dir / "b_experiment_v4.svg",
            args.output_dir / "b_experiment_v4_cm.png",
            args.output_dir / "b_experiment_v4_cm.svg",
            args.output_dir / "b_experiment_v4_em.png",
            args.output_dir / "b_experiment_v4_em.svg",
            args.output_dir / "d.png",
            args.output_dir / "d.svg",
            args.output_dir / "c.png",
            args.output_dir / "c.svg",
            args.output_dir / "b.png",
            args.output_dir / "b.svg",
            args.output_dir / "b_experimental.png",
            args.output_dir / "b_experimental.svg",
        ]
    )
    print(f"Saved PCA coordinates for {pca_df.shape[0]} samples")
    pca_ntop = pca_df["pca_ntop"].iloc[0] if "pca_ntop" in pca_df.columns else "unknown"
    print(f"Rendered PCA coordinates for {pca_df.shape[0]} samples with ntop={pca_ntop}")
    memory_pca_ntop = memory_pca_df["pca_ntop"].iloc[0] if "pca_ntop" in memory_pca_df.columns else "unknown"
    print(f"Rendered memory PCA coordinates for {memory_pca_df.shape[0]} samples with ntop={memory_pca_ntop}")
    print(f"Saved heatmap matrix with {row_scaled.shape[0]} genes x {row_scaled.shape[1]} samples")
    print(
        "Saved experimental heatmap matrix with "
        f"{experimental_row_scaled.shape[0]} genes x {experimental_row_scaled.shape[1]} samples"
    )


if __name__ == "__main__":
    main()
