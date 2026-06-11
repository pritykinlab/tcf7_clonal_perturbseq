#!/usr/bin/env python
"""Generate the reference-only screen correlation heatmap for Figure S1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


SCREEN_FILES = [
    {
        "label": "D5 Tcf7 std1",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D51tcf7hi-low_100_21-03-01-18-13.txt",
    },
    {
        "label": "D5 Tcf7 std2",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D52tcf7hi-low_100_21-03-01-18-14.txt",
    },
    {
        "label": "D7 Tcf7 acute",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7tcf7hi-low_100_21-03-01-18-00.txt",
    },
    {
        "label": "D7 Tcf7 chronic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7conttcf7hi-low_100_21-03-01-18-04.txt",
    },
    {
        "label": "D5 Prdm1 chronic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D5contprdm1hi-low_100_21-03-01-18-17.txt",
    },
    {
        "label": "D7 Prdm1",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7prdm1hi-low_100_21-03-01-18-03.txt",
    },
    {
        "label": "D7 Eomes",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D7eomeshi-low_100_21-03-01-18-01.txt",
    },
    {
        "label": "D5 Tox hypoxic",
        "path": "raw_data/CRISPR_sort_screens/2_26_2021_BRIE_run1/hypergeometric/D5hyptoxhi-low_100_21-03-01-18-19.txt",
    },
]

CONTROL_PREFIXES = ("NO_SITE", "INACTIVE", "POTENTIALLY_ACTIVE")
REQUIRED_GUIDES_PER_TARGET_GENE = 4
PLOT_STEM = "e_old"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manuscript_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_target_gene(series: pd.Series) -> pd.Series:
    return (
        ~series.str.contains("INTERGENIC", na=False)
        & ~series.str.startswith(CONTROL_PREFIXES, na=False)
    )


def load_screen(path: Path, label: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t").rename(
        columns={
            "Gene Symbol": "gene",
            "Average LFC": "average_lfc",
            "Number of perturbations": "number_of_perturbations",
        }
    )
    required = ["gene", "average_lfc", "number_of_perturbations"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df[required].copy()
    df["average_lfc"] = pd.to_numeric(df["average_lfc"], errors="coerce")
    df["number_of_perturbations"] = pd.to_numeric(
        df["number_of_perturbations"], errors="coerce"
    )
    df = df[df["average_lfc"].notna()]
    df = df[is_target_gene(df["gene"])]
    df = df[df["number_of_perturbations"] == REQUIRED_GUIDES_PER_TARGET_GENE]
    df = df.drop_duplicates("gene", keep="first")
    return df.set_index("gene")["average_lfc"].rename(label)


def prepare_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lfc_matrix = pd.concat(
        [
            load_screen(repo_root() / spec["path"], spec["label"])
            for spec in SCREEN_FILES
        ],
        axis=1,
        join="inner",
    )
    lfc_matrix = lfc_matrix.sort_index()
    correlation = lfc_matrix.corr(method="pearson")
    distance = 1 - correlation
    return lfc_matrix, correlation, distance


def make_plot(correlation: pd.DataFrame, distance: pd.DataFrame) -> sns.matrix.ClusterGrid:
    condensed_distance = squareform(distance.values, checks=False)
    screen_linkage = linkage(condensed_distance, method="average")

    grid = sns.clustermap(
        correlation,
        row_linkage=screen_linkage,
        col_linkage=screen_linkage,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        figsize=(6.2, 5.0),
        dendrogram_ratio=(0.18, 0.18),
        cbar_pos=(0.04, 0.86, 0.018, 0.10),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="x", rotation=90, labelsize=8, length=0)
    grid.ax_heatmap.tick_params(axis="y", rotation=0, labelsize=8, length=0)
    grid.ax_cbar.set_ylabel("Pearson r", fontsize=8, labelpad=4)
    grid.ax_cbar.tick_params(labelsize=8)
    return grid


def save_outputs(
    fig: plt.Figure,
    lfc_matrix: pd.DataFrame,
    correlation: pd.DataFrame,
    distance: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{PLOT_STEM}.png"
    svg_path = output_dir / f"{PLOT_STEM}.svg"
    lfc_path = output_dir / f"{PLOT_STEM}_screen_lfc_matrix.tsv"
    correlation_path = output_dir / f"{PLOT_STEM}_correlation_matrix.tsv"
    distance_path = output_dir / f"{PLOT_STEM}_distance_matrix.tsv"

    lfc_matrix.to_csv(lfc_path, sep="\t", float_format="%.6g")
    correlation.to_csv(correlation_path, sep="\t", float_format="%.6g")
    distance.to_csv(distance_path, sep="\t", float_format="%.6g")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")
    print(f"Genes used for screen correlations: {lfc_matrix.shape[0]}")


def main() -> None:
    lfc_matrix, correlation, distance = prepare_matrices()
    grid = make_plot(correlation, distance)
    save_outputs(
        grid.fig,
        lfc_matrix,
        correlation,
        distance,
        manuscript_root() / "plots" / "Sup_Figure_1",
    )
    plt.close(grid.fig)


if __name__ == "__main__":
    main()
