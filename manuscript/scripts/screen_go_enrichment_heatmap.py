#!/usr/bin/env python
"""Generate the GO enrichment heatmap for Figure S2b."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["figure.figsize"] = (8, 12)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


DATA_FILES = [
    "data/screen/pathway_results/GO_enriched_D7tcf7hi-low_100_21-03-01-18-00.txt.csv",
    "data/screen/pathway_results/GO_enriched_D7conttcf7hi-low_100_21-03-01-18-04.txt.csv",
    "data/screen/pathway_results/GO_enriched_D5contprdm1hi-low_100_21-03-01-18-17.txt.csv",
    "data/screen/pathway_results/GO_enriched_D7prdm1hi-low_100_21-03-01-18-03.txt.csv",
    "data/screen/pathway_results/GO_enriched_D7eomeshi-low_100_21-03-01-18-01.txt.csv",
    "data/screen/pathway_results/GO_enriched_D5hyptoxhi-low_100_21-03-01-18-19.txt.csv",
    "data/screen/pathway_results/GO_depleted_D7tcf7hi-low_100_21-03-01-18-00.txt.csv",
    "data/screen/pathway_results/GO_depleted_D7conttcf7hi-low_100_21-03-01-18-04.txt.csv",
    "data/screen/pathway_results/GO_depleted_D5contprdm1hi-low_100_21-03-01-18-17.txt.csv",
    "data/screen/pathway_results/GO_depleted_D7prdm1hi-low_100_21-03-01-18-03.txt.csv",
    "data/screen/pathway_results/GO_depleted_D7eomeshi-low_100_21-03-01-18-01.txt.csv",
    "data/screen/pathway_results/GO_depleted_D5hyptoxhi-low_100_21-03-01-18-19.txt.csv",
]

CONDITION_NAMES = [
    "D7 Tcf7 low vs high acute stim",
    "D7 Tcf7 low vs high chronic stim",
    "D5 Prdm1 low vs high chronic stim",
    "D7 Prdm1 low vs high",
    "D7 Eomes low vs high",
    "D5 Tox low vs high hypoxic",
]

TOP_N = 5
PVALUE_THRESHOLD = 0.001
MIN_OVERLAP = 5
MIN_REPORTED_OVERLAP = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manuscript_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_and_process(file_path: Path, name: str, category: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(file_path)
    too_few_overlaps = pd.to_numeric(df["overlap"], errors="coerce").fillna(0) < MIN_REPORTED_OVERLAP
    df.loc[too_few_overlaps, "log2_odds_ratio"] = np.nan
    df_sig = df.query("pvalue < @PVALUE_THRESHOLD and overlap >= @MIN_OVERLAP")
    chosen_paths = (
        df_sig.sort_values("log2_odds_ratio", ascending=False)
        .head(TOP_N)["GO_name"]
        .tolist()
    )
    df["Condition"] = name
    df["Type"] = category
    return df, chosen_paths


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_terms: list[str] = []
    frames: list[pd.DataFrame] = []

    for enriched_rel, depleted_rel, name in zip(
        DATA_FILES[:6], DATA_FILES[6:], CONDITION_NAMES
    ):
        enriched_df, enriched_paths = load_and_process(
            repo_root() / enriched_rel, name, "enriched"
        )
        depleted_df, depleted_paths = load_and_process(
            repo_root() / depleted_rel, name, "depleted"
        )
        frames.extend([enriched_df, depleted_df])
        selected_terms.extend(enriched_paths)
        selected_terms.extend(depleted_paths)

    selected_terms = list(dict.fromkeys(selected_terms))
    merged_df = pd.concat(frames, ignore_index=True)
    filtered_df = merged_df[merged_df["GO_name"].isin(selected_terms)].copy()

    ordered_columns = pd.MultiIndex.from_product(
        [CONDITION_NAMES, ["enriched", "depleted"]]
    )

    heatmap_data = filtered_df.pivot_table(
        index="GO_name",
        columns=["Condition", "Type"],
        values="log2_odds_ratio",
        aggfunc="first",
    ).reindex(index=selected_terms, columns=ordered_columns)

    pvalue_data = filtered_df.pivot_table(
        index="GO_name",
        columns=["Condition", "Type"],
        values="pvalue",
        aggfunc="first",
    ).reindex(index=selected_terms, columns=ordered_columns)

    return filtered_df, heatmap_data, pvalue_data


def add_annotations(ax: plt.Axes, heatmap_data: pd.DataFrame, pvalue_data: pd.DataFrame) -> None:
    for y, pathway in enumerate(heatmap_data.index):
        for x, column in enumerate(heatmap_data.columns):
            val = heatmap_data.iloc[y, x]
            pval = pvalue_data.iloc[y, x]

            if pd.isna(val) or pd.isna(pval):
                continue

            if pval < 0.00001:
                label = f"{val:.2f}\n**"
                color = "black"
            elif pval < 0.001:
                label = f"{val:.2f}\n*"
                color = "black"
            elif pval < 0.05:
                label = f"{val:.2f}"
                color = "black"
            else:
                ax.add_patch(
                    plt.Rectangle((x, y), 1, 1, fill=True, color="white", ec="white", zorder=3)
                )
                label = f"{val:.2f}"
                color = "white"

            ax.text(
                x + 0.5,
                y + 0.5,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=5.5,
                zorder=4,
            )


def position_colorbar_next_to_y_labels(
    g: sns.matrix.ClusterGrid,
    pad: float = 0.012,
    width: float = 0.025,
    height_fraction: float = 0.5,
) -> None:
    fig = g.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = [
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for label in g.ax_heatmap.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    if label_boxes:
        left = max(box.x1 for box in label_boxes) + pad
    else:
        left = g.ax_heatmap.get_position().x1 + pad
    heatmap_box = g.ax_heatmap.get_position()
    height = heatmap_box.height * height_fraction
    bottom = heatmap_box.y0 + (heatmap_box.height - height) / 2
    g.ax_cbar.set_position([left, bottom, width, height])


def make_plot(heatmap_data: pd.DataFrame, pvalue_data: pd.DataFrame) -> sns.matrix.ClusterGrid:
    plot_data = heatmap_data.fillna(0)
    cmap = sns.color_palette("Reds", as_cmap=True)

    g = sns.clustermap(
        plot_data,
        cmap=cmap,
        row_cluster=True,
        col_cluster=False,
        linewidths=0.5,
        linecolor="white",
        figsize=(8, 12),
        dendrogram_ratio=(0.12, 0.02),
        cbar_pos=(1.16, 0.2, 0.025, 0.55),
        xticklabels=True,
        yticklabels=True,
    )

    row_order = g.dendrogram_row.reordered_ind
    ordered_heatmap = heatmap_data.iloc[row_order]
    ordered_pvalues = pvalue_data.iloc[row_order]

    g.ax_heatmap.clear()
    sns.heatmap(
        ordered_heatmap,
        ax=g.ax_heatmap,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        mask=ordered_heatmap.isna(),
        vmin=np.nanmin(heatmap_data.values),
        vmax=np.nanmax(heatmap_data.values),
    )

    add_annotations(g.ax_heatmap, ordered_heatmap, ordered_pvalues)

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", rotation=90, labelsize=8)
    g.ax_heatmap.tick_params(axis="y", rotation=0, labelsize=8)

    position_colorbar_next_to_y_labels(g)
    g.ax_cbar.set_ylabel("log2(odds_ratio)", fontsize=12)
    g.ax_cbar.tick_params(labelsize=9)

    return g


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=manuscript_root() / "plots" / "Sup_Figure_2",
    )
    parser.add_argument("--panel-stem", default="b")
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=MIN_REPORTED_OVERLAP,
        help="Minimum selected-gene/pathway overlap required to report a log2 odds ratio.",
    )
    return parser.parse_args()


def save_outputs(fig: plt.Figure, output_dir: Path, panel_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{panel_stem}.png"
    svg_path = output_dir / f"{panel_stem}.svg"

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


def main() -> None:
    args = parse_args()
    global MIN_REPORTED_OVERLAP
    MIN_REPORTED_OVERLAP = args.min_overlap
    _, heatmap_data, pvalue_data = prepare_inputs()
    grid = make_plot(heatmap_data, pvalue_data)
    save_outputs(grid.fig, args.output_dir, args.panel_stem)
    plt.close(grid.fig)


if __name__ == "__main__":
    main()
