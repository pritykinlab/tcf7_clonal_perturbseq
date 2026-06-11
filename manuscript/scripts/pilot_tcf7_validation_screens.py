#!/usr/bin/env python
"""
Recreate Figure 1e pilot in vitro Tcf7 validation screen volcano plots.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = (
    REPO_ROOT
    / "raw_data"
    / "CRISPR_sort_screens"
    / "7_8_2020_CP1473_invitro_Tcf7screens"
    / "hypergeometric"
)
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"

P_THRESHOLD = 0.05
NEG_LOG10_P_THRESHOLD = 1.30103

TARGET_HIGH_COLOR = "#c83f4f"
TARGET_LOW_COLOR = "#2f6fa7"
NON_TARGETING_COLOR = "#1a9850"
INTERGENIC_COLOR = "#6a5acd"
NONSIGNIFICANT_COLOR = "#d5d7da"
TEXT_SCALE = 4.0
POINT_DIAMETER_SCALE = 2.0
LINE_SCALE = 2.0
DIRECTION_ARROW_Y = 0.965
DIRECTION_LABEL_Y = 0.995

SCREENS = [
    {
        "stem": "e_top",
        "path": INPUT_DIR / "Tcf7_Ab_(D_C)-(A_B)_100_20-07-11-15-03.txt",
        "title": "In vitro Tcf7 validation screen: intracellular antibody",
        "xlim": (-5.8, 2.6),
        "ylim": (0.0, 4.0),
        "label_sides": {
            "Tcf7": "right",
            "Myb": "right",
            "Foxo1": "right",
            "Ets1": "right",
            "Hmgb2": "right",
            "Hmgb1": "right",
            "Rps6": "left",
            "Rad21": "left",
            "Hif1a": "left",
            "Eomes": "left",
            "Prdm1": "left",
        },
    },
    {
        "stem": "e_bottom",
        "path": INPUT_DIR / "Tcf7_GFP_(D_C)-(A_B)_100_20-07-11-15-03.txt",
        "title": "In vitro Tcf7 validation screen: Tcf7-GFP reporter",
        "xlim": (-6.2, 3.5),
        "ylim": (0.0, 2.5),
        "label_sides": {
            "Rps6": "right",
            "Myb": "right",
            "Foxp1": "left",
            "Foxo1": "right",
            "Yy1": "right",
            "Eef2": "right",
            "Bhlhe40": "left",
            "Ezh2": "left",
            "Eomes": "left",
            "Klf2": "left",
            "Prdm1": "left",
        },
    },
]


def load_screen(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["Gene Symbol", "Average LFC", "Average -log(p-values)", "Number of perturbations"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    df = df.copy()
    df["x"] = df["Average LFC"]
    df["y"] = df["Average -log(p-values)"]
    df["category"] = df["Gene Symbol"].map(categorize_gene)
    df["is_significant_target"] = (
        (df["category"] == "target") & (df["y"] >= NEG_LOG10_P_THRESHOLD)
    )
    df["direction"] = "nonsignificant"
    df.loc[df["is_significant_target"] & (df["x"] > 0), "direction"] = "high"
    df.loc[df["is_significant_target"] & (df["x"] < 0), "direction"] = "low"
    return df


def categorize_gene(gene: str) -> str:
    if gene.startswith("NO_SITE"):
        return "non_targeting"
    if gene.startswith("ONE_NON-GENE_SITE"):
        return "intergenic"
    return "target"


def plot_screen(ax: plt.Axes, config: dict[str, object]) -> None:
    df = load_screen(config["path"])

    background = df[(df["category"] == "target") & (~df["is_significant_target"])]
    high = df[df["direction"] == "high"]
    low = df[df["direction"] == "low"]
    non_targeting = df[df["category"] == "non_targeting"]
    intergenic = df[df["category"] == "intergenic"]

    ax.scatter(
        background["x"],
        background["y"],
        s=scaled_marker_area(7),
        c=NONSIGNIFICANT_COLOR,
        edgecolors="none",
        alpha=0.72,
        zorder=1,
    )
    ax.scatter(
        high["x"],
        high["y"],
        s=scaled_marker_area(9),
        c=TARGET_HIGH_COLOR,
        edgecolors=TARGET_HIGH_COLOR,
        linewidths=0.4 * LINE_SCALE,
        zorder=3,
    )
    ax.scatter(
        low["x"],
        low["y"],
        s=scaled_marker_area(9),
        c=TARGET_LOW_COLOR,
        edgecolors=TARGET_LOW_COLOR,
        linewidths=0.4 * LINE_SCALE,
        zorder=3,
    )
    ax.scatter(
        non_targeting["x"],
        non_targeting["y"],
        s=scaled_marker_area(17),
        facecolors="white",
        edgecolors=NON_TARGETING_COLOR,
        linewidths=0.65 * LINE_SCALE,
        zorder=4,
    )
    ax.scatter(
        intergenic["x"],
        intergenic["y"],
        s=scaled_marker_area(18),
        marker="D",
        facecolors="white",
        edgecolors=INTERGENIC_COLOR,
        linewidths=0.65 * LINE_SCALE,
        zorder=4,
    )

    ax.axhline(
        NEG_LOG10_P_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=0.8 * LINE_SCALE,
        zorder=0,
    )

    label_sides = config["label_sides"]
    for gene, horizontal_alignment in label_sides.items():
        point = df[df["Gene Symbol"] == gene].iloc[0]
        ax.annotate(
            gene,
            xy=(point["x"], point["y"]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=5.0 * TEXT_SCALE,
            ha=horizontal_alignment,
            va="bottom",
            annotation_clip=False,
            zorder=5,
        )

    ax.set_title(config["title"], fontsize=8.4 * TEXT_SCALE, pad=8)
    ax.set_xlabel("log2 FC", fontsize=6.5 * TEXT_SCALE)
    ax.set_ylabel("-log10(p-value)", fontsize=6.5 * TEXT_SCALE)
    ax.set_xlim(*config["xlim"])
    ax.set_ylim(*config["ylim"])
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.45 * LINE_SCALE)
    ax.grid(axis="x", visible=False)
    ax.tick_params(
        axis="both",
        labelsize=6.0 * TEXT_SCALE,
        width=0.65 * LINE_SCALE,
        length=3 * LINE_SCALE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8 * LINE_SCALE)
    ax.spines["bottom"].set_linewidth(0.8 * LINE_SCALE)

    add_tcf7_direction_arrows(ax)


def scaled_marker_area(area: float) -> float:
    return area * POINT_DIAMETER_SCALE**2


def add_tcf7_direction_arrows(ax: plt.Axes) -> None:
    arrow_kwargs = {
        "arrowstyle": "-|>",
        "linewidth": 1.0 * LINE_SCALE,
        "mutation_scale": 9.0 * LINE_SCALE,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(0.055, DIRECTION_ARROW_Y),
        xytext=(0.315, DIRECTION_ARROW_Y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_kwargs, "color": TARGET_LOW_COLOR},
        annotation_clip=False,
        zorder=8,
    )
    ax.text(
        0.185,
        DIRECTION_LABEL_Y,
        "Tcf7 Low",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0 * TEXT_SCALE,
        color=TARGET_LOW_COLOR,
        clip_on=False,
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(0.945, DIRECTION_ARROW_Y),
        xytext=(0.685, DIRECTION_ARROW_Y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_kwargs, "color": TARGET_HIGH_COLOR},
        annotation_clip=False,
        zorder=8,
    )
    ax.text(
        0.815,
        DIRECTION_LABEL_Y,
        "Tcf7 High",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0 * TEXT_SCALE,
        color=TARGET_HIGH_COLOR,
        clip_on=False,
        zorder=8,
    )


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
    fig.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.05)
    strip_trailing_whitespace(svg_path)
    plt.close(fig)
    validate_file_sizes(png_path, svg_path)


def strip_trailing_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def save_split_panels() -> None:
    for screen_config in SCREENS:
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.83)
        plot_screen(ax, screen_config)
        save_figure(fig, screen_config["stem"])


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_split_panels()
    print("Figure 1e pilot validation screens completed successfully!")


if __name__ == "__main__":
    main()
