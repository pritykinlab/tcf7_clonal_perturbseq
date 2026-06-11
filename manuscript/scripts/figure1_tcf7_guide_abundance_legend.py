#!/usr/bin/env python
"""
Generate standalone legend for Figure 1h Tcf7 guide abundance plots.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


NON_GENE_TEXT_SCALE = 1.5

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["legend.fontsize"] = 16 * NON_GENE_TEXT_SCALE
plt.rcParams["legend.title_fontsize"] = 16 * NON_GENE_TEXT_SCALE


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"
STEM = "h_legend"
CONTROL_COLOR = "#9aa0a6"
KO_COLOR = "#d62728"


def validate_file_sizes(png_path: Path, svg_path: Path) -> None:
    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=CONTROL_COLOR,
            markeredgecolor=CONTROL_COLOR,
            markersize=9,
            label="NT",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=KO_COLOR,
            markeredgecolor=KO_COLOR,
            markersize=9,
            label="Gene target",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        edgecolor="dimgray",
        fancybox=False,
    )

    png_path = OUTPUT_DIR / f"{STEM}.png"
    svg_path = OUTPUT_DIR / f"{STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    validate_file_sizes(png_path, svg_path)
    print("Figure 1h legend completed successfully!")


if __name__ == "__main__":
    main()
