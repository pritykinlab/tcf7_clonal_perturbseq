#!/usr/bin/env python
"""
Generate standalone legend for Figure 1e-f in vitro Tcf7 screen volcano plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"  # Use regular font for math text


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Figure1"
STEM = "1ef_legend"

NON_TARGETING_COLOR = "#1a9850"
INTERGENIC_COLOR = "#6a5acd"


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=NON_TARGETING_COLOR,
            markeredgewidth=1.6,
            markersize=8.5,
            label="NO-SITE",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=INTERGENIC_COLOR,
            markeredgewidth=1.6,
            markersize=8.5,
            label="INTERGENIC",
        ),
    ]


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

    fig, ax = plt.subplots(figsize=(5.0, 0.8))
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.legend(
        handles=legend_handles(),
        loc="center",
        ncol=2,
        fontsize=18,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="#bdbdbd",
        framealpha=1,
        borderpad=0.45,
        handletextpad=0.55,
        labelspacing=0.35,
        columnspacing=1.4,
    )

    png_path = OUTPUT_DIR / f"{STEM}.png"
    svg_path = OUTPUT_DIR / f"{STEM}.svg"
    fig.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.05)
    strip_trailing_whitespace(svg_path)
    plt.close(fig)

    validate_file_sizes(png_path, svg_path)
    print("Figure 1e-f legend completed successfully!")


if __name__ == "__main__":
    main()
