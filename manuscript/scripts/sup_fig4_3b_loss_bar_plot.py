#!/usr/bin/env python
"""Generate Figure S9b: per-perturbation loss bar plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.grid"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATIONS_FILE = REPO_ROOT / "scripts" / "clonal_modeling" / "real_data" / "gene10" / "input" / "perturbations.txt"
CLONAL_DIR = REPO_ROOT / "clonal_analysis" / "intermediate_files" / "gene10"
REAL_DATA_DIR = REPO_ROOT / "scripts" / "clonal_modeling" / "real_data" / "gene10" / "input"
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_9"

import sys

UTILS_PATH = REPO_ROOT / "scripts" / "utils"
if str(UTILS_PATH) not in sys.path:
    sys.path.append(str(UTILS_PATH))

from figure_format import perturbation_colors  # type: ignore  # noqa: E402


def load_perturbations() -> list[str]:
    with open(PERTURBATIONS_FILE, "r", encoding="utf-8") as handle:
        perturbations = [line.strip() for line in handle if line.strip()]
    non_nt = sorted(perturbation for perturbation in perturbations if perturbation != "NT")
    return ["NT", *non_nt]


def load_loss(perturbation: str) -> float:
    path = CLONAL_DIR / perturbation / f"lowest_loss_points_best_{perturbation}_parameters.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter summary: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("loss ="):
                return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find loss in {path}")


def load_num_clones(perturbation: str) -> int:
    path = REAL_DATA_DIR / f"{perturbation}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing real-data table: {path}")
    df = pd.read_csv(path)
    return int(df.shape[0])


def build_table(perturbations: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation": perturbations,
            "loss": [load_loss(perturbation) for perturbation in perturbations],
            "num_clones": [load_num_clones(perturbation) for perturbation in perturbations],
        }
    )


def save_plot(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "b.png"
    svg_path = OUTPUT_DIR / "b.svg"

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = [perturbation_colors.get(perturbation, "#808080") for perturbation in df["perturbation"]]
    ax.bar(df["perturbation"], df["loss"], color=colors, edgecolor="black", linewidth=0.6)
    tick_labels = [
        f"{perturbation} (n={num_clones})"
        for perturbation, num_clones in zip(df["perturbation"], df["num_clones"])
    ]

    ax.set_ylabel("Loss")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(tick_labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def main() -> None:
    perturbations = load_perturbations()
    df = build_table(perturbations)
    save_plot(df)


if __name__ == "__main__":
    main()
