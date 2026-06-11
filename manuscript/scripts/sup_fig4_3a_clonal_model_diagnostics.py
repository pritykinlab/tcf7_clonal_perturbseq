#!/usr/bin/env python
"""
Assemble Figure S9a from per-perturbation clonal-model diagnostics.

Rows correspond to modeled perturbations in the gene10 dataset, excluding NT.
Within each row, the ECDF and isotonic diagnostic panels are cropped and
concatenated directly so there is no whitespace between columns. A single title
centered above the pair reports the perturbation and final loss.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"  # Make SVG text editable in Adobe Illustrator
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATIONS_FILE = REPO_ROOT / "scripts" / "clonal_modeling" / "real_data" / "gene10" / "input" / "perturbations.txt"
SOURCE_DIR = REPO_ROOT / "clonal_analysis" / "intermediate_files" / "gene10"
OUTPUT_DIR = REPO_ROOT / "manuscript" / "plots" / "Sup_Figure_9"
OUTPUT_STEM = "a"
WHITE_THRESHOLD = 0.995
TRIM_PADDING = 1
TITLE_PAD = 2


def load_perturbations() -> list[str]:
    with open(PERTURBATIONS_FILE, "r", encoding="utf-8") as handle:
        perturbations = [line.strip() for line in handle if line.strip()]
    return [perturbation for perturbation in perturbations if perturbation != "NT"]


def diagnostic_paths(perturbation: str) -> tuple[Path, Path]:
    base = SOURCE_DIR / perturbation
    ecdf = base / f"lowest_loss_points_best_{perturbation}_ecdf.png"
    isotonic = base / f"lowest_loss_points_best_{perturbation}_isotonic.png"
    return ecdf, isotonic


def parameter_path(perturbation: str) -> Path:
    return SOURCE_DIR / perturbation / f"lowest_loss_points_best_{perturbation}_parameters.txt"


def validate_inputs(perturbations: list[str]) -> None:
    missing: list[str] = []
    for perturbation in perturbations:
        for path in (*diagnostic_paths(perturbation), parameter_path(perturbation)):
            if not path.exists():
                missing.append(str(path))

    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(
            "Missing source diagnostic panels for Figure S9a:\n"
            f"{missing_str}"
        )


def crop_whitespace(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        mask = image < WHITE_THRESHOLD
    else:
        rgb = image[..., :3]
        mask = np.any(rgb < WHITE_THRESHOLD, axis=-1)
        if image.shape[-1] == 4:
            mask |= image[..., 3] > 0

    if not np.any(mask):
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    row_start = max(int(rows[0]) - TRIM_PADDING, 0)
    row_stop = min(int(rows[-1]) + TRIM_PADDING + 1, image.shape[0])
    col_start = max(int(cols[0]) - TRIM_PADDING, 0)
    col_stop = min(int(cols[-1]) + TRIM_PADDING + 1, image.shape[1])
    return image[row_start:row_stop, col_start:col_stop]


def load_image(path: Path) -> np.ndarray:
    return crop_whitespace(plt.imread(path))


def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image

    pad_total = target_height - image.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_width = [(pad_top, pad_bottom), (0, 0)]
    if image.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode="constant", constant_values=1.0)


def combine_row_images(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_height = max(left.shape[0], right.shape[0])
    left_padded = pad_to_height(left, target_height)
    right_padded = pad_to_height(right, target_height)
    return np.concatenate([left_padded, right_padded], axis=1)


def load_loss(perturbation: str) -> float:
    params_path = parameter_path(perturbation)
    with open(params_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("loss ="):
                return float(line.split("=", 1)[1].strip())
    raise ValueError(f"Could not find loss in {params_path}")


def build_figure(perturbations: list[str]) -> plt.Figure:
    n_rows = len(perturbations)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, 1.9 * n_rows + 0.35), squeeze=False)
    axes = axes[:, 0]

    fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.005, hspace=0.12)
    for row_idx, perturbation in enumerate(perturbations):
        ecdf_path, isotonic_path = diagnostic_paths(perturbation)
        ecdf_image = load_image(ecdf_path)
        isotonic_image = load_image(isotonic_path)
        combined_image = combine_row_images(ecdf_image, isotonic_image)
        loss = load_loss(perturbation)

        ax = axes[row_idx]
        ax.imshow(combined_image)
        ax.axis("off")
        ax.set_title(f"{perturbation}, loss={loss:.4g}", fontsize=10.5, pad=TITLE_PAD)

    return fig


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"

    fig.savefig(png_path, dpi=250, bbox_inches="tight")
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
    validate_inputs(perturbations)
    fig = build_figure(perturbations)
    save_figure(fig)


if __name__ == "__main__":
    main()
