#!/usr/bin/env python
"""
Create a bar plot comparing permissible parameter ranges for multiple perturbations.
Each perturbation contributes its lowest-loss points; a shared PCA direction is computed
to derive comparable ranges for p_self, p_split_one, and p_split_two.
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Required font settings for manuscript-style figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = (6, 4.5)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from figure_format import perturbation_colors  # noqa: E402


def parse_dataset_spec(spec: str) -> Tuple[str, str]:
    """Parse a dataset specification of the form LABEL=PATH."""
    if "=" not in spec:
        raise ValueError(f"Dataset specification '{spec}' must be of the form LABEL=PATH")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Dataset specification '{spec}' is invalid")
    return label, path


def get_color(label: str, idx: int) -> str:
    """Return a color for the given label, falling back to a predefined palette."""
    return perturbation_colors.get(label, f"C{idx}")


def load_and_extract_lowest_points(data_file: str, n_points: int = 50) -> np.ndarray:
    """
    Load loss landscape data and extract the n lowest loss points.

    Returns an array of shape (n_points, 4) with columns [p_self, p_split_one, p_split_two, loss].
    """
    data = np.load(data_file)
    p_self_values = data["p_self_values"]
    p_split_one_values = data["p_split_one_values"]
    p_split_two_values = data["p_split_two_values"]
    loss_grid = data["loss_grid"]
    data.close()

    all_points = []
    for i, p_self in enumerate(p_self_values):
        for j, p_split_one in enumerate(p_split_one_values):
            for k, p_split_two in enumerate(p_split_two_values):
                loss = loss_grid[i, j, k]
                if not np.isnan(loss) and (p_self + p_split_one + p_split_two) <= 1.0:
                    all_points.append([p_self, p_split_one, p_split_two, loss])

    if not all_points:
        raise ValueError(f"No valid points found in {data_file}")

    all_points = np.array(all_points)
    sorted_indices = np.argsort(all_points[:, 3])
    return all_points[sorted_indices[: min(n_points, len(all_points))]]


def calculate_shared_direction(datasets: List[Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute a shared PCA direction using centered points from all provided datasets.

    Returns a mapping of label -> centroid and the shared direction vector.
    """
    centroids: Dict[str, np.ndarray] = {}
    centered_blocks = []

    for ds in datasets:
        coords = ds["points"][:, :3]
        centroid = np.mean(coords, axis=0)
        centroids[ds["label"]] = centroid
        centered_blocks.append(coords - centroid)

    combined_centered = np.vstack(centered_blocks)
    pca = PCA(n_components=1)
    pca.fit(combined_centered)
    direction = pca.components_[0]
    if direction[0] < 0:
        direction = -direction

    return centroids, direction


def find_permissible_ranges(centroid: np.ndarray, direction: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Find the permissible ranges for each parameter by extending the line within the valid simplex.
    """
    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        if direction[i] > 0:
            t_max = min(t_max, (1 - centroid[i]) / direction[i])
            t_min = max(t_min, -centroid[i] / direction[i])
        elif direction[i] < 0:
            t_max = min(t_max, -centroid[i] / direction[i])
            t_min = max(t_min, (1 - centroid[i]) / direction[i])

    sum_centroid = np.sum(centroid)
    sum_direction = np.sum(direction)

    if sum_direction > 0:
        t_max = min(t_max, (1 - sum_centroid) / sum_direction)
    elif sum_direction < 0:
        t_min = max(t_min, (1 - sum_centroid) / sum_direction)

    p_min = centroid + t_min * direction
    p_max = centroid + t_max * direction

    return {
        "p_self": (
            max(0.0, min(p_min[0], p_max[0])),
            min(1.0, max(p_min[0], p_max[0])),
        ),
        "p_one": (
            max(0.0, min(p_min[1], p_max[1])),
            min(1.0, max(p_min[1], p_max[1])),
        ),
        "p_two": (
            max(0.0, min(p_min[2], p_max[2])),
            min(1.0, max(p_min[2], p_max[2])),
        ),
        "p_sum": (
            max(0.0, min(p_min.sum(), p_max.sum())),
            min(1.0, max(p_min.sum(), p_max.sum())),
        ),
    }


def create_bar_plot(
    range_map: Dict[str, Dict[str, Tuple[float, float]]],
    datasets: List[Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    """Create a grouped bar plot comparing parameter ranges across perturbations."""
    params = ["p_self", "p_one", "p_two", "p_sum"]
    x_labels = [
        r"$p_{\mathrm{sym}}$",
        r"$p_{\mathrm{asym}}$",
        r"$p_{\mathrm{dif}}$",
        r"$p_{\mathrm{sym}} + p_{\mathrm{asym}} + p_{\mathrm{dif}}$",
    ]

    n_datasets = len(datasets)
    width = 0.6 / n_datasets
    x = np.arange(len(params))

    fig, ax = plt.subplots()

    for idx, ds in enumerate(datasets):
        label = ds["label"]
        color = ds["color"]
        ranges = range_map[label]
        mins = [ranges[p][0] for p in params]
        maxs = [ranges[p][1] for p in params]
        heights = [maxs[i] - mins[i] for i in range(len(params))]
        offsets = x + (idx - (n_datasets - 1) / 2) * width

        ax.bar(
            offsets,
            heights,
            width,
            bottom=mins,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper right", frameon=False)
    max_upper = max(r[p][1] for r in range_map.values() for p in params)
    ax.set_ylim(0, max(max_upper * 1.05, 1.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Bar plot saved to: {output_path}")
    plt.close()


def print_ranges(range_map: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    """Print the parameter ranges in a formatted way."""
    print("\n" + "=" * 60)
    print("PERMISSIBLE PARAMETER RANGES")
    print("=" * 60)

    param_order = ["p_self", "p_one", "p_two", "p_sum"]
    display_names = {
        "p_self": "p_sym",
        "p_one": "p_asym",
        "p_two": "p_dif",
        "p_sum": "p_sym + p_asym + p_dif",
    }

    for label, ranges in range_map.items():
        print(f"\n{label} Dataset:")
        for param in param_order:
            min_val, max_val = ranges[param]
            print(f"  {display_names[param]}: [{min_val:.4f}, {max_val:.4f}] (width: {max_val - min_val:.4f})")
    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create bar plot comparing permissible parameter ranges for multiple perturbations."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset specification in the form LABEL=PATH. Provide at least two entries.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./clonal_analysis/intermediate_files/parameter_ranges_bar_plot.png",
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=50,
        help="Number of lowest loss points to use for line fitting (default: 50)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    if len(args.dataset) < 2:
        raise ValueError("At least two --dataset arguments are required.")

    dataset_specs = [parse_dataset_spec(spec) for spec in args.dataset]

    datasets: List[Dict[str, np.ndarray]] = []
    for idx, (label, path) in enumerate(dataset_specs):
        print(f"Loading dataset '{label}' from {path}")
        points = load_and_extract_lowest_points(path, args.n_points)
        if points.shape[0] < args.n_points:
            raise ValueError(f"Dataset '{label}' does not contain {args.n_points} valid lowest-loss points.")
        datasets.append(
            {
                "label": label,
                "path": path,
                "color": get_color(label, idx),
                "points": points[: args.n_points],
            }
        )

    print("\nCalculating shared direction from combined data...")
    centroids, shared_direction = calculate_shared_direction(datasets)
    print(f"  Shared direction: {shared_direction}")

    print("\nCalculating permissible parameter ranges using shared direction...")
    range_map = {
        ds["label"]: find_permissible_ranges(centroids[ds["label"]], shared_direction) for ds in datasets
    }

    print_ranges(range_map)

    print("\nCreating bar plot...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_bar_plot(range_map, datasets, args.output)

    print(f"\n✓ Successfully created bar plot!")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
