#!/usr/bin/env python
"""
Generate the Figure S11e source panel: Figure 3f-style normalized isotonic
fits for Trim28 clones in the Trim28-Setdb1 dataset.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../scripts/utils')
from figure_format import (  # noqa: E402
    cell_type_colors,
    cell_type_orders_trim28_setdb1,
    get_adata,
)
from isotonic_utils import isotonic_with_ci, prepare_clone_proportions_df  # noqa: E402


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'

plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False

DEFAULT_PNG = "../manuscript/plots/Sup_Figure_11_shared_source/d.png"
DEFAULT_SVG = "../manuscript/plots/Sup_Figure_11_shared_source/d.svg"
PERTURBATION = "Trim28"
CELL_TYPES = [ct for ct in cell_type_orders_trim28_setdb1 if ct != "Progenitor dLN"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the Figure S11e source normalized isotonic fits for Trim28 clones."
    )
    parser.add_argument("--png", default=DEFAULT_PNG, help="Output PNG path.")
    parser.add_argument("--svg", default=DEFAULT_SVG, help="Output SVG path.")
    return parser.parse_args()


def subset_clone_df(clone_df, perturbation: str):
    subset = clone_df[clone_df["perturbation"] == perturbation].copy()
    if subset.empty:
        raise ValueError(f"No clone records found for perturbation '{perturbation}'.")
    subset["log2_size"] = np.log2(subset["num_cells"].clip(lower=1))
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_size"])
    subset = subset.sort_values("log2_size")
    if subset.empty:
        raise ValueError(f"No valid clone sizes for perturbation '{perturbation}'.")
    return subset


def save_figure(fig, png_path: str, svg_path: str):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"WARNING: {os.path.basename(png_path)} is {png_size:.1f}MB (>5MB).")
    if svg_size > 500:
        print(f"WARNING: {os.path.basename(svg_path)} is {svg_size:.0f}KB (>500KB).")


def plot_normalized_panel(clone_df, adata, png_path: str, svg_path: str):
    subset = subset_clone_df(clone_df, PERTURBATION)
    x = subset["log2_size"].to_numpy()
    x_min = max(float(x.min()), 0.0)
    x_max = min(float(x.max()), 8.0)
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_grid = np.linspace(x_min, x_max, 200)

    adata_subset = adata[adata.obs["perturbation"] == PERTURBATION]
    global_props = {
        cell_type: (adata_subset.obs["cell_type"] == cell_type).mean()
        for cell_type in CELL_TYPES
    }

    pseudocount = 1e-10
    fig, ax = plt.subplots(figsize=(12 * 0.7 * 0.8, 6 * 0.8))

    for cell_type in CELL_TYPES:
        if cell_type not in subset.columns:
            continue

        global_prop = global_props.get(cell_type, 0.0)
        if global_prop == 0:
            continue

        y = subset[cell_type].to_numpy()
        if np.allclose(y, 0):
            continue

        transform = lambda pred, gp=global_prop: np.log2((pred + pseudocount) / (gp + pseudocount))
        ci = isotonic_with_ci(
            x=x,
            y=y,
            x_grid=x_grid,
            n_bootstrap=500,
            random_state=42,
            transform=transform,
        )

        ax.plot(
            ci.x_grid,
            ci.fit,
            color=cell_type_colors[cell_type],
            linewidth=2,
            label=cell_type,
        )

        if ci.bootstrap_samples > 0 and np.all(np.isfinite(ci.ci_lower)):
            ax.fill_between(
                ci.x_grid,
                ci.ci_lower,
                ci.ci_upper,
                color=cell_type_colors[cell_type],
                alpha=0.2,
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("log2(clone size)")
    ax.set_ylabel("Normalized cell type\nproportion")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-4, 4)

    save_figure(fig, png_path, svg_path)


def main():
    args = parse_args()

    print("Loading trim28_setdb1 dataset...")
    adata = get_adata(
        "trim28_setdb1",
        use_cells_only_one_construct=False,
        remove_NG=True,
        remove_lymph_node=True,
        get_raw_scaled=True,
    )

    print("Preparing clone-level tables...")
    clone_props = prepare_clone_proportions_df(adata, cell_type_orders_trim28_setdb1)

    print("Generating Figure S11e source panel...")
    plot_normalized_panel(clone_props, adata, args.png, args.svg)
    print("Figure S11e source panel saved successfully.")


if __name__ == "__main__":
    main()
