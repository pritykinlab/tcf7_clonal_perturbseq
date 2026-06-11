#!/usr/bin/env python
"""
Generate ECDF and isotonic regression diagnostics for the best-loss parameter set
associated with a single perturbation. Produces PNG/SVG outputs comparing real data
against the best simulated clone data and writes the parameter summary to disk.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

# Required font settings for publication-quality plots
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_PATH = os.path.join(REPO_ROOT, "scripts", "utils")
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from figure_format import perturbation_colors  # noqa: E402

# Add clonal modeling helpers
CLONAL_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "clonal_modeling", "scripts")
if CLONAL_SCRIPTS not in sys.path:
    sys.path.append(CLONAL_SCRIPTS)

from isotonic_regression import (  # noqa: E402
    build_observed_ecdf,
    build_observed_iso_curve,
    simulate_clone_sizes_advanced,
)


@dataclass
class ParameterSummary:
    label: str
    p_self: float
    p_split_one: float
    p_split_two: float
    loss: float
    mean_divisions: float
    subsample_fraction: float

    def to_text(self) -> str:
        return (
            f"Best {self.label} parameter summary (highlighted lowest-loss point)\n"
            f"p_sym = {self.p_self:.12f}\n"
            f"p_asym = {self.p_split_one:.12f}\n"
            f"p_dif = {self.p_split_two:.12f}\n"
            f"loss = {self.loss:.12f}\n"
            f"mean_divisions = {self.mean_divisions}\n"
            f"subsample_fraction = {self.subsample_fraction}\n"
        )


def find_best_parameters(npz_path: str) -> Tuple[float, float, float, float]:
    """Load the grid search NPZ file and return the minimum-loss parameters."""
    data = np.load(npz_path)
    loss_grid = data["loss_grid"]

    flat_index = np.nanargmin(loss_grid)
    i, j, k = np.unravel_index(flat_index, loss_grid.shape)

    p_self = float(data["p_self_values"][i])
    p_split_one = float(data["p_split_one_values"][j])
    p_split_two = float(data["p_split_two_values"][k])
    loss = float(loss_grid[i, j, k])

    return p_self, p_split_one, p_split_two, loss


def load_observed_data(csv_path: str) -> pd.DataFrame:
    """Load real NT clone data and compute derived columns."""
    df = pd.read_csv(csv_path)
    total = df["Num Progenitor"] + df["Num Effector"]
    df["Clone Size"] = total
    df["Proportion of Progenitors"] = df["Num Progenitor"] / total.clip(lower=1)
    df["Log2 Clone Size"] = np.log2(total.clip(lower=1))
    return df


def simulate_best_clones(
    params: ParameterSummary,
    num_progenitors: int,
    mean_divisions: float,
    subsample_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """Simulate clone sizes using the best parameters."""
    df_sim = simulate_clone_sizes_advanced(
        num_progenitors=num_progenitors,
        mean_divisions=mean_divisions,
        subsample_fraction=subsample_fraction,
        p_self=params.p_self,
        p_split_one=params.p_split_one,
        p_split_two=params.p_split_two,
        division_type="poisson",
        random_seed=random_seed,
    )
    return df_sim


def plot_ecdf(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    xx_axis: np.ndarray,
    output_base: str,
    color: str,
    label: str,
) -> Tuple[str, str]:
    """Create ECDF comparison plot and return PNG/SVG paths."""
    ecdf_real = sm.distributions.ECDF(df_real["Log2 Clone Size"].dropna())
    y_real = ecdf_real(xx_axis)

    ecdf_sim = sm.distributions.ECDF(df_sim["Log2 Clone Size"].dropna())
    y_sim = ecdf_sim(xx_axis)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(xx_axis, y_real, color=color, linestyle="-", linewidth=2, label=f"{label} Real Data")
    ax.plot(xx_axis, y_sim, color=color, linestyle=":", linewidth=2.5, label="Best Simulated Data")

    ax.set_xlabel("Log2(Clone Size)")
    ax.set_ylabel("ECDF")
    ax.set_xlim(xx_axis.min(), xx_axis.max())
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right")

    png_path = f"{output_base}_ecdf.png"
    svg_path = f"{output_base}_ecdf.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_isotonic(
    xx_obs: np.ndarray,
    yy_obs: np.ndarray,
    df_sim: pd.DataFrame,
    output_base: str,
    color: str,
    label: str,
) -> Tuple[str, str]:
    """Create isotonic regression comparison plot and return PNG/SVG paths."""
    iso_sim = IsotonicRegression(increasing=False, out_of_bounds="clip")
    x_sim = df_sim["Log2 Clone Size"].fillna(0).values
    y_sim = df_sim["Proportion of Progenitors"].fillna(0).values
    iso_sim.fit(x_sim, y_sim)
    y_sim_pred = iso_sim.predict(xx_obs)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(xx_obs, yy_obs, color=color, linestyle="-", linewidth=2, label=f"{label} Real Data")
    ax.plot(xx_obs, y_sim_pred, color=color, linestyle=":", linewidth=2.5, label="Best Simulated Data")

    ax.set_xlabel("Log2(Clone Size)")
    ax.set_ylabel("Progenitor Cell Type Proportion")
    ax.set_xlim(xx_obs.min(), xx_obs.max())
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")

    png_path = f"{output_base}_isotonic.png"
    svg_path = f"{output_base}_isotonic.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics for best NT clone modeling parameters.")
    parser.add_argument("--grid", required=True, help="Path to grid_search_loss NPZ file for the perturbation.")
    parser.add_argument("--label", required=True, help="Perturbation label (used for colors and text).")
    parser.add_argument("--real-data", required=True, help="Path to real data CSV for the perturbation.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix (no extension) for generated files.")
    parser.add_argument("--mean-divisions", type=float, default=5.0, help="Mean divisions used in simulation.")
    parser.add_argument("--subsample-fraction", type=float, default=0.15, help="Subsample fraction used in simulation.")
    parser.add_argument("--num-progenitors", type=int, default=1_000_000, help="Number of progenitors to simulate.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for simulation reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    p_self, p_split_one, p_split_two, loss = find_best_parameters(args.grid)
    params = ParameterSummary(
        label=args.label,
        p_self=p_self,
        p_split_one=p_split_one,
        p_split_two=p_split_two,
        loss=loss,
        mean_divisions=args.mean_divisions,
        subsample_fraction=args.subsample_fraction,
    )

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    params_path = f"{args.output_prefix}_parameters.txt"
    with open(params_path, "w", encoding="utf-8") as handle:
        handle.write(params.to_text())

    df_real = load_observed_data(args.real_data)
    real_max = df_real["Log2 Clone Size"].max()
    max_clone_size = 0.9 * real_max if real_max > 0 else 1.0

    xx_obs_ecdf, yy_obs_ecdf = build_observed_ecdf(df_real, max_clone_size)
    xx_obs_iso, yy_obs_iso = build_observed_iso_curve(df_real, max_clone_size)

    df_sim = simulate_best_clones(
        params=params,
        num_progenitors=args.num_progenitors,
        mean_divisions=args.mean_divisions,
        subsample_fraction=args.subsample_fraction,
        random_seed=args.random_seed,
    )

    color = perturbation_colors.get(args.label, "gray")

    plot_ecdf(df_real, df_sim, xx_obs_ecdf, args.output_prefix, color, args.label)
    plot_isotonic(xx_obs_iso, yy_obs_iso, df_sim, args.output_prefix, color, args.label)

    print("Parameter summary written to:", params_path)
    print(f"Finished generating ECDF and isotonic diagnostics for {args.label}.")


if __name__ == "__main__":
    main()
