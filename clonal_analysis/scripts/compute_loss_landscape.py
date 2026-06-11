#!/usr/bin/env python
"""
Compute loss landscape for clonal modeling parameters.
Evaluates loss over a 3D grid of p_self, p_split_one, and p_split_two.
Supports multiple perturbation datasets defined in DATASET_CONFIG.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from multiprocessing import Pool
from functools import partial
import time
from tqdm import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLONAL_ANALYSIS_DIR = os.path.dirname(SCRIPT_DIR)
REAL_DATA_DIR = os.path.join(CLONAL_ANALYSIS_DIR, "intermediate_files", "real_data")

# Add path to clonal modeling scripts
sys.path.append('./scripts/clonal_modeling/scripts')
from isotonic_regression import (
    simulate_clone_sizes_advanced,
    build_observed_ecdf,
    build_observed_iso_curve,
    compute_ecdf_mse,
    compute_isotonic_mse
)

# Dataset configuration (paths relative to project root)
GENE10_PERTURBATIONS = [
    "NT",
    "Bach2",
    "Ifngr1",
    "Ifngr2",
    "Il18r1",
    "Jak2",
    "Jak3",
    "Prdm1",
    "Setdb1",
    "Tnfrsf1b",
    "Trim28",
]
TRIM28_SETDB1_PERTURBATIONS = ["NT", "Trim28", "Setdb1"]
GRID_POINTS = 30

DATASET_CONFIG = {}

for perturbation in GENE10_PERTURBATIONS:
    DATASET_CONFIG[f"gene10_{perturbation}"] = {
        "data_path": os.path.join(REAL_DATA_DIR, "gene10", "input", f"{perturbation}.csv"),
        "n_points": GRID_POINTS,
        "group": "gene10",
        "output_name": perturbation,
    }

for perturbation in TRIM28_SETDB1_PERTURBATIONS:
    DATASET_CONFIG[f"trim28_setdb1_{perturbation}"] = {
        "data_path": os.path.join(REAL_DATA_DIR, "trim28_setdb1", "input", f"{perturbation}.csv"),
        "n_points": GRID_POINTS,
        "group": "trim28_setdb1",
        "output_name": perturbation,
    }

def evaluate_single_point(params, df_obs, xx_obs_ecdf, yy_obs_ecdf, xx_obs_iso, yy_obs_iso, mean_divisions, subsample_fraction):
    """
    Evaluate loss for a single parameter combination.

    Parameters
    ----------
    params : tuple
        (p_self, p_split_one, p_split_two)
    df_obs : pd.DataFrame
        Observed data with columns [Num Progenitor, Num Effector]
    xx_obs_ecdf, yy_obs_ecdf : arrays
        Precomputed observed ECDF
    xx_obs_iso, yy_obs_iso : arrays
        Precomputed observed isotonic regression
    mean_divisions : float
        Mean number of divisions for simulation
    subsample_fraction : float
        Subsample fraction for simulation

    Returns
    -------
    float
        Combined loss value (ECDF MSE + Isotonic MSE)
    """
    p_self, p_split_one, p_split_two = params

    # Check validity
    if p_self < 0 or p_split_one < 0 or p_split_two < 0:
        return np.nan
    if (p_self + p_split_one + p_split_two) > 1:
        return np.nan

    try:
        # Run simulation
        df_sim = simulate_clone_sizes_advanced(
            num_progenitors=1_000_000,  # Reduced for speed
            mean_divisions=mean_divisions,
            subsample_fraction=subsample_fraction,
            p_self=p_self,
            p_split_one=p_split_one,
            p_split_two=p_split_two,
            division_type="poisson",
            random_seed=None
        )

        # Compute losses
        ecdf_loss = compute_ecdf_mse(df_sim, xx_obs_ecdf, yy_obs_ecdf)
        iso_loss = compute_isotonic_mse(df_sim, xx_obs_iso, yy_obs_iso)

        return ecdf_loss + iso_loss

    except Exception as e:
        print(f"Error at ({p_self:.3f}, {p_split_one:.3f}, {p_split_two:.3f}): {e}")
        return np.nan

def compute_loss_landscape(dataset="NT", output_path=None, mean_divisions=5, subsample_fraction=0.15):
    """
    Main function to compute the loss landscape over parameter space.

    Parameters
    ----------
    dataset : str
        Dataset key (see DATASET_CONFIG for valid options)
    output_path : str, optional
        Custom output path for the NPZ file
    mean_divisions : float
        Mean number of divisions for simulation (default: 5)
    subsample_fraction : float
        Subsample fraction for simulation (default: 0.15)
    """

    # Load observed data
    if dataset not in DATASET_CONFIG:
        valid = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of: {valid}")

    dataset_info = DATASET_CONFIG[dataset]
    data_path = dataset_info["data_path"]

    print(f"Loading {dataset} data from {data_path}...")

    df_obs = pd.read_csv(data_path)

    # Prepare observed data
    tot_obs = df_obs["Num Progenitor"] + df_obs["Num Effector"]
    df_obs["Clone Size"] = tot_obs
    df_obs["Proportion of Progenitors"] = df_obs["Num Progenitor"] / tot_obs.clip(lower=1)
    df_obs["Log2 Clone Size"] = np.log2(tot_obs.clip(lower=1))

    # Compute max clone size for metrics
    real_max = df_obs["Log2 Clone Size"].max()
    max_clone_size = 0.9 * real_max if real_max > 0 else 1.0

    # Precompute observed ECDF and isotonic curves
    print("Computing observed ECDF and isotonic regression...")
    xx_obs_ecdf, yy_obs_ecdf = build_observed_ecdf(df_obs, max_clone_size)
    xx_obs_iso, yy_obs_iso = build_observed_iso_curve(df_obs, max_clone_size)

    # Create parameter grids
    print("Creating parameter grids...")
    n_points = dataset_info["n_points"]
    p_self_values = np.linspace(0, 0.5, n_points)
    p_split_one_values = np.linspace(0, 0.4, n_points)
    p_split_two_values = np.linspace(0, 0.4, n_points)

    # Initialize loss grid
    loss_grid = np.full((n_points, n_points, n_points), np.nan)

    # Create all parameter combinations
    param_combinations = []
    indices = []
    for i, p_self in enumerate(p_self_values):
        for j, p_split_one in enumerate(p_split_one_values):
            for k, p_split_two in enumerate(p_split_two_values):
                if (p_self + p_split_one + p_split_two) <= 1:
                    param_combinations.append((p_self, p_split_one, p_split_two))
                    indices.append((i, j, k))

    print(f"Total valid parameter combinations: {len(param_combinations)}")

    # Create partial function with fixed observed data
    eval_func = partial(
        evaluate_single_point,
        df_obs=df_obs,
        xx_obs_ecdf=xx_obs_ecdf,
        yy_obs_ecdf=yy_obs_ecdf,
        xx_obs_iso=xx_obs_iso,
        yy_obs_iso=yy_obs_iso,
        mean_divisions=mean_divisions,
        subsample_fraction=subsample_fraction
    )

    # Compute losses in parallel
    print("Computing losses in parallel...")
    print(f"Using mean_divisions={mean_divisions}, subsample_fraction={subsample_fraction}")
    start_time = time.time()

    n_workers = int(os.environ.get("CLONAL_N_WORKERS", "100"))

    with Pool(n_workers) as pool:
        # Use imap with tqdm for progress tracking
        results = list(tqdm(
            pool.imap(eval_func, param_combinations),
            total=len(param_combinations),
            desc="Computing loss values",
            unit="param_combo"
        ))

    # Fill the loss grid
    for (i, j, k), loss in zip(indices, results):
        loss_grid[i, j, k] = loss

    elapsed = time.time() - start_time
    print(f"Computation completed in {elapsed:.1f} seconds")

    # Save results
    if output_path is None:
        base_dir = "./clonal_analysis/intermediate_files"
        output_dir = os.path.join(base_dir, dataset_info["group"], dataset_info["output_name"])
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "grid_search_loss.npz")
    else:
        # Use provided output path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    np.savez(
        output_path,
        p_self_values=p_self_values,
        p_split_one_values=p_split_one_values,
        p_split_two_values=p_split_two_values,
        loss_grid=loss_grid,
        n_valid=len(param_combinations)
    )

    print(f"Results saved to {output_path}")

    # Print summary statistics
    valid_losses = loss_grid[~np.isnan(loss_grid)]
    if len(valid_losses) > 0:
        print(f"\n{dataset} Loss statistics:")
        print(f"  Min loss: {np.min(valid_losses):.6f}")
        print(f"  Max loss: {np.max(valid_losses):.6f}")
        print(f"  Mean loss: {np.mean(valid_losses):.6f}")
        print(f"  Median loss: {np.median(valid_losses):.6f}")

        # Find minimum
        min_idx = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)
        print(f"\nOptimal {dataset} parameters:")
        print(f"  p_self: {p_self_values[min_idx[0]]:.3f}")
        print(f"  p_split_one: {p_split_one_values[min_idx[1]]:.3f}")
        print(f"  p_split_two: {p_split_two_values[min_idx[2]]:.3f}")
        print(f"  Loss: {loss_grid[min_idx]:.6f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute loss landscape for clonal modeling")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gene10_NT",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the NPZ file (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--mean-divisions",
        type=float,
        default=5,
        help="Mean number of divisions for simulation (default: 5)"
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.15,
        help="Subsample fraction for simulation (default: 0.15)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_loss_landscape(
        dataset=args.dataset,
        output_path=args.output,
        mean_divisions=args.mean_divisions,
        subsample_fraction=args.subsample_fraction
    )
