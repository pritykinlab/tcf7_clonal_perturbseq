#!/usr/bin/env python
"""
Optimize NT parameters using differential evolution with 5 parameters
including subsample_fraction as an optimizable parameter.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from functools import partial

# Add path to clonal modeling scripts
sys.path.append('./scripts/clonal_modeling/scripts')
from isotonic_regression import (
    simulate_clone_sizes_advanced,
    build_observed_ecdf,
    build_observed_iso_curve,
    compute_ecdf_mse,
    compute_isotonic_mse
)

def objective(params, df_obs, max_clone_size, xx_obs, yy_obs_iso, yy_obs_ecdf):
    """
    Objective function for differential evolution optimization.
    Optimizes 5 parameters including subsample_fraction.
    """
    p_self, p_split_one, p_split_two, subsample_fraction, mean_divisions = params

    # Parameter validation
    if (p_self < 0 or p_split_one < 0 or p_split_two < 0 or
        (p_self + p_split_one + p_split_two) > 1 or
        subsample_fraction < 0.05 or subsample_fraction > 0.2 or
        mean_divisions < 2 or mean_divisions > 7):
        return 1e6

    # Run simulation with optimizable subsample_fraction
    df_sim = simulate_clone_sizes_advanced(
        num_progenitors=1_000_000,
        mean_divisions=mean_divisions,
        subsample_fraction=subsample_fraction,  # NOW AN OPTIMIZED PARAMETER!
        p_self=p_self,
        p_split_one=p_split_one,
        p_split_two=p_split_two,
        division_type="poisson"
    )

    # Compute MSE for both ECDF and isotonic regression
    ecdf_val = compute_ecdf_mse(df_sim, xx_obs, yy_obs_ecdf)
    iso_val = compute_isotonic_mse(df_sim, xx_obs, yy_obs_iso)

    return ecdf_val + iso_val

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize NT parameters with differential evolution (5 parameters)"
    )
    parser.add_argument("--input_file", required=True,
                        help="CSV with columns [Num Progenitor, Num Effector]")
    parser.add_argument("--output_file", required=True,
                        help="Output TSV with optimized parameters")
    parser.add_argument("--maxiter", type=int, default=10000,
                        help="Max iterations for differential evolution")
    parser.add_argument("--popsize", type=int, default=15,
                        help="Population size for differential evolution")
    parser.add_argument("--ncpus", type=int, default=64,
                        help="Number of parallel CPU workers")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load observed data
    df_obs = pd.read_csv(args.input_file)
    if "Num Progenitor" not in df_obs.columns or "Num Effector" not in df_obs.columns:
        raise ValueError("Input must have columns: Num Progenitor, Num Effector")

    # Set random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    # Prepare observed dataframe
    tot_obs = df_obs["Num Progenitor"] + df_obs["Num Effector"]
    df_obs["Clone Size"] = tot_obs
    df_obs["Proportion of Progenitors"] = df_obs["Num Progenitor"] / tot_obs.clip(lower=1)
    df_obs["Log2 Clone Size"] = np.log2(tot_obs.clip(lower=1))

    # Set max clone size for comparison
    real_max = df_obs["Log2 Clone Size"].max()
    max_clone_size = 0.9 * real_max if real_max > 0 else 1.0

    # Build observed curves
    print("Building observed curves...")
    xx_obs, yy_obs_iso = build_observed_iso_curve(df_obs, max_clone_size)
    xx_obs_ecdf, yy_obs_ecdf = build_observed_ecdf(df_obs, max_clone_size)

    # Create objective function with partial
    obj_func = partial(
        objective,
        df_obs=df_obs,
        max_clone_size=max_clone_size,
        xx_obs=xx_obs,
        yy_obs_iso=yy_obs_iso,
        yy_obs_ecdf=yy_obs_ecdf
    )

    # Define bounds for 5 parameters
    bounds = [
        (0, 1),       # p_self
        (0, 1),       # p_split_one
        (0, 1),       # p_split_two
        (0.05, 0.2),  # subsample_fraction (OPTIMIZING BETWEEN 0.05 AND 0.2)
        (2, 7),       # mean_divisions
    ]

    print("\n" + "="*60)
    print("NT PARAMETER OPTIMIZATION")
    print("="*60)
    print("\nParameter bounds:")
    print(f"  p_self:             [0, 1]")
    print(f"  p_split_one:        [0, 1]")
    print(f"  p_split_two:        [0, 1]")
    print(f"  subsample_fraction: [0.05, 0.2]")
    print(f"  mean_divisions:     [2, 7]")
    print(f"\nOptimization settings:")
    print(f"  CPU cores:     {args.ncpus}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Population size: {args.popsize}")
    print(f"  Random seed:    {args.random_seed}")
    print("\nStarting differential evolution optimization...")
    print("-"*60)

    # Run differential evolution
    result = differential_evolution(
        obj_func,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        workers=args.ncpus,
        polish=True,
        atol=1e-6,
        tol=1e-6,
        updating='deferred',
        disp=True,
        seed=args.random_seed
    )

    # Extract optimized parameters
    p_self_opt, p_split_one_opt, p_split_two_opt, subsample_opt, divisions_opt = result.x

    # Save results
    results_df = pd.DataFrame({
        "perturbation": ["NT"],
        "p_self": [p_self_opt],
        "p_split_one": [p_split_one_opt],
        "p_split_two": [p_split_two_opt],
        "subsample_fraction": [subsample_opt],
        "mean_divisions": [divisions_opt],
        "objective_value": [result.fun],
        "success": [result.success],
        "n_iterations": [result.nit],
        "n_func_evals": [result.nfev]
    })

    results_df.to_csv(args.output_file, sep='\t', index=False)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nOptimized parameters for NT:")
    print(f"  p_self:             {p_self_opt:.6f}")
    print(f"  p_split_one:        {p_split_one_opt:.6f}")
    print(f"  p_split_two:        {p_split_two_opt:.6f}")
    print(f"  subsample_fraction: {subsample_opt:.6f}")
    print(f"  mean_divisions:     {divisions_opt:.6f}")
    print(f"\nOptimization metrics:")
    print(f"  Objective value: {result.fun:.8f}")
    print(f"  Success:        {result.success}")
    print(f"  Iterations:     {result.nit}")
    print(f"  Function evals: {result.nfev}")
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()