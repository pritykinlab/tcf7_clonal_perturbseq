#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

###############################################################################
# 1) Simulation Helpers
###############################################################################
def simulate_clones_in_bulk(
    p_self: float,
    p_split_one: float,
    p_split_two: float,
    division_type: str,
    divisions: float,
    batch_size: int,
    random_seed: int
):
    """
    Simulates 'batch_size' clones in a single pass and returns the final
    (p_array, e_array) for each clone.

    Parameters
    ----------
    p_self : float
        Probability that a progenitor cell undergoes symmetric self-renewal (+2 progenitors).
    p_split_one : float
        Probability that a progenitor cell splits into (1 progenitor + 1 effector).
    p_split_two : float
        Probability that a progenitor cell splits into (+2 effectors).
    division_type : str
        Either 'poisson' or 'fixed'. If 'poisson', each clone gets Poisson(divisions)
        random divisions; if 'fixed', each clone gets exactly 'divisions' divisions.
    divisions : float
        Mean number of divisions (if 'poisson') or the fixed number of divisions.
    batch_size : int
        Number of clones to simulate in one shot.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Final progenitor (p_array) and effector (e_array) counts (no subsampling).
    """
    if random_seed:
        np.random.seed(random_seed)

    # Decide the number of divisions for each clone
    if division_type.lower() == "poisson":
        d_array = np.random.poisson(divisions, size=batch_size)
    else:
        # 'fixed' or anything else => each clone gets exactly divisions divisions
        d_array = np.full(batch_size, int(divisions), dtype=np.int64)

    max_d = d_array.max()

    # Start with 1 progenitor, 0 effectors for each clone
    p_array = np.ones(batch_size, dtype=np.int64)
    e_array = np.zeros(batch_size, dtype=np.int64)

    # Probability boundaries for np.searchsorted => up to 4 outcomes
    edges = np.array([
        p_self,
        p_self + p_split_one,
        p_self + p_split_one + p_split_two
    ], dtype=np.float64)

    # Simulate in bulk
    for division_idx in range(max_d):
        # Determine which clones are still dividing
        active_mask = (division_idx < d_array) & ((p_array + e_array) > 0)
        if not np.any(active_mask):
            break

        # Effectors of active clones double each division
        e_array[active_mask] *= 2

        # If there are no progenitors among active clones, skip the outcome logic
        total_p = p_array[active_mask].sum()
        if total_p == 0:
            continue

        # Flatten out all active progenitors for sampling
        sim_ids = np.repeat(np.flatnonzero(active_mask), p_array[active_mask])
        rand = np.random.random(sim_ids.size)

        # outcomes => 0: self-renew, 1: split_one, 2: split_two, 3: do-nothing
        outcomes = np.searchsorted(edges, rand)

        # 2D histogram: x-axis = simulation ID, y-axis = outcome
        H, _, _ = np.histogram2d(
            sim_ids,
            outcomes,
            bins=[np.arange(batch_size + 1), np.arange(5)]
        )
        H = H.astype(int)

        count0 = H[:, 0]  # self-renew => +2 progenitors
        count1 = H[:, 1]  # split_one => +1 progenitor, +1 effector
        count2 = H[:, 2]  # split_two => +2 effectors
        count3 = H[:, 3]  # do-nothing => +1 progenitor

        # Update p_array/e_array for active clones only
        p_array[active_mask] = (2 * count0 + count1 + count3)[active_mask]
        e_array[active_mask] += (count1 + 2 * count2)[active_mask]

    return p_array, e_array

def simulate_clone_sizes_advanced(
    num_progenitors: int,
    mean_divisions: float,
    subsample_fraction: float,  # keep this argument for internal use
    p_self: float,
    p_split_one: float,
    p_split_two: float,
    division_type: str = "poisson",
    random_seed: int = None,
    batch_size: int = 10_000
):
    """
    Simulates clone sizes in batches until we collect at least `num_progenitors`
    surviving (post-subsampling) clones, then returns exactly `num_progenitors`.

    Parameters
    ----------
    num_progenitors : int
        Desired number of surviving clones to return.
    mean_divisions : float
        Mean (if Poisson) or fixed number of divisions.
    subsample_fraction : float
        Probability (0-1) with which each cell is kept (binomial) at the end.
    p_self : float
        Probability that a progenitor cell undergoes symmetric self-renewal (+2 progenitors).
    p_split_one : float
        Probability that a progenitor cell splits into (1 progenitor + 1 effector).
    p_split_two : float
        Probability that a progenitor cell splits into (+2 effectors).
    division_type : str
        Either 'poisson' (default) or 'fixed'. Controls how divisions are assigned.
    random_seed : int
        RNG seed for reproducibility.
    batch_size : int
        How many clones to simulate in each bulk pass. Increase if clone yield is small.

    Returns
    -------
    pd.DataFrame
        Columns: [Clone, Clone Size, Progenitor Count, Proportion of Progenitors, Log2 Clone Size]
        Exactly `num_progenitors` rows (may be fewer if probabilities are invalid).
    """
    # Basic sanity check
    p_do_nothing = 1.0 - (p_self + p_split_one + p_split_two)
    if p_do_nothing < 0:
        # If the sum of p_self + p_split_one + p_split_two > 1,
        # that implies negative do-nothing probability -> invalid scenario
        return pd.DataFrame()

    if random_seed:
        np.random.seed(random_seed)
    clones_collected = []

    # Keep going until we have at least num_progenitors *surviving* clones
    while len(clones_collected) < num_progenitors:
        # Each pass simulates a batch of clones in bulk
        # We'll randomly choose a new seed each time to vary results
        seed_for_this_batch = np.random.randint(1_000_000_000)
        p_array, e_array = simulate_clones_in_bulk(
            p_self=p_self,
            p_split_one=p_split_one,
            p_split_two=p_split_two,
            division_type=division_type,
            divisions=mean_divisions,
            batch_size=batch_size,
            random_seed=seed_for_this_batch
        )

        # Binomially subsample progenitors and effectors
        subsampled_p = np.random.binomial(p_array, subsample_fraction)
        subsampled_e = np.random.binomial(e_array, subsample_fraction)

        total = subsampled_p + subsampled_e
        surviving_indices = np.where(total > 0)[0]

        for idx in surviving_indices:
            # We treat each surviving clone as a new row
            clones_collected.append({
                "Clone": len(clones_collected),
                "Clone Size": total[idx],
                "Progenitor Count": subsampled_p[idx],
                "Proportion of Progenitors": subsampled_p[idx] / total[idx],
                "Log2 Clone Size": np.log2(total[idx])
            })

    # Truncate to exactly num_progenitors in case we overshoot
    clones_collected = clones_collected[:num_progenitors]
    df = pd.DataFrame(clones_collected)
    return df

###############################################################################
# 2) ECDF MSE
###############################################################################
# Precompute observed ECDF
def build_observed_ecdf(df_obs, max_clone_size, num_points=100):
    x_obs = df_obs["Log2 Clone Size"].dropna()
    if len(x_obs) < 1:
        return np.linspace(0, max_clone_size, num_points), np.zeros(num_points)

    ecdf_obs = sm.distributions.ECDF(x_obs)
    xx_obs = np.linspace(0, max_clone_size, num_points)
    yy_obs_ecdf = ecdf_obs(xx_obs)

    return xx_obs, yy_obs_ecdf

# Updated ECDF MSE using precomputed xx_obs and yy_obs_ecdf
def compute_ecdf_mse(df_sim, xx_obs, yy_obs_ecdf):
    if df_sim.empty:
        return 1e6

    x_sim = df_sim["Log2 Clone Size"].dropna()
    if len(x_sim) < 1:
        return 1e6

    ecdf_sim = sm.distributions.ECDF(x_sim)
    yy_sim_ecdf = ecdf_sim(xx_obs)

    return np.mean((yy_obs_ecdf - yy_sim_ecdf)**2)

###############################################################################
# 3) Isotonic Regression Precomputation (Observed Data)
###############################################################################
def build_observed_iso_curve(df_obs, max_clone_size):
    """
    Fits a single isotonic regression for the observed data (progenitor fraction
    vs log2 clone size). Then evaluates that regression at 100 equally spaced
    points in [0, max_clone_size]. Returns (xx_obs, yy_obs).
    """

    x_obs = df_obs["Log2 Clone Size"].fillna(0).values
    y_obs = df_obs["Proportion of Progenitors"].fillna(0).values

    # Fit isotonic regression once for observed data
    iso_obs = IsotonicRegression(increasing=False).fit(x_obs, y_obs)

    # Evaluate isotonic regression at 100 equally spaced points
    xx_obs = np.linspace(0, max_clone_size, 100)
    yy_obs = iso_obs.predict(xx_obs)
    print("Isotonic Regression Observed Data:")
    print(xx_obs, yy_obs)

    return xx_obs, yy_obs

###############################################################################
# 4) Isotonic MSE (Simulated vs Observed Curve)
###############################################################################
def compute_isotonic_mse(df_sim, xx_obs, yy_obs):
    """
    Compare the precomputed (xx_obs, yy_obs) from observed data
    to a new isotonic regression curve fit to df_sim.
    Returns MSE of those two curves.
    """
    if df_sim.empty or len(df_sim) < 2 or len(xx_obs) < 1 or len(yy_obs) < 1:
        return 1e6

    x_sim = df_sim["Log2 Clone Size"].fillna(0).values
    y_sim = df_sim["Proportion of Progenitors"].fillna(0).values

    # Fit isotonic for the simulated data
    iso_sim = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x_sim, y_sim)

    # Predict on the same xx_obs grid
    yy_sim = iso_sim.predict(xx_obs)

    # Return MSE
    return np.mean((yy_obs - yy_sim)**2)

###############################################################################
# 5) Objective for Differential Evolution
###############################################################################
def objective(params, df_obs, max_clone_size, xx_obs, yy_obs_iso, yy_obs_ecdf, optimize_p_split_two=True):
    if optimize_p_split_two:
        p_self, p1, p2, mean_divisions = params
    else:
        p_self, p1, mean_divisions = params
        p2 = 0.0  # Fix p_split_two at 0

    if (
        p_self < 0 or p1 < 0 or p2 < 0 or
        (p_self + p1 + p2) > 1 or
        mean_divisions < 2 or mean_divisions > 7 or
        np.isnan(mean_divisions)
    ):
        return 1e6

    # Set subsample_fraction to 0.1 (fixed)
    subsample_fraction = 0.1

    df_sim = simulate_clone_sizes_advanced(
        num_progenitors=1_000_000, # usually 1_000_000
        mean_divisions=mean_divisions,
        subsample_fraction=subsample_fraction,
        p_self=p_self,
        p_split_one=p1,
        p_split_two=p2
    )

    ecdf_val = compute_ecdf_mse(df_sim, xx_obs, yy_obs_ecdf)
    iso_val = compute_isotonic_mse(df_sim, xx_obs, yy_obs_iso)

    return ecdf_val + iso_val

###############################################################################
# 6) Argparse + Main
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Isotonic Regression + advanced simulation, optimizing mean_divisions (subsample_fraction fixed at 0.1).")
    parser.add_argument("--input_file", required=True,
                        help="CSV with columns [Num Progenitor, Num Effector].")
    parser.add_argument("--output_file", required=True,
                        help="Output TSV file with optimized parameters.")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="Max iterations for differential evolution.")
    parser.add_argument("--ncpus", type=int, default=1,
                        help="Number of parallel CPU workers.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Optional random seed.")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument('--optimize_p_split_two', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to optimize p_split_two (default: True)')
    # Remove any argument for subsample_fraction
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Read the observed CSV
    df_obs = pd.read_csv(args.input_file)
    if "Num Progenitor" not in df_obs.columns or "Num Effector" not in df_obs.columns:
        raise ValueError("Input must have columns: Num Progenitor, Num Effector")

    # Sample with replacement if sample_seed is provided
    if args.sample_seed is not None:
        df_obs = df_obs.sample(n=len(df_obs), replace=True, random_state=args.sample_seed).reset_index(drop=True)

    if args.random_seed is not None: # be careful running this
        np.random.seed(args.random_seed)

    # 2) Prepare observed dataframe
    tot_obs = df_obs["Num Progenitor"] + df_obs["Num Effector"]
    df_obs["Clone Size"] = tot_obs
    df_obs["Proportion of Progenitors"] = df_obs["Num Progenitor"] / tot_obs.clip(lower=1)
    df_obs["Log2 Clone Size"] = np.log2(tot_obs.clip(lower=1))

    # 3) Local max clone size
    real_max = df_obs["Log2 Clone Size"].max()
    max_clone_size = 0.9 * real_max if real_max > 0 else 1.0

    xx_obs, yy_obs_iso = build_observed_iso_curve(df_obs, max_clone_size)
    xx_obs_ecdf, yy_obs_ecdf = build_observed_ecdf(df_obs, max_clone_size)

    # 5) Wrap the objective function
    from functools import partial
    obj_func = partial(
        objective,
        df_obs=df_obs,
        max_clone_size=max_clone_size,
        xx_obs=xx_obs,            # Common xx_obs for both iso and ecdf
        yy_obs_iso=yy_obs_iso,    # Isotonic regression curve
        yy_obs_ecdf=yy_obs_ecdf,  # ECDF curve
        optimize_p_split_two=args.optimize_p_split_two
    )

    # 6) Run differential evolution (updated bounds)
    print("Starting Differential Evolution with expanded parameter set")
    if args.optimize_p_split_two:
        bounds = [
            (0, 1),      # p_self
            (0, 1),      # p_split_one
            (0, 1),      # p_split_two
            (2, 7),      # mean_divisions
        ]
    else:
        bounds = [
            (0, 1),      # p_self
            (0, 1),      # p_split_one
            (2, 7),      # mean_divisions
        ]

    result = differential_evolution(
        obj_func,
        bounds,
        maxiter=args.maxiter,
        popsize=15,
        workers=args.ncpus,
        atol=1e-5 # 7.4e-5 is average loss in simulation
    )

    if args.optimize_p_split_two:
        p_self, p1, p2, mean_divisions = result.x
    else:
        p_self, p1, mean_divisions = result.x
        p2 = 0.0

    best_loss = result.fun

    # 7) Output the optimized parameters
    with open(args.output_file, "w") as f:
        f.write(f"p_self\t{p_self}\n")
        f.write(f"p_split_one\t{p1}\n")
        f.write(f"p_split_two\t{p2}\n")
        f.write(f"mean_divisions\t{mean_divisions}\n")
        f.write(f"subsample_fraction\t0.1\n")
        f.write(f"loss\t{best_loss}\n")

    print(f"[isotonic_regression] Best => p_self={p_self:.4f}, p1={p1:.4f}, p2={p2:.4f}, mean_divisions={mean_divisions:.2f}, subsample_fraction=0.1, loss={best_loss:.5f}")

    # Diagnostic plotting (ECDF and Isotonic Regression)
    df_real = df_obs.copy()  # Use the original observed data for plotting
    perturbation = "Optimized Parameters"  # Label for the plot
    seed = args.random_seed if args.random_seed is not None else 0
    output_file = args.output_file.replace(".tsv", "_diagnostic_plot.png")

    df_sim = simulate_clone_sizes_advanced(
        num_progenitors=1_000_000,
        mean_divisions=mean_divisions,
        subsample_fraction=0.1,
        p_self=p_self,
        p_split_one=p1,
        p_split_two=p2,
        division_type="poisson"
    )
    make_diagnostic_plot(df_real, df_sim, perturbation, seed, output_file)


def plot_ecdf(data, label, linestyle='-', color='black', alpha=1.0, max_clone_size=None):
    x = np.linspace(np.min(data), np.max(data), 500)
    if max_clone_size is not None:
        x = x[x <= max_clone_size]
    ecdf = sm.distributions.ECDF(data)
    y = ecdf(x)
    plt.plot(x, y, linestyle=linestyle, label=label, color=color, alpha=alpha)

def plot_isotonic(x, y, label, linestyle='-', color='black', alpha=1.0, max_clone_size=None):
    iso_reg = IsotonicRegression(increasing=False)
    iso_reg.fit(x, y)
    if max_clone_size is not None:
        xx = np.linspace(0, max_clone_size, 100)
    else:
        xx = np.linspace(np.min(x), np.max(x), 100)
    yy = iso_reg.predict(xx)

    # Plot the data points
    plt.scatter(x, y, color=color, alpha=alpha*0.6, s=10)
    # Plot the isotonic regression line
    plt.plot(xx, yy, linestyle=linestyle, label=label, color=color, alpha=alpha, linewidth=2)

def make_diagnostic_plot(df_real, df_sim, perturbation, seed, output_file):
    df_real['Total'] = df_real['Num Progenitor'] + df_real['Num Effector']
    df_real['Log2 Clone Size'] = np.log2(df_real['Total'].clip(lower=1))
    df_real['Proportion of Progenitors'] = df_real['Num Progenitor'] / df_real['Total'].clip(lower=1)
    print(df_real)

    real_max = df_real["Log2 Clone Size"].max()
    max_clone_size = 0.9 * real_max if real_max > 0 else 1.0

    plt.figure(figsize=(15, 6))

    # ECDF Plot
    plt.subplot(1, 2, 1)
    plot_ecdf(list(df_real['Log2 Clone Size']), 'Real ECDF', linestyle='-', color='blue', max_clone_size=max_clone_size)
    plot_ecdf(list(df_sim['Log2 Clone Size']), 'Estimated Params', linestyle='--', color='gray', alpha=0.8, max_clone_size=max_clone_size)
    plt.title(f'ECDF - {perturbation}, Seed {seed}')
    plt.xlabel('Log2 Clone Size')
    plt.ylabel('ECDF')
    plt.xlim([None, max_clone_size])
    plt.legend()

    # Isotonic Regression Plot
    plt.subplot(1, 2, 2)
    plot_isotonic(list(df_real['Log2 Clone Size']), list(df_real['Proportion of Progenitors']),
                  'Real Isotonic', linestyle='-', color='blue', max_clone_size=max_clone_size)
    plot_isotonic(list(df_sim['Log2 Clone Size']), list(df_sim['Proportion of Progenitors']),
                  'Estimated Params', linestyle='--', color='gray', alpha=0.8, max_clone_size=max_clone_size)
    plt.title(f'Isotonic Regression - {perturbation}, Seed {seed}')
    plt.xlabel('Log2 Clone Size')
    plt.ylabel('Proportion of Progenitors')
    plt.xlim([None, max_clone_size])
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    main()
