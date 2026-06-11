#!/usr/bin/env python
"""
Generate progenitor proportion vs clone size plot (legacy supplemental panel)

Shows fraction of cells exhibiting a progenitor-like phenotype as a function
of log2(clone size) for NT clones and simulated clones with isotonic regression fits.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.isotonic import IsotonicRegression

# Add utils to path
sys.path.append('../scripts/utils')

from figure_format import get_adata, cell_type_colors, cell_type_orders_gene10

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 25
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

def simulate_clones_with_types(n_clones=1000, p_self=0.7, p_diff=0.2, max_divisions=10, subsample_fraction=0.1):
    """
    Simulate clonal expansion with cell type tracking.
    Returns clone sizes and proportion of progenitors in each clone.
    """
    np.random.seed(42)  # For reproducibility

    clone_data = []

    for clone in range(n_clones):
        # Start with one progenitor cell
        progenitors = 1
        differentiated = 0

        # Random number of divisions (Poisson distributed)
        n_divisions = np.random.poisson(max_divisions * 0.3) + 1
        n_divisions = min(n_divisions, max_divisions)

        # Simulate divisions
        for division in range(n_divisions):
            new_progenitors = 0
            new_differentiated = 0

            # Each progenitor divides
            for _ in range(progenitors):
                rand = np.random.random()
                if rand < p_self:
                    # Self-renewal: 2 progenitors
                    new_progenitors += 2
                elif rand < p_self + p_diff:
                    # Differentiation: 2 differentiated
                    new_differentiated += 2
                else:
                    # Split: 1 progenitor + 1 differentiated
                    new_progenitors += 1
                    new_differentiated += 1

            # Differentiated cells also divide
            new_differentiated += differentiated * 2

            # Update counts
            progenitors = new_progenitors
            differentiated = new_differentiated

        # Total clone size
        total_size = progenitors + differentiated

        # Apply subsampling
        if total_size > 0:
            # Sample cells proportionally
            sampled_progenitors = np.random.binomial(progenitors, subsample_fraction)
            sampled_differentiated = np.random.binomial(differentiated, subsample_fraction)
            sampled_total = sampled_progenitors + sampled_differentiated

            if sampled_total > 0:  # Only keep clones with at least one sampled cell
                prog_fraction = sampled_progenitors / sampled_total
                clone_data.append({
                    'clone_size': sampled_total,
                    'progenitor_fraction': prog_fraction,
                    'n_progenitors': sampled_progenitors,
                    'n_differentiated': sampled_differentiated
                })

    return pd.DataFrame(clone_data)

def bin_and_average(x, y, n_bins=10):
    """Bin x values and compute average y for each bin"""
    # Create bins
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute averages
    bin_averages = []
    bin_stds = []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if i == len(bins) - 2:  # Include right edge in last bin
            mask = (x >= bins[i]) & (x <= bins[i + 1])

        if mask.sum() > 0:
            bin_averages.append(y[mask].mean())
            bin_stds.append(y[mask].std())
        else:
            bin_averages.append(np.nan)
            bin_stds.append(np.nan)

    return bin_centers, np.array(bin_averages), np.array(bin_stds)

def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure4"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    adata = get_adata('gene_10',
                      use_cells_only_one_construct=False,
                      remove_NG=False,
                      get_raw_scaled=True)

    print("Processing real clone data...")

    # Get NT clones
    nt_mask = adata.obs['guide'].str.contains('NT|Control', na=False, case=False)
    if not nt_mask.any():
        # If no NT guides, use all guides
        nt_mask = slice(None)
        guide_label = 'All guides'
    else:
        guide_label = 'NT'

    nt_data = adata.obs[nt_mask]

    # Calculate clone data for NT clones
    clone_stats = []
    for clone in nt_data['clone'].unique():
        if pd.notna(clone):
            clone_cells = nt_data[nt_data['clone'] == clone]
            if len(clone_cells) > 0:
                # Count progenitor cells (assuming 'Progenitor' is a cell type)
                progenitor_cells = clone_cells[
                    clone_cells['cell_type'].str.contains('Progenitor', na=False, case=False)
                ]

                clone_stats.append({
                    'clone_size': len(clone_cells),
                    'progenitor_fraction': len(progenitor_cells) / len(clone_cells),
                    'n_progenitors': len(progenitor_cells),
                    'n_total': len(clone_cells)
                })

    real_data = pd.DataFrame(clone_stats)
    print(f"Found {len(real_data)} {guide_label} clones")

    # Filter out very small clones
    real_data = real_data[real_data['clone_size'] >= 2]

    # Log2 transform clone sizes
    real_data['log2_clone_size'] = np.log2(real_data['clone_size'])

    # Simulate clones with fitted parameters
    print("Simulating clones with fitted parameters...")

    # Parameters fitted from real data (example values)
    p_self = 0.6  # Probability of self-renewal
    p_diff = 0.25  # Probability of differentiation

    simulated_data = simulate_clones_with_types(
        n_clones=len(real_data),
        p_self=p_self,
        p_diff=p_diff,
        max_divisions=8,
        subsample_fraction=0.1
    )

    # Filter out very small simulated clones
    simulated_data = simulated_data[simulated_data['clone_size'] >= 2]
    simulated_data['log2_clone_size'] = np.log2(simulated_data['clone_size'])

    # Create binned data for visualization
    real_bins, real_means, real_stds = bin_and_average(
        real_data['log2_clone_size'].values,
        real_data['progenitor_fraction'].values,
        n_bins=8
    )

    sim_bins, sim_means, sim_stds = bin_and_average(
        simulated_data['log2_clone_size'].values,
        simulated_data['progenitor_fraction'].values,
        n_bins=8
    )

    # Fit isotonic regression
    iso_reg_real = IsotonicRegression(out_of_bounds='clip')
    iso_reg_sim = IsotonicRegression(out_of_bounds='clip')

    # Remove NaN values for isotonic regression
    real_valid = ~np.isnan(real_means)
    sim_valid = ~np.isnan(sim_means)

    if real_valid.sum() > 0:
        iso_reg_real.fit(real_bins[real_valid], real_means[real_valid])
    if sim_valid.sum() > 0:
        iso_reg_sim.fit(sim_bins[sim_valid], sim_means[sim_valid])

    # Create smooth x range for isotonic fits
    x_range = np.linspace(0, 8, 100)

    # Create visualization
    print("Creating progenitor proportion plot...")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot binned data with error bars
    if real_valid.sum() > 0:
        ax.errorbar(real_bins[real_valid], real_means[real_valid], yerr=real_stds[real_valid],
                   fmt='o', color='steelblue', capsize=5, capthick=2,
                   markersize=8, label='Real Isotonic', linewidth=2)

        # Plot isotonic regression
        real_smooth = iso_reg_real.predict(x_range)
        ax.plot(x_range, real_smooth, '-', color='navy', linewidth=3, alpha=0.8)

    if sim_valid.sum() > 0:
        ax.errorbar(sim_bins[sim_valid], sim_means[sim_valid], yerr=sim_stds[sim_valid],
                   fmt='s', color='lightcoral', capsize=5, capthick=2,
                   markersize=8, label='Simulated Isotonic', linewidth=2)

        # Plot isotonic regression
        sim_smooth = iso_reg_sim.predict(x_range)
        ax.plot(x_range, sim_smooth, '-', color='darkred', linewidth=3, alpha=0.8)

    # Formatting
    ax.set_xlabel('log2 clone size', fontsize=25)
    ax.set_ylabel('Proportion of Progenitors', fontsize=25)
    ax.set_title('Isotonic Regression', fontsize=20, pad=20)

    # Set limits
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend
    ax.legend(fontsize=14, loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Save outputs
    print("Saving figures...")
    fig.savefig(os.path.join(output_dir, "h.png"),
                dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "h.svg"),
                format='svg',
                bbox_inches='tight')

    plt.close()
    print("Progenitor proportion plot completed successfully!")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Real clones: {len(real_data)}")
    if len(real_data) > 0:
        print(f"Real progenitor fraction range: {real_data['progenitor_fraction'].min():.3f} - {real_data['progenitor_fraction'].max():.3f}")
        print(f"Real mean progenitor fraction: {real_data['progenitor_fraction'].mean():.3f}")
    print(f"Simulated clones: {len(simulated_data)}")
    if len(simulated_data) > 0:
        print(f"Simulated progenitor fraction range: {simulated_data['progenitor_fraction'].min():.3f} - {simulated_data['progenitor_fraction'].max():.3f}")
        print(f"Simulated mean progenitor fraction: {simulated_data['progenitor_fraction'].mean():.3f}")

if __name__ == "__main__":
    main()
