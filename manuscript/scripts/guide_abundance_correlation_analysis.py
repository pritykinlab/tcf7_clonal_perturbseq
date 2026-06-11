#!/usr/bin/env python
"""
Generate scatter plots of initial guide abundance vs number of cells/clones
with statistical analysis for multiple datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, nbinom
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import get_adata, perturbation_colors

# Set figure parameters - scaled by 1.5x then 0.9x
plt.rcParams["axes.titlesize"] = 33.75  # 25 * 1.5 * 0.9
plt.rcParams["axes.labelsize"] = 33.75  # 25 * 1.5 * 0.9
plt.rcParams["xtick.labelsize"] = 27  # 20 * 1.5 * 0.9
plt.rcParams["ytick.labelsize"] = 27  # 20 * 1.5 * 0.9
plt.rcParams["lines.markersize"] = 6.75  # 5 * 1.5 * 0.9
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.figsize'] = (7, 7)

# Guide label mapping from long to short names
guide_label_mapping = {
    'NO_SITE_26': 'NO_SITE-1',
    'NO_SITE_80': 'NO_SITE-2',
    'NO_SITE_223': 'NO_SITE-3',
    'NO_SITE_913': 'NO_SITE-4',
    'NO_SITE_738': 'NO_SITE-5',
    'ONE_INTERGENIC_SITE_1': 'INTERGENIC-1',
    'ONE_INTERGENIC_SITE_21': 'INTERGENIC-2',
    'ONE_INTERGENIC_SITE_5': 'INTERGENIC-3',
    'ONE_INTERGENIC_SITE_24': 'INTERGENIC-4',
    'ONE_INTERGENIC_SITE_18': 'INTERGENIC-5',
    'NO_SITE_5': 'NO_SITE-6',
    'NO_SITE_3': 'NO_SITE-7',
    'NO_SITE_1': 'NO_SITE-8',
    'NO_SITE_2': 'NO_SITE-9',
    'NO_SITE_4': 'NO_SITE-10',
    'NO_SITE_6': 'NO_SITE-11',
}

def convert_long_to_short(long_label):
    """Convert long guide label to short form"""
    return guide_label_mapping.get(long_label, long_label)


def infer_perturbation_from_guide(guide_name):
    short_label = convert_long_to_short(guide_name)
    if short_label.startswith('Trim28'):
        return 'Trim28'
    if short_label.startswith('Setdb1'):
        return 'Setdb1'
    if short_label.startswith('NO_SITE') or short_label.startswith('INTERGENIC'):
        return 'NT'
    return None


def format_numeric_nt_label(guide_name):
    short_label = convert_long_to_short(guide_name)
    if short_label.startswith('NO_SITE-'):
        return short_label.split('-')[-1]
    return short_label


def format_numeric_suffix_label(guide_name):
    short_label = convert_long_to_short(guide_name)
    if '-' in short_label:
        return short_label.split('-')[-1]
    return short_label

def get_data_df(adata, guide_conc_dict, analysis_type='cells', guide_names=None):
    """
    Build DataFrame with guide abundance and cell/clone counts

    Parameters:
    -----------
    adata : AnnData object
    guide_conc_dict : dict mapping guide names to concentrations
    analysis_type : str, either 'cells' or 'clones'
    """
    # Convert to str to avoid category issues
    adata.obs['guide'] = adata.obs['guide'].astype(str)

    # Get all guides to report, including guides with zero observations in a subsetted adata.
    if guide_names is None:
        all_guides = np.unique(adata.obs['guide'])
    else:
        all_guides = np.array(list(dict.fromkeys(guide_names)))

    # Build records
    data_records = []
    for g in all_guides:
        abundance = guide_conc_dict.get(g, 0)
        data_records.append({'Guide': g, 'Concentration': abundance})

    df = pd.DataFrame(data_records)

    if analysis_type == 'cells':
        # Count cells per guide
        guide_cell_counts = {}
        for guide in adata.obs['guide']:
            g_part = guide.split('__')[0] if '__' in guide else guide
            guide_cell_counts[g_part] = guide_cell_counts.get(g_part, 0) + 1
        df['count'] = df['Guide'].apply(lambda g: guide_cell_counts.get(g, 0))
    else:  # clones
        # Count clones per guide (unique all_constructs_str values)
        guide_clone_counts = {}
        for clone in np.unique(adata.obs['all_constructs_str']):
            # Extract guide part from clone identifier
            g_part = clone.split('__')[0]
            guide_clone_counts[g_part] = guide_clone_counts.get(g_part, 0) + 1
        df['count'] = df['Guide'].apply(lambda g: guide_clone_counts.get(g, 0))

    df['is_control_guide'] = df['Guide'].apply(lambda x: ('NO_SITE' in x) or ('INTERGENIC' in x))

    return df

def perform_analysis(adata, guide_conc_dict, dataset_name, analysis_type,
                     figure_num, panel_letter, table_suffix=None, remove_guides=None,
                     x_scale_factor=1.0, x_axis_label=None, xtick_values=None,
                     xtick_labels=None, output_dir=None, highlight_guides=None,
                     label_ko_guides=True, ko_color_mode='single_red',
                     control_color=None, ko_label_formatter=None,
                     control_label_formatter=None, y_axis_label=None,
                     guide_names=None, fig_size=None):
    """
    Perform the complete analysis for a dataset and analysis type

    Parameters:
    -----------
    adata : AnnData object
    guide_conc_dict : dict mapping guide names to concentrations
    dataset_name : str, name of dataset (e.g., 'gene10', 'trim28_setdb1')
    analysis_type : str, either 'cells' or 'clones'
    figure_num : str, figure number (e.g., '4', '5')
    panel_letter : str, panel letter (e.g., 'a', 'b', 'k')
    table_suffix : str, suffix for supplementary table (e.g., '4_1', '4_2', '5_1')
    remove_guides : list of guide names to remove (optional)
    x_scale_factor : float, divisor applied to x-axis values for plotting (default 1.0)
    x_axis_label : str, optional label for the x-axis
    xtick_values : list, optional custom x-axis tick positions
    xtick_labels : list, optional custom x-axis tick labels
    label_ko_guides : bool, whether to label non-control guides on the scatter plot
    ko_color_mode : str, either 'single_red' or 'by_perturbation'
    control_color : optional color for NT guides, fit line, and CI
    ko_label_formatter : optional callable mapping guide name to label for KO guides
    control_label_formatter : optional callable mapping guide name to label for NT guides
    y_axis_label : str, optional custom y-axis label
    guide_names : sequence of guide names to include, even if they have zero counts
    fig_size : tuple[float, float], optional matplotlib figure size
    """

    # Create output directories
    if output_dir is None:
        output_dir = f"../manuscript/plots/Figure{figure_num}"
    os.makedirs(output_dir, exist_ok=True)
    table_dir = "../manuscript/supplementary_tables"
    os.makedirs(table_dir, exist_ok=True)

    # Remove specified guides if any
    if remove_guides:
        print(f"Removing guides: {remove_guides}")
        mask = ~adata.obs['guide'].isin(remove_guides)
        adata = adata[mask].copy()

    # Build data DataFrame
    data_df = get_data_df(adata, guide_conc_dict, analysis_type, guide_names=guide_names)

    # Separate control vs KO guides
    df_control = data_df[data_df['is_control_guide']]
    df_ko = data_df[~data_df['is_control_guide']]

    # Fit negative binomial model to control guides
    print(f"Fitting negative binomial model to control guides for {analysis_type}...")
    obs_control = df_control['count']

    def neg_ll_nb(params, obs, x_conc):
        r, s = params
        mu = x_conc * s
        p = r / (r + mu + 1e-10)  # Add small constant to avoid division by zero
        return -np.sum(nbinom.logpmf(obs, r, p))

    r_init = 1.0
    s_init = obs_control.mean() / df_control['Concentration'].mean() if df_control['Concentration'].mean() > 0 else 1.0
    res = minimize(
        neg_ll_nb,
        [r_init, s_init],
        args=(obs_control, df_control['Concentration']),
        bounds=[(0.001, None), (0.001, None)],
        method='L-BFGS-B'
    )
    r_star, s_star = res.x
    alpha_nb = 1 / r_star
    slope = s_star

    # Calculate correlations
    rho_spearman, _ = spearmanr(df_control['Concentration'], df_control['count'])
    r_pearson, _ = pearsonr(df_control['Concentration'], df_control['count'])

    # Finalize axis label and scaling parameters
    if not x_scale_factor:
        x_scale_factor = 1.0
    if x_axis_label is None:
        x_axis_label = "initial guide abundance (reads)"

    # Calculate standardized residuals and NB-based p-values for statistical testing (used for plot text)
    print("\nCalculating standardized residuals and p-values...")

    # For control guides
    mu_control = df_control['Concentration'] * slope
    stdev_control = np.sqrt(mu_control + alpha_nb * (mu_control ** 2))
    z_control = (df_control['count'] - mu_control) / (stdev_control + 1e-10)

    # For KO guides
    mu_ko = df_ko['Concentration'] * slope
    stdev_ko = np.sqrt(mu_ko + alpha_nb * (mu_ko ** 2))
    z_ko = (df_ko['count'] - mu_ko) / (stdev_ko + 1e-10)

    # Combine all guides for p-value calculation
    z_all = pd.concat([z_control, z_ko], ignore_index=True)
    guide_all = pd.concat([df_control['Guide'], df_ko['Guide']], ignore_index=True)
    counts_all = pd.concat([df_control['count'], df_ko['count']], ignore_index=True)
    mu_all = pd.concat([mu_control, mu_ko], ignore_index=True)

    # Two-sided p-values directly from the fitted negative binomial
    def nb_two_sided_pvalue(counts, mus):
        p_param = r_star / (r_star + mus + 1e-10)
        lower_tail = nbinom.cdf(counts, r_star, p_param)
        upper_tail = nbinom.sf(counts - 1, r_star, p_param)  # inclusive of observed count
        return np.clip(2 * np.minimum(lower_tail, upper_tail), 0, 1)

    pvals_nb = nb_two_sided_pvalue(counts_all, mu_all)

    # Apply Benjamini-Hochberg correction
    _, pvals_corr, _, _ = multipletests(pvals_nb, method='fdr_bh')

    # Create results DataFrame with NB-based p-values
    res_df = pd.DataFrame({
        'guide': guide_all.values,
        'count': counts_all.values,
        'expected_mean': mu_all.values,
        'z_score': z_all.values,
        'pvalue': pvals_nb,
        'bh_adj_pvalue': pvals_corr
    }).sort_values('bh_adj_pvalue')

    # Count BH-significant KOs for annotation
    is_control_all = guide_all.str.contains('NO_SITE') | guide_all.str.contains('INTERGENIC')
    ko_bh_sig_count = int(((~is_control_all) & (pvals_corr < 0.05)).sum())

    # Create visualization
    print(f"Creating visualization for Figure {figure_num}{panel_letter}...")
    fig, ax = plt.subplots(figsize=fig_size or (8, 7))

    highlight_guides = set(highlight_guides or [])
    if control_color is None:
        control_color = perturbation_colors.get('NT', 'grey')

    if control_label_formatter is None:
        control_label_formatter = convert_long_to_short
    if ko_label_formatter is None:
        ko_label_formatter = convert_long_to_short

    # Scatter plot
    ax.scatter(df_control['Concentration'] / x_scale_factor, df_control['count'],
               color=control_color, s=50, alpha=0.7, zorder=2)

    if ko_color_mode == 'by_perturbation':
        ko_colors = df_ko['Guide'].map(
            lambda g: perturbation_colors.get(infer_perturbation_from_guide(g), '#d62728')
        )
    else:
        ko_colors = '#d62728'

    ax.scatter(df_ko['Concentration'] / x_scale_factor, df_ko['count'],
               color=ko_colors, s=50, alpha=0.7, zorder=2)

    # Label each point with short name - scaled by 2x then 0.9x
    for _, row in df_control.iterrows():
        short_label = control_label_formatter(row['Guide'])
        text_kwargs = {}
        if row['Guide'] in highlight_guides or short_label in highlight_guides:
            text_kwargs["bbox"] = dict(
                facecolor='none',
                edgecolor='red',
                linewidth=2,
                boxstyle='square,pad=0.15'
            )
        ax.text(row['Concentration'] / x_scale_factor, row['count'], short_label,
                color=control_color, fontsize=14.4, ha='left', va='bottom',
                **text_kwargs)  # 8 * 2 * 0.9 = 14.4

    if label_ko_guides:
        for _, row in df_ko.iterrows():
            short_label = ko_label_formatter(row['Guide'])
            text_kwargs = {}
            if row['Guide'] in highlight_guides or short_label in highlight_guides:
                text_kwargs["bbox"] = dict(
                    facecolor='none',
                    edgecolor='red',
                    linewidth=2,
                    boxstyle='square,pad=0.15'
                )
            if ko_color_mode == 'by_perturbation':
                ko_text_color = perturbation_colors.get(
                    infer_perturbation_from_guide(row['Guide']),
                    '#d62728',
                )
            else:
                ko_text_color = '#d62728'
            ax.text(row['Concentration'] / x_scale_factor, row['count'], short_label,
                    color=ko_text_color, fontsize=14.4, ha='left', va='bottom',
                    **text_kwargs)  # 8 * 2 * 0.9 = 14.4

    # Best-fit line and confidence interval bands (NB predictive 95%)
    x_line_physical = np.linspace(0, data_df['Concentration'].max() * 1.1, 200)
    y_line = slope * x_line_physical
    x_line = x_line_physical / x_scale_factor

    # Plot regression line
    ax.plot(x_line, y_line, color=control_color, linewidth=2, zorder=1)

    # Calculate and plot central 95% negative binomial predictive interval
    p_line = r_star / (r_star + y_line + 1e-10)
    y_lower = nbinom.ppf(0.025, r_star, p_line)
    y_upper = nbinom.ppf(0.975, r_star, p_line)
    ax.fill_between(
        x_line,
        np.clip(y_lower, 0, None),  # Ensure non-negative
        y_upper,
        color=control_color,
        alpha=0.3,
        zorder=0
    )

    # Labels and formatting
    ax.set_xlabel(x_axis_label)
    y_label = y_axis_label or ("number of cells" if analysis_type == 'cells' else "number of clones")
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if xtick_values is not None:
        ax.set_xticks(xtick_values)
        if xtick_labels is None:
            xtick_labels = [str(v) for v in xtick_values]
    if xtick_labels is not None:
        ax.set_xticklabels([str(label) for label in xtick_labels])

    # Add correlation text inside the figure (top-left corner to avoid data points) - changed title as requested
    correlation_text = (
        f"Correlation across control guides\n"
        f"\u00A0\u00A0\u00A0\u00A0Spearman \u03c1={rho_spearman:.2f}\n"
        f"\u00A0\u00A0\u00A0\u00A0Pearson r={r_pearson:.2f}\n"
        f"BH-significant gRNAs={ko_bh_sig_count}"
    )
    ax.text(0.05, 0.95, correlation_text,
            transform=ax.transAxes, fontsize=24.3, verticalalignment='top')  # 18 * 1.5 * 0.9 = 24.3

    # Remove legend if present
    if ax.legend_:
        ax.legend_.remove()

    plt.tight_layout()

    # Save outputs
    print(f"Saving figures for panel {panel_letter}...")
    png_path = os.path.join(output_dir, f"{panel_letter}.png")
    svg_path = os.path.join(output_dir, f"{panel_letter}.svg")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()

    if table_suffix is not None:
        table_path = os.path.join(table_dir, f"{table_suffix}.tsv")
        res_df.to_csv(table_path, sep='\t', index=False, float_format='%.6e')
        print(f"\nSupplementary Table {table_suffix} saved to {table_path}")

    # Print top 10 results
    print(f"\n=== BH-corrected p-values (top 10) for {dataset_name} {analysis_type} ===")
    print(res_df.head(10).to_string(index=True))

    # Print summary statistics
    print(f"\nSummary for {dataset_name} {analysis_type}:")
    print(f"  Control guides: {len(df_control)}")
    print(f"  KO guides: {len(df_ko)}")
    print(f"  Slope: {slope:.2f}")
    print(f"  Alpha (overdispersion): {alpha_nb:.3f}")
    print(f"  Spearman ρ: {rho_spearman:.2f}")
    print(f"  Pearson r: {r_pearson:.2f}")

    return res_df
