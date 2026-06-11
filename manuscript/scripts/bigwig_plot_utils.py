#!/usr/bin/env python
"""
Utility functions for creating gene locus plots with bigwig tracks, ATAC-seq coverage, and expression data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# Ensure Arial font is used (required for manuscript figures)
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # Use only Arial, not DejaVu Sans
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text
import os
from functools import lru_cache
from scipy.ndimage import gaussian_filter1d
try:
    import pyBigWig
    PYBIGWIG_AVAILABLE = True
except ImportError:
    PYBIGWIG_AVAILABLE = False
    print("Warning: pyBigWig not available. BigWig tracks will be simulated.")

@lru_cache()
def read_gtf(gtf_file, chrm, start, end):
    """
    Read GTF file and extract features within specified genomic region.

    Parameters
    ----------
    gtf_file : str
        Path to GTF file
    chrm : str
        Chromosome (e.g., 'chr7')
    start : int
        Start position
    end : int
        End position

    Returns
    -------
    pd.DataFrame
        GTF entries within the region
    """
    gtf_df = pd.read_csv(
        gtf_file,
        sep='\t',
        comment='#',
        header=None,
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'],
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )

    # Keep only numbered chromosomes for analysis
    numbered_chromosomes = {'chr' + str(i) for i in range(1, 20)}
    gtf_df = gtf_df[gtf_df['seqname'].isin(numbered_chromosomes)]

    # Extract gene names and filter the dataframe by the specified region
    gtf_df['gene_name'] = gtf_df['attribute'].str.extract('gene_name "([^"]+)"')
    gtf_df.rename(columns={'gene_name': 'gene_id'}, inplace=True)

    region_df = gtf_df[(gtf_df['seqname'] == chrm) & (gtf_df['start'] <= end) & (gtf_df['end'] >= start)]
    return region_df


def make_annotation_plot(ax, gtf_file, chrm, start, end):
    """
    Create gene annotation track showing genes, exons and strand information.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    gtf_file : str
        Path to GTF file
    chrm : str
        Chromosome
    start : int
        Start position
    end : int
        End position
    """
    region_df = read_gtf(gtf_file, chrm, start, end)

    transcripts = {}
    for _, row in region_df.iterrows():
        gene = row['gene_id']
        color = 'skyblue' if row['strand'] == '+' else 'lightcoral'

        if row['feature'] == 'gene':
            ax.add_patch(patches.Rectangle((row['start'], 0.4 if row['strand'] == '+' else -0.7),
                                           row['end'] - row['start'], 0.3, color=color))
        elif row['feature'] == 'exon':
            ax.add_patch(patches.Rectangle((row['start'], 0.1 if row['strand'] == '+' else -1),
                                           row['end'] - row['start'], 0.9, color=color))

        if gene not in transcripts:
            transcripts[gene] = (row['start'], row['end'], row['strand'])
        else:
            transcripts[gene] = (min(transcripts[gene][0], row['start']),
                                 max(transcripts[gene][1], row['end']), row['strand'])

    for gene, details in transcripts.items():
        text_pos = 1.2 if details[2] == '+' else -1.6
        position = details[0]
        if position < start:
            position = start
        if position > end:
            position = end
        ax.text(position + 10, text_pos, gene, fontsize=15, ha='left', va='bottom',
                style='italic', fontfamily='sans-serif')

    # Label the strands
    ax.text(start, 0.5, 'Positive strand', fontsize=20, color='skyblue', ha='right', va='center')
    ax.text(start, -0.5, 'Negative strand', fontsize=20, color='lightcoral', ha='right', va='center')

    ax.set_xlim((start, end))
    ax.set_ylim((-2.5, 2))
    ax.axis('off')


def make_peak_annotation_plot(ax, peak_bed_file, chrm, start, end):
    """
    Create peak annotation track showing ATAC-seq peaks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    peak_bed_file : str
        Path to BED file with peaks
    chrm : str
        Chromosome
    start : int
        Start position
    end : int
        End position
    """
    # Read the BED file, assuming it's tab-delimited with no header
    peaks_df = pd.read_csv(peak_bed_file, sep='\t', names=['chr', 'start', 'end'])

    # Filter for the specified chromosome and within the given range
    peaks_df = peaks_df[(peaks_df['chr'] == chrm) & (peaks_df['start'] < end) & (peaks_df['end'] > start)]

    # Plot each peak as a rectangle
    for _, row in peaks_df.iterrows():
        peak_start = max(row['start'], start)
        peak_end = min(row['end'], end)
        ax.add_patch(patches.Rectangle((peak_start, -0.5), peak_end - peak_start, 1.0, color='purple'))

    # Setting up the axis properties
    ax.set_xlim((start, end))
    ax.set_ylim((-1, 1))
    ax.axis('off')  # Turn off all spines, ticks, and labels
    ax.text(start, 0, 'Peak', fontsize=20, color='purple', ha='right', va='center')


def process_coverage_data(insertion_file_dir, chrm, start, end, CBCs, peak_smoothing):
    """
    Process ATAC-seq insertion data to generate coverage track.

    Parameters
    ----------
    insertion_file_dir : str
        Directory containing insertion files
    chrm : str
        Chromosome
    start : int
        Start position
    end : int
        End position
    CBCs : list
        List of cell barcodes to include
    peak_smoothing : int
        Gaussian smoothing sigma parameter

    Returns
    -------
    np.ndarray
        Smoothed coverage values
    """
    insertion_df = pd.read_csv(os.path.join(insertion_file_dir, f"{chrm}.tsv"), sep='\t')
    insertion_df = insertion_df[(insertion_df['chr'] == chrm) &
                                (insertion_df['position'] >= start) &
                                (insertion_df['position'] <= end) &
                                (insertion_df['cell_barcode'].isin(CBCs))]

    counts = np.zeros(end - start + 1)
    for _, row in insertion_df.iterrows():
        position_index = row['position'] - start
        if 0 <= position_index < len(counts):
            counts[position_index] += row['norm_factor']

    smoothed_counts = gaussian_filter1d(counts, sigma=peak_smoothing)
    smoothed_counts = smoothed_counts / len(CBCs)  # Normalize by the number of CBCs

    return smoothed_counts


def make_coverage_plots(axs, smoothed_data, start, end, chrm, labels, color_labels,
                        label_rotation=90, label_ha='center', label_va='bottom',
                        labelpad=None):
    """
    Create ATAC-seq coverage plots.

    Parameters
    ----------
    axs : list of matplotlib.axes.Axes
        List of axes to plot on
    smoothed_data : list of np.ndarray
        Coverage data for each group
    start : int
        Start position
    end : int
        End position
    chrm : str
        Chromosome
    labels : list
        Group labels
    color_labels : list
        Colors for each group
    label_rotation : float, optional
        Rotation angle for coverage row labels.
    label_ha : str, optional
        Horizontal alignment for coverage row labels.
    label_va : str, optional
        Vertical alignment for coverage row labels.
    labelpad : float or None, optional
        Padding between the axis and the coverage row labels.
    """
    # Determine the global min and max for consistent y-axis scale
    global_max = max(data.max() for data in smoothed_data)

    for i, (ax, data, label) in enumerate(zip(axs, smoothed_data, labels)):
        # Plot with specified color
        ax.plot(np.arange(start, end + 1) / 1e6, data, color=color_labels[i])
        ax.set_xlim(start / 1e6, end / 1e6)
        ax.set_ylim(0, global_max*1.1)  # Set consistent y-axis scale
        ax.set_xlabel(f'{chrm} (Mb)')
        ax.set_ylabel(
            f'{label}',
            rotation=label_rotation,
            ha=label_ha,
            va=label_va,
            labelpad=labelpad,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Enable grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        if i != len(labels) - 1:
            ax.set_xlabel("")


def make_expression_plots(axs, rna_adata, CBCs_groups, gene, color_labels):
    """
    Generate violin plots for gene expression across cell groups.

    Parameters
    ----------
    axs : list of matplotlib.axes.Axes
        List of axes to draw the violin plots on
    rna_adata : AnnData
        The AnnData object containing the RNA-seq data
    CBCs_groups : list of list
        List of lists of cell barcodes, each list corresponding to a group
    gene : str
        The gene to plot expression data for
    color_labels : list
        A list of colors for each group
    """
    import seaborn as sns

    num_groups = len(CBCs_groups)
    max_expression = 0  # Initialize max expression value

    if num_groups == 1:
        axs = [axs]  # Ensure axs is iterable even if it's a single Axes object

    # Calculate maximum expression value across all groups
    for CBCs in CBCs_groups:
        subset = rna_adata[rna_adata.obs_names.isin(CBCs), gene]
        data = subset.layers['norm'].flatten()
        max_expression = max(max_expression, data.max())

    for i, (ax, CBCs) in enumerate(zip(axs, CBCs_groups)):
        subset = rna_adata[rna_adata.obs_names.isin(CBCs), gene]
        data = subset.layers['norm'].flatten()
        df = pd.DataFrame(data, columns=['Expression'])
        df['Group'] = "Group {}".format(i + 1)  # Update label appropriately

        # Create violin plot with color specification and less smoothing
        sns.violinplot(ax=ax, x='Expression', y='Group', data=df, orient='horizontal',
                       cut=0,  # Don't extend past datapoints
                       scale='width',  # Scale each violin to have the same maximum width
                       inner='box',  # Show box plot inside violin
                       bw=0.1,  # Reduce bandwidth for less smoothing
                       color=color_labels[i])  # Color as per color_labels

        ax.set_xlim(0, max_expression)  # Set x-axis limits based on maximum expression value
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Clean up tick marks and labels
        if i < num_groups - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', length=0)
    plt.xlabel(f"{gene} normalized expr")
    plt.tight_layout()


def read_bw_region(bw_path, chrm, start, end, bin_size=50):
    """
    Read BigWig file and extract binned signal values for a region.

    Parameters
    ----------
    bw_path : str
        Path to BigWig file
    chrm : str
        Chromosome
    start : int
        Start position
    end : int
        End position
    bin_size : int
        Size of bins for averaging signal

    Returns
    -------
    tuple
        (x_positions_in_Mb, signal_values)
    """
    if not PYBIGWIG_AVAILABLE:
        # Create simulated data when pyBigWig is not available
        print(f"Simulating BigWig data for {bw_path}")
        region_size = end - start
        n_points = region_size // bin_size if bin_size > 1 else region_size
        x = np.linspace(start, end, n_points) / 1e6

        # Create cleaner simulated signal with defined peaks
        signal = np.zeros(n_points)

        # Add a few smooth peaks at specific positions
        peak_positions = [0.2, 0.35, 0.6, 0.8]  # Relative positions
        peak_heights = [3.0, 5.0, 2.5, 4.0]
        peak_widths = [0.05, 0.03, 0.04, 0.06]

        for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
            peak_center = int(pos * n_points)
            peak_std = int(width * n_points)
            for i in range(max(0, peak_center - peak_std * 3),
                          min(n_points, peak_center + peak_std * 3)):
                dist = abs(i - peak_center) / peak_std
                signal[i] += height * np.exp(-0.5 * dist**2)

        # Add very slight baseline
        signal = signal + 0.1

        # Smooth the signal slightly
        from scipy.ndimage import gaussian_filter1d
        signal = gaussian_filter1d(signal, sigma=1)

        # Ensure non-negative and starts near 0
        signal = np.maximum(signal, 0)
        signal = signal - signal.min()  # Start from 0

        return x, signal

    bw = pyBigWig.open(bw_path)
    signal = np.nan_to_num(bw.values(chrm, start, end, numpy=True), nan=0.0)
    signal[signal < 0] = 0  # LR files: plot only ≥0

    if bin_size > 1:
        pad = (-len(signal)) % bin_size
        if pad:  # pad so we can reshape
            signal = np.pad(signal, (0, pad))
        signal = signal.reshape(-1, bin_size).mean(axis=1)
        x = np.linspace(start, end, signal.size) / 1e6
    else:
        x = np.arange(start, end) / 1e6

    bw.close()
    return x, signal


def make_plot_with_bigwig(chrm, start, end,
                          insertion_file_dir, gtf_file, peak_bed_file,
                          rna_adata, gene,
                          labels, CBCs, color_labels,
                          peak_smoothing,
                          figsize,
                          bw_files, bw_labels, bw_colors,
                          bw_bin_size=50,
                          coverage_label_rotation=90,
                          coverage_label_ha='center',
                          coverage_label_va='bottom',
                          coverage_labelpad=None):
    """
    Create comprehensive gene locus plot with BigWig tracks, ATAC coverage, and expression.

    Parameters
    ----------
    chrm : str
        Chromosome
    start : int
        Start position
    end : int
        End position
    insertion_file_dir : str
        Directory with insertion files
    gtf_file : str
        Path to GTF annotation file
    peak_bed_file : str
        Path to peak BED file
    rna_adata : AnnData
        RNA expression data
    gene : str
        Gene name to show expression for
    labels : list
        Cell group labels
    CBCs : list of lists
        Cell barcodes for each group
    color_labels : list
        Colors for each group
    peak_smoothing : int
        Smoothing parameter for coverage
    figsize : tuple
        Figure size (width, height)
    bw_files : list
        List of BigWig file paths
    bw_labels : list
        Labels for BigWig tracks
    bw_colors : list
        Colors for BigWig tracks
    bw_bin_size : int
        Bin size for BigWig averaging
    coverage_label_rotation : float, optional
        Rotation angle for ATAC coverage row labels.
    coverage_label_ha : str, optional
        Horizontal alignment for ATAC coverage row labels.
    coverage_label_va : str, optional
        Vertical alignment for ATAC coverage row labels.
    coverage_labelpad : float or None, optional
        Padding between axes and ATAC coverage row labels.

    Returns
    -------
    matplotlib.figure.Figure
        The complete figure
    """
    n_bw = len(bw_files)
    n_cov = len(CBCs)

    # heights: annotation, peak, <n_bw tracks>, <n_cov cov-plots>, expression spacer
    height_ratios = [0.4, 0.05] + [0.30]*n_bw + [0.6]*n_cov + [0.20]
    gs_rows = 2 + n_bw + n_cov + 1  # +1 for expression spacer

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(gs_rows, 2, width_ratios=[3, 0.2],
                          height_ratios=height_ratios)

    ax_annotation = plt.subplot(gs[0, 0])
    ax_peaks = plt.subplot(gs[1, 0])
    ax_bw = [plt.subplot(gs[2+i, 0]) for i in range(n_bw)]
    ax_cov = [plt.subplot(gs[2+n_bw+i, 0]) for i in range(n_cov)]
    ax_expr = [plt.subplot(gs[2+n_bw+i, 1]) for i in range(n_cov)]

    # 1. gene & peak annotation
    make_annotation_plot(ax_annotation, gtf_file, chrm, start, end)
    make_peak_annotation_plot(ax_peaks, peak_bed_file, chrm, start, end)

    # tidy shared x tick labels
    for ax in [ax_peaks] + ax_bw + ax_cov[:-1]:
        ax.tick_params(axis='x', length=0, labelbottom=False)
    for ax in ax_expr[:-1]:
        ax.tick_params(axis='x', length=0, labelbottom=False)

    # 2. BigWig tracks (each in its own axis for clarity)
    for ax, bw_path, lab, col in zip(ax_bw, bw_files, bw_labels, bw_colors):
        x, y = read_bw_region(bw_path, chrm, start, end, bw_bin_size)
        y_max = y.max() if y.max() > 0 else 1
        ax.plot(x, y, color=col, lw=1)
        ax.set_xlim(start/1e6, end/1e6)
        ax.set_ylim(0, y_max*1.05)

        # Horizontal y-axis label
        ax.set_ylabel(lab,
                      rotation=0,  # horizontal
                      ha='right',
                      va='center',
                      fontsize=plt.rcParams['axes.labelsize'])  # = 20 in notebook

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 3. ATAC coverage
    cov_data = [process_coverage_data(insertion_file_dir, chrm, start, end,
                                      cbc, peak_smoothing) for cbc in CBCs]
    make_coverage_plots(
        ax_cov,
        cov_data,
        start,
        end,
        chrm,
        labels,
        color_labels,
        label_rotation=coverage_label_rotation,
        label_ha=coverage_label_ha,
        label_va=coverage_label_va,
        labelpad=coverage_labelpad,
    )

    # 4. Expression violins
    make_expression_plots(ax_expr, rna_adata, CBCs, gene, color_labels)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    return fig
