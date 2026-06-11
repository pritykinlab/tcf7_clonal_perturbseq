#!/usr/bin/env python
"""
Generate Figure 6e: Nars2 locus plot in the standard two-group layout.

This uses the Control-versus-TRM-like format and highlights the Nars2-assigned
ATAC peak that was significant in the Figure 6d comparison.
"""

import os
import sys
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # Use only Arial, not DejaVu Sans
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text
# Ensure Arial is used for all text
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # Use proper minus sign

from figure_format import get_adata, cell_type_colors
import DE_testing

# Import bigwig plotting utilities
sys.path.append('../manuscript/scripts')
from bigwig_plot_utils import make_plot_with_bigwig

# Set figure parameters
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 20  # Set to 20 for bigwig labels
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["lines.markersize"] = 5
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.edgecolor'] = 'dimgray'
plt.rcParams['figure.figsize'] = (7, 7)

SIGNIFICANCE_ALPHA = 0.05


def _parse_peak_name(peak_name):
    chrom, start, end = peak_name.split('-')
    return chrom, int(start), int(end)


def _get_nars2_significant_peak(atac_adata):
    nearest_gene_dict = atac_adata.var['gene'].to_dict()
    mask_trm_like = (
        (atac_adata.obs['cell_type'] == 'TRM-like')
        & (atac_adata.obs['condition'] == 'Trim28')
    )
    mask_control_non_trm = (
        (atac_adata.obs['cell_type'] != 'TRM-like')
        & (atac_adata.obs['condition'] == 'Control')
    )

    results_atac_df = DE_testing.wilcoxon_DE_test(
        atac_adata,
        atac_adata.obs_names[mask_trm_like].tolist(),
        atac_adata.obs_names[mask_control_non_trm].tolist(),
        norm_layer='norm'
    )
    results_atac_df['nearest_gene'] = results_atac_df['Gene'].map(nearest_gene_dict)
    nars2_peaks = results_atac_df[
        (results_atac_df['nearest_gene'] == 'Nars2')
        & (results_atac_df['Adjusted P-value'] < SIGNIFICANCE_ALPHA)
    ].sort_values('Adjusted P-value')
    if nars2_peaks.empty:
        raise ValueError("No significant Figure 6d ATAC peak found for Nars2.")
    return _parse_peak_name(nars2_peaks.iloc[0]['Gene'])


def _highlight_peak(fig, chrom, peak_start, peak_end):
    for ax in fig.get_axes():
        x_min, x_max = ax.get_xlim()
        if max(abs(x_min), abs(x_max)) < 20:
            continue
        ax.axvspan(
            peak_start / 1e6,
            peak_end / 1e6,
            color='goldenrod',
            alpha=0.15,
            linewidth=0,
            zorder=0,
        )


def main():
    # Create output directory
    output_dir = "../manuscript/plots/Figure6"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    rna_adata = get_adata('multiome_rna')
    atac_adata = get_adata('multiome_atac')

    # Define file paths
    gtf_file = "../data/misc/gencode.vM10.basic.annotation.gtf.gz"
    insertion_file_dir = "../data/multiome/insertion_files"
    peak_bed_file = "../data/multiome/analysis_using_atlas_v2/final_peak_set.bed"
    chip_bw_path = "../data/multiome/Trim28_study/chip_trim28.bw"
    cut_tag_bw_path = "../data/multiome/Trim28_study/cut_tag_Trim28.bw"

    # Set up cell groups and labels
    print("Setting up cell groups...")
    labels = ['Control', 'TRM-like']

    # Fetch cell barcodes for each group
    cbc_control = rna_adata[rna_adata.obs['condition'] == 'Control'].obs_names
    cbc_trm_like = rna_adata[rna_adata.obs['cell_type'] == 'TRM-like'].obs_names

    # Collect cell barcodes in the specified order
    CBCs = list(map(list, [cbc_control, cbc_trm_like]))

    # Set color labels
    control_color = 'black'
    color_labels = [control_color, cell_type_colors['TRM-like']]

    # BigWig track settings
    bw_files = [chip_bw_path, cut_tag_bw_path]
    bw_labels = ['Trim28\nChIP', 'Trim28\nCUT&Tag']
    bw_colors = ['black', 'black']

    # Significant Figure 6d peak used for the highlight
    peak_chrom, peak_start, peak_end = _get_nars2_significant_peak(atac_adata)
    print(
        "Highlighting significant Nars2 peak from Figure 6d: "
        f"{peak_chrom}:{peak_start:,}-{peak_end:,}"
    )

    # Generate the figure for the Nars2 locus
    print("Creating Nars2 locus plot (TRM-like vs Control)...")
    print(f"Using font: {plt.rcParams['font.sans-serif']}")
    fig = make_plot_with_bigwig(
        'chr7', 96_900_000, 97_100_000,
        insertion_file_dir, gtf_file, peak_bed_file,
        rna_adata, 'Nars2',
        labels, CBCs, color_labels,
        peak_smoothing=100,
        figsize=(20 * 1.5, 10 * 1.5 * 0.5),
        bw_files=bw_files,
        bw_labels=bw_labels,
        bw_colors=bw_colors,
        bw_bin_size=1,  # 1-bp resolution for maximum detail
        coverage_label_rotation=0,
        coverage_label_ha='right',
        coverage_label_va='center',
        coverage_labelpad=25,
    )
    _highlight_peak(fig, peak_chrom, peak_start, peak_end)

    # Save outputs
    print("Saving figures...")
    png_path = os.path.join(output_dir, "e.png")
    svg_path = os.path.join(output_dir, "e.svg")

    # For scatter plots with many points, enable rasterization for SVG size reduction
    for ax in fig.get_axes():
        for line in ax.get_lines():
            line.set_rasterized(True)  # Rasterize line plots to reduce SVG size

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    # Validate file sizes (critical requirement)
    png_size = os.path.getsize(png_path) / 1024 / 1024  # MB
    svg_size = os.path.getsize(svg_path) / 1024  # KB

    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")

    plt.close()
    print("Figure 6e completed successfully!")

    # Print summary
    print("\nSummary:")
    print(f"- Generated Nars2 locus plot with {len(CBCs)} cell groups")
    print(f"- Included {len(bw_files)} BigWig tracks (ChIP and CUT&Tag)")
    print(f"- Genomic region: chr7:96,900,000-97,100,000")
    print(f"- Highlighted peak: {peak_chrom}:{peak_start:,}-{peak_end:,}")
    print(f"- Output saved to {output_dir}")


if __name__ == "__main__":
    main()
