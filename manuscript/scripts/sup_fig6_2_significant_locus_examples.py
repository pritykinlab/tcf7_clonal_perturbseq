#!/usr/bin/env python
"""
Generate Figure S14 locus examples for Itga1 and Setdb1.

The displayed regions are anchored on the ATAC peaks assigned to each gene that
were significant in the Figure 6d comparison of Trim28 TRM-like cells versus
non-TRM-like Control cells.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Add utils to path
sys.path.append('../scripts/utils')

# Required font settings for all manuscript figures
plt.rcParams['svg.fonttype'] = 'none'  # Make SVG text editable in Adobe Illustrator
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math text

from figure_format import cell_type_colors, get_adata
import DE_testing

# Import bigwig plotting utilities
sys.path.append('../manuscript/scripts')
from bigwig_plot_utils import make_plot_with_bigwig

# Match the Figure 6e styling
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["axes.titlesize"] = 25
plt.rcParams["axes.labelsize"] = 20
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

OUTPUT_DIR = "../manuscript/plots/Sup_Figure_14"
GTF_FILE = "../data/misc/gencode.vM10.basic.annotation.gtf.gz"
INSERTION_FILE_DIR = "../data/multiome/insertion_files"
PEAK_BED_FILE = "../data/multiome/analysis_using_atlas_v2/final_peak_set.bed"
CHIP_BW_PATH = "../data/multiome/Trim28_study/chip_trim28.bw"
CUT_TAG_BW_PATH = "../data/multiome/Trim28_study/cut_tag_Trim28.bw"
SIGNIFICANCE_ALPHA = 0.05
WINDOW_PADDING_BP = 20_000

PANELS = {
    "a": "Itga1",
    "b": "Setdb1",
}


def parse_peak_name(peak_name):
    chrom, start, end = peak_name.split("-")
    return chrom, int(start), int(end)


def load_gene_coordinates(genes):
    gtf_df = pd.read_csv(
        GTF_FILE,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )
    gtf_df = gtf_df[gtf_df["feature"] == "gene"].copy()
    gtf_df["gene_name"] = gtf_df["attribute"].str.extract('gene_name "([^"]+)"')
    coords = (
        gtf_df[gtf_df["gene_name"].isin(genes)]
        .drop_duplicates(subset=["gene_name"])
        .set_index("gene_name")[["seqname", "start", "end", "strand"]]
    )
    missing = sorted(set(genes) - set(coords.index))
    if missing:
        raise ValueError(f"Missing gene coordinates in GTF for: {', '.join(missing)}")
    return coords


def compute_significant_peak_table(rna_adata, atac_adata):
    nearest_gene_dict = atac_adata.var["gene"].to_dict()

    mask_trm_like_atac = (
        (atac_adata.obs["cell_type"] == "TRM-like")
        & (atac_adata.obs["condition"] == "Trim28")
    )
    mask_control_non_trm_atac = (
        (atac_adata.obs["cell_type"] != "TRM-like")
        & (atac_adata.obs["condition"] == "Control")
    )
    mask_trm_like_rna = (
        (rna_adata.obs["cell_type"] == "TRM-like")
        & (rna_adata.obs["condition"] == "Trim28")
    )
    mask_control_non_trm_rna = (
        (rna_adata.obs["cell_type"] != "TRM-like")
        & (rna_adata.obs["condition"] == "Control")
    )

    print("Performing Figure 6d ATAC differential testing...")
    results_atac_df = DE_testing.wilcoxon_DE_test(
        atac_adata,
        atac_adata.obs_names[mask_trm_like_atac].tolist(),
        atac_adata.obs_names[mask_control_non_trm_atac].tolist(),
        norm_layer="norm",
    )
    results_atac_df["nearest_gene"] = results_atac_df["Gene"].map(nearest_gene_dict)

    print("Performing Figure 6d RNA differential testing...")
    results_rna_df = DE_testing.wilcoxon_DE_test(
        rna_adata,
        rna_adata.obs_names[mask_trm_like_rna].tolist(),
        rna_adata.obs_names[mask_control_non_trm_rna].tolist(),
        norm_layer="norm",
    ).rename(
        columns={
            "Gene": "gene",
            "Log2FC": "rna_log2fc",
            "P-value": "rna_p_value",
            "Adjusted P-value": "rna_adjusted_p_value",
        }
    )

    significant_peaks = results_atac_df[
        results_atac_df["Adjusted P-value"] < SIGNIFICANCE_ALPHA
    ].copy()
    significant_peaks = significant_peaks[
        significant_peaks["nearest_gene"].isin(PANELS.values())
    ].copy()
    significant_peaks["chrom"], significant_peaks["peak_start"], significant_peaks["peak_end"] = zip(
        *significant_peaks["Gene"].map(parse_peak_name)
    )
    significant_peaks = significant_peaks.rename(
        columns={
            "Gene": "peak_id",
            "nearest_gene": "gene",
            "Log2FC": "peak_log2fc",
            "P-value": "peak_p_value",
            "Adjusted P-value": "peak_adjusted_p_value",
        }
    )

    peak_counts = significant_peaks.groupby("gene").size().rename("n_significant_peaks")
    significant_peaks = significant_peaks.merge(
        results_rna_df[["gene", "rna_log2fc", "rna_p_value", "rna_adjusted_p_value"]],
        on="gene",
        how="left",
    )
    significant_peaks = significant_peaks.merge(
        peak_counts,
        on="gene",
        how="left",
    )
    significant_peaks["panel"] = significant_peaks["gene"].map(
        {gene: panel for panel, gene in PANELS.items()}
    )

    return significant_peaks.sort_values(["panel", "peak_adjusted_p_value", "peak_id"])


def build_cell_groups(rna_adata):
    labels = ["Control", "TRM-like"]
    cbcs = list(
        map(
            list,
            [
                rna_adata[rna_adata.obs["condition"] == "Control"].obs_names,
                rna_adata[rna_adata.obs["cell_type"] == "TRM-like"].obs_names,
            ],
        )
    )
    colors = ["black", cell_type_colors["TRM-like"]]
    return labels, cbcs, colors


def add_peak_highlights(fig, peak_rows):
    for ax in fig.get_axes():
        x_min, x_max = ax.get_xlim()
        if max(abs(x_min), abs(x_max)) < 20:
            continue
        for _, peak_row in peak_rows.iterrows():
            ax.axvspan(
                peak_row["peak_start"] / 1e6,
                peak_row["peak_end"] / 1e6,
                color="goldenrod",
                alpha=0.15,
                linewidth=0,
                zorder=0,
            )


def rasterize_lines(fig):
    for ax in fig.get_axes():
        for line in ax.get_lines():
            line.set_rasterized(True)


def validate_file_sizes(png_path, svg_path):
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True to plots")
    else:
        print(f"✓ File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def make_panel(panel, gene, gene_coords, peak_rows, rna_adata, labels, cbcs, color_labels):
    chrom = gene_coords.loc[gene, "seqname"]
    plot_start = max(
        0,
        min(int(gene_coords.loc[gene, "start"]), int(peak_rows["peak_start"].min()))
        - WINDOW_PADDING_BP,
    )
    plot_end = max(
        int(gene_coords.loc[gene, "end"]),
        int(peak_rows["peak_end"].max()),
    ) + WINDOW_PADDING_BP

    print(
        f"Creating panel {panel} for {gene}: "
        f"{chrom}:{plot_start:,}-{plot_end:,} with {len(peak_rows)} significant peak(s)"
    )
    fig = make_plot_with_bigwig(
        chrom,
        plot_start,
        plot_end,
        INSERTION_FILE_DIR,
        GTF_FILE,
        PEAK_BED_FILE,
        rna_adata,
        gene,
        labels,
        cbcs,
        color_labels,
        peak_smoothing=100,
        figsize=(20 * 1.5, 10 * 1.5 * 0.5),
        bw_files=[CHIP_BW_PATH, CUT_TAG_BW_PATH],
        bw_labels=["Trim28\nChIP", "Trim28\nCUT&Tag"],
        bw_colors=["black", "black"],
        bw_bin_size=1,
        coverage_label_rotation=0,
        coverage_label_ha="right",
        coverage_label_va="center",
        coverage_labelpad=25,
    )
    add_peak_highlights(fig, peak_rows)
    rasterize_lines(fig)

    png_path = os.path.join(OUTPUT_DIR, f"{panel}.png")
    svg_path = os.path.join(OUTPUT_DIR, f"{panel}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    validate_file_sizes(png_path, svg_path)
    plt.close(fig)

    peak_rows = peak_rows.copy()
    peak_rows["plot_start"] = plot_start
    peak_rows["plot_end"] = plot_end
    return peak_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading multiome RNA data...")
    rna_adata = get_adata("multiome_rna")
    print("Loading multiome ATAC data...")
    atac_adata = get_adata("multiome_atac")

    labels, cbcs, color_labels = build_cell_groups(rna_adata)
    significant_peaks = compute_significant_peak_table(rna_adata, atac_adata)
    gene_coords = load_gene_coordinates(PANELS.values())

    plot_rows = []
    for panel, gene in PANELS.items():
        peak_rows = significant_peaks[significant_peaks["gene"] == gene].copy()
        if peak_rows.empty:
            raise ValueError(f"No significant Figure 6d ATAC peaks found for {gene}.")
        plot_rows.append(
            make_panel(panel, gene, gene_coords, peak_rows, rna_adata, labels, cbcs, color_labels)
        )

    summary_df = pd.concat(plot_rows, ignore_index=True)
    summary_df = summary_df[
        [
            "panel",
            "gene",
            "n_significant_peaks",
            "peak_id",
            "chrom",
            "peak_start",
            "peak_end",
            "peak_log2fc",
            "peak_p_value",
            "peak_adjusted_p_value",
            "rna_log2fc",
            "rna_p_value",
            "rna_adjusted_p_value",
            "plot_start",
            "plot_end",
        ]
    ]
    summary_path = os.path.join(OUTPUT_DIR, "significant_de_peaks.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False, float_format="%.6g")

    print("\nSelected Figure 6d significant peaks:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
