#!/usr/bin/env python
"""
Generate Figure S15 count-reconstruction validation histograms.

Panel a shows per-gene Pearson correlations across cells between RNA counts
reconstructed from the raw multiome gene-expression BAM files and RNA counts
derived from Cell Ranger in the final processed multiome RNA AnnData object.

Panel b shows per-peak Pearson correlations across cells between ATAC insertion
counts reconstructed from the raw multiome ATAC BAM files and ATAC peak counts
derived from Cell Ranger in the final processed multiome ATAC AnnData object.
"""

import os
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from scipy.stats import pearsonr


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams["font.size"] = 22
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.figsize"] = (6, 4.2)


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_15"
RNA_H5 = "../data/multiome/analysis_using_atlas_v2/rna_final.h5ad"
ATAC_H5 = "../data/multiome/analysis_using_atlas_v2/atac_final.h5ad"
GTF_FILE = "../data/misc/gencode.vM10.basic.annotation.gtf.gz"

RNA_BAMS = {
    "Control": "../raw_data/new_data/trim28_multiome/Control/gex_possorted_bam.bam",
    "Trim28": "../raw_data/new_data/trim28_multiome/Trim28/gex_possorted_bam.bam",
}
ATAC_BAMS = {
    "Control": "../raw_data/new_data/trim28_multiome/Control/atac_possorted_bam.bam",
    "Trim28": "../raw_data/new_data/trim28_multiome/Trim28/atac_possorted_bam.bam",
}

RNG_SEED = 603
N_SAMPLED_GENES = 100
N_SAMPLED_PEAKS = 100


def decode_string_array(values):
    return np.array(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def load_feature_metadata(h5_path):
    with h5py.File(h5_path, "r") as handle:
        obs_names = decode_string_array(handle["obs"]["_index"][:])
        var_names = decode_string_array(handle["var"]["_index"][:])
        var_std = np.asarray(handle["var"]["std"][:], dtype=float)
    return obs_names, var_names, var_std


def extract_selected_raw_columns(h5_path, selected_features):
    with h5py.File(h5_path, "r") as handle:
        obs_names = decode_string_array(handle["obs"]["_index"][:])
        var_names = decode_string_array(handle["var"]["_index"][:])
        feature_to_index = {name: idx for idx, name in enumerate(var_names)}
        selected_indices = np.array(
            [feature_to_index[name] for name in selected_features], dtype=np.int32
        )

        selected_map = np.full(len(var_names), -1, dtype=np.int32)
        selected_map[selected_indices] = np.arange(len(selected_features), dtype=np.int32)

        raw_group = handle["layers"]["raw"]
        indptr = raw_group["indptr"][:]
        indices = raw_group["indices"]
        data = raw_group["data"]

        matrix = np.zeros((len(obs_names), len(selected_features)), dtype=np.float64)
        for row_idx in range(len(obs_names)):
            start = indptr[row_idx]
            end = indptr[row_idx + 1]
            row_indices = indices[start:end]
            local_indices = selected_map[row_indices]
            mask = local_indices >= 0
            if np.any(mask):
                matrix[row_idx, local_indices[mask]] = data[start:end][mask]

    return obs_names, matrix


def load_gene_coordinates():
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
    gtf_df = gtf_df.dropna(subset=["gene_name"])
    gtf_df = gtf_df.drop_duplicates(subset=["gene_name"])
    return {
        row.gene_name: (row.seqname, int(row.start), int(row.end))
        for row in gtf_df.itertuples(index=False)
    }


def parse_peak_name(peak_name):
    chrom, start, end = peak_name.split("-")
    return chrom, int(start), int(end)


def sample_features(feature_names, feature_std, eligible_mask, rng, n_to_sample):
    eligible_features = feature_names[(feature_std > 0) & eligible_mask]
    if len(eligible_features) < n_to_sample:
        raise ValueError(
            f"Requested {n_to_sample} features, but only {len(eligible_features)} are eligible."
        )
    sampled = rng.choice(eligible_features, size=n_to_sample, replace=False)
    return sorted(sampled.tolist())


def reconstruct_rna_counts(sampled_genes, gene_coordinates, obs_names):
    obs_to_row = {name: idx for idx, name in enumerate(obs_names)}
    counts = np.zeros((len(obs_names), len(sampled_genes)), dtype=np.int32)
    bam_handles = {
        condition: pysam.AlignmentFile(path, "rb")
        for condition, path in RNA_BAMS.items()
    }

    try:
        for gene_idx, gene_name in enumerate(sampled_genes):
            chrom, start, end = gene_coordinates[gene_name]
            print(f"RNA {gene_idx + 1}/{len(sampled_genes)}: {gene_name}")
            for condition, bam in bam_handles.items():
                cell_to_umis = defaultdict(set)
                for read in bam.fetch(chrom, start, end):
                    if read.is_duplicate:
                        continue
                    if read.reference_start < start or read.reference_end > end:
                        continue
                    if not read.has_tag("CB") or not read.has_tag("UB"):
                        continue
                    if read.has_tag("NH") and read.get_tag("NH") != 1:
                        continue

                    prefixed_cell = f"{condition}_{read.get_tag('CB')}"
                    row_idx = obs_to_row.get(prefixed_cell)
                    if row_idx is None:
                        continue

                    cell_to_umis[row_idx].add(read.get_tag("UB"))

                for row_idx, umi_set in cell_to_umis.items():
                    counts[row_idx, gene_idx] = len(umi_set)
    finally:
        for bam in bam_handles.values():
            bam.close()

    return counts


def reconstruct_atac_counts(sampled_peaks, obs_names):
    obs_to_row = {name: idx for idx, name in enumerate(obs_names)}
    counts = np.zeros((len(obs_names), len(sampled_peaks)), dtype=np.int32)
    bam_handles = {
        condition: pysam.AlignmentFile(path, "rb")
        for condition, path in ATAC_BAMS.items()
    }

    try:
        for peak_idx, peak_name in enumerate(sampled_peaks):
            chrom, start, end = parse_peak_name(peak_name)
            print(f"ATAC {peak_idx + 1}/{len(sampled_peaks)}: {peak_name}")
            for condition, bam in bam_handles.items():
                cell_to_count = defaultdict(int)
                for read in bam.fetch(chrom, start, end):
                    if read.is_duplicate or read.is_unmapped or not read.has_tag("CB"):
                        continue

                    prefixed_cell = f"{condition}_{read.get_tag('CB')}"
                    row_idx = obs_to_row.get(prefixed_cell)
                    if row_idx is None:
                        continue

                    start_insertion = read.reference_start + 4
                    end_insertion = read.reference_end - 5

                    if start <= start_insertion <= end:
                        cell_to_count[row_idx] += 1
                    if start <= end_insertion <= end:
                        cell_to_count[row_idx] += 1

                for row_idx, insertion_count in cell_to_count.items():
                    counts[row_idx, peak_idx] = insertion_count
    finally:
        for bam in bam_handles.values():
            bam.close()

    return counts


def build_correlation_table(feature_names, reference_matrix, reconstructed_matrix):
    records = []
    for feature_idx, feature_name in enumerate(feature_names):
        reference = reference_matrix[:, feature_idx]
        reconstructed = reconstructed_matrix[:, feature_idx]
        if np.std(reference) == 0 or np.std(reconstructed) == 0:
            correlation = np.nan
        else:
            correlation = pearsonr(reference, reconstructed)[0]

        records.append(
            {
                "feature": feature_name,
                "pearson_r": correlation,
                "reference_total": float(reference.sum()),
                "reconstructed_total": float(reconstructed.sum()),
                "reference_nonzero_cells": int(np.count_nonzero(reference)),
                "reconstructed_nonzero_cells": int(np.count_nonzero(reconstructed)),
            }
        )

    correlation_df = pd.DataFrame.from_records(records)
    if correlation_df["pearson_r"].isna().any():
        missing = int(correlation_df["pearson_r"].isna().sum())
        raise ValueError(f"{missing} sampled features had undefined Pearson correlations.")

    return correlation_df.sort_values("feature").reset_index(drop=True)


def validate_file_sizes(png_path, svg_path):
    png_size_mb = os.path.getsize(png_path) / 1024 / 1024
    svg_size_kb = os.path.getsize(svg_path) / 1024

    if png_size_mb > 5:
        print(f"ERROR: PNG {png_size_mb:.2f}MB exceeds 5MB limit.")
    if svg_size_kb > 500:
        print(f"ERROR: SVG {svg_size_kb:.1f}KB exceeds 500KB limit.")
    else:
        print(f"File sizes OK: PNG {png_size_mb:.2f}MB, SVG {svg_size_kb:.1f}KB")


def plot_histogram(
    correlation_df, feature_label, color, png_path, svg_path, title, y_tick_step
):
    fig, ax = plt.subplots()
    bins = np.linspace(-1.0, 1.0, 31)
    values = correlation_df["pearson_r"].to_numpy()
    median_value = float(np.median(values))

    counts, _, _ = ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.8)
    ax.axvline(median_value, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        0.98,
        0.95,
        f"median r = {median_value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Pearson r across cells")
    ax.set_ylabel(f"number of {feature_label}")
    ax.set_title(title, pad=20)
    max_count = int(np.max(counts))
    max_tick = max(y_tick_step, ((max_count + y_tick_step - 1) // y_tick_step) * y_tick_step)
    ax.set_ylim(0, max_tick)
    ax.set_yticks(np.arange(0, max_tick + 1, y_tick_step))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    validate_file_sizes(png_path, svg_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("Loading RNA metadata...")
    rna_obs_names, rna_var_names, rna_var_std = load_feature_metadata(RNA_H5)
    gene_coordinates = load_gene_coordinates()
    gene_mask = np.array([gene in gene_coordinates for gene in rna_var_names], dtype=bool)
    sampled_genes = sample_features(
        rna_var_names,
        rna_var_std,
        gene_mask,
        rng,
        N_SAMPLED_GENES,
    )
    print(f"Sampled {len(sampled_genes)} genes.")

    print("Loading sampled RNA reference counts...")
    _, rna_reference_matrix = extract_selected_raw_columns(RNA_H5, sampled_genes)
    print("Reconstructing sampled RNA counts from BAM files...")
    rna_reconstructed_matrix = reconstruct_rna_counts(
        sampled_genes, gene_coordinates, rna_obs_names
    )
    gene_correlation_df = build_correlation_table(
        sampled_genes, rna_reference_matrix, rna_reconstructed_matrix
    )
    gene_correlation_path = os.path.join(OUTPUT_DIR, "gene_correlations.tsv")
    gene_correlation_df.to_csv(gene_correlation_path, sep="\t", index=False)

    print("Loading ATAC metadata...")
    atac_obs_names, atac_var_names, atac_var_std = load_feature_metadata(ATAC_H5)
    peak_mask = np.array([True] * len(atac_var_names), dtype=bool)
    sampled_peaks = sample_features(
        atac_var_names,
        atac_var_std,
        peak_mask,
        rng,
        N_SAMPLED_PEAKS,
    )
    print(f"Sampled {len(sampled_peaks)} peaks.")

    print("Loading sampled ATAC reference counts...")
    _, atac_reference_matrix = extract_selected_raw_columns(ATAC_H5, sampled_peaks)
    print("Reconstructing sampled ATAC counts from BAM files...")
    atac_reconstructed_matrix = reconstruct_atac_counts(sampled_peaks, atac_obs_names)
    peak_correlation_df = build_correlation_table(
        sampled_peaks, atac_reference_matrix, atac_reconstructed_matrix
    )
    peak_correlation_path = os.path.join(OUTPUT_DIR, "peak_correlations.tsv")
    peak_correlation_df.to_csv(peak_correlation_path, sep="\t", index=False)

    print("Saving Figure S15a...")
    plot_histogram(
        gene_correlation_df,
        feature_label="genes",
        color="#4C78A8",
        png_path=os.path.join(OUTPUT_DIR, "a.png"),
        svg_path=os.path.join(OUTPUT_DIR, "a.svg"),
        title=f"randomly sampled genes (n = {len(gene_correlation_df)})",
        y_tick_step=10,
    )

    print("Saving Figure S15b...")
    plot_histogram(
        peak_correlation_df,
        feature_label="peaks",
        color="#F58518",
        png_path=os.path.join(OUTPUT_DIR, "b.png"),
        svg_path=os.path.join(OUTPUT_DIR, "b.svg"),
        title=f"randomly sampled peaks (n = {len(peak_correlation_df)})",
        y_tick_step=20,
    )

    print("Figure S15 completed successfully.")


if __name__ == "__main__":
    main()
