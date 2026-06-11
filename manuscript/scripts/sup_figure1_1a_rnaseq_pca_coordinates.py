#!/usr/bin/env python
"""Generate Figure S1a PCA coordinates from DESeq2 VST expression."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from figure1c_rnaseq_heatmap import DEFAULT_METADATA, MAIN_GROUPS, clean_sampletype, read_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VST_EXPRESSION = (
    REPO_ROOT / "manuscript" / "derived" / "rnaseq_deseq2" / "deseq2_vst_expression.tsv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "manuscript"
    / "derived"
    / "rnaseq_deseq2"
    / "deseq2_plotpca_sup_figure1_1a_samples.tsv"
)
DEFAULT_MEMORY_OUTPUT = (
    REPO_ROOT
    / "manuscript"
    / "derived"
    / "rnaseq_deseq2"
    / "deseq2_plotpca_sup_figure1_1a_memory_samples.tsv"
)
EXPERIMENTAL_MEMORY_GROUPS = [
    "D30 naive control CM",
    "D30 naive control EM",
    "D30 naive Thy1.1 CM",
    "D30 naive Thy1.1 EM",
    "D30 activated control CM",
    "D30 activated control EM",
    "D30 activated Thy1.1 CM",
    "D30 activated Thy1.1 EM",
    "D30 fresh control CM",
    "D30 fresh control EM",
    "D30 fresh Thy1.1 CM",
    "D30 fresh Thy1.1 EM",
]
PCA_GROUPS = MAIN_GROUPS

EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS = {
    "D30_naive_control_CM": "D30 naive control CM",
    "D30_naive_control_EM": "D30 naive control EM",
    "D30_naive_Thy1.1_CM": "D30 naive Thy1.1 CM",
    "D30_naive_Thy1.1_EM": "D30 naive Thy1.1 EM",
    "D30_activated_control_CM": "D30 activated control CM",
    "D30_activated_control_EM": "D30 activated control EM",
    "D30_activated_Thy1.1_CM": "D30 activated Thy1.1 CM",
    "D30_activated_Thy1.1_EM": "D30 activated Thy1.1 EM",
    "D30_fresh_control_CM": "D30 fresh control CM",
    "D30_fresh_control_EM": "D30 fresh control EM",
    "D30_fresh_Thy1.1_CM": "D30 fresh Thy1.1 CM",
    "D30_fresh_Thy1.1_EM": "D30 fresh Thy1.1 EM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vst-expression", type=Path, default=DEFAULT_VST_EXPRESSION)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-output", type=Path, default=DEFAULT_MEMORY_OUTPUT)
    parser.add_argument("--ntop", type=int, default=500)
    return parser.parse_args()


def read_vst_expression(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    data.columns = [str(col).strip() for col in data.columns]
    index_col = "gene_id" if "gene_id" in data.columns else data.columns[0]
    annotation_cols = {index_col, "gene_name", "gene"}
    sample_cols = [col for col in data.columns if col not in annotation_cols]
    if not sample_cols:
        raise ValueError(f"No sample columns found in VST expression table: {path}")

    expression = data.set_index(index_col)[sample_cols]
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = expression.dropna(axis=0, how="any")
    if expression.empty:
        raise ValueError(f"No numeric VST expression values found in: {path}")
    return expression


def build_pca_sample_metadata(metadata_path: Path, group_order: list[str]) -> pd.DataFrame:
    metadata = read_metadata(metadata_path)
    metadata = metadata.copy()
    metadata["sample_group"] = metadata["sampletype"].map(experimental_sample_group)
    unmapped = sorted(metadata.loc[metadata["sample_group"].isna(), "sampletype"].unique())
    if unmapped:
        raise ValueError(f"Unmapped sampletype values in PCA metadata: {unmapped}")

    sample_records: list[dict[str, object]] = []
    for group in group_order:
        group_samples = metadata.index[metadata["sample_group"] == group].tolist()
        if not group_samples:
            raise ValueError(f"No samples found for required Figure S1a group: {group}")

        for sample in group_samples:
            sample_records.append(
                {
                    "sample": sample,
                    "sampletype": metadata.loc[sample, "sampletype"],
                    "sample_group": group,
                    "display_order": len(sample_records) + 1,
                }
            )

    return pd.DataFrame.from_records(sample_records)


def experimental_sample_group(sampletype: str) -> str | None:
    if sampletype in EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS:
        return EXPERIMENTAL_MEMORY_SAMPLETYPE_GROUPS[sampletype]
    return clean_sampletype(sampletype)


def compute_plotpca_compatible_coordinates(
    expression: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    ntop: int,
) -> pd.DataFrame:
    sample_order = sample_metadata["sample"].tolist()
    missing_samples = [sample for sample in sample_order if sample not in expression.columns]
    if missing_samples:
        raise ValueError(f"VST expression table is missing samples: {missing_samples}")

    expression_subset = expression.loc[:, sample_order]
    ntop = min(ntop, expression.shape[0])
    row_variances = expression_subset.var(axis=1, ddof=1)
    selected_genes = row_variances.sort_values(ascending=False).head(ntop).index
    x = expression_subset.loc[selected_genes].T.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :2] * singular_values[:2]

    for pc_idx in range(scores.shape[1]):
        anchor_idx = np.argmax(np.abs(scores[:, pc_idx]))
        if scores[anchor_idx, pc_idx] < 0:
            scores[:, pc_idx] *= -1

    variance = singular_values**2
    percent_var = variance / variance.sum() * 100

    output = sample_metadata.copy()
    output["PC1"] = scores[:, 0]
    output["PC2"] = scores[:, 1]
    output["PC1_variance_percent"] = percent_var[0]
    output["PC2_variance_percent"] = percent_var[1]
    output["pca_method"] = "numpy.linalg.svd compatible with DESeq2::plotPCA"
    output["pca_transform"] = "vst(blind=TRUE)"
    output["pca_ntop"] = ntop
    return output


def main() -> None:
    args = parse_args()
    expression = read_vst_expression(args.vst_expression)
    sample_metadata = build_pca_sample_metadata(args.metadata, PCA_GROUPS)
    pca_coordinates = compute_plotpca_compatible_coordinates(expression, sample_metadata, args.ntop)
    memory_sample_metadata = build_pca_sample_metadata(args.metadata, EXPERIMENTAL_MEMORY_GROUPS)
    memory_pca_coordinates = compute_plotpca_compatible_coordinates(
        expression,
        memory_sample_metadata,
        args.ntop,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pca_coordinates.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote Figure 1c PCA coordinates for {pca_coordinates.shape[0]} samples: {args.output}")
    print(pca_coordinates["sample_group"].value_counts(sort=False).to_string())
    args.memory_output.parent.mkdir(parents=True, exist_ok=True)
    memory_pca_coordinates.to_csv(args.memory_output, sep="\t", index=False)
    print(f"Wrote memory PCA coordinates for {memory_pca_coordinates.shape[0]} samples: {args.memory_output}")
    print(memory_pca_coordinates["sample_group"].value_counts(sort=False).to_string())


if __name__ == "__main__":
    main()
