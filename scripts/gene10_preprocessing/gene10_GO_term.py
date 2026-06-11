import pandas as pd
import numpy as np
import sys
import tqdm
import scanpy as sc

sys.path.append('./scripts/utils')
from misc import my_score_genes
from figure_format import get_adata
from GO import GOParser

# ---------------------------------------------------
# 1) Parse GO terms
# ---------------------------------------------------
go_parser = GOParser('./data/misc/msigdb_GO_pathways.json')

# Load an AnnData once (any version) just so we know which genes exist in the dataset
adata_for_filtering = get_adata(
    'gene_10',
    use_cells_only_one_construct=False,
    remove_NG=False,
    get_raw_scaled=True
)

var_set = set(adata_for_filtering.var_names)

# Filter GO terms: only keep those with >=5 overlapping genes
go_terms = []
for go_number, go_name, genes in go_parser:
    genes_in_adata = [g for g in genes if g in var_set]
    if len(genes_in_adata) >= 5:
        go_terms.append((go_number, go_name, genes_in_adata))

print(f"Retained {len(go_terms)} GO terms with at least 5 genes in adata.")


# ---------------------------------------------------
# 2) Define a single scoring function
# ---------------------------------------------------
def compute_go_scores_for_one_term(go_term, adata, zscore_layer=None):
    """
    Given a single GO term tuple (go_number, go_name, list_of_genes),
    computes the cell-level gene score for that GO term and returns
    a (go_name, np.array) pair. The array is length = number of cells in `adata`.

    If zscore_layer is not None, we pass that to 'my_score_genes' for scoring.
    """
    go_number, go_name, genes = go_term
    genes_in_adata = [g for g in genes if g in adata.var_names]
    if not genes_in_adata:
        return go_name, None  # no overlapping genes => no score

    # Make a local copy to avoid concurrency issues
    local_adata = adata.copy()

    # Use your custom function or scanpy's built-in (adjust signature as needed)
    if zscore_layer:
        my_score_genes(local_adata, gene_list=genes_in_adata, score_name="temp", zscore_layer=zscore_layer)
    else:
        my_score_genes(local_adata, gene_list=genes_in_adata, score_name="temp")

    # Extract and return the resulting scores
    scores = local_adata.obs["temp"].values
    return go_name, scores


# ---------------------------------------------------
# 3) Define a helper to compute, assemble, and write out GO scores
# ---------------------------------------------------
def compute_go_scores(
    go_terms,
    output_path,
    zscore_layer=None
):
    """
    1) Load adata, possibly with z-score layer if needed
    2) Compute GO scores for each term
    3) Assemble into a DataFrame
    4) Write to 'output_path'
    """
    # Load the appropriate adata
    if zscore_layer == 'norm_log_zscore':
        adata = get_adata(
            'gene_10',
            use_cells_only_one_construct=False,
            remove_NG=False,
            get_raw_scaled=True,
        )
    elif zscore_layer == 'norm_log_zscore_control':
        adata = get_adata(
            'gene_10',
            use_cells_only_one_construct=False,
            remove_NG=False,
            get_norm_log_zscore_control=True
        )


    results_dict = {}
    n_go_terms = len(go_terms)

    for go_term in tqdm.tqdm(go_terms, total=n_go_terms, desc=f"Computing GO scores (zscore_layer={zscore_layer})"):
        go_name, arr = compute_go_scores_for_one_term(go_term, adata, zscore_layer=zscore_layer)
        if arr is not None:
            results_dict[go_name] = arr

    cell_barcodes = adata.obs_names.values
    go_score_df = pd.DataFrame(data=results_dict, index=cell_barcodes)

    go_score_df.to_csv(output_path, sep="\t")
    print(f"Saved GO scores to {output_path} with shape: {go_score_df.shape}")


# ---------------------------------------------------
# 4) Actually run it for both cases
# ---------------------------------------------------
if __name__ == "__main__":
    # Case 1: No control (normal)
    compute_go_scores(
        go_terms,
        output_path="./intermediate_final_plots/gene10_CBC_GO_score.tsv",
        zscore_layer='norm_log_zscore'
    )

    # Case 2: With 'norm_log_zscore_control'
    compute_go_scores(
        go_terms,
        output_path="./intermediate_final_plots/gene10_CBC_GO_score_control.tsv",
        zscore_layer='norm_log_zscore_control'
    )
