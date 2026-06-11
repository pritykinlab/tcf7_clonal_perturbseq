import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ranksums
import statsmodels.stats.multitest as mt
from tqdm import tqdm
from scipy import stats

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as mt
from scipy.stats import gmean
import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats import multitest as mt
from anndata import AnnData
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ranksums
from statsmodels.stats import multitest as mt
import scanpy as sc



def wilcoxon_DE_test(
    adata: AnnData,
    CBC_group1,
    CBC_group2,
    group1_threshold_pct: float = 0.05,
    group2_threshold_pct: float = 0.0,
    pseudo_count: float = 0.01,
    norm_layer: str = None
) -> pd.DataFrame:
    """
    Perform differential expression using the Wilcoxon rank sum test after
    normalizing raw counts in adata.raw via Scanpy's sc.pp.normalize_total.

    This version uses sparse operations to filter genes efficiently, then
    converts to a dense array for the final rank sum test.

    Parameters
    ----------
    adata : AnnData
        The AnnData object with .raw containing unnormalized (raw) counts.
    CBC_group1 : list
        List of cell names (obs_names) for group 1.
    CBC_group2 : list
        List of cell names (obs_names) for group 2.
    group1_threshold_pct : float, optional (default=0.05)
        Fraction of cells in group1 that must express the gene (value > 0).
    group2_threshold_pct : float, optional (default=0.00)
        Fraction of cells in group2 that must express the gene (value > 0).
    pseudo_count : float, optional (default=0.01)
        Small constant added before log2 to avoid log(0).
    norm_layer : str, optional (default=None)
        Name of a pre-normalized layer in `adata.layers`.
        If provided, this will be used directly.

    Returns
    -------
    results_df : pd.DataFrame
        Columns:
          - 'Gene'
          - 'Log2FC'
          - 'rank_sum_statistic'
          - 'P-value'
          - 'Adjusted P-value'
    """
    # ---------------------------------------------------------------------
    # 1) Obtain normalized data (either from a provided layer or via normalization)
    # ---------------------------------------------------------------------
    if norm_layer is not None:
        # Use the pre-normalized layer directly
        data_norm = adata.layers[norm_layer]
        if not sp.issparse(data_norm):
            data_norm = sp.csc_matrix(data_norm)
    else:
        # Create a temporary AnnData from adata.raw for normalization
        adata = sc.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy()
        )
        # Normalize the copied data in-place using Scanpy
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)

        data_norm = adata.X
        # Ensure it's in a sparse format
        if not sp.issparse(data_norm):
            data_norm = sp.csc_matrix(data_norm)

    # ---------------------------------------------------------------------
    # 2) Build boolean masks for the two groups of cells
    # ---------------------------------------------------------------------
    mask_group1 = adata.obs_names.isin(CBC_group1)
    mask_group2 = adata.obs_names.isin(CBC_group2)

    # Subset the sparse matrix by rows (cells)
    data_group1_sp = data_norm[mask_group1, :]
    data_group2_sp = data_norm[mask_group2, :]

    # ---------------------------------------------------------------------
    # 3) Filter genes by expression threshold using sparse operations
    # ---------------------------------------------------------------------
    min_cells_group1 = np.ceil(len(CBC_group1) * group1_threshold_pct)
    min_cells_group2 = np.ceil(len(CBC_group2) * group2_threshold_pct)

    # Count nonzero entries per gene
    expr_mask_group1 = (data_group1_sp > 0).sum(axis=0).A1 >= min_cells_group1
    expr_mask_group2 = (data_group2_sp > 0).sum(axis=0).A1 >= min_cells_group2

    expr_genes_mask = np.logical_or(expr_mask_group1, expr_mask_group2)

    # Subset columns (genes)
    data_group1_sp = data_group1_sp[:, expr_genes_mask]
    data_group2_sp = data_group2_sp[:, expr_genes_mask]

    # ---------------------------------------------------------------------
    # 4) Convert to dense arrays for the rank sum calculations
    # ---------------------------------------------------------------------
    data_group1_norm = data_group1_sp.toarray()
    data_group2_norm = data_group2_sp.toarray()

    # ---------------------------------------------------------------------
    # 5) Wilcoxon rank sum test (per gene)
    # ---------------------------------------------------------------------
    num_genes = data_group1_norm.shape[1]
    rank_sum_stats = np.zeros(num_genes, dtype=float)
    p_values = np.zeros(num_genes, dtype=float)

    for g in range(num_genes):
        stat, pval = ranksums(data_group1_norm[:, g], data_group2_norm[:, g])
        rank_sum_stats[g] = stat
        p_values[g] = pval

    # ---------------------------------------------------------------------
    # 6) Calculate log2 fold change with a pseudo_count
    # ---------------------------------------------------------------------
    mean_group1 = data_group1_norm.mean(axis=0) + pseudo_count
    mean_group2 = data_group2_norm.mean(axis=0) + pseudo_count
    log2fc_values = np.log2(mean_group1 / mean_group2)

    # ---------------------------------------------------------------------
    # 7) Multiple testing correction (Benjamini-Hochberg)
    # ---------------------------------------------------------------------
    _, adj_p_values, _, _ = mt.multipletests(p_values, method='fdr_bh')

    # ---------------------------------------------------------------------
    # 8) Create a DataFrame with the results
    # ---------------------------------------------------------------------
    gene_names = adata.var_names[expr_genes_mask]

    results_df = pd.DataFrame({
        'Gene': gene_names,
        'Log2FC': log2fc_values,
        'rank_sum_statistic': rank_sum_stats,
        'P-value': p_values,
        'Adjusted P-value': adj_p_values
    })

    return results_df

def volcano_plot(results_df, title, label_key='Gene', label_points=True, logFC_label_threshold = 0.5):
    results_df = results_df.copy()
    # Extract relevant columns from results_df
    genes = results_df['Gene']
    logfoldchanges = results_df['Log2FC']
    pvals_adj = results_df['Adjusted P-value']

    # Handle zero p-values to avoid -inf in log scale
    min_nonzero_pval = pvals_adj[pvals_adj > 0].min() / (10**2)
    pvals_adj = np.where(pvals_adj == 0, min_nonzero_pval, pvals_adj)
    results_df['Adjusted P-value'] = pvals_adj

    # Calculate -log10 of the adjusted p-values
    minus_log10_pvals_adj = -np.log10(pvals_adj)

    # Filter genes based on adjusted p-value and logfoldchange thresholds
    significant_genes = genes[(pvals_adj < 0.05) & (np.abs(logfoldchanges) > logFC_label_threshold)]

    plt.figure(figsize=(10, 6))

    # Plotting
    plt.scatter(logfoldchanges, minus_log10_pvals_adj, c='grey', alpha=0.5, s=6)
    plt.scatter(logfoldchanges[significant_genes.index],
                minus_log10_pvals_adj[significant_genes.index],
                c='red', alpha=0.5, s=6)

    # Highlight significant genes with text labels
    if label_points:
        significant_genes = set(list(significant_genes))
        for idx, row in results_df.iterrows():
            gene = row['Gene']
            if gene in significant_genes:
                logFC = row['Log2FC']
                adjusted_pval = row['Adjusted P-value']
                # Ensure p-value is not zero and logFC is finite
                assert np.isfinite(logFC) and adjusted_pval > 0, f"Invalid logFC or adjusted p-value, logFC: {logFC}, adjusted_pval: {adjusted_pval}"
                minus_log10_pvals_adj = -np.log10(adjusted_pval)
                plt.text(logFC, minus_log10_pvals_adj, row[label_key], fontsize=9)

    # Adding horizontal line for p-value threshold
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='dashed')

    plt.title(title)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted P-value')
    plt.grid(True, which="both", ls="--", c='0.7')

    plt.show()


def de_test_and_heatmap_for_clusters(
    adata,
    specific_clusters=None,
    cluster_key='leiden',
    x_scale_heatmap=1
):
    import scanpy as sc
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # If no specific clusters are provided, use all 'leiden' clusters by default
    if specific_clusters is None:
        specific_clusters = adata.obs[cluster_key].unique()

    # Helper to sort clusters numerically if possible
    def custom_sort(e):
        return int(e)

    try:
        specific_clusters = sorted(specific_clusters, key=custom_sort)
    except:
        pass

    all_de_results_l = []

    # Perform Wilcoxon DE test (defaults only) for each cluster vs. the rest
    for cluster in specific_clusters:
        print(f"Processing cluster {cluster}")
        cluster_mask = adata.obs[cluster_key] == cluster
        rest_mask = ~cluster_mask

        CBC_group1 = adata.obs_names[cluster_mask]
        CBC_group2 = adata.obs_names[rest_mask]

        # Use the default params in wilcoxon_DE_test
        de_results = wilcoxon_DE_test(adata, CBC_group1, CBC_group2)
        de_results['group'] = cluster
        all_de_results_l.append(de_results)

    # Combine DE results into one DataFrame
    all_de_results_df = pd.concat(all_de_results_l)

    # Function to get top 30 upregulated genes
    def get_top_upregulated(group):
        top_genes = group.sort_values(by='Log2FC', ascending=False).head(30)
        top_genes = top_genes[['Gene', 'Log2FC']]
        top_genes['Rank'] = range(1, len(top_genes) + 1)

        # If fewer than 30 genes are present, pad rows
        if len(top_genes) < 30:
            additional_rows = 30 - len(top_genes)
            fill_data = {
                'Gene':   [None] * additional_rows,
                'Log2FC': [0] * additional_rows,
                'Rank':   list(range(len(top_genes) + 1, 31))
            }
            fill_df = pd.DataFrame(fill_data)
            top_genes = pd.concat([top_genes, fill_df], ignore_index=True)

        return top_genes[['Gene', 'Log2FC', 'Rank']]

    # Filter for significant genes and gather top hits
    significant_genes = all_de_results_df[all_de_results_df['Adjusted P-value'] < 0.05]
    top_upregulated = significant_genes.groupby('group').apply(get_top_upregulated).reset_index()

    # Convert 'group' to numeric if possible
    try:
        top_upregulated['group'] = pd.to_numeric(top_upregulated['group'])
    except:
        pass

    # Rename level_0 to 'group'
    top_upregulated.rename(columns={'level_0': 'group'}, inplace=True)

    # Pivot tables for log2FC and gene labels
    up_regulated_df = top_upregulated.pivot_table(index='Rank', columns='group', values='Log2FC')
    gene_labels = top_upregulated.pivot_table(index='Rank', columns='group', values='Gene', aggfunc='first')

    # Build heatmap
    num_labels = up_regulated_df.shape[1]
    plt.figure(figsize=(5 + num_labels * x_scale_heatmap, 8))
    heatmap = sns.heatmap(up_regulated_df, annot=gene_labels, fmt='', cmap='viridis')
    plt.title('Top 30 Upregulated Genes per Cluster')
    plt.ylabel('Rank')
    plt.xlabel('Cluster')

    # Colorbar label
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Log2FC')

    plt.show()

    return all_de_results_df


def apply_multi_resolution_clustering_then_heatmaps(adata, resolutions):
    import scanpy as sc

    leiden_keys = []

    # Cluster at each resolution and store the relevant .obs keys
    for resolution in resolutions:
        leiden_key = f'leiden_{resolution}'
        if leiden_key not in adata.obs.columns:
            sc.tl.leiden(adata, resolution=resolution)
            adata.obs[leiden_key] = adata.obs['leiden']
        leiden_keys.append(leiden_key)

    # Plot UMAP for each resolution
    sc.pl.umap(adata, color=leiden_keys, legend_loc='on data')

    # Dictionary to store all DE results
    all_de_results = {}

    # Run DE + heatmap for each resolution’s leiden partition
    for leiden_key in leiden_keys:
        print(f"############################ {leiden_key} ##################################")
        de_result = de_test_and_heatmap_for_clusters(adata, cluster_key=leiden_key)
        all_de_results[leiden_key] = de_result

    return all_de_results
