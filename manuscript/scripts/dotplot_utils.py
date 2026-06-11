#!/usr/bin/env python
"""
Utility functions for creating proper dotplots with z-score colors and expression fractions.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import scanpy as sc
from scipy.sparse import issparse


def create_proper_dotplot(adata, groupby, var_names, categories_order=None, figsize=(8, 2),
                         cmap="Reds", vmin=-1, vmax=1, dendrogram=False):
    """
    Create a dotplot with proper expression fractions and z-score colors.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix. Should be loaded with get_norm_log_zscore_control=True
    groupby : str
        The key in adata.obs to group cells by
    var_names : list
        List of genes to plot
    categories_order : list, optional
        Order of categories to plot
    figsize : tuple
        Figure size
    cmap : str or Colormap
        Colormap for expression values
    vmin, vmax : float
        Min and max values for color scale
    dendrogram : bool
        Whether to add dendrogram

    Returns
    -------
    DotPlot object
    """
    # Get the categories to use
    if categories_order is not None:
        categories = categories_order
    else:
        categories = adata.obs[groupby].cat.categories if pd.api.types.is_categorical_dtype(adata.obs[groupby]) else adata.obs[groupby].unique()

    # Filter adata to only include the specified categories
    mask = adata.obs[groupby].isin(categories)
    adata_subset = adata[mask].copy()

    # Ensure var_names are in adata
    var_names = [g for g in var_names if g in adata_subset.var_names]

    # Get gene indices
    gene_indices = [adata_subset.var_names.get_loc(g) for g in var_names]

    # Extract the layers
    norm_log = adata_subset.layers['norm_log']
    if issparse(norm_log):
        norm_log = norm_log.toarray()

    norm_log_zscore_control = adata_subset.layers['norm_log_zscore_control']
    if issparse(norm_log_zscore_control):
        norm_log_zscore_control = norm_log_zscore_control.toarray()

    # Initialize dataframes for sizes and colors
    fraction_expressing = pd.DataFrame(index=var_names, columns=categories, dtype=float)
    mean_expression = pd.DataFrame(index=var_names, columns=categories, dtype=float)

    # Calculate for each category
    for cat in categories:
        cat_mask = adata_subset.obs[groupby] == cat
        if cat_mask.sum() == 0:
            continue

        # Get the cells for this category
        cat_norm_log = norm_log[cat_mask.values, :][:, gene_indices]
        cat_zscore = norm_log_zscore_control[cat_mask.values, :][:, gene_indices]

        # Calculate fraction expressing (>0 in norm_log)
        frac_expr = (cat_norm_log > 0).mean(axis=0)
        fraction_expressing[cat] = frac_expr

        # Calculate mean z-score expression
        mean_expr = cat_zscore.mean(axis=0)
        mean_expression[cat] = mean_expr

    # Create the dotplot with custom size and color dataframes
    dp = sc.pl.DotPlot(
        adata_subset,
        groupby=groupby,
        var_names=var_names,
        categories_order=categories,
        figsize=figsize,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        dot_size_df=fraction_expressing.T,  # Transpose to match expected format
        dot_color_df=mean_expression.T,     # Transpose to match expected format
        layer=None  # We're providing our own data
    )
    dp.legend(
        size_title="fraction of cells\nin group (%)",
        colorbar_title="mean expression\nin group",
    )

    # Set dendrogram if requested (must be set after creation)
    if dendrogram:
        dp.dendrogram = dendrogram

    return dp
