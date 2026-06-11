import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
import scanpy as sc
import tqdm
import functools
import glob
import gzip as gz
import itertools
import math
import os
import sys
from pathlib import Path
from importlib import reload
from collections import OrderedDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_adata_dict(
    base_dir: str,
    lane_list: list,
    filename: str = "filtered_feature_bc_matrix.h5",
    mito_prefix: str = "mt-"
) -> OrderedDict:
    """
    Loads one AnnData object per lane into an OrderedDict, attaching minimal QC fields.
    Does NOT apply any filtering (mitochondrial or gene/cell thresholds).

    Args:
        base_dir (str): Base directory containing subdirectories for each lane.
        lane_list (list): List of lane folder names (e.g. ["Lane1", "Lane2", ...]).
        filename (str): Name of the 10x H5 file inside each lane directory.
        mito_prefix (str): Prefix identifying mitochondrial genes (default "mt-").

    Returns:
        OrderedDict: Keys are lane names, values are unfiltered AnnData objects.
    """
    adata_dict = OrderedDict()

    for lane_dir in lane_list:
        h5_path = os.path.join(base_dir, lane_dir, filename)
        if not os.path.exists(h5_path):
            print(f"[load_adata_dict] File not found: {h5_path}")
            continue

        # Load
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()

        # Annotate lane
        adata.obs["lane"] = f"l{lane_dir[1:]}"
        adata.obs_names = [f"l{lane_dir[1:]}_{obs}" for obs in adata.obs_names]

        adata.var["mt"] = adata.var_names.str.startswith("mt-")  # mitochondrial genes
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)

        # Mark mitochondrial genes and percent
        adata.obs["percent_mito"] = (
            adata[:, adata.var["mt"]].X.sum(axis=1).A1 / adata.X.sum(axis=1).A1
        ) * 100

        # Store in dictionary
        adata_dict[lane_dir] = adata

    return adata_dict


def plot_qc_metrics(
    adata_dict: dict,
    qc_vars: list = None,
):
    """
    Generates violin and scatter QC plots for each AnnData in the dictionary.

    Args:
        adata_dict (dict): OrderedDict (or dict) of AnnData objects keyed by lane name.
        qc_vars (list): Extra variables for sc.pp.calculate_qc_metrics. E.g. ["mt", "ribo", "hb"].
    """
    if qc_vars is None:
        qc_vars = ["mt"]  # By default, only run QC with "mt" if no list given.

    for lane, adata in adata_dict.items():
        # (Re-)calculate full QC metrics
        sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True, log1p=True)

        print(f"##### Lane: {lane} #####")
        # Violin plots for key metrics
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True,
        )

        # Scatter plot: total_counts vs n_genes_by_counts
        sc.pl.scatter(
            adata,
            x="total_counts",
            y="n_genes_by_counts",
            color="pct_counts_mt",
        )

def get_cell_cycle_genes():
    """
    Reads S-phase and G2M-phase genes from CSV, then returns them in mouse gene format
    (capital first letter, all other letters lowercase).

    Returns
    -------
    s_genes_formatted : list[str]
        Formatted S-phase genes
    g2m_genes_formatted : list[str]
        Formatted G2M-phase genes
    """
    # Read the gene lists
    s_genes = pd.read_csv(REPO_ROOT / "data/misc/s_genes.csv")
    g2m_genes = pd.read_csv(REPO_ROOT / "data/misc/g2m_genes.csv")

    # Convert gene names to mouse gene format
    s_genes_formatted = [gene.lower().capitalize() for gene in s_genes["x"]]
    g2m_genes_formatted = [gene.lower().capitalize() for gene in g2m_genes["x"]]

    return s_genes_formatted, g2m_genes_formatted

def score_cell_cycle_genes(adata):
    """
    Uses Scanpy's built-in `score_genes_cell_cycle` to score cell cycle phase.

    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix.

    Returns
    -------
    AnnData
        Copy of the input `adata` with `S_score`, `G2M_score`, and `phase` in `adata.obs`.
    """
    adata = adata.copy()

    # 1. Get formatted gene lists
    s_genes_formatted, g2m_genes_formatted = get_cell_cycle_genes()

    # 2. Use Scanpy's score_genes_cell_cycle
    sc.tl.score_genes_cell_cycle(adata, s_genes_formatted, g2m_genes_formatted)

    return adata


def my_score_cell_cycle_genes(adata):
    """
    Example function to:
      1) Create a copy of the input AnnData.
      2) Retrieve and format S and G2M genes.
      3) Call `my_score_genes_cell_cycle` to compute cell cycle scores.
      4) Return the modified copy.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.

    Returns
    -------
    AnnData
        Copy of the input `adata` with `S_score`, `G2M_score`, and `phase` in `adata.obs`.
    """
    # 1. Copy
    adata = adata.copy()

    # 2. Get formatted S and G2M gene lists
    s_genes_formatted, g2m_genes_formatted = get_cell_cycle_genes()

    # 3. Score cell cycle with our custom function
    my_score_genes_cell_cycle(
        adata,
        s_genes_formatted,
        g2m_genes_formatted,
        zscore_layer="norm_log_zscore"
    )

    # 4. Return the annotated data
    return adata


import numpy as np
import pandas as pd
from anndata import AnnData

def my_score_genes_cell_cycle(
    adata: AnnData,
    s_genes: list[str],
    g2m_genes: list[str],
    zscore_layer: str = "norm_log_zscore"
):
    """
    Replicates the logic of score_genes_cell_cycle but uses my_score_genes
    instead of scanpy's score_genes. Calculates S- and G2M-phase scores
    and assigns each cell a cell cycle phase.

    Modifies `adata` in place.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    s_genes : list of str
        List of genes associated with S phase.
    g2m_genes : list of str
        List of genes associated with G2M phase.
    zscore_layer : str
        Layer name in `adata.layers` containing z-scored expressions. If missing,
        falls back to `adata.X`.
    """

    # 1. Compute the S score
    my_score_genes(
        adata,
        gene_list=s_genes,
        score_name="S_score",
        zscore_layer=zscore_layer
    )

    # 2. Compute the G2M score
    my_score_genes(
        adata,
        gene_list=g2m_genes,
        score_name="G2M_score",
        zscore_layer=zscore_layer
    )

    # 3. Retrieve both scores
    scores = adata.obs[["S_score", "G2M_score"]].copy()

    # 4. Default phase is S
    phase = pd.Series("S", index=scores.index)

    # 5. If G2M_score is higher than S_score, call it G2M
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"

    # 6. If all scores (S and G2M) are negative, call it G1
    phase[np.all(scores < 0, axis=1)] = "G1"

    # 7. Store the final phase annotations
    adata.obs["phase"] = phase

import re

def remove_TCR_genes(adata):
    """
    Returns a new AnnData object with T-cell receptor genes removed.

    TCR genes are identified by a regular expression matching:
    '^T[cr][abdgr][vjcg]'
    which accounts for e.g., 'TcraV', 'TcrbJ', 'Trav', etc.

    Args:
        adata (AnnData): Input AnnData object. This function does not modify in-place.

    Returns:
        AnnData: A copy of the input adata with TCR genes removed.
    """
    adata = adata.copy()  # safe to copy so we do not modify the original
    pattern = re.compile(r'^T[cr][abdgr][vjcg]')

    # Identify genes that do NOT match the TCR pattern
    non_tcr_genes = [gene for gene in adata.var_names if not pattern.match(gene)]

    # Subset the AnnData to keep only non-TCR genes
    adata = adata[:, non_tcr_genes]

    return adata


def my_score_genes(
    adata,
    gene_list,
    score_name='score',
    zscore_layer='norm_log_zscore'
):
    """
    Compute a custom gene signature score by:
      - Taking the mean z-score of the genes in `gene_list`,
      - Subtracting the mean z-score of all genes,
      - Storing the result in `adata.obs[score_name]`.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data.
    gene_list : list of str
        List of gene names for which to compute the signature score.
    score_name : str, optional (default: 'score')
        Key name under which the score is stored in `adata.obs`.
    zscore_layer : str, optional (default: '')
        The layer in `adata.layers` containing z-scored gene expression.
        If empty string, will use `adata.X`.
    """

    # Use the specified z-score layer if it exists, otherwise fall back to `adata.X`
    if zscore_layer and zscore_layer in adata.layers:
        mat = adata.layers[zscore_layer]
    else:
        mat = adata.X

    # Handle sparse matrices
    if issparse(mat):
        mat = mat.toarray()

    # Make sure the gene_list is a list/array of valid genes present in adata.var_names
    valid_genes = [g for g in gene_list if g in adata.var_names]
    if len(valid_genes) == 0:
        raise ValueError("None of the genes in gene_list are present in adata.var_names.")

    # Get indices of the valid genes
    gene_indices = [adata.var_names.get_loc(g) for g in valid_genes]

    # Calculate the average z-score for the genes of interest per cell
    gene_set_scores = mat[:, gene_indices].mean(axis=1)

    # Calculate the average z-score across all genes per cell
    global_scores = mat.mean(axis=1)

    # Final score is the difference
    final_scores = gene_set_scores - global_scores

    # Store the scores in adata.obs
    adata.obs[score_name] = final_scores


import matplotlib.pyplot as plt

def draw_overlap_umap(adata, column, value_to_highlight, highlight_color, size=5, title=None):
    # Check if UMAP data is stored in adata
    if 'X_umap' not in adata.obsm.keys():
        raise ValueError("UMAP data not found in adata. Make sure UMAP embeddings are computed.")

    # Split groups based on provided column and value
    highlight_group = adata[adata.obs[column] == value_to_highlight]
    other_group = adata[adata.obs[column] != value_to_highlight]

    fig, ax = plt.subplots()

    # Plot 'other' group in grey
    ax.scatter(
        other_group.obsm['X_umap'][:, 0],
        other_group.obsm['X_umap'][:, 1],
        color='lightgrey',
        s=size,
        alpha=0.5,
        label='Others'
    )

    # Plot highlighted group with provided color
    ax.scatter(
        highlight_group.obsm['X_umap'][:, 0],
        highlight_group.obsm['X_umap'][:, 1],
        color=highlight_color,
        s=size*2,
        alpha=1,
        label=value_to_highlight
    )

    ax.axis('off')  # Remove the axis

    if title == 'default':
        ax.set_title(f"{column}: {value_to_highlight}")
    elif title:
        ax.set_title(title)
    plt.show()





