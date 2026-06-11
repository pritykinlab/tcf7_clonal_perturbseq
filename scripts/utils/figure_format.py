import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import gzip as gz
import scipy
from importlib import reload
import glob
import tqdm
import itertools
from scipy.spatial.distance import cdist
import scanpy as sc
import os
from pathlib import Path
from scipy.sparse import issparse

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_rc_params():
    plt.rcParams["axes.titlesize"] = 25
    plt.rcParams["axes.labelsize"] = 25
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
    plt.rcParams['figure.figsize'] = (7,7)

import functools
from collections import OrderedDict

@functools.lru_cache(maxsize=None)
def _get_adata(dataset, use_cells_only_one_construct=False, remove_NG=False, get_raw_scaled=False, remove_lymph_node=False, get_norm_log_zscore_control=False):
    if dataset == 'trim28_setdb1':
        adata = sc.read_h5ad(REPO_ROOT / 'data/trim28_setdb1/preprocessing/final.h5ad')
        if remove_lymph_node:
            adata = adata[list(adata.obs['cell_type'].apply(lambda e: 'Lymph Node' not in e))]
    elif dataset == 'gene_10':
        adata = sc.read_h5ad(REPO_ROOT / "data/gene10/preprocessing/final.h5ad")
    elif dataset == 'comp_screen1':
        adata = sc.read_h5ad(REPO_ROOT / "data/comp_screen1/final.h5ad")
    elif dataset == 'multiome_rna':
        multiome_rna_adata = sc.read_h5ad(REPO_ROOT / 'data/multiome/analysis_using_atlas_v2/rna_final.h5ad')
        return multiome_rna_adata
    elif dataset == 'multiome_atac':
        multiome_atac_adata = sc.read_h5ad(REPO_ROOT / 'data/multiome/analysis_using_atlas_v2/atac_final.h5ad')
        return multiome_atac_adata
    else:
        raise ValueError("Not a valid dataset")
    if use_cells_only_one_construct:
        adata = adata[adata.obs['num_constructs'] == 1]
    if remove_NG:
        adata = adata[adata.obs['perturbation'] != '']
    if get_raw_scaled:
        adata = adata.raw.to_adata()
        adata.layers['raw'] = adata.X.copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=False)
        adata.layers['norm_no_exclude'] = adata.X.copy()
        adata.X = adata.layers['raw'].copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        adata.layers['norm'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['norm_log'] = adata.X.copy()
        sc.pp.scale(adata, max_value=10)
        adata.layers['norm_log_zscore'] = adata.X.copy()
    if get_norm_log_zscore_control:
        adata = adata.raw.to_adata()
        adata.layers['raw'] = adata.X.copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=False)
        adata.layers['norm_no_exclude'] = adata.X.copy()
        adata.X = adata.layers['raw'].copy()
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        adata.layers['norm'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['norm_log'] = adata.X.copy()

        nt_mask = (adata.obs['perturbation'] == 'NT')
        if nt_mask.sum() == 0:
            raise ValueError("No cells with perturbation == 'NT' found. Cannot compute control-based z-score.")

        # Extract the norm_log layer for *all* cells
        X_all = adata.layers['norm_log']
        if issparse(X_all):
            X_all = X_all.toarray()
        else:
            X_all = X_all.copy()

        # Extract norm_log for only the NT cells
        X_nt = adata[nt_mask].layers['norm_log']
        if issparse(X_nt):
            X_nt = X_nt.toarray()
        else:
            X_nt = X_nt.copy()

        # Compute mean & std across NT cells (axis=0 = genes)
        ctrl_mean = np.mean(X_nt, axis=0)
        ctrl_std = np.std(X_nt, axis=0, ddof=1)

        # Avoid division by zero
        ctrl_std[ctrl_std == 0] = 1e-9

        # Z-score transform using NT stats
        X_control_zscore = (X_all - ctrl_mean) / ctrl_std

        # Clip to [-10, 10]
        X_control_zscore = np.clip(X_control_zscore, a_min=-10, a_max=10)

        # Store in a new layer
        adata.layers['norm_log_zscore_control'] = X_control_zscore
    return adata

def _curate_trim28_setdb1_perturbation(raw_value):
    """Map arbitrary guide combinations to the desired Trim28/Setdb1/NT labels."""
    if pd.isna(raw_value):
        return ''
    tokens = [t for t in str(raw_value).split('__') if t]
    token_set = set(tokens)
    if not token_set:
        return ''

    has_trim28 = 'Trim28' in token_set
    has_setdb1 = 'Setdb1' in token_set
    has_nt = 'NT' in token_set

    if has_trim28 and has_setdb1:
        return 'Setdb1+Trim28'
    if has_trim28 and not has_setdb1:
        return 'Trim28'
    if has_setdb1 and not has_trim28:
        return 'Setdb1'
    if has_nt:
        return 'NT'
    return ''

def get_adata(dataset, use_cells_only_one_construct=False, remove_NG=False, get_raw_scaled=False, remove_lymph_node=False, get_norm_log_zscore_control=False):
    adata = _get_adata(
        dataset,
        use_cells_only_one_construct=use_cells_only_one_construct,
        remove_NG=remove_NG,
        get_raw_scaled=get_raw_scaled,
        remove_lymph_node=remove_lymph_node,
        get_norm_log_zscore_control=get_norm_log_zscore_control,
    ).copy()

    # Apply curated labeling (mirrors get_adata)
    if dataset == 'trim28_setdb1':
        adata.obs['perturbation_raw'] = adata.obs['perturbation'].copy()
        adata.obs['perturbation'] = adata.obs['perturbation_raw'].apply(_curate_trim28_setdb1_perturbation)
    # Normalize cell type naming before returning
    if 'cell_type' in adata.obs.columns:
        ct = adata.obs['cell_type']
        if pd.api.types.is_categorical_dtype(ct):
            cats = list(ct.cat.categories)
            if 'Ccl5+' in cats:
                new_cats = [
                    ('Ccl5 High' if c == 'Ccl5+' else c)
                    for c in cats
                ]
                adata.obs['cell_type'] = ct.cat.rename_categories(new_cats)
        else:
            adata.obs['cell_type'] = ct.replace('Ccl5+', 'Ccl5 High')

    return adata

colors2 = plt.get_cmap('Paired')
import matplotlib.pyplot as plt

tableau_colors = plt.get_cmap('tab10')

cell_type_colors = {
    'Cycling': tableau_colors(2),
    'Ccl5 High': tableau_colors(0),
    'TRM-like': tableau_colors(3),
    'Granzyme High': tableau_colors(4),
    'Progenitor': tableau_colors(1),
    'Progenitor Cycling': 'violet',
    'High Mito': tableau_colors(5),
    'Early Activation': 'palegreen',
    'Progenitor dLN': 'gold',
    'High Mito LN': 'goldenrod'
}

colors2 = plt.get_cmap('Paired')

perturbation_colors = {
    'Bach2': colors2(6),
    'Ifngr1': colors2(1),
    'Tnfrsf1b': 'darkturquoise',
    'Il18r1':'bisque',
    'Jak3':colors2(2),
    'Ifngr2':colors2(3),
    'Prdm1':colors2(0),
    'Trim28':colors2(7),
    'Setdb1': 'brown',
    'Jak2': colors2(8),
    'NT':'grey'
}

aPD1_color = 'lightskyblue'
isotype_color = 'salmon'

condition_colors_dict = {
    "Isotype": isotype_color,
    "aPD1": aPD1_color
}

cell_type_orders_gene10 = ['Progenitor', 'Progenitor Cycling', 'Cycling', 'Ccl5 High', 'Granzyme High', 'TRM-like']
cell_type_orders_trim28_setdb1 = ['Progenitor dLN', 'Progenitor', 'Progenitor Cycling', 'Cycling', 'Ccl5 High', 'Granzyme High', 'TRM-like']
cell_type_orders_multiome = ['Progenitor', 'Early Activation', 'Cycling', 'Ccl5 High', 'Granzyme High', 'TRM-like']
perturbation_orders_gene10 = ['NT', 'Bach2', 'Ifngr1', 'Il18r1', 'Jak2', 'Jak3', 'Prdm1', 'Setdb1', 'Tnfrsf1b', 'Trim28']

from matplotlib.colors import LinearSegmentedColormap

def my_RdBu_colormap():
    original_cmap = plt.cm.RdBu_r
    colors = original_cmap(np.linspace(0.1, 0.9, 256))

    # Define the segment to enhance in the middle
    mid_start = int(0.4 * 256)  # Start of the middle segment
    mid_end = int(0.6 * 256)    # End of the middle segment
    mid_colors = colors[mid_start:mid_end]

    # Double the amount of colors in the middle segment
    enhanced_mid = np.repeat(mid_colors, 4, axis=0)

    # Combine the colors with the enhanced middle segment
    new_colors = np.concatenate(
        (colors[:mid_start], enhanced_mid, colors[mid_end:]),
        axis=0
    )

    # Create a new colormap from the modified color list
    return LinearSegmentedColormap.from_list('extended_smooth_mid', new_colors)


dotplot_genes = ['Tcf7',  'Xcl1', 'Slamf6', 'Ccr7', 'Nfkb1', 'Nfkbia', 'Itgb1', 'Bach2',
                      'Mki67','Top2a', 'Mcm3', 'Mcm4', 'Ccnb2',
                        'Havcr2', 'Pdcd1', 'Entpd1', 'Lag3', 'Tigit', 'Tox', 'Cxcr6', 'Ifng',
                      'Gzmb', 'Gzmc', 'Gzmd', 'Prf1', 'Il1r2',
                      'Ccl5',
                      'Runx2', 'Prdm1',
                      'Itga1', 'Epcam', 'Nars2', 'Itm2c', 'Ifngr1',
                     'Trim28', 'Setdb1']


dotplot_genes_LN = ['Id3'] + dotplot_genes
