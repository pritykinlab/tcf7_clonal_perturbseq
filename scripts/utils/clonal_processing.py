#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pysam
import functools
import os

def read_guide_seq_dict(guide_seq_csv):
    """
    Reads a CSV mapping from 20bp sequence to guide name.

    The CSV must have columns [sequence, guide_name] or something similar.
    Adjust as necessary if your column names differ.

    Args:
        guide_seq_csv (str): Path to the guide-sequence CSV.

    Returns:
        dict: {sequence_20bp: guide_name}
    """
    df = pd.read_csv(guide_seq_csv)
    # Adjust column names if needed:
    # e.g. df.set_index("sequence") and the mapping is in "guide_name"
    seq_guide_dict = df.set_index("sequence")["guide_name"].to_dict()
    return seq_guide_dict

def parse_read_name(read_name_str):
    """
    Given a read name, returns (CBC, 10X_UMI).
    Assumes the standard format: e.g. 'lane3_AAACTTTGACCAGTGC-1_...someUMI'
    Adjust if your read names differ.
    """
    parts = read_name_str.split('_')
    if len(parts) < 3:
        # Fallback if the naming doesn't match expectations
        return ("", "")
    # CBC is parts[1], 10X_UMI is last part
    return (parts[1] + "-1", parts[-1])

def build_lane_guide_umi_df(
    assigned_bam_file,
    umi_file,
    seq_guide_dict,
):
    """
    Reads a single lane's data from:
        - assigned BAM with an 'XT' tag for the 20-bp sequence
        - a UMI file mapping read_name -> clone_UMI

    Deduplicates by [clone_UMI, CBC, 10X_UMI, gRNA].

    Args:
        assigned_bam_file (str): Path to the assigned_sorted.bam.
        umi_file (str): Path to the read_UMI.tsv or barcodes.tsv (read_name -> clone_UMI).
        seq_guide_dict (dict): {sequence_20bp: guide_name} from read_guide_seq_dict.

    Returns:
        pd.DataFrame with columns: [clone_UMI, gRNA, CBC, 10X_UMI].
    """

    # 1) read UMI file
    umi_df = pd.read_csv(umi_file, sep='\t', header=None)
    umi_df.columns = ['read_name', 'clone_UMI']
    # e.g. sometimes read_name has '>' or spaces. Adjust:
    umi_df['read_name'] = umi_df['read_name'].apply(lambda e: e.split(' ')[0][1:])

    # 2) read assigned BAM, parse out read_name -> 20-bp sequence
    read_names = []
    lb_refs = []
    bamfile = pysam.AlignmentFile(assigned_bam_file, "rb")
    for r in bamfile.fetch():
        read_names.append(r.query_name)
        # 'XT' tag is assumed to store the 20bp sequence
        lb_refs.append(r.get_tag("XT")[0:20])
    bamfile.close()

    df_tmp = pd.DataFrame({'read_name': read_names, 'sequence_20bp': lb_refs})
    # Convert sequence_20bp -> gRNA name
    df_tmp['gRNA'] = df_tmp['sequence_20bp'].apply(lambda seq: seq_guide_dict.get(seq, 'UNKNOWN_SEQ'))

    # 3) merge with UMI info
    merged = umi_df.merge(df_tmp, on='read_name', how='inner')

    # 4) parse CBC and 10X_UMI
    merged[['CBC', '10X_UMI']] = merged['read_name'].apply(lambda rn: pd.Series(parse_read_name(rn)))

    # 5) drop extraneous columns
    merged.drop(columns=['read_name', 'sequence_20bp'], inplace=True)

    # 6) deduplicate
    dedup = merged.drop_duplicates(['clone_UMI', 'CBC', 'gRNA', '10X_UMI'])

    return dedup

def build_guide_umi_df_for_all_lanes(
    lane_info,
    guide_seq_csv,
    output_all_path=None,
    overwrite=False,
):
    """
    Builds a combined DataFrame for ALL lanes (CBC, clone_UMI, gRNA, 10X_UMI).

    Args:
        lane_info (list of dict):
            e.g. [
              {
                'lane_id': 'lane1',
                'assigned_bam_file': '/path/to/lane1/assigned_sorted.bam',
                'umi_file': '/path/to/lane1/read_UMI.tsv'
              },
              ...
            ]
        guide_seq_csv (str): Path to guide-sequence CSV for mapping sequences -> guide_name.
        output_all_path (str or None): If given, the function will:
            - load from this file if it exists AND overwrite=False
            - else build from scratch and save to this file
        overwrite (bool): Whether to overwrite output_all_path if it exists.

    Returns:
        pd.DataFrame: Combined [clone_UMI, gRNA, CBC, 10X_UMI] across all lanes.
    """
    if (output_all_path is not None) and (not overwrite) and os.path.exists(output_all_path):
        return pd.read_csv(output_all_path, sep='\t')

    seq_guide_dict = read_guide_seq_dict(guide_seq_csv)

    dfs = []
    for lane_dict in lane_info:
        assigned_bam_file = lane_dict['assigned_bam_file']
        umi_file = lane_dict['umi_file']
        lane_id = lane_dict.get('lane_id', 'lane?')  # used only for debugging
        df_lane = build_lane_guide_umi_df(
            assigned_bam_file,
            umi_file,
            seq_guide_dict
        )
        # Optionally prefix the CBC to reflect the lane
        # If you prefer "lane1_XXXX-1" style, do:
        if 'lane_id' in lane_dict:
            df_lane['CBC'] = df_lane.apply(lambda r: f"{lane_dict['lane_id']}_{r['CBC']}", axis=1)
        dfs.append(df_lane)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Optionally save
    if output_all_path is not None:
        combined_df.to_csv(output_all_path, sep='\t', index=False)

    return combined_df

def pivot_CBC_construct_mtx(guide_umi_df, min_count=10, binarize=True):
    """
    Given the combined DataFrame of [CBC, clone_UMI, gRNA, 10X_UMI],
    produce a pivot table with CBC as rows, and "gRNA__clone_UMI" as columns.

    If min_count > 0, we require that each (CBC, clone_UMI, gRNA) triple
    have > min_count occurrences to be retained.

    Args:
        guide_umi_df (pd.DataFrame): [clone_UMI, gRNA, CBC, 10X_UMI].
        min_count (int): filter threshold.
        binarize (bool): if True, sets all nonzero to 1.

    Returns:
        pd.DataFrame: pivoted CBC-by-construct matrix, with counts or 0/1.
    """
    # 1) Count how many times each triple appears
    triple_counts = guide_umi_df.groupby(['CBC', 'clone_UMI', 'gRNA']).size().reset_index(name='count')
    # 2) Filter by min_count if necessary
    if min_count is not None and min_count > 0:
        triple_counts = triple_counts[triple_counts['count'] > min_count]
    # 3) pivot
    triple_counts['construct'] = triple_counts['gRNA'] + '__' + triple_counts['clone_UMI']
    pivot_df = triple_counts.pivot(index='CBC', columns='construct', values='count').fillna(0)
    if binarize:
        pivot_df[pivot_df > 0] = 1
    return pivot_df

def get_all_constructs_str_from_CBC(cbc, cbc_construct_mtx):
    """
    Given a single CBC and a CBC-construct matrix,
    return the string of all constructs joined by '___'.

    Args:
        cbc (str): e.g. 'lane1_AAATTTGG-1'
        cbc_construct_mtx (pd.DataFrame): pivot with rows=CBC, columns=construct, values>0 if present.

    Returns:
        str: e.g. "Trim28-1__AAAA___Setdb1-2__GGGG" or '' if none
    """
    if cbc not in cbc_construct_mtx.index:
        return ''
    row = cbc_construct_mtx.loc[cbc]
    # find columns with non-zero
    constructs = row[row > 0].index.tolist()
    return '___'.join(constructs)

def annotate_adata_with_clones(
    adata,
    cbc_construct_mtx,
    all_constructs_colname="all_constructs_str",
    num_constructs_colname="num_constructs"
):
    """
    Adds columns 'all_constructs_str' and 'num_constructs' to adata.obs
    based on the CBC-construct matrix.

    Args:
        adata (anndata.AnnData): adata whose obs_names are the CBCs.
        cbc_construct_mtx (pd.DataFrame): CBC-by-construct pivot table.
        all_constructs_colname (str): name of the column in adata.obs for the constructs string.
        num_constructs_colname (str): name of the column in adata.obs for the integer count of constructs.
    """
    # For each obs_name, find the constructs
    cbc_list = list(adata.obs_names)
    constructs_list = []
    for cbc in cbc_list:
        constructs_str = get_all_constructs_str_from_CBC(cbc, cbc_construct_mtx)
        constructs_list.append(constructs_str)

    adata.obs[all_constructs_colname] = constructs_list
    adata.obs[num_constructs_colname] = adata.obs[all_constructs_colname].apply(
        lambda s: 0 if s == '' else len(s.split('___'))
    )

##########################################################################################################
###################### Doublet Construct Calling Linear Program ###############################################
##########################################################################################################

import pandas as pd
from collections import Counter

def create_clone_construct_mtx(adata):
    # Create a dictionary with clones as keys and constructs as values
    clone_dict = {}
    for index, constructs_str in adata.obs['all_constructs_str'].items():
        constructs_list = [construct for construct in constructs_str.split('___') if construct != '']
        clone = constructs_str
        if clone not in clone_dict:
            clone_dict[clone] = constructs_list

    # Create the dataframe
    all_constructs = set().union(*[v for v in clone_dict.values() if v])
    clone_construct_count_mtx = pd.DataFrame(
        0, index=clone_dict.keys(), columns=sorted(all_constructs)
    )
    for clone, constructs in clone_dict.items():
        if constructs:  # Ensure that we are not indexing with an empty list
            clone_construct_count_mtx.loc[clone, constructs] = 1

    return clone_construct_count_mtx

def create_clone_cell_count(adata):
    clone_cell_count = Counter(adata.obs['all_constructs_str'])
    return clone_cell_count

from scipy.optimize import linprog

def solve_linear_program(clone_construct_count_mtx, clone_cell_count):
    # Create the objective function coefficients (c_i)
    c = [-clone_cell_count[clone] for clone in clone_construct_count_mtx.index]

    # Create the inequality constraint matrix (A_ub)
    A_ub = clone_construct_count_mtx.values.T

    # Create the inequality constraint vector (b_ub), which is 1 for each construct
    b_ub = [1] * clone_construct_count_mtx.shape[1]

    # Create the bounds for the decision variables
    bounds = [(0, 1)] * clone_construct_count_mtx.shape[0]

    # Solve the linear program
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    return res


def run_doublet_constrct_calling(adata):
    clone_construct_count_mtx = create_clone_construct_mtx(adata)
    clone_cell_count = create_clone_cell_count(adata)
    res = solve_linear_program(clone_construct_count_mtx, clone_cell_count)

    # The clones corresponding to the positions where res.x is 1
    clones_to_keep = clone_construct_count_mtx.index[res.x == 1]

    # The clones corresponding to the positions where res.x is 1
    clones_to_keep = clone_construct_count_mtx.index[res.x == 1]
    adata = adata[adata.obs['all_constructs_str'].isin(clones_to_keep)].copy()
    return adata

