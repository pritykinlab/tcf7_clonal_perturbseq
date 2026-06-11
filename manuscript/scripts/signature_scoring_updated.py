############################
# signature_scoring.py
############################

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional
import misc

###############################################################################
# 1. Custom colormap (unchanged from your code)
###############################################################################
def custom_cmap():
    """
    Return a modified RdBu_r colormap with an enhanced mid-segment.
    """
    original_cmap = plt.cm.RdBu_r
    # Extract the original colors from the RdBu_r colormap
    colors = original_cmap(np.linspace(0.1, 0.9, 256))

    # Define the segment to enhance in the middle
    mid_start = int(0.4 * 256)  # Start of the middle segment
    mid_end   = int(0.6 * 256)  # End of the middle segment
    mid_colors = colors[mid_start:mid_end]

    # Enhance the middle segment by repeating
    # (adjust the factor below as needed)
    enhancement_factor = 2
    enhanced_mid = np.repeat(mid_colors, enhancement_factor, axis=0)

    # Combine the colors with the enhanced middle segment
    new_colors = np.concatenate(
        (colors[:mid_start], enhanced_mid, colors[mid_end:]),
        axis=0
    )

    # Create a new colormap from the modified color list
    return LinearSegmentedColormap.from_list('extended_smooth_mid', new_colors)

###############################################################################
# 2. Gene signature dictionaries
###############################################################################

# Example TRM signatures (including ones from Mackay2013, Mackay2016, Milner2017, Crowl2022)
# Make sure you keep your full gene lists
def get_trm_signatures() -> Dict[str, List[str]]:
    """
    Return a dictionary of TRM-related signatures,
    with keys as signature names and values as lists of genes.
    """

    Mackay2013_up = [
       "Cd244a", "Cdh1", "Chn2", "Ctla4", "Hpgds", "Hspa1a", "Icos",
       "Inpp4b", "Itga1", "Itgae", "Litaf", "Zfp683",
       "Nr4a1", "Nr4a2", "Qpct", "Rgs1", "Rgs2", "Sik1", "Skil",
       "Tmem123", "Vps37b", "Xcl1"
    ] ## added Zfp683, removed LOC641050

    Mackay2016_up = [
        "Insig1", "Gsg2", "Ddx3x", "Dhcr24", "Ppp1r16b", "Klf6", "Btg2", "Cxcr6", "Fosb", "Jun", "Hspa5",
        "Nedd4", "Plk3", "Stard4", "Tnfaip3", "B4galnt4", "Cd244a", "Hobit", "Irf4", "Cish", "Mapkapk3",
        "Sik1", "Pygl", "Ctnna1", "Odc1", "Per1", "Dusp1", "Atf3", "Ldlrad4", "Sc4mol", "Junb", "Gpr56",
        "Nfkbid", "Fosl2", "Rgs2", "Nr4a2", "Dgat1", "Arrdc3", "Frmd4b", "Nr4a1", "Gpr171", "Smim3",
        "Eya2", "Gpr55", "Amica1", "Ly6g5b", "Csrnp1", "Traf4", "Zfp36", "Glrx", "Dusp5", "Litaf",
        "Gadd45b", "Ifng", "Osgin1", "Abi3", "8430419L09Rik", "Egr1", "Hilpda", "Skil", "Rnf149",
        "Hmgcs1", "P4hb", "Pnrc1", "Gpr34", "Ppp1r15a", "Itgae", "P2ry10", "Ehd1", "Dusp6", "Xcl1",
        "Spsb1", "Isg20", "Inpp4b", "Neurl3", "Hpgds", "Rhob", "Vdac1", "Lad1", "Fos", "Cdh1",
        "Cd69", "Qpct", "Hspd1"
    ] # Cd244 -> Cd244a

    Milner2017_up = [
        "2900026A02Rik", "AA467197", "Abcb1b", "Adam19", "Atf3", "B4galnt4",
        "Bag3", "Bhlhe40", "Btg2", "Btg3", "Ccl4", "Cd244a", "Cd69", "Cdh1",
        "Chd7", "Chn2", "Cish", "Coq10b", "Cpd", "Crem", "Ctla4", "Dennd4a",
        "Dgat1", "Dnaja1", "Dnaja4", "Dnajb1", "Dnajb4", "Dnajb6", "Dusp1",
        "Dusp4", "Dusp6", "Egr1", "Ehd1", "Ell2", "Fasl", "Fbxo30", "Fgl2",
        "Fmnl3", "Fos", "Fosl2", "Gabarapl1", "Gadd45b", "Gem", "Glrx",
        "Gpr171", "Adgrg1", "Gzmb", "H3f3b", "Havcr2", "Hip1", "Hpgds",
        "Hsp90aa1", "Hspa5", "Icos", "Ifng", "Ifrd1", "Il21r", "Il4ra",
        "Inpp4b", "Irf4", "Isg20", "Isy1", "Itga1", "Jun", "Junb", "Kdm6b",
        "Litaf", "Lmnb1", "Ly6g5b", "Mxd1", "Mxi1", "Neurl3", "Nfil3",
        "Nfkbie", "Nr4a1", "Nr4a2", "Nr4a3", "Ntan1", "Odc1", "P2ry10",
        "P4hb", "Pdcd1", "Per1", "Plk3", "Plscr1", "Pmepa1", "Ppp1r15a",
        "Prdx5", "Ptp4a1", "Qpct", "Rbpj", "Rgs1", "Rgs10", "Rgs16", "Rhob",
        "Rnf149", "Rrad", "Sik1", "Skil", "Slc16a6", "Slc7a5", "Spty2d1",
        "Ssbp2", "Stk17b", "Tgif1", "Tigit", "Tiparp", "Tjp1", "Tnf",
        "Tnfaip3", "Tnfrsf1b", "Tnfrsf9", "Tra2a", "Trib1", "Trp53inp2",
        "Ube2s", "Vps37b", "Wsb1", "Xcl1", "Zfand5", "Zfp36l1"
    ] ## added Cd244a (removed Cd244), Prdx5 (removed Prdx6), Adgrg1 (removed Gpr56)

    Crowl2022_up = [
        "2900026A02Rik", "5830444F18Rik", "7330423F06Rik", "A630081D01Rik", "AA467197",
        "Abcb1b", "Abcb9", "Aldh7a1", "Arid5a", "Arl5b", "Asb2", "Atf3", "Atf4", "Atg101",
        "B130021K23Rik", "Baiap2", "Bambi", "Bcl2a1b", "Bcl2a1d", "Bcl2l11", "Bend4", "Bhlhe40",
        "Btg2", "C920009B18Rik", "Calca", "Cblb", "Ccl4", "Cd40lg", "Cd69", "Cd86", "Cdkn1a",
        "Cep170b", "Cish", "Ckb", "Clk1", "Coq10b", "Crem", "Csrnp1", "Ctla4", "Cxcl10",
        "Cyp51", "Dbf4", "Ddx3x", "Dennd4a", "Dgat1", "Dhx40", "Dnajb1", "Dnajb9", "Dtl",
        "Dtx4", "Dusp1", "Dusp10", "Dusp4", "Dusp5", "Dusp6", "Dusp8", "Egr1", "Egr2", "Egr3",
        "Ell2", "Elmsan1", "Ern1", "Errfi1", "Fam110a", "Fam83d", "Fasl", "Fbxo34", "Fos",
        "Fosb", "Fosl2", "Frmd4b", "Furin", "Gadd45b", "Gch1", "Gem", "Gm17971", "Gm19327",
        "Gm20743", "Gm24347", "Gm26532", "Gm26982", "Gm31513", "Gm36933", "Gm37033",
        "Gm37039", "Gm37084", "Gm37390", "Gm37531", "Gm37728", "Gm37909", "Gm38126",
        "Gm42748", "Gm43336", "Gm43411", "Gm43430", "Gm43963", "Gm44292", "Gm47088",
        "Gm47586", "Gm47588", "Gm47603", "Gm48302", "Gm48986", "Gm5106", "Gm6109", "Gm8818",
        "Got1", "Gpr132", "Gpr171", "Gpr34", "Grb7", "Hexim1", "Hilpda", "Hk2", "Hspa2",
        "Hspa5", "Icos", "Ier2", "Ifng", "Ifngr1", "Ifrd1", "Il12rb2", "Il21r", "Il4ra",
        "Insig1", "Irak2", "Irf4", "Isca1", "Isg15", "Isy1", "Itpkb", "Jaml", "Jmy", "Jun",
        "Junb", "Kdm2b", "Kdm6b", "Kdm6bos", "Klf4", "Klf6", "Klf9", "Kpna1", "Lilr4b",
        "Lilrb4a", "Litaf", "Lrrc8d", "Lzts1", "Mafk", "Mir22hg", "Mpp7", "Mxd1", "Mxi1",
        "Nabp1", "Neurl3", "Nfil3", "Nfkb2", "Nfkbia", "Nfkbid", "Nfkbiz", "Ninj1", "Nmrk1",
        "Nr4a1", "Nr4a2", "Nr4a3", "Odc1", "Osgin1", "P2rx7", "P2ry10", "Pdcd1", "Pdcd1lg2",
        "Pde4b", "Pde4d", "Per1", "Pim3", "Plac8", "Plcxd2", "Plk2", "Plk3", "Pmaip1",
        "Pmepa1", "Pnrc1", "Polg2", "Ppp1r15a", "Ppp1r16b", "Ppp1r3b", "Ppp2r2c", "Prdx5",
        "Psd", "Ptger4", "Ptpn22", "Pygl", "Qpct", "Ramp3", "Rasl11b", "Rel", "Rgcc", "Rgs1",
        "Rgs16", "Rgs2", "Rhob", "Rnf157", "Rnf19a", "Rnf19b", "Rrad", "Samsn1", "Selenon",
        "Serpinb6b", "Serpinb9", "Sertad1", "Sh2d2a", "Sik1", "Skil", "Slamf1", "Slc16a10",
        "Slc16a6", "Slc25a25", "Slc35d1", "Slc38a2", "Slc3a2", "Smad7", "Smim3", "Smox",
        "Snx18", "Socs1", "Socs3", "Spsb1", "Spty2d1", "Srgn", "Suco", "Taf7", "Tex2",
        "Tgif1", "Tigit", "Tinf2", "Tiparp", "Tnf", "Tnfaip3", "Tnfrsf10b", "Tnfrsf1b",
        "Tnfrsf9", "Tob2", "Traf4", "Trp53inp2", "Tubb2a", "Ubald2", "Ubc", "Usp2", "Vgll4",
        "Vps37b", "Wsb1", "Xcl1", "Ybx3", "Zbtb21", "Zc3h12a", "Zfp36l1", "Zswim4",
        "4930417O13Rik", "5430416N02Rik", "B4galnt4", "Ccpg1", "Cd93", "Cers6", "Chka",
        "Coro2a", "Cry2", "D630030B08Rik", "Dynlt3", "Eif4e3", "Enpp1", "Fgl2", "Gm12167",
        "Gm15638", "Gm30541", "Gm37352", "Gm37510", "Gm37790", "Gm44187", "Gm47585",
        "Gm6225", "Gsn", "Ier5l", "Lin54", "Map1lc3a", "Mthfr", "Rnf125", "Spats2",
        "Sult2b1", "Tmx4", "Trim36", "Zeb1", "Zfp36l2", "Zfp683", "Abcb1a", "Abcb4", "Actn2",
        "Actrt3", "Adgrg1", "Apol9b", "B3galt5", "Bcl6", "Btg1", "Cables1", "Car2", "Cbx4",
        "Ccl3", "Cd38", "Cdh1", "Cemip2", "Chn2", "Cpd", "Dnaja1", "Dusp2", "Ehd1",
        "Eif2ak3", "Enpp5", "Eya2", "Fbxl21", "Fmnl3", "Galns", "Gcnt1", "Gcnt2", "Gm11707",
        "Gm19587", "Gm38257", "Gm47447", "Gm49395", "Grn", "Gzmk", "H1f2", "H2bc4", "Havcr2",
        "Hip1", "Homer1", "Hpgds", "Hspa1b", "Inpp4b", "Isg20", "Itga1", "Ly6g5b", "Maff",
        "Malt1", "Med10", "Mgll", "Mical2", "Mmp11", "Myo6", "Padi2", "Pip5k1b", "Pvrig",
        "Rftn1", "Rgs10", "Rsad2", "Siah2", "Sipa1l1", "Snhg12", "Snx33", "Stat1", "Tmem176b",
        "Tmem69", "Tnip1", "Vpreb3", "Zc3h12d"
    ] ## added Prdx5 (removed Prdx6)



    return {
        "TRM Mackay et al., 2013": Mackay2013_up,
        "TRM Mackay et al., 2016": Mackay2016_up,
        "TRM Milner et al., 2017": Milner2017_up,
        "TRM Crowl et al., 2022":  Crowl2022_up,
    }


def get_cell_cycle_signatures(s_genes_path: str, g2m_genes_path: str) -> Dict[str, List[str]]:
    """
    Read S-phase and G2M-phase gene lists from CSV files.
    Convert them to the typical mouse capitalization if needed.
    Return a dictionary with two keys: 'S phase' and 'G2/M phase'.
    """
    s_genes = pd.read_csv(s_genes_path, header=None).iloc[:, 0].tolist()
    g2m_genes = pd.read_csv(g2m_genes_path, header=None).iloc[:, 0].tolist()

    # Convert to "capital first letter, rest lower" if you like
    s_formatted = [gene.capitalize() for gene in s_genes]
    g2m_formatted = [gene.capitalize() for gene in g2m_genes]

    return {
        "S phase": s_formatted,
        "G2/M phase": g2m_formatted
    }

def get_yuri_signatures(
    comparison_names: Dict[str, str],
    padj_threshold: float = 0.05,
    top_n: int = 20
) -> Dict[str, List[str]]:
    """
    Given a directory containing multiple TSV files from the 'mol_cell_yuri' analysis,
    read each file, parse the significance, and extract top up/down gene sets.

    :param directory_path: path to the folder containing *.tsv files
    :param comparison_names: dict of short file-key -> 'Group1 vs Group2' string
    :param padj_threshold: significance cutoff
    :param top_n: number of top genes by absolute log2FC
    :return: aggregated dictionary of signatures:
             e.g. {'Up Group1 vs Group2': [...], 'Up Group2 vs Group1': [...]}
    """

    directory_path = "../data/signatures/mol_cell_yuri"
    # Read all TSV files into a dictionary
    data_frames = {}
    for filename in os.listdir(directory_path):
        if not filename.endswith(".tsv"):
            continue
        filepath = os.path.join(directory_path, filename)
        df = pd.read_csv(filepath, sep='\t')
        key = filename.replace(".tsv", "")
        data_frames[key] = df

    # The final signatures
    all_signatures = {}

    for filekey, df in data_frames.items():
        if filekey not in comparison_names:
            print(f"Skipping {filekey}, not in comparison_names dictionary.")
            continue

        group1_vs_group2 = comparison_names[filekey]  # e.g. "func vs dysfunc"
        group1, group2 = group1_vs_group2.split(" vs ")

        # Filter by significance
        significant = df[df['padj'] < padj_threshold]

        # Positive log2FC = up in group1
        pos = significant[significant['log2FC'] > 0].sort_values('log2FC', ascending=False).head(top_n)
        # Negative log2FC = up in group2
        neg = significant[significant['log2FC'] < 0].sort_values('log2FC', ascending=True).head(top_n)

        up_g1_key = f'Up {group1} vs {group2}'
        up_g2_key = f'Up {group2} vs {group1}'

        if len(pos) >= 5:
            all_signatures[up_g1_key] = pos['gene'].tolist()

        if len(neg) >= 5:
            all_signatures[up_g2_key] = neg['gene'].tolist()

    return all_signatures

import pandas as pd

def get_miller_signatures(
    file_path: str = "../data/signatures/Miller.xlsx",
    sheet_name: str = "Tumor Prog. vs. Tumor Terminal",
    padj_threshold: float = 0.05,
    top_n: int = 20
) -> dict:
    """
    Load the specified sheet from Miller.xlsx and split genes by sign of log2FoldChange:
      - "Up Tumor Prog vs Tumor Terminal": top genes with log2FoldChange > 0
      - "Up Tumor Terminal vs Tumor Prog": top genes with log2FoldChange < 0

    Both subsets are filtered by padj < padj_threshold and sorted by absolute log2FoldChange.
    The top_n genes from each subset are returned.

    :param file_path: Path to Miller.xlsx
    :param sheet_name: Sheet name to read within Miller.xlsx
    :param padj_threshold: Significance cutoff for padj
    :param top_n: Number of top genes to return by descending absolute log2FoldChange
    :return: A dictionary with two keys:
             {
               'Up Tumor Prog vs Tumor Terminal': [list of top_n gene names],
               'Up Tumor Terminal vs Tumor Prog': [list of top_n gene names]
             }
    """
    # Load the specified sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Filter to significant rows
    df_sig = df[df['padj'] < padj_threshold].copy()

    # 1) Genes up in Tumor Prog (positive log2FoldChange)
    pos_genes_df = (
        df_sig[df_sig['log2FoldChange'] > 0]
        .sort_values('log2FoldChange', ascending=False)
        .head(top_n)
    )
    up_prog_genes = pos_genes_df['Gene'].tolist()

    # 2) Genes up in Tumor Terminal (negative log2FoldChange)
    neg_genes_df = (
        df_sig[df_sig['log2FoldChange'] < 0]
        .sort_values('log2FoldChange', ascending=True)
        .head(top_n)
    )
    up_terminal_genes = neg_genes_df['Gene'].tolist()

    return {
        "Up Tumor Prog vs Tumor Terminal": up_prog_genes,
        "Up Tumor Terminal vs Tumor Prog": up_terminal_genes
    }


def get_immune_signatures_from_excel(
    sheet_name: str = 'T_cell_CD8',
    fdr_threshold: float = 0.05,
    top_n: int = 20
) -> Dict[str, List[str]]:
    """
    Load an Excel file with columns like ["Gene", "FDR", "Avg_log2FC", "Cytokine_Str"],
    extract up/down signatures per cytokine.

    :param excel_path: path to the .xlsx file
    :param sheet_name: name of the sheet to parse
    :param fdr_threshold: significance cutoff
    :param top_n: number of top genes by absolute log2FC
    :return: dict with keys like 'Up IL2', 'Down IL2', etc.
    """
    excel_path = '../data/signatures/immune_cell_dict.xlsx'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cytokine_signatures = {}
    for cytokine in df['Cytokine_Str'].unique():
        sub = df[df['Cytokine_Str'] == cytokine]
        significant = sub[sub['FDR'] < fdr_threshold]

        up   = significant[significant['Avg_log2FC'] > 0].sort_values('Avg_log2FC', ascending=False).head(top_n)
        down = significant[significant['Avg_log2FC'] < 0].sort_values('Avg_log2FC', ascending=True).head(top_n)

        if len(up) >= 5:
            cytokine_signatures[f'Up {cytokine}'] = up['Gene'].tolist()
        if len(down) >= 5:
            cytokine_signatures[f'Down {cytokine}'] = down['Gene'].tolist()

    return cytokine_signatures

def get_burn2025_signatures() -> Dict[str, List[str]]:
    """
    Return Burn et al., 2026 TRM and TEX signatures from text files.

    :return: Dictionary with TRM and TEX signatures from Burn et al., 2026
    """
    # Read TRM signature (keeping original file name)
    with open('../data/signatures/Mackay2025_TRM.txt', 'r') as f:
        burn2025_trm = [line.strip() for line in f if line.strip()]

    # Read TEX signature (keeping original file name)
    with open('../data/signatures/Mackay2025_TEX.txt', 'r') as f:
        burn2025_tex = [line.strip() for line in f if line.strip()]

    return {
        "TRM Burn et al., 2026": burn2025_trm,
        "TEX Burn et al., 2026": burn2025_tex
    }

def get_park2025_signatures() -> Dict[str, List[str]]:
    """
    Return Park et al., 2026 shared TRM/TR-TEX signature (vs TCIRCM) from Excel file.

    :return: Dictionary with shared TRM/TR-TEX signature from Park et al., 2026
    """
    import pandas as pd

    sheet_path = "../data/signatures/Wherry2025_v2.xlsx"
    sheet_name = "UP_TRM_or_TR_TEX_vs_TCIRCM"

    df = pd.read_excel(sheet_path, sheet_name=sheet_name, skiprows=1)
    shared_genes = df.loc[df["Status"] == "Shared", "Gene"].dropna().astype(str).tolist()
    trm_only_genes = df.loc[df["Status"] == "TRM", "Gene"].dropna().astype(str).tolist()

    # Combine TRM-only genes with shared genes for the TRM signature, keeping order and uniqueness
    trm_genes = list(dict.fromkeys(trm_only_genes + shared_genes))

    return {
        "TRM/TR-TEX Park et al., 2026": shared_genes,
        "TRM Park et al., 2026": trm_genes
    }

###############################################################################
# 3. Compute signature scores
###############################################################################

def compute_score(adata, gene_list, score_name, zscore_layer=None):
    """
    Creates adata.obs[score_name] by averaging expression of gene_list
    using 'misc.my_score_genes' (which you have).
    """
    genes_in_data = [g for g in gene_list if g in adata.var_names]
    if not genes_in_data:
        print(f"[Warning] No overlap for {score_name}. Setting to 0.")
        adata.obs[score_name] = 0
        return

    # We’ll call my_score_genes with or without zscore_layer
    # depending on your function signature for my_score_genes
    if zscore_layer is not None:
        misc.my_score_genes(adata, genes_in_data, score_name, zscore_layer=zscore_layer)
    else:
        misc.my_score_genes(adata, genes_in_data, score_name)

def compute_signatures(
    adata: sc.AnnData,
    signatures: dict,
    groupby: str = "cell_type",
    zscore_layer: str = None
) -> pd.DataFrame:
    """
    For each signature, attach a new col in adata.obs,
    then group by `groupby` to get average signature scores.
    """
    # We'll do it in-place on a copy to avoid polluting the original obs
    adata_copy = adata.copy()

    for sig_name, genes in signatures.items():
        compute_score(adata_copy, genes, sig_name, zscore_layer=zscore_layer)

    df_scores = adata_copy.obs.groupby(groupby)[list(signatures.keys())].mean()
    return df_scores  # shape: (groups) x (signatures)


###############################################################################
# 4. Heatmap functions
###############################################################################

def plot_signatures_heatmap(
    score_df: pd.DataFrame,
    row_order: Optional[List[str]] = None,
    col_order: Optional[List[str]] = None,
    center: float = 0,
    vmin: float = -1,
    vmax: float = 1,
    row_cluster: bool = True,
    col_cluster: bool = False,
    figsize: tuple = (12, 8),
    annot: bool = True,
    cmap=None,
    cbar_pos=(1.05, 0.5, 0.02, 0.3),
    fmt: str = ".2f",
    xlabel: str = "",
    ylabel: str = "",
    title: str = ""
):
    """
    Create a clustermap (heatmap) of signature scores
    where rows are signatures and columns are groups (e.g. cell types).
    """
    # row_order and col_order allow you to reorder the heatmap
    plot_df = score_df.copy()
    if row_order is not None:
        row_order = [r for r in row_order if r in plot_df.index]
        plot_df = plot_df.loc[row_order]
    if col_order is not None:
        col_order = [c for c in col_order if c in plot_df.columns]
        plot_df = plot_df[col_order]

    if cmap is None:
        cmap = custom_cmap()

    # We typically want rows=signatures, columns=groups
    g = sns.clustermap(
        plot_df,
        method='average',
        metric='euclidean',
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cbar_pos=cbar_pos
    )

    # Some styling
    g.ax_heatmap.set_xlabel(xlabel)
    g.ax_heatmap.set_ylabel(ylabel)
    if title:
        g.fig.suptitle(title, y=1.02)

    # Rotate x-labels for clarity
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=90)
    # Keep y-labels horizontal
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), rotation=0)

    plt.show()
    return g


def plot_gene_heatmaps_per_signature(
    adata: sc.AnnData,
    signature_dict: Dict[str, List[str]],
    groupby: str,
    zscore_layer: str,
    output_dir: str,
    group_order: Optional[List[str]] = None,
    vmin: float = -2,
    vmax: float = 2,
    cmap=None,
    row_cluster: bool = True,
    col_cluster: bool = False
):
    """
    For each signature in `signature_dict`, create a heatmap that shows
    average z-scored expression of the signature's genes, grouped by `groupby`.

    - If `groupby = "cell_type"`, columns will be cell types.
    - If `groupby = "perturbation"`, columns will be perturbations.

    Saves one heatmap per signature in `output_dir`.
    """
    if cmap is None:
        cmap = custom_cmap()

    # Make sure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert adata.layers[zscore_layer] to a DataFrame: cells x genes
    mat = adata.layers[zscore_layer]
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    df_zscore = pd.DataFrame(mat, index=adata.obs_names, columns=adata.var_names)
    # Add group info
    df_zscore[groupby] = adata.obs[groupby].values

    # Group by and average
    grouped = df_zscore.groupby(groupby).mean()

    # If a group_order is provided, reorder
    if group_order is not None:
        # Only keep the ones actually in grouped
        group_order = [g for g in group_order if g in grouped.index]
        grouped = grouped.loc[group_order]

    for sig_name, gene_list in signature_dict.items():
        # Filter to genes present in grouped.columns
        valid_genes = [g for g in gene_list if g in grouped.columns]
        if len(valid_genes) == 0:
            print(f"No overlapping genes for signature '{sig_name}'. Skipping.")
            continue

        # Subset to these genes, transpose so rows=genes, columns=groups
        matrix = grouped[valid_genes].T

        # Plot as a clustermap
        g = sns.clustermap(
            matrix,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            method='average',
            metric='euclidean',
            xticklabels=True,
            yticklabels=True,
            figsize=(8, max(4, 0.4 * matrix.shape[0] + 3))
        )

        g.fig.suptitle(sig_name, fontsize=14, y=1.02)
        g.ax_heatmap.set_xlabel(groupby.capitalize())
        g.ax_heatmap.set_ylabel("Genes")

        # Save figure
        fig_path = os.path.join(output_dir, f"{sig_name.replace(' ', '_')}.png")
        g.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        print(f"Saved {fig_path}")
