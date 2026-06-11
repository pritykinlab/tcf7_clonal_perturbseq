#!/usr/bin/env python
"""Generate the linked Figure S2a-b gene/pathway panel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"


CONTROL_PREFIXES = ("NO_SITE", "INACTIVE", "POTENTIALLY_ACTIVE")
PLOT_STEM = "a_b"
GENE_P_THRESHOLD = 0.05
GENE_NEG_LOG10_P_THRESHOLD = -math.log10(GENE_P_THRESHOLD)
PATHWAY_P_THRESHOLD = 0.05
REQUIRED_GUIDES_PER_TARGET_GENE = 4

SCREEN_SPECS = [
    {
        "arg": "d7_tcf7",
        "condition": "D7 Tcf7 low vs High Acute Stim",
        "short_label": "D7 Tcf7 acute",
    },
    {
        "arg": "d7_cont_tcf7",
        "condition": "D7 Tcf7 low vs High Chronic Stim",
        "short_label": "D7 Tcf7 chronic",
    },
    {
        "arg": "d5_cont_prdm1",
        "condition": "D5 Prdm1 low vs high Chronic Stim",
        "short_label": "D5 Prdm1 chronic",
    },
    {
        "arg": "d7_prdm1",
        "condition": "D7 Prdm1 low vs high",
        "short_label": "D7 Prdm1",
    },
    {
        "arg": "d7_eomes",
        "condition": "D7 Eomes low vs high",
        "short_label": "D7 Eomes",
    },
    {
        "arg": "d5_tox",
        "condition": "D5 Tox low vs high Hypoxic",
        "short_label": "D5 Tox hypoxic",
    },
]

GENE_GROUPS = [
    ("TCR", ["Cd3d", "Cd3e", "Lcp2", "Zap70", "Lck", "Itk", "Ptpn22"]),
    (
        "Cytokine / JAK-STAT",
        [
            "Il12rb1",
            "Jak2",
            "Tyk2",
            "Stat4",
            "Stat1",
            "Ifngr1",
            "Ifngr2",
            "Stat5a",
            "Stat5b",
            "Jak1",
            "Jak3",
            "Il2rb",
            "Il2rg",
            "Il2ra",
            "Il21r",
            "Stat3",
            "Socs3",
        ],
    ),
    (
        "mTOR / hypoxia / FOXO",
        [
            "Hif1a",
            "Cdkn1a",
            "Mtor",
            "Mlst8",
            "Rptor",
            "Rictor",
            "Lamtor1",
            "Lamtor2",
            "Lamtor3",
            "Pten",
            "Foxo1",
            "Stk11",
            "Tsc2",
            "Vhl",
            "Eloc",
            "Egln3",
        ],
    ),
    (
        "Fate regulators",
        [
            "Tcf7",
            "Prdm1",
            "Eomes",
            "Runx3",
            "Ets1",
            "Myb",
            "Batf",
            "Brd4",
            "Wnt9a",
            "Csnk2b",
            "Myc",
        ],
    ),
    (
        "Chromatin",
        [
            "Arid1a",
            "Smarcb1",
            "Smarcc1",
            "Smarce1",
            "Actl6a",
            "Dpf2",
            "Nipbl",
            "Mau2",
            "Smarca4",
            "Chd8",
            "Ep300",
            "Men1",
            "Tada2b",
            "Actr5",
            "Chd4",
            "Ruvbl1",
        ],
    ),
    (
        "Replication / ribosome",
        [
            "Dbf4",
            "Cdc45",
            "Pcna",
            "Dkc1",
            "Rpl31",
            "Pds5a",
            "Pds5b",
            "Smc3",
            "Cdc6",
            "Gins1",
            "Birc5",
            "Orc6",
            "Cdc7",
            "Bop1",
            "Wdr75",
            "Utp4",
            "Nop10",
            "Dimt1",
            "Pwp2",
        ],
    ),
]

PATHWAY_GROUPS = [
    (
        "TCR signaling",
        [
            ("T cell receptor signaling pathway", "GO", "T Cell Receptor Signaling Pathway (GO:0050852)"),
        ],
    ),
    (
        "Cytokine / JAK-STAT",
        [
            ("Interleukin-2 signaling", "REACTOME", "Interleukin-2 Signaling R-HSA-9020558"),
            ("Interleukin-12-mediated signaling pathway", "GO", "Interleukin-12-Mediated Signaling Pathway (GO:0035722)"),
            ("Type II interferon-mediated signaling pathway", "GO", "Type II Interferon-Mediated Signaling Pathway (GO:0060333)"),
            ("JAK-STAT signaling pathway", "KEGG", "JAK-STAT signaling pathway"),
        ],
    ),
    (
        "mTOR / AKT / hypoxia",
        [
            ("PIP3 activates AKT signaling", "REACTOME", "PIP3 Activates AKT Signaling R-HSA-1257604"),
            ("mTOR signaling pathway", "KEGG", "mTOR signaling pathway"),
            ("HIF-1 signaling pathway", "KEGG", "HIF-1 signaling pathway"),
        ],
    ),
    (
        "Fate / transcription",
        [
            ("Regulation of T cell differentiation", "GO", "Regulation of T Cell Differentiation (GO:0045580)"),
            ("TCF-dependent signaling in response to WNT", "REACTOME", "TCF Dependent Signaling In Response To WNT R-HSA-201681"),
            ("Regulation of transcription by RNA polymerase II", "GO", "Regulation of Transcription by RNA Polymerase II (GO:0006357)"),
        ],
    ),
    (
        "Chromatin",
        [
            ("Chromatin remodeling", "GO", "Chromatin Remodeling (GO:0006338)"),
            ("SWI/SNF complex", "GOCC", "SWI/SNF Complex (GO:0016514)"),
            ("Nucleosomal DNA binding", "GOMF", "Nucleosomal DNA Binding (GO:0031492)"),
            ("Cohesin loading onto chromatin", "REACTOME", "Cohesin Loading Onto Chromatin R-HSA-2470946"),
        ],
    ),
    (
        "Cell-cycle / RNA machinery",
        [
            ("Cell cycle", "REACTOME", "Cell Cycle R-HSA-1640170"),
            ("DNA replication", "REACTOME", "DNA Replication R-HSA-69306"),
            ("rRNA processing", "REACTOME", "rRNA Processing R-HSA-72312"),
        ],
    ),
]

PATHWAY_LINE_COLORS = ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for spec in SCREEN_SPECS:
        parser.add_argument(f"--{spec['arg'].replace('_', '-')}", required=True)
    parser.add_argument("--all-pathway-enrichment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot-stem", default=PLOT_STEM)
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum selected-gene/pathway overlap required to use a pathway log2 odds ratio.",
    )
    return parser.parse_args()


def gene_key(value: object) -> str:
    return str(value).strip().upper()


def selected_genes() -> pd.DataFrame:
    rows = []
    order = 0
    for group, genes in GENE_GROUPS:
        for gene in genes:
            rows.append({"gene": gene, "group": group, "order": order})
            order += 1
    return pd.DataFrame(rows)


def selected_pathways() -> pd.DataFrame:
    rows = []
    order = 0
    for group, pathways in PATHWAY_GROUPS:
        for label, source, pathway in pathways:
            rows.append({"label": label, "source": source, "pathway": pathway, "group": group, "order": order})
            order += 1
    return pd.DataFrame(rows)


def is_target_gene(series: pd.Series) -> pd.Series:
    return ~series.str.contains("INTERGENIC", na=False) & ~series.str.startswith(CONTROL_PREFIXES, na=False)


def load_screen(path: Path, condition: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").rename(
        columns={
            "Gene Symbol": "Gene",
            "Average LFC": "Gene LFC",
            "Average -log(p-values)": "Gene -log10(p)",
            "Number of perturbations": "Guides_used",
            "Perturbations": "Perturbations",
        }
    )
    required = ["Gene", "Gene LFC", "Gene -log10(p)", "Guides_used", "Perturbations"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    df = df[required].copy()
    for column in ["Gene LFC", "Gene -log10(p)", "Guides_used"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[np.isfinite(df["Gene LFC"]) & np.isfinite(df["Gene -log10(p)"])]
    df = df[is_target_gene(df["Gene"])]
    df = df[df["Guides_used"] == REQUIRED_GUIDES_PER_TARGET_GENE]
    df["Gene p-value"] = 10.0 ** (-df["Gene -log10(p)"].astype(float))
    df["significant"] = df["Gene -log10(p)"].astype(float) >= GENE_NEG_LOG10_P_THRESHOLD
    df["screen"] = label
    df["condition"] = condition
    return df.drop_duplicates("Gene", keep="first").reset_index(drop=True)


def load_screens(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    screens = {}
    for spec in SCREEN_SPECS:
        path = Path(getattr(args, spec["arg"]))
        screens[spec["short_label"]] = load_screen(path, spec["condition"], spec["short_label"])
    return screens


def filter_selected_genes_by_significance(genes: pd.DataFrame, screens: dict[str, pd.DataFrame]) -> pd.DataFrame:
    significant_genes = set()
    for df in screens.values():
        significant_genes.update(df.loc[df["significant"], "Gene"].tolist())
    included = genes[genes["gene"].isin(significant_genes)].copy().reset_index(drop=True)
    excluded = genes[~genes["gene"].isin(significant_genes)]["gene"].tolist()
    if excluded:
        print(
            "Excluded genes without PoolQ hypergeometric p < 0.05 in any selected comparison: "
            + ", ".join(excluded)
        )
    if included.empty:
        raise ValueError("None of the selected genes are significant in the PoolQ hypergeometric exports.")
    included["order"] = np.arange(len(included))
    return included


def collect_gene_support(
    screens: dict[str, pd.DataFrame],
    genes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = [spec["short_label"] for spec in SCREEN_SPECS]
    selected_gene_list = genes["gene"].tolist()
    matrix = pd.DataFrame(index=selected_gene_list, columns=labels, dtype=float)
    neglogp_matrix = pd.DataFrame(index=selected_gene_list, columns=labels, dtype=float)
    rows = []
    for label in labels:
        df = screens[label].set_index("Gene")
        for gene in selected_gene_list:
            if gene not in df.index:
                continue
            record = df.loc[gene]
            rows.append(
                {
                    "Gene": gene,
                    "screen": label,
                    "condition": record["condition"],
                    "Gene LFC": record["Gene LFC"],
                    "Gene -log10(p)": record["Gene -log10(p)"],
                    "Gene p-value": record["Gene p-value"],
                    "Guides_used": record["Guides_used"],
                    "significant": bool(record["significant"]),
                }
            )
            if bool(record["significant"]):
                matrix.loc[gene, label] = float(record["Gene LFC"])
                neglogp_matrix.loc[gene, label] = float(record["Gene -log10(p)"])
    all_stats = pd.DataFrame(rows)
    summary_rows = []
    for row in genes.itertuples(index=False):
        sub = all_stats[(all_stats["Gene"] == row.gene) & all_stats["significant"]].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["Gene p-value", "screen"], ascending=[True, True], kind="mergesort").iloc[0]
        summary_rows.append(
            {
                "Gene": row.gene,
                "group": row.group,
                "significant_screen_count": int(sub["screen"].nunique()),
                "best_screen": best["screen"],
                "best_direction": "up" if float(best["Gene LFC"]) > 0 else "down",
                "best_lfc": best["Gene LFC"],
                "best_neg_log10_p": best["Gene -log10(p)"],
                "best_pvalue": best["Gene p-value"],
            }
        )
    return pd.DataFrame(summary_rows), matrix, neglogp_matrix


def order_gene_outputs(
    genes: pd.DataFrame,
    gene_summary: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    gene_neglogp_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_order = {group: idx for idx, group in enumerate(genes["group"].drop_duplicates())}
    screen_order = {screen: idx for idx, screen in enumerate(gene_matrix.columns)}
    rows = []
    for row in genes.itertuples(index=False):
        neglogp = gene_neglogp_matrix.loc[row.gene].dropna()
        lfc = gene_matrix.loc[row.gene, neglogp.index] if not neglogp.empty else pd.Series(dtype=float)
        if neglogp.empty:
            best_screen_order = len(screen_order)
            direction_order = 2
            total_neglogp = 0.0
            max_abs_lfc = 0.0
        else:
            best_screen = neglogp.idxmax()
            best_lfc = float(gene_matrix.loc[row.gene, best_screen])
            best_screen_order = screen_order[best_screen]
            direction_order = 0 if best_lfc > 0 else 1
            total_neglogp = float(neglogp.sum())
            max_abs_lfc = float(lfc.abs().max())
        rows.append(
            {
                "gene": row.gene,
                "group": row.group,
                "initial_order": int(row.order),
                "group_order": group_order[row.group],
                "significant_screen_count": int(neglogp.shape[0]),
                "total_neg_log10_p": total_neglogp,
                "best_screen_order": best_screen_order,
                "direction_order": direction_order,
                "max_abs_lfc": max_abs_lfc,
            }
        )
    ordered = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "group_order",
                "significant_screen_count",
                "total_neg_log10_p",
                "best_screen_order",
                "direction_order",
                "max_abs_lfc",
                "initial_order",
            ],
            ascending=[True, False, False, True, True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    ordered_genes = ordered[["gene", "group"]].copy()
    ordered_genes["order"] = np.arange(len(ordered_genes))
    gene_order = ordered_genes["gene"].tolist()
    ordered_summary = (
        gene_summary.set_index("Gene")
        .loc[gene_order]
        .reset_index()
    )
    return (
        ordered_genes,
        ordered_summary,
        gene_matrix.loc[gene_order],
        gene_neglogp_matrix.loc[gene_order],
    )


def collect_pathway_support(
    all_pathway_enrichment_path: Path,
    pathways: pd.DataFrame,
    min_overlap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    all_results = pd.read_csv(all_pathway_enrichment_path, sep="\t")
    required = [
        "source",
        "pathway",
        "screen",
        "direction",
        "column",
        "tested",
        "log2_odds_ratio",
        "pvalue",
        "fdr",
        "significant",
        "significant_nominal",
        "significant_fdr",
        "overlap",
        "pathway_size_in_universe",
        "significant_gene_count",
        "overlap_gene_keys",
        "overlap_genes",
    ]
    missing = [column for column in required if column not in all_results.columns]
    if missing:
        raise ValueError(f"Missing columns in cached pathway enrichment table: {missing}")

    if all_results["significant"].dtype == bool:
        all_results["significant"] = all_results["significant"].astype(bool)
    else:
        all_results["significant"] = all_results["significant"].astype(str).str.lower().isin({"true", "1"})
    for bool_column in ["tested", "significant_nominal", "significant_fdr"]:
        if all_results[bool_column].dtype == bool:
            all_results[bool_column] = all_results[bool_column].astype(bool)
        else:
            all_results[bool_column] = all_results[bool_column].astype(str).str.lower().isin({"true", "1"})
    selected = pathways[["label", "source", "pathway", "order"]].merge(
        all_results,
        on=["source", "pathway"],
        how="left",
        validate="one_to_many",
    )
    missing_selected = (
        selected.loc[selected["screen"].isna(), ["label", "source", "pathway"]]
        .drop_duplicates()
        .to_dict("records")
    )
    if missing_selected:
        formatted = [f"{row['label']} ({row['source']}: {row['pathway']})" for row in missing_selected]
        raise ValueError("Selected pathways missing from cached enrichment table: " + "; ".join(formatted))

    columns = []
    for spec in SCREEN_SPECS:
        columns.extend([f"{spec['short_label']} up", f"{spec['short_label']} down"])
    heatmap = pd.DataFrame(index=pathways["label"].tolist(), columns=columns, dtype=float)
    pvalue_matrix = pd.DataFrame(index=pathways["label"].tolist(), columns=columns, dtype=float)
    support_rows = []
    all_overlap_keys = {label: set() for label in pathways["label"].tolist()}
    significant_overlap_keys = {label: set() for label in pathways["label"].tolist()}

    screen_order = {spec["short_label"]: idx for idx, spec in enumerate(SCREEN_SPECS)}
    direction_order = {"up": 0, "down": 1}
    selected["_screen_order"] = selected["screen"].map(screen_order)
    selected["_direction_order"] = selected["direction"].map(direction_order)
    selected = selected.sort_values(["order", "_screen_order", "_direction_order"], kind="mergesort")
    for row in selected.itertuples(index=False):
        pvalue = float(row.pvalue) if pd.notna(row.pvalue) else math.nan
        fdr = float(row.fdr) if pd.notna(row.fdr) else math.nan
        log2_or = float(row.log2_odds_ratio) if pd.notna(row.log2_odds_ratio) else math.nan
        overlap = int(row.overlap)
        significant = bool(
            row.tested
            and row.significant_nominal
            and overlap >= min_overlap
            and pvalue < PATHWAY_P_THRESHOLD
            and np.isfinite(log2_or)
        )
        overlap_gene_keys = "" if pd.isna(row.overlap_gene_keys) else str(row.overlap_gene_keys)
        overlap_keys = {key for key in overlap_gene_keys.split(";") if key}
        all_overlap_keys[row.label].update(overlap_keys)
        if significant:
            heatmap.loc[row.label, row.column] = log2_or
            pvalue_matrix.loc[row.label, row.column] = pvalue
            significant_overlap_keys[row.label].update(overlap_keys)
        support_rows.append(
            {
                "pathway_label": row.label,
                "source": row.source,
                "pathway": row.pathway,
                "screen": row.screen,
                "direction": row.direction,
                "column": row.column,
                "tested": bool(row.tested),
                "log2_odds_ratio": log2_or,
                "pvalue": pvalue,
                "fdr": fdr,
                "significant": significant,
                "significant_nominal": bool(row.significant_nominal),
                "significant_fdr": bool(row.significant_fdr),
                "overlap": overlap,
                "pathway_size_in_universe": int(row.pathway_size_in_universe),
                "significant_gene_count": int(row.significant_gene_count),
                "overlap_genes": "" if pd.isna(row.overlap_genes) else row.overlap_genes,
            }
        )

    unsupported = [
        label
        for label in pathways["label"].tolist()
        if not any(row["pathway_label"] == label and row["significant"] for row in support_rows)
    ]
    if unsupported:
        print("Pathways without tested nominal p < 0.05 up/down PoolQ enrichment: " + ", ".join(unsupported))
    return pd.DataFrame(support_rows), heatmap, pvalue_matrix, significant_overlap_keys, all_overlap_keys


def build_link_table(
    pathways: pd.DataFrame,
    genes: pd.DataFrame,
    significant_overlap_keys: dict[str, set[str]],
    all_overlap_keys: dict[str, set[str]],
) -> pd.DataFrame:
    selected_gene_by_key = {gene_key(gene): gene for gene in genes["gene"].tolist()}
    gene_order = dict(zip(genes["gene"], genes["order"]))
    rows = []
    for pathway in pathways.itertuples(index=False):
        linked_gene_keys = set(all_overlap_keys[pathway.label]) & set(selected_gene_by_key)
        linked_genes = sorted(
            (selected_gene_by_key[key] for key in linked_gene_keys),
            key=lambda gene: gene_order[gene],
        )
        for gene in linked_genes:
            rows.append(
                {
                    "gene": gene,
                    "pathway_label": pathway.label,
                    "pathway_source": pathway.source,
                    "pathway": pathway.pathway,
                    "gene_in_significant_pathway_overlap": gene_key(gene) in significant_overlap_keys[pathway.label],
                }
            )
    return pd.DataFrame(rows)


def wrapped_pathway_label(row: pd.Series) -> str:
    return "\n".join(wrap(str(row["label"]), width=28, break_long_words=False))


def gene_stars(neg_log10_p: float) -> str:
    if neg_log10_p >= 3:
        return "***"
    if neg_log10_p >= 2:
        return "**"
    if neg_log10_p >= GENE_NEG_LOG10_P_THRESHOLD:
        return "*"
    return ""


def pathway_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap,
    norm,
    x_tick_fontsize: float,
    y_tick_fontsize: float,
) -> object:
    plot_cmap = cmap.copy()
    plot_cmap.set_bad("white")
    image = ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
        aspect="auto",
        interpolation="nearest",
        cmap=plot_cmap,
        norm=norm,
        rasterized=True,
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=90, ha="center", fontsize=x_tick_fontsize)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=y_tick_fontsize)
    ax.tick_params(axis="both", length=0)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    x_edges = np.arange(-0.5, matrix.shape[1] + 0.5, 1)
    y_edges = np.arange(-0.5, matrix.shape[0] + 0.5, 1)
    ax.vlines(
        x_edges,
        -0.5,
        matrix.shape[0] - 0.5,
        colors="#e6e6e6",
        linewidth=0.35,
        zorder=3,
    )
    ax.hlines(
        y_edges,
        -0.5,
        matrix.shape[1] - 0.5,
        colors="#e6e6e6",
        linewidth=0.35,
        zorder=3,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def add_gene_annotations(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    neglogp_matrix: pd.DataFrame,
    vmax: float,
    fontsize: float,
) -> None:
    for y, gene in enumerate(matrix.index):
        for x, column in enumerate(matrix.columns):
            value = matrix.loc[gene, column]
            neg_log10_p = neglogp_matrix.loc[gene, column]
            if pd.isna(value) or pd.isna(neg_log10_p):
                continue
            stars = gene_stars(float(neg_log10_p))
            label = f"{float(value):.1f}\n{stars}" if stars else f"{float(value):.1f}"
            text_color = "white" if abs(float(value)) > 0.58 * vmax else "black"
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                linespacing=0.78,
                zorder=4,
            )


def add_pathway_annotations(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    pvalue_matrix: pd.DataFrame,
    vmax: float,
    fontsize: float,
) -> None:
    for y, pathway in enumerate(matrix.index):
        for x, column in enumerate(matrix.columns):
            value = matrix.loc[pathway, column]
            pvalue = pvalue_matrix.loc[pathway, column]
            if pd.isna(value) or pd.isna(pvalue):
                continue
            stars = pathway_stars(float(pvalue))
            label = f"{float(value):.1f}\n{stars}" if stars else f"{float(value):.1f}"
            text_color = "white" if float(value) > 0.62 * vmax else "black"
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                linespacing=0.78,
                zorder=4,
            )


def data_to_figure(fig: plt.Figure, ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    return tuple(fig.transFigure.inverted().transform(ax.transData.transform((x, y))))


def draw_link_curve(
    overlay: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    strong: bool,
) -> None:
    dx = x1 - x0
    path = MplPath(
        [(x0, y0), (x0 + dx * 0.45, y0), (x1 - dx * 0.45, y1), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    overlay.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=1.64 if strong else 1.0,
            alpha=0.24 if strong else 0.11,
            zorder=1,
        )
    )


def pathway_color_lookup(pathways: pd.DataFrame) -> dict[str, str]:
    return {
        row.label: PATHWAY_LINE_COLORS[idx % len(PATHWAY_LINE_COLORS)]
        for idx, row in enumerate(pathways.itertuples(index=False))
    }


def color_pathway_labels(ax: plt.Axes, path_labels: list[str], pathways: pd.DataFrame) -> None:
    colors_by_label = pathway_color_lookup(pathways)
    lookup = dict(zip(path_labels, pathways["label"]))
    for tick in ax.get_yticklabels():
        tick.set_color(colors_by_label[lookup[tick.get_text()]])


def plot_panel(
    gene_matrix: pd.DataFrame,
    gene_neglogp_matrix: pd.DataFrame,
    pathway_matrix: pd.DataFrame,
    pathway_pvalue_matrix: pd.DataFrame,
    genes: pd.DataFrame,
    pathways: pd.DataFrame,
    links: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_vmax = float(np.nanmax(np.abs(gene_matrix.to_numpy(dtype=float))))
    pathway_vmax = float(np.nanmax(pathway_matrix.to_numpy(dtype=float)))
    gene_vmax = max(1.0, math.ceil(gene_vmax * 10.0) / 10.0)
    pathway_vmax = max(1.0, math.ceil(pathway_vmax * 10.0) / 10.0)

    axis_label_fontsize = 11.6
    gene_label_fontsize = 11.2
    pathway_label_fontsize = 11.6
    gene_annotation_fontsize = 7.8
    pathway_annotation_fontsize = 8.8
    cbar_label_fontsize = 13.0
    cbar_tick_fontsize = 11.6

    fig_height = max(17.4, 0.31 * len(gene_matrix.index) + 1.2)
    fig = plt.figure(figsize=(18.2, fig_height))
    colorbar_y = 0.368
    colorbar_height = 0.375
    gene_ax = fig.add_axes([0.115, 0.18, 0.153, 0.75], zorder=2)
    gene_cbar_ax = fig.add_axes([0.052, colorbar_y, 0.014, colorbar_height], zorder=2)
    path_ax = fig.add_axes([0.47, 0.18, 0.32, 0.75], zorder=2)
    path_cbar_ax = fig.add_axes([0.935, colorbar_y, 0.014, colorbar_height], zorder=2)

    gene_cmap = sns.diverging_palette(240, 10, as_cmap=True)
    pathway_cmap = sns.color_palette("Reds", as_cmap=True)
    gene_image = draw_heatmap(
        gene_ax,
        gene_matrix,
        gene_cmap,
        TwoSlopeNorm(vcenter=0.0, vmin=-gene_vmax, vmax=gene_vmax),
        axis_label_fontsize,
        gene_label_fontsize,
    )
    pathway_image = draw_heatmap(
        path_ax,
        pathway_matrix,
        pathway_cmap,
        Normalize(vmin=0.0, vmax=pathway_vmax),
        axis_label_fontsize,
        pathway_label_fontsize,
    )

    add_gene_annotations(gene_ax, gene_matrix, gene_neglogp_matrix, gene_vmax, gene_annotation_fontsize)
    add_pathway_annotations(path_ax, pathway_matrix, pathway_pvalue_matrix, pathway_vmax, pathway_annotation_fontsize)

    path_labels = [wrapped_pathway_label(row) for _, row in pathways.iterrows()]
    path_ax.set_yticklabels(path_labels, fontsize=pathway_label_fontsize)
    path_ax.yaxis.tick_right()
    path_ax.tick_params(axis="y", pad=4)

    color_pathway_labels(path_ax, path_labels, pathways)

    gene_cbar = fig.colorbar(gene_image, cax=gene_cbar_ax, orientation="vertical")
    gene_cbar.set_label("Average LFC", fontsize=cbar_label_fontsize, labelpad=8)
    gene_cbar.ax.yaxis.set_ticks_position("left")
    gene_cbar.ax.yaxis.set_label_position("left")
    gene_cbar.ax.tick_params(labelsize=cbar_tick_fontsize, length=3, pad=2)
    path_cbar = fig.colorbar(pathway_image, cax=path_cbar_ax, orientation="vertical")
    path_cbar.set_label("log2(odds_ratio)", fontsize=cbar_label_fontsize, labelpad=8)
    path_cbar.ax.yaxis.set_ticks_position("right")
    path_cbar.ax.yaxis.set_label_position("right")
    path_cbar.ax.tick_params(labelsize=cbar_tick_fontsize, length=3, pad=2)

    fig.canvas.draw()
    overlay = fig.add_axes([0, 0, 1, 1], zorder=1)
    overlay.set_axis_off()
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)

    gene_y = {gene: idx for idx, gene in enumerate(gene_matrix.index)}
    path_y = {label: idx for idx, label in enumerate(pathway_matrix.index)}
    pathway_colors = pathway_color_lookup(pathways)
    for row in links.itertuples(index=False):
        x0, y0 = data_to_figure(fig, gene_ax, gene_matrix.shape[1] - 0.47, gene_y[row.gene])
        x1, y1 = data_to_figure(fig, path_ax, -0.53, path_y[row.pathway_label])
        draw_link_curve(
            overlay,
            x0 + 0.006,
            y0,
            x1 - 0.006,
            y1,
            pathway_colors[row.pathway_label],
            bool(row.gene_in_significant_pathway_overlap),
        )

    png_path = output_dir / f"{PLOT_STEM}.png"
    svg_path = output_dir / f"{PLOT_STEM}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    png_size = png_path.stat().st_size / 1024 / 1024
    svg_size = svg_path.stat().st_size / 1024
    if png_size > 5:
        print(f"ERROR: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"ERROR: SVG {svg_size:.1f}KB exceeds 500KB limit! Add rasterized=True")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def write_outputs(
    output_dir: Path,
    genes: pd.DataFrame,
    pathways: pd.DataFrame,
    gene_summary: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    gene_neglogp_matrix: pd.DataFrame,
    pathway_support: pd.DataFrame,
    pathway_matrix: pd.DataFrame,
    pathway_pvalue_matrix: pd.DataFrame,
    links: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    genes.to_csv(output_dir / f"{PLOT_STEM}_selected_genes.tsv", sep="\t", index=False)
    pathways.to_csv(output_dir / f"{PLOT_STEM}_selected_pathways.tsv", sep="\t", index=False)
    gene_summary.to_csv(output_dir / f"{PLOT_STEM}_gene_summary.tsv", sep="\t", index=False)
    gene_matrix.to_csv(output_dir / f"{PLOT_STEM}_gene_heatmap.tsv", sep="\t")
    gene_neglogp_matrix.to_csv(output_dir / f"{PLOT_STEM}_gene_neglog10_p.tsv", sep="\t")
    pathway_support.to_csv(output_dir / f"{PLOT_STEM}_pathway_support.tsv", sep="\t", index=False)
    pathway_matrix.to_csv(output_dir / f"{PLOT_STEM}_pathway_heatmap.tsv", sep="\t")
    pathway_pvalue_matrix.to_csv(output_dir / f"{PLOT_STEM}_pathway_pvalues.tsv", sep="\t")
    links.to_csv(output_dir / f"{PLOT_STEM}_links.tsv", sep="\t", index=False)
    write_pathway_label_key(output_dir, pathways)


def write_pathway_label_key(output_dir: Path, pathways: pd.DataFrame) -> None:
    lines = [
        "Figure S2a-b pathway label key",
        "",
        "Figure label (original database pathway name)",
        "",
    ]
    for group, group_pathways in pathways.groupby("group", sort=False):
        lines.append(group)
        for row in group_pathways.itertuples(index=False):
            lines.append(f"- {row.label} ({row.pathway})")
        lines.append("")

    (output_dir / f"{PLOT_STEM}_pathway_label_key.txt").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    global PLOT_STEM
    PLOT_STEM = args.plot_stem
    output_dir = Path(args.output_dir)

    screens = load_screens(args)
    genes = filter_selected_genes_by_significance(selected_genes(), screens)
    pathways = selected_pathways()

    gene_summary, gene_matrix, gene_neglogp_matrix = collect_gene_support(screens, genes)
    genes, gene_summary, gene_matrix, gene_neglogp_matrix = order_gene_outputs(
        genes,
        gene_summary,
        gene_matrix,
        gene_neglogp_matrix,
    )
    (
        pathway_support,
        pathway_matrix,
        pathway_pvalue_matrix,
        significant_overlap_keys,
        all_overlap_keys,
    ) = collect_pathway_support(
        Path(args.all_pathway_enrichment),
        pathways,
        args.min_overlap,
    )
    links = build_link_table(pathways, genes, significant_overlap_keys, all_overlap_keys)

    write_outputs(
        output_dir,
        genes,
        pathways,
        gene_summary,
        gene_matrix,
        gene_neglogp_matrix,
        pathway_support,
        pathway_matrix,
        pathway_pvalue_matrix,
        links,
    )
    plot_panel(gene_matrix, gene_neglogp_matrix, pathway_matrix, pathway_pvalue_matrix, genes, pathways, links, output_dir)

    print(f"Selected genes: {len(genes)}")
    print(f"Selected pathways: {len(pathways)}")
    print(f"Programmatic links: {len(links)}")


if __name__ == "__main__":
    main()
