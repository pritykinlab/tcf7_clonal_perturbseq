#!/usr/bin/env python
"""
Generate gene-level signature heatmaps for Figure S3c-e.

Rows are manually curated representative genes grouped by the Figure 2e
signature/program order. Candidate genes are ranked within each program by:

    max_c(mu_g,c) - mean(other cell types)

where mu_g,c is the mean NT-referenced z-scored expression of gene g in cell
type c. Five genes are then manually selected from the top 20 ranked genes for
each program to emphasize interpretable markers while preserving the Figure 2e
program structure.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from scipy.sparse import issparse

sys.path.append("../scripts/utils")

from GO import GOParser  # noqa: E402
from figure_format import cell_type_orders_gene10, get_adata  # noqa: E402
import misc  # noqa: E402
from signature_scoring_updated import (  # noqa: E402
    get_burn2025_signatures,
    get_miller_signatures,
    get_park2025_signatures,
    get_trm_signatures,
    get_yuri_signatures,
)


# Required font settings for all manuscript figures
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 6.5


OUTPUT_DIR = "../manuscript/plots/Sup_Figure_3"
OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "c_candidates.tsv")
OUTPUT_SELECTED = os.path.join(OUTPUT_DIR, "c_selected_genes.tsv")
OUTPUT_LEGEND_PNG = os.path.join(OUTPUT_DIR, "signature_heatmap_legend.png")
OUTPUT_LEGEND_SVG = os.path.join(OUTPUT_DIR, "signature_heatmap_legend.svg")
CURATION_CONFIG = (
    "../manuscript/config/"
    "sup_figure2_1c_manual_curation.tsv"
)
SECTION_SPECS = OrderedDict(
    [
        (
            "c",
            {
                "title": "GO terms",
                "programs": [
                    "GOBP_THYMOCYTE_MIGRATION",
                    "GOBP_DENDRITIC_CELL_CHEMOTAXIS",
                    "GOBP_POSITIVE_REGULATION_OF_RESPONSE_TO_TUMOR_CELL",
                    "GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_INTEGRIN",
                    "GOCC_MHC_CLASS_I_PROTEIN_COMPLEX",
                    "GOCC_CYTOLYTIC_GRANULE",
                    "GOMF_CCR1_CHEMOKINE_RECEPTOR_BINDING",
                    "GOBP_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_CADHERIN",
                    "GOMF_CELL_ADHESION_MEDIATOR_ACTIVITY",
                ],
            },
        ),
        (
            "d",
            {
                "title": "Progenitor/exhaustion signatures",
                "programs": [
                    "Progenitor Pritykin et al., 2021",
                    "Dysfunction Pritykin et al., 2021",
                    "Progenitor Miller et al., 2019",
                    "Terminal Miller et al., 2019",
                ],
            },
        ),
        (
            "e",
            {
                "title": "TRM signature",
                "programs": [
                    "TRM Mackay et al., 2013",
                    "TRM Mackay et al., 2016",
                    "TRM Milner et al., 2017",
                    "TRM Crowl et al., 2022",
                    "TRM Burn et al., 2026",
                    "TEX Burn et al., 2026",
                    "TRM/TR-TEX Park et al., 2026",
                    "TRM Park et al., 2026",
                ],
            },
        ),
    ]
)

GO_DISPLAY_LABELS = {
    "GOBP_THYMOCYTE_MIGRATION": "Thymocyte migration",
    "GOBP_DENDRITIC_CELL_CHEMOTAXIS": "Dendritic cell chemotaxis",
    "GOBP_POSITIVE_REGULATION_OF_RESPONSE_TO_TUMOR_CELL": "Pos. reg. response to tumor cell",
    "GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_INTEGRIN": "Pos. reg. cell-cell adhesion by integrin",
    "GOCC_MHC_CLASS_I_PROTEIN_COMPLEX": "MHC-I complex",
    "GOCC_CYTOLYTIC_GRANULE": "Cytolytic granule",
    "GOMF_CCR1_CHEMOKINE_RECEPTOR_BINDING": "CCR1 chemokine binding",
    "GOBP_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_CADHERIN": "Reg. cell-cell adhesion by cadherin",
    "GOMF_CELL_ADHESION_MEDIATOR_ACTIVITY": "Cell adhesion mediator activity",
}


def _validate_file_sizes(png_path: str, svg_path: str) -> None:
    png_size = os.path.getsize(png_path) / 1024 / 1024
    svg_size = os.path.getsize(svg_path) / 1024
    if png_size > 5:
        print(f"WARNING: PNG {png_size:.1f}MB exceeds 5MB limit!")
    if svg_size > 500:
        print(f"WARNING: SVG {svg_size:.1f}KB exceeds 500KB limit!")
    else:
        print(f"File sizes OK: PNG {png_size:.2f}MB, SVG {svg_size:.1f}KB")


def _panel_paths(section_key: str) -> tuple[str, str]:
    return (
        os.path.join(OUTPUT_DIR, f"{section_key}.png"),
        os.path.join(OUTPUT_DIR, f"{section_key}.svg"),
    )


def _build_programs() -> OrderedDict[str, list[str]]:
    go_parser = GOParser()
    go_terms = [
        "GOBP_THYMOCYTE_MIGRATION",
        "GOBP_DENDRITIC_CELL_CHEMOTAXIS",
        "GOBP_POSITIVE_REGULATION_OF_RESPONSE_TO_TUMOR_CELL",
        "GOBP_POSITIVE_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_INTEGRIN",
        "GOCC_MHC_CLASS_I_PROTEIN_COMPLEX",
        "GOCC_CYTOLYTIC_GRANULE",
        "GOMF_CCR1_CHEMOKINE_RECEPTOR_BINDING",
        "GOBP_REGULATION_OF_CELL_CELL_ADHESION_MEDIATED_BY_CADHERIN",
        "GOMF_CELL_ADHESION_MEDIATOR_ACTIVITY",
    ]

    go_dict = {term: go_parser.get_genes_for_term(term) for term in go_terms}
    yuri_sigs = get_yuri_signatures(
        {"func_dysfunc": "func vs dysfunc", "dysfunc_prog": "dysfunc vs prog"}
    )
    miller_sigs = get_miller_signatures()
    trm_sigs = get_trm_signatures()
    burn2025_sigs = get_burn2025_signatures()
    park2025_sigs = get_park2025_signatures()
    s_genes, g2m_genes = misc.get_cell_cycle_genes()

    programs: OrderedDict[str, list[str]] = OrderedDict()
    for term, genes in go_dict.items():
        programs[term] = genes
    programs["Progenitor Pritykin et al., 2021"] = yuri_sigs["Up prog vs dysfunc"]
    programs["Dysfunction Pritykin et al., 2021"] = yuri_sigs["Up dysfunc vs prog"]
    programs["Progenitor Miller et al., 2019"] = miller_sigs["Up Tumor Prog vs Tumor Terminal"]
    programs["Terminal Miller et al., 2019"] = miller_sigs["Up Tumor Terminal vs Tumor Prog"]
    for source in (trm_sigs, burn2025_sigs, park2025_sigs):
        for term, genes in source.items():
            programs[term] = genes
    programs["S phase"] = s_genes
    programs["G2/M phase"] = g2m_genes
    return programs


def _load_manual_curation(config_path: str) -> pd.DataFrame:
    curation = pd.read_csv(config_path, sep="\t")
    required_cols = {
        "program",
        "selection_order",
        "gene",
        "gene_function",
        "selection_reason",
    }
    missing = required_cols - set(curation.columns)
    if missing:
        raise ValueError(f"Missing columns in manual curation file: {sorted(missing)}")

    curation["selection_order"] = curation["selection_order"].astype(int)
    if curation.duplicated(["program", "selection_order"]).any():
        raise ValueError("Manual curation file has duplicate program/selection_order entries.")
    if curation.duplicated(["program", "gene"]).any():
        raise ValueError("Manual curation file has duplicate program/gene entries.")

    counts = curation.groupby("program", sort=False)["gene"].count()
    bad_counts = counts[counts != 5]
    if not bad_counts.empty:
        raise ValueError(
            "Each program must have exactly 5 curated genes. "
            f"Bad counts: {bad_counts.to_dict()}"
        )

    curation["display_program"] = curation["program"].map(lambda x: GO_DISPLAY_LABELS.get(x, x))
    program_order = {program: idx for idx, program in enumerate(curation["program"].drop_duplicates())}
    curation["program_order"] = curation["program"].map(program_order)
    return curation.sort_values(["program_order", "selection_order"]).reset_index(drop=True)


def _compute_mean_by_cell_type(adata, cell_types: list[str]) -> pd.DataFrame:
    layer = adata.layers["norm_log_zscore_control"]
    if issparse(layer):
        layer = layer.toarray()
    layer = np.asarray(layer)

    mean_by_ct = {}
    cell_type_values = adata.obs["cell_type"].astype(str).to_numpy()
    for cell_type in cell_types:
        mask = cell_type_values == cell_type
        if not np.any(mask):
            continue
        mean_by_ct[cell_type] = layer[mask].mean(axis=0)
    return pd.DataFrame(mean_by_ct, index=adata.var_names)


def _rank_candidate_genes(
    programs: OrderedDict[str, list[str]], mean_by_ct: pd.DataFrame, cell_types: list[str]
) -> pd.DataFrame:
    rows = []
    program_order = {program: idx for idx, program in enumerate(programs)}
    for program, genes in programs.items():
        valid = [gene for gene in genes if gene in mean_by_ct.index]
        if not valid:
            continue
        sub = mean_by_ct.loc[valid, cell_types]
        values = sub.to_numpy()
        max_idx = values.argmax(axis=1)
        max_vals = values[np.arange(len(valid)), max_idx]
        other_means = (values.sum(axis=1) - max_vals) / (values.shape[1] - 1)
        ranking_score = max_vals - other_means
        for gene, score, idx in zip(valid, ranking_score, max_idx):
            row = {
                "program": program,
                "display_program": GO_DISPLAY_LABELS.get(program, program),
                "gene": gene,
                "ranking_score": float(score),
                "max_cell_type": cell_types[idx],
            }
            for cell_type in cell_types:
                row[cell_type] = float(sub.loc[gene, cell_type])
            rows.append(row)

    ranked = pd.DataFrame(rows)
    ranked["program_order"] = ranked["program"].map(program_order)
    ranked = ranked.sort_values(
        ["program_order", "ranking_score", "gene"], ascending=[True, False, True]
    )
    ranked["candidate_rank"] = (
        ranked.groupby("program")["ranking_score"].rank(method="first", ascending=False).astype(int)
    )
    return ranked


def _select_curated_genes(ranked: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    curation = _load_manual_curation(CURATION_CONFIG)
    selected_rows = []
    for program, program_curation in curation.groupby("program", sort=False):
        program_candidates = ranked.loc[ranked["program"] == program].copy()
        if program_candidates.empty:
            raise ValueError(f"No ranked candidates found for program: {program}")
        top20 = set(program_candidates.nsmallest(20, "candidate_rank")["gene"])
        for _, curation_row in program_curation.iterrows():
            gene = curation_row["gene"]
            if gene not in top20:
                raise ValueError(
                    f"Curated gene {gene} for {program} is not in the top 20 candidates."
                )
            row = program_candidates.loc[program_candidates["gene"] == gene].iloc[0].to_dict()
            row["selection_order"] = int(curation_row["selection_order"])
            row["gene_function"] = curation_row["gene_function"]
            row["selection_reason"] = curation_row["selection_reason"]
            row["program_order"] = int(curation_row["program_order"])
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values(["program_order", "selection_order"]).reset_index(drop=True)
    selected["display_program"] = selected["program"].map(lambda x: GO_DISPLAY_LABELS.get(x, x))
    ordered_cols = [
        "program",
        "display_program",
        "program_order",
        "selection_order",
        "gene",
        "candidate_rank",
        "ranking_score",
        "max_cell_type",
        "gene_function",
        "selection_reason",
    ] + cell_types
    return selected.loc[:, ordered_cols]


def _build_plot_matrix(selected: pd.DataFrame, cell_types: list[str]):
    plot_rows = []
    yticklabels: list[str] = []
    program_spans = []
    row_idx = 0

    programs_in_order = selected.loc[:, ["program_order", "program"]].drop_duplicates()
    for program_index, (_, program) in enumerate(programs_in_order.itertuples(index=False)):
        program_genes = (
            selected.loc[selected["program"] == program]
            .sort_values("selection_order")["gene"]
            .tolist()
        )
        program_df = selected.loc[selected["program"] == program].set_index("gene")
        start = row_idx
        for gene in program_genes:
            plot_rows.append(program_df.loc[gene, cell_types].to_numpy(dtype=float))
            yticklabels.append(gene)
            row_idx += 1
        end = row_idx - 1
        program_spans.append(
            {
                "program": program,
                "display_label": GO_DISPLAY_LABELS.get(program, program),
                "start": start,
                "end": end,
            }
        )
        if program_index < len(programs_in_order) - 1:
            plot_rows.append(np.full(len(cell_types), np.nan))
            yticklabels.append("")
            row_idx += 1

    return np.vstack(plot_rows), yticklabels, program_spans


def _draw_program_bracket(ax, x_inner: float, x_outer: float, start: int, end: int, label: str) -> None:
    y0 = start - 0.5
    y1 = end + 0.5
    y_mid = (start + end) / 2
    ax.plot([x_inner, x_outer], [y0, y0], color="black", lw=0.6, clip_on=False)
    ax.plot([x_outer, x_outer], [y0, y1], color="black", lw=0.6, clip_on=False)
    ax.plot([x_inner, x_outer], [y1, y1], color="black", lw=0.6, clip_on=False)
    ax.text(
        x_outer + 0.12,
        y_mid,
        label,
        va="center",
        ha="left",
        fontsize=7.5,
    )


def _render_section(
    selected: pd.DataFrame,
    cell_types: list[str],
    section_key: str,
    section_title: str,
    section_programs: list[str],
) -> None:
    section_selected = selected.loc[selected["program"].isin(section_programs)].copy()
    if section_selected.empty:
        raise ValueError(f"No selected genes found for section {section_key}.")

    plot_matrix, yticklabels, program_spans = _build_plot_matrix(section_selected, cell_types)

    fig_height = max(5.0, 0.16 * plot_matrix.shape[0] + 1.7)
    fig = plt.figure(figsize=(12.0, fig_height))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.18, 0.08, 0.50, 0.84])
    ax.set_facecolor("none")

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    masked = np.ma.masked_invalid(plot_matrix)
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=-2,
        vmax=2,
        aspect="auto",
        interpolation="none",
    )
    im.set_rasterized(True)

    ax.set_xticks(np.arange(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=90, ha="center")
    ax.tick_params(
        axis="x",
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        length=0,
        pad=2,
    )

    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(section_title, pad=18, fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ncols = len(cell_types)
    x_inner = ncols - 0.5 + 0.05
    x_outer = ncols - 0.5 + 0.28
    for span in program_spans:
        _draw_program_bracket(ax, x_inner, x_outer, span["start"], span["end"], span["display_label"])

    ax.set_xlim(-0.5, ncols + 4.3)

    output_png, output_svg = _panel_paths(section_key)
    fig.savefig(output_png, dpi=250, bbox_inches="tight", transparent=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    _validate_file_sizes(output_png, output_svg)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_svg}")


def _render_heatmap_legend() -> None:
    fig = plt.figure(figsize=(1.4, 2.6))
    cax = fig.add_axes([0.38, 0.12, 0.22, 0.76])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=2), cmap=plt.get_cmap("RdBu_r"))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.set_title("Mean\nNT-ref.\nZ-score", fontsize=9, pad=6)

    fig.savefig(OUTPUT_LEGEND_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_LEGEND_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)

    _validate_file_sizes(OUTPUT_LEGEND_PNG, OUTPUT_LEGEND_SVG)
    print(f"Wrote {OUTPUT_LEGEND_PNG}")
    print(f"Wrote {OUTPUT_LEGEND_SVG}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading gene_10 dataset with NT-referenced z-scored expression...")
    adata = get_adata(
        "gene_10",
        use_cells_only_one_construct=False,
        remove_NG=False,
        get_raw_scaled=False,
        get_norm_log_zscore_control=True,
    )

    cell_types = [ct for ct in cell_type_orders_gene10 if ct in adata.obs["cell_type"].unique()]
    mean_by_ct = _compute_mean_by_cell_type(adata, cell_types)

    print("Building Figure 2e program definitions and ranking candidate genes...")
    programs = _build_programs()
    ranked = _rank_candidate_genes(programs, mean_by_ct, cell_types)
    selected = _select_curated_genes(ranked, cell_types)
    selected_programs = selected["program"].drop_duplicates().tolist()
    ranked.loc[
        ranked["program"].isin(selected_programs) & (ranked["candidate_rank"] <= 20)
    ].to_csv(OUTPUT_CANDIDATES, sep="\t", index=False)

    print("Applying manual curation from the top 20 genes per program...")
    selected.to_csv(OUTPUT_SELECTED, sep="\t", index=False)

    for section_key, section_spec in SECTION_SPECS.items():
        _render_section(
            selected=selected,
            cell_types=cell_types,
            section_key=section_key,
            section_title=section_spec["title"],
            section_programs=section_spec["programs"],
        )
    _render_heatmap_legend()
    print(f"Wrote {OUTPUT_CANDIDATES}")
    print(f"Wrote {OUTPUT_SELECTED}")


if __name__ == "__main__":
    main()
