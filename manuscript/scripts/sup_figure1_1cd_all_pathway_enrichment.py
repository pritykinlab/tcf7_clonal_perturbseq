#!/usr/bin/env python
"""Cache all pathway enrichment tests for Figure S2a-b."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_PREFIXES = ("NO_SITE", "INACTIVE", "POTENTIALLY_ACTIVE")
GENE_P_THRESHOLD = 0.05
GENE_NEG_LOG10_P_THRESHOLD = -math.log10(GENE_P_THRESHOLD)
PATHWAY_P_THRESHOLD = 0.05
REQUIRED_GUIDES_PER_TARGET_GENE = 4

CATALOGS = {
    "GO": "GO_Biological_Process_2025",
    "GOCC": "GO_Cellular_Component_2025",
    "GOMF": "GO_Molecular_Function_2025",
    "KEGG": "KEGG_2019_Mouse",
    "REACTOME": "Reactome_2022",
    "WikiPathways": "WikiPathways_2024_Mouse",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for spec in SCREEN_SPECS:
        parser.add_argument(f"--{spec['arg'].replace('_', '-')}", required=True)
    parser.add_argument("--my-utils-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help=(
            "Minimum selected-gene/pathway overlap required to report a finite "
            "Fisher log2 odds ratio. Rows below this threshold keep overlap "
            "counts but do not report the half-count fallback odds ratio."
        ),
    )
    parser.add_argument(
        "--custom-catalog",
        action="append",
        nargs=3,
        metavar=("SOURCE", "LIBRARY", "PATH"),
        default=[],
        help="Additional GMT catalog as source, library name, and path.",
    )
    return parser.parse_args()


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


def load_pathway_tools(
    my_utils_dir: Path,
    custom_catalogs: list[list[str]],
) -> tuple[object, object, object, list[object]]:
    sys.path.insert(0, str(my_utils_dir.resolve().parent))
    from my_utils.gene_list_interpretation import (  # pylint: disable=import-error,import-outside-toplevel
        available_cpus,
        gene_key,
        load_pathway_catalog,
        load_pathway_catalogs,
        test_pathways,
    )

    catalog_dir = my_utils_dir / "pathways" / "enrichr"
    catalog_paths = {
        source: (library, catalog_dir / f"{library}.gmt")
        for source, library in CATALOGS.items()
    }
    terms = load_pathway_catalogs(catalog_paths)
    for source, library, path in custom_catalogs:
        terms.extend(load_pathway_catalog(Path(path), source=source, library=library))
    return available_cpus, gene_key, test_pathways, terms


def prepare_pathway_inputs(df: pd.DataFrame, gene_key_fn: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = df.sort_values(
        ["Gene LFC", "Gene p-value", "Gene"],
        ascending=[False, True, True],
        kind="mergesort",
    ).copy()
    ranked["gene_key"] = ranked["Gene"].map(gene_key_fn)
    ranked = ranked[ranked["gene_key"] != ""].drop_duplicates("gene_key", keep="first").reset_index(drop=True)
    ranked["rank_position"] = np.arange(1, len(ranked) + 1, dtype=int)
    up = ranked[ranked["significant"] & (ranked["Gene LFC"].astype(float) > 0)].copy()
    down = ranked[ranked["significant"] & (ranked["Gene LFC"].astype(float) < 0)].copy()
    columns = ["Gene", "gene_key", "rank_position"]
    return ranked[columns], up[columns], down[columns]


def _cell_value(row: object, attr: str, default: object = "") -> object:
    value = getattr(row, attr, default)
    if pd.isna(value):
        return default
    return value


def pathway_results_to_long(
    pathway_results: pd.DataFrame,
    screen: str,
    condition: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for direction, prefix, display_direction in [("up", "up", "up"), ("down", "down", "down")]:
        column = f"{screen} {display_direction}"
        for row in pathway_results.itertuples(index=False):
            tested = bool(_cell_value(row, f"{prefix}_tested", False))
            pvalue = _cell_value(row, f"{prefix}_pvalue", math.nan)
            fdr = _cell_value(row, f"{prefix}_fdr", math.nan)
            significant_nominal = bool(_cell_value(row, f"{prefix}_significant_nominal", False))
            significant_fdr = bool(_cell_value(row, f"{prefix}_significant_fdr", False))
            rows.append(
                {
                    "term_id": row.term_id,
                    "source": row.source,
                    "library": row.library,
                    "pathway": row.pathway,
                    "screen": screen,
                    "condition": condition,
                    "direction": direction,
                    "column": column,
                    "tested": tested,
                    "log2_odds_ratio": _cell_value(row, f"{prefix}_log2_odds_ratio", math.nan),
                    "pvalue": pvalue,
                    "fdr": fdr,
                    "significant": significant_nominal,
                    "significant_nominal": significant_nominal,
                    "significant_fdr": significant_fdr,
                    "overlap": int(_cell_value(row, f"{prefix}_overlap", 0)),
                    "pathway_size_in_universe": int(_cell_value(row, f"{prefix}_pathway_size_in_universe", 0)),
                    "significant_gene_count": int(_cell_value(row, f"{prefix}_selected_gene_count", 0)),
                    "universe_gene_count": int(_cell_value(row, f"{prefix}_universe_gene_count", 0)),
                    "overlap_gene_keys": _cell_value(row, f"{prefix}_gene_keys", ""),
                    "overlap_genes": _cell_value(row, f"{prefix}_genes", ""),
                    "ranked_pvalue": row.ranked_pvalue,
                    "ranked_fdr": row.ranked_fdr,
                    "ranked_direction": row.ranked_direction,
                    "ranked_significant_nominal": row.ranked_significant_nominal,
                    "ranked_significant_fdr": row.ranked_significant_fdr,
                    "ranked_significant": row.ranked_significant,
                    "ranked_overlap": int(row.ranked_overlap),
                    "ranked_mean_rank": row.ranked_mean_rank,
                    "ranked_genes": row.ranked_genes,
                }
            )
    return rows


def collect_all_pathway_enrichment(
    screens: dict[str, pd.DataFrame],
    terms: list[object],
    test_pathways_fn: object,
    gene_key_fn: object,
    threads: int,
    source_order_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in SCREEN_SPECS:
        screen = spec["short_label"]
        ranked, up, down = prepare_pathway_inputs(screens[screen], gene_key_fn)
        pathway_results = test_pathways_fn(
            ranked_genes=ranked,
            significant_up_genes=up,
            significant_down_genes=down,
            pathway_catalogs=terms,
            n_jobs=threads,
        )
        rows.extend(pathway_results_to_long(pathway_results, screen, spec["condition"]))

    results = pd.DataFrame(rows)
    screen_order = {spec["short_label"]: idx for idx, spec in enumerate(SCREEN_SPECS)}
    source_order = {source: idx for idx, source in enumerate(source_order_names)}
    direction_order = {"up": 0, "down": 1}
    results["_screen_order"] = results["screen"].map(screen_order)
    results["_direction_order"] = results["direction"].map(direction_order)
    results["_source_order"] = results["source"].map(source_order).fillna(len(source_order))
    results = results.sort_values(
        ["_screen_order", "_direction_order", "_source_order", "pathway"],
        kind="mergesort",
    )
    return results.drop(columns=["_screen_order", "_direction_order", "_source_order"]).reset_index(drop=True)


def mask_half_count_odds_ratios(results: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Remove log2 odds ratios that would require the shared 0.5-count fallback."""
    masked = results.copy()
    overlap = pd.to_numeric(masked["overlap"], errors="coerce")
    selected = pd.to_numeric(masked["significant_gene_count"], errors="coerce")
    pathway = pd.to_numeric(masked["pathway_size_in_universe"], errors="coerce")
    universe = pd.to_numeric(masked["universe_gene_count"], errors="coerce")
    selected_not_pathway = selected - overlap
    pathway_not_selected = pathway - overlap
    neither = universe - overlap - selected_not_pathway - pathway_not_selected

    uses_half_count = (
        overlap.lt(min_overlap)
        | overlap.le(0)
        | selected_not_pathway.le(0)
        | pathway_not_selected.le(0)
        | neither.le(0)
    ).fillna(True)

    masked.loc[uses_half_count, "log2_odds_ratio"] = pd.NA
    for column in ["significant", "significant_nominal", "significant_fdr"]:
        masked.loc[uses_half_count, column] = False
    return masked


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    custom_catalogs = args.custom_catalog or []
    available_cpus_fn, gene_key_fn, test_pathways_fn, terms = load_pathway_tools(Path(args.my_utils_dir), custom_catalogs)
    threads = max(1, int(args.threads or available_cpus_fn()))
    source_order_names = list(CATALOGS) + [catalog[0] for catalog in custom_catalogs]

    screens = load_screens(args)
    results = collect_all_pathway_enrichment(
        screens,
        terms,
        test_pathways_fn,
        gene_key_fn,
        threads,
        source_order_names,
    )
    results = mask_half_count_odds_ratios(results, args.min_overlap)

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, sep="\t", index=False)

    print(f"Pathway enrichment threads: {min(threads, len(terms))}")
    print(f"Pathway terms tested: {len(terms)}")
    print(f"Screen-direction tests: {len(SCREEN_SPECS) * 2}")
    print(f"Rows written: {len(results)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
