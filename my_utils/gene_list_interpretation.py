from __future__ import annotations

import html
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm


SOURCE_ORDER = ("GO", "GOCC", "GOMF", "KEGG", "REACTOME", "WikiPathways")
DEFAULT_ENRICHR_LIBRARIES = {
    "GO": "GO_Biological_Process_2025",
    "GOCC": "GO_Cellular_Component_2025",
    "GOMF": "GO_Molecular_Function_2025",
    "KEGG": "KEGG_2019_Mouse",
    "REACTOME": "Reactome_2022",
    "WikiPathways": "WikiPathways_2024_Mouse",
}
PATHWAY_FDR_THRESHOLD = 0.05
MIN_DIRECTIONAL_PATHWAY_GENES = 0
COMMON_NOISE_PATHWAY_KEYWORDS = [
    "ABC TRANSPORTER DISORDERS",
    "CANCER",
    "CARCINOMA",
    "DISEASE",
    "EPSTEIN-BARR",
    "GLIOMA",
    "HEPATITIS",
    "HIV",
    "HOST",
    "HUMAN PAPILLOMAVIRUS",
    "INFECTION",
    "INFECTIOUS",
    "INFLUENZA",
    "LEUKEMIA",
    "MELANOMA",
    "POTENTIAL THERAPEUTICS",
    "SARS",
    "THERMOGENESIS",
    "VIRAL",
]
GENERIC_DISPLAY_PATHWAY_KEYWORDS = [
    "CELLULAR RESPONSES TO STIMULI",
    "CELLULAR RESPONSES TO STRESS",
    "GENE EXPRESSION (TRANSCRIPTION)",
    "GENERIC TRANSCRIPTION PATHWAY",
    "IMMUNE SYSTEM",
    "METABOLISM OF PROTEINS",
    "METABOLISM OF RNA",
    "METABOLISM R-HSA",
    "INTRACELLULAR SIGNALING BY SECOND MESSENGERS",
    "PREINITIATION COMPLEX",
    "RNA POLYMERASE III TRANSCRIPTION",
    "RNA POLYMERASE II TRANSCRIPTION",
    "SIGNAL TRANSDUCTION",
    "TRANSCRIPTION INITIATION",
    "TRANSCRIPTION PREINITIATION",
]
GENE_DESCRIPTION_GENERIC_PATHWAY_KEYWORDS = [
    "CYTOKINE SIGNALING IN IMMUNE SYSTEM",
    "INNATE IMMUNE SYSTEM",
    "SIGNALING BY INTERLEUKINS",
]
INTERPRETABLE_PROGRAMS = [
    {
        "id": "t_cell_cytokine",
        "name": "TCR, cytokine, and JAK/STAT signaling",
        "short_label": "TCR/cytokine signaling",
        "keywords": [
            "ANTIGEN RECEPTOR",
            "CD3",
            "IFNG",
            "INTERFERON",
            "INTERLEUKIN",
            "IL 2",
            "IL-2",
            "IL 12",
            "IL-12",
            "IL 15",
            "IL-15",
            "JAK",
            "JAK-STAT",
            "JAK STAT",
            "LCK",
            "LYMPHOCYTE ACTIVATION",
            "T CELL ACTIVATION",
            "T CELL DIFFERENTIATION",
            "T CELL RECEPTOR",
            "TCR",
            "CYTOKINE SIGNALING",
            "CYTOKINE-MEDIATED",
            "CYTOKINE PRODUCTION",
            "CYTOKINE RECEPTOR",
            "RESPONSE TO CYTOKINE",
            "STAT1",
            "STAT2",
            "STAT3",
            "STAT4",
            "STAT5",
            "STAT6",
            "TH1",
            "TH2",
            "TH17",
            "ZAP70",
        ],
        "literature": (
            "Strong antigen, IL-2/IL-12/IFN, and JAK/STAT cues are established drivers of activated effector CD8 T cell "
            "differentiation. These signals can induce Prdm1/Blimp1-like effector programs and can oppose a quieter "
            "TCF1/TCF7 progenitor-like state when stimulation is strong or chronic."
        ),
    },
    {
        "id": "fate_tf",
        "name": "TCF/FOXO/Blimp1/RUNX cell-fate control",
        "short_label": "cell-fate transcriptional regulators",
        "keywords": [
            "BCL6",
            "BLIMP",
            "FOXO",
            "LEF",
            "MYB",
            "NOTCH",
            "PRDM1",
            "PTEN",
            "REGULATION OF T CELL DIFFERENTIATION",
            "RUNX",
            "TCF",
            "T CELL DIFFERENTIATION",
            "WNT",
        ],
        "literature": (
            "TCF1/TCF7, FOXO1, BCL6, and related transcriptional circuits maintain memory-like or progenitor-like CD8 "
            "states, whereas Prdm1/Blimp1 and RUNX-linked effector circuits promote terminal effector differentiation. "
            "This axis is therefore directly interpretable in Tcf7, Prdm1, Eomes, and Tox reporter-bin screens."
        ),
    },
    {
        "id": "pi3k_mtor_metabolism",
        "name": "PI3K/AKT/mTOR, nutrient sensing, and glycolysis",
        "short_label": "PI3K/AKT/mTOR-metabolic signaling",
        "keywords": [
            "AKT",
            "AMPK",
            "CELLULAR RESPONSE TO STARVATION",
            "ELECTRON TRANSPORT",
            "FATTY ACID",
            "GLYCOLYSIS",
            "GLUCONEOGENESIS",
            "HIF",
            "HYPOXIA",
            "INSULIN",
            "LKB1",
            "MTOR",
            "MTORC1",
            "OXIDATIVE PHOSPHORYLATION",
            "PENTOSE PHOSPHATE",
            "PI3K",
            "PIP3",
            "PTEN",
            "STARVATION",
            "TCA CYCLE",
        ],
        "literature": (
            "PI3K/AKT/mTOR and nutrient-sensing pathways couple TCR and cytokine signals to anabolic growth. In CD8 T cells, "
            "strong anabolic signaling supports effector differentiation, while lower AKT/mTOR tone and FOXO activity are "
            "often associated with memory-like or progenitor-like programs."
        ),
    },
    {
        "id": "cell_cycle_dna",
        "name": "Cell cycle, DNA replication, and DNA repair",
        "short_label": "cell-cycle and DNA-replication machinery",
        "keywords": [
            "CELL CYCLE",
            "CHECKPOINT",
            "DNA REPAIR",
            "DNA REPLICATION",
            "DNA STRAND ELONGATION",
            "G1",
            "G2",
            "HOMOLOGOUS",
            "MISMATCH REPAIR",
            "MITOTIC",
            "NUCLEOTIDE-EXCISION",
            "PRE-REPLICATIVE",
            "RECOMBINATION",
            "REPLICATION STRESS",
            "S PHASE",
            "ATR",
        ],
        "literature": (
            "Effector differentiation and exhaustion are tightly coupled to proliferation history and replication stress. A cell-cycle "
            "signal is interpretable when it is directional, but it is usually a state-fitness mechanism rather than a lineage-specific "
            "transcription factor pathway."
        ),
    },
    {
        "id": "translation_ribosome",
        "name": "Translation, ribosome biogenesis, and RNA processing",
        "short_label": "translation/ribosome machinery",
        "keywords": [
            "MRNA",
            "MRNA PROCESSING",
            "PEPTIDE CHAIN",
            "RIBONUCLEOPROTEIN",
            "RIBOSOME",
            "RIBOSOMAL",
            "RRNA",
            "RRNA PROCESSING",
            "SPLIC",
            "SRP-DEPENDENT",
            "TRANSLATION",
        ],
        "exclude_keywords": [
            "MITOCHONDRIAL GENE EXPRESSION",
            "MITOCHONDRIAL TRANSLATION",
        ],
        "literature": (
            "CD8 T cell activation and terminal effector differentiation require large increases in protein synthesis, ribosome output, "
            "and RNA processing. Directional hits here usually mean the reporter phenotype is coupled to biosynthetic fitness, growth rate, "
            "or tolerance of activation-induced protein-production load."
        ),
    },
    {
        "id": "mitochondrial_fitness",
        "name": "Mitochondrial translation and respiratory fitness",
        "short_label": "mitochondrial fitness",
        "keywords": [
            "ELECTRON TRANSPORT",
            "MITOCHONDRIAL GENE EXPRESSION",
            "MITOCHONDRIAL TRANSLATION",
            "MITOCHONDR",
            "OXIDATIVE PHOSPHORYLATION",
            "RESPIRATORY",
            "TCA",
        ],
        "literature": (
            "Mitochondrial fitness helps determine whether activated CD8 T cells persist, become dysfunctional, or maintain effector function. "
            "Mitochondrial translation and respiration signals are therefore interpretable as survival and metabolic-capacity dependencies."
        ),
    },
    {
        "id": "chromatin_remodeling",
        "name": "Chromatin remodeling and histone regulation",
        "short_label": "chromatin-remodeling machinery",
        "keywords": [
            "ARID",
            "BAF",
            "CHROMATIN",
            "HISTONE",
            "EPIGEN",
            "METHYLATE HISTONE",
            "NUCLEOSOME",
            "RMT",
            "SMARC",
            "SWI/SNF",
        ],
        "exclude_keywords": [
            "GENE EXPRESSION (TRANSCRIPTION)",
            "GENERIC TRANSCRIPTION PATHWAY",
            "RNA POLYMERASE II TRANSCRIPTION",
        ],
        "literature": (
            "Activation-induced T cell fate changes require chromatin remodeling. SWI/SNF/BAF, histone-modification, and enhancer-accessibility "
            "modules can determine whether a stimulated CD8 T cell can open effector or exhaustion programs such as Prdm1, Eomes, and Tox."
        ),
    },
    {
        "id": "proteostasis",
        "name": "Proteostasis, ubiquitin, proteasome, and ER targeting",
        "short_label": "proteostasis and ubiquitin-proteasome control",
        "keywords": [
            "ANTIGEN PROCESSING: UBIQUITINATION",
            "ENDOPLASMIC",
            "NEDDYLATION",
            "PROTEASOME",
            "PROTEASOMAL",
            "PROTEIN DEGRADATION",
            "PROTEIN TARGETING",
            "PROTEOLYSIS",
            "UBIQUITIN",
            "UNFOLDED",
        ],
        "literature": (
            "The ubiquitin-proteasome and ER-stress systems control receptor abundance, transcription-factor stability, antigen presentation, "
            "and tolerance of activation-induced protein load. These are interpretable as signaling and fitness regulators rather than generic pathway noise."
        ),
    },
    {
        "id": "tgfb_smad",
        "name": "TGF-beta/SMAD signaling",
        "short_label": "TGF-beta/SMAD signaling",
        "keywords": [
            "SMAD",
            "TGF-BETA",
            "TGFB",
            "TRANSFORMING GROWTH FACTOR",
        ],
        "literature": (
            "TGF-beta signaling restrains effector CD8 function and can promote tissue-resident or dysfunctional programs depending on context. "
            "A directional signal here is interpretable because TGF-beta can oppose cytotoxic differentiation while also shaping exhausted-cell persistence."
        ),
    },
]


@dataclass(frozen=True)
class PathwayTerm:
    term_id: str
    source: str
    library: str
    term: str
    genes: tuple[str, ...]
    gene_keys: frozenset[str]


@dataclass(frozen=True)
class ScreenContext:
    title: str
    low_label: str
    high_label: str
    high_marker: str | None
    high_state: str
    low_state: str
    marker_biology: str


def available_cpus() -> int:
    try:
        return max(1, int(subprocess.check_output(["nproc"], text=True).strip()))
    except (OSError, subprocess.SubprocessError, ValueError):
        return max(1, os.cpu_count() or 1)


def default_catalog_paths(utils_dir: str | os.PathLike[str] | None = None) -> dict[str, tuple[str, Path]]:
    base_dir = Path(utils_dir) if utils_dir is not None else Path(__file__).resolve().parent
    catalog_dir = base_dir / "pathways" / "enrichr"
    return {
        source: (library, catalog_dir / f"{library}.gmt")
        for source, library in DEFAULT_ENRICHR_LIBRARIES.items()
    }


def _ordered_catalog_sources(
    paths: Mapping[str, str | os.PathLike[str] | tuple[str, str | os.PathLike[str]]],
) -> list[str]:
    ordered = [source for source in SOURCE_ORDER if source in paths]
    ordered.extend(source for source in paths if source not in SOURCE_ORDER)
    return ordered


def gene_key(value: object) -> str:
    return str(value).strip().upper()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def load_pathway_catalog(path: str | os.PathLike[str], source: str, library: str | None = None) -> list[PathwayTerm]:
    catalog_path = Path(path)
    library_name = library or catalog_path.stem
    terms: list[PathwayTerm] = []
    with catalog_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            term_name = parts[0].strip()
            genes: list[str] = []
            seen: set[str] = set()
            for raw_gene in parts[2:]:
                gene = raw_gene.strip()
                key = gene_key(gene)
                if gene and key not in seen:
                    genes.append(gene)
                    seen.add(key)
            terms.append(
                PathwayTerm(
                    term_id=f"{source}|{library_name}|{idx}",
                    source=source,
                    library=library_name,
                    term=term_name,
                    genes=tuple(genes),
                    gene_keys=frozenset(seen),
                )
            )
    return terms


def load_pathway_catalogs(
    catalog_paths: Mapping[str, str | os.PathLike[str] | tuple[str, str | os.PathLike[str]]] | None = None,
) -> list[PathwayTerm]:
    paths = catalog_paths or default_catalog_paths()
    terms: list[PathwayTerm] = []
    for source in _ordered_catalog_sources(paths):
        value = paths[source]
        if isinstance(value, tuple):
            library, path = value
        else:
            path = value
            library = Path(path).stem
        terms.extend(load_pathway_catalog(path, source=source, library=str(library)))
    return terms


def _clean_gene_info_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text in {"", "-", "nan", "None", "<NA>"} else text


def _split_gene_info_values(value: object) -> list[str]:
    values: list[str] = []
    for part in _clean_gene_info_value(value).split("|"):
        cleaned = _clean_gene_info_value(part)
        if cleaned:
            values.append(cleaned)
    return values


def _gene_info_row_dict(row: object) -> dict[str, object]:
    full_name = _clean_gene_info_value(getattr(row, "Full_name_from_nomenclature_authority"))
    description = _clean_gene_info_value(getattr(row, "description"))
    if not full_name:
        full_name = description
    return {
        "full_name": full_name,
        "description": description,
        "synonyms": _split_gene_info_values(getattr(row, "Synonyms")),
        "other_designations": _split_gene_info_values(getattr(row, "Other_designations")),
        "type_of_gene": _clean_gene_info_value(getattr(row, "type_of_gene")),
    }


def load_gene_info_lookup(gene_info_path: str | os.PathLike[str]) -> dict[str, dict[str, object]]:
    gene_info = pd.read_csv(gene_info_path, sep="\t", dtype=str)
    lookup: dict[str, dict[str, object]] = {}
    synonym_lookup: dict[str, dict[str, object]] = {}
    synonym_counts: dict[str, int] = {}
    for row in gene_info.itertuples(index=False):
        info = _gene_info_row_dict(row)
        if not info["full_name"]:
            continue
        for symbol_col in ["Symbol", "Symbol_from_nomenclature_authority"]:
            symbol = _clean_gene_info_value(getattr(row, symbol_col))
            key = gene_key(symbol)
            if key and key not in lookup:
                lookup[key] = info
        for synonym in info["synonyms"]:
            key = gene_key(synonym)
            if key:
                synonym_lookup[key] = info
                synonym_counts[key] = synonym_counts.get(key, 0) + 1
    for key, info in synonym_lookup.items():
        if key not in lookup and synonym_counts[key] == 1:
            lookup[key] = info
    return lookup


def load_gene_full_name_lookup(gene_info_path: str | os.PathLike[str]) -> dict[str, str]:
    return {
        key: str(info["full_name"])
        for key, info in load_gene_info_lookup(gene_info_path).items()
        if info.get("full_name")
    }


def _benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    n_tests = len(values)
    if n_tests == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n_tests, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _prepare_gene_table(
    genes: pd.DataFrame,
    gene_col: str,
    logfc_col: str,
    pvalue_col: str,
    include_rank: bool,
) -> pd.DataFrame:
    out = genes[[gene_col, logfc_col, pvalue_col]].copy()
    out = out.rename(columns={gene_col: "Gene", logfc_col: "logFC", pvalue_col: "pvalue"})
    out["Gene"] = out["Gene"].astype(str)
    out["gene_key"] = out["Gene"].map(gene_key)
    out = out[out["gene_key"] != ""].drop_duplicates("gene_key", keep="first").reset_index(drop=True)
    out["logFC"] = pd.to_numeric(out["logFC"], errors="coerce")
    out["pvalue"] = pd.to_numeric(out["pvalue"], errors="coerce")
    if include_rank:
        out["rank_position"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _add_gene_annotations(genes: pd.DataFrame, gene_info_lookup: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    out = genes.copy()
    out["gene_full_name"] = out["gene_key"].map(
        lambda key: str(gene_info_lookup.get(key, {}).get("full_name", ""))
    )
    out["gene_synonyms"] = out["gene_key"].map(
        lambda key: list(gene_info_lookup.get(key, {}).get("synonyms", []))
    )
    out["gene_other_designations"] = out["gene_key"].map(
        lambda key: list(gene_info_lookup.get(key, {}).get("other_designations", []))
    )
    return out


def _add_gene_full_names(genes: pd.DataFrame, gene_full_name_lookup: Mapping[str, str]) -> pd.DataFrame:
    return _add_gene_annotations(
        genes,
        {
            gene_key(gene): {
                "full_name": name,
                "synonyms": [],
                "other_designations": [],
            }
            for gene, name in gene_full_name_lookup.items()
        },
    )


def _rank_shift_pvalue(pathway_positions: list[int], n_universe: int) -> tuple[float, str, float]:
    n_pathway = len(pathway_positions)
    n_background = n_universe - n_pathway
    if n_pathway == 0 or n_background == 0:
        return 1.0, "NA", float("nan")

    rank_sum = float(np.sum(pathway_positions))
    u_stat = rank_sum - (n_pathway * (n_pathway + 1) / 2.0)
    expected = n_pathway * n_background / 2.0
    variance = n_pathway * n_background * (n_universe + 1.0) / 12.0
    z_score = (u_stat - expected) / math.sqrt(variance)
    pvalue = min(1.0, 2.0 * norm.sf(abs(z_score)))
    mean_rank = rank_sum / n_pathway
    direction = "top" if mean_rank < (n_universe + 1.0) / 2.0 else "bottom"
    return pvalue, direction, mean_rank


def _worker_count(n_items: int, n_jobs: int | None = None) -> int:
    return min(max(1, int(n_jobs or available_cpus())), max(1, n_items))


def _pathway_chunks(pathway_catalogs: Sequence[PathwayTerm], worker_count: int) -> list[Sequence[PathwayTerm]]:
    chunk_size = max(1, math.ceil(len(pathway_catalogs) / worker_count))
    return [
        pathway_catalogs[start:start + chunk_size]
        for start in range(0, len(pathway_catalogs), chunk_size)
    ]


def _gene_lookup(genes: pd.DataFrame | pd.Series | Sequence[object], gene_col: str = "Gene") -> dict[str, str]:
    lookup: dict[str, str] = {}
    if isinstance(genes, pd.DataFrame):
        if "gene_key" in genes.columns:
            key_values = genes["gene_key"]
        elif gene_col in genes.columns:
            key_values = genes[gene_col].map(gene_key)
        else:
            raise ValueError(f"Gene table must contain 'gene_key' or '{gene_col}'.")

        if gene_col in genes.columns:
            name_values = genes[gene_col]
        elif "Gene" in genes.columns:
            name_values = genes["Gene"]
        else:
            name_values = key_values

        for raw_key, raw_name in zip(key_values, name_values):
            key = gene_key(raw_key)
            if key and key not in lookup:
                lookup[key] = str(raw_name)
        return lookup

    values = genes.tolist() if isinstance(genes, pd.Series) else list(genes)
    for value in values:
        key = gene_key(value)
        if key and key not in lookup:
            lookup[key] = str(value)
    return lookup


def _fisher_enrichment(
    pathway_keys: set[str],
    selected_keys: set[str],
    universe_keys: set[str],
) -> tuple[float, float]:
    overlap = len(pathway_keys & selected_keys)
    selected_not_pathway = len(selected_keys) - overlap
    pathway_not_selected = len(pathway_keys) - overlap
    neither = len(universe_keys) - overlap - selected_not_pathway - pathway_not_selected
    table = [[overlap, selected_not_pathway], [pathway_not_selected, neither]]
    test = fisher_exact(table, alternative="greater")
    odds_ratio = float(test.statistic)
    if not np.isfinite(odds_ratio) or odds_ratio <= 0:
        odds_ratio = ((overlap + 0.5) * (neither + 0.5)) / (
            (selected_not_pathway + 0.5) * (pathway_not_selected + 0.5)
        )
    log2_odds_ratio = math.log2(odds_ratio) if odds_ratio > 0 else math.nan
    return float(test.pvalue), log2_odds_ratio


def _test_gene_set_pathway_chunk(
    terms: Sequence[PathwayTerm],
    selected_keys: set[str],
    universe_keys: set[str],
    gene_names: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for term in terms:
        pathway_keys = set(term.gene_keys) & universe_keys
        overlap_keys = pathway_keys & selected_keys
        tested = bool(selected_keys) and bool(pathway_keys) and bool(universe_keys)
        if tested:
            pvalue, log2_odds_ratio = _fisher_enrichment(pathway_keys, selected_keys, universe_keys)
        else:
            pvalue = math.nan
            log2_odds_ratio = math.nan
        rows.append(
            {
                "term_id": term.term_id,
                "source": term.source,
                "library": term.library,
                "pathway": term.term,
                "tested": tested,
                "pvalue": pvalue,
                "log2_odds_ratio": log2_odds_ratio,
                "overlap": len(overlap_keys),
                "pathway_size_in_universe": len(pathway_keys),
                "selected_gene_count": len(selected_keys),
                "universe_gene_count": len(universe_keys),
                "overlap_gene_keys": ";".join(sorted(overlap_keys)),
                "overlap_genes": ";".join(sorted(gene_names[key] for key in overlap_keys)),
            }
        )
    return rows


def test_gene_set_pathways(
    selected_genes: pd.DataFrame | pd.Series | Sequence[object],
    universe_genes: pd.DataFrame | pd.Series | Sequence[object],
    pathway_catalogs: Sequence[PathwayTerm] | None = None,
    catalog_paths: Mapping[str, str | os.PathLike[str] | tuple[str, str | os.PathLike[str]]] | None = None,
    fdr_threshold: float = PATHWAY_FDR_THRESHOLD,
    pvalue_threshold: float = 0.05,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    catalogs = list(pathway_catalogs) if pathway_catalogs is not None else load_pathway_catalogs(catalog_paths)
    universe_lookup = _gene_lookup(universe_genes)
    selected_lookup = _gene_lookup(selected_genes)
    universe_keys = set(universe_lookup)
    selected_keys = set(selected_lookup) & universe_keys

    if not catalogs:
        return pd.DataFrame()

    worker_count = _worker_count(len(catalogs), n_jobs)
    chunks = _pathway_chunks(catalogs, worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_results = list(
            executor.map(
                lambda chunk: _test_gene_set_pathway_chunk(
                    chunk,
                    selected_keys=selected_keys,
                    universe_keys=universe_keys,
                    gene_names=universe_lookup,
                ),
                chunks,
            )
        )

    rows = [row for chunk in chunk_results for row in chunk]
    results = pd.DataFrame(rows)
    results["fdr"] = np.nan
    for source, source_idx in results.groupby("source").groups.items():
        idx = list(source_idx)
        tested_idx = results.loc[idx].index[results.loc[idx, "tested"].astype(bool)]
        if len(tested_idx):
            results.loc[tested_idx, "fdr"] = _benjamini_hochberg(
                results.loc[tested_idx, "pvalue"].to_numpy(dtype=float)
            )
    results["significant_nominal"] = (
        results["tested"].astype(bool)
        & (results["pvalue"].astype(float) < pvalue_threshold)
    )
    results["significant_fdr"] = (
        results["tested"].astype(bool)
        & (results["fdr"].astype(float) < fdr_threshold)
    )
    return results


def _test_ranked_pathway_chunk(
    terms: Sequence[PathwayTerm],
    rank_positions: dict[str, int],
    universe_keys: set[str],
    gene_names: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    universe_size = len(universe_keys)
    for term in terms:
        pathway_keys = set(term.gene_keys) & universe_keys
        pathway_positions = [rank_positions[key] for key in pathway_keys]
        ranked_pvalue, ranked_direction, mean_rank = _rank_shift_pvalue(pathway_positions, universe_size)
        if ranked_direction == "top":
            ranked_gene_keys = sorted(pathway_keys, key=lambda key: rank_positions[key])[:12]
        elif ranked_direction == "bottom":
            ranked_gene_keys = sorted(pathway_keys, key=lambda key: rank_positions[key], reverse=True)[:12]
        else:
            ranked_gene_keys = []
        rows.append(
            {
                "term_id": term.term_id,
                "source": term.source,
                "library": term.library,
                "pathway": term.term,
                "pathway_size_in_ranked_list": len(pathway_keys),
                "ranked_overlap": len(pathway_keys),
                "ranked_mean_rank": mean_rank,
                "ranked_direction": ranked_direction,
                "ranked_pvalue": ranked_pvalue,
                "ranked_genes": ";".join(gene_names[key] for key in ranked_gene_keys),
            }
        )
    return rows


def test_ranked_pathways(
    ranked_genes: pd.DataFrame,
    pathway_catalogs: Sequence[PathwayTerm] | None = None,
    catalog_paths: Mapping[str, str | os.PathLike[str] | tuple[str, str | os.PathLike[str]]] | None = None,
    fdr_threshold: float = PATHWAY_FDR_THRESHOLD,
    pvalue_threshold: float = 0.05,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    if "gene_key" not in ranked_genes.columns or "rank_position" not in ranked_genes.columns:
        raise ValueError("ranked_genes must contain 'gene_key' and 'rank_position'.")

    catalogs = list(pathway_catalogs) if pathway_catalogs is not None else load_pathway_catalogs(catalog_paths)
    rank_positions = dict(zip(ranked_genes["gene_key"], ranked_genes["rank_position"].astype(int)))
    universe_keys = set(rank_positions)
    if "Gene" in ranked_genes.columns:
        gene_names = dict(zip(ranked_genes["gene_key"], ranked_genes["Gene"]))
    else:
        gene_names = {key: key for key in universe_keys}

    if not catalogs:
        return pd.DataFrame()

    worker_count = _worker_count(len(catalogs), n_jobs)
    chunks = _pathway_chunks(catalogs, worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_results = list(
            executor.map(
                lambda chunk: _test_ranked_pathway_chunk(
                    chunk,
                    rank_positions=rank_positions,
                    universe_keys=universe_keys,
                    gene_names=gene_names,
                ),
                chunks,
            )
        )

    rows = [row for chunk in chunk_results for row in chunk]
    results = pd.DataFrame(rows)
    for source, source_idx in results.groupby("source").groups.items():
        idx = list(source_idx)
        results.loc[idx, "ranked_fdr"] = _benjamini_hochberg(
            results.loc[idx, "ranked_pvalue"].to_numpy(dtype=float)
        )
    results["ranked_significant_nominal"] = results["ranked_pvalue"].astype(float) < pvalue_threshold
    results["ranked_significant_fdr"] = results["ranked_fdr"].astype(float) < fdr_threshold
    return results


def test_pathways(
    ranked_genes: pd.DataFrame,
    significant_up_genes: pd.DataFrame,
    significant_down_genes: pd.DataFrame,
    pathway_catalogs: Sequence[PathwayTerm],
    fdr_threshold: float = PATHWAY_FDR_THRESHOLD,
    min_directional_genes: int = MIN_DIRECTIONAL_PATHWAY_GENES,
    pvalue_threshold: float = 0.05,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    _ = min_directional_genes  # Compatibility argument; Fisher tests no longer use a gene-count threshold.
    catalogs = list(pathway_catalogs)
    ranked = test_ranked_pathways(
        ranked_genes=ranked_genes,
        pathway_catalogs=catalogs,
        fdr_threshold=fdr_threshold,
        pvalue_threshold=pvalue_threshold,
        n_jobs=n_jobs,
    )
    up = test_gene_set_pathways(
        selected_genes=significant_up_genes,
        universe_genes=ranked_genes,
        pathway_catalogs=catalogs,
        fdr_threshold=fdr_threshold,
        pvalue_threshold=pvalue_threshold,
        n_jobs=n_jobs,
    )
    down = test_gene_set_pathways(
        selected_genes=significant_down_genes,
        universe_genes=ranked_genes,
        pathway_catalogs=catalogs,
        fdr_threshold=fdr_threshold,
        pvalue_threshold=pvalue_threshold,
        n_jobs=n_jobs,
    )

    merge_cols = ["term_id", "source", "library", "pathway"]
    up = up.rename(
        columns={
            "tested": "up_tested",
            "pvalue": "up_pvalue",
            "fdr": "up_fdr",
            "log2_odds_ratio": "up_log2_odds_ratio",
            "overlap": "up_overlap",
            "pathway_size_in_universe": "up_pathway_size_in_universe",
            "selected_gene_count": "up_selected_gene_count",
            "universe_gene_count": "up_universe_gene_count",
            "overlap_gene_keys": "up_gene_keys",
            "overlap_genes": "up_genes",
            "significant_nominal": "up_significant_nominal",
            "significant_fdr": "up_significant_fdr",
        }
    )
    down = down.rename(
        columns={
            "tested": "down_tested",
            "pvalue": "down_pvalue",
            "fdr": "down_fdr",
            "log2_odds_ratio": "down_log2_odds_ratio",
            "overlap": "down_overlap",
            "pathway_size_in_universe": "down_pathway_size_in_universe",
            "selected_gene_count": "down_selected_gene_count",
            "universe_gene_count": "down_universe_gene_count",
            "overlap_gene_keys": "down_gene_keys",
            "overlap_genes": "down_genes",
            "significant_nominal": "down_significant_nominal",
            "significant_fdr": "down_significant_fdr",
        }
    )

    up_cols = merge_cols + [
        "up_tested",
        "up_pvalue",
        "up_fdr",
        "up_log2_odds_ratio",
        "up_overlap",
        "up_pathway_size_in_universe",
        "up_selected_gene_count",
        "up_universe_gene_count",
        "up_gene_keys",
        "up_genes",
        "up_significant_nominal",
        "up_significant_fdr",
    ]
    down_cols = merge_cols + [
        "down_tested",
        "down_pvalue",
        "down_fdr",
        "down_log2_odds_ratio",
        "down_overlap",
        "down_pathway_size_in_universe",
        "down_selected_gene_count",
        "down_universe_gene_count",
        "down_gene_keys",
        "down_genes",
        "down_significant_nominal",
        "down_significant_fdr",
    ]
    results = ranked.merge(up[up_cols], on=merge_cols, how="left", validate="one_to_one")
    results = results.merge(down[down_cols], on=merge_cols, how="left", validate="one_to_one")

    results["ranked_significant"] = results["ranked_significant_fdr"].astype(bool)
    results["up_significant"] = results["up_significant_fdr"].fillna(False).astype(bool)
    results["down_significant"] = results["down_significant_fdr"].fillna(False).astype(bool)
    results["significant"] = results[["ranked_significant", "up_significant", "down_significant"]].any(axis=1)
    results["significant_tests"] = results.apply(_significant_test_labels, axis=1)
    results["_up_sort_fdr"] = results["up_fdr"].fillna(1.0)
    results["_down_sort_fdr"] = results["down_fdr"].fillna(1.0)
    return results.sort_values(
        ["significant", "source", "ranked_fdr", "_up_sort_fdr", "_down_sort_fdr", "pathway"],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_up_sort_fdr", "_down_sort_fdr"]).reset_index(drop=True)


def _significant_test_labels(row: pd.Series) -> str:
    labels: list[str] = []
    if bool(row["ranked_significant"]):
        labels.append(f"ranked:{row['ranked_direction']}")
    if bool(row["up_significant"]):
        labels.append("up")
    if bool(row["down_significant"]):
        labels.append("down")
    return ";".join(labels)


def _significant_term_direction_lookup(pathway_results: pd.DataFrame) -> dict[str, frozenset[str]]:
    lookup: dict[str, frozenset[str]] = {}
    for row in pathway_results[pathway_results["significant"]].itertuples(index=False):
        directions: set[str] = set()
        if bool(row.up_significant):
            directions.add("up")
        if bool(row.down_significant):
            directions.add("down")
        if bool(row.ranked_significant):
            if row.ranked_direction == "top":
                directions.add("up")
            elif row.ranked_direction == "bottom":
                directions.add("down")
        if directions:
            lookup[str(row.term_id)] = frozenset(directions)
    return lookup


def build_gene_membership(
    ranked_genes: pd.DataFrame,
    pathway_catalogs: Sequence[PathwayTerm],
    term_direction_lookup: Mapping[str, frozenset[str]],
) -> dict[str, dict[str, list[tuple[str, frozenset[str]]]]]:
    universe_keys = set(ranked_genes["gene_key"])
    sources = list(SOURCE_ORDER)
    for term in pathway_catalogs:
        if term.source not in sources:
            sources.append(term.source)
    membership: dict[str, dict[str, list[tuple[str, frozenset[str]]]]] = {
        key: {source: [] for source in sources}
        for key in universe_keys
    }
    for term in pathway_catalogs:
        directions = term_direction_lookup.get(term.term_id, frozenset())
        if not directions:
            continue
        for key in sorted(term.gene_keys & universe_keys):
            membership[key].setdefault(term.source, []).append((term.term, directions))
    return membership


def _term_matches_direction(directions: frozenset[str], direction: str | None) -> bool:
    return direction is None or direction in directions


def _gene_direction_from_logfc(value: object) -> str:
    return "up" if float(value) > 0 else "down"


def _plain_pathway_cell(terms: list[tuple[str, frozenset[str]]], direction: str | None = None) -> str:
    return "; ".join(term for term, directions in terms if _term_matches_direction(directions, direction))


def _html_pathway_cell(terms: list[tuple[str, frozenset[str]]], direction: str | None = None) -> str:
    parts: list[str] = []
    for term, directions in terms:
        if not _term_matches_direction(directions, direction):
            continue
        escaped = html.escape(term)
        parts.append(f'<span class="significant-pathway">{escaped}</span>')
    return "; ".join(parts)


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        key = cleaned.upper()
        if cleaned and key not in seen:
            out.append(cleaned)
            seen.add(key)
    return out


def _gene_membership_terms(
    membership: Mapping[str, Mapping[str, list[tuple[str, frozenset[str]]]]],
    key: str,
    direction: str | None = None,
) -> list[str]:
    terms: list[str] = []
    for source in SOURCE_ORDER:
        for term, directions in membership.get(key, {}).get(source, []):
            if _term_matches_direction(directions, direction):
                terms.append(term)
    return _dedupe_preserve_order(terms)


def _informative_pathway_labels(terms: Sequence[str], max_terms: int = 3) -> list[str]:
    labels: list[str] = []
    excluded_keywords = [
        *COMMON_NOISE_PATHWAY_KEYWORDS,
        *GENERIC_DISPLAY_PATHWAY_KEYWORDS,
        *GENE_DESCRIPTION_GENERIC_PATHWAY_KEYWORDS,
    ]
    for term in terms:
        cleaned = _clean_pathway_name(term)
        if not cleaned or _contains_keyword(cleaned, excluded_keywords):
            continue
        labels.append(cleaned)
    if not labels:
        labels = [_clean_pathway_name(term) for term in terms if _clean_pathway_name(term)]
    return _dedupe_preserve_order(labels)[:max_terms]


def _program_labels_from_terms(terms: Sequence[str], max_labels: int = 3) -> list[str]:
    labels: list[str] = []
    for program in INTERPRETABLE_PROGRAMS:
        exclude_keywords = [
            *COMMON_NOISE_PATHWAY_KEYWORDS,
            *GENERIC_DISPLAY_PATHWAY_KEYWORDS,
            *GENE_DESCRIPTION_GENERIC_PATHWAY_KEYWORDS,
            *list(program.get("exclude_keywords", [])),
        ]
        for term in terms:
            if _contains_keyword(term, program["keywords"]) and not _contains_keyword(term, exclude_keywords):
                labels.append(str(program["short_label"]))
                break
    return _dedupe_preserve_order(labels)[:max_labels]


def _designation_examples(row: object, max_designations: int = 3) -> list[str]:
    blocked = {
        str(getattr(row, "Gene", "")).upper(),
        str(getattr(row, "gene_full_name", "")).upper(),
    }
    other_designations = list(getattr(row, "gene_other_designations", []) or [])
    synonyms = list(getattr(row, "gene_synonyms", []) or [])

    def filtered(values: Sequence[str]) -> list[str]:
        kept: list[str] = []
        for candidate in _dedupe_preserve_order(values):
            upper = candidate.upper()
            if upper in blocked or upper == "":
                continue
            kept.append(candidate)
        return sorted(kept, key=lambda value: (len(value) > 70, len(value), value.lower()))

    kept = filtered(other_designations)
    if len(kept) < max_designations:
        kept.extend(filtered(synonyms))
    return _dedupe_preserve_order(kept)[:max_designations]


def _cohit_pathway_terms(terms: Sequence[str]) -> list[str]:
    excluded_keywords = [
        *COMMON_NOISE_PATHWAY_KEYWORDS,
        *GENERIC_DISPLAY_PATHWAY_KEYWORDS,
        *GENE_DESCRIPTION_GENERIC_PATHWAY_KEYWORDS,
    ]
    filtered = [
        term
        for term in terms
        if not _contains_keyword(term, excluded_keywords)
    ]
    return filtered if filtered else list(terms)


def _cohit_label(shared_terms: Sequence[str]) -> list[str]:
    informative_terms = _cohit_pathway_terms(shared_terms)
    return _informative_pathway_labels(informative_terms, max_terms=1)


def _program_function_sentence(program_labels: Sequence[str]) -> str:
    label_sentences = {
        "TCR/cytokine signaling": (
            "This points to receptor or cytokine signaling that controls how strongly the T cell receives activation and differentiation cues."
        ),
        "cell-fate transcriptional regulators": (
            "This points to transcriptional state control, where the KO may move cells between progenitor-like, effector, or exhausted-like programs."
        ),
        "PI3K/AKT/mTOR-metabolic signaling": (
            "This points to nutrient sensing and anabolic growth control rather than a simple marker gene effect."
        ),
        "cell-cycle and DNA-replication machinery": (
            "This points to cycling, S-phase entry, DNA replication, or mitotic fitness rather than a T cell lineage marker."
        ),
        "translation/ribosome machinery": (
            "This points to protein-synthesis capacity and growth fitness, which can strongly bias reporter-bin recovery."
        ),
        "mitochondrial fitness": (
            "This points to respiratory capacity or mitochondrial stress tolerance in the sorted population."
        ),
        "chromatin-remodeling machinery": (
            "This points to chromatin accessibility or nucleosome regulation that can change whether cell-state programs are reachable."
        ),
        "proteostasis and ubiquitin-proteasome control": (
            "This points to protein turnover, ER stress, or ubiquitin-proteasome control of signaling and transcription-factor stability."
        ),
        "TGF-beta/SMAD signaling": (
            "This points to TGF-beta/SMAD signaling, a pathway that often restrains effector T cell function and reshapes differentiated states."
        ),
    }
    for label in program_labels:
        if label in label_sentences:
            return label_sentences[label]
    return ""


def _keyword_gene_function_note(gene: str, full_name: str, label_sentence: str) -> str:
    text = full_name.lower()
    keyword_notes = [
        (
            ["transcription factor", "zinc finger", "dna binding", "dna-binding"],
            f"{gene} likely acts at the transcriptional-regulation layer, changing which cell-state genes can be turned on or off.",
        ),
        (
            ["kinase"],
            f"{gene} likely controls signaling by phosphorylating target proteins, so the screen effect is best read as altered pathway tone.",
        ),
        (
            ["phosphatase"],
            f"{gene} likely removes phosphorylation marks from signaling or cell-cycle proteins, changing pathway timing and intensity.",
        ),
        (
            ["ubiquitin", "e3 ligase", "ligase"],
            f"{gene} likely controls protein turnover, which can shift signaling by changing how long key regulators persist.",
        ),
        (
            ["proteasome"],
            f"{gene} likely affects proteasome-mediated protein degradation and the ability of activated cells to handle protein-turnover stress.",
        ),
        (
            ["pre-mrna", "splicing", "spliceosome", "snrnp", "rna processing"],
            f"{gene} likely affects RNA splicing or RNA maturation, which can reshape many activation and survival programs at once.",
        ),
        (
            ["ribosome", "ribosomal", "rrna", "nucleolar"],
            f"{gene} likely affects ribosome production or translation capacity, a common growth and activation-fitness axis in T cells.",
        ),
        (
            ["mitochond"],
            f"{gene} likely affects mitochondrial function, so the screen effect may reflect respiratory fitness or stress tolerance.",
        ),
        (
            ["dna polymerase"],
            f"{gene} likely helps copy or repair DNA, so the hit points to replication fidelity and cell-cycle fitness.",
        ),
        (
            ["rna polymerase"],
            f"{gene} likely helps make RNA, so the hit points to transcriptional capacity rather than one specific downstream pathway.",
        ),
        (
            ["helicase"],
            f"{gene} likely unwinds DNA or RNA, so the screen effect may come from replication, repair, or RNA-processing stress.",
        ),
        (
            ["cohesin", "condensin", "centromere", "kinetochore", "chromosome"],
            f"{gene} likely affects chromosome organization or segregation, so the hit is a mitotic-fitness signal.",
        ),
        (
            ["histone", "chromatin", "nucleosome"],
            f"{gene} likely affects chromatin packaging, which can change both proliferation and access to cell-state programs.",
        ),
        (
            ["actin", "arp2/3", "tubulin", "microtubule"],
            f"{gene} likely affects the cytoskeleton, which can influence mitosis, cell shape, and signaling-complex organization.",
        ),
        (
            ["chaperone", "heat shock"],
            f"{gene} likely helps proteins fold or remain stable under stress, making it a proteostasis and activation-load hit.",
        ),
        (
            ["h+ transporting", "vacuolar", "lysosomal"],
            f"{gene} likely affects lysosome acidification or vesicle function, which can feed into nutrient sensing and mTOR signaling.",
        ),
        (
            ["glycosyltransferase", "galactosyltransferase", "glycosylation"],
            f"{gene} likely affects glycosylation, which can change receptor trafficking, surface protein stability, and cell signaling.",
        ),
        (
            ["transporter", "solute carrier", "abc transporter"],
            f"{gene} likely affects transport across membranes, changing nutrient, ion, or metabolite availability.",
        ),
        (
            ["receptor", "cd3", "lymphocyte"],
            f"{gene} likely affects receptor signaling, so the hit may change how strongly cells sense activation cues.",
        ),
        (
            ["adaptor", "adapter", "scaffold"],
            f"{gene} likely works as a scaffold/adaptor that assembles signaling complexes rather than acting as an enzyme itself.",
        ),
        (
            ["dna repair", "repair"],
            f"{gene} likely protects genome integrity, so the screen effect may reflect DNA-damage tolerance during proliferation.",
        ),
        (
            ["methyltransferase", "acetyltransferase", "deacetylase", "demethylase"],
            f"{gene} likely edits chromatin or protein modification marks, which can redirect transcriptional state and stress responses.",
        ),
        (
            ["apoptosis", "cell death"],
            f"{gene} likely affects survival or programmed cell death, so the hit may change which cells persist through the sort.",
        ),
        (
            ["gator", "ragulator", "mtor"],
            f"{gene} likely controls nutrient sensing upstream of mTOR, connecting amino-acid availability to growth and differentiation.",
        ),
    ]
    for keywords, note in keyword_notes:
        if any(keyword in text for keyword in keywords):
            return note
    if label_sentence:
        return f"{gene} is best read here through its screen pathway context. {label_sentence}"
    if full_name:
        return f"{gene} has no sharp curated mechanism in this report; treat it as a gene-level candidate and use the direction, co-hits, and validation data to interpret it."
    return f"{gene} has no clear functional description in this report; treat it as a candidate for follow-up annotation and validation."


def _curated_gene_function_note(row: object, program_labels: Sequence[str]) -> str:
    gene = str(getattr(row, "Gene", "")).strip()
    key = gene_key(gene)
    full_name = str(getattr(row, "gene_full_name", "")).strip()
    label_sentence = _program_function_sentence(program_labels)

    exact_notes = {
        "BIRC5": (
            "Birc5 is survivin, an inhibitor-of-apoptosis-repeat protein that also works at the chromosomal passenger complex during mitosis. "
            "A Birc5 hit is usually a cycling/mitotic fitness signal, especially chromosome segregation and cytokinesis, not a T cell fate marker by itself."
        ),
        "DBF4": (
            "Dbf4 stands for dumbbell former 4; the name comes from yeast mutants that stalled with dumbbell-shaped cells. "
            "Functionally, mammalian DBF4 activates CDC7 kinase to fire DNA replication origins, so a Dbf4 hit is an S-phase/origin-firing signal."
        ),
        "CDC45": (
            "Cdc45 is a cell division cycle gene that joins MCM and GINS to form the CMG helicase, the engine that unwinds DNA at active replication forks."
        ),
        "GMNN": (
            "Gmnn encodes geminin, a licensing brake that prevents replication origins from firing more than once per cell cycle."
        ),
        "ESPL1": (
            "Espl1 encodes separase, the protease that cleaves cohesin so sister chromatids can separate at anaphase."
        ),
        "NIPBL": (
            "Nipbl loads cohesin onto chromatin, so hits here usually read as chromosome cohesion, enhancer architecture, and cell-cycle control."
        ),
        "MAU2": (
            "Mau2 partners with NIPBL to load cohesin onto chromosomes, tying this hit to sister-chromatid cohesion and chromatin architecture."
        ),
        "SMC3": (
            "Smc3 is a structural maintenance of chromosomes cohesin subunit; it holds sister chromatids and helps organize chromatin loops."
        ),
        "SMC4": (
            "Smc4 is part of condensin, the chromosome-condensation machinery needed to package chromosomes for mitosis."
        ),
        "SKP1": (
            "Skp1 is the scaffold of SCF ubiquitin ligases, which mark cell-cycle and signaling proteins for degradation."
        ),
        "SKP2": (
            "Skp2 is an SCF ubiquitin-ligase substrate receptor that promotes cell-cycle progression by targeting inhibitors such as p27-like proteins."
        ),
        "MTOR": (
            "Mtor is the mechanistic target of rapamycin kinase, the central nutrient-and-growth sensor of mTORC1/mTORC2 signaling."
        ),
        "RPTOR": (
            "Rptor encodes raptor, the scaffold that helps mTORC1 recognize substrates and respond to nutrients."
        ),
        "MLST8": (
            "Mlst8 encodes mLST8/GbetaL, a core mTOR complex subunit that stabilizes the kinase domain."
        ),
        "LAMTOR2": (
            "Lamtor2 is part of the lysosomal Ragulator complex that helps recruit and activate mTORC1 in response to amino acids."
        ),
        "HIF1A": (
            "Hif1a encodes hypoxia-inducible factor 1 alpha, the oxygen-sensitive transcription factor that turns on hypoxia and glycolysis programs."
        ),
        "ZAP70": (
            "Zap70 means zeta-chain-associated protein kinase of 70 kDa; it is the proximal kinase that transmits TCR signaling from CD3 zeta chains."
        ),
        "LCP2": (
            "Lcp2 encodes SLP-76, an adaptor that organizes LAT/TCR signaling complexes downstream of ZAP70."
        ),
        "CD3E": (
            "Cd3e encodes the CD3 epsilon signaling chain of the T cell receptor complex, so it is a direct TCR-signal-strength hit."
        ),
        "CD3D": (
            "Cd3d encodes the CD3 delta signaling chain of the T cell receptor complex, so it is a direct TCR-signal-strength hit."
        ),
        "CDKN1A": (
            "Cdkn1a encodes p21, a cyclin-dependent kinase inhibitor induced by p53 and stress signals to slow cell-cycle progression."
        ),
        "MAPKAPK2": (
            "Mapkapk2 encodes MK2, a stress-activated kinase downstream of p38 MAPK that controls inflammatory signaling and RNA stability."
        ),
        "PPP6C": (
            "Ppp6c encodes the catalytic subunit of protein phosphatase 6, a phosphatase tied to mitotic spindle control, DNA damage responses, and signaling tone."
        ),
        "DR1": (
            "Dr1 is a down-regulator of transcription that represses TBP/TFIID-dependent transcription, so the screen hit is most naturally read as transcriptional control."
        ),
    }
    if key in exact_notes:
        return exact_notes[key]

    if key.startswith("MCM"):
        return (
            f"{gene} is a minichromosome-maintenance helicase component. MCM proteins license replication origins and become part of the helicase that unwinds DNA during S phase."
        )
    if key.startswith("ORC"):
        return (
            f"{gene} is an origin-recognition-complex component. ORC marks replication origins and helps load the MCM helicase before S phase."
        )
    if key.startswith("POLD"):
        return (
            f"{gene} is part of DNA polymerase delta, the replication and repair polymerase that extends DNA, especially during lagging-strand synthesis."
        )
    if key.startswith("ANAPC"):
        return (
            f"{gene} is part of the anaphase-promoting complex/cyclosome, an E3 ubiquitin ligase that drives mitotic progression and exit."
        )
    if key.startswith("NDUF"):
        return (
            f"{gene} is a Complex I/NADH:ubiquinone oxidoreductase subunit, so this hit points to mitochondrial electron transport and respiratory fitness."
        )
    if key.startswith("UQCR"):
        return (
            f"{gene} is a Complex III ubiquinol-cytochrome c reductase component, so this hit points to mitochondrial electron transport."
        )
    if key.startswith("COX"):
        return (
            f"{gene} is a cytochrome c oxidase/Complex IV component, so this hit points to mitochondrial respiration."
        )
    if key.startswith("ATP5"):
        return (
            f"{gene} is part of mitochondrial ATP synthase, the machine that converts the proton gradient into ATP."
        )
    if key.startswith("ATP6V"):
        return (
            f"{gene} is a vacuolar H+-ATPase subunit. V-ATPase acidifies lysosomes and helps connect amino-acid sensing to mTOR signaling."
        )
    if key.startswith("MRPL") or key.startswith("MRPS"):
        return (
            f"{gene} is a mitochondrial ribosomal protein, so this hit points to mitochondrial translation and respiratory-chain maintenance."
        )
    if key.startswith("RPL") or key.startswith("RPS"):
        return (
            f"{gene} is a cytosolic ribosomal protein. These hits usually report protein-synthesis capacity, growth rate, or sensitivity to activation load."
        )
    if key.startswith("EIF") or key.startswith("EEF"):
        return (
            f"{gene} is a translation factor, so this hit points to protein-synthesis initiation or elongation rather than a lineage-specific regulator."
        )
    if re.match(r"^[A-Z]{1,4}ARS2?$", key):
        return (
            f"{gene} is an aminoacyl-tRNA synthetase, the enzyme class that charges tRNAs with amino acids for translation."
        )
    if key.startswith("POLR"):
        return (
            f"{gene} is an RNA polymerase subunit, so this hit points to transcriptional capacity and RNA production."
        )
    if key.startswith("PSM"):
        return (
            f"{gene} is a proteasome subunit or proteasome-associated factor, so this hit points to protein degradation and stress tolerance."
        )
    if key.startswith("CCT"):
        return (
            f"{gene} is part of the CCT/TRiC chaperonin that folds actin, tubulin, and other complex proteins needed for growth and division."
        )
    if key.startswith("H2") or key.startswith("H3") or key.startswith("H4"):
        return (
            f"{gene} is a histone gene, so this hit points to chromatin packaging and DNA-replication-coupled nucleosome assembly."
        )
    if key.startswith("TUB"):
        return (
            f"{gene} is a tubulin-family gene, so this hit points to microtubules, spindle assembly, and mitotic mechanics."
        )

    if label_sentence and full_name:
        return _keyword_gene_function_note(gene, full_name, label_sentence)
    if label_sentence:
        return f"{gene} is best read here through its pathway context. {label_sentence}"
    if full_name:
        return _keyword_gene_function_note(gene, full_name, label_sentence)
    return f"{gene} has no clear functional description in this report; treat it as a candidate for follow-up annotation and validation."


def _same_direction_cohit_lookup(
    up: pd.DataFrame,
    down: pd.DataFrame,
    membership: Mapping[str, Mapping[str, list[tuple[str, frozenset[str]]]]],
    max_genes: int = 6,
) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for direction, genes in [("up", up), ("down", down)]:
        key_to_terms: dict[str, set[str]] = {}
        key_to_name: dict[str, str] = {}
        key_to_pvalue: dict[str, float] = {}
        term_to_keys: dict[str, set[str]] = {}
        for row in genes.itertuples(index=False):
            key = row.gene_key
            terms = set(_cohit_pathway_terms(_gene_membership_terms(membership, key, direction)))
            key_to_terms[key] = terms
            key_to_name[key] = str(row.Gene)
            key_to_pvalue[key] = float(row.pvalue) if not pd.isna(row.pvalue) else 1.0
            for term in terms:
                term_to_keys.setdefault(term, set()).add(key)

        for key, terms in key_to_terms.items():
            shared_counts: dict[str, int] = {}
            shared_terms: dict[str, list[str]] = {}
            for term in terms:
                for other_key in term_to_keys.get(term, set()):
                    if other_key == key:
                        continue
                    shared_counts[other_key] = shared_counts.get(other_key, 0) + 1
                    shared_terms.setdefault(other_key, []).append(term)
            ranked_neighbors = sorted(
                shared_counts,
                key=lambda other_key: (
                    -shared_counts[other_key],
                    key_to_pvalue.get(other_key, 1.0),
                    key_to_name.get(other_key, other_key),
                ),
            )
            neighbor_labels: list[str] = []
            for other_key in ranked_neighbors[:max_genes]:
                labels = _cohit_label(shared_terms.get(other_key, []))
                if labels:
                    neighbor_labels.append(f"{key_to_name.get(other_key, other_key)} ({labels[0]})")
                else:
                    neighbor_labels.append(key_to_name.get(other_key, other_key))
            lookup[key] = neighbor_labels
    return lookup


def _gene_function_description(
    row: object,
    membership: Mapping[str, Mapping[str, list[tuple[str, frozenset[str]]]]],
    cohit_lookup: Mapping[str, list[str]],
) -> str:
    direction = _gene_direction_from_logfc(row.logFC)
    terms = _gene_membership_terms(membership, row.gene_key, direction)
    program_labels = _program_labels_from_terms(terms)
    pathway_labels = _informative_pathway_labels(terms, max_terms=2)
    parts: list[str] = [_curated_gene_function_note(row, program_labels)]

    if program_labels and pathway_labels:
        parts.append(
            f"Direction-specific pathway support: {', '.join(program_labels)} "
            f"({'; '.join(pathway_labels)})."
        )
    elif program_labels:
        parts.append(f"Direction-specific pathway support: {', '.join(program_labels)}.")
    elif pathway_labels:
        parts.append(f"Direction-specific pathway support: {'; '.join(pathway_labels)}.")
    else:
        parts.append("No direction-specific pathway passed FDR for this gene.")

    neighbors = cohit_lookup.get(row.gene_key, [])
    if neighbors:
        cohit_text = f"Same-direction pathway co-hits: {'; '.join(neighbors)}."
    else:
        cohit_text = "No same-direction pathway co-hits passed FDR."
    return f"{' '.join(parts)}\n\n{cohit_text}"


def _add_gene_function_descriptions(
    genes: pd.DataFrame,
    membership: Mapping[str, Mapping[str, list[tuple[str, frozenset[str]]]]],
    cohit_lookup: Mapping[str, list[str]],
) -> pd.DataFrame:
    out = genes.copy()
    out["gene_function_description"] = [
        _gene_function_description(row, membership, cohit_lookup)
        for row in out.itertuples(index=False)
    ]
    return out


def _format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}g}"


def _gene_table_with_membership(
    genes: pd.DataFrame,
    membership: dict[str, dict[str, list[tuple[str, frozenset[str]]]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in genes.itertuples(index=False):
        key = row.gene_key
        out_row: dict[str, object] = {
            "Gene": row.Gene,
            "Gene full name": row.gene_full_name,
            "Functional description": row.gene_function_description,
        }
        direction = _gene_direction_from_logfc(row.logFC)
        for source in SOURCE_ORDER:
            out_row[f"{source} pathways"] = _plain_pathway_cell(membership[key][source], direction)
        out_row["logFC"] = row.logFC
        out_row["pvalue"] = row.pvalue
        rows.append(out_row)
    return pd.DataFrame(rows)


def _render_gene_table(
    genes: pd.DataFrame,
    membership: dict[str, dict[str, list[tuple[str, frozenset[str]]]]],
) -> str:
    columns = [
        "Gene",
        "Gene full name",
        "Functional description",
        *[f"{source} pathways" for source in SOURCE_ORDER],
        "logFC",
        "pvalue",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    row_html: list[str] = []
    for row in genes.itertuples(index=False):
        direction = _gene_direction_from_logfc(row.logFC)
        cells = [
            f"<td>{html.escape(str(row.Gene))}</td>",
            f"<td>{html.escape(str(row.gene_full_name))}</td>",
            f'<td class="description-cell">{html.escape(str(row.gene_function_description))}</td>',
        ]
        for source in SOURCE_ORDER:
            cells.append(f'<td class="pathway-cell">{_html_pathway_cell(membership[row.gene_key][source], direction)}</td>')
        cells.append(f'<td class="numeric">{_format_float(row.logFC, digits=5)}</td>')
        cells.append(f'<td class="numeric">{_format_float(row.pvalue, digits=4)}</td>')
        row_html.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-box">'
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        "</div>"
    )


def _render_pathway_summary(pathway_results: pd.DataFrame) -> str:
    significant = pathway_results[pathway_results["significant"]].copy()
    if significant.empty:
        return '<p class="empty">No pathways pass FDR &lt; 0.05 in the ranked, up-gene, or down-gene tests.</p>'
    columns = [
        "source",
        "pathway",
        "significant_tests",
        "ranked_fdr",
        "up_fdr",
        "down_fdr",
        "up_overlap",
        "down_overlap",
    ]
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows: list[str] = []
    for row in significant[columns].itertuples(index=False):
        values = [
            row.source,
            row.pathway,
            row.significant_tests,
            _format_float(row.ranked_fdr, digits=3),
            _format_float(row.up_fdr, digits=3),
            _format_float(row.down_fdr, digits=3),
            row.up_overlap,
            row.down_overlap,
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    return (
        '<div class="table-box pathway-summary-box">'
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _split_gene_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [gene.strip() for gene in str(value).split(";") if gene.strip()]


def _clean_pathway_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"\s+R-HSA-\d+\b", "", text)
    return text.strip()


def _humanize_label(value: object) -> str:
    text = re.sub(r"[_]+", " ", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def _title_derived_states(title: str) -> tuple[str | None, str | None]:
    if re.search(r"\blow\s+vs\s+high\b", title, flags=re.IGNORECASE):
        high = re.sub(r"\blow\s+vs\s+high\b", "high", title, flags=re.IGNORECASE)
        low = re.sub(r"\blow\s+vs\s+high\b", "low", title, flags=re.IGNORECASE)
        return _humanize_label(low), _humanize_label(high)
    if " to " in title:
        low, high = title.split(" to ", 1)
        return _humanize_label(low), _humanize_label(high)
    return None, None


def _is_raw_column_label(value: object) -> bool:
    text = str(value)
    return "_" in text or text.islower()


def _infer_high_marker(*values: object) -> str | None:
    combined = " ".join(str(value) for value in values if value is not None)
    for marker in ["Tcf7", "Prdm1", "Eomes", "Tox"]:
        if re.search(rf"\b{marker}\b", combined, flags=re.IGNORECASE):
            return marker
    return None


def _marker_biology(marker: str | None) -> str:
    if marker == "Tcf7":
        return (
            "Literature frame: Tcf7/TCF1 marks memory-like and progenitor-like CD8 T cells, including the stem-like "
            "precursor exhausted compartment. Pathways that reduce terminal effector instruction can plausibly enrich "
            "Tcf7-high cells, while loss of Tcf7/TCF1, FOXO1, or related progenitor regulators should deplete that bin."
        )
    if marker == "Prdm1":
        return (
            "Literature frame: Prdm1/Blimp1 is an activation-induced effector and terminal-differentiation regulator. "
            "Pathways that deliver strong TCR/cytokine, chromatin-opening, and anabolic-growth signals are expected to "
            "support the Prdm1-high state."
        )
    if marker == "Eomes":
        return (
            "Literature frame: Eomes is tied to CD8 effector, cytotoxic, memory, and exhausted-cell differentiation "
            "depending on context. Eomes-high cells are therefore expected to be sensitive to pathways that control "
            "effector fitness, mitochondrial capacity, and activation-linked transcription."
        )
    if marker == "Tox":
        return (
            "Literature frame: Tox is induced by persistent stimulation and is a core regulator of exhausted-cell programs. "
            "Stress, hypoxia, chronic TCR signaling, and terminal-differentiation pressure can all make a Tox-high bin more likely."
        )
    return (
        "Literature frame: this report interprets KO enrichment as a direction-aware screen result. Positive logFC means "
        "KO guides accumulate in the high-bin state; negative logFC means KO guides are depleted from that high-bin state."
    )


def _infer_screen_context(
    title: str,
    low_label: str | None = None,
    high_label: str | None = None,
) -> ScreenContext:
    title_low, title_high = _title_derived_states(title)
    if low_label is None or _is_raw_column_label(low_label):
        low_state = title_low or _humanize_label(low_label or "low state")
    else:
        low_state = _humanize_label(low_label)
    if high_label is None or _is_raw_column_label(high_label):
        high_state = title_high or _humanize_label(high_label or "high state")
    else:
        high_state = _humanize_label(high_label)
    high_marker = _infer_high_marker(high_state, title)
    return ScreenContext(
        title=title,
        low_label=low_state,
        high_label=high_state,
        high_marker=high_marker,
        high_state=high_state,
        low_state=low_state,
        marker_biology=_marker_biology(high_marker),
    )


def _direction_relevant_pathways(pathway_results: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "up":
        mask = pathway_results["up_significant"] | (
            pathway_results["ranked_significant"] & pathway_results["ranked_direction"].eq("top")
        )
    else:
        mask = pathway_results["down_significant"] | (
            pathway_results["ranked_significant"] & pathway_results["ranked_direction"].eq("bottom")
        )
    return pathway_results[mask].copy()


def _direction_fdr(row: pd.Series, direction: str) -> float:
    values: list[float] = []
    if direction == "up":
        if bool(row.get("up_significant", False)):
            values.append(float(row["up_fdr"]))
        if bool(row.get("ranked_significant", False)) and row.get("ranked_direction") == "top":
            values.append(float(row["ranked_fdr"]))
    else:
        if bool(row.get("down_significant", False)):
            values.append(float(row["down_fdr"]))
        if bool(row.get("ranked_significant", False)) and row.get("ranked_direction") == "bottom":
            values.append(float(row["ranked_fdr"]))
    return min(values) if values else 1.0


def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    upper = text.upper()
    return any(keyword in upper for keyword in keywords)


def _matching_program_rows(pathways: pd.DataFrame, program: Mapping[str, object]) -> pd.DataFrame:
    keywords = list(program["keywords"])
    exclude_keywords = [*COMMON_NOISE_PATHWAY_KEYWORDS, *list(program.get("exclude_keywords", []))]
    keep_rows: list[bool] = []
    for pathway in pathways["pathway"].astype(str):
        upper = pathway.upper()
        keep_rows.append(
            _contains_keyword(upper, keywords)
            and not _contains_keyword(upper, exclude_keywords)
        )
    return pathways[keep_rows].copy()


def _displayable_program_rows(rows: pd.DataFrame, program: Mapping[str, object]) -> pd.DataFrame:
    exclude_keywords = [
        *GENERIC_DISPLAY_PATHWAY_KEYWORDS,
        *COMMON_NOISE_PATHWAY_KEYWORDS,
        *list(program.get("display_exclude_keywords", [])),
    ]
    keep_rows = [
        not _contains_keyword(str(pathway), exclude_keywords)
        for pathway in rows["pathway"].astype(str)
    ]
    return rows[keep_rows].copy()


def _top_direction_genes(genes: pd.DataFrame, n: int = 12) -> list[str]:
    if genes.empty:
        return []
    return genes.sort_values(["pvalue", "Gene"], ascending=[True, True], kind="mergesort")["Gene"].head(n).tolist()


def _program_gene_counts(program_rows: pd.DataFrame, direction: str) -> list[str]:
    column = "up_genes" if direction == "up" else "down_genes"
    counts: dict[str, int] = {}
    for value in program_rows[column].tolist():
        for gene in _split_gene_list(value):
            counts[gene] = counts.get(gene, 0) + 1
    ranked_direction = "top" if direction == "up" else "bottom"
    ranked_rows = program_rows[
        program_rows["ranked_significant"].astype(bool)
        & program_rows["ranked_direction"].eq(ranked_direction)
    ]
    for value in ranked_rows.get("ranked_genes", pd.Series(dtype=object)).tolist():
        for gene in _split_gene_list(value):
            counts[gene] = counts.get(gene, 0) + 0.25
    return [
        gene
        for gene, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]


def _direct_program_gene_count(program_rows: pd.DataFrame, direction: str) -> int:
    column = "up_genes" if direction == "up" else "down_genes"
    genes: set[str] = set()
    for value in program_rows[column].tolist():
        genes.update(_split_gene_list(value))
    return len(genes)


def _evidence_label(rows: pd.DataFrame, direction: str) -> str:
    labels: list[str] = []
    if direction == "up":
        up_rows = rows[rows["up_significant"].astype(bool)]
        ranked_rows = rows[
            rows["ranked_significant"].astype(bool)
            & rows["ranked_direction"].eq("top")
        ]
        if not up_rows.empty:
            labels.append(f"positive-gene Fisher FDR {_format_float(up_rows['up_fdr'].min(), digits=3)}")
        if not ranked_rows.empty:
            labels.append(f"ranked-list top shift FDR {_format_float(ranked_rows['ranked_fdr'].min(), digits=3)}")
    else:
        down_rows = rows[rows["down_significant"].astype(bool)]
        ranked_rows = rows[
            rows["ranked_significant"].astype(bool)
            & rows["ranked_direction"].eq("bottom")
        ]
        if not down_rows.empty:
            labels.append(f"negative-gene Fisher FDR {_format_float(down_rows['down_fdr'].min(), digits=3)}")
        if not ranked_rows.empty:
            labels.append(f"ranked-list bottom shift FDR {_format_float(ranked_rows['ranked_fdr'].min(), digits=3)}")
    return "; ".join(labels) if labels else "directional enrichment in the pathway table"


def _program_summaries(pathway_results: pd.DataFrame, direction: str) -> list[dict[str, object]]:
    relevant = _direction_relevant_pathways(pathway_results, direction)
    summaries: list[dict[str, object]] = []
    for priority, program in enumerate(INTERPRETABLE_PROGRAMS):
        rows = _matching_program_rows(relevant, program)
        if rows.empty:
            continue
        rows = rows.copy()
        rows["_direction_fdr"] = rows.apply(lambda row: _direction_fdr(row, direction), axis=1)
        display_rows = _displayable_program_rows(rows, program)
        if display_rows.empty:
            continue
        display_rows = display_rows.sort_values(["_direction_fdr", "source", "pathway"], kind="mergesort")
        direct_gene_count = _direct_program_gene_count(rows, direction)
        if direct_gene_count == 0 and len(display_rows) < 2:
            continue
        best_fdr = float(rows["_direction_fdr"].min())
        evidence_tier = 0 if direct_gene_count > 0 else 1
        summaries.append(
            {
                "id": program["id"],
                "name": program["name"],
                "short_label": program["short_label"],
                "literature": program["literature"],
                "best_fdr": best_fdr,
                "evidence": _evidence_label(rows, direction),
                "evidence_tier": evidence_tier,
                "direct_gene_count": direct_gene_count,
                "priority": priority,
                "pathways": [
                    f"{row.source}: {_clean_pathway_name(row.pathway)}"
                    for row in display_rows.head(4).itertuples(index=False)
                ],
                "genes": _program_gene_counts(rows, direction),
                "n_pathways": int(len(display_rows)),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (int(item["evidence_tier"]), float(item["best_fdr"]), int(item["priority"])),
    )[:4]


def _direction_screen_sentence(direction: str, context: ScreenContext) -> str:
    if direction == "up":
        return (
            f"Positive logFC means KO guides are enriched in {context.high_state} relative to "
            f"{context.low_state}."
        )
    return (
        f"Negative logFC means KO guides are depleted from {context.high_state} and are relatively enriched "
        f"toward {context.low_state}."
    )


def _clean_report_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _population_label(direction: str, context: ScreenContext) -> str:
    state = "High" if direction == "up" else "Low"
    if context.high_marker:
        return f"{context.high_marker} {state} Population"
    return f"{state} Population"


def _program_heading(summary: Mapping[str, object], direction: str, context: ScreenContext) -> str:
    return f"KO of {summary['short_label']} enriches {_population_label(direction, context)}"


def _gene_set_contains(genes: Sequence[str], *candidates: str) -> bool:
    keys = {gene_key(gene) for gene in genes}
    return any(gene_key(candidate) in keys for candidate in candidates)


def _program_mechanistic_read(summary: Mapping[str, object], direction: str, context: ScreenContext) -> str:
    program_id = str(summary["id"])
    marker = context.high_marker
    genes = list(summary.get("genes", []))
    sign_sentence = ""

    if marker == "Tcf7":
        if program_id == "fate_tf" and direction == "up":
            extra = (
                "This is a strong mechanistic fit: removing effector-fate regulators can release cells from terminal "
                "differentiation pressure and leave more cells in a Tcf7/TCF1-high progenitor-like state."
            )
            if _gene_set_contains(genes, "Prdm1"):
                extra += " The presence of Prdm1 is especially interpretable because Blimp1 is a canonical antagonist of memory/progenitor-like programs."
            return f"{sign_sentence} {extra}"
        if program_id == "fate_tf" and direction == "down":
            extra = (
                "This is the expected dependency direction when the hit genes include the TCF1/FOXO/progenitor-state machinery: "
                "knocking out those regulators removes the biology needed to stay Tcf7 high."
            )
            if _gene_set_contains(genes, "Tcf7", "Foxo1", "Myb"):
                extra += " Tcf7, Foxo1, or Myb in this direction should be treated as positive-control-like evidence."
            return f"{sign_sentence} {extra}"
        if program_id == "t_cell_cytokine" and direction == "up":
            return (
                f"{sign_sentence} This makes sense if strong cytokine/TCR signaling is pushing cells away from the "
                "TCF1-positive progenitor-like state and toward effector differentiation. KO of IFN/JAK/STAT, IL-2, or TCR-proximal "
                "nodes can dampen that instruction and enrich the Tcf7-high bin."
            )
        if program_id == "t_cell_cytokine" and direction == "down":
            return (
                f"{sign_sentence} This suggests that this particular antigen-receptor or cytokine module is not only an effector "
                "driver but also supports survival or maintenance of the Tcf7-high compartment under the stimulation condition. "
                "This is most interpretable when the evidence is a focused gene set rather than broad immune-system terms."
            )
        if program_id == "pi3k_mtor_metabolism":
            return (
                f"{sign_sentence} PI3K/AKT/mTOR is a known anabolic effector-differentiation axis. Lowering this axis can preserve "
                "FOXO/TCF-like memory programs, while loss of brakes such as PTEN can have the opposite effect by increasing AKT tone."
            )
        if program_id == "proteostasis":
            return (
                f"{sign_sentence} Tcf7-high cells still need controlled protein turnover to keep receptor, signaling-adaptor, and "
                "transcription-factor levels in the right range. A proteasome/ubiquitin dependency can therefore move cells out of "
                "the TCF1-positive bin even though it is not a lineage pathway by itself."
            )

    if marker == "Prdm1":
        if direction == "down" and program_id in {"t_cell_cytokine", "chromatin_remodeling", "fate_tf"}:
            extra = (
                "This is the expected direction for a Blimp1-high reporter: strong activation signals and chromatin opening are needed "
                "to induce and maintain the Prdm1-high effector state."
            )
            if _gene_set_contains(genes, "Prdm1"):
                extra += " Prdm1 itself appearing on the negative side is a direct positive-control-like result."
            return f"{sign_sentence} {extra}"
        if program_id == "pi3k_mtor_metabolism":
            return (
                f"{sign_sentence} Prdm1-high cells are usually more effector-like and anabolic. A PI3K/mTOR-metabolic dependency fits "
                "the idea that Blimp1-high differentiation requires growth-factor and nutrient-sensing support."
            )
        if program_id in {"translation_ribosome", "cell_cycle_dna"}:
            return (
                f"{sign_sentence} This reads as a fitness-coupled Prdm1 signal. Blimp1-high effector cells often sit in a proliferative, "
                "biosynthetically active state, so perturbing replication or translation can change how well that compartment is represented."
            )
        if program_id == "fate_tf":
            return (
                f"{sign_sentence} This is a direct fate-control signal. The genes in this axis likely change the balance between "
                "Blimp1-high effector differentiation and alternative TCF/FOXO/Notch/RUNX-regulated states."
            )
        if program_id == "proteostasis":
            return (
                f"{sign_sentence} Proteasome and neddylation machinery can tune Blimp1-high differentiation by changing the stability "
                "of signaling adaptors, cell-cycle regulators, and transcription factors. The direction tells whether that turnover "
                "supports or restrains the Prdm1-high compartment."
            )

    if marker == "Eomes":
        if program_id in {"mitochondrial_fitness", "translation_ribosome"}:
            return (
                f"{sign_sentence} This fits an effector-fitness model: Eomes-high cells need mitochondrial and translational capacity "
                "to sustain cytotoxic or chronically stimulated programs. KO of those pathways can therefore deplete the Eomes-high bin."
            )
        if program_id in {"t_cell_cytokine", "fate_tf"}:
            return (
                f"{sign_sentence} Eomes is downstream of activation and fate-setting circuitry in CD8 T cells. Directional enrichment here "
                "suggests the pathway is changing whether cells can enter or maintain an Eomes-high effector-like state."
            )
        if program_id in {"cell_cycle_dna", "proteostasis", "chromatin_remodeling"}:
            return (
                f"{sign_sentence} This is consistent with Eomes-high cells depending on activation-linked remodeling, protein turnover, "
                "and proliferative fitness. These pathways can determine whether an effector-like Eomes program is stable enough to be recovered."
            )

    if marker == "Tox":
        if program_id in {"cell_cycle_dna", "pi3k_mtor_metabolism", "mitochondrial_fitness"}:
            return (
                f"{sign_sentence} This is plausible because TOX-high states are linked to chronic stimulation, stress adaptation, and "
                "terminal/exhausted differentiation. Perturbing replication, hypoxia/metabolic, or mitochondrial programs can create the "
                "stress and altered fitness context that makes the Tox-high bin expand or collapse."
            )
        if program_id == "t_cell_cytokine":
            return (
                f"{sign_sentence} Chronic antigen and cytokine signaling are upstream of TOX induction. A directional signal in this axis "
                "therefore directly suggests the KO changes the strength or persistence of exhaustion-inducing stimulation."
            )
        if program_id in {"translation_ribosome", "proteostasis", "chromatin_remodeling", "fate_tf"}:
            return (
                f"{sign_sentence} TOX-high cells are shaped by chronic-stimulation stress and transcriptional remodeling. Perturbing this "
                "axis can either create stress that enriches the TOX-high bin or remove machinery needed to sustain that exhausted-like state."
            )

    if program_id in {"cell_cycle_dna", "translation_ribosome"}:
        return (
            f"{sign_sentence} This is best interpreted as state-fitness coupling rather than a clean lineage switch: the high-bin cells "
            "may be more or less dependent on proliferation, RNA processing, and protein synthesis than the low-bin cells."
        )
    if program_id == "proteostasis":
        return (
            f"{sign_sentence} Protein turnover can change the abundance of receptors, signaling proteins, and transcription factors, so "
            "a ubiquitin/proteasome signal is interpretable as a mechanism that changes how stable the high-bin cell state is."
        )
    if program_id == "chromatin_remodeling":
        return (
            f"{sign_sentence} Chromatin-remodeling KOs can block or redirect the enhancer accessibility needed for activation-induced "
            "state changes, making this a plausible direct mechanism for changing the reporter-high compartment."
        )
    if program_id == "fate_tf":
        return (
            f"{sign_sentence} These genes are close to the transcriptional circuitry that defines T cell state, so the pathway direction "
            "is interpretable as a shift in fate-regulator balance rather than generic growth noise."
        )
    return f"{sign_sentence} The sign is interpretable because this pathway axis has a known role in CD8 T cell activation, fate, or fitness."


def _program_counter_read(summary: Mapping[str, object], direction: str, context: ScreenContext) -> str:
    program_id = str(summary["id"])
    marker = context.high_marker
    population = _population_label(direction, context)

    if marker == "Tox" and program_id == "cell_cycle_dna":
        return (
            "TOX-high states are often more differentiated or repeatedly stimulated, and those cells can be tied to proliferation history. "
            "A cell-cycle signal may therefore reflect division or viability bias rather than direct regulation of the TOX program."
        )
    if marker == "Tox" and program_id == "pi3k_mtor_metabolism":
        return (
            "mTOR and metabolic signaling can support both effector differentiation and survival, so the same pathway can push cells toward "
            "or away from a TOX-high state depending on timing and stress."
        )
    if marker == "Tox" and program_id == "mitochondrial_fitness":
        return (
            "Mitochondrial hits can be generic survival effects. They may change which cells survive the sort rather than specifically changing exhaustion biology."
        )
    if marker == "Tox" and program_id == "translation_ribosome":
        return (
            "Translation and ribosome hits often report growth-rate or stress-load differences, so they are weaker evidence for direct TOX-state instruction."
        )
    if marker == "Tcf7" and program_id in {"cell_cycle_dna", "translation_ribosome", "mitochondrial_fitness"}:
        return (
            "TCF7-high biology is a fate/state program, whereas this pathway is mostly a fitness axis. The signal could be indirect through growth or survival."
        )
    if marker == "Prdm1" and program_id in {"cell_cycle_dna", "translation_ribosome"}:
        return (
            "Blimp1-high cells are often activated and biosynthetic, so this may be a general effector fitness dependency rather than a direct Prdm1 regulator."
        )
    if marker == "Eomes" and program_id in {"cell_cycle_dna", "translation_ribosome", "mitochondrial_fitness"}:
        return (
            "Eomes-high cells can differ in fitness and activation history, so the signal may reflect recovery of fitter cells rather than direct Eomes control."
        )
    if program_id == "t_cell_cytokine":
        return (
            "TCR and cytokine pathways change activation, survival, and differentiation at the same time, so this mechanism can be directionally real but not uniquely state-specific."
        )
    if program_id == "fate_tf":
        return (
            "Fate regulators are highly context-dependent. A KO can move cells between states indirectly by changing activation strength, survival, or timing."
        )
    if program_id == "chromatin_remodeling":
        return (
            "Chromatin regulators have broad effects on viability and transcriptional capacity, so the screen may not identify one specific downstream state program."
        )
    if program_id == "proteostasis":
        return (
            "Proteostasis hits can reflect stress tolerance and protein-load handling, not necessarily a direct mechanism for the reporter state."
        )
    if program_id == "tgfb_smad":
        return (
            "TGF-beta effects are context-dependent across activation, suppression, and persistence, so the same pathway can have different meanings across culture conditions."
        )
    return (
        f"This pathway could still be a broad fitness effect in the {population}, so it should be treated as a hypothesis unless supported by focused genes."
    )


def _render_program_cards(
    program_summaries: list[dict[str, object]],
    direction: str,
    context: ScreenContext,
    fallback: str,
) -> str:
    if not program_summaries:
        return f'<p class="empty">{html.escape(fallback)}</p>'
    cards: list[str] = []
    for summary in program_summaries:
        pathways = "".join(
            f"<li>{html.escape(pathway)}</li>"
            for pathway in summary["pathways"]
        )
        gene_text = ", ".join(summary["genes"]) if summary["genes"] else "No single overlapping gene dominates the significant set."
        why_text = _clean_report_sentence(_program_mechanistic_read(summary, direction, context))
        why_not_text = _clean_report_sentence(_program_counter_read(summary, direction, context))
        cards.append(
            '<div class="interpretation-card">'
            f'<div class="interpretation-title">{html.escape(_program_heading(summary, direction, context))}</div>'
            f'<p><strong>Why this makes sense:</strong> {html.escape(why_text)}</p>'
            f'<p><strong>Why this does not make sense:</strong> {html.escape(why_not_text)}</p>'
            f'<p><strong>Key KO genes in this report:</strong> {html.escape(gene_text)}</p>'
            '<p><strong>Interpretable pathway evidence:</strong></p>'
            f'<ul>{pathways}</ul>'
            '</div>'
        )
    return "".join(cards)


def _canonical_gene_notes(up: pd.DataFrame, down: pd.DataFrame, context: ScreenContext) -> str:
    up_genes = {gene_key(gene): gene for gene in up["Gene"].astype(str).tolist()} if not up.empty else {}
    down_genes = {gene_key(gene): gene for gene in down["Gene"].astype(str).tolist()} if not down.empty else {}
    notes: list[str] = []
    marker = context.high_marker
    if marker and gene_key(marker) in down_genes:
        notes.append(
            f"{down_genes[gene_key(marker)]}: KO depletes the {context.high_state} bin. This is a direct positive-control-like result for a reporter-gated screen."
        )
    if marker == "Tcf7":
        if "PRDM1" in up_genes:
            notes.append(
                f"{up_genes['PRDM1']}: Prdm1/Blimp1 KO enriches {context.high_state}, matching the known antagonism between Blimp1-driven effector differentiation and TCF1/TCF7 progenitor-like state."
            )
        if "FOXO1" in down_genes:
            notes.append(
                f"{down_genes['FOXO1']}: Foxo1 KO depletes {context.high_state}, consistent with FOXO1 supporting memory/progenitor-like Tcf7 programs."
            )
        if "PTEN" in down_genes:
            notes.append(
                f"{down_genes['PTEN']}: Pten KO depletes {context.high_state}; this fits the idea that unchecked PI3K/AKT signaling can push cells away from FOXO/TCF-like memory biology."
            )
    if marker == "Prdm1":
        activation_hits = [down_genes[key] for key in ["CD3D", "CD3E", "ZAP70", "JAK2", "STAT4", "IL12RB1"] if key in down_genes]
        if activation_hits:
            notes.append(
                f"{', '.join(activation_hits)}: KO of activation/cytokine signaling genes depletes {context.high_state}, which fits Prdm1/Blimp1 induction downstream of strong stimulation."
            )
    if marker == "Tox":
        if "HIF1A" in up_genes or "HIF1AN" in up_genes:
            gene = up_genes.get("HIF1A", up_genes.get("HIF1AN"))
            notes.append(
                f"{gene}: hypoxia/HIF-axis perturbation points toward the known connection between hypoxic stress and terminal exhausted-cell programs."
            )
    if not notes:
        return (
            '<p class="empty">No canonical single-gene sanity-check stood out after the curated filter. Use the pathway cards below and the full gene tables above.</p>'
        )
    return "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes[:5]) + "</ul>"


def _render_gene_watchlist(up: pd.DataFrame, down: pd.DataFrame) -> str:
    up_genes = _top_direction_genes(up)
    down_genes = _top_direction_genes(down)
    up_text = ", ".join(up_genes) if up_genes else "none"
    down_text = ", ".join(down_genes) if down_genes else "none"
    return (
        "<ul>"
        f"<li><strong>Positive-logFC genes to inspect first:</strong> {html.escape(up_text)}.</li>"
        f"<li><strong>Negative-logFC genes to inspect first:</strong> {html.escape(down_text)}.</li>"
        "</ul>"
    )


def _render_codex_interpretation_report(
    title: str,
    ranked: pd.DataFrame,
    up: pd.DataFrame,
    down: pd.DataFrame,
    pathway_results: pd.DataFrame,
    low_label: str | None = None,
    high_label: str | None = None,
) -> str:
    context = _infer_screen_context(title, low_label=low_label, high_label=high_label)
    significant_pathways = pathway_results[pathway_results["significant"]].copy()
    up_programs = _program_summaries(pathway_results, "up")
    down_programs = _program_summaries(pathway_results, "down")
    top_up_pathways = _direction_relevant_pathways(pathway_results, "up").copy()
    top_down_pathways = _direction_relevant_pathways(pathway_results, "down").copy()
    up_count = len(up)
    down_count = len(down)
    up_population_text = html.escape(_population_label("up", context))
    down_population_text = html.escape(_population_label("down", context))
    all_programs = [*up_programs, *down_programs]
    literature_program_ids = {"t_cell_cytokine", "pi3k_mtor_metabolism", "fate_tf"}
    new_program_ids = {"cell_cycle_dna", "translation_ribosome", "mitochondrial_fitness", "proteostasis", "chromatin_remodeling"}
    agree_programs = [program for program in all_programs if str(program["id"]) in literature_program_ids]
    new_programs = [program for program in all_programs if str(program["id"]) in new_program_ids]
    def unique_program_labels(programs: Sequence[Mapping[str, object]], fallback: str) -> str:
        labels: list[str] = []
        for program in programs:
            label = str(program["short_label"])
            if label not in labels:
                labels.append(label)
        return ", ".join(labels[:4]) if labels else fallback

    up_program_text = unique_program_labels(up_programs, "no curated positive-logFC pathway axis")
    down_program_text = unique_program_labels(down_programs, "no curated negative-logFC pathway axis")
    agree_text = (
        unique_program_labels(agree_programs, "no curated canonical literature axis passed the interpretation filter")
    )
    new_text = (
        unique_program_labels(new_programs, "no extra fitness or remodeling axis stood out beyond the canonical literature frame")
    )
    confidence_rows: list[str] = []
    for label, direction, programs in [
        ("More high-bin cells", "up", up_programs),
        ("Fewer high-bin cells", "down", down_programs),
    ]:
        for program in programs:
            direct_gene_count = int(program.get("direct_gene_count", 0))
            best_fdr = float(program["best_fdr"])
            if direct_gene_count > 0 and best_fdr <= 0.001:
                confidence = "Strong screen support"
            elif direct_gene_count > 0:
                confidence = "Moderate screen support"
            else:
                confidence = "Speculative: ranked-list/pathway shift without many direct significant genes"
            confidence_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(program['short_label']))}</td>"
                f"<td>{html.escape(confidence)}</td>"
                f"<td>{_format_float(best_fdr, digits=3)}</td>"
                f"<td>{html.escape(', '.join(program['genes'][:6]))}</td>"
                "</tr>"
            )
    confidence_table = ""
    if confidence_rows:
        confidence_table = (
            '<div class="table-box interpretation-table"><table>'
            "<thead><tr><th>Effect</th><th>Mechanism</th><th>Confidence</th><th>Best FDR</th><th>Example genes</th></tr></thead>"
            f"<tbody>{''.join(confidence_rows)}</tbody></table></div>"
        )
    return f"""
    <section class="panel interpretation-report">
      <h2>Screen Interpretation Summary</h2>
      <p>
        This summary interprets <strong>{html.escape(title)}</strong> from the direction-aware KO effects and the significant pathway tests above.
        It separates direct screen support from broader pathway-level hypotheses.
      </p>
      <p>
        The full ranked test used <code>{len(ranked):,}</code> genes; the displayed gene tables focus on
        <code>{up_count:,}</code> KOs enriching <strong>{up_population_text}</strong> and
        <code>{down_count:,}</code> KOs enriching <strong>{down_population_text}</strong>.
        There are <code>{len(significant_pathways):,}</code> significant pathways.
      </p>
      <p><strong>Biological frame:</strong> {html.escape(context.marker_biology)}</p>
      <p>
        <strong>What enriches {up_population_text}:</strong> {html.escape(up_program_text)}.
        <strong>What enriches {down_population_text}:</strong> {html.escape(down_program_text)}.
      </p>
      <p>
        <strong>Where this agrees with the literature:</strong> {html.escape(agree_text)} fit established CD8 T cell activation,
        fate, or nutrient-sensing biology for this reporter context.
      </p>
      <p>
        <strong>Potential new biology from this screen:</strong> {html.escape(new_text)} appear as perturbable axes in this assay.
        These are strongest as screen-derived dependencies or state-fitness mechanisms; they need follow-up before being treated as direct
        lineage-instruction pathways.
      </p>
      <p>
        <strong>Speculative layer:</strong> ranked-list-only pathways, broad disease/infection labels, and generic transcription or metabolism
        terms are useful for triage but should not be over-interpreted unless the same axis contains direct significant genes in the expected direction.
      </p>
      <h3>Canonical Gene Sanity Checks</h3>
      {_canonical_gene_notes(up, down, context)}
      <h3>Top Individual KO Genes</h3>
      {_render_gene_watchlist(up, down)}
      <h3>Confidence Triage</h3>
      {confidence_table if confidence_table else '<p class="empty">No curated mechanism passed the interpretation filter.</p>'}
      <h3>Mechanisms Enriching {up_population_text}</h3>
      {_render_program_cards(up_programs, "up", context, f"No curated mechanism enriching {_population_label('up', context)} passed the interpretation filter.")}
      <h3>Mechanisms Enriching {down_population_text}</h3>
      {_render_program_cards(down_programs, "down", context, f"No curated mechanism enriching {_population_label('down', context)} passed the interpretation filter.")}
      <h3>Practical Takeaway</h3>
      <p>
        Treat this as a mechanistic triage layer over the statistical tests. The strongest candidates are genes that have the correct KO
        direction, sit in one of the curated pathways above, and make sense given the reporter biology. A pathway that is significant but
        appears only as a broad disease, viral infection, generic transcription, or generic metabolism term should not be treated as a
        mechanistic explanation unless its specific genes support one of the curated axes.
      </p>
      <p class="theme-meta">
        Directional pathway counts: <code>{len(top_up_pathways):,}</code> positive/top pathways and
        <code>{len(top_down_pathways):,}</code> negative/bottom pathways.
      </p>
    </section>
"""


def _relative_href(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def build_gene_list_interpretation_report(
    ranked_genes: pd.DataFrame,
    significant_up_genes: pd.DataFrame,
    significant_down_genes: pd.DataFrame,
    output_html: str | os.PathLike[str],
    title: str,
    gene_col: str = "Gene",
    logfc_col: str = "logFC",
    pvalue_col: str = "pvalue",
    low_label: str | None = None,
    high_label: str | None = None,
    pathway_catalogs: Sequence[PathwayTerm] | None = None,
    catalog_paths: Mapping[str, str | os.PathLike[str] | tuple[str, str | os.PathLike[str]]] | None = None,
    gene_full_name_lookup: Mapping[str, str] | None = None,
    gene_info_path: str | os.PathLike[str] | None = None,
    fdr_threshold: float = PATHWAY_FDR_THRESHOLD,
    min_directional_genes: int = MIN_DIRECTIONAL_PATHWAY_GENES,
) -> dict[str, Path]:
    ranked = _prepare_gene_table(ranked_genes, gene_col, logfc_col, pvalue_col, include_rank=True)
    up = _prepare_gene_table(significant_up_genes, gene_col, logfc_col, pvalue_col, include_rank=False)
    down = _prepare_gene_table(significant_down_genes, gene_col, logfc_col, pvalue_col, include_rank=False)
    gene_info_lookup: dict[str, dict[str, object]] = {}
    if gene_info_path is not None:
        gene_info_lookup.update(load_gene_info_lookup(gene_info_path))
    if gene_full_name_lookup is not None:
        for gene, name in gene_full_name_lookup.items():
            key = gene_key(gene)
            info = dict(gene_info_lookup.get(key, {}))
            info["full_name"] = name
            info.setdefault("synonyms", [])
            info.setdefault("other_designations", [])
            gene_info_lookup[key] = info
    ranked = _add_gene_annotations(ranked, gene_info_lookup)
    up = _add_gene_annotations(up, gene_info_lookup)
    down = _add_gene_annotations(down, gene_info_lookup)
    catalogs = list(pathway_catalogs) if pathway_catalogs is not None else load_pathway_catalogs(catalog_paths)

    pathway_results = test_pathways(
        ranked_genes=ranked,
        significant_up_genes=up,
        significant_down_genes=down,
        pathway_catalogs=catalogs,
        fdr_threshold=fdr_threshold,
        min_directional_genes=min_directional_genes,
    )
    significant_pathway_results = pathway_results[pathway_results["significant"]].copy()
    term_direction_lookup = _significant_term_direction_lookup(pathway_results)
    membership = build_gene_membership(ranked, catalogs, term_direction_lookup)
    significant_gene_keys = set(up["gene_key"]) | set(down["gene_key"])
    cohit_lookup = _same_direction_cohit_lookup(up, down, membership)
    ranked = _add_gene_function_descriptions(ranked, membership, cohit_lookup)
    up = _add_gene_function_descriptions(up, membership, cohit_lookup)
    down = _add_gene_function_descriptions(down, membership, cohit_lookup)
    displayed_ranked = ranked[ranked["gene_key"].isin(significant_gene_keys)].copy()

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    pathway_tsv = stem.with_name(f"{stem.name}_pathway_tests.tsv")
    ranked_tsv = stem.with_name(f"{stem.name}_ranked_genes.tsv")
    up_tsv = stem.with_name(f"{stem.name}_significant_up_genes.tsv")
    down_tsv = stem.with_name(f"{stem.name}_significant_down_genes.tsv")

    significant_pathway_results.to_csv(pathway_tsv, sep="\t", index=False)
    _gene_table_with_membership(displayed_ranked, membership).to_csv(ranked_tsv, sep="\t", index=False)
    _gene_table_with_membership(up, membership).to_csv(up_tsv, sep="\t", index=False)
    _gene_table_with_membership(down, membership).to_csv(down_tsv, sep="\t", index=False)

    links_html = " | ".join(
        f'<a href="{html.escape(_relative_href(path, output_path.parent), quote=True)}">{label}</a>'
        for label, path in [
            ("significant pathway tests TSV", pathway_tsv),
            ("significant ranked genes TSV", ranked_tsv),
            ("significant up genes TSV", up_tsv),
            ("significant down genes TSV", down_tsv),
        ]
    )
    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} Gene List Interpretation</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #101828; background: #f6f7fb; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 600; }}
    h2 {{ margin: 0 0 8px; font-size: 21px; font-weight: 600; }}
    h3 {{ margin: 16px 0 8px; font-size: 16px; font-weight: 600; }}
    p {{ color: #475467; font-size: 14px; line-height: 1.5; margin: 0; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; color: #344054; font-size: 14px; line-height: 1.45; }}
    li {{ margin: 4px 0; }}
    a {{ color: #175cd3; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .panel {{ background: #fff; border: 1px solid #e4e7ec; border-radius: 8px; padding: 14px; margin: 0 0 18px; }}
    .links {{ margin-top: 10px; font-size: 14px; }}
    .table-box {{ border: 1px solid #eaecf0; border-radius: 8px; overflow: auto; max-height: 780px; margin-top: 12px; }}
    .pathway-summary-box {{ max-height: 460px; }}
    .interpretation-table {{ max-height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
    th, td {{ border: 1px solid #eaecf0; padding: 7px 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; z-index: 1; background: #f9fafb; text-align: left; }}
    td.numeric {{ text-align: right; white-space: nowrap; }}
    td.pathway-cell {{ min-width: 240px; white-space: normal; line-height: 1.35; }}
    td.description-cell {{ min-width: 340px; max-width: 520px; white-space: pre-line; line-height: 1.4; }}
    .significant-pathway {{ color: #c1121f; font-weight: 700; }}
    .empty {{ padding: 10px 12px; border: 1px dashed #d0d5dd; border-radius: 8px; background: #f9fafb; }}
    .interpretation-report {{ border-color: #b2ddff; }}
    .interpretation-card {{ border: 1px solid #d0d5dd; border-radius: 8px; padding: 12px; margin-top: 10px; background: #fcfcfd; }}
    .interpretation-title {{ font-weight: 700; color: #101828; margin-bottom: 4px; }}
    .theme-meta {{ color: #667085; font-size: 13px; margin-top: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>{html.escape(title)} Gene List Interpretation</h1>
      <p>
        Pathways are grouped by source. Pathway names shown in gene tables are restricted to pathways
        passing FDR &lt; {fdr_threshold:g} in at least one of the ranked-list Wilcoxon rank-shift test,
        up-gene Fisher test, or down-gene Fisher test. The ranked-gene table below is restricted to genes with
        BH FDR &lt; 0.05 in the screen comparison. Pathway tests run with <code>nproc</code> workers.
      </p>
      <div class="links">{links_html}</div>
    </section>
    <section class="panel">
      <h2>Significant Pathways</h2>
      {_render_pathway_summary(pathway_results)}
    </section>
    <section class="panel">
      <h2>Significant Ranked Genes ({len(displayed_ranked):,})</h2>
      {_render_gene_table(displayed_ranked, membership) if len(displayed_ranked) else '<p class="empty">No significant ranked genes.</p>'}
    </section>
    <section class="panel">
      <h2>Significant Up Genes ({len(up):,})</h2>
      {_render_gene_table(up, membership) if len(up) else '<p class="empty">No significant up genes.</p>'}
    </section>
    <section class="panel">
      <h2>Significant Down Genes ({len(down):,})</h2>
      {_render_gene_table(down, membership) if len(down) else '<p class="empty">No significant down genes.</p>'}
    </section>
    {_render_codex_interpretation_report(title, ranked, up, down, pathway_results, low_label=low_label, high_label=high_label)}
  </div>
</body>
</html>
"""
    output_path.write_text(report_html, encoding="utf-8")
    return {
        "html": output_path,
        "pathway_tests": pathway_tsv,
        "ranked_genes": ranked_tsv,
        "significant_up_genes": up_tsv,
        "significant_down_genes": down_tsv,
    }
