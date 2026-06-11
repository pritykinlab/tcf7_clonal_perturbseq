#!/usr/bin/env python
"""
Export per-perturbation clone count tables (Num Progenitor / Num Effector)
from the processed AnnData H5AD files.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Sequence

import h5py
import pandas as pd


GENE10_DEFAULT_ADATA = "./data/gene10/preprocessing/final.h5ad"
TRIM28_DEFAULT_ADATA = "./data/trim28_setdb1/preprocessing/final.h5ad"
EXCLUDED_CELL_TYPES = {"progenitor dln"}


def _decode_categories(categories: Iterable) -> List[str]:
    decoded = []
    for c in categories:
        if isinstance(c, (bytes, bytearray)):
            decoded.append(c.decode("utf-8"))
        else:
            decoded.append(str(c))
    return decoded


def _read_obs_cols(h5ad_path: str, columns: Sequence[str]) -> pd.DataFrame:
    data = {}
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        for col in columns:
            if col not in obs:
                raise KeyError(f"Column '{col}' not found in obs for {h5ad_path}")
            obj = obs[col]
            if isinstance(obj, h5py.Group):
                categories = _decode_categories(obj["categories"][...])
                codes = obj["codes"][...]
                values = [categories[c] if c >= 0 else "" for c in codes]
                data[col] = pd.Series(values)
            else:
                arr = obj[...]
                if arr.dtype.kind in {"S", "O"}:
                    arr = _decode_categories(arr)
                data[col] = pd.Series(arr)
    return pd.DataFrame(data)


def _curate_trim28_setdb1(raw_value: str) -> str:
    if raw_value is None:
        return ""
    raw_value = str(raw_value)
    if raw_value == "" or raw_value == "nan":
        return ""
    tokens = [t for t in raw_value.split("__") if t]
    token_set = set(tokens)
    if not token_set:
        return ""

    has_trim28 = "Trim28" in token_set
    has_setdb1 = "Setdb1" in token_set
    has_nt = "NT" in token_set

    if has_trim28 and has_setdb1:
        return "Setdb1+Trim28"
    if has_trim28 and not has_setdb1:
        return "Trim28"
    if has_setdb1 and not has_trim28:
        return "Setdb1"
    if has_nt:
        return "NT"
    return ""


def _prepare_gene10(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["num_constructs"] == 1].copy()
    df = df[df["perturbation"].astype(str) != ""]
    df = df[~df["perturbation"].astype(str).str.contains("__")]
    return df


def _prepare_trim28_setdb1(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["num_constructs"] == 1].copy()
    df["perturbation"] = df["perturbation"].map(_curate_trim28_setdb1)
    df = df[df["perturbation"].isin(["NT", "Trim28", "Setdb1"])]
    return df


def _drop_forbidden_cell_types(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["cell_type"].astype(str).str.lower().isin(EXCLUDED_CELL_TYPES)
    return df[mask].copy()


def _get_progenitor_types() -> List[str]:
    return ["Progenitor", "Progenitor Cycling"]


def _write_clone_counts(df: pd.DataFrame, output_dir: str, perturbations: Sequence[str], progenitor_types: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for perturbation in perturbations:
        subset = df[df["perturbation"] == perturbation].copy()
        if subset.empty:
            raise ValueError(f"No cells found for perturbation '{perturbation}'")
        subset["is_progenitor"] = subset["cell_type"].isin(progenitor_types)
        # Preserve first-observed clone order to match legacy CSV generation.
        grouped = subset.groupby("clone", sort=False)["is_progenitor"].agg(["sum", "count"]).reset_index()
        grouped.rename(columns={"sum": "Num Progenitor", "count": "Total"}, inplace=True)
        grouped["Num Effector"] = grouped["Total"] - grouped["Num Progenitor"]
        grouped = grouped.reset_index(drop=True)
        grouped.insert(0, "Clone", range(len(grouped)))
        output_path = os.path.join(output_dir, f"{perturbation}.csv")
        grouped[["Clone", "Num Progenitor", "Num Effector"]].to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clone count tables from AnnData.")
    parser.add_argument("--dataset", required=True, choices=["gene10", "trim28_setdb1"])
    parser.add_argument("--adata", default=None, help="Path to H5AD file (defaults to project data).")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-perturbation CSVs.")
    parser.add_argument(
        "--perturbations",
        default="",
        help="Comma-separated perturbation names to export (default: all allowed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "gene10":
        adata_path = args.adata or GENE10_DEFAULT_ADATA
    else:
        adata_path = args.adata or TRIM28_DEFAULT_ADATA

    obs = _read_obs_cols(adata_path, ["clone", "perturbation", "num_constructs", "cell_type"])
    obs = _drop_forbidden_cell_types(obs)
    if args.dataset == "gene10":
        obs = _prepare_gene10(obs)
    else:
        obs = _prepare_trim28_setdb1(obs)

    if args.perturbations:
        perturbations = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    else:
        perturbations = sorted(obs["perturbation"].unique())

    progenitor_types = _get_progenitor_types()
    _write_clone_counts(obs, args.output_dir, perturbations, progenitor_types)


if __name__ == "__main__":
    main()
