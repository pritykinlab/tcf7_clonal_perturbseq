#!/usr/bin/env python
"""Prepare sample and transcript-to-gene tables for bulk RNA-seq normalization."""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd


ATTR_RE = re.compile(r'([A-Za-z0-9_]+) "([^"]*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tx2gene-output", type=Path, required=True)
    parser.add_argument("--gene-metadata-output", type=Path, required=True)
    parser.add_argument("--sample-table-output", type=Path, required=True)
    return parser.parse_args()


def read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, dtype=str)
    metadata.columns = [str(col).strip() for col in metadata.columns]
    if "sampletype" not in metadata.columns:
        raise ValueError(f"Metadata table must include a sampletype column: {path}")

    id_candidates = ["sample", "sample_id", "Sample", "Unnamed: 0", "X", ""]
    sample_col = next((col for col in id_candidates if col in metadata.columns), metadata.columns[0])
    if sample_col == "sampletype":
        raise ValueError(f"Could not identify sample IDs in metadata table: {path}")

    metadata[sample_col] = metadata[sample_col].astype(str).str.strip()
    metadata["sampletype"] = metadata["sampletype"].astype(str).str.strip()
    metadata = metadata.rename(columns={sample_col: "sample"})
    return metadata[["sample", "sampletype"]]


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_gtf_attributes(attributes: str) -> dict[str, str]:
    return {key: value for key, value in ATTR_RE.findall(attributes)}


def build_tx2gene(gtf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transcript_records: list[dict[str, str]] = []
    gene_records: dict[str, dict[str, str]] = {}

    with open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue

            attrs = parse_gtf_attributes(fields[8])
            transcript_id = attrs.get("transcript_id")
            gene_id = attrs.get("gene_id")
            if not transcript_id or not gene_id:
                continue

            gene_name = attrs.get("gene_name", gene_id)
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            transcript_records.append(
                {
                    "transcript_id": transcript_id,
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                }
            )
            gene_records.setdefault(
                gene_id,
                {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "gene_type": gene_type,
                },
            )

    if not transcript_records:
        raise ValueError(f"No transcript records were parsed from GTF: {gtf_path}")

    tx2gene = pd.DataFrame.from_records(transcript_records).drop_duplicates("transcript_id")
    gene_metadata = pd.DataFrame.from_records(list(gene_records.values())).sort_values("gene_id")
    return tx2gene.sort_values("transcript_id"), gene_metadata


def build_sample_table(metadata: pd.DataFrame, quant_dir: Path) -> pd.DataFrame:
    sample_table = metadata.copy()
    sample_table["quant_file"] = sample_table["sample"].map(
        lambda sample: str(quant_dir / f"{sample}_quant" / "quant.sf")
    )
    missing = [path for path in sample_table["quant_file"] if not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} Salmon quant.sf files under {quant_dir}. First missing paths:\n{preview}"
        )
    return sample_table


def main() -> None:
    args = parse_args()
    metadata = read_metadata(args.metadata)
    sample_table = build_sample_table(metadata, args.quant_dir)
    tx2gene, gene_metadata = build_tx2gene(args.gtf)

    args.tx2gene_output.parent.mkdir(parents=True, exist_ok=True)
    tx2gene.to_csv(args.tx2gene_output, sep="\t", index=False)
    gene_metadata.to_csv(args.gene_metadata_output, sep="\t", index=False)
    sample_table.to_csv(args.sample_table_output, sep="\t", index=False)

    print(f"Wrote {len(tx2gene):,} transcript-to-gene records: {args.tx2gene_output}")
    print(f"Wrote {len(gene_metadata):,} gene records: {args.gene_metadata_output}")
    print(f"Wrote {len(sample_table):,} sample records: {args.sample_table_output}")


if __name__ == "__main__":
    main()
