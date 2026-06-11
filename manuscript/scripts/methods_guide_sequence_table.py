#!/usr/bin/env python3
"""Generate the methods supplementary guide-sequence table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HEADER = [
    "library",
    "guide_id",
    "target_gene",
    "guide_class",
    "sequence_5to3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Table M1 with custom sgRNA guide sequences."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tcf7-pilot", required=True, type=Path)
    parser.add_argument("--in-vivo-screen-1", required=True, type=Path)
    parser.add_argument("--in-vivo-screen-2", required=True, type=Path)
    parser.add_argument("--ten-gene-perturbseq", required=True, type=Path)
    parser.add_argument("--trim28-setdb1-perturbseq", required=True, type=Path)
    parser.add_argument("--trim28-validation", required=True, type=Path)
    parser.add_argument("--trim28-validation-two-guide", required=True, type=Path)
    return parser.parse_args()


def clean(value: str) -> str:
    return value.strip().replace("\r", "")


def classify_guide(guide_id: str, target_gene: str) -> str:
    guide_upper = guide_id.upper()
    gene_upper = target_gene.upper()
    if "INTERGENIC" in guide_upper or "NON-GENE" in guide_upper:
        return "intergenic control"
    if guide_upper == "NT" or "NO_SITE" in guide_upper:
        return "non-targeting control"
    if gene_upper in {"NON-TARGETING", "NT"}:
        return "non-targeting control"
    return "targeting"


def infer_target_gene(guide_id: str) -> str:
    guide_id = clean(guide_id)
    guide_upper = guide_id.upper()
    if guide_upper == "NT" or guide_upper.startswith("NO_SITE"):
        return "Non-targeting"
    if "INTERGENIC" in guide_upper or "NON-GENE" in guide_upper:
        return "intergenic"
    if "-" in guide_id:
        return guide_id.rsplit("-", 1)[0]
    return guide_id


def validate_sequence(sequence: str, path: Path) -> str:
    sequence = clean(sequence).upper()
    invalid = sorted(set(sequence) - set("ACGTN"))
    if invalid:
        invalid_text = "".join(invalid)
        raise ValueError(f"Invalid sequence characters in {path}: {invalid_text}")
    return sequence


def make_row(
    *,
    library: str,
    guide_id: str,
    target_gene: str,
    sequence: str,
    path: Path,
) -> dict[str, str]:
    guide_id = clean(guide_id)
    target_gene = clean(target_gene)
    return {
        "library": library,
        "guide_id": guide_id,
        "target_gene": target_gene,
        "guide_class": classify_guide(guide_id, target_gene),
        "sequence_5to3": validate_sequence(sequence, path),
    }


def read_csv_guides(
    path: Path,
    *,
    library: str,
    sequence_field: str,
    guide_id_field: str,
    target_gene_field: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            if not any(clean(value) for value in record.values() if value is not None):
                continue
            rows.append(
                make_row(
                    library=library,
                    guide_id=record[guide_id_field],
                    target_gene=record[target_gene_field],
                    sequence=record[sequence_field],
                    path=path,
                )
            )
    return rows


def read_poolq_reference(
    path: Path,
    *,
    library: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for record in reader:
            if not record:
                continue
            sequence, guide_id = record[:2]
            guide_id = clean(guide_id)
            if not sequence or not guide_id:
                continue
            rows.append(
                make_row(
                    library=library,
                    guide_id=guide_id,
                    target_gene=infer_target_gene(guide_id),
                    sequence=sequence,
                    path=path,
                )
            )
    return rows


def build_table(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        read_csv_guides(
            args.tcf7_pilot,
            library="Tcf7 pilot in vitro CRISPR library",
            sequence_field="guide",
            guide_id_field="grna",
            target_gene_field="gene",
        )
    )
    rows.extend(
        read_poolq_reference(
            args.in_vivo_screen_1,
            library="first in vivo Tcf7 sort-screen library",
        )
    )
    rows.extend(
        read_poolq_reference(
            args.in_vivo_screen_2,
            library="second in vivo Tcf7 sort-screen library",
        )
    )
    rows.extend(
        read_csv_guides(
            args.ten_gene_perturbseq,
            library="10-gene Perturb-seq",
            sequence_field="guide",
            guide_id_field="grna",
            target_gene_field="gene",
        )
    )
    rows.extend(
        read_csv_guides(
            args.trim28_setdb1_perturbseq,
            library="Trim28/Setdb1 Perturb-seq",
            sequence_field="sequence",
            guide_id_field="guide_name",
            target_gene_field="gene",
        )
    )
    rows.extend(
        read_poolq_reference(
            args.trim28_validation,
            library="Trim28 validation guide-count reference",
        )
    )
    rows.extend(
        read_poolq_reference(
            args.trim28_validation_two_guide,
            library="Trim28 validation two-guide reference",
        )
    )
    return rows


def write_table(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = build_table(args)
    write_table(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
