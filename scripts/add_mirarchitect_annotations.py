#!/usr/bin/env python3
"""Add Manakov2022 annotation columns to MiRarchitect result CSVs."""

from __future__ import annotations

import argparse
import codecs
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_COLUMNS = ("gene_cluster_ID", "noncodingRNA")


def detect_file_format(path: Path) -> tuple[bool, str]:
    data = path.read_bytes()
    has_bom = data.startswith(codecs.BOM_UTF8)
    newline = "\r\n" if b"\r\n" in data else "\n"
    return has_bom, newline


def read_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]], bool, str]:
    has_bom, newline = detect_file_format(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        return list(reader.fieldnames), list(reader), has_bom, newline


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    delimiter: str,
    has_bom: bool,
    newline: str,
) -> None:
    encoding = "utf-8-sig" if has_bom else "utf-8"
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            lineterminator=newline,
        )
        writer.writeheader()
        writer.writerows(rows)


def require_columns(path: Path, fieldnames: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def append_annotation_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name not in ANNOTATION_COLUMNS] + list(
        ANNOTATION_COLUMNS
    )


def build_annotation_lookup(
    predicted_rows: list[dict[str, str]],
    wanted_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    annotations: dict[tuple[str, str], dict[str, str]] = {}
    conflicts: list[tuple[tuple[str, str], dict[str, str], dict[str, str]]] = []
    repaired_keys = 0

    for row in predicted_rows:
        value = {column: row[column] for column in ANNOTATION_COLUMNS}
        candidate_keys = [((row["gene"], row["generated_mirna"]), False)]

        seed_types = row.get("seed_types", "")
        repaired_mirna = seed_types[2:] if seed_types.startswith("NA") else ""
        if (
            not row["generated_mirna"]
            and repaired_mirna
            and set(repaired_mirna) <= set("ACGTU")
        ):
            candidate_keys.append(((row["gene"], repaired_mirna), True))

        for key, was_repaired in candidate_keys:
            if key not in wanted_keys:
                continue
            previous = annotations.get(key)
            if previous is not None and previous != value:
                conflicts.append((key, previous, value))
                continue
            annotations[key] = value
            if was_repaired:
                repaired_keys += 1

    if conflicts:
        examples = "\n".join(
            f"  gene={key[0]} generated_mirna={key[1]}: {first} vs {second}"
            for key, first, second in conflicts[:5]
        )
        raise ValueError(
            "Predicted TSV contains conflicting annotations for the same "
            f"(gene, generated_mirna) key. Examples:\n{examples}"
        )

    return annotations, repaired_keys


def annotate_specimir(
    specimir_rows: list[dict[str, str]],
    annotations: dict[tuple[str, str], dict[str, str]],
) -> int:
    missing: list[tuple[int, str, str]] = []

    for line_number, row in enumerate(specimir_rows, start=2):
        key = (row["gene"], row["generated_mirna"])
        value = annotations.get(key)
        if value is None:
            missing.append((line_number, key[0], key[1]))
            continue
        row.update(value)

    if missing:
        examples = "\n".join(
            f"  line {line_number}: gene={gene} generated_mirna={generated_mirna}"
            for line_number, gene, generated_mirna in missing[:10]
        )
        raise ValueError(
            "No annotation was found for these specimir rows. Examples:\n"
            f"{examples}"
        )

    return len(specimir_rows)


def build_gene_lookup(
    specimir_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    by_gene: dict[str, dict[str, str]] = {}
    conflicts: list[tuple[str, dict[str, str], dict[str, str]]] = []

    for row in specimir_rows:
        value = {column: row[column] for column in ANNOTATION_COLUMNS}
        previous = by_gene.get(row["gene"])
        if previous is not None and previous != value:
            conflicts.append((row["gene"], previous, value))
            continue
        by_gene[row["gene"]] = value

    if conflicts:
        examples = "\n".join(
            f"  gene={gene}: {first} vs {second}"
            for gene, first, second in conflicts[:5]
        )
        raise ValueError(
            "Specimir CSV contains conflicting annotations for the same gene. "
            f"Examples:\n{examples}"
        )

    return by_gene


def annotate_mirarchitect(
    mirarchitect_rows: list[dict[str, str]],
    specimir_by_gene: dict[str, dict[str, str]],
) -> int:
    missing: list[tuple[int, str]] = []

    for line_number, row in enumerate(mirarchitect_rows, start=2):
        value = specimir_by_gene.get(row["gene"])
        if value is None:
            missing.append((line_number, row["gene"]))
            continue
        row.update(value)

    if missing:
        examples = "\n".join(
            f"  line {line_number}: gene={gene}" for line_number, gene in missing[:10]
        )
        raise ValueError(
            "No specimir annotation was found for these mirarchitect rows. "
            f"Examples:\n{examples}"
        )

    return len(mirarchitect_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add gene_cluster_ID and noncodingRNA from the Manakov2022 predicted TSV "
            "to specimir_mirna.csv, then copy them to mirarchitect-amirnas.csv by gene."
        )
    )
    parser.add_argument(
        "--specimir-csv",
        type=Path,
        default=ROOT / "mirarchitect_data" / "specimir_mirna.csv",
        help="Path to specimir_mirna.csv",
    )
    parser.add_argument(
        "--mirarchitect-csv",
        type=Path,
        default=ROOT / "mirarchitect_data" / "mirarchitect-amirnas.csv",
        help="Path to mirarchitect-amirnas.csv",
    )
    parser.add_argument(
        "--predicted-tsv",
        type=Path,
        default=ROOT / "Manakov2022" / "AGO2_eCLIP_Manakov2022_test_predicted.tsv",
        help="Path to AGO2_eCLIP_Manakov2022_test_predicted.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec_fields, spec_rows, spec_bom, spec_newline = read_rows(args.specimir_csv, ",")
    mir_fields, mir_rows, mir_bom, mir_newline = read_rows(args.mirarchitect_csv, ",")
    pred_fields, pred_rows, _, _ = read_rows(args.predicted_tsv, "\t")

    require_columns(args.specimir_csv, spec_fields, {"gene", "generated_mirna"})
    require_columns(args.mirarchitect_csv, mir_fields, {"gene"})
    require_columns(
        args.predicted_tsv,
        pred_fields,
        {"gene", "generated_mirna", *ANNOTATION_COLUMNS},
    )

    wanted_keys = {(row["gene"], row["generated_mirna"]) for row in spec_rows}
    annotations, repaired_keys = build_annotation_lookup(pred_rows, wanted_keys)
    spec_count = annotate_specimir(spec_rows, annotations)
    specimir_by_gene = build_gene_lookup(spec_rows)
    mir_count = annotate_mirarchitect(mir_rows, specimir_by_gene)

    write_rows(
        args.specimir_csv,
        append_annotation_columns(spec_fields),
        spec_rows,
        ",",
        spec_bom,
        spec_newline,
    )
    write_rows(
        args.mirarchitect_csv,
        append_annotation_columns(mir_fields),
        mir_rows,
        ",",
        mir_bom,
        mir_newline,
    )

    print(
        f"Updated {spec_count} specimir rows and {mir_count} mirarchitect rows "
        f"with columns: {', '.join(ANNOTATION_COLUMNS)}"
    )
    if repaired_keys:
        print(
            "Resolved "
            f"{repaired_keys} annotation key from a blank generated_mirna TSV row "
            "where seed_types contained NA plus the generated miRNA sequence."
        )


if __name__ == "__main__":
    main()
