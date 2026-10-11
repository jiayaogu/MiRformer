#!/usr/bin/env python3
"""Export full RNAhybrid duplexes for generated miRNA and siRNA targets."""

import argparse
import csv
import re
import subprocess
from pathlib import Path

from Global_parameters import PROJ_HOME, RNAHYBRID_BIN


DEFAULT_INPUT_PATH = Path(PROJ_HOME) / "siRNA_targets_data" / "fda_sirna_targets.csv"
DEFAULT_OUTPUT_PATH = (
    Path(PROJ_HOME) / "siRNA_targets_data" / "generated_mirna_target_rnahybrid_duplexes.tsv"
)
DEFAULT_REPORT_PATH = (
    Path(PROJ_HOME) / "siRNA_targets_data" / "generated_mirna_target_rnahybrid_duplexes.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--rnahybrid-bin", type=Path, default=Path(RNAHYBRID_BIN))
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def normalize_dna(sequence: object) -> str:
    return re.sub(r"[^ACGTU]", "", str(sequence).upper()).replace("U", "T")


def to_rna(sequence: str) -> str:
    return sequence.replace("T", "U")


def run_rnahybrid(
    target_rna: str,
    mirna_rna_5to3: str,
    rnahybrid_bin: Path,
    timeout: int,
) -> str:
    result = subprocess.run(
        [
            str(rnahybrid_bin),
            "-b",
            "1",
            "-d",
            "1.28,0.9",
            target_rna,
            mirna_rna_5to3,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"RNAhybrid failed with code {result.returncode}")
    if not result.stdout.strip():
        raise RuntimeError("RNAhybrid returned empty output.")
    return result.stdout.rstrip()


def parse_rnahybrid_output(output: str) -> dict[str, object]:
    mfe_match = re.search(r"mfe:\s+(-?\d+(?:\.\d+)?)", output)
    p_value_match = re.search(r"p-value:\s+([0-9.eE+-]+)", output)
    position_match = re.search(r"position\s+(\d+)", output)

    lines = output.splitlines()
    position_index = next(
        (index for index, line in enumerate(lines) if line.startswith("position")),
        None,
    )

    alignment_block = {
        "target_alignment_line": "",
        "target_pairing_line": "",
        "mirna_pairing_line": "",
        "mirna_alignment_line": "",
    }
    if position_index is not None and position_index + 4 < len(lines):
        alignment_block = {
            "target_alignment_line": lines[position_index + 1].rstrip(),
            "target_pairing_line": lines[position_index + 2].rstrip(),
            "mirna_pairing_line": lines[position_index + 3].rstrip(),
            "mirna_alignment_line": lines[position_index + 4].rstrip(),
        }

    return {
        "mfe_kcal_per_mol": float(mfe_match.group(1)) if mfe_match else "",
        "p_value": float(p_value_match.group(1)) if p_value_match else "",
        "position_1based": int(position_match.group(1)) if position_match else "",
        **alignment_block,
    }


def load_inputs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        rows = [{key: value or "" for key, value in row.items()} for row in reader]

    required_columns = {"drug", "gene_name", "gene", "generated_mirna"}
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return rows


def write_tsv(records: list[dict[str, object]], output_path: Path) -> None:
    if not records:
        raise ValueError("No RNAhybrid duplex records were generated.")

    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def build_report(records: list[dict[str, object]], report_path: Path) -> None:
    with report_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{'=' * 80}\n")
            handle.write(f"{record['drug']} ({record['gene_name']})\n")
            handle.write(f"accession: {record.get('accession', '')}\n")
            handle.write(f"region: {record.get('region', '')}\n")
            handle.write(f"generated_mirna_5to3_dna: {record['generated_mirna_5to3_dna']}\n")
            handle.write(f"generated_mirna_5to3_rna: {record['generated_mirna_5to3_rna']}\n")
            handle.write(f"target_mrna_window_dna: {record['target_mrna_window_dna']}\n")
            handle.write(f"mfe_kcal_per_mol: {record['mfe_kcal_per_mol']}\n")
            handle.write(f"p_value: {record['p_value']}\n")
            handle.write(f"position_1based: {record['position_1based']}\n\n")
            handle.write(f"{record['raw_output']}\n\n")


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args.input_path)
    records: list[dict[str, object]] = []

    for row in inputs:
        target_dna = normalize_dna(row["gene"])
        generated_5to3_dna = normalize_dna(row["generated_mirna"])

        raw_output = run_rnahybrid(
            target_rna=to_rna(target_dna),
            mirna_rna_5to3=to_rna(generated_5to3_dna),
            rnahybrid_bin=args.rnahybrid_bin,
            timeout=args.timeout,
        )
        parsed = parse_rnahybrid_output(raw_output)

        record = {
            "drug": row.get("drug", ""),
            "gene_name": row.get("gene_name", ""),
            "accession": row.get("accession", ""),
            "region": row.get("region", ""),
            "binding_start": row.get("binding_start", ""),
            "binding_end": row.get("binding_end", ""),
            "target_mrna_window_dna": target_dna,
            "target_mrna_window_rna": to_rna(target_dna),
            "generated_mirna_5to3_dna": generated_5to3_dna,
            "generated_mirna_5to3_rna": to_rna(generated_5to3_dna),
            "raw_output": raw_output,
            **parsed,
        }
        records.append(record)

    write_tsv(records, args.output_path)
    build_report(records, args.report_path)
    print(f"Saved TSV to {args.output_path}")
    print(f"Saved text report to {args.report_path}")


if __name__ == "__main__":
    main()
