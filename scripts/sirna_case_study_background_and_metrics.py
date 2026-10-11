#!/usr/bin/env python3
"""Sample siRNA case-study background miRNAs and compute RNAhybrid overlap metrics."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

from Global_parameters import PROJ_HOME


DATA_DIR = Path(PROJ_HOME) / "siRNA_targets_data"
FDA_PATH = DATA_DIR / "fda_sirna_targets.csv"
FDA_DUPLEX_PATH = DATA_DIR / "generated_mirna_target_rnahybrid_duplexes.tsv"
BACKGROUND_POOL_PATH = Path(PROJ_HOME) / "Manakov2022" / "AGO2_eCLIP_Manakov2022_test_predicted.tsv"
BACKGROUND_OUTPUT_PATH = DATA_DIR / "background_mirnas.tsv"
METRICS_OUTPUT_PATH = DATA_DIR / "generated_mirna_target_rnahybrid_metrics.tsv"

RNA_BASES = set("ACGU")
DNA_BASES = set("ACGT")
MISSING_VALUES = {"", "0", "NA", "NAN", "NONE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fda-path", type=Path, default=FDA_PATH)
    parser.add_argument("--fda-duplex-path", type=Path, default=FDA_DUPLEX_PATH)
    parser.add_argument("--background-pool-path", type=Path, default=BACKGROUND_POOL_PATH)
    parser.add_argument("--background-output-path", type=Path, default=BACKGROUND_OUTPUT_PATH)
    parser.add_argument("--metrics-output-path", type=Path, default=METRICS_OUTPUT_PATH)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_dna(sequence: object) -> str:
    return re.sub(r"[^ACGTU]", "", str(sequence).upper()).replace("U", "T")


def to_rna(sequence: str) -> str:
    return sequence.replace("T", "U")


def load_delimited_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(handle, dialect=dialect)
        rows = []
        for row_id, row in enumerate(reader, start=1):
            cleaned = {key: (value or "") for key, value in row.items()}
            cleaned["_row_id"] = str(row_id)
            rows.append(cleaned)
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def exact_match_interval(target_window_dna: str, site_sequence_dna: str) -> tuple[int, int] | None:
    start_index = target_window_dna.find(site_sequence_dna)
    if start_index < 0:
        return None
    second_index = target_window_dna.find(site_sequence_dna, start_index + 1)
    if second_index >= 0:
        raise ValueError(
            f"Site sequence matched multiple positions in target window: {site_sequence_dna}"
        )
    start_1based = start_index + 1
    end_1based = start_1based + len(site_sequence_dna) - 1
    return start_1based, end_1based


def resolve_site_interval(row: dict[str, str]) -> tuple[int, int]:
    target_window = normalize_dna(row["gene"])
    sense_strand = normalize_dna(row["sense_strand"])
    exact_interval = exact_match_interval(target_window, sense_strand)
    if exact_interval is not None:
        return exact_interval

    binding_start = int(row["binding_start"])
    binding_end = int(row["binding_end"])
    if 1 <= binding_start <= binding_end <= len(target_window):
        return binding_start, binding_end

    window_start_0based = int(row["Start"])
    site_start_1based = binding_start - window_start_0based + 1
    site_end_1based = binding_end - window_start_0based
    if not (1 <= site_start_1based <= site_end_1based <= len(target_window)):
        raise ValueError(
            f"Could not resolve local site interval for {row.get('drug', '')}: "
            f"binding_start={binding_start}, binding_end={binding_end}"
        )
    return site_start_1based, site_end_1based


def seed_region_7mer(sequence_5to3_dna: str) -> str:
    if len(sequence_5to3_dna) < 8:
        raise ValueError(f"Sequence shorter than 8 nt cannot provide positions 2-8: {sequence_5to3_dna}")
    return sequence_5to3_dna[1:8]


def extract_alignment_contents(raw_output: str) -> tuple[int, str, str]:
    lines = raw_output.splitlines()
    position_index = next(
        (index for index, line in enumerate(lines) if line.startswith("position")),
        None,
    )
    if position_index is None or position_index + 4 >= len(lines):
        raise ValueError(f"RNAhybrid output missing alignment block:\n{raw_output}")

    position_match = re.search(r"position\s+(\d+)", lines[position_index])
    if position_match is None:
        raise ValueError(f"RNAhybrid output missing position value:\n{raw_output}")

    target_top = lines[position_index + 1]
    target_paired = lines[position_index + 2]
    content_start = len("target 5'")
    content_end = target_top.rfind(" 3'")
    if content_end <= content_start:
        raise ValueError(f"RNAhybrid target line was not parseable:\n{raw_output}")

    def aligned_slice(line: str) -> str:
        return line.ljust(content_end)[content_start:content_end]

    return (
        int(position_match.group(1)),
        aligned_slice(target_top),
        aligned_slice(target_paired),
    )


def paired_target_positions(raw_output: str) -> list[int]:
    start_1based, target_top, target_paired = extract_alignment_contents(raw_output)
    positions = []
    target_position = start_1based - 1
    for top_char, paired_char in zip(target_top, target_paired):
        has_target_base = top_char in RNA_BASES or paired_char in RNA_BASES
        if not has_target_base:
            continue
        target_position += 1
        if paired_char in RNA_BASES:
            positions.append(target_position)
    if not positions:
        raise ValueError(f"No paired target positions were recovered from RNAhybrid output:\n{raw_output}")
    return positions


def region_jaccard(interval_a: tuple[int, int], interval_b: tuple[int, int]) -> float:
    a_start, a_end = interval_a
    b_start, b_end = interval_b
    intersection_len = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union_len = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection_len
    return intersection_len / union_len


def fda_metrics_rows(
    fda_rows: list[dict[str, str]],
    duplex_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    fda_by_drug = {row["drug"]: row for row in fda_rows}
    metrics_rows = []
    for duplex_row in duplex_rows:
        drug = duplex_row["drug"]
        if drug not in fda_by_drug:
            raise ValueError(f"FDA row missing for duplex row: {drug}")

        fda_row = fda_by_drug[drug]
        site_interval = resolve_site_interval(fda_row)
        paired_positions = paired_target_positions(duplex_row["raw_output"])
        duplex_interval = (min(paired_positions), max(paired_positions))
        center = (duplex_interval[0] + duplex_interval[1]) / 2.0
        center_hit = int(site_interval[0] <= center <= site_interval[1])

        metrics_rows.append(
            {
                "drug": drug,
                "gene_name": fda_row["gene_name"],
                "generated_mirna_5to3_dna": normalize_dna(fda_row["generated_mirna"]),
                "sirna_guide_5to3_dna": normalize_dna(fda_row["noncodingRNA"]),
                "sense_strand_dna": normalize_dna(fda_row["sense_strand"]),
                "target_window_dna": normalize_dna(fda_row["gene"]),
                "site_start_1based": site_interval[0],
                "site_end_1based": site_interval[1],
                "paired_target_positions_1based": ",".join(str(pos) for pos in paired_positions),
                "duplex_start_1based": duplex_interval[0],
                "duplex_end_1based": duplex_interval[1],
                "duplex_center": center,
                "center_hit": center_hit,
                "region_jaccard": region_jaccard(duplex_interval, site_interval),
                "mfe_kcal_per_mol": duplex_row.get("mfe_kcal_per_mol", ""),
            }
        )
    return metrics_rows


def background_sample_rows(
    fda_rows: list[dict[str, str]],
    background_rows: list[dict[str, str]],
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    fda_target_windows = {normalize_dna(row["gene"]) for row in fda_rows}
    sirna_seeds = {seed_region_7mer(normalize_dna(row["noncodingRNA"])) for row in fda_rows}

    eligible_rows = []
    for row in background_rows:
        mirna_seq = normalize_dna(row.get("generated_mirna", ""))
        if mirna_seq in MISSING_VALUES:
            continue
        if len(mirna_seq) < 8:
            continue
        if normalize_dna(row.get("gene", "")) in fda_target_windows:
            continue
        if seed_region_7mer(mirna_seq) in sirna_seeds:
            continue
        eligible_rows.append(row)

    if len(eligible_rows) < sample_size:
        raise ValueError(
            f"Only {len(eligible_rows)} eligible background rows remain after filtering, "
            f"but sample_size={sample_size} was requested."
        )

    sampler = random.Random(seed)
    sampled_rows = sampler.sample(eligible_rows, sample_size)
    return [
        {
            "bg_id": f"bg_{index:04d}",
            "mirna_seq": normalize_dna(row["generated_mirna"]),
            "source_target_id": row["_row_id"],
        }
        for index, row in enumerate(sampled_rows, start=1)
    ]


def main() -> None:
    args = parse_args()
    fda_rows = load_delimited_rows(args.fda_path)
    duplex_rows = load_delimited_rows(args.fda_duplex_path)
    background_rows = load_delimited_rows(args.background_pool_path)

    sampled_background = background_sample_rows(
        fda_rows=fda_rows,
        background_rows=background_rows,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    metric_rows = fda_metrics_rows(fda_rows=fda_rows, duplex_rows=duplex_rows)

    write_tsv(args.background_output_path, sampled_background)
    write_tsv(args.metrics_output_path, metric_rows)

    print(f"Saved background sample to {args.background_output_path}")
    print(f"Saved FDA RNAhybrid metrics to {args.metrics_output_path}")


if __name__ == "__main__":
    main()
