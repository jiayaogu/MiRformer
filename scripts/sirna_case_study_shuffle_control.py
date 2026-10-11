#!/usr/bin/env python3
"""Run dinucleotide-preserving shuffle controls for the FDA siRNA case study."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from Global_parameters import PROJ_HOME, RNAHYBRID_BIN
from sirna_case_study_background_and_metrics import (
    load_delimited_rows,
    normalize_dna,
    paired_target_positions,
    region_jaccard,
    to_rna,
    write_tsv,
)


DATA_DIR = Path(PROJ_HOME) / "siRNA_targets_data"
FDA_METRICS_PATH = DATA_DIR / "generated_mirna_target_rnahybrid_metrics.tsv"
SHUFFLE_CONTROL_PATH = DATA_DIR / "shuffle_control.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fda-metrics-path", type=Path, default=FDA_METRICS_PATH)
    parser.add_argument("--output-path", type=Path, default=SHUFFLE_CONTROL_PATH)
    parser.add_argument("--rnahybrid-bin", type=Path, default=Path(RNAHYBRID_BIN))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def dinucleotide_counts(sequence: str) -> Counter[str]:
    return Counter(sequence[i : i + 2] for i in range(len(sequence) - 1))


def random_eulerian_shuffle(sequence: str, rng: random.Random) -> str:
    if len(sequence) < 2:
        return sequence

    adjacency: dict[str, list[str]] = defaultdict(list)
    for left, right in zip(sequence[:-1], sequence[1:]):
        adjacency[left].append(right)
    for node in adjacency:
        rng.shuffle(adjacency[node])

    stack = [sequence[0]]
    path: list[str] = []
    while stack:
        node = stack[-1]
        if adjacency[node]:
            stack.append(adjacency[node].pop())
        else:
            path.append(stack.pop())

    shuffled = "".join(reversed(path))
    if len(shuffled) != len(sequence):
        raise ValueError(f"Eulerian shuffle changed sequence length: {sequence} -> {shuffled}")
    if dinucleotide_counts(shuffled) != dinucleotide_counts(sequence):
        raise ValueError(f"Eulerian shuffle failed to preserve dinucleotides: {sequence} -> {shuffled}")
    return shuffled


def make_shuffles(sequence: str, n_shuffles: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffles: list[str] = []
    for index in range(n_shuffles):
        shuffled = sequence
        max_attempts = 500
        for _ in range(max_attempts):
            shuffled = random_eulerian_shuffle(sequence, rng)
            if shuffled != sequence:
                break
        shuffles.append(shuffled)
        rng.seed(seed + index + 1)
    return shuffles


def run_rnahybrid(target_rna: str, mirna_rna_5to3: str, rnahybrid_bin: Path, timeout: int) -> str:
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


def shuffle_stat_row(
    target_row: dict[str, str],
    shuffle_id: int,
    shuffled_mirna: str,
    rnahybrid_bin: Path,
    timeout: int,
) -> tuple[str, int, int, float]:
    raw_output = run_rnahybrid(
        target_rna=to_rna(normalize_dna(target_row["target_window_dna"])),
        mirna_rna_5to3=to_rna(shuffled_mirna),
        rnahybrid_bin=rnahybrid_bin,
        timeout=timeout,
    )
    paired_positions = paired_target_positions(raw_output)
    duplex_interval = (min(paired_positions), max(paired_positions))
    site_interval = (int(target_row["site_start_1based"]), int(target_row["site_end_1based"]))
    center = (duplex_interval[0] + duplex_interval[1]) / 2.0
    center_hit = int(site_interval[0] <= center <= site_interval[1])
    return target_row["drug"], shuffle_id, center_hit, region_jaccard(duplex_interval, site_interval)


def main() -> None:
    args = parse_args()
    target_rows = load_delimited_rows(args.fda_metrics_path)

    jobs = []
    for target_row in target_rows:
        observed_mirna = normalize_dna(target_row["generated_mirna_5to3_dna"])
        shuffles = make_shuffles(observed_mirna, args.n_shuffles, args.seed)
        for shuffle_id, shuffled in enumerate(shuffles, start=1):
            jobs.append((target_row, shuffle_id, shuffled))

    results_by_target: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                shuffle_stat_row,
                target_row,
                shuffle_id,
                shuffled,
                args.rnahybrid_bin,
                args.timeout,
            ): (target_row["drug"], shuffle_id)
            for target_row, shuffle_id, shuffled in jobs
        }
        for idx, future in enumerate(as_completed(future_map), start=1):
            drug, shuffle_id, center_hit, jaccard = future.result()
            results_by_target[drug].append((center_hit, jaccard))
            if idx % 100 == 0 or idx == len(future_map):
                print(f"Completed {idx}/{len(future_map)} shuffle RNAhybrid runs", flush=True)

    summary_rows = []
    for target_row in target_rows:
        drug = target_row["drug"]
        shuffle_values = results_by_target[drug]
        center_hits = [value[0] for value in shuffle_values]
        jaccards = [value[1] for value in shuffle_values]
        observed_center = int(target_row["center_hit"])
        observed_jaccard = float(target_row["region_jaccard"])
        mean_center = mean(center_hits)
        mean_jaccard = mean(jaccards)
        summary = (
            f"Observed center-hit={observed_center} vs shuffle mean={mean_center:.3f}; "
            f"observed Region Jaccard={observed_jaccard:.3f} vs shuffle mean={mean_jaccard:.3f}."
        )
        summary_rows.append(
            {
                "drug": drug,
                "generated_mirna_5to3_dna": normalize_dna(target_row["generated_mirna_5to3_dna"]),
                "n_shuffles": args.n_shuffles,
                "observed_center_hit": observed_center,
                "shuffle_mean_center_hit": mean_center,
                "observed_region_jaccard": observed_jaccard,
                "shuffle_mean_region_jaccard": mean_jaccard,
                "summary": summary,
            }
        )

    write_tsv(args.output_path, summary_rows)
    print(f"Saved shuffle control summary to {args.output_path}")


if __name__ == "__main__":
    main()
