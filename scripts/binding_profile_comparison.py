#!/usr/bin/env python3
"""Run RNAhybrid alignments for MiRarchitect/SpeciMiR Experiments 2 and 3."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Global_parameters import PROJ_HOME, RNAHYBRID_BIN


DATA_DIR = Path(PROJ_HOME) / "mirarchitect_data"
DEFAULT_SPECIMIR_PATH = DATA_DIR / "specimir_mirna.csv"
DEFAULT_MIRARCHITECT_PATH = DATA_DIR / "mirarchitect-amirnas.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "rnahybrid_alignment_results"
VECTOR_LENGTH = 22
FAILURE_VALUES = {"", "0", "NA", "NAN", "NONE"}


@dataclass
class HybridAlignment:
    mfe: float
    p_value: float
    target_start: int
    target_end: int
    pairing_vector: list[int]
    raw_output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimir-path", type=Path, default=DEFAULT_SPECIMIR_PATH)
    parser.add_argument("--mirarchitect-path", type=Path, default=DEFAULT_MIRARCHITECT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rnahybrid-bin", type=Path, default=Path(RNAHYBRID_BIN))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N targets for a smoke test.",
    )
    return parser.parse_args()


def normalize_sequence(value: object) -> str:
    sequence = str(value).upper().replace("T", "U")
    return re.sub(r"[^ACGU]", "", sequence)


def has_candidate(value: object) -> bool:
    return str(value).strip().upper() not in FAILURE_VALUES


def inclusive_iou(
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    first_start, first_end = first
    second_start, second_end = second
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    first_len = first_end - first_start + 1
    second_len = second_end - second_start + 1
    union = first_len + second_len - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def fixed_pairing_vector(vector: list[int], length: int = VECTOR_LENGTH) -> list[int]:
    if len(vector) >= length:
        return vector[:length]
    return vector + [0] * (length - len(vector))


def vector_distance(first: list[int] | None, second: list[int] | None) -> float:
    if first is None or second is None:
        return 1.0
    first_fixed = fixed_pairing_vector(first)
    second_fixed = fixed_pairing_vector(second)
    differences = sum(abs(a - b) for a, b in zip(first_fixed, second_fixed))
    return differences / VECTOR_LENGTH


def count_bases(*lines: str) -> int:
    return sum(1 for line in lines for char in line if char in "ACGU")


def parse_rnahybrid_output(output: str) -> HybridAlignment:
    lines = output.splitlines()
    mfe_match = re.search(r"mfe:\s+(-?\d+(?:\.\d+)?)", output)
    p_value_match = re.search(r"p-value:\s+([0-9.eE+-]+)", output)
    position_index = next(
        (index for index, line in enumerate(lines) if line.startswith("position")),
        None,
    )

    if mfe_match is None or p_value_match is None or position_index is None:
        raise ValueError(f"RNAhybrid output was not parseable:\n{output}")
    if position_index + 4 >= len(lines):
        raise ValueError(f"RNAhybrid alignment block was incomplete:\n{output}")

    position_match = re.search(r"position\s+(\d+)", lines[position_index])
    if position_match is None:
        raise ValueError(f"RNAhybrid position line was not parseable:\n{output}")

    target_top = lines[position_index + 1]
    target_paired = lines[position_index + 2]
    mirna_paired = lines[position_index + 3]
    mirna_bottom = lines[position_index + 4]

    content_start = len("target 5'")
    content_end = target_top.rfind(" 3'")
    if content_end <= content_start:
        raise ValueError(f"RNAhybrid target line was not parseable:\n{output}")

    target_top_content = target_top[content_start:content_end]
    target_paired_content = target_paired[content_start:content_end]
    mirna_paired_content = mirna_paired[content_start:content_end]
    mirna_bottom_content = mirna_bottom[content_start:content_end]

    display_order_vector = []
    for paired_base, unpaired_base in zip(mirna_paired_content, mirna_bottom_content):
        if paired_base in "ACGU":
            display_order_vector.append(1)
        elif unpaired_base in "ACGU":
            display_order_vector.append(0)

    target_start = int(position_match.group(1))
    target_span = count_bases(target_top_content, target_paired_content)
    if target_span <= 0 or not display_order_vector:
        raise ValueError(f"RNAhybrid alignment block had no aligned bases:\n{output}")

    return HybridAlignment(
        mfe=float(mfe_match.group(1)),
        p_value=float(p_value_match.group(1)),
        target_start=target_start,
        target_end=target_start + target_span - 1,
        pairing_vector=fixed_pairing_vector(list(reversed(display_order_vector))),
        raw_output=output,
    )


def run_rnahybrid(
    target_sequence: str,
    mirna_sequence: str,
    rnahybrid_bin: Path,
    timeout: int,
) -> HybridAlignment:
    result = subprocess.run(
        [
            str(rnahybrid_bin),
            "-b",
            "1",
            "-d",
            "1.28, 0.9",
            target_sequence,
            mirna_sequence,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RNAhybrid failed with code {result.returncode}: {result.stderr}"
        )
    return parse_rnahybrid_output(result.stdout)


def vector_to_string(vector: list[int] | None) -> str:
    if vector is None:
        return ""
    return "".join(str(value) for value in fixed_pairing_vector(vector))


def load_inputs(specimir_path: Path, mirarchitect_path: Path, limit: int | None) -> pd.DataFrame:
    specimir = pd.read_csv(specimir_path, dtype=str)
    mirarchitect = pd.read_csv(mirarchitect_path, dtype=str)

    required = {"No", "gene", "generated_mirna", "noncodingRNA"}
    for path, frame in [(specimir_path, specimir), (mirarchitect_path, mirarchitect)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

    if len(specimir) != len(mirarchitect):
        raise ValueError("SpeciMiR and miRArchitect input files have different row counts.")
    if not specimir["gene"].reset_index(drop=True).equals(
        mirarchitect["gene"].reset_index(drop=True)
    ):
        raise ValueError("SpeciMiR and miRArchitect gene columns are not row-aligned.")

    inputs = specimir[["No", "gene", "noncodingRNA", "generated_mirna"]].rename(
        columns={"generated_mirna": "specimir_generated_mirna"}
    )
    inputs["mirarchitect_generated_mirna"] = mirarchitect["generated_mirna"].to_numpy()
    if limit is not None:
        inputs = inputs.head(limit).copy()
    return inputs


def alignment_record(
    row: pd.Series,
    method: str,
    mirna: str,
    alignment: HybridAlignment | None,
    generated: bool,
    error: str = "",
) -> dict[str, object]:
    return {
        "No": row["No"],
        "gene": row["gene"],
        "method": method,
        "mirna": mirna,
        "generated": int(generated),
        "mfe": alignment.mfe if alignment else pd.NA,
        "p_value": alignment.p_value if alignment else pd.NA,
        "target_start": alignment.target_start if alignment else pd.NA,
        "target_end": alignment.target_end if alignment else pd.NA,
        "pairing_vector_22": vector_to_string(alignment.pairing_vector if alignment else None),
        "error": error,
    }


def method_summary(metrics: pd.DataFrame, method: str, total_targets: int) -> dict[str, object]:
    method_metrics = metrics[metrics["method"] == method].copy()
    generated = int(method_metrics["generated"].sum())
    conditional = method_metrics[method_metrics["generated"] == 1]
    return {
        "method": method,
        "n_targets": total_targets,
        "n_generated": generated,
        "design_coverage": generated / total_targets if total_targets else pd.NA,
        "iou_conditional": conditional["iou"].mean() if len(conditional) else pd.NA,
        "iou_coverage_adjusted": method_metrics["iou"].mean(),
        "D_pair_conditional": conditional["D_pair"].mean() if len(conditional) else pd.NA,
        "D_pair_coverage_adjusted": method_metrics["D_pair_coverage_adjusted"].mean(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args.specimir_path, args.mirarchitect_path, args.limit)
    alignment_rows = []
    metric_rows = []

    for index, row in inputs.iterrows():
        target = normalize_sequence(row["gene"])
        observed_mirna = normalize_sequence(row["noncodingRNA"])
        print(
            f"[{index + 1}/{len(inputs)}] aligning observed miRNA for No={row['No']}",
            flush=True,
        )
        observed_alignment = run_rnahybrid(
            target, observed_mirna, args.rnahybrid_bin, args.timeout
        )
        alignment_rows.append(
            alignment_record(row, "observed", observed_mirna, observed_alignment, True)
        )
        observed_interval = (observed_alignment.target_start, observed_alignment.target_end)
        observed_vector = observed_alignment.pairing_vector

        for method, column in [
            ("specimir", "specimir_generated_mirna"),
            ("mirarchitect", "mirarchitect_generated_mirna"),
        ]:
            raw_mirna = row[column]
            generated = has_candidate(raw_mirna)
            normalized_mirna = normalize_sequence(raw_mirna)
            alignment = None
            error = ""

            if generated and normalized_mirna:
                print(
                    f"[{index + 1}/{len(inputs)}] aligning {method} candidate for No={row['No']}",
                    flush=True,
                )
                try:
                    alignment = run_rnahybrid(
                        target, normalized_mirna, args.rnahybrid_bin, args.timeout
                    )
                except Exception as exc:
                    generated = False
                    error = str(exc)
            else:
                error = "missing generated_mirna"

            alignment_rows.append(
                alignment_record(
                    row,
                    method,
                    normalized_mirna,
                    alignment,
                    generated,
                    error,
                )
            )

            candidate_interval = (
                (alignment.target_start, alignment.target_end) if alignment else None
            )
            d_pair = vector_distance(
                alignment.pairing_vector if alignment else None,
                observed_vector,
            )
            metric_rows.append(
                {
                    "No": row["No"],
                    "gene": row["gene"],
                    "method": method,
                    "observed_mirna": observed_mirna,
                    "generated_mirna": normalized_mirna,
                    "generated": int(generated),
                    "observed_start": observed_interval[0],
                    "observed_end": observed_interval[1],
                    "candidate_start": candidate_interval[0] if candidate_interval else pd.NA,
                    "candidate_end": candidate_interval[1] if candidate_interval else pd.NA,
                    "iou": inclusive_iou(observed_interval, candidate_interval),
                    "observed_pairing_vector_22": vector_to_string(observed_vector),
                    "candidate_pairing_vector_22": vector_to_string(
                        alignment.pairing_vector if alignment else None
                    ),
                    "D_pair": d_pair,
                    "D_pair_coverage_adjusted": d_pair if generated else 1.0,
                    "error": error,
                }
            )

    alignments = pd.DataFrame(alignment_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(
        [
            method_summary(metrics, "specimir", len(inputs)),
            method_summary(metrics, "mirarchitect", len(inputs)),
        ]
    )

    alignments_path = args.output_dir / "rnahybrid_alignments.csv"
    metrics_path = args.output_dir / "site_recovery_and_pairing_metrics.csv"
    summary_path = args.output_dir / "site_recovery_and_pairing_summary.csv"
    summary_tex_path = args.output_dir / "site_recovery_and_pairing_summary.tex"

    alignments.to_csv(alignments_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_latex(summary_tex_path, index=False, float_format="%.4f")

    print(f"Saved {alignments_path}", flush=True)
    print(f"Saved {metrics_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {summary_tex_path}", flush=True)
    print("\nSummary:", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
