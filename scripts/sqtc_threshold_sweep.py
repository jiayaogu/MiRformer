#!/usr/bin/env python3
"""Run an SQTC threshold sweep from saved per-target SQTC percentiles.

This script does not rescore predictor pairs. It reads ``sqtc_per_target.csv``
from ``evaluate_sqtc.py`` and recomputes qualification at several
TargetPercentile thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from specimir_panel_analysis import CONTINUOUS_PREDICTORS

from Global_parameters import PROJ_HOME


DATA_DIR = Path(PROJ_HOME) / "mirarchitect_data"
DEFAULT_RESULTS_DIR = DATA_DIR / "sqtc_results"
DEFAULT_PER_TARGET_PATH = DEFAULT_RESULTS_DIR / "sqtc_per_target.csv"
DEFAULT_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-target-path", type=Path, default=DEFAULT_PER_TARGET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="TargetPercentile thresholds to evaluate.",
    )
    parser.add_argument("--min-predictors", type=int, default=2)
    parser.add_argument("--denominator", type=int, default=100)
    return parser.parse_args()


def read_generated_flags(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def compute_sweep(
    per_target: pd.DataFrame,
    thresholds: list[float],
    min_predictors: int,
    denominator: int,
) -> pd.DataFrame:
    percentile_columns = [
        f"{predictor}_target_percentile" for predictor in CONTINUOUS_PREDICTORS
    ]
    require_columns(
        per_target,
        ["method", "generated_mirna_present"] + percentile_columns,
        Path("sqtc_per_target.csv"),
    )
    generated = read_generated_flags(per_target["generated_mirna_present"])
    percentile_values = per_target[percentile_columns].astype(float)

    rows = []
    for threshold in thresholds:
        n_predictors_passing = percentile_values.ge(threshold).sum(axis=1)
        qualified = generated & n_predictors_passing.ge(min_predictors)
        working = per_target[["method"]].copy()
        working["generated_mirna_present"] = generated
        working["qualified"] = qualified

        for method, group in working.groupby("method", sort=False):
            generated_count = int(group["generated_mirna_present"].sum())
            qualified_count = int(group["qualified"].sum())
            rows.append(
                {
                    "threshold": threshold,
                    "min_predictors": min_predictors,
                    "method": method,
                    "total_targets": len(group),
                    "denominator": denominator,
                    "generated_mirna": generated_count,
                    "qualified": qualified_count,
                    "design_coverage": generated_count / denominator,
                    "conditional_specificity": (
                        qualified_count / generated_count
                        if generated_count
                        else np.nan
                    ),
                    "SQTC": qualified_count / denominator,
                }
            )
    return pd.DataFrame(rows)


def write_latex_sweep(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SQTC sensitivity to TargetPercentile threshold.}",
        r"\label{tab:sqtc-threshold-sweep}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (
            r"Threshold & Method & Generated candidates & Qualified candidates & "
            r"Design coverage & Conditional specificity & SQTC \\"
        ),
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.threshold:.1f} & "
            f"{row.method} & "
            f"{int(row.generated_mirna)}/{int(row.denominator)} & "
            f"{int(row.qualified)} & "
            f"{row.design_coverage:.2f} & "
            f"{row.conditional_specificity:.2f} & "
            f"{row.SQTC:.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{flushleft}\footnotesize "
                r"Thresholds were applied to each predictor's TargetPercentile. "
                r"Candidates were specificity-qualified if at least two of four "
                r"external predictors met or exceeded the threshold."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    per_target = pd.read_csv(args.per_target_path)
    summary = compute_sweep(
        per_target,
        thresholds=args.thresholds,
        min_predictors=args.min_predictors,
        denominator=args.denominator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sqtc_threshold_sweep.csv"
    tex_path = args.output_dir / "sqtc_threshold_sweep.tex"
    summary.to_csv(csv_path, index=False)
    write_latex_sweep(summary, tex_path)

    print(summary.to_string(index=False), flush=True)
    print(f"\nWrote {csv_path}", flush=True)
    print(f"Wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
