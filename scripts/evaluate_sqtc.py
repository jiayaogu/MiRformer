#!/usr/bin/env python
"""Evaluate SQTC for generated miRNAs using external predictor specificity.

SQTC definition:
    TargetPercentile = (1 / K) * sum_k [S(miRNA, target) > S(miRNA, off_target_k)]

A candidate passes specificity if at least ``min_predictors`` of the external
predictors have TargetPercentile >= ``threshold``. Missing design candidates
encoded as generated_mirna = 0 are counted as not generated and not qualified.

Default inputs:
    mirarchitect_data/specimir_mirna_off_targets.csv
    mirarchitect_data/mirarchitect_amirnas_off_targets.csv

Default outputs:
    mirarchitect_data/sqtc_results/sqtc_summary.csv
    mirarchitect_data/sqtc_results/sqtc_per_target.csv
    mirarchitect_data/sqtc_results/sqtc_pair_scores.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mirarchitect_external import PREDICTORS
from evaluate_mirarchitect_external import score_interactions
from specimir_panel_analysis import PREDICTOR_LABELS

from Global_parameters import PROJ_HOME


DATA_DIR = Path(PROJ_HOME) / "mirarchitect_data"
DEFAULT_METHOD_INPUTS = {
    "SpeciMiR": DATA_DIR / "specimir_mirna_off_targets.csv",
    "MiRArchitect": DATA_DIR / "mirarchitect_amirnas_off_targets.csv",
}
DEFAULT_OUTPUT_DIR = DATA_DIR / "sqtc_results"
REQUIRED_COLUMNS = [
    "No",
    "gene",
    "generated_mirna",
    "gene_cluster_ID",
    "noncodingRNA",
    "off_target_gene",
]
KEY_COLUMNS = ["method", "No", "gene"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specimir-path",
        type=Path,
        default=DEFAULT_METHOD_INPUTS["SpeciMiR"],
    )
    parser.add_argument(
        "--mirarchitect-path",
        type=Path,
        default=DEFAULT_METHOD_INPUTS["MiRArchitect"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-off-targets", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-predictors", type=int, default=2)
    parser.add_argument("--denominator", type=int, default=100)
    parser.add_argument(
        "--limit-targets",
        type=int,
        default=None,
        help="Use only the first N targets per method for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scoring and metric calculation without writing output files.",
    )
    return parser.parse_args()


def is_generated_candidate(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text not in {"", "0", "NA", "NaN", "nan", "None"}


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_method_pairs(
    method: str,
    path: Path,
    k_off_targets: int,
    limit_targets: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    require_columns(df, path, REQUIRED_COLUMNS)
    df = df.copy()
    df.insert(0, "method", method)

    grouped = df.groupby(["No", "gene"], sort=False)
    if limit_targets is not None:
        keep_keys = list(grouped.groups)[:limit_targets]
        keep_index = []
        for key in keep_keys:
            keep_index.extend(grouped.groups[key])
        df = df.loc[keep_index].reset_index(drop=True)
        grouped = df.groupby(["No", "gene"], sort=False)

    counts = grouped.size()
    bad_counts = counts[counts != k_off_targets]
    if not bad_counts.empty:
        preview = bad_counts.head().to_dict()
        raise ValueError(
            f"{path} must have exactly {k_off_targets} off-target rows per target. "
            f"Unexpected counts: {preview}"
        )

    varying_generated = grouped["generated_mirna"].nunique(dropna=False)
    bad_generated = varying_generated[varying_generated != 1]
    if not bad_generated.empty:
        preview = bad_generated.head().to_dict()
        raise ValueError(
            f"{path} has targets with multiple generated_mirna values: {preview}"
        )

    df["off_target_index"] = grouped.cumcount() + 1
    return df


def collect_targets_and_pairs(
    method_pairs: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_records = []
    score_target_records = []
    score_off_target_records = []

    for method, pairs in method_pairs.items():
        for (_, _), group in pairs.groupby(["No", "gene"], sort=False):
            first = group.iloc[0]
            generated = is_generated_candidate(first["generated_mirna"])
            target_record = {
                "method": method,
                "No": first["No"],
                "gene": first["gene"],
                "gene_cluster_ID": first["gene_cluster_ID"],
                "noncodingRNA": first["noncodingRNA"],
                "generated_mirna": first["generated_mirna"],
                "generated_mirna_present": generated,
                "n_off_targets": len(group),
            }
            target_records.append(target_record)

            if not generated:
                continue

            score_target_records.append(
                {
                    "method": method,
                    "No": first["No"],
                    "gene": first["gene"],
                    "generated_mirna": first["generated_mirna"],
                }
            )
            for row in group.itertuples(index=False):
                score_off_target_records.append(
                    {
                        "method": method,
                        "No": row.No,
                        "gene": row.gene,
                        "off_target_index": row.off_target_index,
                        "off_target_gene": row.off_target_gene,
                        "generated_mirna": row.generated_mirna,
                    }
                )

    return (
        pd.DataFrame(target_records),
        pd.DataFrame(score_target_records),
        pd.DataFrame(score_off_target_records),
    )


def score_pairs(
    score_targets: pd.DataFrame,
    score_off_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if score_targets.empty:
        target_scores = score_targets.copy()
        off_target_scores = score_off_targets.copy()
        for predictor in PREDICTORS:
            target_scores[predictor] = np.nan
            off_target_scores[predictor] = np.nan
        return target_scores, off_target_scores

    print(
        f"\n=== SQTC target scoring ({len(score_targets):,} generated candidates) ===",
        flush=True,
    )
    target_score_values = score_interactions(
        score_targets, mirna_col="generated_mirna", gene_col="gene"
    )
    target_scores = pd.concat(
        [score_targets.reset_index(drop=True), target_score_values.reset_index(drop=True)],
        axis=1,
    )

    print(
        f"\n=== SQTC off-target scoring ({len(score_off_targets):,} pairs) ===",
        flush=True,
    )
    off_target_score_values = score_interactions(
        score_off_targets, mirna_col="generated_mirna", gene_col="off_target_gene"
    )
    off_target_scores = pd.concat(
        [
            score_off_targets.reset_index(drop=True),
            off_target_score_values.reset_index(drop=True),
        ],
        axis=1,
    )
    return target_scores, off_target_scores


def build_per_target_results(
    targets: pd.DataFrame,
    target_scores: pd.DataFrame,
    off_target_scores: pd.DataFrame,
    threshold: float,
    min_predictors: int,
) -> pd.DataFrame:
    rows = []
    target_score_index = target_scores.set_index(KEY_COLUMNS).sort_index()
    off_target_score_index = off_target_scores.set_index(KEY_COLUMNS).sort_index()

    for target in targets.itertuples(index=False):
        row = target._asdict()
        key = (target.method, target.No, target.gene)
        n_predictors_passing = 0

        if target.generated_mirna_present:
            target_score_row = target_score_index.loc[key]
            off_target_score_rows = off_target_score_index.loc[key]
            if isinstance(off_target_score_rows, pd.Series):
                off_target_score_rows = off_target_score_rows.to_frame().T

            for predictor in PREDICTORS:
                target_score = float(target_score_row[predictor])
                off_target_values = off_target_score_rows[predictor].to_numpy(dtype=float)
                target_percentile = float((target_score > off_target_values).mean())
                predictor_passes = target_percentile >= threshold
                n_predictors_passing += int(predictor_passes)
                row[f"{predictor}_target_score"] = target_score
                row[f"{predictor}_off_target_mean"] = off_target_values.mean()
                row[f"{predictor}_off_target_median"] = np.median(off_target_values)
                row[f"{predictor}_target_percentile"] = target_percentile
                row[f"{predictor}_passes"] = predictor_passes
        else:
            for predictor in PREDICTORS:
                row[f"{predictor}_target_score"] = np.nan
                row[f"{predictor}_off_target_mean"] = np.nan
                row[f"{predictor}_off_target_median"] = np.nan
                row[f"{predictor}_target_percentile"] = np.nan
                row[f"{predictor}_passes"] = False

        row["n_predictors_passing"] = n_predictors_passing
        row["qualified"] = (
            bool(target.generated_mirna_present)
            and n_predictors_passing >= min_predictors
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_pair_scores(
    all_pairs: pd.DataFrame,
    target_scores: pd.DataFrame,
    off_target_scores: pd.DataFrame,
) -> pd.DataFrame:
    target_score_columns = {
        predictor: f"{predictor}_target_score" for predictor in PREDICTORS
    }
    off_target_score_columns = {
        predictor: f"{predictor}_off_target_score" for predictor in PREDICTORS
    }

    target_score_subset = target_scores[
        KEY_COLUMNS + PREDICTORS
    ].rename(columns=target_score_columns)
    off_target_score_subset = off_target_scores[
        KEY_COLUMNS + ["off_target_index", "off_target_gene"] + PREDICTORS
    ].rename(columns=off_target_score_columns)

    pair_scores = all_pairs.merge(target_score_subset, on=KEY_COLUMNS, how="left")
    pair_scores = pair_scores.merge(
        off_target_score_subset,
        on=KEY_COLUMNS + ["off_target_index", "off_target_gene"],
        how="left",
    )
    pair_scores["generated_mirna_present"] = pair_scores["generated_mirna"].map(
        is_generated_candidate
    )
    return pair_scores


def build_summary(
    per_target: pd.DataFrame,
    denominator: int,
    threshold: float,
    min_predictors: int,
    k_off_targets: int,
) -> pd.DataFrame:
    rows = []
    for method, group in per_target.groupby("method", sort=False):
        generated_count = int(group["generated_mirna_present"].sum())
        qualified_count = int(group["qualified"].sum())
        rows.append(
            {
                "method": method,
                "total_targets": len(group),
                "denominator": denominator,
                "k_off_targets": k_off_targets,
                "threshold": threshold,
                "min_predictors": min_predictors,
                "generated_mirna": generated_count,
                "qualified": qualified_count,
                "design_coverage": generated_count / denominator,
                "conditional_specificity": (
                    qualified_count / generated_count if generated_count else np.nan
                ),
                "SQTC": qualified_count / denominator,
            }
        )
    return pd.DataFrame(rows)


def write_latex_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Specificity-qualified target coverage (SQTC) for generated miRNA design methods.}",
        r"\label{tab:sqtc-results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        (
            r"Method & Generated candidates & Qualified candidates & "
            r"Design coverage & Conditional specificity & SQTC \\"
        ),
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
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
                r"SQTC was computed with $K=20$ off-target mRNAs per target. "
                r"A candidate was specificity-qualified if at least two of four "
                r"external predictors had TargetPercentile $\geq 0.70$. "
                r"MiRArchitect failures with generated\_mirna = 0 were counted "
                r"as not generated and not qualified."
                r"\end{flushleft}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    method_paths = {
        "SpeciMiR": args.specimir_path,
        "MiRArchitect": args.mirarchitect_path,
    }
    method_pairs = {
        method: load_method_pairs(
            method,
            path,
            k_off_targets=args.k_off_targets,
            limit_targets=args.limit_targets,
        )
        for method, path in method_paths.items()
    }
    for method, pairs in method_pairs.items():
        n_targets = pairs.groupby(["No", "gene"], sort=False).ngroups
        n_generated = (
            pairs.groupby(["No", "gene"], sort=False)["generated_mirna"]
            .first()
            .map(is_generated_candidate)
            .sum()
        )
        print(
            f"{method}: {n_targets:,} targets, {n_generated:,} generated candidates, "
            f"{len(pairs):,} off-target rows",
            flush=True,
        )

    targets, score_targets, score_off_targets = collect_targets_and_pairs(method_pairs)
    target_scores, off_target_scores = score_pairs(score_targets, score_off_targets)
    per_target = build_per_target_results(
        targets,
        target_scores,
        off_target_scores,
        threshold=args.threshold,
        min_predictors=args.min_predictors,
    )
    all_pairs = pd.concat(method_pairs.values(), ignore_index=True)
    pair_scores = build_pair_scores(all_pairs, target_scores, off_target_scores)
    summary = build_summary(
        per_target,
        denominator=args.denominator,
        threshold=args.threshold,
        min_predictors=args.min_predictors,
        k_off_targets=args.k_off_targets,
    )

    print("\nSQTC summary:")
    print(summary.to_string(index=False), flush=True)

    if args.dry_run:
        print("\nDry run requested; no files written.", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sqtc_summary.csv"
    latex_summary_path = args.output_dir / "sqtc_summary.tex"
    per_target_path = args.output_dir / "sqtc_per_target.csv"
    pair_scores_path = args.output_dir / "sqtc_pair_scores.csv"
    summary.to_csv(summary_path, index=False)
    write_latex_summary(summary, latex_summary_path)
    per_target.to_csv(per_target_path, index=False)
    pair_scores.to_csv(pair_scores_path, index=False)
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {latex_summary_path}", flush=True)
    print(f"Wrote {per_target_path}", flush=True)
    print(f"Wrote {pair_scores_path}", flush=True)


if __name__ == "__main__":
    main()
