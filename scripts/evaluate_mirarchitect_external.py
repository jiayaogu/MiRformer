#!/usr/bin/env python
"""Score MiRArchitect amiRNA interactions with continuous miRBench predictors.

The script appends external predictor scores to the existing MiRArchitect score
tables:

    mirarchitect_scores.csv
        Scores generated_mirna against gene.

    mirarchitect_off_target_scores.tsv
        Scores generated_mirna against off_target_gene, preserving row order
        from mirarchitect_off_targets.tsv.

Run from the project root with the miRBench environment:

    mirbench_venv/bin/python -u scripts/evaluate_mirarchitect_external.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from miRBench.encoder import get_encoder
from miRBench.predictor import get_predictor

from Global_parameters import PROJ_HOME


PREDICTORS = [
    "TargetScanCnn_McGeary2019",
    "InteractionAwareModel_Yang2024",
    "TargetNet_Min2021",
    "miRBind_Klimentova2022",
]
DNA_INPUT_PREDICTORS = {"TargetScanCnn_McGeary2019"}

DATA_DIR = Path(PROJ_HOME) / "mirarchitect_data"
DEFAULT_AMIRNA_PATH = DATA_DIR / "mirarchitect-amirnas.csv"
DEFAULT_TARGET_SCORE_PATH = DATA_DIR / "mirarchitect_scores.csv"
DEFAULT_OFF_TARGET_PATH = DATA_DIR / "mirarchitect_off_targets.tsv"
DEFAULT_OFF_TARGET_SCORE_PATH = DATA_DIR / "mirarchitect_off_target_scores.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amirna-path", type=Path, default=DEFAULT_AMIRNA_PATH)
    parser.add_argument("--target-score-path", type=Path, default=DEFAULT_TARGET_SCORE_PATH)
    parser.add_argument("--off-target-path", type=Path, default=DEFAULT_OFF_TARGET_PATH)
    parser.add_argument(
        "--off-target-score-path",
        type=Path,
        default=DEFAULT_OFF_TARGET_SCORE_PATH,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N rows from each table for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scoring and validation without writing output files.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_table(df: pd.DataFrame, path: Path, sep: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, sep=sep, index=False)
    tmp_path.replace(path)


def require_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def require_no_missing_sequences(
    df: pd.DataFrame, path: Path, columns: list[str]
) -> None:
    for column in columns:
        values = df[column].astype(str).str.strip()
        invalid = values.isin(["", "NA", "NaN", "nan"])
        if invalid.any():
            bad_rows = ", ".join(str(index) for index in df.index[invalid][:5])
            raise ValueError(
                f"{path} has missing/invalid {column} values at row indexes: {bad_rows}"
            )


def validate_target_alignment(
    amirnas: pd.DataFrame,
    target_scores: pd.DataFrame,
    amirna_path: Path,
    target_score_path: Path,
) -> None:
    require_columns(amirnas, amirna_path, ["gene", "generated_mirna"])
    require_columns(target_scores, target_score_path, ["gene", "noncodingRNA"])
    if len(amirnas) != len(target_scores):
        raise ValueError(
            f"Target row-count mismatch: {amirna_path} has {len(amirnas)} rows, "
            f"but {target_score_path} has {len(target_scores)} rows."
        )
    if not target_scores["gene"].reset_index(drop=True).equals(
        amirnas["gene"].reset_index(drop=True)
    ):
        raise ValueError(f"Target gene columns are not aligned by row order.")
    if not target_scores["noncodingRNA"].reset_index(drop=True).equals(
        amirnas["generated_mirna"].reset_index(drop=True)
    ):
        raise ValueError(
            "Target score noncodingRNA column does not match generated_mirna by row order."
        )
    require_no_missing_sequences(amirnas, amirna_path, ["gene", "generated_mirna"])


def validate_off_target_alignment(
    off_targets: pd.DataFrame,
    off_target_scores: pd.DataFrame,
    off_target_path: Path,
    off_target_score_path: Path,
) -> None:
    require_columns(
        off_targets,
        off_target_path,
        ["gene", "off_target_gene", "noncodingRNA", "generated_mirna"],
    )
    require_columns(off_target_scores, off_target_score_path, ["gene", "noncodingRNA"])
    if len(off_targets) != len(off_target_scores):
        raise ValueError(
            f"Off-target row-count mismatch: {off_target_path} has {len(off_targets)} "
            f"rows, but {off_target_score_path} has {len(off_target_scores)} rows."
        )
    if not off_target_scores["gene"].reset_index(drop=True).equals(
        off_targets["gene"].reset_index(drop=True)
    ):
        raise ValueError(f"Off-target gene columns are not aligned by row order.")
    if not off_target_scores["noncodingRNA"].reset_index(drop=True).equals(
        off_targets["noncodingRNA"].reset_index(drop=True)
    ):
        raise ValueError(
            "Off-target score noncodingRNA column does not match source noncodingRNA by row order."
        )
    require_no_missing_sequences(
        off_targets, off_target_path, ["off_target_gene", "generated_mirna"]
    )


def score_interactions(
    df: pd.DataFrame,
    mirna_col: str,
    gene_col: str,
    predictors: list[str] = PREDICTORS,
) -> pd.DataFrame:
    scores = {}
    for predictor in predictors:
        print(
            f"  scoring {predictor} on {len(df):,} pairs "
            f"({mirna_col} vs {gene_col}) ...",
            flush=True,
        )
        encoder = get_encoder(predictor)
        model = get_predictor(predictor)
        encoded_df = df.copy()
        encoded_df[mirna_col] = encoded_df[mirna_col].astype(str).str.upper()
        encoded_df[gene_col] = encoded_df[gene_col].astype(str).str.upper()
        if predictor in DNA_INPUT_PREDICTORS:
            encoded_df[mirna_col] = encoded_df[mirna_col].str.replace(
                "U", "T", regex=False
            )
            encoded_df[gene_col] = encoded_df[gene_col].str.replace(
                "U", "T", regex=False
            )
        encoded = encoder(encoded_df, miRNA_col=mirna_col, gene_col=gene_col)
        scores[predictor] = np.asarray(model(encoded)).ravel()
    return pd.DataFrame(scores, index=df.index)


def add_target_scores(
    target_scores: pd.DataFrame,
    external_scores: pd.DataFrame,
) -> pd.DataFrame:
    updated = target_scores.copy()
    for predictor in PREDICTORS:
        updated[predictor] = external_scores[predictor].to_numpy()
    return updated


def add_off_target_scores(
    off_target_scores: pd.DataFrame,
    external_scores: pd.DataFrame,
) -> pd.DataFrame:
    updated = off_target_scores.copy()
    for predictor in PREDICTORS:
        updated[f"{predictor}_off_target_score"] = external_scores[
            predictor
        ].to_numpy()
    return updated


def print_score_summary(label: str, scores: pd.DataFrame) -> None:
    print(f"\n{label} score summary:", flush=True)
    for predictor in PREDICTORS:
        values = scores[predictor].to_numpy(dtype=float)
        print(
            f"  {predictor}: mean={values.mean():.6g}, "
            f"std={values.std(ddof=1):.6g}, median={np.median(values):.6g}",
            flush=True,
        )


def maybe_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    return df.head(limit).copy()


def main() -> None:
    args = parse_args()

    amirnas = maybe_limit(read_csv(args.amirna_path), args.limit)
    target_scores = maybe_limit(read_csv(args.target_score_path), args.limit)
    off_targets = maybe_limit(read_tsv(args.off_target_path), args.limit)
    off_target_scores = maybe_limit(read_tsv(args.off_target_score_path), args.limit)

    validate_target_alignment(
        amirnas, target_scores, args.amirna_path, args.target_score_path
    )
    validate_off_target_alignment(
        off_targets,
        off_target_scores,
        args.off_target_path,
        args.off_target_score_path,
    )

    print(f"Loaded {len(amirnas):,} target pairs from {args.amirna_path}", flush=True)
    print(
        f"Loaded {len(off_targets):,} off-target pairs from {args.off_target_path}",
        flush=True,
    )

    print("\n=== Target interactions ===", flush=True)
    target_external = score_interactions(
        amirnas, mirna_col="generated_mirna", gene_col="gene"
    )
    print_score_summary("Target", target_external)

    print("\n=== Off-target interactions ===", flush=True)
    off_target_external = score_interactions(
        off_targets, mirna_col="generated_mirna", gene_col="off_target_gene"
    )
    print_score_summary("Off-target", off_target_external)

    if args.dry_run:
        print("\nDry run requested; no files written.", flush=True)
        return

    updated_target_scores = add_target_scores(target_scores, target_external)
    updated_off_target_scores = add_off_target_scores(
        off_target_scores, off_target_external
    )

    write_table(updated_target_scores, args.target_score_path, sep=",")
    write_table(updated_off_target_scores, args.off_target_score_path, sep="\t")
    print(f"\nUpdated {args.target_score_path}", flush=True)
    print(f"Updated {args.off_target_score_path}", flush=True)


if __name__ == "__main__":
    main()
