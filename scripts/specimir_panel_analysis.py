#!/usr/bin/env python3
"""Analyze score distributions from the four SpeciMiR external datasets.

The workflow contains only external miRBench predictors. It deliberately does
not load, calculate, or compare against D_phi.

Outputs:
    panel_distribution_summary.csv
    panel_pairwise_arm_tests.csv
    panel_normalized_means.csv
    panel_elevation_and_effects.csv
    panel_paired_permirna.csv

Usage:
    python scripts/specimir_panel_analysis.py [INPUT_DIR] [OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Manakov2022"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "specimir_panel_results"

ARM_FILES = {
    "random_on": "specimir_external_scores_random_miRNA_target_mRNA.tsv",
    "real_on": "specimir_external_scores_real_miRNA_target_mRNA.tsv",
    "generated_on": "specimir_external_scores_generated_miRNA_target_mRNA.tsv",
    "generated_off": "specimir_external_scores_generated_miRNA_off_target_mRNA.tsv",
}

ARM_LABELS = {
    "random_on": "Random miRNA → target mRNA",
    "real_on": "Real miRNA → target mRNA",
    "generated_on": "Generated miRNA → target mRNA",
    "generated_off": "Generated miRNA → off-target mRNA",
}

SEED_PREDICTORS = [
    "Seed6mer",
    "Seed7mer",
    "Seed8mer",
    "Seed6merBulgeOrMismatch",
]

CONTINUOUS_PREDICTORS = [
    "TargetScanCnn_McGeary2019",
    "InteractionAwareModel_Yang2024",
    "TargetNet_Min2021",
    "miRBind_Klimentova2022",
]

PREDICTORS = SEED_PREDICTORS + CONTINUOUS_PREDICTORS

PREDICTOR_LABELS = {
    "Seed6mer": "Seed 6mer",
    "Seed7mer": "Seed 7mer",
    "Seed8mer": "Seed 8mer",
    "Seed6merBulgeOrMismatch": "Seed 6mer bulge/mismatch",
    "TargetScanCnn_McGeary2019": "TargetScanCNN",
    "InteractionAwareModel_Yang2024": "InteractionAware",
    "TargetNet_Min2021": "TargetNet",
    "miRBind_Klimentova2022": "miRBind",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_arms(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate all four score tables."""
    arms = {}
    for arm, filename in ARM_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input for {arm}: {path}")
        frame = pd.read_csv(path, sep="\t")
        missing = [predictor for predictor in PREDICTORS if predictor not in frame.columns]
        if missing:
            raise ValueError(f"{path} lacks predictor columns: {missing}")
        for predictor in PREDICTORS:
            frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce")
            if frame[predictor].isna().any():
                count = int(frame[predictor].isna().sum())
                raise ValueError(f"{path} has {count} invalid {predictor} scores")
        arms[arm] = frame

    for arm, frame in arms.items():
        for predictor in SEED_PREDICTORS:
            values = set(frame[predictor].unique())
            if not values.issubset({0, 1}):
                raise ValueError(f"{arm}/{predictor} is not binary: {sorted(values)}")
    return arms


def adjust_bh(p_values: list[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order] * len(finite) / np.arange(1, len(finite) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    local = np.empty_like(ranked)
    local[order] = np.minimum(ranked, 1.0)
    adjusted[finite_indices] = local
    return adjusted


def distribution_summary(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return descriptive statistics for every predictor and dataset."""
    rows = []
    for predictor in PREDICTORS:
        score_type = "binary seed indicator" if predictor in SEED_PREDICTORS else "continuous"
        for arm, frame in arms.items():
            values = frame[predictor].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            rows.append(
                {
                    "predictor": predictor,
                    "predictor_label": PREDICTOR_LABELS[predictor],
                    "score_type": score_type,
                    "dataset": arm,
                    "dataset_label": ARM_LABELS[arm],
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "min": values.min(),
                    "max": values.max(),
                    "positive_rate": values.mean() if predictor in SEED_PREDICTORS else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pairwise_arm_tests(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare all six dataset pairs separately for each predictor."""
    rows = []
    arm_pairs = list(combinations(ARM_FILES, 2))
    for predictor in PREDICTORS:
        predictor_rows = []
        for left_arm, right_arm in arm_pairs:
            left = arms[left_arm][predictor].to_numpy(dtype=float)
            right = arms[right_arm][predictor].to_numpy(dtype=float)
            result = stats.mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
            predictor_rows.append(
                {
                    "predictor": predictor,
                    "left_dataset": left_arm,
                    "right_dataset": right_arm,
                    "left_n": len(left),
                    "right_n": len(right),
                    "left_mean": left.mean(),
                    "right_mean": right.mean(),
                    "mean_difference": left.mean() - right.mean(),
                    "left_median": np.median(left),
                    "right_median": np.median(right),
                    "probability_left_greater": result.statistic / (len(left) * len(right)),
                    "p_value": result.pvalue,
                }
            )
        adjusted = adjust_bh([row["p_value"] for row in predictor_rows])
        for row, p_adjusted in zip(predictor_rows, adjusted):
            row["p_fdr_bh_within_predictor"] = p_adjusted
            rows.append(row)
    return pd.DataFrame(rows)


def normalized_means(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Place every predictor's four means on its random-to-real scale."""
    rows = []
    for predictor in PREDICTORS:
        random_on = arms["random_on"][predictor].to_numpy(dtype=float)
        real_on = arms["real_on"][predictor].to_numpy(dtype=float)
        generated_on = arms["generated_on"][predictor].to_numpy(dtype=float)
        generated_off = arms["generated_off"][predictor].to_numpy(dtype=float)
        means = {
            "random_on": random_on.mean(),
            "real_on": real_on.mean(),
            "generated_on": generated_on.mean(),
            "generated_off": generated_off.mean(),
        }
        reference_span = means["real_on"] - means["random_on"]
        if np.isclose(reference_span, 0.0):
            normalized_on = np.nan
            normalized_off = np.nan
        else:
            normalized_on = (means["generated_on"] - means["random_on"]) / reference_span
            normalized_off = (means["generated_off"] - means["random_on"]) / reference_span
        rows.append(
            {
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS[predictor],
                "score_type": "binary seed indicator" if predictor in SEED_PREDICTORS else "continuous",
                **means,
                "normalized_generated_on": normalized_on,
                "normalized_generated_off": normalized_off,
            }
        )
    return pd.DataFrame(rows)


def elevation_and_effects(
    arms: dict[str, pd.DataFrame], normalized: pd.DataFrame
) -> pd.DataFrame:
    """Test generated-target separation for continuous predictors only."""
    normalized_index = normalized.set_index("predictor")
    rows = []
    for predictor in CONTINUOUS_PREDICTORS:
        generated_on = arms["generated_on"][predictor].to_numpy(dtype=float)
        generated_off = arms["generated_off"][predictor].to_numpy(dtype=float)
        means = normalized_index.loc[predictor]
        result = stats.mannwhitneyu(
            generated_on,
            generated_off,
            alternative="greater",
            method="asymptotic",
        )
        average_sd = np.sqrt(
            (generated_on.var(ddof=1) + generated_off.var(ddof=1)) / 2.0
        )
        rows.append(
            {
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS[predictor],
                "score_type": "continuous",
                "random_on": means["random_on"],
                "real_on": means["real_on"],
                "generated_on": means["generated_on"],
                "generated_off": means["generated_off"],
                "normalized_generated_on": means["normalized_generated_on"],
                "normalized_generated_off": means["normalized_generated_off"],
                "generated_on_minus_off": means["generated_on"] - means["generated_off"],
                "cohens_d_on_vs_off": (
                    (means["generated_on"] - means["generated_off"]) / average_sd
                    if average_sd > 0
                    else np.nan
                ),
                "auc_generated_on_vs_off": result.statistic / (len(generated_on) * len(generated_off)),
                "p_value_generated_on_greater": result.pvalue,
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_fdr_bh_generated_on_greater"] = adjust_bh(
        frame["p_value_generated_on_greater"].tolist()
    )
    return frame


def paired_per_mirna(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run paired target/off-target tests for continuous predictors only."""
    generated_on = arms["generated_on"].groupby("mirna")[CONTINUOUS_PREDICTORS].mean()
    generated_off = arms["generated_off"].groupby("mirna")[CONTINUOUS_PREDICTORS].mean()
    generated_on, generated_off = generated_on.align(generated_off, join="inner")
    rows = []
    for predictor in CONTINUOUS_PREDICTORS:
        delta = (generated_on[predictor] - generated_off[predictor]).to_numpy(dtype=float)
        target_higher = float((delta > 0).mean())
        tied = float((delta == 0).mean())
        off_target_higher = float((delta < 0).mean())
        untied = target_higher + off_target_higher
        if np.all(delta == 0):
            p_value = 1.0
        else:
            p_value = stats.wilcoxon(delta, alternative="greater").pvalue
        rows.append(
            {
                "predictor": predictor,
                "predictor_label": PREDICTOR_LABELS[predictor],
                "n_unique_mirna": len(delta),
                "mean_delta": delta.mean(),
                "median_delta": np.median(delta),
                "fraction_target_higher": target_higher,
                "fraction_tied": tied,
                "fraction_off_target_higher": off_target_higher,
                "win_rate_untied": target_higher / untied if untied > 0 else np.nan,
                "p_value_wilcoxon_greater": p_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_fdr_bh_wilcoxon_greater"] = adjust_bh(
        frame["p_value_wilcoxon_greater"].tolist()
    )
    return frame


def run_analysis(input_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Run the complete analysis and write result tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = load_arms(input_dir)
    normalized = normalized_means(arms)
    tables = {
        "panel_distribution_summary.csv": distribution_summary(arms),
        "panel_pairwise_arm_tests.csv": pairwise_arm_tests(arms),
        "panel_normalized_means.csv": normalized,
        "panel_elevation_and_effects.csv": elevation_and_effects(arms, normalized),
        "panel_paired_permirna.csv": paired_per_mirna(arms),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    print("Loaded " + ", ".join(f"{arm} n={len(frame):,}" for arm, frame in arms.items()))
    print(f"Analyzed {len(PREDICTORS)} external predictors; D_phi was not included")
    print(f"Wrote {len(tables)} analysis tables to {output_dir}")
    return tables


def main() -> None:
    args = parse_args()
    run_analysis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
