#!/usr/bin/env python3
"""Plot D_phi-free distributions for the SpeciMiR external predictor panel.

The companion analysis script must be run first with the same input and output
directories.

Outputs are written in both PNG and SVG formats:
    panel_continuous_distributions
    panel_continuous_ecdf
    panel_seed_match_rates
    panel_normalized_common_scale
    panel_generated_on_vs_off

Usage:
    python scripts/specimir_panel_figures.py [INPUT_DIR] [OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


sys.path.insert(0, str(Path(__file__).resolve().parent))
from specimir_panel_analysis import ARM_FILES
from specimir_panel_analysis import ARM_LABELS
from specimir_panel_analysis import CONTINUOUS_PREDICTORS
from specimir_panel_analysis import DEFAULT_INPUT_DIR
from specimir_panel_analysis import DEFAULT_OUTPUT_DIR
from specimir_panel_analysis import PREDICTOR_LABELS
from specimir_panel_analysis import PREDICTORS
from specimir_panel_analysis import SEED_PREDICTORS
from specimir_panel_analysis import load_arms


ARM_ORDER = list(ARM_FILES)
ARM_COLORS = {
    "random_on": "#9E9E9E",
    "real_on": "#313131",
    "generated_on": "#D55E00",
    "generated_off": "#56B4E9",
}
ARM_SHORT_LABELS = {
    "random_on": "Random\n→ target",
    "real_on": "Real\n→ target",
    "generated_on": "Generated\n→ target",
    "generated_off": "Generated\n→ off-target",
}
FOCAL_COLOR = ARM_COLORS["generated_on"]
OFF_COLOR = ARM_COLORS["generated_off"]
DARK = "#313131"
REFERENCE = "#777777"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "legend.frameon": False,
            "text.color": DARK,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")


def arm_legend(arms: dict[str, pd.DataFrame]) -> list[Patch]:
    return [
        Patch(
            facecolor=ARM_COLORS[arm],
            alpha=0.78,
            label=f"{ARM_LABELS[arm]} (n={len(arms[arm]):,})",
        )
        for arm in ARM_ORDER
    ]


def format_adjusted_p_value(p_value: float) -> str:
    """Format a BH-adjusted p-value as a significance level."""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.1:
        return "*"
    return "ns"


def annotate_pairwise_tests(
    ax: plt.Axes,
    predictor: str,
    pairwise_tests: pd.DataFrame,
    values: list[np.ndarray],
) -> None:
    """Add all six BH-adjusted Mann–Whitney comparisons above a distribution."""
    display_order = [
        ("random_on", "real_on"),
        ("real_on", "generated_on"),
        ("generated_on", "generated_off"),
        ("random_on", "generated_on"),
        ("real_on", "generated_off"),
        ("random_on", "generated_off"),
    ]
    predictor_tests = pairwise_tests[pairwise_tests["predictor"] == predictor]
    lookup = {
        (row.left_dataset, row.right_dataset): row
        for row in predictor_tests.itertuples()
    }
    missing = [pair for pair in display_order if pair not in lookup]
    if missing:
        raise ValueError(f"Missing pairwise tests for {predictor}: {missing}")
    data_min = min(float(array.min()) for array in values)
    data_max = max(float(array.max()) for array in values)
    data_range = data_max - data_min
    if data_range == 0:
        data_range = max(abs(data_max), 1.0)
    base = data_max + 0.07 * data_range
    step = 0.10 * data_range
    bracket_height = 0.025 * data_range
    arm_position = {arm: position for position, arm in enumerate(ARM_ORDER, start=1)}
    for level, pair in enumerate(display_order):
        row = lookup[pair]
        left = arm_position[pair[0]]
        right = arm_position[pair[1]]
        y = base + level * step
        ax.plot(
            [left, left, right, right],
            [y, y + bracket_height, y + bracket_height, y],
            color=DARK,
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2,
            y + bracket_height,
            format_adjusted_p_value(row.p_fdr_bh_within_predictor),
            ha="center",
            va="bottom",
            fontsize=7,
            color=DARK,
            clip_on=False,
        )
    lower_limit = data_min - 0.05 * data_range
    upper_limit = base + (len(display_order) - 1) * step + 0.10 * data_range
    ax.set_ylim(lower_limit, upper_limit)


def continuous_distributions(
    arms: dict[str, pd.DataFrame],
    pairwise_tests: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    """Draw violin and box overlays for all continuous predictors."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    axes = axes.ravel()
    positions = np.arange(1, len(ARM_ORDER) + 1)
    for ax, predictor in zip(axes, CONTINUOUS_PREDICTORS):
        values = [arms[arm][predictor].to_numpy(dtype=float) for arm in ARM_ORDER]
        violins = ax.violinplot(
            values,
            positions=positions,
            widths=0.82,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, arm in zip(violins["bodies"], ARM_ORDER):
            body.set_facecolor(ARM_COLORS[arm])
            body.set_edgecolor(ARM_COLORS[arm])
            body.set_alpha(0.62)
        boxes = ax.boxplot(
            values,
            positions=positions,
            widths=0.19,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": DARK, "linewidth": 0.8},
            capprops={"color": DARK, "linewidth": 0.8},
        )
        for box, arm in zip(boxes["boxes"], ARM_ORDER):
            box.set_facecolor(ARM_COLORS[arm])
            box.set_edgecolor(DARK)
            box.set_alpha(0.95)
        means = [array.mean() for array in values]
        ax.scatter(positions, means, s=24, facecolor="white", edgecolor=DARK, zorder=5)
        ax.set_title(PREDICTOR_LABELS[predictor], fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([ARM_SHORT_LABELS[arm] for arm in ARM_ORDER], fontsize=8)
        ax.set_ylabel("Predicted binding score")
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
        annotate_pairwise_tests(ax, predictor, pairwise_tests, values)
    fig.legend(
        handles=arm_legend(arms)
        + [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor=DARK,
                label="Mean",
            )
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "External predictor score distributions across four interaction datasets",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Violin = density; box = interquartile range and median; white point = mean. "
        "Brackets: BH-corrected two-sided Mann–Whitney U test P-values within predictor; "
        "* P<=0.1, ** P<=0.01, *** P<=0.001, ns = not significant.",
        ha="center",
        fontsize=10,
        color=REFERENCE,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save_figure(fig, output_dir, "panel_continuous_distributions")
    return fig


def continuous_ecdf(arms: dict[str, pd.DataFrame], output_dir: Path) -> plt.Figure:
    """Draw empirical cumulative distributions without density smoothing."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axes = axes.ravel()
    for ax, predictor in zip(axes, CONTINUOUS_PREDICTORS):
        for arm in ARM_ORDER:
            values = np.sort(arms[arm][predictor].to_numpy(dtype=float))
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                values,
                cumulative,
                where="post",
                color=ARM_COLORS[arm],
                linewidth=1.7,
                alpha=0.92,
            )
        ax.set_title(PREDICTOR_LABELS[predictor], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted binding score")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.01)
        ax.grid(color="#E7E7E7", linewidth=0.7)
        if predictor != "TargetScanCnn_McGeary2019":
            ax.set_xlim(-0.03, 1.03)
    fig.legend(
        handles=arm_legend(arms),
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "Empirical cumulative score distributions",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    save_figure(fig, output_dir, "panel_continuous_ecdf")
    return fig


def wilson_interval(successes: float, total: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval for a binary proportion."""
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def seed_match_rates(arms: dict[str, pd.DataFrame], output_dir: Path) -> plt.Figure:
    """Plot positive rates and Wilson intervals for binary seed predictors."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
    positions = np.arange(len(ARM_ORDER))
    for ax, predictor in zip(axes.ravel(), SEED_PREDICTORS):
        rates = []
        lower_errors = []
        upper_errors = []
        for arm in ARM_ORDER:
            values = arms[arm][predictor].to_numpy(dtype=float)
            rate = values.mean()
            lower, upper = wilson_interval(values.sum(), len(values))
            rates.append(rate)
            lower_errors.append(max(0.0, rate - lower))
            upper_errors.append(max(0.0, upper - rate))
        bars = ax.bar(
            positions,
            rates,
            color=[ARM_COLORS[arm] for arm in ARM_ORDER],
            alpha=0.82,
            yerr=np.vstack([lower_errors, upper_errors]),
            capsize=3,
            linewidth=0,
        )
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(rate + 0.045, 0.96),
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(PREDICTOR_LABELS[predictor], fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([ARM_SHORT_LABELS[arm] for arm in ARM_ORDER], fontsize=8)
        ax.set_ylabel("Fraction with a detected seed match")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
    fig.suptitle(
        "Seed-match positive rates across four interaction datasets",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.text(0.5, 0.005, "Error bars show 95% Wilson intervals", ha="center", color=REFERENCE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, output_dir, "panel_seed_match_rates")
    return fig


def normalized_common_scale(normalized: pd.DataFrame, output_dir: Path) -> plt.Figure:
    """Show all predictors on their random-to-real normalized mean scale."""
    ordered = normalized.copy()
    ordered["type_order"] = ordered["score_type"].map(
        {"binary seed indicator": 0, "continuous": 1}
    )
    ordered = ordered.sort_values(
        ["type_order", "normalized_generated_on"], ascending=[True, False]
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    positions = np.arange(len(ordered))[::-1]
    for position, row in zip(positions, ordered.itertuples()):
        ax.plot(
            [row.normalized_generated_off, row.normalized_generated_on],
            [position, position],
            color="#C8C8C8",
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            row.normalized_generated_off,
            position,
            s=55,
            facecolor="white",
            edgecolor=OFF_COLOR,
            linewidth=1.8,
            zorder=3,
        )
        ax.scatter(
            row.normalized_generated_on,
            position,
            s=58,
            facecolor=FOCAL_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.axvline(0, color=REFERENCE, linewidth=1.0)
    ax.axvline(1, color=DARK, linewidth=1.1, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["predictor_label"], fontsize=10)
    all_values = np.concatenate(
        [
            ordered["normalized_generated_on"].to_numpy(dtype=float),
            ordered["normalized_generated_off"].to_numpy(dtype=float),
            np.array([0.0, 1.0]),
        ]
    )
    value_range = np.nanmax(all_values) - np.nanmin(all_values)
    margin = max(0.12 * value_range, 0.15)
    ax.set_xlim(np.nanmin(all_values) - margin, np.nanmax(all_values) + margin)
    ax.set_xlabel("Normalized mean score (random → target = 0; real → target = 1)")
    ax.set_title(
        "Generated target and off-target scores on a common external-panel scale",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=FOCAL_COLOR,
                markeredgecolor="white",
                markersize=8,
                label="Generated → target",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor=OFF_COLOR,
                markersize=8,
                label="Generated → off-target",
            ),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "panel_normalized_common_scale")
    return fig


def generated_on_vs_off(
    effects: pd.DataFrame, paired: pd.DataFrame, output_dir: Path
) -> plt.Figure:
    """Plot unpaired ranking AUC and paired within-miRNA win rates."""
    ordered = effects[
        effects["predictor"].isin(CONTINUOUS_PREDICTORS)
    ].sort_values("auc_generated_on_vs_off").reset_index(drop=True)
    paired_indexed = paired.set_index("predictor")
    positions = np.arange(len(ordered))
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(12.0, 5.5), sharey=True)

    left_values = ordered["auc_generated_on_vs_off"].to_numpy(dtype=float)
    left_ax.barh(
        positions,
        left_values - 0.5,
        left=0.5,
        height=0.62,
        color=FOCAL_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )
    left_ax.axvline(0.5, color=DARK, linestyle="--", linewidth=1.0)
    left_ax.set_yticks(positions)
    left_ax.set_yticklabels(ordered["predictor_label"], fontsize=10)
    left_ax.set_xlim(0.5, 1.0)
    left_ax.set_xlabel("P(generated target score > off-target score)\nMann–Whitney ranking AUC")
    left_ax.set_title("Unpaired target/off-target separation", fontweight="bold")
    left_ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)
    for position, value in zip(positions, left_values):
        label_x = min(value + 0.012, 0.985)
        ha = "left" if label_x > value else "right"
        left_ax.text(
            label_x,
            position,
            f"AUC={value:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            color=DARK,
        )

    paired_ordered = paired_indexed.loc[ordered["predictor"]]
    right_values = paired_ordered["win_rate_untied"].to_numpy(dtype=float)
    right_ax.barh(
        positions,
        right_values - 0.5,
        left=0.5,
        height=0.62,
        color=OFF_COLOR,
        edgecolor="white",
        linewidth=0.7,
    )
    right_ax.axvline(0.5, color=DARK, linestyle="--", linewidth=1.0)
    right_ax.set_xlim(0.5, 1.0)
    right_ax.set_xlabel("Fraction target-higher among untied miRNAs")
    right_ax.set_title("Paired within-miRNA separation", fontweight="bold")
    right_ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)
    for position, row in zip(positions, paired_ordered.itertuples()):
        label_x = min(row.win_rate_untied + 0.012, 0.985)
        ha = "left" if label_x > row.win_rate_untied else "right"
        right_ax.text(
            label_x,
            position,
            f"Win={row.win_rate_untied:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            color=DARK,
        )
        if row.fraction_tied >= 0.1:
            right_ax.text(
                0.985,
                position - 0.26,
                f"{row.fraction_tied:.0%} tied",
                va="center",
                ha="right",
                fontsize=8,
                color=REFERENCE,
            )
    fig.suptitle(
        "Continuous external predictors distinguish generated targets from off-targets",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "panel_generated_on_vs_off")
    return fig


def load_analysis_tables(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load tables required by the summary figures."""
    normalized_path = output_dir / "panel_normalized_means.csv"
    effects_path = output_dir / "panel_elevation_and_effects.csv"
    paired_path = output_dir / "panel_paired_permirna.csv"
    pairwise_path = output_dir / "panel_pairwise_arm_tests.csv"
    required = [normalized_path, effects_path, paired_path, pairwise_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing analysis output: {joined}. Run specimir_panel_analysis.py first."
        )
    return (
        pd.read_csv(normalized_path),
        pd.read_csv(effects_path),
        pd.read_csv(paired_path),
        pd.read_csv(pairwise_path),
    )


def main() -> None:
    args = parse_args()
    apply_style()
    arms = load_arms(args.input_dir)
    normalized, effects, paired, pairwise_tests = load_analysis_tables(args.output_dir)
    figures = [
        continuous_distributions(arms, pairwise_tests, args.output_dir),
        continuous_ecdf(arms, args.output_dir),
        seed_match_rates(arms, args.output_dir),
        normalized_common_scale(normalized, args.output_dir),
        generated_on_vs_off(effects, paired, args.output_dir),
    ]
    for figure in figures:
        plt.close(figure)
    expected_predictors = set(PREDICTORS)
    continuous_predictors = set(CONTINUOUS_PREDICTORS)
    if set(normalized["predictor"]) != expected_predictors:
        raise ValueError("Normalized table does not contain the expected predictors")
    if set(effects["predictor"]) != continuous_predictors:
        raise ValueError("Effect table does not contain exactly the continuous predictors")
    if set(paired["predictor"]) != continuous_predictors:
        raise ValueError("Paired table does not contain exactly the continuous predictors")
    print(f"Rendered {len(figures)} D_phi-free figures in PNG and SVG format")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
