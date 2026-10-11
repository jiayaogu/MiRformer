#!/usr/bin/env python3
"""Compute empirical null distributions and p-values for the FDA siRNA case study."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binom, chi2

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
BACKGROUND_PATH = DATA_DIR / "background_mirnas.tsv"
FDA_TARGETS_PATH = DATA_DIR / "fda_sirna_targets.csv"
FDA_METRICS_PATH = DATA_DIR / "generated_mirna_target_rnahybrid_metrics.tsv"
NULL_STATS_PATH = DATA_DIR / "null_stats.tsv"
PER_TARGET_PVALUES_PATH = DATA_DIR / "per_target_pvalues.tsv"
COMBINED_PVALUES_PATH = DATA_DIR / "combined_pvalues.txt"
FIGURE_NULL_DISTRIBUTIONS_PATH = DATA_DIR / "figure_null_distributions.svg"
FIGURE_SUMMARY_PATH = DATA_DIR / "figure_summary.svg"

SVG_NS = "http://www.w3.org/2000/svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-path", type=Path, default=BACKGROUND_PATH)
    parser.add_argument("--fda-targets-path", type=Path, default=FDA_TARGETS_PATH)
    parser.add_argument("--fda-metrics-path", type=Path, default=FDA_METRICS_PATH)
    parser.add_argument("--null-stats-path", type=Path, default=NULL_STATS_PATH)
    parser.add_argument("--per-target-pvalues-path", type=Path, default=PER_TARGET_PVALUES_PATH)
    parser.add_argument("--combined-pvalues-path", type=Path, default=COMBINED_PVALUES_PATH)
    parser.add_argument("--figure-null-distributions-path", type=Path, default=FIGURE_NULL_DISTRIBUTIONS_PATH)
    parser.add_argument("--figure-summary-path", type=Path, default=FIGURE_SUMMARY_PATH)
    parser.add_argument("--rnahybrid-bin", type=Path, default=Path(RNAHYBRID_BIN))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--hist-bins", type=int, default=20)
    return parser.parse_args()


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


def extract_mfe(raw_output: str) -> float:
    for line in raw_output.splitlines():
        if line.startswith("mfe:"):
            value = line.split(":", 1)[1].strip().split()[0]
            return float(value)
    raise ValueError(f"Could not extract MFE from RNAhybrid output:\n{raw_output}")


def compute_null_row(
    target_row: dict[str, str],
    background_row: dict[str, str],
    rnahybrid_bin: Path,
    timeout: int,
) -> dict[str, object]:
    target_rna = to_rna(normalize_dna(target_row["target_window_dna"]))
    mirna_rna = to_rna(normalize_dna(background_row["mirna_seq"]))
    raw_output = run_rnahybrid(target_rna, mirna_rna, rnahybrid_bin, timeout)
    paired_positions = paired_target_positions(raw_output)
    duplex_interval = (min(paired_positions), max(paired_positions))
    site_interval = (int(target_row["site_start_1based"]), int(target_row["site_end_1based"]))
    center = (duplex_interval[0] + duplex_interval[1]) / 2.0
    return {
        "target_id": target_row["drug"],
        "bg_id": background_row["bg_id"],
        "center_hit": int(site_interval[0] <= center <= site_interval[1]),
        "region_jaccard": region_jaccard(duplex_interval, site_interval),
        "mfe": extract_mfe(raw_output),
    }


def empirical_percentile(values: np.ndarray, observed: float) -> float:
    return 100.0 * float(np.mean(values <= observed))


def fisher_combined_p(p_values: Iterable[float]) -> float:
    p_list = list(p_values)
    statistic = -2.0 * sum(math.log(p) for p in p_list)
    return float(chi2.sf(statistic, df=2 * len(p_list)))


def format_sig(value: float) -> str:
    return f"{value:.4g}"


def draw_svg_text(parts: list[str], x: float, y: float, text: str, size: int = 12, anchor: str = "start", fill: str = "#111111", extra: str = "") -> None:
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" {extra}>{text}</text>'
    )


def make_null_histogram_svg(
    target_rows: list[dict[str, str]],
    per_target_rows: list[dict[str, object]],
    path: Path,
    bins: int,
) -> None:
    width, height = 1320, 880
    cols, rows = 3, 2
    panel_w, panel_h = 400, 360
    gap_x, gap_y = 30, 40
    left_margin, top_margin = 40, 40
    svg: list[str] = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    per_target_map = {row["drug"]: row for row in per_target_rows}
    for idx, target_row in enumerate(target_rows):
        panel_col = idx % cols
        panel_row = idx // cols
        x0 = left_margin + panel_col * (panel_w + gap_x)
        y0 = top_margin + panel_row * (panel_h + gap_y)
        plot_x = x0 + 55
        plot_y = y0 + 40
        plot_w = panel_w - 90
        plot_h = panel_h - 100

        stats_row = per_target_map[target_row["drug"]]
        null_values = np.asarray(stats_row["_null_jaccard_values"], dtype=float)
        observed = float(stats_row["region_jaccard_obs"])
        p_jaccard = float(stats_row["p_jaccard"])

        counts, edges = np.histogram(null_values, bins=bins, range=(0.0, 1.0))
        y_max = int(max(counts.max(), 1))

        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="none" stroke="#d0d0d0"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{plot_y + plot_h:.2f}" x2="{plot_x + plot_w:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>')

        for bar_idx, count in enumerate(counts):
            x_left = plot_x + plot_w * (edges[bar_idx] / 1.0)
            x_right = plot_x + plot_w * (edges[bar_idx + 1] / 1.0)
            bar_w = max(1.0, x_right - x_left - 1.0)
            bar_h = 0.0 if y_max == 0 else plot_h * (count / y_max)
            svg.append(
                f'<rect x="{x_left:.2f}" y="{plot_y + plot_h - bar_h:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" '
                f'fill="#bdbdbd" stroke="#7a7a7a" stroke-width="0.5"/>'
            )

        obs_x = plot_x + plot_w * observed
        svg.append(f'<line x1="{obs_x:.2f}" y1="{plot_y:.2f}" x2="{obs_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#d62728" stroke-width="3"/>')

        for tick, label in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
            tx = plot_x + plot_w * tick
            svg.append(f'<line x1="{tx:.2f}" y1="{plot_y + plot_h:.2f}" x2="{tx:.2f}" y2="{plot_y + plot_h + 6:.2f}" stroke="#222222" stroke-width="1"/>')
            draw_svg_text(svg, tx, plot_y + plot_h + 22, label, size=11, anchor="middle")

        for tick_value in [0, y_max // 2, y_max]:
            ty = plot_y + plot_h - (0 if y_max == 0 else plot_h * (tick_value / y_max))
            svg.append(f'<line x1="{plot_x - 6:.2f}" y1="{ty:.2f}" x2="{plot_x:.2f}" y2="{ty:.2f}" stroke="#222222" stroke-width="1"/>')
            draw_svg_text(svg, plot_x - 10, ty + 4, str(tick_value), size=11, anchor="end")

        draw_svg_text(svg, x0 + panel_w / 2, y0 + 24, target_row["drug"], size=16, anchor="middle")
        draw_svg_text(svg, x0 + panel_w / 2, y0 + panel_h - 12, "Region Jaccard", size=12, anchor="middle")
        draw_svg_text(svg, x0 + 18, y0 + panel_h / 2, "Count", size=12, anchor="middle", extra='transform="rotate(-90 {:.2f} {:.2f})"'.format(x0 + 18, y0 + panel_h / 2))
        draw_svg_text(svg, x0 + panel_w - 12, y0 + 26, f"p = {format_sig(p_jaccard)}", size=13, anchor="end", fill="#d62728")

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def make_summary_svg(per_target_rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1180, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 50, 100
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    all_null = np.concatenate([np.asarray(row["_null_jaccard_values"], dtype=float) for row in per_target_rows])
    grand_mean = float(np.mean(all_null))

    def y_to_px(value: float) -> float:
        return plot_y + plot_h * (1.0 - value)

    svg: list[str] = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{plot_x:.2f}" y1="{plot_y + plot_h:.2f}" x2="{plot_x + plot_w:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>',
        f'<line x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ty = y_to_px(tick)
        svg.append(f'<line x1="{plot_x - 6:.2f}" y1="{ty:.2f}" x2="{plot_x:.2f}" y2="{ty:.2f}" stroke="#222222" stroke-width="1"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{ty:.2f}" x2="{plot_x + plot_w:.2f}" y2="{ty:.2f}" stroke="#eeeeee" stroke-width="1"/>')
        draw_svg_text(svg, plot_x - 10, ty + 4, f"{tick:.2f}", size=11, anchor="end")

    mean_y = y_to_px(grand_mean)
    svg.append(f'<line x1="{plot_x:.2f}" y1="{mean_y:.2f}" x2="{plot_x + plot_w:.2f}" y2="{mean_y:.2f}" stroke="#888888" stroke-dasharray="6,4" stroke-width="2"/>')
    draw_svg_text(svg, plot_x + plot_w - 4, mean_y - 8, f"Mean null Jaccard = {format_sig(grand_mean)}", size=12, anchor="end", fill="#666666")

    x_positions = np.linspace(plot_x + 90, plot_x + plot_w - 90, len(per_target_rows))
    box_width = 54
    for idx, row in enumerate(per_target_rows):
        x = x_positions[idx]
        values = np.asarray(row["_null_jaccard_values"], dtype=float)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        vmin, vmax = float(np.min(values)), float(np.max(values))
        obs = float(row["region_jaccard_obs"])
        color = colors[idx % len(colors)]

        svg.append(f'<line x1="{x:.2f}" y1="{y_to_px(vmin):.2f}" x2="{x:.2f}" y2="{y_to_px(vmax):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x - 14:.2f}" y1="{y_to_px(vmin):.2f}" x2="{x + 14:.2f}" y2="{y_to_px(vmin):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x - 14:.2f}" y1="{y_to_px(vmax):.2f}" x2="{x + 14:.2f}" y2="{y_to_px(vmax):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(
            f'<rect x="{x - box_width / 2:.2f}" y="{y_to_px(q3):.2f}" width="{box_width:.2f}" '
            f'height="{max(1.0, y_to_px(q1) - y_to_px(q3)):.2f}" fill="#d9d9d9" stroke="#666666" stroke-width="1.5"/>'
        )
        svg.append(f'<line x1="{x - box_width / 2:.2f}" y1="{y_to_px(med):.2f}" x2="{x + box_width / 2:.2f}" y2="{y_to_px(med):.2f}" stroke="#333333" stroke-width="2"/>')
        svg.append(f'<circle cx="{x:.2f}" cy="{y_to_px(obs):.2f}" r="5.5" fill="{color}" stroke="#222222" stroke-width="1"/>')

        draw_svg_text(svg, x, plot_y + plot_h + 26, row["drug"], size=12, anchor="middle", extra='transform="rotate(25 {:.2f} {:.2f})"'.format(x, plot_y + plot_h + 26))

    draw_svg_text(svg, width / 2, 28, "Observed Region Jaccard vs Null Distributions", size=18, anchor="middle")
    draw_svg_text(svg, width / 2, height - 24, "FDA Target", size=13, anchor="middle")
    draw_svg_text(svg, 26, height / 2, "Region Jaccard", size=13, anchor="middle", extra='transform="rotate(-90 26 {:.2f})"'.format(height / 2))
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    background_rows = load_delimited_rows(args.background_path)
    target_rows = load_delimited_rows(args.fda_metrics_path)

    jobs = [
        (target_row, background_row)
        for target_row in target_rows
        for background_row in background_rows
    ]

    null_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                compute_null_row,
                target_row,
                background_row,
                args.rnahybrid_bin,
                args.timeout,
            ): (target_row["drug"], background_row["bg_id"])
            for target_row, background_row in jobs
        }
        for idx, future in enumerate(as_completed(future_map), start=1):
            null_rows.append(future.result())
            if idx % 500 == 0 or idx == len(future_map):
                print(f"Completed {idx}/{len(future_map)} null RNAhybrid runs", flush=True)

    target_order = [row["drug"] for row in target_rows]
    bg_order = [row["bg_id"] for row in background_rows]
    target_rank = {drug: i for i, drug in enumerate(target_order)}
    bg_rank = {bg_id: i for i, bg_id in enumerate(bg_order)}
    null_rows.sort(key=lambda row: (target_rank[row["target_id"]], bg_rank[row["bg_id"]]))
    write_tsv(args.null_stats_path, null_rows)

    null_by_target: dict[str, list[dict[str, object]]] = {drug: [] for drug in target_order}
    for row in null_rows:
        null_by_target[row["target_id"]].append(row)

    per_target_rows: list[dict[str, object]] = []
    for target_row in target_rows:
        drug = target_row["drug"]
        target_null = null_by_target[drug]
        center_null = np.asarray([int(row["center_hit"]) for row in target_null], dtype=float)
        jaccard_null = np.asarray([float(row["region_jaccard"]) for row in target_null], dtype=float)
        mfe_null = np.asarray([float(row["mfe"]) for row in target_null], dtype=float)

        center_obs = int(target_row["center_hit"])
        jaccard_obs = float(target_row["region_jaccard"])
        mfe_obs = float(target_row["mfe_kcal_per_mol"])
        n = len(target_null)

        per_target_rows.append(
            {
                "drug": drug,
                "center_hit_obs": center_obs,
                "p_center": (int(np.sum(center_null >= center_obs)) + 1) / (n + 1),
                "region_jaccard_obs": jaccard_obs,
                "null_jaccard_mean": float(np.mean(jaccard_null)),
                "null_jaccard_median": float(np.median(jaccard_null)),
                "null_jaccard_sd": float(np.std(jaccard_null, ddof=0)),
                "observed_percentile_jaccard": empirical_percentile(jaccard_null, jaccard_obs),
                "p_jaccard": (int(np.sum(jaccard_null >= jaccard_obs)) + 1) / (n + 1),
                "mfe_obs": mfe_obs,
                "null_mfe_mean": float(np.mean(mfe_null)),
                "null_mfe_median": float(np.median(mfe_null)),
                "null_mfe_sd": float(np.std(mfe_null, ddof=0)),
                "observed_percentile_mfe": empirical_percentile(mfe_null, mfe_obs),
                "p_mfe": (int(np.sum(mfe_null <= mfe_obs)) + 1) / (n + 1),
                "null_center_hit_rate": float(np.mean(center_null)),
                "_null_jaccard_values": jaccard_null.tolist(),
                "_null_mfe_values": mfe_null.tolist(),
            }
        )

    p_jaccard_list = [float(row["p_jaccard"]) for row in per_target_rows]
    p_mfe_list = [float(row["p_mfe"]) for row in per_target_rows]
    p_bg_list = [float(row["null_center_hit_rate"]) for row in per_target_rows]
    k = sum(int(row["center_hit_obs"]) for row in per_target_rows)
    p_bar = float(np.mean(p_bg_list))

    combined_p_jaccard = fisher_combined_p(p_jaccard_list)
    combined_p_mfe = fisher_combined_p(p_mfe_list)
    combined_p_center = float(binom.sf(k - 1, n=len(per_target_rows), p=p_bar))

    clean_per_target_rows = []
    for row in per_target_rows:
        clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
        clean_per_target_rows.append(clean_row)
    write_tsv(args.per_target_pvalues_path, clean_per_target_rows)

    args.combined_pvalues_path.write_text(
        "\n".join(
            [
                f"Fisher combined p (Region Jaccard) = {format_sig(combined_p_jaccard)}",
                f"Fisher combined p (MFE) = {format_sig(combined_p_mfe)}",
                f"Binomial exact p (Center-hit, k={k}/6, p_bar={format_sig(p_bar)}) = {format_sig(combined_p_center)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    make_null_histogram_svg(target_rows, per_target_rows, args.figure_null_distributions_path, args.hist_bins)
    make_summary_svg(per_target_rows, args.figure_summary_path)

    print(f"Saved null stats to {args.null_stats_path}")
    print(f"Saved per-target p-values to {args.per_target_pvalues_path}")
    print(f"Saved combined p-values to {args.combined_pvalues_path}")
    print(f"Saved histogram figure to {args.figure_null_distributions_path}")
    print(f"Saved summary figure to {args.figure_summary_path}")


if __name__ == "__main__":
    main()
