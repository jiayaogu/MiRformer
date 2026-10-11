#!/usr/bin/env python3
"""Render MFE null-distribution figures for the FDA siRNA case study."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Global_parameters import PROJ_HOME
from sirna_case_study_background_and_metrics import load_delimited_rows


DATA_DIR = Path(PROJ_HOME) / "siRNA_targets_data"
NULL_STATS_PATH = DATA_DIR / "null_stats.tsv"
PER_TARGET_PVALUES_PATH = DATA_DIR / "per_target_pvalues.tsv"
HISTOGRAM_FIGURE_PATH = DATA_DIR / "figure_mfe_null_distributions.svg"
SUMMARY_FIGURE_PATH = DATA_DIR / "figure_mfe_summary.svg"
SVG_NS = "http://www.w3.org/2000/svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-stats-path", type=Path, default=NULL_STATS_PATH)
    parser.add_argument("--per-target-pvalues-path", type=Path, default=PER_TARGET_PVALUES_PATH)
    parser.add_argument("--histogram-figure-path", type=Path, default=HISTOGRAM_FIGURE_PATH)
    parser.add_argument("--summary-figure-path", type=Path, default=SUMMARY_FIGURE_PATH)
    parser.add_argument("--hist-bins", type=int, default=20)
    return parser.parse_args()


def draw_svg_text(parts: list[str], x: float, y: float, text: str, size: int = 12, anchor: str = "start", fill: str = "#111111", extra: str = "") -> None:
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" {extra}>{text}</text>'
    )


def format_sig(value: float) -> str:
    return f"{value:.4g}"


def global_mfe_range(per_target_rows: list[dict[str, str]], null_by_drug: dict[str, np.ndarray]) -> tuple[float, float]:
    observed = [float(row["mfe_obs"]) for row in per_target_rows]
    null_values = np.concatenate([null_by_drug[row["drug"]] for row in per_target_rows])
    min_val = min(float(np.min(null_values)), min(observed))
    max_val = max(float(np.max(null_values)), max(observed))
    span = max(max_val - min_val, 1.0)
    pad = 0.05 * span
    return min_val - pad, max_val + pad


def make_mfe_histogram_svg(
    per_target_rows: list[dict[str, str]],
    null_by_drug: dict[str, np.ndarray],
    path: Path,
    bins: int,
) -> None:
    width, height = 1320, 880
    cols, rows = 3, 2
    panel_w, panel_h = 400, 360
    gap_x, gap_y = 30, 40
    left_margin, top_margin = 40, 40
    x_min, x_max = global_mfe_range(per_target_rows, null_by_drug)
    svg: list[str] = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for idx, row in enumerate(per_target_rows):
        drug = row["drug"]
        panel_col = idx % cols
        panel_row = idx // cols
        x0 = left_margin + panel_col * (panel_w + gap_x)
        y0 = top_margin + panel_row * (panel_h + gap_y)
        plot_x = x0 + 55
        plot_y = y0 + 40
        plot_w = panel_w - 90
        plot_h = panel_h - 100

        null_values = null_by_drug[drug]
        observed = float(row["mfe_obs"])
        p_value = float(row["p_mfe"])
        counts, edges = np.histogram(null_values, bins=bins, range=(x_min, x_max))
        y_max = int(max(counts.max(), 1))

        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="none" stroke="#d0d0d0"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{plot_y + plot_h:.2f}" x2="{plot_x + plot_w:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>')

        for bar_idx, count in enumerate(counts):
            frac_left = (edges[bar_idx] - x_min) / (x_max - x_min)
            frac_right = (edges[bar_idx + 1] - x_min) / (x_max - x_min)
            x_left = plot_x + plot_w * frac_left
            x_right = plot_x + plot_w * frac_right
            bar_w = max(1.0, x_right - x_left - 1.0)
            bar_h = 0.0 if y_max == 0 else plot_h * (count / y_max)
            svg.append(
                f'<rect x="{x_left:.2f}" y="{plot_y + plot_h - bar_h:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" '
                f'fill="#bdbdbd" stroke="#7a7a7a" stroke-width="0.5"/>'
            )

        obs_x = plot_x + plot_w * ((observed - x_min) / (x_max - x_min))
        svg.append(f'<line x1="{obs_x:.2f}" y1="{plot_y:.2f}" x2="{obs_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#d62728" stroke-width="3"/>')

        for tick in np.linspace(x_min, x_max, 4):
            tx = plot_x + plot_w * ((tick - x_min) / (x_max - x_min))
            svg.append(f'<line x1="{tx:.2f}" y1="{plot_y + plot_h:.2f}" x2="{tx:.2f}" y2="{plot_y + plot_h + 6:.2f}" stroke="#222222" stroke-width="1"/>')
            draw_svg_text(svg, tx, plot_y + plot_h + 22, f"{tick:.1f}", size=11, anchor="middle")

        for tick_value in [0, y_max // 2, y_max]:
            ty = plot_y + plot_h - (0 if y_max == 0 else plot_h * (tick_value / y_max))
            svg.append(f'<line x1="{plot_x - 6:.2f}" y1="{ty:.2f}" x2="{plot_x:.2f}" y2="{ty:.2f}" stroke="#222222" stroke-width="1"/>')
            draw_svg_text(svg, plot_x - 10, ty + 4, str(tick_value), size=11, anchor="end")

        draw_svg_text(svg, x0 + panel_w / 2, y0 + 24, drug, size=16, anchor="middle")
        draw_svg_text(svg, x0 + panel_w / 2, y0 + panel_h - 12, "MFE (kcal/mol)", size=12, anchor="middle")
        draw_svg_text(svg, x0 + 18, y0 + panel_h / 2, "Count", size=12, anchor="middle", extra='transform="rotate(-90 {:.2f} {:.2f})"'.format(x0 + 18, y0 + panel_h / 2))
        draw_svg_text(svg, x0 + panel_w - 12, y0 + 26, f"p = {format_sig(p_value)}", size=13, anchor="end", fill="#d62728")

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def make_mfe_summary_svg(
    per_target_rows: list[dict[str, str]],
    null_by_drug: dict[str, np.ndarray],
    path: Path,
) -> None:
    width, height = 1180, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 50, 100
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]

    x_min, x_max = global_mfe_range(per_target_rows, null_by_drug)
    all_null = np.concatenate([null_by_drug[row["drug"]] for row in per_target_rows])
    grand_mean = float(np.mean(all_null))

    def y_to_px(value: float) -> float:
        frac = (value - x_min) / (x_max - x_min)
        return plot_y + plot_h * (1.0 - frac)

    svg: list[str] = [
        f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{plot_x:.2f}" y1="{plot_y + plot_h:.2f}" x2="{plot_x + plot_w:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>',
        f'<line x1="{plot_x:.2f}" y1="{plot_y:.2f}" x2="{plot_x:.2f}" y2="{plot_y + plot_h:.2f}" stroke="#222222" stroke-width="1.5"/>',
    ]

    for tick in np.linspace(x_min, x_max, 5):
        ty = y_to_px(tick)
        svg.append(f'<line x1="{plot_x - 6:.2f}" y1="{ty:.2f}" x2="{plot_x:.2f}" y2="{ty:.2f}" stroke="#222222" stroke-width="1"/>')
        svg.append(f'<line x1="{plot_x:.2f}" y1="{ty:.2f}" x2="{plot_x + plot_w:.2f}" y2="{ty:.2f}" stroke="#eeeeee" stroke-width="1"/>')
        draw_svg_text(svg, plot_x - 10, ty + 4, f"{tick:.1f}", size=11, anchor="end")

    mean_y = y_to_px(grand_mean)
    svg.append(f'<line x1="{plot_x:.2f}" y1="{mean_y:.2f}" x2="{plot_x + plot_w:.2f}" y2="{mean_y:.2f}" stroke="#888888" stroke-dasharray="6,4" stroke-width="2"/>')
    draw_svg_text(svg, plot_x + plot_w - 4, mean_y - 8, f"Mean null MFE = {format_sig(grand_mean)}", size=12, anchor="end", fill="#666666")

    x_positions = np.linspace(plot_x + 90, plot_x + plot_w - 90, len(per_target_rows))
    box_width = 54
    for idx, row in enumerate(per_target_rows):
        drug = row["drug"]
        values = null_by_drug[drug]
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        vmin, vmax = float(np.min(values)), float(np.max(values))
        obs = float(row["mfe_obs"])
        color = colors[idx % len(colors)]
        x = x_positions[idx]

        svg.append(f'<line x1="{x:.2f}" y1="{y_to_px(vmin):.2f}" x2="{x:.2f}" y2="{y_to_px(vmax):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x - 14:.2f}" y1="{y_to_px(vmin):.2f}" x2="{x + 14:.2f}" y2="{y_to_px(vmin):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(f'<line x1="{x - 14:.2f}" y1="{y_to_px(vmax):.2f}" x2="{x + 14:.2f}" y2="{y_to_px(vmax):.2f}" stroke="#666666" stroke-width="1.5"/>')
        svg.append(
            f'<rect x="{x - box_width / 2:.2f}" y="{y_to_px(q3):.2f}" width="{box_width:.2f}" '
            f'height="{max(1.0, y_to_px(q1) - y_to_px(q3)):.2f}" fill="#d9d9d9" stroke="#666666" stroke-width="1.5"/>'
        )
        svg.append(f'<line x1="{x - box_width / 2:.2f}" y1="{y_to_px(med):.2f}" x2="{x + box_width / 2:.2f}" y2="{y_to_px(med):.2f}" stroke="#333333" stroke-width="2"/>')
        svg.append(f'<circle cx="{x:.2f}" cy="{y_to_px(obs):.2f}" r="5.5" fill="{color}" stroke="#222222" stroke-width="1"/>')
        draw_svg_text(svg, x, plot_y + plot_h + 26, drug, size=12, anchor="middle", extra='transform="rotate(25 {:.2f} {:.2f})"'.format(x, plot_y + plot_h + 26))

    draw_svg_text(svg, width / 2, 28, "Observed MFE vs Null Distributions", size=18, anchor="middle")
    draw_svg_text(svg, width / 2, height - 24, "FDA Target", size=13, anchor="middle")
    draw_svg_text(svg, 26, height / 2, "MFE (kcal/mol)", size=13, anchor="middle", extra='transform="rotate(-90 26 {:.2f})"'.format(height / 2))
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    per_target_rows = load_delimited_rows(args.per_target_pvalues_path)
    null_rows = load_delimited_rows(args.null_stats_path)

    null_by_drug: dict[str, list[float]] = {row["drug"]: [] for row in per_target_rows}
    for row in null_rows:
        null_by_drug[row["target_id"]].append(float(row["mfe"]))
    null_by_drug_np = {drug: np.asarray(values, dtype=float) for drug, values in null_by_drug.items()}

    make_mfe_histogram_svg(per_target_rows, null_by_drug_np, args.histogram_figure_path, args.hist_bins)
    make_mfe_summary_svg(per_target_rows, null_by_drug_np, args.summary_figure_path)
    print(f"Saved MFE histogram figure to {args.histogram_figure_path}")
    print(f"Saved MFE summary figure to {args.summary_figure_path}")


if __name__ == "__main__":
    main()
