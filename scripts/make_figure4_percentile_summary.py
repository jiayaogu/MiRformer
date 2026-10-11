#!/usr/bin/env python3
"""Create the publication-ready percentile summary SVG for the 6 FDA siRNA drugs."""

from __future__ import annotations

from pathlib import Path

from Global_parameters import PROJ_HOME


OUTPUT_PATH = Path(PROJ_HOME) / "siRNA_targets_data" / "figure4_percentile_summary.svg"
FONT_FAMILY = "Arial, Helvetica, 'DejaVu Sans', sans-serif"

RED = "#D62728"
BLUE = "#1F77B4"
GREEN = "#2E7D32"
DARK_RED = "#C62828"
GREY = "#888888"
GRID = "#DDDDDD"
LEFT_FILL = "#E8F1F8"
TOP_FILL = "#E8F5E9"
BOTTOM_FILL = "#FFEBEE"

WIDTH = 700
HEIGHT = 500
PLOT_LEFT = 78
PLOT_RIGHT = 36
PLOT_TOP = 24
PLOT_BOTTOM = 142
PLOT_WIDTH = WIDTH - PLOT_LEFT - PLOT_RIGHT
PLOT_HEIGHT = HEIGHT - PLOT_TOP - PLOT_BOTTOM

X_MIN = -0.6
X_MAX = 5.6

DRUGS = [
    {"drug": "Lumasiran", "x": 0.0, "center_hit": 1, "jaccard": 90.3, "mfe_strength": 57.8},
    {"drug": "Vutrisiran", "x": 1.0, "center_hit": 1, "jaccard": 82.0, "mfe_strength": 70.4},
    {"drug": "Nedosiran", "x": 2.0, "center_hit": 1, "jaccard": 80.6, "mfe_strength": 69.0},
    {"drug": "Patisiran", "x": 3.0, "center_hit": 0, "jaccard": 44.8, "mfe_strength": 86.3},
    {"drug": "Givosiran", "x": 4.0, "center_hit": 1, "jaccard": 51.5, "mfe_strength": 51.8},
    {"drug": "Inclisiran", "x": 5.0, "center_hit": 0, "jaccard": 16.2, "mfe_strength": 39.9},
]


def x_to_px(x_value: float) -> float:
    return PLOT_LEFT + (x_value - X_MIN) / (X_MAX - X_MIN) * PLOT_WIDTH


def y_to_px(y_value: float) -> float:
    return PLOT_TOP + (100.0 - y_value) / 100.0 * PLOT_HEIGHT


def text(x: float, y: float, label: str, size: int = 14, anchor: str = "start", fill: str = "#111111", weight: str = "normal", extra: str = "") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT_FAMILY}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" {extra}>{label}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#000000", width: float = 1.0, dash: str | None = None, opacity: float | None = None) -> str:
    attrs = [
        f'x1="{x1:.2f}"',
        f'y1="{y1:.2f}"',
        f'x2="{x2:.2f}"',
        f'y2="{y2:.2f}"',
        f'stroke="{stroke}"',
        f'stroke-width="{width:.2f}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'stroke-opacity="{opacity:.2f}"')
    return f"<line {' '.join(attrs)}/>"


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str | None = None, stroke_width: float = 0.0, fill_opacity: float | None = None) -> str:
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'width="{width:.2f}"',
        f'height="{height:.2f}"',
        f'fill="{fill}"',
    ]
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width:.2f}"')
    if fill_opacity is not None:
        attrs.append(f'fill-opacity="{fill_opacity:.2f}"')
    return f"<rect {' '.join(attrs)}/>"


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "#000000", stroke_width: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
    )


def main() -> None:
    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="7in" height="5in" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, fill="white"),
    ]

    # Background bands
    elements.append(rect(PLOT_LEFT, y_to_px(100), PLOT_WIDTH, y_to_px(70) - y_to_px(100), fill=TOP_FILL, fill_opacity=0.3))
    elements.append(rect(PLOT_LEFT, y_to_px(30), PLOT_WIDTH, y_to_px(0) - y_to_px(30), fill=BOTTOM_FILL, fill_opacity=0.3))

    # Horizontal grid lines
    for tick in [0, 25, 50, 75, 100]:
        y = y_to_px(tick)
        elements.append(line(PLOT_LEFT, y, PLOT_LEFT + PLOT_WIDTH, y, stroke=GRID, width=1.0, opacity=0.5))

    # Minor ticks
    for tick in range(0, 101, 10):
        y = y_to_px(float(tick))
        elements.append(line(PLOT_LEFT - 4, y, PLOT_LEFT, y, stroke="#000000", width=0.8))

    # Main axes
    elements.append(line(PLOT_LEFT, PLOT_TOP, PLOT_LEFT, PLOT_TOP + PLOT_HEIGHT, stroke="#000000", width=1.2))
    elements.append(line(PLOT_LEFT, PLOT_TOP + PLOT_HEIGHT, PLOT_LEFT + PLOT_WIDTH, PLOT_TOP + PLOT_HEIGHT, stroke="#000000", width=1.2))

    # Major tick labels
    for tick in [0, 25, 50, 75, 100]:
        y = y_to_px(float(tick))
        elements.append(line(PLOT_LEFT - 6, y, PLOT_LEFT, y, stroke="#000000", width=1.0))
        elements.append(text(PLOT_LEFT - 10, y + 4, str(tick), size=10, anchor="end"))

    # Median line
    median_y = y_to_px(50)
    elements.append(line(PLOT_LEFT, median_y, PLOT_LEFT + PLOT_WIDTH, median_y, stroke=GREY, width=1.0, dash="6,4"))
    elements.append(text(PLOT_LEFT + PLOT_WIDTH - 4, median_y - 6, "Null median", size=10, anchor="end", fill=GREY))

    # Band labels
    elements.append(text(PLOT_LEFT + PLOT_WIDTH - 4, y_to_px(85), "Top 30%", size=10, anchor="end", fill=GREEN))
    elements.append(text(PLOT_LEFT + PLOT_WIDTH - 4, y_to_px(15), "Bottom 30%", size=10, anchor="end", fill=DARK_RED))

    # Data points
    marker_radius = 7.5
    x_offset = 0.12
    for row in DRUGS:
        x_center = x_to_px(row["x"])
        x_red = x_to_px(row["x"] - x_offset)
        x_blue = x_to_px(row["x"] + x_offset)
        y_red = y_to_px(row["jaccard"])
        y_blue = y_to_px(row["mfe_strength"])

        elements.append(line(x_center, y_red, x_center, y_blue, stroke=GREY, width=1.0, opacity=0.3))
        elements.append(circle(x_red, y_red, marker_radius, RED))
        elements.append(circle(x_blue, y_blue, marker_radius, BLUE))

        elements.append(text(x_red - 8, y_red - 10, f"{int(round(row['jaccard']))}%", size=9, anchor="end", fill=RED))
        elements.append(text(x_blue + 8, y_blue - 10, f"{int(round(row['mfe_strength']))}%", size=9, anchor="start", fill=BLUE))

        tick_x = x_to_px(row["x"])
        axis_y = PLOT_TOP + PLOT_HEIGHT
        elements.append(line(tick_x, axis_y, tick_x, axis_y + 6, stroke="#000000", width=1.0))
        elements.append(text(tick_x, axis_y + 26, row["drug"], size=10, anchor="middle"))
        mark = "✓" if row["center_hit"] == 1 else "✗"
        mark_color = GREEN if row["center_hit"] == 1 else DARK_RED
        elements.append(text(tick_x, axis_y + 48, mark, size=14, anchor="middle", fill=mark_color, weight="bold"))

    # Axis label
    axis_label_x = 22
    axis_label_y = PLOT_TOP + PLOT_HEIGHT / 2
    elements.append(
        text(
            axis_label_x,
            axis_label_y,
            "Percentile in cross-target null distribution",
            size=12,
            anchor="middle",
            extra=f'transform="rotate(-90 {axis_label_x:.2f} {axis_label_y:.2f})"',
        )
    )

    # Legend
    legend_x = PLOT_LEFT + 14
    legend_y = PLOT_TOP + PLOT_HEIGHT - 86
    legend_w = 230
    legend_h = 74
    elements.append(rect(legend_x, legend_y, legend_w, legend_h, fill="white", stroke="#CCCCCC", stroke_width=1.0))
    elements.append(circle(legend_x + 16, legend_y + 20, 6.5, RED))
    elements.append(circle(legend_x + 120, legend_y + 20, 6.5, BLUE))
    elements.append(text(legend_x + 30, legend_y + 24, "Region Jaccard", size=11))
    elements.append(text(legend_x + 134, legend_y + 24, "MFE strength", size=11))
    elements.append(text(legend_x + 16, legend_y + 52, "✓", size=14, anchor="middle", fill=GREEN, weight="bold"))
    elements.append(text(legend_x + 120, legend_y + 52, "✗", size=14, anchor="middle", fill=DARK_RED, weight="bold"))
    elements.append(text(legend_x + 30, legend_y + 56, "center-hit", size=11))
    elements.append(text(legend_x + 134, legend_y + 56, "non-center-hit", size=11))

    elements.append("</svg>")
    OUTPUT_PATH.write_text("\n".join(elements), encoding="utf-8")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
