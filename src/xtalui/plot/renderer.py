"""Terminal rendering for 2D scatter and line plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from xtalui.renderer import (
    BRAILLE_BASE,
    BRAILLE_DOTS,
    HORIZONTAL_EDGE,
    VERTICAL_EDGE,
    Viewport,
)

ScaleType = Literal["linear", "log"]
PlotMode = Literal["scatter", "line", "both"]

TextFragment = tuple[str, str]

SERIES_COLORS = [
    "fg:#4fc3f7",
    "fg:#ff8a65",
    "fg:#81c784",
    "fg:#ce93d8",
    "fg:#fff176",
    "fg:#4dd0e1",
    "fg:#f48fb1",
    "fg:#a5d6a7",
]

_LOG_EPSILON = 1e-300


@dataclass(frozen=True)
class PlotBounds:
    """Data-range bounds for the plot area."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class PlotLayout:
    """Computed pixel regions within the terminal viewport."""

    plot_left: int
    plot_right: int
    plot_top: int
    plot_bottom: int
    y_label_width: int
    x_label_height: int
    title_height: int


@dataclass(frozen=True)
class TickMark:
    """A single tick mark on an axis."""

    value: float
    label: str
    position: float


def _apply_scale(value: float, scale: ScaleType) -> float:
    if scale == "log":
        return math.log10(max(value, _LOG_EPSILON))
    return value


def _nice_tick_values(data_min: float, data_max: float, max_ticks: int = 8) -> list[float]:
    """Generate aesthetically spaced tick values using the 1-2-5 algorithm."""
    if data_min >= data_max:
        if data_min == 0.0:
            return [0.0]
        margin = abs(data_min) * 0.1 or 0.5
        data_min -= margin
        data_max += margin
    range_val = data_max - data_min
    rough_step = range_val / max(max_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(max(rough_step, _LOG_EPSILON)))
    residual = rough_step / magnitude
    if residual <= 1.5:
        nice_step = 1 * magnitude
    elif residual <= 3.0:
        nice_step = 2 * magnitude
    elif residual <= 7.0:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude
    start = math.floor(data_min / nice_step) * nice_step
    ticks: list[float] = []
    value = start
    while value <= data_max + nice_step * 0.01:
        ticks.append(value)
        value += nice_step
    return ticks


def _format_tick(value: float, step: float) -> str:
    """Format a tick value with appropriate precision."""
    if step >= 1:
        return f"{value:.0f}"
    if step <= 0:
        return f"{value:.6g}"
    decimal_places = max(0, -int(math.floor(math.log10(max(step, _LOG_EPSILON)))))
    return f"{value:.{decimal_places}f}"


def _data_to_pixel(
    x: float,
    y: float,
    bounds: PlotBounds,
    layout: PlotLayout,
) -> tuple[float, float]:
    """Map a data point to sub-pixel (col, row) in the plot area.

    The mapping uses the same convention as tick positions:
    x: left edge = x_min, right edge = x_max
    y: bottom edge = y_min, top edge = y_max
    """
    plot_width = layout.plot_right - layout.plot_left
    plot_height = layout.plot_bottom - layout.plot_top
    if bounds.x_max <= bounds.x_min:
        col = layout.plot_left + plot_width / 2.0
    else:
        frac_x = (x - bounds.x_min) / (bounds.x_max - bounds.x_min)
        col = layout.plot_left + frac_x * plot_width
    if bounds.y_max <= bounds.y_min:
        row = layout.plot_top + plot_height / 2.0
    else:
        frac_y = (y - bounds.y_min) / (bounds.y_max - bounds.y_min)
        row = layout.plot_bottom - 1 - frac_y * (plot_height - 1)
    return col, row


def _compute_layout(viewport: Viewport, y_ticks: list[TickMark], x_ticks: list[TickMark], title: str) -> PlotLayout:
    """Compute the pixel layout for the plot area within the viewport."""
    y_label_width = 2
    if y_ticks:
        y_label_width = max(len(t.label) for t in y_ticks) + 2
    y_label_width = max(y_label_width, 4)
    x_label_height = 2 if x_ticks else 1
    title_height = 1 if title else 0
    plot_left = y_label_width
    plot_right = viewport.width
    plot_top = title_height
    plot_bottom = viewport.height - x_label_height
    if plot_bottom <= plot_top:
        plot_bottom = plot_top + 1
    return PlotLayout(
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        y_label_width=y_label_width,
        x_label_height=x_label_height,
        title_height=title_height,
    )


def _line_points_bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int, float]]:
    """Bresenham line algorithm returning (x, y, t) points with parametric t."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points: list[tuple[int, int, float]] = []
    steps = max(dx, dy, 1)
    i = 0
    while True:
        t = i / steps if steps else 0.0
        points.append((x0, y0, t))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        i += 1
    return points


def _braille_scatter(
    width: int,
    height: int,
    points: list[tuple[float, float]],
    colors: list[str],
    bounds: PlotBounds,
    layout: PlotLayout,
) -> tuple[list[str], list[list[str]]]:
    """Render scatter points as Braille dots."""
    masks = [[0] * width for _ in range(height)]
    styles = [["" for _ in range(width)] for _ in range(height)]

    for (x, y), color in zip(points, colors):
        col, row = _data_to_pixel(x, y, bounds, layout)
        col -= layout.plot_left
        row -= layout.plot_top
        sub_x = int(round(col * 2))
        sub_y = int(round(row * 4))

        cell_x = sub_x // 2
        cell_y = sub_y // 4
        dot_x = sub_x % 2
        dot_y = sub_y % 4
        if 0 <= cell_x < width and 0 <= cell_y < height:
            masks[cell_y][cell_x] |= BRAILLE_DOTS[(dot_x, dot_y)]
            if not styles[cell_y][cell_x]:
                styles[cell_y][cell_x] = color

    rows: list[str] = []
    for cy in range(height):
        row_chars = []
        for cx in range(width):
            mask = masks[cy][cx]
            row_chars.append(chr(BRAILLE_BASE + mask) if mask else " ")
        rows.append("".join(row_chars))
    return rows, styles


def _braille_lines(
    width: int,
    height: int,
    series_points: list[list[tuple[float, float]]],
    colors: list[str],
    bounds: PlotBounds,
    layout: PlotLayout,
) -> tuple[list[str], list[list[str]]]:
    """Render line series using Braille sub-pixel Bresenham."""
    masks = [[0] * width for _ in range(height)]
    styles = [["" for _ in range(width)] for _ in range(height)]
    for points, color in zip(series_points, colors):
        if len(points) < 2:
            # Single point — render as a dot.
            if points:
                x, y = points[0]
                col, row = _data_to_pixel(x, y, bounds, layout)
                col -= layout.plot_left
                row -= layout.plot_top
                sub_x = int(round(col * 2))
                sub_y = int(round(row * 4))
                cell_x = sub_x // 2
                cell_y = sub_y // 4
                dot_x = sub_x % 2
                dot_y = sub_y % 2
                if 0 <= cell_x < width and 0 <= cell_y < height:
                    masks[cell_y][cell_x] |= BRAILLE_DOTS[(dot_x, dot_y)]
                    if not styles[cell_y][cell_x]:
                        styles[cell_y][cell_x] = color
            continue
        # Convert to pixel coordinates.
        pixel_points: list[tuple[float, float]] = []
        for x, y in points:
            col, row = _data_to_pixel(x, y, bounds, layout)
            pixel_points.append((col - layout.plot_left, row - layout.plot_top))
        # Draw line segments on the Braille sub-grid.
        for i in range(len(pixel_points) - 1):
            x0, y0 = pixel_points[i]
            x1, y1 = pixel_points[i + 1]
            sub_start = (int(round(x0 * 2)), int(round(y0 * 4)))
            sub_end = (int(round(x1 * 2)), int(round(y1 * 4)))
            for sx, sy, _ in _line_points_bresenham(sub_start, sub_end):
                cell_x = sx // 2
                cell_y = sy // 4
                dot_x = sx % 2
                dot_y = sy % 4
                if 0 <= cell_x < width and 0 <= cell_y < height:
                    masks[cell_y][cell_x] |= BRAILLE_DOTS[(dot_x, dot_y)]
                    if not styles[cell_y][cell_x]:
                        styles[cell_y][cell_x] = color
    rows: list[str] = []
    for cy in range(height):
        row_chars = []
        for cx in range(width):
            mask = masks[cy][cx]
            row_chars.append(chr(BRAILLE_BASE + mask) if mask else " ")
        rows.append("".join(row_chars))
    return rows, styles


def _render_legend(
    series_list: list,
    series_colors: list[str],
    show_color: bool,
    layout: PlotLayout,
    viewport_width: int,
    set_char: Callable[[int, int, str, str], None],
) -> None:
    """Overlay a compact legend in the top-right corner of the plot area."""
    if len(series_list) < 2:
        return

    # Compute legend dimensions.
    max_name_width = min(max(len(s.name) for s in series_list), 20)
    box_inner_width = max_name_width + 4  # "■ " + name + padding
    box_width = box_inner_width + 2  # +2 for border chars
    n_entries = min(len(series_list), (layout.plot_bottom - layout.plot_top) // 2)
    box_height = n_entries + 2  # +2 for top/bottom border

    # Position: top-right of plot area with 2-cell padding.
    box_right = min(layout.plot_right - 2, viewport_width - 1)
    box_left = box_right - box_width + 1
    box_top = layout.plot_top + 1
    box_bottom = box_top + box_height - 1

    if box_left < layout.plot_left + 2:
        return  # Not enough room.

    # Draw border.
    set_char(box_top, box_left, "┌")
    set_char(box_top, box_right, "┐")
    set_char(box_bottom, box_left, "└")
    set_char(box_bottom, box_right, "┘")
    for cx in range(box_left + 1, box_right):
        set_char(box_top, cx, "─")
        set_char(box_bottom, cx, "─")
    for ry in range(box_top + 1, box_bottom):
        set_char(ry, box_left, "│")
        set_char(ry, box_right, "│")
        # Clear interior to prevent data from showing through.
        for cx in range(box_left + 1, box_right):
            set_char(ry, cx, " ")

    # Draw entries.
    for i in range(n_entries):
        name = series_list[i].name[:max_name_width]
        color = series_colors[i] if show_color else ""
        row = box_top + 1 + i
        # Swatch.
        set_char(row, box_left + 2, "■", color)
        # Name.
        for j, ch in enumerate(name):
            set_char(row, box_left + 4 + j, ch, color)


def render_plot(
    series_list: list,
    viewport: Viewport,
    x_column_name: str,
    y_column_name: str,
    bounds: PlotBounds | None = None,
    show_grid: bool = True,
    show_color: bool = False,
    plot_mode: PlotMode = "scatter",
    x_scale: ScaleType = "linear",
    y_scale: ScaleType = "linear",
    show_legend: bool = False,
) -> list[TextFragment]:
    """Render a scatter/line plot as styled text fragments for prompt_toolkit."""

    if not series_list:
        return [("", "No data to plot\n")]

    all_x = np.concatenate([s.x for s in series_list])
    all_y = np.concatenate([s.y for s in series_list])

    # Handle log scale: filter non-positive values for tick computation.
    if x_scale == "log":
        valid_x = all_x[all_x > 0]
        if len(valid_x) == 0:
            return [("", "Cannot use log scale: all x values <= 0\n")]
    else:
        valid_x = all_x
    if y_scale == "log":
        valid_y = all_y[all_y > 0]
        if len(valid_y) == 0:
            return [("", "Cannot use log scale: all y values <= 0\n")]
    else:
        valid_y = all_y

    if bounds is None:
        x_margin = (float(np.max(valid_x)) - float(np.min(valid_x))) * 0.05
        if x_margin == 0:
            x_margin = 0.5
        y_margin = (float(np.max(valid_y)) - float(np.min(valid_y))) * 0.05
        if y_margin == 0:
            y_margin = 0.5
        bounds = PlotBounds(
            x_min=float(np.min(valid_x)) - x_margin,
            x_max=float(np.max(valid_x)) + x_margin,
            y_min=float(np.min(valid_y)) - y_margin,
            y_max=float(np.max(valid_y)) + y_margin,
        )

    # Compute tick marks.
    # Bounds are always in display space (log-transformed when log scale is active).
    # Generate linear ticks on the display-space bounds.
    x_tick_values = _nice_tick_values(bounds.x_min, bounds.x_max)
    x_step = x_tick_values[1] - x_tick_values[0] if len(x_tick_values) >= 2 else 1.0
    y_tick_values = _nice_tick_values(bounds.y_min, bounds.y_max)
    y_step = y_tick_values[1] - y_tick_values[0] if len(y_tick_values) >= 2 else 1.0

    def x_tick_pos(v: float) -> float:
        if bounds.x_max <= bounds.x_min:
            return 0.0
        return (v - bounds.x_min) / (bounds.x_max - bounds.x_min)

    def y_tick_pos(v: float) -> float:
        if bounds.y_max <= bounds.y_min:
            return 0.0
        return (v - bounds.y_min) / (bounds.y_max - bounds.y_min)

    def _format_log_axis_label(v: float) -> str:
        """Format a log-axis tick label: display 10^v."""
        if abs(v) > 300:
            return f"1e{v:+.0f}"
        val = 10**v
        if abs(val) >= 1e6 or (abs(val) < 1e-3 and val != 0):
            return f"{val:.2e}"
        if val == int(val):
            return str(int(val))
        return f"{val:.4g}"

    x_ticks = [
        TickMark(
            value=v,
            label=_format_log_axis_label(v) if x_scale == "log" else _format_tick(v, x_step),
            position=x_tick_pos(v),
        )
        for v in x_tick_values
    ]
    y_ticks = [
        TickMark(
            value=v,
            label=_format_log_axis_label(v) if y_scale == "log" else _format_tick(v, y_step),
            position=y_tick_pos(v),
        )
        for v in y_tick_values
    ]

    title = f"{x_column_name} vs {y_column_name}"
    layout = _compute_layout(viewport, y_ticks, x_ticks, title)

    plot_width = layout.plot_right - layout.plot_left
    plot_height = layout.plot_bottom - layout.plot_top

    # Initialize character and style buffers.
    buffer = [[" " for _ in range(viewport.width)] for _ in range(viewport.height)]
    styles = [["" for _ in range(viewport.width)] for _ in range(viewport.height)]

    def set_char(row: int, col: int, char: str, style: str = "") -> None:
        if 0 <= row < viewport.height and 0 <= col < viewport.width:
            buffer[row][col] = char
            if style:
                styles[row][col] = style

    # Draw title.
    if title:
        title_col = max((viewport.width - len(title)) // 2, 0)
        for i, ch in enumerate(title):
            set_char(0, title_col + i, ch)

    # Draw plot area border.
    for col in range(layout.plot_left, layout.plot_right):
        set_char(layout.plot_top, col, HORIZONTAL_EDGE)
        set_char(layout.plot_bottom, col, HORIZONTAL_EDGE)
    for row in range(layout.plot_top, layout.plot_bottom + 1):
        set_char(row, layout.plot_left, VERTICAL_EDGE)
        # Right edge is optional — omit if it's the viewport edge.
        if layout.plot_right < viewport.width:
            set_char(row, layout.plot_right, VERTICAL_EDGE)
    # Corners.
    set_char(layout.plot_top, layout.plot_left, "┌")
    set_char(layout.plot_top, layout.plot_right, "┐")
    set_char(layout.plot_bottom, layout.plot_left, "└")
    set_char(layout.plot_bottom, layout.plot_right, "┘")

    # Draw grid lines and tick marks.
    if show_grid:
        for tick in x_ticks:
            col = layout.plot_left + int(tick.position * plot_width)
            if layout.plot_left < col < layout.plot_right:
                for row in range(layout.plot_top + 1, layout.plot_bottom):
                    set_char(row, col, "┊")
        for tick in y_ticks:
            row = layout.plot_bottom - 1 - int(tick.position * (plot_height - 1))
            if layout.plot_top < row < layout.plot_bottom:
                for col in range(layout.plot_left + 1, layout.plot_right):
                    set_char(row, col, "─")

    # Draw tick labels on y-axis (right-aligned to the left of the plot area).
    for tick in y_ticks:
        row = layout.plot_bottom - 1 - int(tick.position * (plot_height - 1))
        if layout.plot_top <= row <= layout.plot_bottom:
            label_col = layout.plot_left - len(tick.label) - 1
            for i, ch in enumerate(tick.label):
                set_char(row, label_col + i, ch)
            # Tick mark on the axis.
            set_char(row, layout.plot_left - 1, "┤")

    # Draw tick labels on x-axis (centered below the plot area).
    for tick in x_ticks:
        col = layout.plot_left + int(tick.position * plot_width)
        if layout.plot_left <= col <= layout.plot_right:
            label_row = layout.plot_bottom + 1
            label_col = col - len(tick.label) // 2
            for i, ch in enumerate(tick.label):
                set_char(label_row, label_col + i, ch)
            # Tick mark on the axis.
            set_char(layout.plot_bottom, col, "┬")

    # Draw data points using Braille.
    # Map data through scale transform before rendering.
    scaled_series_points: list[list[tuple[float, float]]] = []
    series_colors: list[str] = []
    for idx, s in enumerate(series_list):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)] if show_color else ""
        series_colors.append(color)
        pts: list[tuple[float, float]] = []
        for xv, yv in zip(s.x, s.y):
            sx = _apply_scale(float(xv), x_scale)
            sy = _apply_scale(float(yv), y_scale)
            pts.append((sx, sy))
        scaled_series_points.append(pts)

    if plot_width > 0 and plot_height > 0:
        if plot_mode == "scatter":
            all_pts: list[tuple[float, float]] = []
            all_colors: list[str] = []
            for pts, color in zip(scaled_series_points, series_colors):
                all_pts.extend(pts)
                all_colors.extend([color] * len(pts))
            braille_rows, braille_styles = _braille_scatter(
                plot_width, plot_height, all_pts, all_colors, bounds, layout
            )
        elif plot_mode == "both":
            # Draw lines first, then overlay scatter dots on top.
            braille_rows, braille_styles = _braille_lines(
                plot_width, plot_height, scaled_series_points, series_colors, bounds, layout
            )
            # Overlay scatter points.
            all_pts: list[tuple[float, float]] = []
            all_colors: list[str] = []
            for pts, color in zip(scaled_series_points, series_colors):
                all_pts.extend(pts)
                all_colors.extend([color] * len(pts))
            dot_rows, dot_styles = _braille_scatter(plot_width, plot_height, all_pts, all_colors, bounds, layout)
            # Merge: dot content overwrites line content.
            for cy in range(min(len(dot_rows), len(braille_rows))):
                for cx in range(min(len(dot_rows[cy]), len(braille_rows[cy]))):
                    if dot_rows[cy][cx] != " ":
                        braille_rows[cy] = braille_rows[cy][:cx] + dot_rows[cy][cx] + braille_rows[cy][cx + 1 :]
                        braille_styles[cy][cx] = dot_styles[cy][cx]
        else:
            braille_rows, braille_styles = _braille_lines(
                plot_width, plot_height, scaled_series_points, series_colors, bounds, layout
            )

        # Overlay Braille data onto the buffer.
        for by, braille_row in enumerate(braille_rows):
            screen_row = layout.plot_top + by
            if screen_row >= layout.plot_bottom:
                break
            for bx, ch in enumerate(braille_row):
                screen_col = layout.plot_left + bx
                if screen_col >= layout.plot_right:
                    break
                if ch != " ":
                    buffer[screen_row][screen_col] = ch
                    if braille_styles[by][bx]:
                        styles[screen_row][screen_col] = braille_styles[by][bx]

    # Draw legend.
    if show_legend:
        _render_legend(series_list, series_colors, show_color, layout, viewport.width, set_char)

    # Convert buffer to TextFragment list.
    fragments: list[TextFragment] = []
    for row_chars, row_styles in zip(buffer, styles):
        current_style = row_styles[0]
        current_text = [row_chars[0]]
        for char, style in zip(row_chars[1:], row_styles[1:]):
            if style == current_style:
                current_text.append(char)
                continue
            fragments.append((current_style, "".join(current_text)))
            current_style = style
            current_text = [char]
        fragments.append((current_style, "".join(current_text)))
        fragments.append(("", "\n"))
    if fragments:
        fragments.pop()
    return fragments


def render_plot_ascii(
    series_list: list,
    viewport: Viewport,
    x_column_name: str,
    y_column_name: str,
    bounds: PlotBounds | None = None,
    show_grid: bool = True,
    show_color: bool = False,
    plot_mode: PlotMode = "scatter",
    x_scale: ScaleType = "linear",
    y_scale: ScaleType = "linear",
    show_legend: bool = False,
) -> list[str]:
    """Render a plot as plain text rows (no style metadata)."""
    fragments = render_plot(
        series_list,
        viewport,
        x_column_name,
        y_column_name,
        bounds=bounds,
        show_grid=show_grid,
        show_color=show_color,
        plot_mode=plot_mode,
        x_scale=x_scale,
        y_scale=y_scale,
        show_legend=show_legend,
    )
    text = "".join(t for _, t in fragments)
    return text.split("\n")


def render_panels(
    series_groups: list[list],
    viewport: Viewport,
    x_column_name: str,
    bounds: PlotBounds | None = None,
    show_grid: bool = True,
    plot_mode: PlotMode = "scatter",
    x_scale: ScaleType = "linear",
    y_scale: ScaleType = "linear",
) -> list[TextFragment]:
    """Render multiple panels sharing the same x-axis range.

    Each group in *series_groups* is rendered as a separate panel stacked
    vertically. All panels share the same x-axis bounds and scale. Each
    panel gets its own y-axis bounds computed from its data.
    """
    if not series_groups:
        return [("", "No data to plot\n")]

    all_x = []
    for group in series_groups:
        for s in group:
            all_x.extend(s.x.tolist())
    if not all_x:
        return [("", "No data to plot\n")]

    all_x_arr = np.array(all_x)
    valid_x = all_x_arr[all_x_arr > 0] if x_scale == "log" else all_x_arr

    # Shared x bounds (in display space).
    if bounds is None:
        x_margin = (float(np.max(valid_x)) - float(np.min(valid_x))) * 0.05
        if x_margin == 0:
            x_margin = 0.5
        shared_x_min = float(np.min(valid_x)) - x_margin
        shared_x_max = float(np.max(valid_x)) + x_margin
    else:
        shared_x_min = bounds.x_min
        shared_x_max = bounds.x_max

    # Apply log transform to x bounds if needed.
    if x_scale == "log":
        shared_x_min = math.log10(max(shared_x_min, _LOG_EPSILON))
        shared_x_max = math.log10(max(shared_x_max, _LOG_EPSILON))

    n_panels = len(series_groups)
    # Reserve 1 row between panels as separator, 2 rows for shared x-axis at bottom.
    separator_rows = max(n_panels - 1, 0)
    total_height = viewport.height - separator_rows - 2  # 2 for x-axis labels
    panel_height = max(total_height // n_panels, 3)

    all_fragments: list[TextFragment] = []
    for panel_idx, group in enumerate(series_groups):
        # Compute per-panel y bounds (in display space).
        all_y = np.concatenate([s.y for s in group])
        valid_y = all_y[all_y > 0] if y_scale == "log" else all_y
        if y_scale == "log":
            valid_y = np.log10(valid_y)
        y_margin = (float(np.max(valid_y)) - float(np.min(valid_y))) * 0.05
        if y_margin == 0:
            y_margin = 0.5
        panel_bounds = PlotBounds(
            x_min=shared_x_min,
            x_max=shared_x_max,
            y_min=float(np.min(valid_y)) - y_margin,
            y_max=float(np.max(valid_y)) + y_margin,
        )

        # Determine y-column name from the first series.
        y_name = group[0].name if group else "y"
        panel_viewport = Viewport(viewport.width, panel_height)

        if panel_idx < n_panels - 1:
            # Hide x-axis labels for all but the last panel.
            panel_fragments = render_plot(
                group,
                panel_viewport,
                x_column_name,
                y_name,
                bounds=panel_bounds,
                show_grid=show_grid,
                plot_mode=plot_mode,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            all_fragments.extend(panel_fragments)
            all_fragments.append(("", "\n"))
        else:
            # Last panel: render with full x-axis labels.
            # Use a slightly taller viewport to include x-axis labels.
            last_height = viewport.height - panel_height * (n_panels - 1) - separator_rows
            last_viewport = Viewport(viewport.width, max(last_height, panel_height))
            panel_fragments = render_plot(
                group,
                last_viewport,
                x_column_name,
                y_name,
                bounds=panel_bounds,
                show_grid=show_grid,
                plot_mode=plot_mode,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            all_fragments.extend(panel_fragments)

    return all_fragments
