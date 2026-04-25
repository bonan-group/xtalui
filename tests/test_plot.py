"""Tests for the tplot data parser and renderer."""

from __future__ import annotations

import math

import numpy as np

from xtalui.plot.parser import (
    Series,
    auto_series,
    detect_numeric_columns,
    parse_text,
)
from xtalui.plot.renderer import (
    PlotBounds,
    PlotLayout,
    Viewport,
    _apply_scale,
    _data_to_pixel,
    _format_tick,
    _nice_tick_values,
    render_panels,
    render_plot,
    render_plot_ascii,
)


class TestParseText:
    def test_extracts_numeric_columns(self) -> None:
        text = (
            "Epoch: 165 20.580 3445.343 RMSE Train    0.35602 eV | Val    0.50984 eV\n"
            "Epoch: 166 20.644 3465.988 RMSE Train    0.35737 eV | Val    0.51064 eV\n"
        )
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert len(columns) >= 4

    def test_skips_comment_lines(self) -> None:
        text = "# This is a header\n1.0 2.0\n3.0 4.0\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert len(columns) == 2
        assert len(columns[0].values) == 2

    def test_handles_scientific_notation(self) -> None:
        text = "1.5e-3 2.0E+10\n4.0 5.0\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert len(columns) == 2
        assert abs(columns[0].values[0] - 1.5e-3) < 1e-10
        assert abs(columns[1].values[0] - 2.0e10) < 1e-10

    def test_handles_ragged_lines(self) -> None:
        text = "1.0 2.0 3.0\n4.0 5.0\n6.0 7.0 8.0 9.0\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        # col_0 has 3 values, col_1 has 3 values, col_2 has 2, col_3 has 1
        assert len(columns) >= 3
        assert len(columns[0].values) == 3
        assert len(columns[2].values) == 2
        assert len(columns[3].values) == 1

    def test_skips_empty_lines(self) -> None:
        text = "\n1.0 2.0\n\n3.0 4.0\n\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert len(columns) == 2
        assert len(columns[0].values) == 2

    def test_source_name_preserved(self) -> None:
        data = parse_text("1.0 2.0\n", source_name="test.log")
        assert data.source_name == "test.log"

    def test_column_naming_from_adjacent_text(self) -> None:
        text = "Train 0.35602 eV\nTrain 0.35737 eV\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert columns[0].name == "Train"

    def test_default_column_names(self) -> None:
        text = "1.0 2.0\n3.0 4.0\n"
        data = parse_text(text)
        columns = detect_numeric_columns(data)
        assert columns[0].name == "col_0"
        assert columns[1].name == "col_1"


class TestAutoSeries:
    def test_single_series_no_grouping(self) -> None:
        text = "1.0 2.0\n3.0 4.0\n5.0 6.0\n"
        data = parse_text(text)
        series_list = auto_series(data, 0, 1)
        assert len(series_list) == 1
        assert len(series_list[0].x) == 3
        assert len(series_list[0].y) == 3

    def test_detects_grouping_column(self) -> None:
        text = "Train 0.1 0.2\nTrain 0.3 0.4\nVal 0.5 0.6\nVal 0.7 0.8\n"
        data = parse_text(text)
        series_list = auto_series(data, 0, 1)
        assert len(series_list) == 2
        assert series_list[0].name == "Train"
        assert series_list[1].name == "Val"
        assert len(series_list[0].x) == 2
        assert len(series_list[1].x) == 2

    def test_empty_data_returns_empty(self) -> None:
        text = "no numeric data here\n"
        data = parse_text(text)
        series_list = auto_series(data, 0, 1)
        assert series_list == []

    def test_out_of_range_columns_returns_empty(self) -> None:
        text = "1.0 2.0\n"
        data = parse_text(text)
        series_list = auto_series(data, 5, 6)
        assert series_list == []


# --- Renderer tests ---


def _make_series(name: str, x: list[float], y: list[float]) -> Series:
    return Series(name=name, x=np.array(x), y=np.array(y))


class TestNiceTickValues:
    def test_basic_range(self) -> None:
        ticks = _nice_tick_values(0.0, 100.0, max_ticks=8)
        assert 4 <= len(ticks)
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 100.0

    def test_small_range(self) -> None:
        ticks = _nice_tick_values(0.0, 1.0, max_ticks=5)
        assert len(ticks) >= 2
        assert all(0 <= t <= 1.0 for t in ticks)

    def test_equal_bounds(self) -> None:
        ticks = _nice_tick_values(5.0, 5.0, max_ticks=5)
        assert len(ticks) >= 1

    def test_zero_range(self) -> None:
        ticks = _nice_tick_values(0.0, 0.0, max_ticks=5)
        assert len(ticks) >= 1


class TestFormatTick:
    def test_integer_step(self) -> None:
        assert _format_tick(10.0, 1.0) == "10"

    def test_decimal_step(self) -> None:
        assert _format_tick(0.05, 0.01) == "0.05"

    def test_tiny_step(self) -> None:
        result = _format_tick(1.234, 0.001)
        assert "1.234" in result


class TestApplyScale:
    def test_linear(self) -> None:
        assert _apply_scale(5.0, "linear") == 5.0

    def test_log_positive(self) -> None:
        assert abs(_apply_scale(100.0, "log") - 2.0) < 1e-10

    def test_log_zero(self) -> None:
        result = _apply_scale(0.0, "log")
        assert result == math.log10(1e-300)


class TestDataToPixel:
    def test_maps_bottom_left(self) -> None:
        bounds = PlotBounds(0.0, 10.0, 0.0, 10.0)
        layout = PlotLayout(
            plot_left=5, plot_right=55, plot_top=2, plot_bottom=22, y_label_width=5, x_label_height=2, title_height=1
        )
        col, row = _data_to_pixel(0.0, 0.0, bounds, layout)
        assert col == 5.0
        assert row == 21.0  # bottom edge (plot_bottom - 1)

    def test_maps_top_right(self) -> None:
        bounds = PlotBounds(0.0, 10.0, 0.0, 10.0)
        layout = PlotLayout(
            plot_left=5, plot_right=55, plot_top=2, plot_bottom=22, y_label_width=5, x_label_height=2, title_height=1
        )
        col, row = _data_to_pixel(10.0, 10.0, bounds, layout)
        assert col == 55.0
        assert row == 2.0  # top edge


class TestRenderPlot:
    def test_produces_non_empty_output(self) -> None:
        series = [_make_series("test", [1.0, 2.0, 3.0], [1.0, 4.0, 2.0])]
        fragments = render_plot(series, Viewport(60, 20), "x", "y")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0

    def test_contains_axis_labels(self) -> None:
        series = [_make_series("test", [1.0, 2.0, 3.0], [1.0, 4.0, 2.0])]
        fragments = render_plot(series, Viewport(60, 20), "x_col", "y_col")
        text = "".join(t for _, t in fragments)
        assert "x_col" in text or "y_col" in text

    def test_multiple_series_have_colors(self) -> None:
        series = [
            _make_series("a", [1.0, 5.0], [1.0, 5.0]),
            _make_series("b", [2.0, 6.0], [3.0, 7.0]),
        ]
        fragments = render_plot(series, Viewport(60, 20), "x", "y", show_color=True)
        styles = {style for style, text in fragments if style and text.strip()}
        assert len(styles) >= 2

    def test_line_mode(self) -> None:
        series = [_make_series("test", [1.0, 2.0, 3.0], [1.0, 4.0, 2.0])]
        fragments = render_plot(series, Viewport(60, 20), "x", "y", plot_mode="line")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0

    def test_log_scale(self) -> None:
        series = [_make_series("test", [1.0, 10.0, 100.0], [1.0, 100.0, 10000.0])]
        fragments = render_plot(series, Viewport(60, 20), "x", "y", x_scale="log", y_scale="log")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0

    def test_empty_series_returns_message(self) -> None:
        fragments = render_plot([], Viewport(60, 20), "x", "y")
        text = "".join(t for _, t in fragments)
        assert "No data" in text

    def test_custom_bounds(self) -> None:
        series = [_make_series("test", [5.0, 6.0], [5.0, 6.0])]
        bounds = PlotBounds(0.0, 10.0, 0.0, 10.0)
        fragments = render_plot(series, Viewport(60, 20), "x", "y", bounds=bounds)
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0


class TestRenderPlotAscii:
    def test_returns_strings(self) -> None:
        series = [_make_series("test", [1.0, 2.0], [3.0, 4.0])]
        rows = render_plot_ascii(series, Viewport(40, 15), "x", "y")
        assert all(isinstance(row, str) for row in rows)
        assert len(rows) > 0

    def test_no_style_metadata(self) -> None:
        series = [_make_series("test", [1.0, 2.0], [3.0, 4.0])]
        rows = render_plot_ascii(series, Viewport(40, 15), "x", "y")
        for row in rows:
            assert "(" not in row or "fg:" not in row


class TestRenderPanels:
    def test_single_panel(self) -> None:
        groups = [[_make_series("a", [1.0, 2.0], [3.0, 4.0])]]
        fragments = render_panels(groups, Viewport(60, 24), "x")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0

    def test_two_panels(self) -> None:
        groups = [
            [_make_series("Train", [1.0, 2.0], [0.5, 0.4])],
            [_make_series("Val", [1.0, 2.0], [0.8, 0.7])],
        ]
        fragments = render_panels(groups, Viewport(60, 30), "epoch")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0

    def test_empty_groups(self) -> None:
        fragments = render_panels([], Viewport(60, 20), "x")
        text = "".join(t for _, t in fragments)
        assert "No data" in text


# --- State tests ---


class TestPlotState:
    def test_initializes_with_defaults(self) -> None:
        data = parse_text("1.0 2.0\n3.0 4.0\n5.0 6.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        assert state.x_col_index == 0
        assert state.y_col_index == 1
        assert state.zoom == 1.0
        assert state.pan_x == 0.0
        assert state.pan_y == 0.0

    def test_zoom_changes_effective_bounds(self) -> None:
        data = parse_text("0.0 0.0\n10.0 10.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        initial_range = state.effective_bounds.x_max - state.effective_bounds.x_min
        state.zoom_in()
        zoomed_range = state.effective_bounds.x_max - state.effective_bounds.x_min
        assert zoomed_range < initial_range

    def test_pan_shifts_effective_bounds(self) -> None:
        data = parse_text("0.0 0.0\n10.0 10.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        initial_cx = (state.effective_bounds.x_min + state.effective_bounds.x_max) / 2
        state.pan(1, 0)
        new_cx = (state.effective_bounds.x_min + state.effective_bounds.x_max) / 2
        assert new_cx > initial_cx

    def test_reset_view_restores_defaults(self) -> None:
        data = parse_text("0.0 0.0\n10.0 10.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        state.zoom_in()
        state.pan(1, 1)
        state.reset_view()
        assert state.zoom == 1.0
        assert state.pan_x == 0.0
        assert state.pan_y == 0.0

    def test_toggle_plot_mode(self) -> None:
        data = parse_text("1.0 2.0\n3.0 4.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        assert state.plot_mode == "scatter"
        state.toggle_plot_mode()
        assert state.plot_mode == "line"
        state.toggle_plot_mode()
        assert state.plot_mode == "both"
        state.toggle_plot_mode()
        assert state.plot_mode == "scatter"

    def test_toggle_scale(self) -> None:
        data = parse_text("1.0 2.0\n10.0 20.0\n100.0 200.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        assert state.x_scale == "linear"
        state.toggle_x_scale()
        assert state.x_scale == "log"
        state.toggle_x_scale()
        assert state.x_scale == "linear"

    def test_column_select_updates_series(self) -> None:
        data = parse_text("Train 0.1 0.2\nTrain 0.3 0.4\nVal 0.5 0.6\nVal 0.7 0.8\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        assert len(state.series_list) == 2
        # Simulate the two-step column selector: pick x, then pick y.
        state.begin_column_select()
        state.col_cursor = 0
        state.confirm_column_select()  # confirms x
        state.col_cursor = 1
        state.confirm_column_select()  # confirms y
        assert state.x_col_index == 0
        assert state.y_col_index == 1
        assert len(state.series_list) == 2

    def test_column_selector_fragments_non_empty(self) -> None:
        data = parse_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        state.begin_column_select()
        frags = state.column_selector_fragments()
        text = "".join(t for _, t in frags)
        assert "Select X-AXIS" in text
        assert "col_0" in text
        assert "col_1" in text
        assert "col_2" in text

    def test_column_selector_shows_y_prompt_after_x(self) -> None:
        data = parse_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        state.begin_column_select()
        state.confirm_column_select()  # confirm x at cursor=0
        frags = state.column_selector_fragments()
        text = "".join(t for _, t in frags)
        assert "Select Y-AXIS" in text

    def test_column_cancel(self) -> None:
        data = parse_text("1.0 2.0\n3.0 4.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        state.begin_column_select()
        assert state.column_select_mode is True
        state.cancel_column_select()
        assert state.column_select_mode is False

    def test_col_cursor_movement(self) -> None:
        data = parse_text("1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        state.begin_column_select()
        assert state.col_cursor == 0
        state.col_move_down()
        assert state.col_cursor == 1
        state.col_move_down()
        assert state.col_cursor == 2
        state.col_move_up()
        assert state.col_cursor == 1
        state.col_move_up()
        assert state.col_cursor == 0
        state.col_move_up()  # stays at 0
        assert state.col_cursor == 0

    def test_status_includes_column_info(self) -> None:
        data = parse_text("1.0 2.0\n3.0 4.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        status = state.status()
        assert "zoom=1.00" in status

    def test_info_text_lists_columns(self) -> None:
        data = parse_text("1.0 2.0 3.0\n4.0 5.0 6.0\n")
        from xtalui.plot.app import PlotState

        state = PlotState(data, x_col_index=0, y_col_index=1)
        info = state.info_text()
        assert "3 numeric columns" in info


# --- CLI tests ---


class TestCLI:
    def test_accepts_file_argument(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt"])
        assert args.file == "data.txt"

    def test_accepts_column_indices(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "-x", "2", "-y", "5"])
        assert args.x_column == 2
        assert args.y_column == 5

    def test_accepts_no_grid_flag(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--no-grid"])
        assert args.no_grid is True

    def test_accepts_ascii_flag(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--ascii"])
        assert args.ascii is True

    def test_accepts_log_flags(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--log-x", "--log-y"])
        assert args.log_x is True
        assert args.log_y is True

    def test_accepts_line_flag(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--line"])
        assert args.line is True

    def test_file_defaults_to_none(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.file is None

    def test_column_defaults_to_none(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt"])
        assert args.x_column is None
        assert args.y_column is None

    def test_accepts_color_flag(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--color"])
        assert args.color is True

    def test_accepts_axis_limits(self) -> None:
        from xtalui.plot.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["data.txt", "--xmin", "0", "--xmax", "100", "--ymin", "-1", "--ymax", "1"])
        assert args.xmin == 0.0
        assert args.xmax == 100.0
        assert args.ymin == -1.0
        assert args.ymax == 1.0

    def test_both_plot_mode_renders(self) -> None:
        series = [_make_series("test", [1.0, 2.0, 3.0], [1.0, 4.0, 2.0])]
        fragments = render_plot(series, Viewport(60, 20), "x", "y", plot_mode="both")
        text = "".join(t for _, t in fragments)
        assert len(text.strip()) > 0
