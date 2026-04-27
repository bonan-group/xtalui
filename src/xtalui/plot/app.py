"""Interactive prompt_toolkit application for the terminal plot viewer."""

from __future__ import annotations

import numpy as np
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.controls import FormattedTextControl

from xtalui.plot.parser import ParsedData, Series, auto_series, detect_numeric_columns, multi_series
from xtalui.plot.renderer import (
    PlotBounds,
    PlotMode,
    ScaleType,
    Viewport,
    render_plot,
    render_plot_ascii,
)

TextFragment = tuple[str, str]

INFO_PANEL_LINES = 3
COLUMN_SELECTOR_WIDTH = 64
COLUMN_SELECTOR_HEIGHT = 20
COLUMN_SELECTOR_X_COL = 0
COLUMN_SELECTOR_Y_COL = 1
HIGHLIGHT_STYLE = "bg:#ffffff fg:#000000"
SELECTED_X_STYLE = "fg:#4fc3f7 bold"
SELECTED_Y_STYLE = "fg:#ff8a65 bold"


class PlotState:
    """Mutable runtime state for the interactive plot viewer."""

    def __init__(
        self,
        data: ParsedData,
        x_col_index: int = 0,
        y_col_index: int = 1,
        y_col_indices: list[int] | None = None,
        multi_column: bool = False,
        show_grid: bool = True,
        plot_mode: PlotMode = "scatter",
        x_scale: ScaleType = "linear",
        y_scale: ScaleType = "linear",
        show_color: bool = False,
        show_legend: bool = False,
        auto_group: bool = False,
    ) -> None:
        self.data = data
        self.columns = detect_numeric_columns(data)
        self.x_col_index = x_col_index
        self.y_col_index = y_col_index
        self.y_col_indices: list[int] = y_col_indices if y_col_indices is not None else [y_col_index]
        self.multi_column = multi_column
        self.auto_group = auto_group
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.show_grid = show_grid
        self.show_color = show_color
        self.show_legend = show_legend
        self.plot_mode = plot_mode
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.show_help = False
        self.status_message = ""
        self.column_select_mode = False
        self.col_cursor = 0  # cursor row in the column selector
        self.col_selecting_axis = COLUMN_SELECTOR_X_COL  # 0 = picking x, 1 = picking y
        self._pending_x: int | None = None
        self._custom_bounds: PlotBounds | None = None
        self.col_selected_y: set[int] = set()
        self._refresh_series()

    def _refresh_series(self) -> None:
        if self.multi_column and len(self.y_col_indices) > 1:
            self.series_list: list[Series] = multi_series(self.data, self.x_col_index, self.y_col_indices)
        else:
            self.series_list = auto_series(self.data, self.x_col_index, self.y_col_index, auto_group=self.auto_group)
        self._auto_bounds = self._compute_auto_bounds()

    def _compute_auto_bounds(self) -> PlotBounds:
        if not self.series_list:
            return PlotBounds(0.0, 1.0, 0.0, 1.0)
        all_x = np.concatenate([s.x for s in self.series_list])
        all_y = np.concatenate([s.y for s in self.series_list])
        if self.x_scale == "log":
            all_x = np.log10(all_x[all_x > 0])
        if self.y_scale == "log":
            all_y = np.log10(all_y[all_y > 0])
        if len(all_x) == 0 or len(all_y) == 0:
            return PlotBounds(0.0, 1.0, 0.0, 1.0)
        x_margin = (float(np.max(all_x)) - float(np.min(all_x))) * 0.05
        if x_margin == 0:
            x_margin = 0.5
        y_margin = (float(np.max(all_y)) - float(np.min(all_y))) * 0.05
        if y_margin == 0:
            y_margin = 0.5
        return PlotBounds(
            x_min=float(np.min(all_x)) - x_margin,
            x_max=float(np.max(all_x)) + x_margin,
            y_min=float(np.min(all_y)) - y_margin,
            y_max=float(np.max(all_y)) + y_margin,
        )

    @property
    def effective_bounds(self) -> PlotBounds:
        b = self._custom_bounds if self._custom_bounds is not None else self._auto_bounds
        cx = (b.x_min + b.x_max) / 2 + self.pan_x
        cy = (b.y_min + b.y_max) / 2 + self.pan_y
        half_w = (b.x_max - b.x_min) / 2 / self.zoom
        half_h = (b.y_max - b.y_min) / 2 / self.zoom
        return PlotBounds(cx - half_w, cx + half_w, cy - half_h, cy + half_h)

    def set_bounds(self, bounds: PlotBounds) -> None:
        import math

        self._custom_bounds = PlotBounds(
            x_min=math.log10(max(bounds.x_min, 1e-300)) if self.x_scale == "log" else bounds.x_min,
            x_max=math.log10(max(bounds.x_max, 1e-300)) if self.x_scale == "log" else bounds.x_max,
            y_min=math.log10(max(bounds.y_min, 1e-300)) if self.y_scale == "log" else bounds.y_min,
            y_max=math.log10(max(bounds.y_max, 1e-300)) if self.y_scale == "log" else bounds.y_max,
        )

    def render(self, width: int, height: int) -> list[TextFragment]:
        viewport = Viewport(width=width, height=height)
        return render_plot(
            self.series_list,
            viewport,
            self.columns[self.x_col_index].name if self.x_col_index < len(self.columns) else "?",
            self._y_axis_name(),
            bounds=self.effective_bounds,
            show_grid=self.show_grid,
            show_color=self.show_color,
            plot_mode=self.plot_mode,
            x_scale=self.x_scale,
            y_scale=self.y_scale,
            show_legend=self.show_legend,
        )

    def _y_axis_name(self) -> str:
        if self.multi_column and len(self.y_col_indices) > 1:
            names = [self.columns[i].name for i in self.y_col_indices if i < len(self.columns)]
            return "[" + ",".join(names) + "]"
        return self.columns[self.y_col_index].name if self.y_col_index < len(self.columns) else "?"

    def render_ascii(self, width: int, height: int, bounds: PlotBounds | None = None) -> list[str]:
        viewport = Viewport(width=width, height=height)
        return render_plot_ascii(
            self.series_list,
            viewport,
            self.columns[self.x_col_index].name if self.x_col_index < len(self.columns) else "?",
            self._y_axis_name(),
            bounds=bounds if bounds is not None else self.effective_bounds,
            show_grid=self.show_grid,
            show_color=self.show_color,
            plot_mode=self.plot_mode,
            x_scale=self.x_scale,
            y_scale=self.y_scale,
            show_legend=self.show_legend,
        )

    def zoom_in(self) -> None:
        self.zoom = min(self.zoom * 1.3, 1000.0)
        self.status_message = f"zoom={self.zoom:.2f}"

    def zoom_out(self) -> None:
        self.zoom = max(self.zoom / 1.3, 0.01)
        self.status_message = f"zoom={self.zoom:.2f}"

    def pan(self, dx: float, dy: float) -> None:
        b = self._auto_bounds
        x_range = (b.x_max - b.x_min) / self.zoom
        y_range = (b.y_max - b.y_min) / self.zoom
        self.pan_x += dx * x_range * 0.05
        self.pan_y += dy * y_range * 0.05

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._custom_bounds = None
        self.status_message = "view reset"

    def toggle_grid(self) -> None:
        self.show_grid = not self.show_grid
        self.status_message = f"grid {'on' if self.show_grid else 'off'}"

    def toggle_color(self) -> None:
        self.show_color = not self.show_color
        self.status_message = f"color {'on' if self.show_color else 'off'}"

    def toggle_multi_column(self) -> None:
        self.multi_column = not self.multi_column
        if self.multi_column:
            if len(self.y_col_indices) <= 1:
                self.y_col_indices = [self.y_col_index]
            self.show_color = True
            self.show_legend = True
            self.status_message = "multi-column on (press c to add y-columns)"
        else:
            self.y_col_indices = [self.y_col_index]
            self.show_legend = False
            self.status_message = "multi-column off"
        self._refresh_series()

    def toggle_legend(self) -> None:
        self.show_legend = not self.show_legend
        self.status_message = f"legend {'on' if self.show_legend else 'off'}"

    def toggle_auto_group(self) -> None:
        self.auto_group = not self.auto_group
        self._refresh_series()
        self.status_message = f"auto-group {'on' if self.auto_group else 'off'}  series={len(self.series_list)}"

    def toggle_plot_mode(self) -> None:
        modes = ["scatter", "line", "both"]
        idx = modes.index(self.plot_mode)
        self.plot_mode = modes[(idx + 1) % len(modes)]
        self.status_message = f"mode={self.plot_mode}"

    def toggle_x_scale(self) -> None:
        self.x_scale: ScaleType = "log" if self.x_scale == "linear" else "linear"
        self._custom_bounds = None  # bounds are in wrong space after toggle
        self._refresh_series()
        self.status_message = f"x-scale={self.x_scale}"

    def toggle_y_scale(self) -> None:
        self.y_scale: ScaleType = "log" if self.y_scale == "linear" else "linear"
        self._custom_bounds = None  # bounds are in wrong space after toggle
        self._refresh_series()
        self.status_message = f"y-scale={self.y_scale}"

    def begin_column_select(self) -> None:
        self.column_select_mode = True
        self.col_cursor = self.x_col_index
        self.col_selecting_axis = COLUMN_SELECTOR_X_COL
        self._pending_x = None
        self.col_selected_y = set(self.y_col_indices)
        self.status_message = "selecting x-axis column"

    def confirm_column_select(self) -> None:
        idx = self.col_cursor
        if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
            self._pending_x = idx
            self.col_selecting_axis = COLUMN_SELECTOR_Y_COL
            self.col_cursor = self.y_col_index
            self.col_selected_y = set(self.y_col_indices)
            self.status_message = f"x={self.columns[idx].name} — now selecting y-axis column"
        else:
            x_idx = self._pending_x if self._pending_x is not None else self.x_col_index
            if self.multi_column:
                self.x_col_index = x_idx
                if self.col_selected_y:
                    self.y_col_indices = sorted(self.col_selected_y)
                    self.y_col_index = self.y_col_indices[0]
                else:
                    self.status_message = "select at least one y-column"
                    return
                self._refresh_series()
            else:
                y_idx = idx
                self.x_col_index = x_idx
                self.y_col_index = y_idx
                self.y_col_indices = [y_idx]
                self._refresh_series()
            self.column_select_mode = False
            self._pending_x = None
            self.col_selected_y = set()
            self.status_message = f"x={self.columns[self.x_col_index].name}, y={self._y_axis_name()}"

    def cancel_column_select(self) -> None:
        self.column_select_mode = False
        self._pending_x = None
        self.col_selected_y = set()
        self.status_message = "cancelled"

    def toggle_y_column_in_selector(self) -> None:
        """Toggle the cursor column in the y-selection set (multi-column mode only)."""
        if not self.column_select_mode or self.col_selecting_axis != COLUMN_SELECTOR_Y_COL:
            return
        idx = self.col_cursor
        if idx in self.col_selected_y:
            self.col_selected_y.discard(idx)
        else:
            self.col_selected_y.add(idx)

    def col_move_up(self) -> None:
        if self.col_cursor > 0:
            self.col_cursor -= 1

    def col_move_down(self) -> None:
        if self.col_cursor < len(self.columns) - 1:
            self.col_cursor += 1

    def status(self) -> str:
        x_name = self.columns[self.x_col_index].name if self.x_col_index < len(self.columns) else "?"
        if self.multi_column and len(self.y_col_indices) > 1:
            y_str = "[" + ",".join(str(i) for i in self.y_col_indices) + "]"
        else:
            y_name = self.columns[self.y_col_index].name if self.y_col_index < len(self.columns) else "?"
            y_str = f"{y_name}({self.y_col_index})"
        parts = [
            f"x={x_name}({self.x_col_index})",
            f"y={y_str}",
            f"zoom={self.zoom:.2f}",
            f"{self.plot_mode}",
            f"x:{self.x_scale}",
            f"y:{self.y_scale}",
            f"grid={'on' if self.show_grid else 'off'}",
            f"color={'on' if self.show_color else 'off'}",
            f"series={len(self.series_list)}",
            f"pts={sum(len(s.x) for s in self.series_list)}",
        ]
        if self.multi_column:
            parts.append("multi")
        if self.auto_group:
            parts.append("group")
        if self.show_legend:
            parts.append("legend")
        if self.status_message:
            parts.append(self.status_message)
        return "  ".join(parts)

    def info_text(self) -> str:
        n = len(self.columns)
        lines = [f"Source: {self.data.source_name}  |  {len(self.data.raw_lines)} lines  |  {n} numeric columns"]
        if n > 0:
            col_info = "  ".join(f"{i}:{c.name}" for i, c in enumerate(self.columns))
            lines.append(col_info)
        else:
            lines.append("no numeric columns detected")
        return "\n".join(lines)

    def help_text(self) -> str:
        if not self.show_help:
            return ""
        return (
            "Arrow=pan  +/-=zoom  m=scatter/line  g=grid  M=multi-col  L=legend  "
            "G=auto-group  X=x-scale  Y=y-scale  C=color  c=columns  Ctrl-R=reset  ?=help  q=quit"
        )

    def column_selector_text(self) -> str:
        """Render the interactive column selector as styled fragments."""
        n = len(self.columns)
        lines: list[str] = []
        lines.append("")
        if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
            lines.append(f"  Select X-AXIS column (arrows+enter, or digit 0-{n - 1}):")
        else:
            x_name = self.columns[self._pending_x].name if self._pending_x is not None else "?"
            lines.append(f"  X = {x_name}  |  Select Y-AXIS column (arrows+enter, or digit 0-{n - 1}):")
        lines.append(f"  {'':>3s}  {'Column':<30s}  {'Min':>14s}  {'Max':>14s}  {'Count':>6s}")
        lines.append(f"  {'---':>3s}  {'------':->30s}  {'---':->14s}  {'---':->14s}  {'-----':->6s}")

        for i, col in enumerate(self.columns):
            marker = ">>>" if i == self.col_cursor else "   "
            prefix = "x" if i == self.x_col_index else ("y" if i == self.y_col_index else " ")
            if i == self.col_cursor:
                prefix = "X" if self.col_selecting_axis == COLUMN_SELECTOR_X_COL else "Y"
            name = col.name[:30]
            mn = f"{min(col.values):.6g}" if col.values else "—"
            mx = f"{max(col.values):.6g}" if col.values else "—"
            cnt = str(len(col.values))
            lines.append(f"  {marker} [{prefix}] {name:<30s}  {mn:>14s}  {mx:>14s}  {cnt:>6s}")

        lines.append("")
        if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
            lines.append("  Enter=confirm x  Esc=cancel  or type digit to jump")
        else:
            lines.append("  Enter=confirm y & plot  Esc=cancel  or type digit to jump")

        return "\n".join(lines)

    def column_selector_fragments(self) -> list[TextFragment]:
        """Return styled fragments for the column selector."""
        n = len(self.columns)
        fragments: list[TextFragment] = []

        def add(text: str, style: str = "") -> None:
            fragments.append((style, text))

        add("\n")
        if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
            add(f"  Select X-AXIS column (arrows+enter, or digit 0-{n - 1}):\n")
        else:
            x_name = self.columns[self._pending_x].name if self._pending_x is not None else "?"
            add("  X = ", SELECTED_X_STYLE)
            add(f"{x_name}")
            if self.multi_column:
                add("  |  Select Y-AXIS columns (Space=toggle, Enter=done):\n")
            else:
                add(f"  |  Select Y-AXIS column (arrows+enter, or digit 0-{n - 1}):\n")

        header = f"  {'':>3s}  {'Column':<30s}  {'Min':>14s}  {'Max':>14s}  {'Count':>6s}"
        add(header + "\n")
        sep = f"  {'---':>3s}  {'------':->30s}  {'---':->14s}  {'---':->14s}  {'-----':->6s}"
        add(sep + "\n")

        for i, col in enumerate(self.columns):
            is_cursor = i == self.col_cursor

            if self.col_selecting_axis == COLUMN_SELECTOR_Y_COL and self.multi_column:
                # Multi-select mode: show check marks.
                if i in self.col_selected_y:
                    prefix = "Y* "
                else:
                    prefix = "   "
            else:
                prefix = " "
                if i == self.x_col_index and i == self.y_col_index:
                    prefix = "x,y"
                elif i == self.x_col_index:
                    prefix = "x  "
                elif i == self.y_col_index:
                    prefix = " y "

            name = col.name[:30].ljust(30)
            mn = f"{min(col.values):>14.6g}" if col.values else f"{'—':>14s}"
            mx = f"{max(col.values):>14.6g}" if col.values else f"{'—':>14s}"
            cnt = f"{len(col.values):>6d}"

            if is_cursor:
                add("  >>> ", HIGHLIGHT_STYLE)
                if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
                    add("[X] ", HIGHLIGHT_STYLE)
                elif self.multi_column:
                    check = "*" if i in self.col_selected_y else " "
                    add(f"[{check}] ", HIGHLIGHT_STYLE)
                else:
                    add("[Y] ", HIGHLIGHT_STYLE)
                add(f"{name}  {mn}  {mx}  {cnt}", HIGHLIGHT_STYLE)
            else:
                add(f"      [{prefix}] {name}  {mn}  {mx}  {cnt}")
            add("\n")

        add("\n")
        if self.col_selecting_axis == COLUMN_SELECTOR_X_COL:
            add("  Enter=confirm x  Esc=cancel  or type digit to jump\n")
        elif self.multi_column:
            add("  Space=toggle  Enter=confirm & plot  Esc=cancel\n")
        else:
            add("  Enter=confirm y & plot  Esc=cancel  or type digit to jump\n")

        return fragments


def build_plot_application(state: PlotState) -> Application:
    """Build a prompt_toolkit Application for the interactive plot viewer."""

    def get_body_fragments() -> list[TextFragment]:
        try:
            from prompt_toolkit.output.defaults import create_output

            out = create_output()
            size = out.get_size()
            return state.render(size.columns, size.rows - INFO_PANEL_LINES - 2)
        except Exception:
            return state.render(80, 20)

    body_control = FormattedTextControl(get_body_fragments)
    info_control = FormattedTextControl(lambda: state.info_text())
    footer_control = FormattedTextControl(lambda: state.status())
    help_control = FormattedTextControl(lambda: state.help_text())
    col_selector_control = FormattedTextControl(lambda: state.column_selector_fragments())

    # Column selector as a float overlay on top of the body.
    col_selector_float = Float(
        content=ConditionalContainer(
            content=Window(
                content=col_selector_control,
                height=Dimension(min=8),
                dont_extend_height=True,
                always_hide_cursor=True,
            ),
            filter=Condition(lambda: state.column_select_mode),
        ),
    )

    layout = Layout(
        FloatContainer(
            content=HSplit(
                [
                    Window(
                        content=info_control, height=INFO_PANEL_LINES, dont_extend_height=True, always_hide_cursor=True
                    ),
                    Window(content=body_control, always_hide_cursor=True),
                    Window(
                        content=footer_control,
                        height=Dimension(min=1),
                        dont_extend_height=True,
                        wrap_lines=True,
                        always_hide_cursor=True,
                    ),
                    ConditionalContainer(
                        content=Window(
                            content=help_control,
                            height=Dimension(min=1),
                            dont_extend_height=True,
                            wrap_lines=True,
                            always_hide_cursor=True,
                        ),
                        filter=Condition(lambda: state.show_help),
                    ),
                ]
            ),
            floats=[col_selector_float],
        )
    )

    bindings = KeyBindings()

    @bindings.add("q")
    @bindings.add("c-c")
    def _quit(event: object) -> None:
        event.app.exit()

    # --- Normal mode bindings ---

    @bindings.add("left", filter=~Condition(lambda: state.column_select_mode))
    def _pan_left(event: object) -> None:
        state.pan(-1, 0)
        event.app.invalidate()

    @bindings.add("right", filter=~Condition(lambda: state.column_select_mode))
    def _pan_right(event: object) -> None:
        state.pan(1, 0)
        event.app.invalidate()

    @bindings.add("up", filter=~Condition(lambda: state.column_select_mode))
    def _pan_up(event: object) -> None:
        state.pan(0, 1)
        event.app.invalidate()

    @bindings.add("down", filter=~Condition(lambda: state.column_select_mode))
    def _pan_down(event: object) -> None:
        state.pan(0, -1)
        event.app.invalidate()

    @bindings.add("+")
    @bindings.add("=")
    def _zoom_in(event: object) -> None:
        state.zoom_in()
        event.app.invalidate()

    @bindings.add("-")
    def _zoom_out(event: object) -> None:
        state.zoom_out()
        event.app.invalidate()

    @bindings.add("g")
    def _toggle_grid(event: object) -> None:
        state.toggle_grid()
        event.app.invalidate()

    @bindings.add("C")
    def _toggle_color(event: object) -> None:
        state.toggle_color()
        event.app.invalidate()

    @bindings.add("m")
    def _toggle_mode(event: object) -> None:
        state.toggle_plot_mode()
        event.app.invalidate()

    @bindings.add("X")
    def _toggle_x_scale(event: object) -> None:
        state.toggle_x_scale()
        event.app.invalidate()

    @bindings.add("Y")
    def _toggle_y_scale(event: object) -> None:
        state.toggle_y_scale()
        event.app.invalidate()

    @bindings.add("M")
    def _toggle_multi(event: object) -> None:
        state.toggle_multi_column()
        event.app.invalidate()

    @bindings.add("L")
    def _toggle_legend(event: object) -> None:
        state.toggle_legend()
        event.app.invalidate()

    @bindings.add("G")
    def _toggle_auto_group(event: object) -> None:
        state.toggle_auto_group()
        event.app.invalidate()

    @bindings.add("c")
    def _column_select(event: object) -> None:
        state.begin_column_select()
        event.app.invalidate()

    @bindings.add("c-r")
    def _reset(event: object) -> None:
        state.reset_view()
        event.app.invalidate()

    @bindings.add("?")
    def _toggle_help(event: object) -> None:
        state.show_help = not state.show_help
        event.app.invalidate()

    # --- Column selector bindings ---

    @bindings.add("up", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_up(event: object) -> None:
        state.col_move_up()
        event.app.invalidate()

    @bindings.add("down", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_down(event: object) -> None:
        state.col_move_down()
        event.app.invalidate()

    @bindings.add("enter", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_confirm(event: object) -> None:
        state.confirm_column_select()
        event.app.invalidate()

    @bindings.add("escape", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_cancel(event: object) -> None:
        state.cancel_column_select()
        event.app.invalidate()

    @bindings.add("space", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_toggle_y(event: object) -> None:
        if state.multi_column and state.col_selecting_axis == COLUMN_SELECTOR_Y_COL:
            state.toggle_y_column_in_selector()
            event.app.invalidate()

    @bindings.add("<any>", filter=Condition(lambda: state.column_select_mode), eager=True)
    def _col_digit(event: object) -> None:
        key = event.key_sequence[0].key
        if key is None or not key.isdigit():
            return
        idx = int(key)
        if 0 <= idx < len(state.columns):
            state.col_cursor = idx
        event.app.invalidate()

    return Application(layout=layout, key_bindings=bindings, full_screen=True)
