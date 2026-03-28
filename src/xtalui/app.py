from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase.data import atomic_names
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.controls import FormattedTextControl

from xtalui.renderer import BRAILLE_BASE, BRAILLE_DOTS, Viewport, render_ascii, render_formatted
from xtalui.scene import (
    CameraState,
    RenderOptions,
    bond_records,
    cycle_line_mode,
    load_structures,
    reset_camera,
    structure_info,
    toggle_flag,
    view_along,
    with_pan,
    with_rotation,
    with_zoom,
)

INFO_PANEL_LINES = 12
AXIS_WIDGET_WIDTH = 23
AXIS_WIDGET_HEIGHT = 10
AXIS_ASPECT_RATIO = 2.0
AXIS_LENGTH_SCALE = 1.5
POSITIONS_PANEL_LINES = 9
AUTOROTATE_INTERVAL = 0.05
AUTOROTATE_STEP = 0.08
FRAME_PLAY_INTERVAL = 0.25
ASPECT_RATIO_STEP = 0.1
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 4.0


class ViewerState:
    def __init__(
        self,
        paths: Sequence[str | Path],
        repeat: tuple[int, int, int],
        show_cell: bool,
        symprec: float,
        show_color: bool,
        image_number: str = ":",
    ) -> None:
        self.paths = tuple(paths)
        self.image_number = image_number
        self.initial_repeat = tuple(int(value) for value in repeat)
        self.repeat = self.initial_repeat
        self.symprec = symprec
        self.camera = CameraState(show_cell=show_cell, show_color=show_color)
        self.pending_repeat_command: str | None = None
        self.status_message = ""
        self.autorotate = False
        self.frame_autoplay = False
        self.calibration_mode = False
        self.aspect_ratio = RenderOptions().aspect_ratio
        self.show_positions = False
        self.position_scroll = 0
        self.show_bond_lengths = False
        self.bond_scroll = 0
        self._next_frame_time = 0.0
        self.frame_index = 0
        self.scenes: list = []
        self.infos: list = []
        self.reload_scene()

    @property
    def scene(self):
        return self.scenes[self.frame_index]

    @property
    def info(self):
        return self.infos[self.frame_index]

    @property
    def frame_count(self) -> int:
        return len(self.scenes)

    def reload_scene(self) -> None:
        self.scenes = load_structures(self.paths, self.repeat, image_number=self.image_number)
        self.infos = [structure_info(scene, symprec=self.symprec) for scene in self.scenes]
        self.frame_index = min(self.frame_index, len(self.scenes) - 1)
        self.position_scroll = 0
        self.bond_scroll = 0

    def begin_repeat_command(self) -> None:
        self.pending_repeat_command = ""
        self.status_message = "repeat: type rXYZ, for example r222"

    def cancel_repeat_command(self) -> None:
        self.pending_repeat_command = None
        self.status_message = "repeat command cancelled"

    def append_repeat_digit(self, digit: str) -> None:
        if self.pending_repeat_command is None:
            return
        if digit not in "123456789":
            self.status_message = "repeat digits must be in 1..9"
            return
        self.pending_repeat_command += digit
        if len(self.pending_repeat_command) < 3:
            return
        self.repeat = tuple(int(value) for value in self.pending_repeat_command[:3])
        self.pending_repeat_command = None
        self.reload_scene()
        self.status_message = f"repeat set to {self.repeat[0]}x{self.repeat[1]}x{self.repeat[2]}"

    def step_frame(self, delta: int) -> None:
        if self.frame_count <= 1:
            self.status_message = "single-frame structure"
            return
        self.frame_index = (self.frame_index + delta) % self.frame_count
        self.position_scroll = 0
        self.bond_scroll = 0
        self.status_message = f"frame {self.frame_index + 1}/{self.frame_count}"
        self._next_frame_time = time.monotonic() + FRAME_PLAY_INTERVAL

    def toggle_autorotate(self) -> None:
        self.autorotate = not self.autorotate
        self.status_message = f"autorotate {'on' if self.autorotate else 'off'}"

    def toggle_frame_autoplay(self) -> None:
        if self.frame_count <= 1:
            self.frame_autoplay = False
            self.status_message = "single-frame structure"
            return
        self.frame_autoplay = not self.frame_autoplay
        self._next_frame_time = time.monotonic() + FRAME_PLAY_INTERVAL
        self.status_message = f"frame playback {'on' if self.frame_autoplay else 'off'}"

    def tick_autorotate(self) -> bool:
        if not self.autorotate:
            return False
        self.camera = with_rotation(self.camera, dx=AUTOROTATE_STEP, dy=0.0)
        return True

    def tick_frame_autoplay(self, now: float) -> bool:
        if not self.frame_autoplay or self.frame_count <= 1 or now < self._next_frame_time:
            return False
        self.frame_index = (self.frame_index + 1) % self.frame_count
        self.position_scroll = 0
        self.bond_scroll = 0
        self._next_frame_time = now + FRAME_PLAY_INTERVAL
        return True

    def repeat_prompt(self) -> str:
        if self.pending_repeat_command is None:
            return ""
        digits = self.pending_repeat_command.ljust(3, "_")
        return f"cmd=r{digits}"

    def reset_viewer(self) -> None:
        self.camera = reset_camera(self.camera)
        self.pending_repeat_command = None
        self.position_scroll = 0
        self.bond_scroll = 0
        if self.repeat != self.initial_repeat:
            self.repeat = self.initial_repeat
            self.reload_scene()
        self.status_message = f"view reset; repeat={self.repeat[0]}x{self.repeat[1]}x{self.repeat[2]}"

    def render(self, width: int, height: int) -> str:
        if self.calibration_mode:
            return "\n".join(render_calibration_lines(width, self.body_height(width, height), self.aspect_ratio))
        body_height = self.body_height(width, height)
        rows = render_ascii(
            self.scene,
            self.camera,
            Viewport(width=width, height=body_height),
            RenderOptions(aspect_ratio=self.aspect_ratio),
        )
        return "\n".join(rows)

    def render_formatted(self, width: int, height: int):
        if self.calibration_mode:
            return render_calibration_formatted(width, self.body_height(width, height), self.aspect_ratio)
        body_height = self.body_height(width, height)
        return render_formatted(
            self.scene,
            self.camera,
            Viewport(width=width, height=body_height),
            RenderOptions(aspect_ratio=self.aspect_ratio),
        )

    def body_height(self, width: int, height: int) -> int:
        footer_lines = wrapped_line_count(self.status(), width)
        help_lines = wrapped_line_count(self.help_text(), width) if self.camera.show_help else 0
        return max(height - INFO_PANEL_LINES - footer_lines - help_lines, 1)

    def info_text(self) -> str:
        a_vec, b_vec, c_vec = self.info.lattice_vectors
        a_len, b_len, c_len = self.info.lengths
        alpha, beta, gamma = self.info.angles
        lines = [
            f"{self.scene.title}  frame={self.frame_index + 1}/{self.frame_count}  SG={self.info.spacegroup}",
            f"Formula: full={self.info.formula_full}  reduced={self.info.formula_reduced}",
            f"a = ({a_vec[0]:7.3f}, {a_vec[1]:7.3f}, {a_vec[2]:7.3f})",
            f"b = ({b_vec[0]:7.3f}, {b_vec[1]:7.3f}, {b_vec[2]:7.3f})",
            f"c = ({c_vec[0]:7.3f}, {c_vec[1]:7.3f}, {c_vec[2]:7.3f})",
            f"|a|={a_len:7.3f}  |b|={b_len:7.3f}  |c|={c_len:7.3f}  V={self.info.volume:9.3f}",
            f"alpha={alpha:6.2f}  beta={beta:6.2f}  gamma={gamma:6.2f}",
        ]
        if not self.camera.show_labels:
            lines.append(element_legend(self.scene))
        return "\n".join(lines)

    def axis_text(self) -> str:
        return render_lattice_direction_widget(
            self.scene.cell,
            self.camera,
            width=AXIS_WIDGET_WIDTH,
            height=AXIS_WIDGET_HEIGHT,
            aspect_ratio=self.aspect_ratio,
        )

    def cartesian_axis_text(self) -> str:
        return render_cartesian_direction_widget(
            self.camera,
            width=AXIS_WIDGET_WIDTH,
            height=AXIS_WIDGET_HEIGHT,
            aspect_ratio=self.aspect_ratio,
        )

    def status(self) -> str:
        parts = [
            f"frame={self.frame_index + 1}/{self.frame_count}",
            f"repeat={self.repeat[0]}x{self.repeat[1]}x{self.repeat[2]}",
            f"aspect={self.aspect_ratio:.2f}",
            f"zoom={self.camera.zoom:.2f}  "
            f"pan=({self.camera.pan_x:.1f}, {self.camera.pan_y:.1f})  "
            f"mode={self.camera.line_mode}  "
            f"cal={'on' if self.calibration_mode else 'off'}  "
            f"spin={'on' if self.autorotate else 'off'}  "
            f"play={'on' if self.frame_autoplay else 'off'}  "
            f"pos={'on' if self.show_positions else 'off'}  "
            f"bondlen={'on' if self.show_bond_lengths else 'off'}  "
            f"color={'on' if self.camera.show_color else 'off'}  "
            f"spheres={'on' if self.camera.show_spheres else 'off'}  "
            f"abc={'on' if self.camera.show_abc_panel else 'off'}  "
            f"xyz={'on' if self.camera.show_xyz_panel else 'off'}  "
            f"cell={'on' if self.camera.show_cell else 'off'}  "
            f"bonds={'on' if self.camera.show_bonds else 'off'}  "
            f"labels={'on' if self.camera.show_labels else 'off'}",
        ]
        if self.pending_repeat_command is not None:
            parts.append(self.repeat_prompt())
        if self.status_message:
            parts.append(self.status_message)
        return "  ".join(parts)

    def help_text(self) -> str:
        if not self.camera.show_help:
            return ""
        return "Arrows rotate | a autorotate | p positions | B bond lengths | t play frames | T calibration | [/] frames | x/y/z align view | r123 repeat | Ctrl-R reset | 1 abc panel | 2 xyz panel | m mode | s spheres | S-Arrows pan | +/- zoom | Left/Right or +/- adjust aspect in calibration | b bonds | c cell | l labels | C color | j/k or Up/Down scroll active overlay | Esc cancel cmd | ? help | q quit"

    def toggle_calibration(self) -> None:
        self.calibration_mode = not self.calibration_mode
        self.status_message = f"calibration {'on' if self.calibration_mode else 'off'}"

    def adjust_aspect_ratio(self, delta: float) -> None:
        self.aspect_ratio = min(max(self.aspect_ratio + delta, ASPECT_RATIO_MIN), ASPECT_RATIO_MAX)
        self.status_message = f"aspect ratio set to {self.aspect_ratio:.2f}"

    def toggle_positions(self) -> None:
        self.show_positions = not self.show_positions
        if self.show_positions:
            self.show_bond_lengths = False
        self._clamp_position_scroll()
        self.status_message = f"positions {'shown' if self.show_positions else 'hidden'}"

    def toggle_bond_lengths(self) -> None:
        self.show_bond_lengths = not self.show_bond_lengths
        if self.show_bond_lengths:
            self.show_positions = False
        self._clamp_bond_scroll()
        self.status_message = f"bond lengths {'shown' if self.show_bond_lengths else 'hidden'}"

    def scroll_positions(self, delta: int) -> None:
        if not self.show_positions:
            return
        max_scroll = self.max_position_scroll()
        if max_scroll <= 0:
            self.position_scroll = 0
            self.status_message = "positions fit in panel"
            return
        self.position_scroll = min(max(self.position_scroll + delta, 0), max_scroll)
        self.status_message = f"positions {self.position_scroll + 1}/{max_scroll + 1}"

    def scroll_bond_lengths(self, delta: int) -> None:
        if not self.show_bond_lengths:
            return
        max_scroll = self.max_bond_scroll()
        if max_scroll <= 0:
            self.bond_scroll = 0
            self.status_message = "bond lengths fit in panel"
            return
        self.bond_scroll = min(max(self.bond_scroll + delta, 0), max_scroll)
        self.status_message = f"bond lengths {self.bond_scroll + 1}/{max_scroll + 1}"

    def scroll_overlay(self, delta: int) -> None:
        if self.show_bond_lengths:
            self.scroll_bond_lengths(delta)
            return
        if self.show_positions:
            self.scroll_positions(delta)

    def max_position_scroll(self) -> int:
        return max(len(self.position_lines()) - POSITIONS_PANEL_LINES, 0)

    def _clamp_position_scroll(self) -> None:
        self.position_scroll = min(max(self.position_scroll, 0), self.max_position_scroll())

    def max_bond_scroll(self) -> int:
        return max(len(self.bond_lines()) - POSITIONS_PANEL_LINES, 0)

    def _clamp_bond_scroll(self) -> None:
        self.bond_scroll = min(max(self.bond_scroll, 0), self.max_bond_scroll())

    def position_lines(self) -> list[str]:
        lines = [
            " idx  el        x        y        z  |      fx       fy       fz",
            "---------------------------------------------------------------",
        ]
        cart_positions = np.asarray(self.scene.atoms.get_positions(), dtype=float)
        frac_positions = np.asarray(self.scene.atoms.get_scaled_positions(), dtype=float)
        for index, (symbol, cart, frac) in enumerate(zip(self.scene.symbols, cart_positions, frac_positions), start=1):
            lines.append(
                f"{index:4d}  {symbol:<2}  "
                f"{cart[0]:7.3f}  {cart[1]:7.3f}  {cart[2]:7.3f}  |  "
                f"{frac[0]:7.4f}  {frac[1]:7.4f}  {frac[2]:7.4f}"
            )
        if len(lines) == 2:
            lines.append(" no atoms")
        return lines

    def positions_text(self) -> str:
        lines = self.position_lines()
        self._clamp_position_scroll()
        visible = lines[self.position_scroll : self.position_scroll + POSITIONS_PANEL_LINES]
        return "\n".join(visible)

    def bond_lines(self) -> list[str]:
        lines = [
            " idx   atom i       atom j       length    offset",
            "--------------------------------------------------------",
        ]
        records = bond_records(self.scene)
        for index, (i, j, distance, offset) in enumerate(records, start=1):
            atom_i = f"{self.scene.symbols[i]}[{i + 1}]"
            atom_j = f"{self.scene.symbols[j]}[{j + 1}]"
            offset_text = f"({offset[0]},{offset[1]},{offset[2]})"
            lines.append(f"{index:4d}  {atom_i:<11}  {atom_j:<11}  {distance:7.3f}  {offset_text}")
        if len(lines) == 2:
            lines.append(" no bonds")
        return lines

    def bond_text(self) -> str:
        lines = self.bond_lines()
        self._clamp_bond_scroll()
        visible = lines[self.bond_scroll : self.bond_scroll + POSITIONS_PANEL_LINES]
        return "\n".join(visible)


def element_legend(scene) -> str:
    seen: set[str] = set()
    entries: list[str] = []
    for atom in scene.atoms:
        if atom.symbol in seen:
            continue
        seen.add(atom.symbol)
        entries.append(f"{atom.symbol}={atomic_names[atom.number]}")
    if not entries:
        return "Legend: none"
    return "Legend: " + "  ".join(entries)


def wrapped_line_count(text: str, width: int) -> int:
    usable_width = max(width, 1)
    if not text:
        return 0
    total = 0
    for line in text.splitlines() or [""]:
        total += max((len(line) + usable_width - 1) // usable_width, 1)
    return total


def render_calibration_lines(width: int, height: int, aspect_ratio: float) -> list[str]:
    width = max(width, 1)
    height = max(height, 1)
    center_x = (width * 2 - 1) / 2.0
    center_y = (height * 4 - 1) / 2.0
    radius = max(min((width - 2) / max(2.0 * aspect_ratio, 1e-12), (height - 2) / 2.0) * 0.8, 1.0)
    rows = [[" " for _ in range(width)] for _ in range(height)]

    for cell_y in range(height):
        for cell_x in range(width):
            mask = 0
            for dot_x in range(2):
                for dot_y in range(4):
                    sub_x = cell_x * 2 + dot_x
                    sub_y = cell_y * 4 + dot_y
                    dx = (sub_x - center_x) / max(2.0 * aspect_ratio, 1e-12)
                    dy = (sub_y - center_y) / 4.0
                    distance = float(np.hypot(dx, dy))
                    if abs(distance - radius) <= 0.18:
                        mask |= BRAILLE_DOTS[(dot_x, dot_y)]
            if mask:
                rows[cell_y][cell_x] = chr(BRAILLE_BASE + mask)

    rows[int(round((height - 1) / 2.0))][int(round((width - 1) / 2.0))] = "+"
    return ["".join(row) for row in rows]


def render_calibration_formatted(width: int, height: int, aspect_ratio: float) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    for row in render_calibration_lines(width, height, aspect_ratio):
        fragments.append(("", row))
        fragments.append(("", "\n"))
    if fragments:
        fragments.pop()
    return fragments


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points: list[tuple[int, int]] = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def _braille_line_points(start: tuple[float, float], end: tuple[float, float]) -> list[tuple[int, int]]:
    sub_start = (int(round(start[0] * 2)), int(round(start[1] * 4)))
    sub_end = (int(round(end[0] * 2)), int(round(end[1] * 4)))
    return _line_points(sub_start, sub_end)


def direction_endpoints(
    vectors,
    labels: tuple[str, str, str],
    camera: CameraState,
    width: int = AXIS_WIDGET_WIDTH,
    height: int = AXIS_WIDGET_HEIGHT,
    aspect_ratio: float = AXIS_ASPECT_RATIO,
) -> list[tuple[str, float, float, float]]:
    width = max(width, 9)
    height = max(height, 7)
    center_x = width // 2
    center_y = height // 2

    projected = []
    for vector in vectors[:3]:
        vector = np.asarray(vector, dtype=float)
        direction = vector / max((sum(component * component for component in vector) ** 0.5), 1e-12)
        rotated = camera.orientation @ direction
        projected.append((rotated[0] * aspect_ratio, -rotated[1], rotated[2]))

    usable_x = max((width - 2) / 2.0, 1.0)
    usable_y = max((height - 2) / 2.0, 1.0)
    radius = min(usable_x, usable_y)
    max_planar_norm = max(((x * x + y * y) ** 0.5 for x, y, _ in projected), default=1.0)
    scale = radius * AXIS_LENGTH_SCALE / max(max_planar_norm, 1e-12)

    endpoints: list[tuple[str, float, float, float]] = []
    for label, (x, y, z) in zip(labels, projected):
        planar_norm = (x * x + y * y) ** 0.5
        if planar_norm > 1e-9:
            end_xf = center_x + x * scale
            end_yf = center_y + y * scale
        else:
            end_xf = float(center_x)
            end_yf = float(center_y)
        if int(round(end_xf)) == center_x and int(round(end_yf)) == center_y:
            end_xf = float(center_x + (1 if z >= 0.0 else -1))
            end_yf = float(center_y)
        end_xf = min(max(end_xf, 1.0), width - 2.0)
        end_yf = min(max(end_yf, 1.0), height - 2.0)
        endpoints.append((label, end_xf, end_yf, z))
    return endpoints


def lattice_direction_endpoints(
    cell,
    camera: CameraState,
    width: int = AXIS_WIDGET_WIDTH,
    height: int = AXIS_WIDGET_HEIGHT,
    aspect_ratio: float = AXIS_ASPECT_RATIO,
) -> list[tuple[str, float, float, float]]:
    cell_array = cell if cell is not None else ()
    vectors = [
        vector for vector in cell_array if len(vector) == 3 and any(abs(component) > 1e-12 for component in vector)
    ]
    if not vectors:
        vectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    return direction_endpoints(vectors, ("a", "b", "c"), camera, width, height, aspect_ratio)


def cartesian_direction_endpoints(
    camera: CameraState,
    width: int = AXIS_WIDGET_WIDTH,
    height: int = AXIS_WIDGET_HEIGHT,
    aspect_ratio: float = AXIS_ASPECT_RATIO,
) -> list[tuple[str, float, float, float]]:
    return direction_endpoints(
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("x", "y", "z"),
        camera,
        width,
        height,
        aspect_ratio,
    )


def _render_direction_widget(
    title: str,
    endpoints: list[tuple[str, float, float, float]],
    width: int,
    height: int,
) -> str:
    width = max(width, 9)
    height = max(height, 7)
    buffer = [[" " for _ in range(width)] for _ in range(height)]
    braille_masks = [[0 for _ in range(width)] for _ in range(height)]

    for x in range(width):
        buffer[0][x] = "-"
        buffer[height - 1][x] = "-"
    for y in range(height):
        buffer[y][0] = "|"
        buffer[y][width - 1] = "|"
    buffer[0][0] = "+"
    buffer[0][width - 1] = "+"
    buffer[height - 1][0] = "+"
    buffer[height - 1][width - 1] = "+"

    start_col = max((width - len(title)) // 2, 1)
    for index, char in enumerate(title[: width - 2]):
        buffer[0][start_col + index] = char

    center_x = width // 2
    center_y = height // 2
    buffer[center_y][center_x] = "o"
    label_positions: list[tuple[str, int, int]] = []

    for label, end_xf, end_yf, z in endpoints:
        end_x = int(round(end_xf))
        end_y = int(round(end_yf))
        if end_x == center_x and end_y == center_y:
            end_x = center_x + (1 if z >= 0.0 else -1)
            end_xf = float(end_x)
            end_yf = float(end_y)
        end_x = min(max(end_x, 1), width - 2)
        end_y = min(max(end_y, 1), height - 2)
        end_xf = min(max(end_xf, 1.0), width - 2.0)
        end_yf = min(max(end_yf, 1.0), height - 2.0)
        for sub_x, sub_y in _braille_line_points((float(center_x), float(center_y)), (end_xf, end_yf))[1:-1]:
            cell_x = sub_x // 2
            cell_y = sub_y // 4
            dot_x = sub_x % 2
            dot_y = sub_y % 4
            if not (1 <= cell_x < width - 1 and 1 <= cell_y < height - 1):
                continue
            braille_masks[cell_y][cell_x] |= BRAILLE_DOTS[(dot_x, dot_y)]
        label_positions.append((label, end_x, end_y))

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if braille_masks[y][x] and buffer[y][x] == " ":
                buffer[y][x] = chr(BRAILLE_BASE + braille_masks[y][x])

    buffer[center_y][center_x] = "o"
    for label, end_x, end_y in label_positions:
        buffer[end_y][end_x] = label

    return "\n".join("".join(row) for row in buffer)


def render_lattice_direction_widget(
    cell,
    camera: CameraState,
    width: int = AXIS_WIDGET_WIDTH,
    height: int = AXIS_WIDGET_HEIGHT,
    aspect_ratio: float = AXIS_ASPECT_RATIO,
) -> str:
    return _render_direction_widget(
        " abc dirs ", lattice_direction_endpoints(cell, camera, width, height, aspect_ratio), width, height
    )


def render_cartesian_direction_widget(
    camera: CameraState,
    width: int = AXIS_WIDGET_WIDTH,
    height: int = AXIS_WIDGET_HEIGHT,
    aspect_ratio: float = AXIS_ASPECT_RATIO,
) -> str:
    return _render_direction_widget(
        " xyz dirs ", cartesian_direction_endpoints(camera, width, height, aspect_ratio), width, height
    )


def build_application(state: ViewerState) -> Application:
    body_control = FormattedTextControl(
        lambda: state.render_formatted(app.output.get_size().columns, app.output.get_size().rows)
    )
    info_control = FormattedTextControl(lambda: state.info_text())
    axis_control = FormattedTextControl(lambda: state.axis_text())
    cartesian_axis_control = FormattedTextControl(lambda: state.cartesian_axis_text())
    positions_control = FormattedTextControl(lambda: state.positions_text())
    bond_control = FormattedTextControl(lambda: state.bond_text())
    footer_control = FormattedTextControl(lambda: state.status())
    help_control = FormattedTextControl(lambda: state.help_text())

    info_text_window = Window(content=info_control, always_hide_cursor=True)
    axis_window = ConditionalContainer(
        content=Window(
            content=axis_control,
            width=Dimension(preferred=AXIS_WIDGET_WIDTH, min=AXIS_WIDGET_WIDTH, max=AXIS_WIDGET_WIDTH),
            always_hide_cursor=True,
        ),
        filter=Condition(lambda: state.camera.show_abc_panel),
    )
    cartesian_axis_window = ConditionalContainer(
        content=Window(
            content=cartesian_axis_control,
            width=Dimension(preferred=AXIS_WIDGET_WIDTH, min=AXIS_WIDGET_WIDTH, max=AXIS_WIDGET_WIDTH),
            always_hide_cursor=True,
        ),
        filter=Condition(lambda: state.camera.show_xyz_panel),
    )
    info = VSplit([info_text_window, axis_window, cartesian_axis_window], height=INFO_PANEL_LINES)
    body = FloatContainer(
        content=Window(content=body_control, always_hide_cursor=True),
        floats=[
            Float(
                left=0,
                right=0,
                bottom=0,
                content=ConditionalContainer(
                    content=Window(
                        content=positions_control,
                        height=POSITIONS_PANEL_LINES,
                        always_hide_cursor=True,
                    ),
                    filter=Condition(lambda: state.show_positions),
                ),
            ),
            Float(
                left=0,
                right=0,
                bottom=0,
                content=ConditionalContainer(
                    content=Window(
                        content=bond_control,
                        height=POSITIONS_PANEL_LINES,
                        always_hide_cursor=True,
                    ),
                    filter=Condition(lambda: state.show_bond_lengths),
                ),
            ),
        ],
    )
    footer = Window(
        content=footer_control,
        height=Dimension(min=1),
        dont_extend_height=True,
        wrap_lines=True,
        always_hide_cursor=True,
    )
    help_window = ConditionalContainer(
        content=Window(
            content=help_control,
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
            always_hide_cursor=True,
        ),
        filter=Condition(lambda: state.camera.show_help),
    )

    layout = Layout(HSplit([info, body, footer, help_window]))
    bindings = KeyBindings()

    @bindings.add("q")
    @bindings.add("c-c")
    def _quit(event) -> None:
        event.app.exit()

    @bindings.add("left")
    def _left(event) -> None:
        if state.calibration_mode:
            state.adjust_aspect_ratio(-ASPECT_RATIO_STEP)
            event.app.invalidate()
            return
        state.camera = with_rotation(state.camera, dx=-0.12, dy=0.0)
        event.app.invalidate()

    @bindings.add("right")
    def _right(event) -> None:
        if state.calibration_mode:
            state.adjust_aspect_ratio(ASPECT_RATIO_STEP)
            event.app.invalidate()
            return
        state.camera = with_rotation(state.camera, dx=0.12, dy=0.0)
        event.app.invalidate()

    @bindings.add("up")
    def _up(event) -> None:
        if state.show_positions or state.show_bond_lengths:
            state.scroll_overlay(-1)
            event.app.invalidate()
            return
        state.camera = with_rotation(state.camera, dx=0.0, dy=-0.12)
        event.app.invalidate()

    @bindings.add("down")
    def _down(event) -> None:
        if state.show_positions or state.show_bond_lengths:
            state.scroll_overlay(1)
            event.app.invalidate()
            return
        state.camera = with_rotation(state.camera, dx=0.0, dy=0.12)
        event.app.invalidate()

    @bindings.add("a")
    def _toggle_autorotate(event) -> None:
        state.toggle_autorotate()
        event.app.invalidate()

    @bindings.add("p")
    def _toggle_positions(event) -> None:
        state.toggle_positions()
        event.app.invalidate()

    @bindings.add("B")
    def _toggle_bond_lengths(event) -> None:
        state.toggle_bond_lengths()
        event.app.invalidate()

    @bindings.add("t")
    def _toggle_frame_autoplay(event) -> None:
        state.toggle_frame_autoplay()
        event.app.invalidate()

    @bindings.add("T")
    def _toggle_calibration(event) -> None:
        state.toggle_calibration()
        event.app.invalidate()

    @bindings.add("j")
    def _scroll_positions_down(event) -> None:
        if not (state.show_positions or state.show_bond_lengths):
            return
        state.scroll_overlay(1)
        event.app.invalidate()

    @bindings.add("k")
    def _scroll_positions_up(event) -> None:
        if not (state.show_positions or state.show_bond_lengths):
            return
        state.scroll_overlay(-1)
        event.app.invalidate()

    @bindings.add("s-left")
    def _pan_left(event) -> None:
        state.camera = with_pan(state.camera, dx=-2.0, dy=0.0)
        event.app.invalidate()

    @bindings.add("s-right")
    def _pan_right(event) -> None:
        state.camera = with_pan(state.camera, dx=2.0, dy=0.0)
        event.app.invalidate()

    @bindings.add("s-up")
    def _pan_up(event) -> None:
        state.camera = with_pan(state.camera, dx=0.0, dy=1.0)
        event.app.invalidate()

    @bindings.add("s-down")
    def _pan_down(event) -> None:
        state.camera = with_pan(state.camera, dx=0.0, dy=-1.0)
        event.app.invalidate()

    @bindings.add("+")
    @bindings.add("=")
    def _zoom_in(event) -> None:
        if state.calibration_mode:
            state.adjust_aspect_ratio(ASPECT_RATIO_STEP)
            event.app.invalidate()
            return
        state.camera = with_zoom(state.camera, 1.12)
        event.app.invalidate()

    @bindings.add("-")
    def _zoom_out(event) -> None:
        if state.calibration_mode:
            state.adjust_aspect_ratio(-ASPECT_RATIO_STEP)
            event.app.invalidate()
            return
        state.camera = with_zoom(state.camera, 0.9)
        event.app.invalidate()

    @bindings.add("r")
    def _begin_repeat(event) -> None:
        state.begin_repeat_command()
        event.app.invalidate()

    @bindings.add("c-r")
    def _reset(event) -> None:
        state.reset_viewer()
        event.app.invalidate()

    @bindings.add("escape", filter=Condition(lambda: state.pending_repeat_command is not None))
    def _cancel_repeat(event) -> None:
        state.cancel_repeat_command()
        event.app.invalidate()

    @bindings.add("backspace", filter=Condition(lambda: state.pending_repeat_command is not None))
    def _repeat_backspace(event) -> None:
        if not state.pending_repeat_command:
            state.cancel_repeat_command()
        else:
            state.pending_repeat_command = state.pending_repeat_command[:-1]
            state.status_message = "repeat: type rXYZ, for example r222"
        event.app.invalidate()

    def _append_repeat_digit_if_active(digit: str) -> bool:
        if state.pending_repeat_command is None:
            return False
        state.append_repeat_digit(digit)
        return True

    @bindings.add("<any>", filter=Condition(lambda: state.pending_repeat_command is not None))
    def _repeat_digits(event) -> None:
        key = event.key_sequence[0].key
        if key is None:
            return
        state.append_repeat_digit(key)
        event.app.invalidate()

    @bindings.add("c")
    def _toggle_cell(event) -> None:
        state.camera = toggle_flag(state.camera, "show_cell")
        event.app.invalidate()

    @bindings.add("C")
    def _toggle_color(event) -> None:
        state.camera = toggle_flag(state.camera, "show_color")
        event.app.invalidate()

    @bindings.add("b")
    def _toggle_bonds(event) -> None:
        state.camera = toggle_flag(state.camera, "show_bonds")
        event.app.invalidate()

    @bindings.add("l")
    def _toggle_labels(event) -> None:
        state.camera = toggle_flag(state.camera, "show_labels")
        event.app.invalidate()

    @bindings.add("1")
    def _toggle_abc_panel(event) -> None:
        if _append_repeat_digit_if_active("1"):
            event.app.invalidate()
            return
        state.camera = toggle_flag(state.camera, "show_abc_panel")
        event.app.invalidate()

    @bindings.add("2")
    def _toggle_xyz_panel(event) -> None:
        if _append_repeat_digit_if_active("2"):
            event.app.invalidate()
            return
        state.camera = toggle_flag(state.camera, "show_xyz_panel")
        event.app.invalidate()

    @bindings.add("m")
    def _toggle_line_mode(event) -> None:
        state.camera = cycle_line_mode(state.camera)
        event.app.invalidate()

    @bindings.add("s")
    def _toggle_spheres(event) -> None:
        state.camera = toggle_flag(state.camera, "show_spheres")
        event.app.invalidate()

    @bindings.add("[")
    def _prev_frame(event) -> None:
        state.step_frame(-1)
        event.app.invalidate()

    @bindings.add("]")
    def _next_frame(event) -> None:
        state.step_frame(1)
        event.app.invalidate()

    @bindings.add("?")
    def _toggle_help(event) -> None:
        state.camera = toggle_flag(state.camera, "show_help")
        event.app.invalidate()

    @bindings.add("x")
    def _view_x(event) -> None:
        state.camera = view_along(state.camera, "x")
        event.app.invalidate()

    @bindings.add("y")
    def _view_y(event) -> None:
        state.camera = view_along(state.camera, "y")
        event.app.invalidate()

    @bindings.add("z")
    def _view_z(event) -> None:
        state.camera = view_along(state.camera, "z")
        event.app.invalidate()

    async def _autorotate_loop() -> None:
        while True:
            now = time.monotonic()
            rotated = state.tick_autorotate()
            advanced = state.tick_frame_autoplay(now)
            if rotated or advanced:
                app.invalidate()
            await asyncio.sleep(AUTOROTATE_INTERVAL)

    def _start_background_tasks() -> None:
        app.create_background_task(_autorotate_loop())

    app = Application(layout=layout, key_bindings=bindings, full_screen=True)
    app.pre_run_callables.append(_start_background_tasks)
    return app


def run_viewer(
    paths: Sequence[str | Path],
    repeat: tuple[int, int, int],
    image_number: str = ":",
    show_cell: bool = True,
    symprec: float = 1e-5,
    show_color: bool = False,
) -> None:
    state = ViewerState(
        paths,
        repeat=repeat,
        show_cell=show_cell,
        symprec=symprec,
        show_color=show_color,
        image_number=image_number,
    )
    app = build_application(state)
    app.run()
