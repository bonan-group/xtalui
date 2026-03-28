"""Projection and terminal rendering helpers for xtalui scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from xtalui.scene import (
    CameraState,
    RenderOptions,
    RenderPrimitive,
    SceneData,
    scene_radius,
    transformed_cell_axis_labels,
    transformed_bond_segments,
    transformed_cell_edges,
    transformed_positions,
)


DEPTH_GLYPHS = "˙·∘◦•●"
HORIZONTAL_EDGE = "─"
VERTICAL_EDGE = "│"
DIAGONAL_ASC_EDGE = "╲"
DIAGONAL_DESC_EDGE = "╱"
ORTHOGONAL_CROSS = "┼"
DIAGONAL_CROSS = "╳"
EDGE_GLYPHS = frozenset(
    {
        HORIZONTAL_EDGE,
        VERTICAL_EDGE,
        DIAGONAL_ASC_EDGE,
        DIAGONAL_DESC_EDGE,
        ORTHOGONAL_CROSS,
        DIAGONAL_CROSS,
    }
)
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}
SPHERE_GLYPH = "●"
TextFragment = tuple[str, str]
ScreenPoint3D = tuple[float, float, float]


@dataclass(frozen=True)
class Viewport:
    """Terminal drawing area measured in character cells."""

    width: int
    height: int


@dataclass(frozen=True)
class ProjectedSegment:
    """Projected line segment with rendering priority."""

    start: ScreenPoint3D
    end: ScreenPoint3D
    priority: int


@dataclass(frozen=True)
class ProjectedSphere:
    """Projected atom footprint used for braille sphere rendering."""

    x: float
    y: float
    z: float
    radius: float
    style: str


def _project_coords(
    point: np.ndarray, viewport: Viewport, scale: float, camera: CameraState, aspect_ratio: float
) -> tuple[float, float, float]:
    x = point[0] * scale * aspect_ratio + camera.pan_x
    y = point[1] * scale + camera.pan_y
    z = point[2]
    screen_x = viewport.width / 2 + x
    screen_y = viewport.height / 2 - y
    return screen_x, screen_y, z


def _project_point(
    point: np.ndarray, viewport: Viewport, scale: float, camera: CameraState, aspect_ratio: float
) -> tuple[int, int, float]:
    screen_x, screen_y, z = _project_coords(point, viewport, scale, camera, aspect_ratio)
    screen_x = int(round(screen_x))
    screen_y = int(round(screen_y))
    return screen_x, screen_y, z


def _depth_char(z: float, z_min: float, z_max: float) -> str:
    if z_max <= z_min:
        return DEPTH_GLYPHS[-2]
    ratio = (z - z_min) / (z_max - z_min)
    index = int(round(ratio * (len(DEPTH_GLYPHS) - 1)))
    return DEPTH_GLYPHS[index]


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int, float]]:
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
        depth = i / steps if steps else 0.0
        points.append((x0, y0, depth))
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


def _edge_char(start: tuple[int, int], end: tuple[int, int]) -> str:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    if abs_dx >= 2 * abs_dy:
        return HORIZONTAL_EDGE
    if abs_dy >= 2 * abs_dx:
        return VERTICAL_EDGE
    return DIAGONAL_ASC_EDGE if dx * dy >= 0 else DIAGONAL_DESC_EDGE


def _merged_edge_char(existing_char: str, new_char: str) -> str:
    if existing_char == new_char:
        return existing_char
    diagonal_chars = {DIAGONAL_ASC_EDGE, DIAGONAL_DESC_EDGE}
    if existing_char in diagonal_chars and new_char in diagonal_chars:
        return DIAGONAL_CROSS
    return ORTHOGONAL_CROSS


def _place_cell_label(
    label: str,
    x: int,
    y: int,
    z: float,
    occupied: set[tuple[int, int]],
    viewport: Viewport,
) -> RenderPrimitive:
    fallback_offsets = {
        "a": [(1, 0), (2, 0), (1, -1), (1, 1)],
        "b": [(0, -1), (0, -2), (1, -1), (-1, -1)],
        "c": [(1, 1), (2, 1), (1, 2), (2, 0)],
    }
    candidates = [(x, y)] + [(x + dx, y + dy) for dx, dy in fallback_offsets.get(label, [])]
    for candidate_x, candidate_y in candidates:
        if not (0 <= candidate_x < viewport.width and 0 <= candidate_y < viewport.height):
            continue
        if (candidate_x, candidate_y) in occupied:
            continue
        occupied.add((candidate_x, candidate_y))
        priority = 19 if label == "o" else 18
        return RenderPrimitive(x=candidate_x, y=candidate_y, z=z, char=label, priority=priority)
    occupied.add((x, y))
    priority = 19 if label == "o" else 18
    return RenderPrimitive(x=x, y=y, z=z, char=label, priority=priority)


def _element_style(atomic_number: int) -> str:
    red, green, blue = jmol_colors[atomic_number]
    return f"fg:#{int(round(red * 255)):02x}{int(round(green * 255)):02x}{int(round(blue * 255)):02x}"


def _atom_radius_in_cells(atomic_number: int, scale: float, options: RenderOptions) -> float:
    return max(float(covalent_radii[atomic_number]) * scale * options.atom_radius_scale, 0.75)


def _sphere_primitives(
    center_x: int,
    center_y: int,
    z: float,
    char: str,
    atomic_number: int,
    scale: float,
    options: RenderOptions,
    style: str,
) -> list[RenderPrimitive]:
    radius = _atom_radius_in_cells(atomic_number, scale, options)
    radius_x = max(radius * options.aspect_ratio, 0.75)
    min_dx = int(np.floor(-radius_x))
    max_dx = int(np.ceil(radius_x))
    min_dy = int(np.floor(-radius))
    max_dy = int(np.ceil(radius))
    primitives: list[RenderPrimitive] = []
    for dy in range(min_dy, max_dy + 1):
        for dx in range(min_dx, max_dx + 1):
            norm = (dx / max(radius_x, 1e-12)) ** 2 + (dy / max(radius, 1e-12)) ** 2
            if norm > 1.0:
                continue
            primitives.append(
                RenderPrimitive(x=center_x + dx, y=center_y + dy, z=z, char=char, priority=17, style=style)
            )
    return primitives


def _braille_sphere_rows(
    width: int, height: int, spheres: list[ProjectedSphere], aspect_ratio: float
) -> tuple[list[str], list[list[str]], list[list[float]]]:
    masks = [[0 for _ in range(width)] for _ in range(height)]
    styles = [["" for _ in range(width)] for _ in range(height)]
    depths = [[float("-inf") for _ in range(width)] for _ in range(height)]
    for sphere in sorted(spheres, key=lambda item: item.z):
        radius_x_sub = max(sphere.radius * aspect_ratio * 2.0, 1.0)
        radius_y_sub = max(sphere.radius * 4.0, 1.0)
        center_x_sub = sphere.x * 2.0
        center_y_sub = sphere.y * 4.0
        min_cell_x = max(int(np.floor((center_x_sub - radius_x_sub) / 2.0)), 0)
        max_cell_x = min(int(np.ceil((center_x_sub + radius_x_sub) / 2.0)), width - 1)
        min_cell_y = max(int(np.floor((center_y_sub - radius_y_sub) / 4.0)), 0)
        max_cell_y = min(int(np.ceil((center_y_sub + radius_y_sub) / 4.0)), height - 1)
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                mask = 0
                for dot_x in range(2):
                    for dot_y in range(4):
                        sub_x = cell_x * 2 + dot_x
                        sub_y = cell_y * 4 + dot_y
                        norm = ((sub_x - center_x_sub) / radius_x_sub) ** 2 + (
                            (sub_y - center_y_sub) / radius_y_sub
                        ) ** 2
                        if norm <= 1.0:
                            mask |= BRAILLE_DOTS[(dot_x, dot_y)]
                if not mask or sphere.z < depths[cell_y][cell_x]:
                    continue
                masks[cell_y][cell_x] = mask
                styles[cell_y][cell_x] = sphere.style
                depths[cell_y][cell_x] = sphere.z
    rows: list[str] = []
    for y in range(height):
        row_chars = []
        for x in range(width):
            mask = masks[y][x]
            row_chars.append(chr(BRAILLE_BASE + mask) if mask else " ")
        rows.append("".join(row_chars))
    return rows, styles, depths


def _project_scene(
    scene: SceneData,
    camera: CameraState,
    viewport: Viewport,
    options: RenderOptions,
) -> tuple[list[RenderPrimitive], list[ProjectedSegment], list[ProjectedSphere]]:
    transformed_atoms = transformed_positions(scene, camera)
    bond_segments = transformed_bond_segments(scene, camera) if camera.show_bonds else []
    cell_edges = transformed_cell_edges(scene, camera) if camera.show_cell else []
    cell_axis_labels = transformed_cell_axis_labels(scene, camera) if camera.show_cell else []
    # Keep the fitted scale stable when the cell visibility is toggled.
    # Rendering may hide the frame, but the camera fit stays anchored to the
    # same scene extent so the structure does not appear to zoom.
    extent = scene_radius(scene, include_cell=True)
    usable_width = max(viewport.width - 4, 1)
    usable_height = max(viewport.height - 4, 1)
    scale_x = usable_width / (2.0 * extent * options.aspect_ratio)
    scale_y = usable_height / (2.0 * extent)
    scale = min(scale_x, scale_y) * camera.zoom

    atom_primitives: list[RenderPrimitive] = []
    projected_spheres: list[ProjectedSphere] = []
    z_values = transformed_atoms[:, 2] if len(transformed_atoms) else np.array([0.0])
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))

    for index, point in enumerate(transformed_atoms):
        screen_x, screen_y, z = _project_coords(point, viewport, scale, camera, options.aspect_ratio)
        x = int(round(screen_x))
        y = int(round(screen_y))
        char = _depth_char(z, z_min, z_max)
        style = _element_style(scene.atoms[index].number) if camera.show_color else ""
        if camera.show_spheres:
            atom_primitives.extend(
                _sphere_primitives(x, y, z, SPHERE_GLYPH, scene.atoms[index].number, scale, options, style)
            )
            projected_spheres.append(
                ProjectedSphere(
                    x=screen_x,
                    y=screen_y,
                    z=z,
                    radius=_atom_radius_in_cells(scene.atoms[index].number, scale, options),
                    style=style,
                )
            )
            if camera.show_labels:
                atom_primitives.append(
                    RenderPrimitive(x=x, y=y, z=z, char=scene.symbols[index], priority=21, style=style)
                )
        else:
            if camera.show_labels:
                char = scene.symbols[index]
            atom_primitives.append(RenderPrimitive(x=x, y=y, z=z, char=char, priority=20, style=style))

    occupied_cell_labels: set[tuple[int, int]] = set()
    for label, point in cell_axis_labels:
        x, y, z = _project_point(point, viewport, scale, camera, options.aspect_ratio)
        atom_primitives.append(_place_cell_label(label, x, y, z, occupied_cell_labels, viewport))

    projected_segments: list[ProjectedSegment] = []
    for priority, segments in ((15, bond_segments), (10, cell_edges)):
        for start, end in segments:
            projected_segments.append(
                ProjectedSegment(
                    start=_project_coords(start, viewport, scale, camera, options.aspect_ratio),
                    end=_project_coords(end, viewport, scale, camera, options.aspect_ratio),
                    priority=priority,
                )
            )
    return atom_primitives, projected_segments, projected_spheres


def _unicode_line_primitives(segments: list[ProjectedSegment]) -> list[RenderPrimitive]:
    primitives: list[RenderPrimitive] = []
    for segment in segments:
        sx, sy, sz = segment.start
        ex, ey, ez = segment.end
        start = (int(round(sx)), int(round(sy)))
        end = (int(round(ex)), int(round(ey)))
        edge_char = _edge_char(start, end)
        for x, y, ratio in _line_points(start, end):
            z = float((1.0 - ratio) * sz + ratio * ez)
            primitives.append(RenderPrimitive(x=x, y=y, z=z, char=edge_char, priority=segment.priority))
    return primitives


def _braille_line_rows(width: int, height: int, segments: list[ProjectedSegment]) -> list[str]:
    masks = [[0 for _ in range(width)] for _ in range(height)]
    for segment in segments:
        sx, sy, _ = segment.start
        ex, ey, _ = segment.end
        sub_start = (int(round(sx * 2)), int(round(sy * 4)))
        sub_end = (int(round(ex * 2)), int(round(ey * 4)))
        for sub_x, sub_y, _ in _line_points(sub_start, sub_end):
            cell_x = sub_x // 2
            cell_y = sub_y // 4
            dot_x = sub_x % 2
            dot_y = sub_y % 4
            if not (0 <= cell_x < width and 0 <= cell_y < height):
                continue
            masks[cell_y][cell_x] |= BRAILLE_DOTS[(dot_x, dot_y)]
    rows: list[str] = []
    for y in range(height):
        row_chars = []
        for x in range(width):
            mask = masks[y][x]
            row_chars.append(chr(BRAILLE_BASE + mask) if mask else " ")
        rows.append("".join(row_chars))
    return rows


def build_primitives(
    scene: SceneData, camera: CameraState, viewport: Viewport, options: RenderOptions | None = None
) -> list[RenderPrimitive]:
    """Project a scene into draw-order-aware terminal primitives."""

    options = options or RenderOptions()
    atoms, segments, _ = _project_scene(scene, camera, viewport, options)
    return atoms + _unicode_line_primitives(segments)


def render_ascii(
    scene: SceneData, camera: CameraState, viewport: Viewport, options: RenderOptions | None = None
) -> list[str]:
    """Render a scene as plain text rows without style metadata."""

    return "".join(text for _, text in render_formatted(scene, camera, viewport, options)).split("\n")


def render_formatted(
    scene: SceneData, camera: CameraState, viewport: Viewport, options: RenderOptions | None = None
) -> list[TextFragment]:
    """Render a scene as prompt-toolkit style/text fragments."""

    options = options or RenderOptions()
    width = max(viewport.width, 1)
    height = max(viewport.height, 1)
    atom_primitives, segments, spheres = _project_scene(scene, camera, viewport, options)
    if camera.line_mode == "braille":
        buffer = [list(row) for row in _braille_line_rows(width, height, segments)]
    else:
        buffer = [[" " for _ in range(width)] for _ in range(height)]
    styles = [["" for _ in range(width)] for _ in range(height)]
    depth = [[float("-inf") for _ in range(width)] for _ in range(height)]
    priority = [[-1 for _ in range(width)] for _ in range(height)]
    if camera.line_mode == "braille" and camera.show_spheres:
        sphere_rows, sphere_styles, sphere_depths = _braille_sphere_rows(width, height, spheres, options.aspect_ratio)
        for y in range(height):
            for x in range(width):
                sphere_char = sphere_rows[y][x]
                if sphere_char == " ":
                    continue
                buffer[y][x] = sphere_char
                styles[y][x] = sphere_styles[y][x]
                depth[y][x] = sphere_depths[y][x]
                priority[y][x] = 17
    primitives = atom_primitives + ([] if camera.line_mode == "braille" else _unicode_line_primitives(segments))
    if camera.line_mode == "braille" and camera.show_spheres:
        primitives = [primitive for primitive in primitives if primitive.priority != 17]
    for primitive in sorted(primitives, key=lambda item: (item.z, item.priority)):
        y = primitive.y
        if not (0 <= y < height):
            continue
        start_x = primitive.x
        if primitive.priority >= 20 and len(primitive.char) > 1:
            start_x = primitive.x - (len(primitive.char) - 1) // 2
        for offset, char in enumerate(primitive.char):
            x = start_x + offset
            if not (0 <= x < width):
                continue
            existing_char = buffer[y][x]
            if (
                primitive.priority == priority[y][x]
                and primitive.priority >= 10
                and existing_char in EDGE_GLYPHS
                and char in EDGE_GLYPHS
                and existing_char != char
            ):
                buffer[y][x] = _merged_edge_char(existing_char, char)
                depth[y][x] = max(depth[y][x], primitive.z)
                continue
            if primitive.priority > priority[y][x] or primitive.z >= depth[y][x]:
                buffer[y][x] = char
                styles[y][x] = primitive.style
                depth[y][x] = primitive.z
                priority[y][x] = primitive.priority
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
