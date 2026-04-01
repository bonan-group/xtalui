"""Scene loading, geometry transforms, and viewer camera state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from numpy.typing import NDArray

from xtalui.abacus_stru import looks_like_abacus_stru, read_abacus_stru

try:
    import spglib
except ImportError:  # pragma: no cover - exercised in integration, not unit tests
    spglib = None


AtomPositions = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
Vector3 = NDArray[np.float64]
BondRecord = tuple[int, int, float, tuple[int, int, int]]
PathInput = str | Path
StructurePaths = PathInput | Sequence[PathInput]


@dataclass(frozen=True)
class SceneData:
    """Immutable structure data prepared for terminal rendering."""

    atoms: Atoms
    positions: AtomPositions
    symbols: list[str]
    cell: Matrix3
    title: str


@dataclass(frozen=True)
class StructureInfo:
    """Derived structural metadata displayed in the viewer info panel."""

    formula_full: str
    formula_reduced: str
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    volume: float
    spacegroup: str


@dataclass(frozen=True)
class CameraState:
    """Interactive view state used to project a structure into the terminal."""

    orientation: Matrix3 = field(default_factory=lambda: default_orientation())
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 1.0
    show_abc_panel: bool = True
    show_xyz_panel: bool = True
    line_mode: str = "braille"
    show_cell: bool = True
    show_bonds: bool = True
    show_labels: bool = False
    show_indices: bool = False
    show_color: bool = False
    show_spheres: bool = False
    show_help: bool = False


@dataclass(frozen=True)
class RenderOptions:
    """Rendering parameters that control projected atom appearance."""

    atom_radius_scale: float = 0.55
    depth_scale: float = 1.0
    aspect_ratio: float = 2.0
    selected_indices: frozenset[int] = frozenset()
    selection_style: str = "fg:#ffd54f"


@dataclass(frozen=True)
class RenderPrimitive:
    """Projected character cell with z-order and optional style."""

    x: int
    y: int
    z: float
    char: str
    priority: int
    style: str = ""


def load_structure(
    paths: StructurePaths, repeat: tuple[int, int, int] = (1, 1, 1), image_number: str = ":"
) -> SceneData:
    """Load the first structure frame from one or more path inputs."""

    return load_structures(paths, repeat, image_number=image_number)[0]


def load_structures(
    paths: StructurePaths, repeat: tuple[int, int, int] = (1, 1, 1), image_number: str = ":"
) -> list[SceneData]:
    """Load one or more structures, expanding repeats and frame selections."""

    scenes: list[SceneData] = []
    for path_input in _normalize_paths(paths):
        path, file_image_number, title = _resolve_path_input(path_input, image_number)
        atoms_series = read_structure_series(path, image_number=file_image_number)
        scenes.extend(_scene_from_atoms(_repeat_atoms(atoms, repeat), title) for atoms in atoms_series)
    return scenes


def _normalize_paths(paths: StructurePaths) -> tuple[PathInput, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    normalized = tuple(paths)
    if not normalized:
        raise ValueError("at least one structure path is required")
    return normalized


def _resolve_path_input(path_input: PathInput, default_image_number: str) -> tuple[Path, str, str]:
    if isinstance(path_input, Path):
        path = path_input
        image_number = default_image_number
    else:
        raw = str(path_input)
        path_text, image_override = _split_path_and_image_number(raw)
        path = Path(path_text)
        image_number = image_override if image_override is not None else default_image_number
    title = path.name if image_number == ":" else f"{path.name}@{image_number}"
    return path, image_number, title


def _split_path_and_image_number(raw: str) -> tuple[str, str | None]:
    """Split `path@slice` inputs without mis-parsing real paths that contain `@`."""

    if "@" not in raw or Path(raw).exists():
        return raw, None
    path_text, image_number = raw.rsplit("@", 1)
    if not path_text or not image_number:
        return raw, None
    try:
        _parse_image_number(image_number)
    except ValueError:
        return raw, None
    return path_text, image_number


def _parse_image_number(image_number: str) -> int | slice:
    """Parse ASE-style image selectors expressed as an index or slice."""

    if ":" not in image_number:
        return int(image_number)
    parts = image_number.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid image-number slice: {image_number}")
    values = [None if part == "" else int(part) for part in parts]
    while len(values) < 3:
        values.append(None)
    return slice(*values)


def _select_images(atoms_series: Sequence[Atoms], image_number: str) -> list[Atoms]:
    if image_number == ":":
        return list(atoms_series)
    selection = atoms_series[_parse_image_number(image_number)]
    if isinstance(selection, Atoms):
        return [selection]
    return list(selection)


def read_structure_series(path: Path, image_number: str = ":") -> list[Atoms]:
    """Read one file into a list of ASE `Atoms` frames."""

    if looks_like_abacus_stru(path):
        return _select_images([read_abacus_stru(path)], image_number)
    atoms = read(path, index=image_number)
    if isinstance(atoms, Atoms):
        return [atoms]
    if atoms:
        return list(atoms)
    fallback = read(path)
    if isinstance(fallback, Atoms):
        return _select_images([fallback], image_number)
    return _select_images(list(fallback), image_number)


def _repeat_atoms(atoms: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    if repeat == (1, 1, 1):
        return atoms
    return atoms.repeat(repeat)


def _scene_from_atoms(atoms: Atoms, title: str) -> SceneData:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    return SceneData(
        atoms=atoms,
        positions=positions,
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title=title,
    )


def structure_info(scene: SceneData, symprec: float = 1e-5) -> StructureInfo:
    """Compute display metadata for a rendered scene."""

    cell = np.asarray(scene.atoms.cell.array, dtype=float)
    cellpar = tuple(float(value) for value in scene.atoms.cell.cellpar())
    return StructureInfo(
        formula_full=scene.atoms.get_chemical_formula(mode="metal"),
        formula_reduced=scene.atoms.get_chemical_formula(mode="metal", empirical=True),
        lattice_vectors=tuple(tuple(float(component) for component in vector) for vector in cell),
        lengths=cellpar[:3],
        angles=cellpar[3:],
        volume=float(scene.atoms.cell.volume),
        spacegroup=spacegroup_label(scene.atoms, symprec=symprec),
    )


def spacegroup_label(atoms: Atoms, symprec: float = 1e-5) -> str:
    """Return a human-readable space-group label for a structure."""

    if spglib is None:
        return "Unavailable (install spglib)"
    cell = (
        np.asarray(atoms.cell.array, dtype=float),
        np.asarray(atoms.get_scaled_positions(), dtype=float),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
    )
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        return "Unknown"
    if hasattr(dataset, "international"):
        symbol = dataset.international
        number = dataset.number
    else:
        symbol = dataset["international"]
        number = dataset["number"]
    return f"{symbol} (#{number})"


def rotation_matrix(rot_x: float, rot_y: float) -> Matrix3:
    """Build a combined x/y rotation matrix."""

    cx, sx = np.cos(rot_x), np.sin(rot_x)
    cy, sy = np.cos(rot_y), np.sin(rot_y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return ry @ rx


def axis_rotation_matrix(axis: str, angle: float) -> Matrix3:
    """Build a right-handed rotation matrix around a named Cartesian axis."""

    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"unsupported axis: {axis}")


def default_orientation() -> Matrix3:
    """Return the default isometric viewing orientation."""

    # Isometric default: equal axes in a cubic cell project to equal lengths.
    return rotation_matrix(-np.pi / 4.0, np.arcsin(np.tan(np.pi / 6.0)))


def normalize_orientation(matrix: Matrix3) -> Matrix3:
    """Re-orthogonalize a rotation matrix after incremental updates."""

    u, _, vh = np.linalg.svd(matrix)
    return u @ vh


def orientation_for_view(axis: str) -> Matrix3:
    """Return a camera orientation aligned to the requested axis."""

    axis = axis.lower()
    if axis == "x":
        orientation = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    elif axis == "y":
        orientation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    elif axis == "z":
        orientation = np.eye(3)
    else:
        raise ValueError(f"unsupported view axis: {axis}")
    return normalize_orientation(orientation)


def centered_positions(scene: SceneData) -> AtomPositions:
    """Center atomic positions around the rendered scene origin."""

    positions = scene.positions
    if len(positions) == 0:
        return positions
    if np.any(scene.cell):
        center = np.sum(scene.cell, axis=0) / 2.0
    else:
        center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    return positions - center


def cell_corners(scene: SceneData) -> AtomPositions:
    """Return cell corner positions in the centered scene frame."""

    if not np.any(scene.cell):
        return np.empty((0, 3), dtype=float)
    origin = -np.sum(scene.cell, axis=0) / 2.0
    return np.array(
        [
            origin,
            origin + scene.cell[0],
            origin + scene.cell[1],
            origin + scene.cell[2],
            origin + scene.cell[0] + scene.cell[1],
            origin + scene.cell[0] + scene.cell[2],
            origin + scene.cell[1] + scene.cell[2],
            origin + np.sum(scene.cell, axis=0),
        ],
        dtype=float,
    )


def transformed_positions(scene: SceneData, camera: CameraState) -> AtomPositions:
    """Rotate centered atomic positions into camera space."""

    return centered_positions(scene) @ camera.orientation.T


def transformed_cell_edges(scene: SceneData, camera: CameraState) -> list[tuple[Vector3, Vector3]]:
    """Rotate unit-cell edge segments into camera space."""

    if not np.any(scene.cell):
        return []
    corners = cell_corners(scene)
    rotated = corners @ camera.orientation.T
    edge_indices = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return [(rotated[start], rotated[end]) for start, end in edge_indices]


def transformed_cell_axis_labels(scene: SceneData, camera: CameraState) -> list[tuple[str, Vector3]]:
    """Return labeled origin and lattice-axis marker positions in camera space."""

    if not np.any(scene.cell):
        return []
    origin = -np.sum(scene.cell, axis=0) / 2.0
    axis_fraction = 0.22
    label_points = [
        ("o", origin),
        ("a", origin + axis_fraction * scene.cell[0]),
        ("b", origin + axis_fraction * scene.cell[1]),
        ("c", origin + axis_fraction * scene.cell[2]),
    ]
    return [(label, point @ camera.orientation.T) for label, point in label_points]


def transformed_bond_segments(
    scene: SceneData, camera: CameraState, cutoff_scale: float = 1.0
) -> list[tuple[Vector3, Vector3]]:
    """Rotate bond segments into camera space."""

    records = bond_records(scene, cutoff_scale=cutoff_scale)
    positions = centered_positions(scene)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i, j, _, offset in records:
        start = positions[i]
        end = positions[j] + np.dot(offset, scene.cell)
        segments.append((start @ camera.orientation.T, end @ camera.orientation.T))
    return segments


def bond_records(scene: SceneData, cutoff_scale: float = 1.0) -> list[BondRecord]:
    """Return unique bond records using ASE neighbor-list detection."""

    if len(scene.positions) < 2:
        return []
    # Match ASE GUI bond detection: periodic neighbor list with a 1.5x
    # covalent-radii cutoff.
    cutoffs = natural_cutoffs(scene.atoms, mult=1.5 * cutoff_scale)
    indices_i, indices_j, offsets = neighbor_list("ijS", scene.atoms, cutoffs)
    positions = scene.positions
    records: list[BondRecord] = []
    seen_pairs: set[tuple[int, int, tuple[int, int, int]]] = set()
    for i, j, offset in zip(indices_i, indices_j, offsets):
        if i == j:
            continue
        offset_tuple = tuple(int(value) for value in offset)
        pair = (int(i), int(j), offset_tuple)
        reverse_pair = (int(j), int(i), tuple(-value for value in offset_tuple))
        if reverse_pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        start = positions[int(i)]
        end = positions[int(j)] + np.dot(offset, scene.cell)
        distance = float(np.linalg.norm(end - start))
        records.append((int(i), int(j), distance, offset_tuple))
    return records


def with_rotation(camera: CameraState, dx: float, dy: float) -> CameraState:
    """Return a camera rotated by incremental x/y view angles."""

    orientation = camera.orientation
    if dx:
        orientation = axis_rotation_matrix("y", dx) @ orientation
    if dy:
        orientation = axis_rotation_matrix("x", dy) @ orientation
    return replace(camera, orientation=normalize_orientation(orientation))


def with_pan(camera: CameraState, dx: float, dy: float) -> CameraState:
    """Return a camera translated in screen space."""

    return replace(camera, pan_x=camera.pan_x + dx, pan_y=camera.pan_y + dy)


def with_zoom(camera: CameraState, factor: float) -> CameraState:
    """Return a camera with multiplicative zoom clamped to a safe range."""

    return replace(camera, zoom=max(0.1, min(8.0, camera.zoom * factor)))


def toggle_flag(camera: CameraState, flag_name: str) -> CameraState:
    """Flip one boolean display flag on the camera state."""

    current = getattr(camera, flag_name)
    return replace(camera, **{flag_name: not current})


def reset_camera(camera: CameraState) -> CameraState:
    """Reset the camera transform while preserving display toggles."""

    return replace(
        camera,
        orientation=default_orientation(),
        pan_x=0.0,
        pan_y=0.0,
        zoom=1.0,
    )


def view_along(camera: CameraState, axis: str) -> CameraState:
    """Align the camera to look down a named Cartesian axis."""

    return replace(camera, orientation=orientation_for_view(axis))


def cycle_line_mode(camera: CameraState) -> CameraState:
    """Toggle between braille and box-drawing line rendering."""

    next_mode = "unicode" if camera.line_mode == "braille" else "braille"
    return replace(camera, line_mode=next_mode)


def scene_radius(scene: SceneData, include_cell: bool = True) -> float:
    """Estimate a scene radius for view fitting."""

    points = [centered_positions(scene)]
    if include_cell:
        corners = cell_corners(scene)
        if len(corners):
            points.append(corners)
    arrays = [array for array in points if len(array)]
    if not arrays:
        return 1.0
    stacked = np.vstack(arrays)
    radius = np.max(np.linalg.norm(stacked, axis=1))
    return float(radius) if radius > 0.0 else 1.0
