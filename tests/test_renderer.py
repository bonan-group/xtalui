from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.units import Angstrom, Bohr

from xtalui.renderer import (
    BRAILLE_BASE,
    DIAGONAL_ASC_EDGE,
    DIAGONAL_DESC_EDGE,
    DEPTH_GLYPHS,
    HORIZONTAL_EDGE,
    ORTHOGONAL_CROSS,
    RenderOptions,
    VERTICAL_EDGE,
    Viewport,
    _edge_char,
    build_primitives,
    render_ascii,
)
from xtalui.scene import (
    CameraState,
    SceneData,
    load_structure,
    orientation_for_view,
    scene_radius,
    structure_info,
    transformed_bond_segments,
    transformed_cell_edges,
    transformed_positions,
    view_along,
    with_rotation,
    with_zoom,
)


def make_scene() -> SceneData:
    atoms = Atoms(
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=np.diag([3.0, 3.0, 3.0]),
        pbc=True,
    )
    return SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="test",
    )


def test_cif_loading(tmp_path: Path) -> None:
    atoms = Atoms(
        symbols=["Si", "Si"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        cell=np.diag([5.4, 5.4, 5.4]),
        pbc=True,
    )
    path = tmp_path / "si.cif"
    write(path, atoms)
    scene = load_structure(path)
    assert scene.positions.shape == (2, 3)
    assert scene.symbols == ["Si", "Si"]


def test_abacus_stru_loading_with_direct_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "STRU"
    path.write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "Si 28.085 Si.upf",
                "",
                "LATTICE_CONSTANT",
                f"{Angstrom / Bohr}",
                "",
                "LATTICE_VECTORS",
                "2.0 0.0 0.0",
                "0.0 3.0 0.0",
                "0.0 0.0 4.0",
                "",
                "ATOMIC_POSITIONS",
                "Direct",
                "",
                "Si",
                "0.0",
                "2",
                "0.0 0.0 0.0",
                "0.25 0.5 0.75 m 1 1 1",
            ]
        ),
        encoding="utf-8",
    )
    scene = load_structure(path)
    assert scene.symbols == ["Si", "Si"]
    assert np.allclose(scene.cell, np.diag([2.0, 3.0, 4.0]))
    assert np.allclose(scene.positions[1], np.array([0.5, 1.5, 3.0]))


def test_abacus_stru_loading_with_centered_cartesian_angstrom(tmp_path: Path) -> None:
    path = tmp_path / "example.stru"
    path.write_text(
        "\n".join(
            [
                "ATOMIC_SPECIES",
                "C 12.011 C.upf",
                "",
                "LATTICE_CONSTANT",
                f"{Angstrom / Bohr}",
                "",
                "LATTICE_VECTORS",
                "4.0 0.0 0.0",
                "0.0 4.0 0.0",
                "0.0 0.0 10.0",
                "",
                "ATOMIC_POSITIONS",
                "Cartesian_angstrom_center_xy",
                "",
                "C",
                "0.0",
                "1",
                "0.1 -0.2 1.5",
            ]
        ),
        encoding="utf-8",
    )
    scene = load_structure(path)
    assert scene.symbols == ["C"]
    assert np.allclose(scene.positions[0], np.array([2.1, 1.8, 1.5]))


def test_cell_edges_count() -> None:
    scene = make_scene()
    edges = transformed_cell_edges(scene, CameraState())
    assert len(edges) == 12


def test_zoom_changes_projection_density() -> None:
    scene = make_scene()
    viewport = Viewport(width=40, height=20)
    base = build_primitives(scene, CameraState(), viewport)
    zoomed = build_primitives(scene, with_zoom(CameraState(), 1.5), viewport)
    assert max(item.x for item in zoomed) >= max(item.x for item in base)


def test_nearer_atom_wins_depth_buffer() -> None:
    atoms = Atoms(
        symbols=["C", "O"],
        positions=[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="depth",
    )
    camera = CameraState(orientation=np.eye(3), show_cell=False, show_labels=True)
    viewport = Viewport(width=21, height=11)
    rows = render_ascii(scene, camera, viewport)
    primitives = build_primitives(scene, camera, viewport)
    front = max(primitives, key=lambda item: item.z)
    assert rows[front.y][front.x] == "O"


def test_scene_radius_is_rotation_invariant() -> None:
    scene = make_scene()
    base_radius = scene_radius(scene)
    assert base_radius > 0.0


def test_rotation_keeps_camera_orthonormal() -> None:
    camera = CameraState()
    camera = with_rotation(camera, dx=0.8, dy=0.0)
    camera = with_rotation(camera, dx=0.0, dy=0.9)
    identity = camera.orientation @ camera.orientation.T
    assert np.allclose(identity, np.eye(3), atol=1e-8)


def test_structure_info_exposes_requested_metadata() -> None:
    scene = make_scene()
    info = structure_info(scene)
    assert info.formula_full == "NaCl"
    assert info.formula_reduced == "NaCl"
    assert len(info.lattice_vectors) == 3
    assert info.lengths == (3.0, 3.0, 3.0)
    assert info.angles == (90.0, 90.0, 90.0)
    assert info.volume == 27.0


def test_orientation_for_view_maps_requested_axis_to_depth() -> None:
    orientation = orientation_for_view("x")
    transformed = np.array([1.0, 0.0, 0.0]) @ orientation.T
    assert np.allclose(transformed, np.array([0.0, 0.0, 1.0]), atol=1e-8)

    orientation = orientation_for_view("y")
    transformed = np.array([0.0, 1.0, 0.0]) @ orientation.T
    assert np.allclose(transformed, np.array([0.0, 0.0, 1.0]), atol=1e-8)

    orientation = orientation_for_view("z")
    transformed = np.array([0.0, 0.0, 1.0]) @ orientation.T
    assert np.allclose(transformed, np.array([0.0, 0.0, 1.0]), atol=1e-8)
    transformed = np.array([1.0, 0.0, 0.0]) @ orientation_for_view("y").T
    assert np.allclose(transformed, np.array([1.0, 0.0, 0.0]), atol=1e-8)


def test_view_along_sets_camera_to_axis_aligned_orientation() -> None:
    scene = make_scene()
    camera = view_along(CameraState(), "z")
    positions = transformed_positions(scene, camera)
    assert np.allclose(camera.orientation, np.eye(3), atol=1e-8)
    assert positions.shape == scene.positions.shape


def test_cube_projection_respects_terminal_character_aspect_ratio() -> None:
    atoms = Atoms(cell=np.diag([4.0, 4.0, 4.0]), pbc=True)
    scene = SceneData(
        atoms=atoms,
        positions=np.empty((0, 3), dtype=float),
        symbols=[],
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="cube",
    )
    camera = view_along(CameraState(show_cell=True, line_mode="unicode"), "z")
    viewport = Viewport(width=80, height=32)
    primitives = build_primitives(scene, camera, viewport, RenderOptions(aspect_ratio=2.0))
    xs = [primitive.x for primitive in primitives]
    ys = [primitive.y for primitive in primitives]
    width_in_cells = max(xs) - min(xs)
    height_in_cells = max(ys) - min(ys)
    assert abs((width_in_cells / 2.0) - height_in_cells) <= 1.0


def test_edge_char_uses_directional_line_glyphs() -> None:
    assert _edge_char((1, 1), (7, 1)) == HORIZONTAL_EDGE
    assert _edge_char((3, 1), (3, 8)) == VERTICAL_EDGE
    assert _edge_char((1, 1), (5, 5)) == DIAGONAL_ASC_EDGE
    assert _edge_char((1, 5), (5, 1)) == DIAGONAL_DESC_EDGE


def test_cell_edges_render_with_line_glyphs_not_only_crosses() -> None:
    scene = make_scene()
    camera = view_along(CameraState(show_cell=True, line_mode="unicode"), "z")
    rows = render_ascii(scene, camera, Viewport(width=60, height=24))
    text = "\n".join(rows)
    assert any(glyph in text for glyph in (HORIZONTAL_EDGE, VERTICAL_EDGE, DIAGONAL_ASC_EDGE, DIAGONAL_DESC_EDGE))


def test_bond_segments_are_detected_for_neighboring_atoms() -> None:
    atoms = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="bond",
    )
    segments = transformed_bond_segments(scene, CameraState(orientation=np.eye(3), show_cell=False))
    assert len(segments) == 1


def test_bond_segments_are_detected_for_periodic_silicon() -> None:
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="silicon",
    )
    segments = transformed_bond_segments(scene, CameraState(orientation=np.eye(3), show_cell=False))
    assert len(segments) == 16


def test_bonds_render_as_sticks_between_atoms() -> None:
    atoms = Atoms(
        symbols=["C", "C"],
        positions=[[0.0, 0.0, 0.0], [1.45, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="bond",
    )
    rows = render_ascii(scene, CameraState(orientation=np.eye(3), line_mode="unicode", show_cell=False), Viewport(width=40, height=15))
    text = "\n".join(rows)
    assert HORIZONTAL_EDGE in text


def test_two_letter_element_labels_render_fully() -> None:
    atoms = Atoms(
        symbols=["Na"],
        positions=[[0.0, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="labels",
    )
    rows = render_ascii(scene, CameraState(orientation=np.eye(3), show_cell=False, show_bonds=False, show_labels=True), Viewport(width=20, height=9))
    assert any("Na" in row for row in rows)


def test_crossing_edges_use_unicode_cross_glyph() -> None:
    atoms = Atoms(cell=np.diag([4.0, 4.0, 4.0]), pbc=True)
    scene = SceneData(
        atoms=atoms,
        positions=np.empty((0, 3), dtype=float),
        symbols=[],
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="cube",
    )
    camera = with_rotation(view_along(CameraState(show_cell=True, line_mode="unicode"), "z"), dx=0.0, dy=0.6)
    text = "\n".join(render_ascii(scene, camera, Viewport(width=80, height=32)))
    assert ORTHOGONAL_CROSS in text or "╳" in text


def test_braille_mode_renders_high_resolution_line_glyphs() -> None:
    scene = make_scene()
    camera = view_along(CameraState(show_cell=True, line_mode="braille"), "z")
    text = "\n".join(render_ascii(scene, camera, Viewport(width=60, height=24)))
    assert any(ord(char) >= BRAILLE_BASE for char in text if char.strip())


def test_braille_mode_keeps_atom_labels_on_top_of_lines() -> None:
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="braille",
    )
    rows = render_ascii(
        scene,
        CameraState(orientation=np.eye(3), line_mode="braille", show_cell=False, show_labels=True),
        Viewport(width=40, height=15),
    )
    assert any("Si" in row for row in rows)


def test_unlabeled_atoms_use_unicode_depth_dots() -> None:
    atoms = Atoms(
        symbols=["C", "O"],
        positions=[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        cell=np.zeros((3, 3)),
        pbc=False,
    )
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="depth-dots",
    )
    rows = render_ascii(
        scene,
        CameraState(orientation=np.eye(3), show_cell=False, show_bonds=False, show_labels=False),
        Viewport(width=20, height=9),
    )
    text = "\n".join(rows)
    assert any(char in DEPTH_GLYPHS for char in text)


def test_cell_display_includes_origin_and_abc_axis_labels() -> None:
    atoms = Atoms(cell=np.diag([4.0, 4.0, 4.0]), pbc=True)
    scene = SceneData(
        atoms=atoms,
        positions=np.empty((0, 3), dtype=float),
        symbols=[],
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="cube",
    )
    rows = render_ascii(scene, view_along(CameraState(show_cell=True), "z"), Viewport(width=60, height=24))
    text = "\n".join(rows)
    assert "o" in text
    assert "a" in text
    assert "b" in text
    assert "c" in text
    assert "b" in text
    assert "c" in text
