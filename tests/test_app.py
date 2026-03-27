from __future__ import annotations

import numpy as np

from ase import Atoms

from atomtui.app import (
    AXIS_WIDGET_HEIGHT,
    AXIS_WIDGET_WIDTH,
    cartesian_direction_endpoints,
    element_legend,
    lattice_direction_endpoints,
    render_cartesian_direction_widget,
    render_lattice_direction_widget,
)
from atomtui.renderer import BRAILLE_BASE
from atomtui.scene import CameraState, SceneData, cycle_line_mode


def test_lattice_direction_widget_contains_axis_labels() -> None:
    text = render_lattice_direction_widget(np.eye(3), CameraState(orientation=np.eye(3)))
    assert "a" in text
    assert "b" in text
    assert "c" in text
    assert "o" in text


def test_lattice_direction_widget_changes_with_rotation() -> None:
    identity = render_lattice_direction_widget(np.eye(3), CameraState(orientation=np.eye(3)))
    rotated = render_lattice_direction_widget(
        np.eye(3),
        CameraState(
            orientation=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        ),
    )
    assert identity != rotated


def test_lattice_direction_widget_normalizes_vector_lengths() -> None:
    identity = render_lattice_direction_widget(np.eye(3), CameraState(orientation=np.eye(3)))
    stretched = render_lattice_direction_widget(np.diag([5.0, 2.0, 9.0]), CameraState(orientation=np.eye(3)))
    assert identity == stretched


def test_lattice_direction_widget_uses_braille_for_vector_shafts() -> None:
    text = render_lattice_direction_widget(np.eye(3), CameraState(orientation=np.eye(3)))
    assert any(ord(char) >= BRAILLE_BASE for char in text if char.strip())


def test_lattice_direction_widget_preserves_foreshortening() -> None:
    identity = dict((label, (x, y)) for label, x, y, _ in lattice_direction_endpoints(np.eye(3), CameraState(orientation=np.eye(3))))
    rotated = dict(
        (label, (x, y))
        for label, x, y, _ in lattice_direction_endpoints(
            np.eye(3),
            CameraState(
                orientation=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
                        [0.0, np.sqrt(0.5), np.sqrt(0.5)],
                    ]
                )
            ),
        )
    )
    center = (AXIS_WIDGET_WIDTH // 2, AXIS_WIDGET_HEIGHT // 2)
    identity_b = np.hypot(identity["b"][0] - center[0], identity["b"][1] - center[1])
    rotated_b = np.hypot(rotated["b"][0] - center[0], rotated["b"][1] - center[1])
    identity_c = np.hypot(identity["c"][0] - center[0], identity["c"][1] - center[1])
    rotated_c = np.hypot(rotated["c"][0] - center[0], rotated["c"][1] - center[1])
    assert rotated_b < identity_b
    assert rotated_c > identity_c


def test_cartesian_direction_widget_contains_xyz_labels() -> None:
    text = render_cartesian_direction_widget(CameraState(orientation=np.eye(3)))
    assert "x" in text
    assert "y" in text
    assert "z" in text
    assert "o" in text


def test_cartesian_direction_widget_changes_with_rotation() -> None:
    identity = render_cartesian_direction_widget(CameraState(orientation=np.eye(3)))
    rotated = render_cartesian_direction_widget(
        CameraState(
            orientation=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
        )
    )
    assert identity != rotated


def test_cartesian_direction_widget_preserves_foreshortening() -> None:
    identity = dict((label, (x, y)) for label, x, y, _ in cartesian_direction_endpoints(CameraState(orientation=np.eye(3))))
    rotated = dict(
        (label, (x, y))
        for label, x, y, _ in cartesian_direction_endpoints(
            CameraState(
                orientation=np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
                        [0.0, np.sqrt(0.5), np.sqrt(0.5)],
                    ]
                )
            )
        )
    )
    center = (AXIS_WIDGET_WIDTH // 2, AXIS_WIDGET_HEIGHT // 2)
    identity_y = np.hypot(identity["y"][0] - center[0], identity["y"][1] - center[1])
    rotated_y = np.hypot(rotated["y"][0] - center[0], rotated["y"][1] - center[1])
    identity_z = np.hypot(identity["z"][0] - center[0], identity["z"][1] - center[1])
    rotated_z = np.hypot(rotated["z"][0] - center[0], rotated["z"][1] - center[1])
    assert rotated_y < identity_y
    assert rotated_z > identity_z


def test_element_legend_lists_full_element_names() -> None:
    atoms = Atoms("NaCl", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scene = SceneData(
        atoms=atoms,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        symbols=list(atoms.get_chemical_symbols()),
        cell=np.asarray(atoms.cell.array, dtype=float),
        title="legend",
    )
    legend = element_legend(scene)
    assert "Na=Sodium" in legend
    assert "Cl=Chlorine" in legend


def test_cycle_line_mode_toggles_between_braille_and_unicode() -> None:
    camera = CameraState()
    assert camera.line_mode == "braille"
    camera = cycle_line_mode(camera)
    assert camera.line_mode == "unicode"
    camera = cycle_line_mode(camera)
    assert camera.line_mode == "braille"


def test_camera_defaults_show_both_direction_panels() -> None:
    camera = CameraState()
    assert camera.show_abc_panel is True
    assert camera.show_xyz_panel is True
