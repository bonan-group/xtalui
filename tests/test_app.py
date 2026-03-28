from __future__ import annotations

import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.io import write

from xtalui.app import (
    AXIS_ASPECT_RATIO,
    AXIS_WIDGET_HEIGHT,
    AXIS_WIDGET_WIDTH,
    ViewerState,
    build_application,
    cartesian_direction_endpoints,
    element_legend,
    lattice_direction_endpoints,
    render_calibration_lines,
    render_cartesian_direction_widget,
    render_lattice_direction_widget,
    wrapped_line_count,
)
from xtalui.renderer import BRAILLE_BASE
from xtalui.scene import CameraState, SceneData, cycle_line_mode, default_orientation, view_along


def _visual_length(point: tuple[float, float], center: tuple[int, int]) -> float:
    return float(np.hypot((point[0] - center[0]) / AXIS_ASPECT_RATIO, point[1] - center[1]))


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
    identity = dict(
        (label, (x, y)) for label, x, y, _ in lattice_direction_endpoints(np.eye(3), CameraState(orientation=np.eye(3)))
    )
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
    identity = dict(
        (label, (x, y)) for label, x, y, _ in cartesian_direction_endpoints(CameraState(orientation=np.eye(3)))
    )
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


def test_default_orientation_projects_cartesian_axes_to_equal_lengths() -> None:
    endpoints = dict(
        (label, (x, y))
        for label, x, y, _ in cartesian_direction_endpoints(CameraState(orientation=default_orientation()))
    )
    center = (AXIS_WIDGET_WIDTH // 2, AXIS_WIDGET_HEIGHT // 2)
    lengths = {label: _visual_length(point, center) for label, point in endpoints.items()}

    assert abs(lengths["x"] - lengths["y"]) <= 1.0
    assert abs(lengths["y"] - lengths["z"]) <= 1.0


def test_x_view_balances_horizontal_and_vertical_axis_lengths_visually() -> None:
    endpoints = dict(
        (label, (x, y)) for label, x, y, _ in cartesian_direction_endpoints(view_along(CameraState(), "x"))
    )
    center = (AXIS_WIDGET_WIDTH // 2, AXIS_WIDGET_HEIGHT // 2)

    assert abs(_visual_length(endpoints["y"], center) - _visual_length(endpoints["z"], center)) <= 1.0


def test_default_orientation_axes_make_better_use_of_widget_space() -> None:
    endpoints = dict(
        (label, (x, y))
        for label, x, y, _ in cartesian_direction_endpoints(CameraState(orientation=default_orientation()))
    )
    center = (AXIS_WIDGET_WIDTH // 2, AXIS_WIDGET_HEIGHT // 2)

    assert _visual_length(endpoints["x"], center) >= 2.0
    assert _visual_length(endpoints["y"], center) >= 2.0
    assert _visual_length(endpoints["z"], center) >= 2.0


def test_wrapped_line_count_grows_for_narrow_widths() -> None:
    assert wrapped_line_count("abcdef", 10) == 1
    assert wrapped_line_count("abcdef", 3) == 2
    assert wrapped_line_count("abc\ndefgh", 3) == 3


def test_calibration_renderer_draws_circle_glyphs() -> None:
    lines = render_calibration_lines(40, 16, 2.0)

    assert any(any(ord(char) >= BRAILLE_BASE for char in line if char.strip()) for line in lines)
    assert any("+" in line for line in lines)


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
    assert camera.show_color is False
    assert camera.show_spheres is False


def test_repeat_command_updates_scene_and_status(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    initial_atoms = len(state.scene.atoms)
    state.begin_repeat_command()
    state.append_repeat_digit("2")
    state.append_repeat_digit("2")
    state.append_repeat_digit("2")

    assert state.repeat == (2, 2, 2)
    assert state.pending_repeat_command is None
    assert len(state.scene.atoms) == initial_atoms * 8
    assert "repeat=2x2x2" in state.status()
    assert "repeat set to 2x2x2" in state.status()


def test_repeat_command_rejects_zero_digit(tmp_path) -> None:
    path = tmp_path / "al.cif"
    write(path, bulk("Al", "fcc", a=4.05, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    state.begin_repeat_command()
    state.append_repeat_digit("0")

    assert state.repeat == (1, 1, 1)
    assert state.pending_repeat_command == ""
    assert "repeat digits must be in 1..9" in state.status()


def test_reset_viewer_restores_initial_repeat(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(2, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    state.begin_repeat_command()
    state.append_repeat_digit("3")
    state.append_repeat_digit("1")
    state.append_repeat_digit("1")
    assert state.repeat == (3, 1, 1)

    state.reset_viewer()

    assert state.repeat == (2, 1, 1)
    assert state.pending_repeat_command is None
    assert "repeat=2x1x1" in state.status()
    assert "view reset" in state.status()


def test_autorotate_tick_updates_orientation(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)
    before = state.camera.orientation.copy()

    state.toggle_autorotate()
    rotated = state.tick_autorotate()

    assert rotated is True
    assert not np.allclose(state.camera.orientation, before)
    assert f"spin={'on' if state.autorotate else 'off'}" in state.status()


def test_series_frame_navigation_wraps(tmp_path) -> None:
    atoms_a = bulk("Si", "diamond", a=5.431, cubic=True)
    atoms_b = bulk("Si", "diamond", a=5.531, cubic=True)
    path = tmp_path / "series.xyz"
    write(path, [atoms_a, atoms_b], format="extxyz")
    state = ViewerState(paths=[path], repeat=(2, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    assert state.frame_count == 2
    assert len(state.scene.atoms) == len(atoms_a) * 2

    state.step_frame(1)
    assert state.frame_index == 1
    assert len(state.scene.atoms) == len(atoms_b) * 2

    state.step_frame(1)
    assert state.frame_index == 0


def test_repeat_reload_preserves_image_number_selection(tmp_path) -> None:
    frames = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
        Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
        Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True),
    ]
    path = tmp_path / "series.xyz"
    write(path, frames, format="extxyz")
    state = ViewerState(
        paths=[path],
        repeat=(1, 1, 1),
        show_cell=True,
        symprec=1e-5,
        show_color=False,
        image_number="1:",
    )

    assert state.frame_count == 2
    assert state.scene.symbols == ["He"]

    state.begin_repeat_command()
    state.append_repeat_digit("2")
    state.append_repeat_digit("1")
    state.append_repeat_digit("1")

    assert state.frame_count == 2
    assert state.scene.symbols == ["He", "He"]
    assert state.scenes[1].symbols == ["Li", "Li"]


def test_frame_autoplay_advances_series(tmp_path) -> None:
    atoms_a = bulk("Si", "diamond", a=5.431, cubic=True)
    atoms_b = bulk("Si", "diamond", a=5.531, cubic=True)
    path = tmp_path / "series.xyz"
    write(path, [atoms_a, atoms_b], format="extxyz")
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    state.toggle_frame_autoplay()
    assert state.frame_autoplay is True
    advanced = state.tick_frame_autoplay(state._next_frame_time)

    assert advanced is True
    assert state.frame_index == 1
    assert "play=on" in state.status()


def test_frame_autoplay_rejects_single_frame(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    state.toggle_frame_autoplay()

    assert state.frame_autoplay is False
    assert "single-frame structure" in state.status()


def test_viewer_state_accepts_multiple_paths_as_frame_series(tmp_path) -> None:
    path_a = tmp_path / "a.cif"
    path_b = tmp_path / "b.cif"
    write(path_a, bulk("Al", "fcc", a=4.05, cubic=True))
    write(path_b, bulk("Cu", "fcc", a=3.615, cubic=True))

    state = ViewerState(paths=[path_a, path_b], repeat=(2, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    assert state.frame_count == 2
    assert state.scene.title == "a.cif"
    state.step_frame(1)
    assert state.scene.title == "b.cif"
    assert "frame=2/2" in state.status()


def test_footer_and_help_windows_are_configured_to_wrap(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)
    app = build_application(state)
    root = app.layout.container

    footer = root.children[2]
    help_container = root.children[3]

    assert footer.wrap_lines() is True
    assert help_container.content.wrap_lines() is True


def test_positions_panel_does_not_change_render_body_height(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    without_panel = state.body_height(80, 32)
    state.toggle_positions()
    state.status_message = ""
    with_panel = state.body_height(80, 32)

    assert with_panel == without_panel


def test_positions_panel_is_hidden_by_default_and_can_be_toggled(tmp_path) -> None:
    path = tmp_path / "nacl.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    assert state.show_positions is False

    state.toggle_positions()

    assert state.show_positions is True
    assert "positions shown" in state.status()
    assert "idx" in state.positions_text()
    assert "fx" in state.positions_text()


def test_bond_lengths_panel_can_be_toggled(tmp_path) -> None:
    path = tmp_path / "nacl.cif"
    write(path, bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)

    state.toggle_bond_lengths()

    assert state.show_bond_lengths is True
    assert state.show_positions is False
    assert "bond lengths shown" in state.status()
    assert "length" in state.bond_text()
    assert "[" in state.bond_text()


def test_positions_panel_scrolls_when_atom_list_is_long(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(2, 2, 2), show_cell=True, symprec=1e-5, show_color=False)
    state.toggle_positions()

    initial_text = state.positions_text()
    initial_scroll = state.position_scroll

    state.scroll_positions(1)

    assert state.position_scroll == initial_scroll + 1
    assert state.positions_text() != initial_text


def test_bond_lengths_panel_scrolls_when_list_is_long(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(2, 2, 2), show_cell=True, symprec=1e-5, show_color=False)
    state.toggle_bond_lengths()

    initial_text = state.bond_text()
    initial_scroll = state.bond_scroll

    state.scroll_overlay(1)

    assert state.bond_scroll == initial_scroll + 1
    assert state.bond_text() != initial_text


def test_positions_panel_clamps_scroll_across_frame_changes(tmp_path) -> None:
    atoms_a = bulk("Si", "diamond", a=5.431, cubic=True).repeat((2, 2, 2))
    atoms_b = bulk("Si", "diamond", a=5.431, cubic=True)
    path = tmp_path / "series.xyz"
    write(path, [atoms_a, atoms_b], format="extxyz")
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)
    state.toggle_positions()
    state.scroll_positions(100)

    assert state.position_scroll > 0

    state.step_frame(1)

    assert state.position_scroll == 0


def test_viewer_state_can_start_with_color_enabled(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=True)

    assert state.camera.show_color is True
    assert "color=on" in state.status()


def test_sphere_mode_state_appears_in_status() -> None:
    camera = CameraState(show_spheres=True)
    assert camera.show_spheres is True


def test_calibration_mode_toggles_and_adjusts_aspect_ratio(tmp_path) -> None:
    path = tmp_path / "si.cif"
    write(path, bulk("Si", "diamond", a=5.431, cubic=True))
    state = ViewerState(paths=[path], repeat=(1, 1, 1), show_cell=True, symprec=1e-5, show_color=False)
    initial_aspect = state.aspect_ratio

    state.toggle_calibration()
    state.adjust_aspect_ratio(0.2)

    assert state.calibration_mode is True
    assert state.aspect_ratio == initial_aspect + 0.2
    assert "cal=on" in state.status()
