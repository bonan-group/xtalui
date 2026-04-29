from __future__ import annotations

from xtalui.cli import build_parser


def test_cli_repeat_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--repeat", "2", "1", "3"])
    assert tuple(args.repeat) == (2, 1, 3)


def test_cli_repeat_short_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-r", "2", "1", "3"])
    assert tuple(args.repeat) == (2, 1, 3)


def test_cli_accepts_symprec() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--symprec", "0.001"])
    assert args.symprec == 0.001


def test_cli_accepts_symprec_short_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-s", "0.001"])
    assert args.symprec == 0.001


def test_cli_accepts_multiple_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(["frame1.xyz", "frame2.xyz"])
    assert args.paths == ["frame1.xyz", "frame2.xyz"]


def test_cli_accepts_color_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--color"])
    assert args.color is True


def test_cli_accepts_color_short_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-c"])
    assert args.color is True


def test_cli_enables_color_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif"])
    assert args.color is True


def test_cli_accepts_no_color_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--no-color"])
    assert args.color is False


def test_cli_accepts_global_image_number() -> None:
    parser = build_parser()
    args = parser.parse_args(["-n", "1::2", "series.xyz"])
    assert args.image_number == "1::2"


def test_cli_accepts_filename_slice_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["frames.xyz@-2:", "other.xyz"])
    assert args.paths == ["frames.xyz@-2:", "other.xyz"]


def test_cli_accepts_wrap_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--wrap"])
    assert args.wrap is True


def test_cli_accepts_wrap_short_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-w"])
    assert args.wrap is True


def test_cli_wrap_defaults_to_false() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif"])
    assert args.wrap is False


def test_cli_accepts_refine_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--refine"])
    assert args.refine is True


def test_cli_refine_defaults_to_false() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif"])
    assert args.refine is False


def test_cli_accepts_filter_label() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--filter-label", "alpha"])
    assert args.filter_label == ["alpha"]


def test_cli_accepts_filter_label_short() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-f", "alpha"])
    assert args.filter_label == ["alpha"]


def test_cli_accepts_multiple_filter_labels() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "-f", "alpha", "-f", "beta"])
    assert args.filter_label == ["alpha", "beta"]


def test_cli_filter_label_defaults_to_none() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif"])
    assert args.filter_label is None
