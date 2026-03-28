from __future__ import annotations

from xtalui.cli import build_parser


def test_cli_repeat_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--repeat", "2", "1", "3"])
    assert tuple(args.repeat) == (2, 1, 3)


def test_cli_accepts_symprec() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--symprec", "0.001"])
    assert args.symprec == 0.001


def test_cli_accepts_multiple_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(["frame1.xyz", "frame2.xyz"])
    assert [path.name for path in args.paths] == ["frame1.xyz", "frame2.xyz"]


def test_cli_accepts_color_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--color"])
    assert args.color is True
