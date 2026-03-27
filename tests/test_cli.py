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
