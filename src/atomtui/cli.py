from __future__ import annotations

import argparse
from pathlib import Path

from atomtui.app import run_viewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomtui", description="Render an atomic structure directly in the terminal.")
    parser.add_argument("path", type=Path, help="Path to a structure file")
    parser.add_argument(
        "--repeat",
        metavar=("NX", "NY", "NZ"),
        type=int,
        nargs=3,
        default=(1, 1, 1),
        help="Repeat the unit cell along each lattice vector",
    )
    parser.add_argument(
        "--hide-cell",
        action="store_true",
        help="Start with the unit cell hidden",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance passed to spglib for space-group detection",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_viewer(args.path, tuple(args.repeat), show_cell=not args.hide_cell, symprec=args.symprec)


if __name__ == "__main__":
    main()
