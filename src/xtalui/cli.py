from __future__ import annotations

import argparse

from xtalui.app import run_viewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xtal", description="Render an atomic structure directly in the terminal.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to structure files. Append @SLICE to override frame selection for one file.",
    )
    parser.add_argument(
        "-n",
        "--image-number",
        default=":",
        metavar="SLICE",
        help="Pick individual image(s) from each file using Python slice syntax like 0, 1:, or ::2.",
    )
    parser.add_argument(
        "-r",
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
        "-s",
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance passed to spglib for space-group detection",
    )
    parser.add_argument(
        "-c",
        "--color",
        action="store_true",
        help="Start with element colors enabled",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_viewer(
        args.paths,
        tuple(args.repeat),
        image_number=args.image_number,
        show_cell=not args.hide_cell,
        symprec=args.symprec,
        show_color=args.color,
    )


if __name__ == "__main__":
    main()
