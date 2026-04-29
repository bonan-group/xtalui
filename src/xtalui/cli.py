"""Command-line entry points for the terminal structure viewer."""

from __future__ import annotations

import argparse

from xtalui.app import run_viewer


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for `xtal`."""

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
        "--color",
        "-c",
        dest="color",
        action="store_true",
        default=True,
        help="Start with element colors enabled (default)",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Start with element colors disabled",
    )
    parser.add_argument(
        "-w",
        "--wrap",
        action="store_true",
        help="Wrap atoms back into the unit cell on load",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine the cell using spglib symmetry operations",
    )
    parser.add_argument(
        "-f",
        "--filter-label",
        action="append",
        default=None,
        metavar="LABEL",
        help="Only show structures whose atoms.info label/dft_label matches LABEL (repeatable)",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and launch the interactive viewer."""

    parser = build_parser()
    args = parser.parse_args()
    run_viewer(
        args.paths,
        tuple(args.repeat),
        image_number=args.image_number,
        show_cell=not args.hide_cell,
        symprec=args.symprec,
        show_color=args.color,
        wrap=args.wrap,
        refine=args.refine,
        filter_labels=tuple(args.filter_label) if args.filter_label else None,
    )


if __name__ == "__main__":
    main()
