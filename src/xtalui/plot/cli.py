"""Command-line entry point for the terminal plot viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtalui.plot.app import PlotState, build_plot_application
from xtalui.plot.parser import detect_numeric_columns, parse_file, parse_stdin
from xtalui.plot.renderer import PlotBounds

EPILOG = """\
examples:
  tpot data.txt                        Interactive plot, auto-detect columns
  tpot data.txt -x 0 -y 3             Plot column 0 vs column 3
  tpot data.txt -Y 1 2 3              Multi-column: plot cols 1,2,3 vs col 0
  tpot data.txt --line                 Start in line mode
  tpot data.txt --log-x --log-y        Start with log axes
  tpot data.txt --ascii                Plain text output to stdout
  tpot data.txt --ascii --xmin 0 --xmax 100 --ymin 0 --ymax 1
  tpot data.txt --ascii --color        Enable colored output
  cat log.txt | tpot                   Read from stdin (non-interactive)

interactive controls:
  Arrow keys    Pan the view
  +/-           Zoom in/out
  m             Toggle scatter/line mode
  g             Toggle grid lines
  M             Toggle multi-column mode
  L             Toggle legend
  C             Toggle colors
  X / Y         Toggle x/y axis between linear and log scale
  c             Open column selector
  Ctrl-R        Reset view (zoom, pan)
  ?             Toggle help overlay
  q             Quit
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpot",
        description="Interactive terminal scatter and line plot viewer. "
        "Auto-detects numeric columns from free-form text files or stdin "
        "and renders high-resolution Braille plots directly in the terminal.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to a data file. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "-x",
        "--x-column",
        type=int,
        default=None,
        metavar="IDX",
        help="0-based index of the numeric column for the x-axis (default: 0).",
    )
    parser.add_argument(
        "-y",
        "--y-column",
        type=int,
        default=None,
        metavar="IDX",
        help="0-based index of the numeric column for the y-axis (default: 1).",
    )
    parser.add_argument(
        "-Y",
        "--y-columns",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="0-based indices of multiple y-axis columns for multi-column mode.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        metavar="VAL",
        help="Lower bound for the x-axis.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        metavar="VAL",
        help="Upper bound for the x-axis.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        metavar="VAL",
        help="Lower bound for the y-axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        metavar="VAL",
        help="Upper bound for the y-axis.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Start with grid lines hidden.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Output plain text to stdout and exit (no interaction).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        default=False,
        help="Enable colored output (for terminals that support ANSI colors).",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic x-axis scale.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis scale.",
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="Use line plot mode (default: scatter).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Read data.
    if args.file is not None:
        data = parse_file(Path(args.file))
    elif not sys.stdin.isatty():
        data = parse_stdin()
    else:
        parser.error("no input file specified and stdin is a terminal")

    # Detect numeric columns.
    columns = detect_numeric_columns(data)
    if len(columns) < 2:
        print(f"Error: need at least 2 numeric columns, found {len(columns)}", file=sys.stderr)
        sys.exit(1)

    x_col = args.x_column if args.x_column is not None else 0
    y_col = args.y_column if args.y_column is not None else 1

    multi_column = args.y_columns is not None
    y_col_indices = args.y_columns if args.y_columns is not None else [y_col]

    # Validate indices.
    all_indices = [x_col] + y_col_indices
    if any(idx < 0 or idx >= len(columns) for idx in all_indices):
        print(
            f"Error: column indices must be in 0..{len(columns) - 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    x_scale = "log" if args.log_x else "linear"
    y_scale = "log" if args.log_y else "linear"

    state = PlotState(
        data,
        x_col_index=x_col,
        y_col_index=y_col_indices[0],
        y_col_indices=y_col_indices,
        multi_column=multi_column,
        show_grid=not args.no_grid,
        plot_mode="line" if args.line else "scatter",
        x_scale=x_scale,
        y_scale=y_scale,
        show_color=args.color or multi_column,
        show_legend=multi_column,
    )

    # Build explicit bounds from CLI limits, if any were provided.
    bounds = None
    if any(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax)):
        import numpy as np

        all_x = np.concatenate([s.x for s in state.series_list])
        all_y = np.concatenate([s.y for s in state.series_list])
        b_xmin = args.xmin if args.xmin is not None else float(np.min(all_x))
        b_xmax = args.xmax if args.xmax is not None else float(np.max(all_x))
        b_ymin = args.ymin if args.ymin is not None else float(np.min(all_y))
        b_ymax = args.ymax if args.ymax is not None else float(np.max(all_y))
        bounds = PlotBounds(x_min=b_xmin, x_max=b_xmax, y_min=b_ymin, y_max=b_ymax)

    # Non-interactive mode: piped stdin or piped stdout or explicit --ascii.
    if not sys.stdin.isatty() or not sys.stdout.isatty() or args.ascii:
        rows = state.render_ascii(80, 24, bounds=bounds)
        print("\n".join(rows))
        return

    # Interactive mode.
    if bounds is not None:
        state.set_bounds(bounds)
    app = build_plot_application(state)
    app.run()


if __name__ == "__main__":
    main()
