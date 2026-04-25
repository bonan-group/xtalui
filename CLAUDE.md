# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`xtalui` is a terminal-first crystal structure viewer. It renders atomic/crystalline structures in the terminal using ASCII, Unicode, and Braille glyphs with interactive camera controls. The Python package is `xtalui`; the installed CLI commands are `xtal` (crystal viewer) and `tpot` (scatter/line plot viewer).

## Commands

```bash
# Setup
uv venv --python /usr/bin/python3 .venv
uv sync --extra dev

# Run
uv run xtal examples/silicon_diamond.cif
uv run python -m xtalui --help
uv run tpot data.txt            # interactive scatter plot
uv run tpot data.txt --line     # line mode
echo "data" | uv run tpot       # from stdin
uv run tpot data.txt --ascii    # non-interactive text output

# Test
uv run pytest -q

# Lint and format
uv run ruff check .
uv run ruff format --check .
uv run ruff format .   # fix formatting

# Build
uv build

# Run a single test
uv run pytest tests/test_renderer.py::test_name -q
uv run pytest tests/test_plot.py::TestPlotState -q
```

## Architecture

The source lives in `src/xtalui/` with four core modules:

- **`cli.py`** — argparse-based CLI entry point (`xtal` command). Parses file paths, frame slicing (`@SLICE`, `-n`), cell repeat, and display options, then delegates to `app.run_viewer`.
- **`app.py`** — Interactive terminal application built on `prompt_toolkit`. Owns the main event loop, key bindings, and overlay panels. Calls `scene` for data and `renderer` for drawing.
- **`scene.py`** — Data layer with three key types:
  - `SceneData` (immutable structure: positions, cell, bonds)
  - `CameraState` (mutable view: rotation, pan, zoom)
  - `StructureInfo` (metadata: formula, space group, lattice params)
  - Handles structure loading via ASE, ABACUS STRU parsing, bond detection (ASE neighbor list with 1.5x covalent-radius cutoff), and supercell repeats.
- **`renderer.py`** — 3D-to-2D projection and terminal rendering. Supports three line modes (Braille default, Unicode, ASCII), depth-sorted atom drawing, sphere rasterization, and ASE Jmol element colors.
- **`abacus_stru.py`** — Parser for ABACUS STRU files with explicit `LATTICE_VECTORS`. Only this variant is supported (not `LATTICE_PARAMETER(S)` + `latname`).
- **`plot/`** — Sub-package for the `tpot` terminal scatter/line plot viewer:
  - **`parser.py`** — Extracts numeric columns from free-form text. Auto-detects numeric positions (≥60% threshold), names columns from adjacent non-numeric tokens, and detects grouping columns (e.g., "Train"/"Val") to split into multiple series.
  - **`renderer.py`** — 2D plot rendering with Braille high-resolution scatter and line modes. Supports linear/log axis scales, nice-number tick algorithm (1-2-5), grid lines, box-drawing axis frame, multi-panel layouts with shared x-axes, and per-series colors.
  - **`app.py`** — Interactive prompt_toolkit viewer (`PlotState`, `build_plot_application`). Pan, zoom, toggle scatter/line mode, toggle axis scales, grid, column selection, help overlay.
  - **`cli.py`** — argparse CLI for `tpot` command. Supports file/stdin input, column selection, log scale, line mode, and non-interactive ASCII output.

## Key Dependencies

- **ASE** — structure I/O, neighbor lists, element data
- **numpy** — array math for projections and geometry
- **prompt_toolkit** — terminal UI framework for the interactive app
- **spglib** — space group detection (via ASE interface)

## Conventions

- Ruff with 120-char line length, targeting Python 3.10+
- Tests use pytest with `testpaths = ["tests"]` and `pythonpath = ["src"]`
- CI runs on Python 3.10/3.11/3.12 across Ubuntu and macOS
- Releases are tag-driven: `git tag v0.x.y && git push origin v0.x.y`
- Frame slicing uses Python slice syntax on filenames (`file.xyz@::10`) and the `-n` flag
