# xtalui

`xtalui` is a terminal-first crystal structure viewer for atomistic and crystalline structures.

It renders structures directly in the terminal using:

- Unicode and Braille line rendering for cell edges and bonds
- depth-aware atom glyphs or element labels
- interactive camera controls without launching a GUI
- a single-command CLI: `xtal STRUCTURE_FILE`

## Features

- Load structures through ASE
- Show atoms, bonds, and the unit cell in the terminal
- Display chemical formula, lattice vectors, cell lengths and angles, volume, and space group
- Show both lattice-frame `a/b/c` and Cartesian `x/y/z` direction widgets
- Toggle labels, bonds, cell frame, and direction panels at runtime
- Align the view along `x`, `y`, or `z` with one keypress

## Installation

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3 .venv
UV_CACHE_DIR=/tmp/uv-cache uv sync --dev
```

You can also run it without activating the environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run xtal --help
UV_CACHE_DIR=/tmp/uv-cache uv run python -m xtalui --help
```

## Usage

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run xtal structure.cif
UV_CACHE_DIR=/tmp/uv-cache uv run xtal POSCAR --repeat 2 2 1
UV_CACHE_DIR=/tmp/uv-cache uv run xtal structure.cif --symprec 1e-3
```

The repository also ships with generated sample structures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run xtal examples/silicon_diamond.cif
UV_CACHE_DIR=/tmp/uv-cache uv run xtal examples/gaas_zincblende.cif
UV_CACHE_DIR=/tmp/uv-cache uv run xtal examples/graphite_hexagonal.cif
```

## CLI Options

- `PATH`: structure file to open with ASE
- `--repeat NX NY NZ`: repeat the structure along each lattice direction
- `--hide-cell`: start with the unit cell hidden
- `--symprec FLOAT`: set the symmetry tolerance used for space-group detection

## Controls

- `Left` / `Right`: rotate around Y
- `Up` / `Down`: rotate around X
- `x` / `y` / `z`: align the view along the Cartesian axes
- `rXYZ`: rebuild the displayed structure as an in-app `XxYxZ` supercell, for example `r222`
- `1`: toggle the `abc dirs` panel
- `2`: toggle the `xyz dirs` panel
- `Shift+Left` / `Shift+Right`: pan X
- `Shift+Up` / `Shift+Down`: pan Y
- `+` / `-`: zoom in/out
- `m`: toggle line mode between Braille and Unicode wireframe
- `b`: toggle bonds
- `c`: toggle unit cell
- `Ctrl-R`: reset the view camera and restore the launch repeat
- `l`: toggle labels
- `Esc`: cancel an in-progress repeat command
- `?`: toggle help
- `q`: quit

## Notes

- The Python package name is `xtalui`, while the installed CLI command is `xtal`.
- Space-group detection uses `spglib` through ASE-compatible structure data.
- Bond detection follows the ASE GUI heuristic: a periodic neighbor list with a `1.5x` covalent-radius cutoff.
- Braille mode is the default line renderer because it provides smoother terminal line quality.
- Example CIF files in [`examples/`](/home/bonan/appdir/atomtui/examples) are generated with ASE for common crystal prototypes.
