# atomtui

`atomtui` is a terminal-first crystal structure viewer for atomistic and crystalline structures.

It renders structures directly in the terminal using:

- Unicode and Braille line rendering for cell edges and bonds
- depth-aware atom glyphs or element labels
- interactive camera controls without launching a GUI
- a single-command CLI: `atomtui STRUCTURE_FILE`

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
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv/bin/python -e '.[dev]'
```

You can also run it without activating the environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui --help
UV_CACHE_DIR=/tmp/uv-cache uv run python -m atomtui --help
```

## Usage

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui structure.cif
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui POSCAR --repeat 2 2 1
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui structure.cif --symprec 1e-3
```

The repository also ships with generated sample structures:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui examples/silicon_diamond.cif
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui examples/gaas_zincblende.cif
UV_CACHE_DIR=/tmp/uv-cache uv run atomtui examples/graphite_hexagonal.cif
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
- `1`: toggle the `abc dirs` panel
- `2`: toggle the `xyz dirs` panel
- `Shift+Left` / `Shift+Right`: pan X
- `Shift+Up` / `Shift+Down`: pan Y
- `+` / `-`: zoom in/out
- `m`: toggle line mode between Braille and Unicode wireframe
- `b`: toggle bonds
- `c`: toggle unit cell
- `l`: toggle labels
- `r`: reset view
- `?`: toggle help
- `q`: quit

## Notes

- Space-group detection uses `spglib` through ASE-compatible structure data.
- Bond detection uses ASE natural cutoffs based on covalent radii.
- Braille mode is the default line renderer because it provides smoother terminal line quality.
- Example CIF files in [`examples/`](/home/bonan/appdir/atomtui/examples) are generated with ASE for common crystal prototypes.
