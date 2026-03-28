# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

## [0.1.1] - 2026-03-28

### Added

- ASE-style frame selection with `-n/--image-number` and per-file `@SLICE` syntax
- Short CLI flags `-r` for repeat, `-s` for symmetry tolerance, and `-c` for color

### Changed

- README installation and usage documentation expanded for overlays, sphere mode, calibration, and frame selection

## [0.1.0] - 2026-03-27

### Added

- Initial terminal crystal viewer release with interactive rotation, pan, zoom, and supercell repetition
- Unicode and Braille rendering modes for atoms, bonds, and cell edges
- Metadata display for formula, lattice parameters, volume, and space group
- Direction widgets for lattice and Cartesian axes
- Example crystal CIF files generated with ASE
- ABACUS `STRU` support for files with explicit `LATTICE_VECTORS`
