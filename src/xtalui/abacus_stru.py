from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Bohr


STRU_BLOCK_TITLES = {
    "ATOMIC_SPECIES",
    "NUMERICAL_ORBITAL",
    "LATTICE_CONSTANT",
    "LATTICE_PARAMETER",
    "LATTICE_PARAMETERS",
    "LATTICE_VECTORS",
    "ATOMIC_POSITIONS",
}


def looks_like_abacus_stru(path: Path) -> bool:
    if path.suffix.lower() == ".stru" or path.name.upper() == "STRU":
        return True
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = _trim_line(raw_line)
                if line:
                    return line == "ATOMIC_SPECIES"
    except OSError:
        return False
    return False


def read_abacus_stru(path: Path) -> Atoms:
    blocks = _read_blocks(path)
    if "LATTICE_CONSTANT" not in blocks:
        raise ValueError(f"{path} is missing the LATTICE_CONSTANT block")
    if "ATOMIC_SPECIES" not in blocks:
        raise ValueError(f"{path} is missing the ATOMIC_SPECIES block")
    if "ATOMIC_POSITIONS" not in blocks:
        raise ValueError(f"{path} is missing the ATOMIC_POSITIONS block")

    lattice_constant = float(blocks["LATTICE_CONSTANT"][0])
    cell = _cell_from_blocks(blocks, lattice_constant)
    species_order = _parse_species_order(blocks["ATOMIC_SPECIES"])
    symbols, positions = _parse_positions(blocks["ATOMIC_POSITIONS"], species_order, lattice_constant, cell)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=bool(np.any(cell)))
    return atoms


def _trim_line(line: str) -> str:
    return line.split("#", 1)[0].split("//", 1)[0].strip(" \t\r\n")


def _read_blocks(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [_trim_line(line).replace("\t", " ") for line in handle]
    lines = [line for line in lines if line]

    delimiters = [index for index, line in enumerate(lines) if line in STRU_BLOCK_TITLES]
    delimiters.append(len(lines))
    blocks: dict[str, list[str]] = {}
    for start, end in zip(delimiters, delimiters[1:]):
        blocks[lines[start]] = lines[start + 1 : end]
    return blocks


def _cell_from_blocks(blocks: dict[str, list[str]], lattice_constant: float) -> np.ndarray:
    if "LATTICE_VECTORS" in blocks:
        vectors = np.array([[float(value) for value in line.split()] for line in blocks["LATTICE_VECTORS"]], dtype=float)
        return vectors * lattice_constant * Bohr
    if "LATTICE_PARAMETERS" in blocks or "LATTICE_PARAMETER" in blocks:
        raise ValueError(
            "STRU files without explicit LATTICE_VECTORS are not supported. "
            "ABACUS requires INPUT latname information to reconstruct the cell."
        )
    raise ValueError("STRU file is missing the LATTICE_VECTORS block")


def _parse_species_order(lines: list[str]) -> list[str]:
    order: list[str] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"invalid ATOMIC_SPECIES entry: {line!r}")
        order.append(fields[0])
    return order


def _parse_positions(
    lines: list[str], species_order: list[str], lattice_constant: float, cell: np.ndarray
) -> tuple[list[str], np.ndarray]:
    coord_type = lines[0].strip()
    symbols: list[str] = []
    coords: list[np.ndarray] = []

    index = 1
    while index < len(lines):
        symbol = lines[index]
        _mag_each = float(lines[index + 1])
        natom = int(lines[index + 2])
        for offset in range(natom):
            parsed = _parse_atom_line(lines[index + 3 + offset])
            symbols.append(symbol)
            coords.append(_convert_coord(parsed, coord_type, lattice_constant, cell))
        index += 3 + natom

    # Keep species order stable if the file grouped symbols unexpectedly.
    species_rank = {symbol: rank for rank, symbol in enumerate(species_order)}
    permutation = sorted(range(len(symbols)), key=lambda idx: species_rank.get(symbols[idx], len(species_rank)))
    ordered_symbols = [symbols[idx] for idx in permutation]
    ordered_coords = np.array([coords[idx] for idx in permutation], dtype=float)
    return ordered_symbols, ordered_coords


def _parse_atom_line(line: str) -> dict[str, list[float] | float | tuple[str, list[float]]]:
    fields = line.split()
    result: dict[str, list[float] | float | tuple[str, list[float]]] = {
        "coord": [float(value) for value in fields[:3]]
    }

    idx = 3
    while idx < len(fields):
        token = fields[idx]
        lowered = token.lower()
        if lowered.isdigit():
            idx += 3
        elif lowered == "m":
            idx += 4
        elif lowered in {"v", "vel", "velocity"}:
            idx += 4
        elif lowered in {"mag", "magmom"}:
            if idx + 5 < len(fields) and fields[idx + 2].lower() == "angle1":
                idx += 6
            elif idx + 3 < len(fields) and _is_number(fields[idx + 1]) and _is_number(fields[idx + 2]) and _is_number(fields[idx + 3]):
                idx += 4
            else:
                idx += 2
        elif lowered in {"angle1", "angle2"}:
            idx += 2
        elif lowered in {"lambda", "sc"}:
            idx += 1 + _count_numeric_tail(fields, idx + 1, max_count=3)
        else:
            raise ValueError(f"unknown ATOMIC_POSITIONS keyword {token!r} in line {line!r}")
    return result


def _convert_coord(
    parsed: dict[str, list[float] | float | tuple[str, list[float]]],
    coord_type: str,
    lattice_constant: float,
    cell: np.ndarray,
) -> np.ndarray:
    coord = np.array(parsed["coord"], dtype=float)
    lowered = coord_type.lower()

    if lowered == "direct":
        return coord @ cell
    if lowered == "cartesian":
        return coord * lattice_constant * Bohr
    if lowered == "cartesian_au":
        return coord * Bohr
    if lowered == "cartesian_angstrom":
        return coord
    if lowered.startswith("cartesian_angstrom_center_"):
        offsets = {
            "cartesian_angstrom_center_xy": np.array([0.5, 0.5, 0.0]),
            "cartesian_angstrom_center_xz": np.array([0.5, 0.0, 0.5]),
            "cartesian_angstrom_center_yz": np.array([0.0, 0.5, 0.5]),
            "cartesian_angstrom_center_xyz": np.array([0.5, 0.5, 0.5]),
        }
        if lowered not in offsets:
            raise ValueError(f"unsupported STRU coordinate type {coord_type!r}")
        return coord + offsets[lowered] @ cell
    raise ValueError(f"unsupported STRU coordinate type {coord_type!r}")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _count_numeric_tail(fields: list[str], start: int, max_count: int) -> int:
    count = 0
    while start + count < len(fields) and count < max_count and _is_number(fields[start + count]):
        count += 1
    return count
