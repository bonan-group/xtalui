"""Extract numeric columns from free-form text for plotting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class Column:
    """A detected numeric column extracted from input text."""

    name: str
    values: list[float]


@dataclass(frozen=True)
class Series:
    """A named series of (x, y) data points."""

    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ParsedData:
    """Complete parsed result from a text input."""

    columns: tuple[Column, ...]
    raw_lines: tuple[str, ...]
    source_name: str


def _is_number_token(token: str) -> bool:
    return bool(_NUMBER_RE.match(token))


def _tokenize_lines(lines: list[str]) -> list[list[str]]:
    """Split non-comment, non-empty lines into token lists."""
    result: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append(stripped.split())
    return result


def _detect_numeric_positions(token_rows: list[list[str]]) -> list[bool]:
    """Return a list indicating which column positions are predominantly numeric."""
    if not token_rows:
        return []
    max_cols = max(len(row) for row in token_rows)
    if max_cols == 0:
        return []
    numeric_count = [0] * max_cols
    total_count = [0] * max_cols
    for row in token_rows:
        for i, token in enumerate(row):
            total_count[i] += 1
            if _is_number_token(token):
                numeric_count[i] += 1
    threshold = 0.6
    return [total_count[i] > 0 and numeric_count[i] / total_count[i] >= threshold for i in range(max_cols)]


def _name_column(token_rows: list[list[str]], col_index: int, is_numeric: bool) -> str:
    """Derive a column name from adjacent non-numeric tokens or generate a default."""
    if not is_numeric:
        # For non-numeric columns, collect distinct values as the name.
        values = []
        for row in token_rows:
            if col_index < len(row):
                values.append(row[col_index])
        distinct = sorted(set(values))
        if len(distinct) <= 3:
            return "/".join(distinct)
        return distinct[0] if distinct else f"col_{col_index}"
    # Numeric column: look at the token to the left for a label.
    for row in token_rows:
        if col_index < len(row) and _is_number_token(row[col_index]):
            if col_index > 0 and not _is_number_token(row[col_index - 1]):
                return row[col_index - 1]
            break
    return f"col_{col_index}"


def _extract_column_values(token_rows: list[list[str]], col_index: int) -> list[float]:
    """Extract float values from a specific column position."""
    values: list[float] = []
    for row in token_rows:
        if col_index < len(row) and _is_number_token(row[col_index]):
            values.append(float(row[col_index]))
    return values


def _find_grouping_column(token_rows: list[list[str]], numeric_positions: list[bool]) -> int | None:
    """Find a non-numeric column that could serve as a series grouping key.

    A grouping column has 2–10 distinct values and each value appears
    multiple times across the rows.
    """
    for col_index, is_numeric in enumerate(numeric_positions):
        if is_numeric:
            continue
        values: list[str] = []
        for row in token_rows:
            if col_index < len(row):
                values.append(row[col_index])
        distinct = set(values)
        if 2 <= len(distinct) <= 10:
            counts: dict[str, int] = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            if all(c >= 2 for c in counts.values()):
                return col_index
    return None


def parse_text(text: str, source_name: str = "<text>") -> ParsedData:
    """Parse a string into detected numeric columns.

    Lines are split on whitespace. Positions where ≥60% of tokens look
    like numbers are extracted as numeric columns. Column names are
    derived from the nearest non-numeric token to the left, or default
    to ``col_0``, ``col_1``, etc.
    """
    raw_lines = text.splitlines()
    token_rows = _tokenize_lines(raw_lines)
    numeric_positions = _detect_numeric_positions(token_rows)

    columns: list[Column] = []
    for i, is_num in enumerate(numeric_positions):
        name = _name_column(token_rows, i, is_num)
        if is_num:
            values = _extract_column_values(token_rows, i)
            if values:
                columns.append(Column(name=name, values=values))

    return ParsedData(
        columns=tuple(columns),
        raw_lines=tuple(raw_lines),
        source_name=source_name,
    )


def parse_file(path: Path) -> ParsedData:
    """Read and parse a file."""
    text = path.read_text()
    return parse_text(text, source_name=str(path))


def parse_stdin() -> ParsedData:
    """Read from stdin (for piped input)."""
    import sys

    text = sys.stdin.read()
    return parse_text(text, source_name="<stdin>")


def detect_numeric_columns(data: ParsedData) -> list[Column]:
    """Return only the numeric columns, in order of detection."""
    return list(data.columns)


def auto_series(data: ParsedData, x_col: int, y_col: int) -> list[Series]:
    """Create one or more Series from parsed data given x/y column indices.

    If a grouping column (non-numeric, repeating values like "Train"/"Val")
    is detected, rows are split into separate series by group. Otherwise,
    all rows form a single series.
    """
    token_rows = _tokenize_lines(list(data.raw_lines))
    numeric_positions = _detect_numeric_positions(token_rows)

    group_col = _find_grouping_column(token_rows, numeric_positions)

    # Map numeric column indices to their actual token-row positions.
    numeric_positions_list = [i for i, is_num in enumerate(numeric_positions) if is_num]
    if x_col >= len(numeric_positions_list) or y_col >= len(numeric_positions_list):
        return []

    if group_col is None:
        columns = list(data.columns)
        x_values = columns[x_col].values
        y_values = columns[y_col].values
        n = min(len(x_values), len(y_values))
        if n == 0:
            return []
        return [
            Series(
                name="data",
                x=np.array(x_values[:n]),
                y=np.array(y_values[:n]),
            )
        ]

    # Map numeric column indices to their actual token-row positions.
    numeric_positions_list = [i for i, is_num in enumerate(numeric_positions) if is_num]
    if x_col >= len(numeric_positions_list) or y_col >= len(numeric_positions_list):
        return []
    x_pos = numeric_positions_list[x_col]
    y_pos = numeric_positions_list[y_col]

    # Rebuild value lists with group tags using actual token positions.
    x_vals: list[float] = []
    y_vals: list[float] = []
    tags: list[str] = []
    for row in token_rows:
        if x_pos < len(row) and y_pos < len(row) and _is_number_token(row[x_pos]) and _is_number_token(row[y_pos]):
            x_vals.append(float(row[x_pos]))
            y_vals.append(float(row[y_pos]))
            tag = row[group_col] if group_col is not None and group_col < len(row) else "data"
            tags.append(tag)

    grouped: dict[str, list[tuple[float, float]]] = {}
    tag_order: list[str] = []
    for xv, yv, tag in zip(x_vals, y_vals, tags):
        if tag not in grouped:
            grouped[tag] = []
            tag_order.append(tag)
        grouped[tag].append((xv, yv))

    result: list[Series] = []
    for tag in tag_order:
        pairs = grouped[tag]
        result.append(
            Series(
                name=tag,
                x=np.array([p[0] for p in pairs]),
                y=np.array([p[1] for p in pairs]),
            )
        )
    return result
