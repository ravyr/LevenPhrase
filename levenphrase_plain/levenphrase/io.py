"""Input helpers for batch checking text files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

PairFormat = Literal["auto", "tsv", "csv", "pipe", "pairs", "triples", "single"]


@dataclass(frozen=True, slots=True)
class TextPair:
    reference: str
    hypothesis: str
    label: str | None = None
    line_number: int | None = None


def _clean_lines(path: str | Path, *, encoding: str = "utf-8", skip_blank: bool = True) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with Path(path).open("r", encoding=encoding) as file:
        for number, raw in enumerate(file, start=1):
            line = raw.rstrip("\n\r")
            if skip_blank and not line.strip():
                continue
            rows.append((number, line.strip()))
    return rows


def detect_pair_format(lines: list[tuple[int, str]]) -> PairFormat:
    if not lines:
        return "single"
    if any("\t" in line for _, line in lines):
        return "tsv"
    if any("||" in line for _, line in lines):
        return "pipe"
    # Historical format: number/id, reference, hypothesis repeated.
    if len(lines) % 3 == 0 and all(lines[i][1].strip().isdigit() for i in range(0, len(lines), 3)):
        return "triples"
    if len(lines) % 2 == 0:
        return "pairs"
    return "single"


def load_text_pairs(
    path: str | Path,
    *,
    format: PairFormat = "auto",
    encoding: str = "utf-8",
    reference_column: int = 0,
    hypothesis_column: int = 1,
    label_column: int | None = None,
    delimiter: str | None = None,
) -> list[TextPair]:
    """Load text pairs from a file.

    Supported formats:

    * ``tsv``: one pair per line, reference<TAB>hypothesis,
    * ``csv``: configurable columns,
    * ``pipe``: one pair per line, reference||hypothesis,
    * ``pairs``: two lines per pair: reference, hypothesis,
    * ``triples``: historical format: id, reference, hypothesis,
    * ``single``: one text per line; the line is compared with itself unless
      you later pair it manually.
    """
    lines = _clean_lines(path, encoding=encoding)
    chosen = detect_pair_format(lines) if format == "auto" else format

    if chosen == "tsv":
        output: list[TextPair] = []
        for line_number, line in lines:
            parts = line.split("\t")
            if len(parts) <= max(reference_column, hypothesis_column):
                raise ValueError(f"Line {line_number} has too few TSV columns: {line!r}")
            label = parts[label_column] if label_column is not None and label_column < len(parts) else None
            output.append(TextPair(parts[reference_column], parts[hypothesis_column], label, line_number))
        return output

    if chosen == "csv":
        output = []
        with Path(path).open("r", encoding=encoding, newline="") as file:
            reader = csv.reader(file, delimiter=delimiter or ",")
            for line_number, parts in enumerate(reader, start=1):
                if not parts:
                    continue
                if len(parts) <= max(reference_column, hypothesis_column):
                    raise ValueError(f"Line {line_number} has too few CSV columns: {parts!r}")
                label = parts[label_column] if label_column is not None and label_column < len(parts) else None
                output.append(TextPair(parts[reference_column].strip(), parts[hypothesis_column].strip(), label, line_number))
        return output

    if chosen == "pipe":
        output = []
        for line_number, line in lines:
            parts = [part.strip() for part in line.split("||")]
            if len(parts) < 2:
                raise ValueError(f"Line {line_number} is not a pipe pair: {line!r}")
            output.append(TextPair(parts[0], parts[1], parts[2] if len(parts) > 2 else None, line_number))
        return output

    if chosen == "pairs":
        if len(lines) % 2:
            raise ValueError("pairs format requires an even number of nonblank lines")
        return [
            TextPair(lines[i][1], lines[i + 1][1], label=str(i // 2 + 1), line_number=lines[i][0])
            for i in range(0, len(lines), 2)
        ]

    if chosen == "triples":
        if len(lines) % 3:
            raise ValueError("triples format requires groups of id/reference/hypothesis")
        output = []
        for i in range(0, len(lines), 3):
            output.append(TextPair(lines[i + 1][1], lines[i + 2][1], label=lines[i][1], line_number=lines[i][0]))
        return output

    if chosen == "single":
        return [TextPair(line, line, label=str(idx + 1), line_number=line_number) for idx, (line_number, line) in enumerate(lines)]

    raise ValueError(f"Unsupported pair format: {chosen!r}")


def load_lines(path: str | Path, *, encoding: str = "utf-8", skip_blank: bool = True) -> list[str]:
    return [line for _, line in _clean_lines(path, encoding=encoding, skip_blank=skip_blank)]
