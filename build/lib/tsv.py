"""TSV read/write helpers.

All stage outputs use Python's csv module with dialect='excel-tab' and
QUOTE_MINIMAL. This handles the one source line (262) with internal "folk"
quotes correctly.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

# Maximum field size — bumped above default to handle long en_all and source_line
csv.field_size_limit(10_000_000)


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a TSV file into a list of dict rows. Empty file → empty list."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        return list(reader)


def iter_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    """Stream rows from a TSV without loading the whole file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        yield from reader


def write_tsv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> int:
    """Write rows to a TSV file. Returns row count written.

    Coerces None → "" and non-string values → str() so the output is always valid TSV.
    Atomic: writes a sibling temp file and `os.replace`s it over the target, so
    a crash mid-write can never leave a truncated file behind.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                dialect="excel-tab",
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                sanitized = {k: ("" if v is None else str(v)) for k, v in row.items()}
                writer.writerow(sanitized)
                written += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return written


def append_tsv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> int:
    """Append rows to a TSV file, creating with header if missing."""
    p = Path(path)
    exists = p.exists() and p.stat().st_size > 0
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            dialect="excel-tab",
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            sanitized = {k: ("" if v is None else str(v)) for k, v in row.items()}
            writer.writerow(sanitized)
            written += 1
    return written
