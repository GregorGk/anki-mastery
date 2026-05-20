"""Stage 18 — deterministic pilot selection + Stage-17 TSV reader.

The pilot `.apkg` is the import smoke-test run before the full build. It
must (a) exercise the real beginning of the deck, (b) include the known
audio trouble cases so they can be eyeballed inside real Anki, and (c)
hit every template branch (optional fields, risk notes, genders, POS).

`select_pilot_sense_ids` is fully deterministic (ordered by `anki_order`)
so the pilot is byte-stable across rebuilds. Shared by `18_0`
(fetch only the pilot's media) and `18_1` (build the pilot deck).

`read_anki_export` skips the two `#…` directive lines the Stage-17 clean
TSV starts with — `csv.DictReader` would otherwise treat `#separator:tab`
as the header.
"""
from __future__ import annotations

import csv
from pathlib import Path

# The 12 spike-bad senses (from build/phoneme_asr_spike.py BATCH, BAD side):
# 6 legacy human-confirmed mispronunciations + 6 v3 Gemini-flagged non_bp.
SPIKE_BAD_SENSE_IDS = [
    "0937.00.01", "3009.00.01", "1175.00.01", "3536.00.01", "3013.00.01", "4064.00.01",
    "0534.00.01", "0002.00.01", "0304.00.01", "0913.00.01", "1454.00.01", "0580.00.01",
]

PILOT_FIRST_N = 100

# First row (by anki_order) matching each predicate is pulled into the
# pilot, so every template branch is exercised at least once. `long_example`
# is handled separately (max length, not first-match).
PILOT_EDGE_CASE_PREDICATES = [
    ("usage_hint",     lambda r: bool((r.get("usage_hint") or "").strip())),
    ("risk_note",      lambda r: bool((r.get("risk_note") or "").strip())),
    ("false_friend",   lambda r: "false_friend" in (r.get("risk_flags") or "")),
    ("risk_flags",     lambda r: (r.get("risk_flags") or "").strip() not in ("", "none")),
    ("bp_nonstandard", lambda r: (r.get("bp_validity") or "").strip() not in ("", "standard")),
    ("family_root",    lambda r: bool((r.get("family_root") or "").strip())),
    ("noun_gender",    lambda r: (r.get("pos") or "") == "noun" and bool((r.get("gender") or "").strip())),
    ("verb",           lambda r: (r.get("pos") or "") == "verb"),
    ("adjective",      lambda r: (r.get("pos") or "") == "adj"),
    ("empty_optional", lambda r: not (r.get("usage_hint") or "").strip()
                                 and not (r.get("risk_note") or "").strip()),
]


def read_anki_export(path: str | Path) -> list[dict]:
    """Read the Stage-17 clean TSV, skipping its `#…` directive lines."""
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return list(csv.DictReader(lines, dialect="excel-tab"))


def _anki_order(row: dict) -> int:
    try:
        return int(row.get("anki_order") or 0)
    except (TypeError, ValueError):
        return 0


def select_pilot_sense_ids(rows: list[dict]) -> list[str]:
    """Deterministic pilot set: first-100 + spike-bads + edge cases.

    `rows` are Stage-17 export dicts (any order; sorted here by anki_order).
    Returns sense_ids in selection order, deduped (~120–130 senses).
    """
    ordered = sorted(rows, key=_anki_order)
    by_sense = {r["sense_id"]: r for r in ordered}

    selected: list[str] = []
    seen: set[str] = set()

    def add(sid: str) -> None:
        if sid and sid not in seen and sid in by_sense:
            seen.add(sid)
            selected.append(sid)

    # 1. first 100 by anki_order — tests the real start of the deck
    for r in ordered[:PILOT_FIRST_N]:
        add(r["sense_id"])

    # 2. known audio trouble cases
    for sid in SPIKE_BAD_SENSE_IDS:
        add(sid)

    # 3. the single longest example sentence (tie-break by anki_order)
    longest = max(ordered, key=lambda r: (len(r.get("example_pt") or ""), -_anki_order(r)),
                  default=None)
    if longest is not None:
        add(longest["sense_id"])

    # 4. first-match edge cases — one card per template branch
    for _label, pred in PILOT_EDGE_CASE_PREDICATES:
        for r in ordered:
            if pred(r):
                add(r["sense_id"])
                break

    return selected
