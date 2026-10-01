"""Stage 19 — table-driven post-fixes applied by build/derive_final.py.

1. EP→BP spelling. `data/_ep_spelling_map.tsv` (Stage 1b's table) is applied
   whole-token and case-preserving to the learner-facing text fields
   (`pt`, `pt_display`, `example_pt`, `target_word_used`, and `family_root`,
   which must keep matching a `pt`) of 06-final. Stage 1b
   only normalized source headwords, so a handful of EP spellings survived
   into later stages (génio, ingénuo, polémica, cómodo). Upstream stage
   outputs stay untouched; the fix is re-applied on every derive.

2. IPA precedence for the card-displayed `ipa_word` / `ipa_example`:
       data/_manual_ipa.tsv  (user overrides, read-only here)
     > data/_ipa_v2.tsv      (Stage 19 validated São Paulo IPA)
     > data/05-ipa.tsv       (Stage 5 `ipa_*_final`)
   resolved field by field, so a manual word override doesn't clobber a v2
   sentence, and vice versa.
"""
from __future__ import annotations

import re
from pathlib import Path

from build.lib.tsv import read_tsv

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

SPELLING_FIELDS = ("pt", "pt_display", "example_pt", "target_word_used", "family_root")


def load_spelling_map(path: Path) -> dict[str, str]:
    """{lower-case EP form: BP form} from the Stage 1b spelling table."""
    out: dict[str, str] = {}
    for r in read_tsv(path):
        src = (r.get("source_form") or "").strip()
        dst = (r.get("bp_form") or "").strip()
        if src and dst:
            out[src.lower()] = dst
    return out


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_spelling_map(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Replace whole word tokens found in `mapping`; returns (text, n_changed)."""
    if not text or not mapping:
        return text, 0
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        tok = m.group(0)
        bp = mapping.get(tok.lower())
        if bp is None:
            return tok
        n += 1
        return _match_case(tok, bp)

    return _TOKEN_RE.sub(_sub, text), n


def apply_spelling_to_row(row: dict, mapping: dict[str, str]) -> int:
    """In-place fix of SPELLING_FIELDS; returns the number of tokens changed."""
    changed = 0
    for col in SPELLING_FIELDS:
        if col in row:
            row[col], n = apply_spelling_map(row[col] or "", mapping)
            changed += n
    return changed


def load_ipa_v2(path: Path) -> dict[str, dict]:
    """{sense_id: row} from data/_ipa_v2.tsv (missing file → {})."""
    return {r["sense_id"]: r for r in read_tsv(path) if r.get("sense_id")}


def load_manual_ipa(path: Path) -> dict[str, dict]:
    """{sense_id: row} from the user's data/_manual_ipa.tsv (never written here)."""
    return {r["sense_id"]: r for r in read_tsv(path) if r.get("sense_id")}


def resolve_ipa(
    sense_id: str,
    stage5_row: dict,
    ipa_v2: dict[str, dict],
    manual: dict[str, dict],
) -> tuple[str, str, str, str]:
    """Return (ipa_word, ipa_example, word_source, example_source)."""
    m = manual.get(sense_id) or {}
    v2 = ipa_v2.get(sense_id) or {}

    def pick(manual_key: str, v2_key: str, s5_key: str) -> tuple[str, str]:
        val = (m.get(manual_key) or "").strip()
        if val:
            return val, "manual"
        val = (v2.get(v2_key) or "").strip()
        if val:
            return val, "ipa_v2"
        return (stage5_row.get(s5_key) or ""), "stage5"

    word, wsrc = pick("ipa_word_override", "ipa_word", "ipa_word_final")
    example, esrc = pick("ipa_example_override", "ipa_example", "ipa_example_final")
    return word, example, wsrc, esrc
