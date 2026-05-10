"""Stage 8 / Step 0 — Risk pre-classifier.

Walks data/05-ipa.tsv joined to data/_audio_manifest.tsv (for voice_id),
applies 12 word-level + 2 voice-level patterns, emits
data/_audio_risk_classification.tsv with one row per word-clip sense_id.

Pure Python, no API calls, no LLM, idempotent. Input is read each time;
output is rewritten atomically.

Patterns (priority: P0 high → P3 low; min priority across matched patterns
becomes the row's `priority`):

  P0 | voice_high_risk_critical | voice_id ∈ high-risk seed (CRITICAL tier)
  P0 | user_reported            | pt ∈ user-reported failures seed
  P0 | ep_leftover_named        | pt ∈ Stage-1 EP audit seed
  P1 | voice_high_risk_elevated | voice_id ∈ high-risk seed (ELEVATED tier)
  P1 | same_spelling_en_pt      | pt ∈ EN-PT homograph seed
  P1 | final_l_vocalization     | regex `[a-z]+(al|el|il|ol)$` AND BP IPA ends in [w]
  P1 | h_initial_english_risk   | pt starts with 'h' (BP h- silent; EN aspirated)
  P1 | english_loanword         | pt ∈ EN loanword seed
  P2 | de_te_palatalization     | regex `[a-z]+(de|te)$` AND BP IPA contains [dʒi]/[tʃi]
  P2 | final_unstressed_e       | regex `[^aáâãeéêiíoóôõuú]e$`
  P2 | initial_r_or_rr          | starts with 'r' OR contains 'rr'
  P3 | coda_s_ep_risk           | regex `s[^aáâãeéêiíoóôõuú]`
  P3 | spanish_collision        | pt ∈ ES collision seed
  P3 | french_loanword          | pt ∈ FR loanword seed

Output schema:
  sense_id  pt  voice_id  risk_patterns  priority  ipa_word_final  rank

Usage:
    .venv/bin/python build/08_0_classify_risk.py [--out PATH]

Exit: 0 success.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
OUT_PATH = DATA_DIR / "_audio_risk_classification.tsv"

USER_REPORTED_SEED = DATA_DIR / "_audio_user_reported_failures.tsv"
HIGH_RISK_VOICES_SEED = DATA_DIR / "_risk_seeds_high_risk_voices.tsv"
EN_LOANWORDS_SEED = DATA_DIR / "_risk_seeds_en_loanwords.tsv"
ES_COLLISIONS_SEED = DATA_DIR / "_risk_seeds_es_collisions.tsv"
FR_LOANWORDS_SEED = DATA_DIR / "_risk_seeds_fr_loanwords.tsv"
SAME_SPELLING_SEED = DATA_DIR / "_risk_seeds_same_spelling_en_pt.tsv"

# EP-leftover named list — explicit, hand-curated (Stage-1 audit + plan additions)
EP_LEFTOVER_NAMED = {
    "camisola", "fazenda", "marcha", "troço", "vosso",
    "comboio", "equipa", "registo", "utilizador", "paragem",
    "desporto", "golo",
}

# Pattern → priority mapping (P0 = highest priority = lowest int)
PATTERN_PRIORITY: dict[str, int] = {
    "voice_high_risk_critical": 0,
    "user_reported": 0,
    "ep_leftover_named": 0,
    "voice_high_risk_elevated": 1,
    "same_spelling_en_pt": 1,
    "final_l_vocalization": 1,
    "h_initial_english_risk": 1,
    "english_loanword": 1,
    "de_te_palatalization": 2,
    "final_unstressed_e": 2,
    "initial_r_or_rr": 2,
    "coda_s_ep_risk": 3,
    "spanish_collision": 3,
    "french_loanword": 3,
}

# Regexes for patterns
_RE_FINAL_L = re.compile(r"[a-záàâãäéèêíìîóòôõúùûç]+(al|el|il|ol)$", re.IGNORECASE)
_RE_DE_TE = re.compile(r"[a-záàâãäéèêíìîóòôõúùûç]+(de|te)$", re.IGNORECASE)
_RE_FINAL_UNSTRESSED_E = re.compile(r"[^aáâãeéêiíoóôõuúyAÁÂÃEÉÊIÍOÓÔÕUÚY\s\-]e$")
_RE_CODA_S = re.compile(r"s[^aáâãeéêiíoóôõuúyAÁÂÃEÉÊIÍOÓÔÕUÚY\s\-]", re.IGNORECASE)


def _read_seed_pt(path: Path) -> set[str]:
    """Read a seed TSV and return the set of pt values (strips, dedups)."""
    if not path.exists():
        return set()
    rows = read_tsv(path)
    pts: set[str] = set()
    for r in rows:
        pt = (r.get("pt") or "").strip()
        if pt and not pt.startswith("#"):
            pts.add(pt)
    return pts


def _read_high_risk_voices(path: Path) -> dict[str, str]:
    """Return voice_id → tier ('CRITICAL' / 'ELEVATED')."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, str] = {}
    for r in rows:
        vid = (r.get("voice_id") or "").strip()
        tier = (r.get("risk_tier") or "").strip().upper()
        if vid and not vid.startswith("#") and tier in {"CRITICAL", "ELEVATED"}:
            out[vid] = tier
    return out


def _voice_id_for_word_clip(manifest_rows: list[dict]) -> dict[str, str]:
    """Build sense_id → voice_id from manifest's word clips."""
    out: dict[str, str] = {}
    for r in manifest_rows:
        if r.get("clip_type") == "word":
            sid = r.get("sense_id", "")
            vid = r.get("voice_id", "")
            if sid and vid:
                out[sid] = vid
    return out


def classify_row(
    pt: str,
    ipa_word_final: str,
    voice_id: str,
    *,
    user_reported: set[str],
    high_risk_voices: dict[str, str],
    en_loanwords: set[str],
    es_collisions: set[str],
    fr_loanwords: set[str],
    same_spelling: set[str],
) -> list[str]:
    """Return sorted list of matched patterns for a single sense."""
    pt_low = pt.lower().strip()
    ipa = ipa_word_final or ""
    patterns: list[str] = []

    # Voice-level (independent of pt)
    if voice_id in high_risk_voices:
        tier = high_risk_voices[voice_id]
        if tier == "CRITICAL":
            patterns.append("voice_high_risk_critical")
        elif tier == "ELEVATED":
            patterns.append("voice_high_risk_elevated")

    # Word-level: P0 named lists
    if pt_low in user_reported:
        patterns.append("user_reported")
    if pt_low in EP_LEFTOVER_NAMED:
        patterns.append("ep_leftover_named")

    # Word-level: P1 named lists
    if pt_low in same_spelling:
        patterns.append("same_spelling_en_pt")
    if pt_low in en_loanwords:
        patterns.append("english_loanword")

    # Word-level: P1 regex
    if _RE_FINAL_L.match(pt_low) and (ipa.endswith("w") or ipa.endswith("ʊ")):
        patterns.append("final_l_vocalization")
    if pt_low.startswith("h"):
        patterns.append("h_initial_english_risk")

    # Word-level: P2 regex
    if _RE_DE_TE.match(pt_low) and ("dʒi" in ipa or "tʃi" in ipa or "dʒ" in ipa or "tʃ" in ipa):
        patterns.append("de_te_palatalization")
    if _RE_FINAL_UNSTRESSED_E.search(pt_low):
        patterns.append("final_unstressed_e")
    if pt_low.startswith("r") or "rr" in pt_low:
        patterns.append("initial_r_or_rr")

    # Word-level: P3
    if _RE_CODA_S.search(pt_low):
        patterns.append("coda_s_ep_risk")
    if pt_low in es_collisions:
        patterns.append("spanish_collision")
    if pt_low in fr_loanwords:
        patterns.append("french_loanword")

    return sorted(set(patterns))


def priority_for_patterns(patterns: list[str]) -> str:
    """Return min-priority label P0/P1/P2/P3 (or — if no patterns)."""
    if not patterns:
        return "—"
    p = min(PATTERN_PRIORITY.get(pat, 99) for pat in patterns)
    return f"P{p}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    # Load all inputs
    if not IPA_PATH.exists():
        parser.error(f"missing {IPA_PATH}")
    if not MANIFEST_PATH.exists():
        parser.error(f"missing {MANIFEST_PATH}")

    ipa_rows = read_tsv(IPA_PATH)
    manifest_rows = read_tsv(MANIFEST_PATH)

    # Manifest may not have all senses if examples-only or other; build per-sense voice map from word clips
    voice_map = _voice_id_for_word_clip(manifest_rows)

    # Seeds
    user_reported = _read_seed_pt(USER_REPORTED_SEED)
    high_risk_voices = _read_high_risk_voices(HIGH_RISK_VOICES_SEED)
    en_loanwords = _read_seed_pt(EN_LOANWORDS_SEED)
    es_collisions = _read_seed_pt(ES_COLLISIONS_SEED)
    fr_loanwords = _read_seed_pt(FR_LOANWORDS_SEED)
    same_spelling = _read_seed_pt(SAME_SPELLING_SEED)

    # Filter ipa_rows to only senses that have a word clip in the manifest
    out_rows: list[dict] = []
    pattern_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    senses_with_no_word_clip = 0

    for r in ipa_rows:
        sid = r.get("sense_id", "")
        pt = r.get("pt", "")
        ipa = r.get("ipa_word_final", "")
        rank = r.get("rank", "")
        if sid not in voice_map:
            senses_with_no_word_clip += 1
            continue
        vid = voice_map[sid]
        patterns = classify_row(
            pt,
            ipa,
            vid,
            user_reported=user_reported,
            high_risk_voices=high_risk_voices,
            en_loanwords=en_loanwords,
            es_collisions=es_collisions,
            fr_loanwords=fr_loanwords,
            same_spelling=same_spelling,
        )
        priority = priority_for_patterns(patterns)
        for p in patterns:
            pattern_counts[p] += 1
        priority_counts[priority] += 1

        out_rows.append({
            "sense_id": sid,
            "pt": pt,
            "voice_id": vid,
            "risk_patterns": ",".join(patterns) if patterns else "none",
            "priority": priority,
            "ipa_word_final": ipa,
            "rank": rank,
        })

    # Write atomically
    fieldnames = ["sense_id", "pt", "voice_id", "risk_patterns", "priority", "ipa_word_final", "rank"]
    write_tsv(Path(args.out), out_rows, fieldnames=fieldnames)

    # Summary
    print(f"Wrote {len(out_rows)} rows to {args.out}")
    print(f"  senses_with_no_word_clip skipped: {senses_with_no_word_clip}")
    print()
    print("Priority distribution:")
    for prio in ("P0", "P1", "P2", "P3", "—"):
        print(f"  {prio:<3}  {priority_counts.get(prio, 0):>5}")
    print()
    print("Pattern counts (a clip can match multiple patterns):")
    for pat in sorted(PATTERN_PRIORITY.keys(), key=lambda p: (PATTERN_PRIORITY[p], p)):
        prio = PATTERN_PRIORITY[pat]
        n = pattern_counts.get(pat, 0)
        print(f"  P{prio}  {pat:<30}  {n:>5}")

    risky_total = sum(priority_counts.get(p, 0) for p in ("P0", "P1", "P2", "P3"))
    print()
    print(f"Risky bucket size: {risky_total} / {len(out_rows)} ({100 * risky_total / max(len(out_rows), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
