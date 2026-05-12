"""Derive `data/06-final.tsv` — the project's final deliverable.

Joins:
  - data/03-enriched.tsv      (base: sense_id, source_pt, annotation, family_root,
                               source_line, source_line_number, tags, en_all)
  - data/05-ipa.tsv           (example_pt, example_en, target_word_used,
                               ipa_word_final, ipa_example_final + most other cols)
  - data/_audio_manifest.tsv  (audio URLs + md5 + voice_id, per (sense_id, clip_type).
                               clip_type ∈ {word, example, en_ex})
  - config/voices.tsv         (voice_gender from BP voice_id; en_voice_id pairing)
  - data/_audio_asr_override.tsv (notes flag if sense's audio is ASR-untranscribable-but-bp-ok)
  - data/_usage_hints.tsv     (Stage 12: usage_hint, usage_hint_priority, risk_note
                               per sense_id for verb/idiom rows)
  - data/_family_roots.tsv    (Stage 13: family_root for content-word rows;
                               LLM-classified, post-processed, cluster-validated)

Produces 37 columns per the schema in docs/plan.md §"Final TSV schema". History:
  v1: 30 cols (Stage 9 baseline)
  +3 EN audio cols (Stage 10):     audio_en_example, audio_en_example_md5, voice_id_en
  +1 POS-safe display (Stage 10.5): pt_display_safe (placed after notes)
  +3 usage-hint cols (Stage 12):   usage_hint, usage_hint_priority, risk_note (appended)
  Stage 13: no schema change — fills the previously-empty `family_root` column
            from data/_family_roots.tsv, validated against pt set + cluster
            size ≥ 2 distinct pt lemmas.

Export filter for Stage 12 columns:
  usage_hint        = essential always; useful only if rank<=1000 or row has
                      #reflexive / #false-friend tag or non-empty risk_note
  usage_hint_priority = same gate as usage_hint; for risk_note-only rows
                      (gozar), priority stays empty
  risk_note         = always carried through when sidecar has it

Audit trail of the join (which sidecar row produced which output) is appended
to audit/12_usage_hints.jsonl as one JSONL record per sense_id with a hint.

Usage:
    .venv/bin/python build/derive_final.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
AUDIT_DIR = REPO_ROOT / "audit"

ENRICHED_PATH = DATA_DIR / "03-enriched.tsv"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
OVERRIDE_PATH = DATA_DIR / "_audio_asr_override.tsv"
USAGE_HINTS_PATH = DATA_DIR / "_usage_hints.tsv"
FAMILY_ROOTS_PATH = DATA_DIR / "_family_roots.tsv"
TOPIC_TAGS_PATH = DATA_DIR / "_topic_tags.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"

FINAL_PATH = DATA_DIR / "06-final.tsv"
LOG_PATH = AUDIT_DIR / "derive_final.log"
USAGE_HINT_AUDIT = AUDIT_DIR / "12_usage_hints.jsonl"
FAMILY_ROOT_AUDIT = AUDIT_DIR / "13_family_root.jsonl"
TOPIC_TAG_AUDIT = AUDIT_DIR / "14_topic_tags.jsonl"

# Content POS — Stage 13 only ships `family_root` for these.
FAMILY_ROOT_CONTENT_POS = ("noun", "verb", "adj", "adv")
# Provenance values that bypass the high-confidence requirement.
FAMILY_ROOT_DETERMINISTIC_SOURCES = ("postprocess_self_root", "manual")

# Tags that promote a `useful`-priority hint to be included in the final
# learner-facing column even when rank > 1000.
USAGE_HINT_RISK_TAGS = ("#reflexive", "#false-friend")
USAGE_HINT_USEFUL_RANK_CEILING = 1000

# Hint length budget. Anything longer is allowed (the LLM/post-process
# already aimed for ≤ 70 chars) but flagged by verify_all.
USAGE_HINT_MAX_CHARS = 120

FINAL_FIELDS = [
    "sense_id", "rank", "source_pt", "pt", "pt_type", "gender",
    "pt_display", "pos", "sense_index", "en_primary", "en_all",
    "annotation", "bp_status", "normalization_action", "ipa_word",
    "example_pt", "example_en", "target_word_used", "ipa_example",
    "audio_word", "audio_example", "audio_word_md5", "audio_example_md5",
    "audio_en_example", "audio_en_example_md5",
    "voice_id", "voice_id_en", "voice_gender", "family_root", "tags",
    "source_line", "source_line_number", "notes",
    "pt_display_safe",
    "usage_hint", "usage_hint_priority", "risk_note",
]


# Maps every non-noun POS to the short tag wrapped in parentheses in
# `pt_display_safe`. Nouns are omitted intentionally — they keep the article-
# prefixed form unchanged (e.g. "o banco") so the gender is taught directly.
SHORT_POS_TAG = {
    "adj":    "adj",
    "verb":   "v",
    "adv":    "adv",
    "prep":   "prep",
    "conj":   "conj",
    "num":    "num",
    "pron":   "pron",
    "interj": "interj",
    "idiom":  "idiom",
    "art":    "art",
}

# Strip a leading Portuguese definite or indefinite article from a display
# string. Anchored on `^` + whitespace so internal articles in multi-word
# entries (e.g. "à medida que" — `à` is not in this set anyway) are left
# alone. Longer / more specific patterns come first so `o/a banco` strips
# `o/a `, not `o `.
_LEADING_ARTICLE_RE = re.compile(r"^(o/a|os|as|um|uma|uns|umas|o|a)\s+")


def _pt_display_safe(pt_display: str, pos: str) -> str:
    """Return a POS-safe display string.

    Nouns: returned unchanged (the article prefix teaches gender).
    Everything else: leading article stripped, short POS tag prepended in
    parens. See SHORT_POS_TAG for the abbreviation set.
    """
    if pos == "noun":
        return pt_display
    bare = _LEADING_ARTICLE_RE.sub("", pt_display, count=1)
    tag = SHORT_POS_TAG.get(pos, pos)
    return f"({tag}) {bare}"


def _voice_gender_map() -> dict[str, str]:
    return {r["voice_id"]: r.get("gender", "")
            for r in read_tsv(VOICES_PATH)}


def _en_voice_pairing() -> dict[str, str]:
    """BP voice_id → paired EN voice_id from config/voices.tsv."""
    return {r["voice_id"]: r.get("en_voice_id", "")
            for r in read_tsv(VOICES_PATH)}


def _load_usage_hints() -> dict[str, dict]:
    """Read data/_usage_hints.tsv as {sense_id: row}. Empty if file missing."""
    if not USAGE_HINTS_PATH.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(USAGE_HINTS_PATH)}


def _load_family_roots() -> dict[str, dict]:
    """Read data/_family_roots.tsv as {sense_id: row}. Empty if file missing."""
    if not FAMILY_ROOTS_PATH.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(FAMILY_ROOTS_PATH)}


def _load_topic_tags() -> dict[str, dict]:
    """Read data/_topic_tags.tsv as {sense_id: row}. Empty if file missing."""
    if not TOPIC_TAGS_PATH.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(TOPIC_TAGS_PATH)}


def _merge_topic_tag(current_tags: str, topic: str) -> str:
    """Append the Stage-14 topic tag to `tags`, space-separated, dedupe.

    Order: existing tags preserved verbatim; `#topic-*` appended at the end.
    Matches the natural Stage-N append convention already in 06-final.tsv.
    """
    if not topic:
        return current_tags
    parts = current_tags.split() if current_tags else []
    if topic in parts:
        return current_tags
    parts.append(topic)
    return " ".join(parts)


def _resolve_family_root(
    *,
    pos: str,
    sidecar_row: dict | None,
    all_pts: set[str],
) -> str:
    """Apply the Stage-13 export filter; return the family_root to ship.

    Rules:
      - pos not in content set → empty (defense-in-depth)
      - sidecar missing or family_root empty → empty
      - family_root not in all_pts → empty (rejects hallucinated/stale roots)
      - confidence != 'high' AND source not in deterministic sources → empty
    """
    if pos not in FAMILY_ROOT_CONTENT_POS:
        return ""
    if sidecar_row is None:
        return ""
    root = (sidecar_row.get("family_root") or "").strip()
    if not root or root not in all_pts:
        return ""
    confidence = (sidecar_row.get("confidence") or "").strip().lower()
    source = (sidecar_row.get("source") or "llm").strip()
    if source in FAMILY_ROOT_DETERMINISTIC_SOURCES:
        return root
    if confidence != "high":
        return ""
    return root


def _resolve_usage_hint_fields(
    *,
    sense_id: str,
    rank_str: str,
    tags: str,
    sidecar_row: dict | None,
) -> tuple[str, str, str]:
    """Apply the Stage-12 export filter and return the three final-TSV fields.

    Returns (usage_hint, usage_hint_priority, risk_note). Defaults: all empty.

    Rules:
      - risk_note is always carried through if the sidecar has it.
      - Gozar-style rows (risk_note populated, usage_hint empty) ship with
        an empty `usage_hint_priority` so the learner-facing card doesn't
        light up the priority chip — only the risk_note surfaces.
      - usage_hint priority "essential" always ships.
      - "useful" ships only if rank <= USAGE_HINT_USEFUL_RANK_CEILING, OR
        the row carries a risk tag (#reflexive / #false-friend), OR the
        row has a non-empty risk_note (defensive — risk_note is its own
        gate but a useful hint paired with one is still worth surfacing).
      - "omit" never ships a hint.
    """
    if sidecar_row is None:
        return "", "", ""
    hint = (sidecar_row.get("usage_hint") or "").strip().replace("\n", " ")
    priority = sidecar_row.get("hint_priority") or "omit"
    risk = (sidecar_row.get("risk_note") or "").strip().replace("\n", " ")

    # risk_note-only rows: ship risk_note, suppress usage_hint_priority.
    if not hint and risk:
        return "", "", risk

    include = False
    if priority == "essential":
        include = True
    elif priority == "useful":
        try:
            rank = int(rank_str or "0")
        except ValueError:
            rank = 0
        has_risk_tag = any(t in tags for t in USAGE_HINT_RISK_TAGS)
        if rank and rank <= USAGE_HINT_USEFUL_RANK_CEILING:
            include = True
        elif has_risk_tag:
            include = True
        elif risk:
            include = True

    if not include:
        return "", "", risk
    return hint, priority, risk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute stats, don't write 06-final.tsv")
    args = parser.parse_args()

    enriched = read_tsv(ENRICHED_PATH)
    ipa = read_tsv(IPA_PATH)
    manifest = read_tsv(MANIFEST_PATH)
    overrides = read_tsv(OVERRIDE_PATH) if OVERRIDE_PATH.exists() else []

    voice_gender = _voice_gender_map()
    en_pairing = _en_voice_pairing()
    usage_hints = _load_usage_hints()
    family_roots = _load_family_roots()
    topic_tags = _load_topic_tags()

    enriched_idx = {r["sense_id"]: r for r in enriched}
    ipa_idx = {r["sense_id"]: r for r in ipa}
    manifest_idx: dict[tuple[str, str], dict] = {
        (r["sense_id"], r["clip_type"]): r for r in manifest
    }
    override_sids = {r["sense_id"] for r in overrides}

    # Cross-check: same sense_ids in 03 and 05?
    en_sids = set(enriched_idx.keys())
    ipa_sids = set(ipa_idx.keys())
    only_en = en_sids - ipa_sids
    only_ipa = ipa_sids - en_sids
    if only_en or only_ipa:
        print(f"WARN: sense_id mismatch: only in 03={len(only_en)}, only in 05={len(ipa_sids - en_sids)}",
              file=sys.stderr)
    sids = sorted(en_sids & ipa_sids)
    print(f"Joining {len(sids)} senses (intersection of 03 + 05)")

    # Build the universe of pt values once. Stage 13's _resolve_family_root
    # validates against this set so hallucinated / stale roots don't ship.
    all_pts: set[str] = set()
    for sid in sids:
        e = enriched_idx[sid]
        ip = ipa_idx[sid]
        pt = e.get("pt", "") or ip.get("pt", "")
        if pt:
            all_pts.add(pt)

    rows: list[dict] = []
    gap_stats = {
        "missing_audio_word": 0,
        "missing_audio_example": 0,
        "missing_audio_en_example": 0,
        "missing_voice_id": 0,
        "missing_voice_id_en": 0,
        "missing_md5_word": 0,
        "missing_md5_example": 0,
        "missing_md5_en_example": 0,
        "voice_mismatch": 0,
        "en_voice_pairing_mismatch": 0,
        "asr_override_applied": 0,
        "family_root_shipped": 0,
        "family_root_dropped_low_conf": 0,
        "family_root_dropped_not_in_pts": 0,
        "family_root_dropped_non_content_pos": 0,
        "family_root_self_root_shipped": 0,
        "family_root_blank_content": 0,
        "family_root_blank_function": 0,
        "source_line_blank": 0,
        # Stage 12 usage hints
        "usage_hint_essential": 0,
        "usage_hint_useful_kept": 0,
        "usage_hint_useful_dropped": 0,
        "usage_hint_omit": 0,
        "usage_hint_risk_note_only": 0,
        "usage_hint_total_shipped": 0,
        "risk_note_shipped": 0,
        # Stage 14 topic tags
        "topic_tag_shipped": 0,
        "topic_tag_missing": 0,
        "topic_tag_low_conf": 0,
        "topic_tag_deterministic": 0,
        "topic_tag_llm": 0,
    }

    for sid in sids:
        e = enriched_idx[sid]
        ip = ipa_idx[sid]
        mw = manifest_idx.get((sid, "word"), {})
        mex = manifest_idx.get((sid, "example"), {})
        men = manifest_idx.get((sid, "en_ex"), {})

        if not mw.get("url"):
            gap_stats["missing_audio_word"] += 1
        if not mex.get("url"):
            gap_stats["missing_audio_example"] += 1
        if not men.get("url"):
            gap_stats["missing_audio_en_example"] += 1
        if not mw.get("md5"):
            gap_stats["missing_md5_word"] += 1
        if not mex.get("md5"):
            gap_stats["missing_md5_example"] += 1
        if not men.get("md5"):
            gap_stats["missing_md5_en_example"] += 1

        voice_id = mw.get("voice_id") or mex.get("voice_id") or ""
        if not voice_id:
            gap_stats["missing_voice_id"] += 1
        elif mw.get("voice_id") and mex.get("voice_id") and mw["voice_id"] != mex["voice_id"]:
            gap_stats["voice_mismatch"] += 1
            voice_id = mw["voice_id"]  # prefer word
        vgender_word = voice_gender.get(voice_id, "")
        vgender_short = "m" if vgender_word == "male" else ("f" if vgender_word == "female" else "")

        # voice_id_en: prefer the en_ex manifest row's voice_id (authoritative —
        # the voice that actually spoke). Fall back to config/voices.tsv pairing
        # if the manifest row is missing. Flag if they disagree.
        voice_id_en_manifest = men.get("voice_id") or ""
        voice_id_en_pairing = en_pairing.get(voice_id, "")
        voice_id_en = voice_id_en_manifest or voice_id_en_pairing
        if not voice_id_en:
            gap_stats["missing_voice_id_en"] += 1
        elif (voice_id_en_manifest and voice_id_en_pairing
              and voice_id_en_manifest != voice_id_en_pairing):
            gap_stats["en_voice_pairing_mismatch"] += 1

        if not e.get("source_line"):
            gap_stats["source_line_blank"] += 1

        notes_parts: list[str] = []
        if sid in override_sids:
            notes_parts.append("audio_asr_status=untranscribable_audio_verified_bp")
            gap_stats["asr_override_applied"] += 1
        notes = " | ".join(notes_parts)

        pt_display_val = e.get("pt_display", "") or ip.get("pt_display", "")
        pos_val = e.get("pos", "") or ip.get("pos", "")
        rank_val = e.get("rank", "") or ip.get("rank", "")
        tags_val = e.get("tags", "") or ip.get("tags", "")

        sidecar_row = usage_hints.get(sid)
        uh_hint, uh_priority, uh_risk = _resolve_usage_hint_fields(
            sense_id=sid, rank_str=rank_val, tags=tags_val,
            sidecar_row=sidecar_row,
        )

        # Stage 13: family_root from sidecar with export filter.
        fr_sidecar = family_roots.get(sid)
        family_root = _resolve_family_root(
            pos=pos_val, sidecar_row=fr_sidecar, all_pts=all_pts,
        )
        # Account for the why-empty paths for the log.
        if pos_val not in FAMILY_ROOT_CONTENT_POS:
            if not family_root:
                gap_stats["family_root_blank_function"] += 1
        else:
            if family_root:
                gap_stats["family_root_shipped"] += 1
                source = (fr_sidecar.get("source") or "").strip() if fr_sidecar else ""
                if source in FAMILY_ROOT_DETERMINISTIC_SOURCES:
                    gap_stats["family_root_self_root_shipped"] += 1
            else:
                gap_stats["family_root_blank_content"] += 1
                if fr_sidecar:
                    sc_root = (fr_sidecar.get("family_root") or "").strip()
                    sc_conf = (fr_sidecar.get("confidence") or "").strip().lower()
                    sc_source = (fr_sidecar.get("source") or "llm").strip()
                    if sc_root and sc_root not in all_pts:
                        gap_stats["family_root_dropped_not_in_pts"] += 1
                    elif (sc_root and sc_conf != "high"
                          and sc_source not in FAMILY_ROOT_DETERMINISTIC_SOURCES):
                        gap_stats["family_root_dropped_low_conf"] += 1
        if sidecar_row is not None:
            raw_priority = sidecar_row.get("hint_priority") or "omit"
            raw_hint = (sidecar_row.get("usage_hint") or "").strip()
            raw_risk = (sidecar_row.get("risk_note") or "").strip()
            if not raw_hint and raw_risk:
                gap_stats["usage_hint_risk_note_only"] += 1
            elif raw_priority == "essential":
                gap_stats["usage_hint_essential"] += 1
            elif raw_priority == "useful":
                if uh_hint:
                    gap_stats["usage_hint_useful_kept"] += 1
                else:
                    gap_stats["usage_hint_useful_dropped"] += 1
            elif raw_priority == "omit":
                gap_stats["usage_hint_omit"] += 1
        if uh_hint:
            gap_stats["usage_hint_total_shipped"] += 1
        if uh_risk:
            gap_stats["risk_note_shipped"] += 1

        # Stage 14: merge topic tag into the existing tags string.
        topic_sidecar = topic_tags.get(sid) or {}
        topic_primary = (topic_sidecar.get("topic_primary") or "").strip()
        if topic_primary:
            tags_val = _merge_topic_tag(tags_val, topic_primary)
            gap_stats["topic_tag_shipped"] += 1
            src = (topic_sidecar.get("source") or "").strip()
            if src == "deterministic":
                gap_stats["topic_tag_deterministic"] += 1
            elif src == "llm":
                gap_stats["topic_tag_llm"] += 1
            if (topic_sidecar.get("confidence") or "").strip().lower() != "high":
                gap_stats["topic_tag_low_conf"] += 1
        else:
            gap_stats["topic_tag_missing"] += 1

        rows.append({
            "sense_id": sid,
            "rank": rank_val,
            "source_pt": e.get("source_pt", ""),
            "pt": e.get("pt", "") or ip.get("pt", ""),
            "pt_type": e.get("pt_type", "") or ip.get("pt_type", ""),
            "gender": e.get("gender", "") or ip.get("gender", ""),
            "pt_display": pt_display_val,
            "pos": pos_val,
            "sense_index": e.get("sense_index", "") or ip.get("sense_index", ""),
            "en_primary": e.get("en_primary", "") or ip.get("en_primary", ""),
            "en_all": e.get("en_all", "") or ip.get("en_all", ""),
            "annotation": e.get("annotation", ""),
            "bp_status": e.get("bp_status", "") or ip.get("bp_status", ""),
            "normalization_action": e.get("normalization_action", ""),
            "ipa_word": ip.get("ipa_word_final", ""),
            "example_pt": ip.get("example_pt", ""),
            "example_en": ip.get("example_en", ""),
            "target_word_used": ip.get("target_word_used", ""),
            "ipa_example": ip.get("ipa_example_final", ""),
            "audio_word": mw.get("url", ""),
            "audio_example": mex.get("url", ""),
            "audio_word_md5": mw.get("md5", ""),
            "audio_example_md5": mex.get("md5", ""),
            "audio_en_example": men.get("url", ""),
            "audio_en_example_md5": men.get("md5", ""),
            "voice_id": voice_id,
            "voice_id_en": voice_id_en,
            "voice_gender": vgender_short,
            "family_root": family_root,
            "tags": tags_val,
            "source_line": e.get("source_line", ""),
            "source_line_number": e.get("source_line_number", ""),
            "notes": notes,
            "pt_display_safe": _pt_display_safe(pt_display_val, pos_val),
            "usage_hint": uh_hint,
            "usage_hint_priority": uh_priority,
            "risk_note": uh_risk,
        })

    # Write log
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"derive_final: joined {len(sids)} senses",
        f"  enriched (03) rows: {len(enriched)}",
        f"  ipa (05) rows:      {len(ipa)}",
        f"  manifest rows:      {len(manifest)}",
        f"  overrides:          {len(overrides)}",
        "",
        "Gap stats:",
    ]
    for k, v in gap_stats.items():
        log_lines.append(f"  {k}: {v}")
    log_text = "\n".join(log_lines)
    LOG_PATH.write_text(log_text + "\n", encoding="utf-8")
    print(log_text)

    if args.dry_run:
        print("\n--dry-run: not writing 06-final.tsv")
        return 0

    write_tsv(FINAL_PATH, rows, fieldnames=FINAL_FIELDS)
    print(f"\nWrote {FINAL_PATH} ({len(rows)} rows + 1 header)")

    # Append a per-row JSONL record for every sense that shipped a hint or
    # a risk_note. Carries the derive run's timestamp + the sidecar's
    # original priority + the export decision (kept / dropped) so the
    # audit trail joins back to the LLM classifier's call log at
    # audit/12_usage_hints.jsonl (which has prompt_hash / response_hash).
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    derive_ts = _dt.now(_tz.utc).isoformat()
    with USAGE_HINT_AUDIT.open("a", encoding="utf-8") as af:
        for r in rows:
            sid = r["sense_id"]
            shipped_hint = r.get("usage_hint", "")
            shipped_risk = r.get("risk_note", "")
            if not shipped_hint and not shipped_risk:
                continue
            sc = usage_hints.get(sid) or {}
            af.write(_json.dumps({
                "event": "derive_final_export",
                "sense_id": sid,
                "rank": r.get("rank", ""),
                "pos": r.get("pos", ""),
                "pt": r.get("pt", ""),
                "tags": r.get("tags", ""),
                "shipped_usage_hint": shipped_hint,
                "shipped_usage_hint_priority": r.get("usage_hint_priority", ""),
                "shipped_risk_note": shipped_risk,
                "sidecar_priority": sc.get("hint_priority", ""),
                "sidecar_hint": sc.get("usage_hint", ""),
                "sidecar_confidence": sc.get("confidence", ""),
                "model_id": sc.get("model_id", ""),
                "derive_ts": derive_ts,
            }, ensure_ascii=False) + "\n")
    print(f"Appended {gap_stats['usage_hint_total_shipped']} hint-export records "
          f"+ {gap_stats['risk_note_shipped']} risk_note records to {USAGE_HINT_AUDIT}")

    # Stage 13: append a per-row JSONL record for every sense whose family_root
    # shipped, capturing the sidecar provenance side-by-side with the derive
    # output. Joins to the LLM classifier log at audit/13_family_root.jsonl
    # (which has prompt_hash / response_hash) via sense_id.
    with FAMILY_ROOT_AUDIT.open("a", encoding="utf-8") as af:
        n_fr_records = 0
        for r in rows:
            sid = r["sense_id"]
            shipped = r.get("family_root", "")
            if not shipped:
                continue
            sc = family_roots.get(sid) or {}
            af.write(_json.dumps({
                "event": "derive_final_export",
                "sense_id": sid,
                "rank": r.get("rank", ""),
                "pos": r.get("pos", ""),
                "pt": r.get("pt", ""),
                "shipped_family_root": shipped,
                "sidecar_family_root": sc.get("family_root", ""),
                "sidecar_family_relation": sc.get("family_relation", ""),
                "sidecar_confidence": sc.get("confidence", ""),
                "sidecar_source": sc.get("source", ""),
                "sidecar_reason": sc.get("reason", ""),
                "model_id": sc.get("model_id", ""),
                "derive_ts": derive_ts,
            }, ensure_ascii=False) + "\n")
            n_fr_records += 1
    print(f"Appended {n_fr_records} family_root export records to "
          f"{FAMILY_ROOT_AUDIT}")

    # Stage 14: append a per-row JSONL record for every sense whose
    # `#topic-*` tag was merged into `tags`. Joins to the classifier log at
    # audit/14_topic_tags.jsonl (which has prompt_hash / response_hash per
    # LLM call) via sense_id. Deterministic rows ship too — their sidecar
    # `source` field reads "deterministic".
    with TOPIC_TAG_AUDIT.open("a", encoding="utf-8") as af:
        n_topic_records = 0
        for r in rows:
            sid = r["sense_id"]
            sc = topic_tags.get(sid) or {}
            shipped_topic = (sc.get("topic_primary") or "").strip()
            if not shipped_topic:
                continue
            af.write(_json.dumps({
                "event": "derive_final_export",
                "sense_id": sid,
                "rank": r.get("rank", ""),
                "pos": r.get("pos", ""),
                "pt": r.get("pt", ""),
                "shipped_topic": shipped_topic,
                "sidecar_confidence": sc.get("confidence", ""),
                "sidecar_source": sc.get("source", ""),
                "sidecar_reason": sc.get("reason", ""),
                "model_id": sc.get("model_id", ""),
                "derive_ts": derive_ts,
            }, ensure_ascii=False) + "\n")
            n_topic_records += 1
    print(f"Appended {n_topic_records} topic-tag export records to "
          f"{TOPIC_TAG_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
