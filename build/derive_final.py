"""Derive `data/06-final.tsv` — the project's final deliverable.

Joins:
  - data/03-enriched.tsv      (base: sense_id, source_pt, annotation, family_root,
                               source_line, source_line_number, tags, en_all)
  - data/05-ipa.tsv           (example_pt, example_en, target_word_used,
                               ipa_word_final, ipa_example_final + most other cols)
  - data/_audio_manifest.tsv  (audio URLs + md5 + voice_id, per (sense_id, clip_type))
  - config/voices.tsv         (voice_gender from voice_id)
  - data/_audio_asr_override.tsv (notes flag if sense's audio is ASR-untranscribable-but-bp-ok)

Produces 30 columns per the schema in docs/plan.md §"Final TSV schema".

Usage:
    .venv/bin/python build/derive_final.py
"""
from __future__ import annotations

import argparse
import csv
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
VOICES_PATH = CONFIG_DIR / "voices.tsv"

FINAL_PATH = DATA_DIR / "06-final.tsv"
LOG_PATH = AUDIT_DIR / "derive_final.log"

FINAL_FIELDS = [
    "sense_id", "rank", "source_pt", "pt", "pt_type", "gender",
    "pt_display", "pos", "sense_index", "en_primary", "en_all",
    "annotation", "bp_status", "normalization_action", "ipa_word",
    "example_pt", "example_en", "target_word_used", "ipa_example",
    "audio_word", "audio_example", "audio_word_md5", "audio_example_md5",
    "voice_id", "voice_gender", "family_root", "tags",
    "source_line", "source_line_number", "notes",
]


def _voice_gender_map() -> dict[str, str]:
    return {r["voice_id"]: r.get("gender", "")
            for r in read_tsv(VOICES_PATH)}


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

    rows: list[dict] = []
    gap_stats = {
        "missing_audio_word": 0,
        "missing_audio_example": 0,
        "missing_voice_id": 0,
        "missing_md5_word": 0,
        "missing_md5_example": 0,
        "voice_mismatch": 0,
        "asr_override_applied": 0,
        "family_root_blank": 0,
        "source_line_blank": 0,
    }

    for sid in sids:
        e = enriched_idx[sid]
        ip = ipa_idx[sid]
        mw = manifest_idx.get((sid, "word"), {})
        mex = manifest_idx.get((sid, "example"), {})

        if not mw.get("url"):
            gap_stats["missing_audio_word"] += 1
        if not mex.get("url"):
            gap_stats["missing_audio_example"] += 1
        if not mw.get("md5"):
            gap_stats["missing_md5_word"] += 1
        if not mex.get("md5"):
            gap_stats["missing_md5_example"] += 1

        voice_id = mw.get("voice_id") or mex.get("voice_id") or ""
        if not voice_id:
            gap_stats["missing_voice_id"] += 1
        elif mw.get("voice_id") and mex.get("voice_id") and mw["voice_id"] != mex["voice_id"]:
            gap_stats["voice_mismatch"] += 1
            voice_id = mw["voice_id"]  # prefer word
        vgender_word = voice_gender.get(voice_id, "")
        vgender_short = "m" if vgender_word == "male" else ("f" if vgender_word == "female" else "")

        family_root = e.get("family_root", "")
        if not family_root:
            gap_stats["family_root_blank"] += 1
        if not e.get("source_line"):
            gap_stats["source_line_blank"] += 1

        notes_parts: list[str] = []
        if sid in override_sids:
            notes_parts.append("audio_asr_status=untranscribable_audio_verified_bp")
            gap_stats["asr_override_applied"] += 1
        notes = " | ".join(notes_parts)

        rows.append({
            "sense_id": sid,
            "rank": e.get("rank", "") or ip.get("rank", ""),
            "source_pt": e.get("source_pt", ""),
            "pt": e.get("pt", "") or ip.get("pt", ""),
            "pt_type": e.get("pt_type", "") or ip.get("pt_type", ""),
            "gender": e.get("gender", "") or ip.get("gender", ""),
            "pt_display": e.get("pt_display", "") or ip.get("pt_display", ""),
            "pos": e.get("pos", "") or ip.get("pos", ""),
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
            "voice_id": voice_id,
            "voice_gender": vgender_short,
            "family_root": family_root,
            "tags": e.get("tags", "") or ip.get("tags", ""),
            "source_line": e.get("source_line", ""),
            "source_line_number": e.get("source_line_number", ""),
            "notes": notes,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
