"""Stage 17 / Step 1 — Clean Anki export + media manifest.

Reads (read-only):  data/06-final.sqlite  (senses, anki_ordering, audio_clips)
Writes:             data/07-anki-listening-general.tsv   (clean learner import)
                    data/07-anki-media-manifest.tsv      (audio provenance)

The clean Anki TSV holds ONLY learner/card/filter fields — see
ANKI_EXPORT_FIELDS. No debug ordering columns, no md5s, no provenance,
no raw R2 URLs. Audio is emitted as Anki `[sound:<basename>.mp3]` tags;
the raw URL + md5 per clip live in the media manifest instead.

Rows are ordered by `anki_ordering.spaced_topic_order` — the
topic-ordered, polysemy-spaced teaching sequence.

NEVER mutates data/06-final.tsv. Pipeline order is strict: run
build/16_0_build_sqlite.py then build/17_0_build_ordering.py first.

Usage:
    .venv/bin/python build/17_1_export_anki.py
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import order_rules  # noqa: E402
from build.lib.tsv import write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
DB_PATH = DATA / "06-final.sqlite"
ANKI_OUT = DATA / "07-anki-listening-general.tsv"
MEDIA_OUT = DATA / "07-anki-media-manifest.tsv"

# The single source of truth for the clean export header. Imported by
# tests/test_stage_17_ordering.py so the writer and the test never drift.
ANKI_EXPORT_FIELDS = [
    "anki_order",
    "sense_id",
    "topic_primary",
    "rank",
    "pt",
    "pt_display_safe",
    "pos",
    "gender",
    "en_primary",
    "en_all",
    "annotation",
    "example_pt",
    "example_en",
    "ipa_word",
    "ipa_example",
    "audio_word",
    "audio_example",
    "audio_en_example",
    "usage_hint",
    "risk_note",
    "bp_validity",
    "register",
    "risk_flags",
    "family_root",
    "anki_tags",
]

MEDIA_FIELDS = [
    "sense_id", "clip_type", "spaced_topic_order", "object_key",
    "media_filename", "url", "md5", "tts_model", "version", "status",
]

# Anki import directives prepended to the clean TSV.
ANKI_DIRECTIVES = ["#separator:tab", "#html:true"]


class ExportError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (17_1_export_anki): {msg}")


def _require_tables(conn: sqlite3.Connection) -> None:
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for needed in ("senses", "anki_ordering", "audio_clips"):
        if needed not in have:
            raise ExportError(
                f"table '{needed}' missing from {DB_PATH.name}. Run "
                f"build/16_0_build_sqlite.py then build/17_0_build_ordering.py.")


def _basename(object_key: str) -> str:
    """'audio/0001.00.01-word-eleven_v3-v1.mp3' -> '0001.00.01-word-eleven_v3-v1.mp3'.

    Strips the 'audio/' prefix off the STORED object_key — never
    reconstructs from clip_type (the 'example' clip uses an '-ex-'
    filename segment; the legacy 2801.00.01 holdout uses a 'flash' model
    segment). Prefix-strip handles both quirks for free.
    """
    return object_key.split("/", 1)[1] if "/" in object_key else object_key


def _sound_field(object_key: str | None) -> str:
    if not object_key:
        return ""
    return f"[sound:{_basename(object_key)}]"


def _build_anki_rows(conn: sqlite3.Connection) -> list[dict]:
    sql = """
    SELECT
      o.spaced_topic_order AS anki_order,
      s.sense_id, o.topic_primary, s.rank, s.pt, s.pt_display_safe, s.pos,
      s.gender, s.en_primary, s.en_all, s.annotation, s.example_pt,
      s.example_en, s.ipa_word, s.ipa_example,
      s.usage_hint, s.risk_note, s.bp_validity, s.register, s.risk_flags,
      s.family_root, s.tags,
      aw.object_key AS word_key,
      ax.object_key AS ex_key,
      ae.object_key AS en_ex_key
    FROM senses s
    JOIN anki_ordering o     ON o.sense_id = s.sense_id
    LEFT JOIN audio_clips aw ON aw.sense_id = s.sense_id AND aw.clip_type = 'word'
    LEFT JOIN audio_clips ax ON ax.sense_id = s.sense_id AND ax.clip_type = 'example'
    LEFT JOIN audio_clips ae ON ae.sense_id = s.sense_id AND ae.clip_type = 'en_ex'
    ORDER BY o.spaced_topic_order
    """
    rows: list[dict] = []
    for r in conn.execute(sql):
        d = dict(r)
        out = {
            "anki_order": d["anki_order"],
            "sense_id": d["sense_id"],
            "topic_primary": d["topic_primary"],
            "rank": d["rank"],
            "pt": d["pt"],
            "pt_display_safe": d["pt_display_safe"],
            "pos": d["pos"],
            "gender": d["gender"],
            "en_primary": d["en_primary"],
            "en_all": d["en_all"],
            "annotation": d["annotation"],
            "example_pt": d["example_pt"],
            "example_en": d["example_en"],
            "ipa_word": d["ipa_word"],
            "ipa_example": d["ipa_example"],
            "audio_word": _sound_field(d["word_key"]),
            "audio_example": _sound_field(d["ex_key"]),
            "audio_en_example": _sound_field(d["en_ex_key"]),
            "usage_hint": d["usage_hint"],
            "risk_note": d["risk_note"],
            "bp_validity": d["bp_validity"],
            "register": d["register"],
            "risk_flags": d["risk_flags"],
            "family_root": d["family_root"],
            "anki_tags": order_rules.anki_tags_for(
                topic_primary=d["topic_primary"], pos=d["pos"] or "",
                risk_flags=d["risk_flags"] or "",
                bp_validity=d["bp_validity"] or "", tags=d["tags"] or ""),
        }
        rows.append(out)
    return rows


def _build_media_rows(conn: sqlite3.Connection) -> list[dict]:
    sql = """
    SELECT a.sense_id, a.clip_type, o.spaced_topic_order, a.object_key,
           a.url, a.md5, a.tts_model, a.version, a.status
    FROM audio_clips a
    JOIN anki_ordering o ON o.sense_id = a.sense_id
    ORDER BY o.spaced_topic_order, a.clip_type
    """
    rows: list[dict] = []
    for r in conn.execute(sql):
        d = dict(r)
        d["media_filename"] = _basename(d["object_key"] or "")
        rows.append(d)
    return rows


def _assert_clean(anki_rows: list[dict], media_rows: list[dict],
                  n_senses: int, n_clips: int) -> None:
    # 1. row counts
    if len(anki_rows) != n_senses:
        raise ExportError(f"anki TSV has {len(anki_rows)} rows, expected "
                          f"{n_senses} (one per sense).")
    if len(media_rows) != n_clips:
        raise ExportError(f"media manifest has {len(media_rows)} rows, "
                          f"expected {n_clips} (one per audio clip).")
    # 2. anki_order is a contiguous 1..N permutation
    orders = sorted(int(r["anki_order"]) for r in anki_rows)
    if orders != list(range(1, len(orders) + 1)):
        raise ExportError("anki_order is not a contiguous 1..N permutation.")
    # 3. no empty sound fields, no raw URLs anywhere
    sound_basenames: set[str] = set()
    for r in anki_rows:
        for col in ("audio_word", "audio_example", "audio_en_example"):
            v = r[col]
            if not (v.startswith("[sound:") and v.endswith(".mp3]")):
                raise ExportError(
                    f"{r['sense_id']}: {col}={v!r} is not a [sound:....mp3] tag.")
            sound_basenames.add(v[len("[sound:"):-1])
        for col, v in r.items():
            if "http" in str(v).lower():
                raise ExportError(
                    f"{r['sense_id']}: column {col} contains a raw URL: {v!r}")
    # 4. bidirectional media closure
    media_filenames = {r["media_filename"] for r in media_rows}
    missing = sound_basenames - media_filenames
    if missing:
        raise ExportError(f"{len(missing)} [sound:...] refs have no media "
                          f"manifest row, e.g. {sorted(missing)[:3]}")
    orphan = media_filenames - sound_basenames
    if orphan:
        raise ExportError(f"{len(orphan)} media manifest files are unused by "
                          f"any [sound:...] ref, e.g. {sorted(orphan)[:3]}")


def _write_anki_tsv(rows: list[dict]) -> None:
    """Write the clean export: #directives, header row, then data rows."""
    ANKI_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ANKI_OUT.open("w", encoding="utf-8", newline="") as f:
        for directive in ANKI_DIRECTIVES:
            f.write(directive + "\n")
        w = csv.DictWriter(f, fieldnames=ANKI_EXPORT_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else str(r.get(k)))
                        for k in ANKI_EXPORT_FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    if not args.db.exists():
        raise ExportError(f"{args.db} does not exist.")

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _require_tables(conn)
        n_senses = conn.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
        n_clips = conn.execute("SELECT COUNT(*) FROM audio_clips").fetchone()[0]
        anki_rows = _build_anki_rows(conn)
        media_rows = _build_media_rows(conn)
    finally:
        conn.close()

    _assert_clean(anki_rows, media_rows, n_senses, n_clips)

    _write_anki_tsv(anki_rows)
    n_media = write_tsv(MEDIA_OUT, media_rows, fieldnames=MEDIA_FIELDS)

    first, last = anki_rows[0], anki_rows[-1]
    print(f"=== Stage 17.1 — clean Anki export ===")
    print(f"  {ANKI_OUT.name}:  {len(anki_rows):,} rows, "
          f"{len(ANKI_EXPORT_FIELDS)} fields")
    print(f"  {MEDIA_OUT.name}:  {n_media:,} rows")
    print(f"  first card:  #{first['anki_order']} {first['sense_id']} "
          f"{first['pt']!r} [{first['topic_primary']}]")
    print(f"  last card:   #{last['anki_order']} {last['sense_id']} "
          f"{last['pt']!r} [{last['topic_primary']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
