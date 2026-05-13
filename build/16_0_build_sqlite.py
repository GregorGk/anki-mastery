"""Stage 16 / Step 0 — Build `data/06-final.sqlite` from canonical TSVs.

The TSVs (`data/06-final.tsv`, `data/_audio_manifest.tsv`, `config/voices.tsv`)
remain the source of truth. This script wipes and rebuilds a SQLite mirror
that lets us run SQL queries (cross-tabs, FTS, JOINs) without writing ad-hoc
Python every time.

Idempotent — re-running drops the DB and recreates from scratch.

Tables:
  senses                 1 row per sense_id (40 cols mirroring 06-final.tsv)
  sense_tags             1 row per (sense_id, tag) — exploded from senses.tags
  sense_risk_flags       1 row per (sense_id, flag) — exploded from senses.risk_flags
                         ('none' is dropped — absence in this table means none)
  sense_en_glosses       1 row per (sense_id, gloss_index, gloss) — split en_all on '/'
  audio_clips            from data/_audio_manifest.tsv — 3 rows per sense
  voices                 from config/voices.tsv (post-status column)
  senses_fts             FTS5 over PT/EN text fields

Usage:
    .venv/bin/python build/16_0_build_sqlite.py
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"

DB_PATH      = DATA / "06-final.sqlite"
FINAL_PATH   = DATA / "06-final.tsv"
MANIFEST     = DATA / "_audio_manifest.tsv"
VOICES_PATH  = CONFIG / "voices.tsv"

SCHEMA = """
CREATE TABLE senses (
  sense_id              TEXT PRIMARY KEY,
  rank                  INTEGER,
  source_pt             TEXT,
  pt                    TEXT NOT NULL,
  pt_type               TEXT,
  gender                TEXT,
  pt_display            TEXT,
  pos                   TEXT,
  sense_index           INTEGER,
  en_primary            TEXT,
  en_all                TEXT,
  annotation            TEXT,
  bp_status             TEXT,
  normalization_action  TEXT,
  ipa_word              TEXT,
  example_pt            TEXT,
  example_en            TEXT,
  target_word_used      TEXT,
  ipa_example           TEXT,
  audio_word            TEXT,
  audio_example         TEXT,
  audio_word_md5        TEXT,
  audio_example_md5     TEXT,
  audio_en_example      TEXT,
  audio_en_example_md5  TEXT,
  voice_id              TEXT,
  voice_id_en           TEXT,
  voice_gender          TEXT,
  family_root           TEXT,
  tags                  TEXT,
  source_line           TEXT,
  source_line_number    INTEGER,
  notes                 TEXT,
  pt_display_safe       TEXT,
  usage_hint            TEXT,
  usage_hint_priority   TEXT,
  risk_note             TEXT,
  bp_validity           TEXT,
  register              TEXT,
  risk_flags            TEXT
);
CREATE INDEX idx_senses_pt          ON senses(pt);
CREATE INDEX idx_senses_pos         ON senses(pos);
CREATE INDEX idx_senses_bp_status   ON senses(bp_status);
CREATE INDEX idx_senses_bp_validity ON senses(bp_validity);
CREATE INDEX idx_senses_register    ON senses(register);
CREATE INDEX idx_senses_family_root ON senses(family_root);
CREATE INDEX idx_senses_rank        ON senses(rank);
CREATE INDEX idx_senses_voice_id    ON senses(voice_id);

CREATE TABLE sense_tags (
  sense_id TEXT NOT NULL REFERENCES senses(sense_id),
  tag      TEXT NOT NULL,
  PRIMARY KEY (sense_id, tag)
);
CREATE INDEX idx_sense_tags_tag ON sense_tags(tag);

CREATE TABLE sense_risk_flags (
  sense_id TEXT NOT NULL REFERENCES senses(sense_id),
  flag     TEXT NOT NULL,
  PRIMARY KEY (sense_id, flag)
);
CREATE INDEX idx_sense_risk_flags_flag ON sense_risk_flags(flag);

CREATE TABLE sense_en_glosses (
  sense_id    TEXT NOT NULL REFERENCES senses(sense_id),
  gloss_index INTEGER NOT NULL,
  gloss       TEXT NOT NULL,
  PRIMARY KEY (sense_id, gloss_index)
);
CREATE INDEX idx_sense_en_glosses_gloss ON sense_en_glosses(gloss);

CREATE TABLE audio_clips (
  sense_id                  TEXT NOT NULL REFERENCES senses(sense_id),
  clip_type                 TEXT NOT NULL,
  voice_gender              TEXT,
  tts_provider              TEXT,
  tts_model                 TEXT,
  voice_id                  TEXT,
  text_input                TEXT,
  text_hash                 TEXT,
  object_key                TEXT,
  url                       TEXT,
  version                   INTEGER,
  md5                       TEXT,
  asr_transcript            TEXT,
  asr_similarity            REAL,
  asr_decision              TEXT,
  applied_gain_db           REAL,
  final_lufs                REAL,
  final_tp                  REAL,
  loudness_within_tolerance INTEGER,
  tp_limited                INTEGER,
  generated_at              TEXT,
  status                    TEXT,
  notes                     TEXT,
  PRIMARY KEY (sense_id, clip_type)
);
CREATE INDEX idx_audio_clips_voice  ON audio_clips(voice_id);
CREATE INDEX idx_audio_clips_model  ON audio_clips(tts_model);
CREATE INDEX idx_audio_clips_status ON audio_clips(status);

CREATE TABLE voices (
  voice_id    TEXT PRIMARY KEY,
  gender      TEXT NOT NULL,
  pool_index  INTEGER,
  bp_name     TEXT,
  en_voice_id TEXT,
  en_name     TEXT,
  status      TEXT,
  notes       TEXT
);
CREATE INDEX idx_voices_status ON voices(status);

CREATE VIRTUAL TABLE senses_fts USING fts5(
  sense_id UNINDEXED,
  pt, pt_display, en_primary, en_all,
  example_pt, example_en,
  annotation, usage_hint, risk_note,
  tokenize = 'unicode61 remove_diacritics 1'
);

CREATE VIEW v_topic_counts AS
  SELECT tag, COUNT(*) AS n
  FROM sense_tags
  WHERE tag LIKE '#topic-%'
  GROUP BY tag
  ORDER BY n DESC;

CREATE VIEW v_senses_with_audio AS
  SELECT s.*,
         aw.url       AS audio_word_url,    aw.tts_model AS audio_word_model,
         ax.url       AS audio_ex_url,      ax.tts_model AS audio_ex_model,
         ae.url       AS audio_en_ex_url,   ae.tts_model AS audio_en_ex_model
  FROM senses s
  LEFT JOIN audio_clips aw ON aw.sense_id = s.sense_id AND aw.clip_type = 'word'
  LEFT JOIN audio_clips ax ON ax.sense_id = s.sense_id AND ax.clip_type = 'example'
  LEFT JOIN audio_clips ae ON ae.sense_id = s.sense_id AND ae.clip_type = 'en_ex';
"""


def _read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _coerce_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _coerce_real(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _coerce_bool_int(s: str) -> int | None:
    """'true'/'false' (manifest convention) → 1/0; '' → NULL."""
    s = (s or "").strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no"):
        return 0
    return None


def _load_senses(conn: sqlite3.Connection) -> int:
    cols, rows = _read_tsv(FINAL_PATH)
    int_cols = {"rank", "sense_index", "source_line_number"}
    placeholders = ",".join("?" * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT INTO senses ({colnames}) VALUES ({placeholders})"

    def _row_values(r: dict) -> list:
        out = []
        for c in cols:
            v = r.get(c, "")
            if c in int_cols:
                out.append(_coerce_int(v))
            else:
                out.append(v)
        return out

    conn.executemany(sql, [_row_values(r) for r in rows])
    return len(rows)


def _load_sense_tags(conn: sqlite3.Connection) -> int:
    _, rows = _read_tsv(FINAL_PATH)
    triples = [
        (r["sense_id"], t)
        for r in rows
        for t in (r.get("tags") or "").split()
        if t
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO sense_tags(sense_id, tag) VALUES (?, ?)",
        triples,
    )
    return len(triples)


def _load_sense_risk_flags(conn: sqlite3.Connection) -> int:
    _, rows = _read_tsv(FINAL_PATH)
    pairs = []
    for r in rows:
        for f in (r.get("risk_flags") or "").split("|"):
            f = f.strip()
            if not f or f == "none":
                continue
            pairs.append((r["sense_id"], f))
    conn.executemany(
        "INSERT OR IGNORE INTO sense_risk_flags(sense_id, flag) VALUES (?, ?)",
        pairs,
    )
    return len(pairs)


def _load_sense_en_glosses(conn: sqlite3.Connection) -> int:
    _, rows = _read_tsv(FINAL_PATH)
    triples = []
    for r in rows:
        en_all = r.get("en_all") or ""
        seen_indices = 0
        for gloss in en_all.split("/"):
            gloss = gloss.strip()
            if not gloss:
                continue
            triples.append((r["sense_id"], seen_indices, gloss))
            seen_indices += 1
    conn.executemany(
        "INSERT INTO sense_en_glosses(sense_id, gloss_index, gloss) VALUES (?, ?, ?)",
        triples,
    )
    return len(triples)


def _load_audio_clips(conn: sqlite3.Connection) -> int:
    cols, rows = _read_tsv(MANIFEST)
    int_cols = {"version"}
    real_cols = {"asr_similarity", "applied_gain_db", "final_lufs", "final_tp"}
    bool_cols = {"loudness_within_tolerance", "tp_limited"}

    placeholders = ",".join("?" * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT INTO audio_clips ({colnames}) VALUES ({placeholders})"

    def _row_values(r: dict) -> list:
        out = []
        for c in cols:
            v = r.get(c, "")
            if c in int_cols:
                out.append(_coerce_int(v))
            elif c in real_cols:
                out.append(_coerce_real(v))
            elif c in bool_cols:
                out.append(_coerce_bool_int(v))
            else:
                out.append(v)
        return out

    conn.executemany(sql, [_row_values(r) for r in rows])
    return len(rows)


def _load_voices(conn: sqlite3.Connection) -> int:
    cols, rows = _read_tsv(VOICES_PATH)
    expected = ["voice_id", "gender", "pool_index", "bp_name",
                "en_voice_id", "en_name", "status", "notes"]
    # Tolerate column-order drift but fail loudly if a column is missing.
    missing = [c for c in expected if c not in cols]
    if missing:
        raise RuntimeError(f"voices.tsv missing columns: {missing}")
    sql = ("INSERT INTO voices "
           "(voice_id, gender, pool_index, bp_name, en_voice_id, en_name, status, notes) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
    conn.executemany(sql, [
        (r["voice_id"], r["gender"], _coerce_int(r["pool_index"]),
         r["bp_name"], r["en_voice_id"], r["en_name"],
         r["status"] or "active", r.get("notes", ""))
        for r in rows
    ])
    return len(rows)


def _populate_fts(conn: sqlite3.Connection) -> int:
    conn.execute("""
        INSERT INTO senses_fts(sense_id, pt, pt_display, en_primary, en_all,
                               example_pt, example_en, annotation,
                               usage_hint, risk_note)
        SELECT sense_id, pt, pt_display, en_primary, en_all,
               example_pt, example_en, annotation, usage_hint, risk_note
        FROM senses
    """)
    return conn.execute("SELECT COUNT(*) FROM senses_fts").fetchone()[0]


def main() -> int:
    if DB_PATH.exists():
        print(f"  wiping existing {DB_PATH}")
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")  # we INSERT in dep order; FKs informational
    conn.executescript(SCHEMA)

    n_senses    = _load_senses(conn)
    n_tags      = _load_sense_tags(conn)
    n_flags     = _load_sense_risk_flags(conn)
    n_en_glosses = _load_sense_en_glosses(conn)
    n_audio     = _load_audio_clips(conn)
    n_voices    = _load_voices(conn)
    n_fts       = _populate_fts(conn)

    conn.execute("ANALYZE")  # let the query planner build stats once.
    conn.commit()
    conn.close()

    size_kb = DB_PATH.stat().st_size // 1024
    print(f"\n  wrote {DB_PATH} ({size_kb:,} KB)")
    print()
    print(f"  senses             {n_senses:>6,}")
    print(f"  sense_tags         {n_tags:>6,}")
    print(f"  sense_risk_flags   {n_flags:>6,}")
    print(f"  sense_en_glosses   {n_en_glosses:>6,}")
    print(f"  audio_clips        {n_audio:>6,}")
    print(f"  voices             {n_voices:>6,}")
    print(f"  senses_fts         {n_fts:>6,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
