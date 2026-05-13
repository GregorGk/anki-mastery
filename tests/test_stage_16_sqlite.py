"""Stage 16 tests — SQLite mirror of 06-final.tsv.

Asserts that EVERY cell of `data/06-final.tsv` is faithfully persisted in
`data/06-final.sqlite`, plus that the derived multi-valued tables
(`sense_tags`, `sense_risk_flags`, `sense_en_glosses`) and the joined
`audio_clips` / `voices` tables match their canonical TSVs.

A session-scoped fixture rebuilds the SQLite once from the current TSVs
via `build/16_0_build_sqlite.py`, so tests are reproducible regardless
of the DB's prior state.
"""
from __future__ import annotations

import csv
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

FINAL_TSV   = DATA_DIR / "06-final.tsv"
MANIFEST    = DATA_DIR / "_audio_manifest.tsv"
VOICES_TSV  = CONFIG_DIR / "voices.tsv"
BUILD_SCRIPT = REPO_ROOT / "build" / "16_0_build_sqlite.py"
DB_PATH     = DATA_DIR / "06-final.sqlite"

# Columns coerced to non-TEXT types in the senses table.
SENSES_INT_COLS = {"rank", "sense_index", "source_line_number"}

# Audio clip manifest coercions (mirror build/16_0_build_sqlite.py).
AUDIO_INT_COLS  = {"version"}
AUDIO_REAL_COLS = {"asr_similarity", "applied_gain_db", "final_lufs", "final_tp"}
AUDIO_BOOL_COLS = {"loudness_within_tolerance", "tp_limited"}


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="session")
def db_path() -> Path:
    """Rebuild the SQLite once for the whole test session from current TSVs.

    We exercise the real build script (no mocking) so any drift between
    the build code and the schema surfaces immediately.
    """
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"build/16_0_build_sqlite.py failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert DB_PATH.exists(), f"build script did not write {DB_PATH}"
    return DB_PATH


@pytest.fixture(scope="session")
def conn(db_path: Path):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def _read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, dialect="excel-tab"))


@pytest.fixture(scope="session")
def tsv_rows() -> list[dict]:
    return _read_tsv(FINAL_TSV)


@pytest.fixture(scope="session")
def manifest_rows() -> list[dict]:
    return _read_tsv(MANIFEST)


@pytest.fixture(scope="session")
def voices_rows() -> list[dict]:
    return _read_tsv(VOICES_TSV)


# --- senses table ------------------------------------------------------------


def test_senses_row_count(conn, tsv_rows):
    n = conn.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    assert n == len(tsv_rows), (
        f"senses table has {n} rows but 06-final.tsv has {len(tsv_rows)}"
    )


def test_senses_no_duplicate_sense_ids(conn):
    total = conn.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    distinct = conn.execute("SELECT COUNT(DISTINCT sense_id) FROM senses").fetchone()[0]
    assert total == distinct


def test_senses_no_null_pt(conn):
    n = conn.execute("SELECT COUNT(*) FROM senses WHERE pt IS NULL OR pt = ''").fetchone()[0]
    assert n == 0


def test_senses_sense_id_set_parity(conn, tsv_rows):
    tsv_ids = {r["sense_id"] for r in tsv_rows}
    db_ids = {r[0] for r in conn.execute("SELECT sense_id FROM senses")}
    missing_in_db = tsv_ids - db_ids
    extra_in_db = db_ids - tsv_ids
    assert not missing_in_db, f"missing in DB: {sorted(missing_in_db)[:5]}"
    assert not extra_in_db, f"extra in DB: {sorted(extra_in_db)[:5]}"


def test_senses_full_cell_parity(conn, tsv_rows):
    """Every cell value in 06-final.tsv equals the SQLite cell, modulo
    integer coercion for rank / sense_index / source_line_number and the
    convention that empty strings in the TSV map to NULL in INTEGER columns."""
    tsv_cols = list(tsv_rows[0].keys())
    db_rows = {r["sense_id"]: dict(r) for r in conn.execute("SELECT * FROM senses")}
    mismatches: list[str] = []
    for tsv_row in tsv_rows:
        sid = tsv_row["sense_id"]
        db_row = db_rows.get(sid)
        if db_row is None:
            mismatches.append(f"{sid}: missing in DB")
            continue
        for c in tsv_cols:
            tsv_val = tsv_row.get(c, "")
            db_val = db_row.get(c)
            if c in SENSES_INT_COLS:
                expected = None if (not tsv_val or not tsv_val.strip()) else int(tsv_val)
                if db_val != expected:
                    mismatches.append(f"{sid}.{c}: tsv={tsv_val!r} db={db_val!r}")
            else:
                # TEXT col: empty TSV ('') round-trips as '' or None (sqlite quirk).
                db_str = "" if db_val is None else db_val
                if db_str != tsv_val:
                    mismatches.append(f"{sid}.{c}: tsv={tsv_val!r} db={db_val!r}")
            if len(mismatches) > 10:
                break
        if len(mismatches) > 10:
            break
    assert not mismatches, "first mismatches:\n  " + "\n  ".join(mismatches)


# --- sense_tags (derived from senses.tags) -----------------------------------


def test_sense_tags_parity(conn, tsv_rows):
    expected = set()
    for r in tsv_rows:
        for t in (r.get("tags") or "").split():
            if t:
                expected.add((r["sense_id"], t))
    actual = {(s, t) for s, t in conn.execute("SELECT sense_id, tag FROM sense_tags")}
    assert expected == actual, (
        f"sense_tags drift: tsv-only={list(expected-actual)[:5]} "
        f"db-only={list(actual-expected)[:5]}"
    )


def test_sense_tags_no_orphans(conn):
    n = conn.execute("""
        SELECT COUNT(*) FROM sense_tags
        WHERE sense_id NOT IN (SELECT sense_id FROM senses)
    """).fetchone()[0]
    assert n == 0


# --- sense_risk_flags (derived; 'none' dropped) ------------------------------


def test_sense_risk_flags_parity(conn, tsv_rows):
    expected: set[tuple[str, str]] = set()
    for r in tsv_rows:
        for f in (r.get("risk_flags") or "").split("|"):
            f = f.strip()
            if not f or f == "none":
                continue
            expected.add((r["sense_id"], f))
    actual = {(s, f) for s, f in conn.execute(
        "SELECT sense_id, flag FROM sense_risk_flags"
    )}
    assert expected == actual


def test_sense_risk_flags_none_dropped(conn):
    """The literal 'none' default must never appear as an exploded flag."""
    n = conn.execute(
        "SELECT COUNT(*) FROM sense_risk_flags WHERE flag = 'none'"
    ).fetchone()[0]
    assert n == 0


def test_sense_risk_flags_no_orphans(conn):
    n = conn.execute("""
        SELECT COUNT(*) FROM sense_risk_flags
        WHERE sense_id NOT IN (SELECT sense_id FROM senses)
    """).fetchone()[0]
    assert n == 0


# --- sense_en_glosses (derived from senses.en_all on '/') --------------------


def test_sense_en_glosses_parity(conn, tsv_rows):
    expected: set[tuple[str, int, str]] = set()
    for r in tsv_rows:
        idx = 0
        for g in (r.get("en_all") or "").split("/"):
            g = g.strip()
            if not g:
                continue
            expected.add((r["sense_id"], idx, g))
            idx += 1
    actual = {(s, i, g) for s, i, g in conn.execute(
        "SELECT sense_id, gloss_index, gloss FROM sense_en_glosses"
    )}
    assert expected == actual


def test_sense_en_glosses_no_orphans(conn):
    n = conn.execute("""
        SELECT COUNT(*) FROM sense_en_glosses
        WHERE sense_id NOT IN (SELECT sense_id FROM senses)
    """).fetchone()[0]
    assert n == 0


# --- audio_clips (from _audio_manifest.tsv) ---------------------------------


def test_audio_clips_row_count(conn, manifest_rows):
    n = conn.execute("SELECT COUNT(*) FROM audio_clips").fetchone()[0]
    assert n == len(manifest_rows)


def test_audio_clips_key_set_parity(conn, manifest_rows):
    expected = {(r["sense_id"], r["clip_type"]) for r in manifest_rows}
    actual = {(s, c) for s, c in conn.execute(
        "SELECT sense_id, clip_type FROM audio_clips"
    )}
    assert expected == actual


def test_audio_clips_full_cell_parity(conn, manifest_rows):
    cols = list(manifest_rows[0].keys())
    db_rows = {(r["sense_id"], r["clip_type"]): dict(r)
               for r in conn.execute("SELECT * FROM audio_clips")}
    mismatches: list[str] = []
    for mr in manifest_rows:
        key = (mr["sense_id"], mr["clip_type"])
        db_row = db_rows.get(key)
        if db_row is None:
            mismatches.append(f"{key}: missing in DB")
            continue
        for c in cols:
            tsv_val = mr.get(c, "")
            db_val = db_row.get(c)
            if c in AUDIO_INT_COLS:
                exp = None if (not tsv_val or not tsv_val.strip()) else int(tsv_val)
                if db_val != exp:
                    mismatches.append(f"{key[0]}/{key[1]}.{c}: tsv={tsv_val!r} db={db_val!r}")
            elif c in AUDIO_REAL_COLS:
                if not tsv_val or not tsv_val.strip():
                    if db_val is not None:
                        mismatches.append(f"{key[0]}/{key[1]}.{c}: expected NULL, got {db_val!r}")
                else:
                    exp_f = float(tsv_val)
                    if db_val is None or abs(db_val - exp_f) > 1e-9:
                        mismatches.append(f"{key[0]}/{key[1]}.{c}: tsv={tsv_val!r} db={db_val!r}")
            elif c in AUDIO_BOOL_COLS:
                lo = (tsv_val or "").strip().lower()
                if lo in ("true", "1", "yes"):
                    exp = 1
                elif lo in ("false", "0", "no"):
                    exp = 0
                else:
                    exp = None
                if db_val != exp:
                    mismatches.append(f"{key[0]}/{key[1]}.{c}: tsv={tsv_val!r} db={db_val!r}")
            else:
                db_str = "" if db_val is None else db_val
                if db_str != tsv_val:
                    mismatches.append(f"{key[0]}/{key[1]}.{c}: tsv={tsv_val!r} db={db_val!r}")
            if len(mismatches) > 10:
                break
        if len(mismatches) > 10:
            break
    assert not mismatches, "first mismatches:\n  " + "\n  ".join(mismatches)


def test_audio_clips_no_orphans(conn):
    n = conn.execute("""
        SELECT COUNT(*) FROM audio_clips
        WHERE sense_id NOT IN (SELECT sense_id FROM senses)
    """).fetchone()[0]
    assert n == 0


# --- voices ------------------------------------------------------------------


def test_voices_row_count(conn, voices_rows):
    n = conn.execute("SELECT COUNT(*) FROM voices").fetchone()[0]
    assert n == len(voices_rows)


def test_voices_full_cell_parity(conn, voices_rows):
    db = {r["voice_id"]: dict(r) for r in conn.execute("SELECT * FROM voices")}
    mismatches: list[str] = []
    for v in voices_rows:
        vid = v["voice_id"]
        db_row = db.get(vid)
        if db_row is None:
            mismatches.append(f"{vid}: missing in DB")
            continue
        # pool_index is INTEGER
        expected_pool = int(v["pool_index"])
        if db_row["pool_index"] != expected_pool:
            mismatches.append(f"{vid}.pool_index: tsv={v['pool_index']!r} db={db_row['pool_index']!r}")
        for c in ("gender", "bp_name", "en_voice_id", "en_name", "status", "notes"):
            tsv_val = v.get(c, "") or ""
            # status defaults to 'active' if blank in TSV
            if c == "status" and not tsv_val:
                tsv_val = "active"
            db_val = db_row.get(c) or ""
            if db_val != tsv_val:
                mismatches.append(f"{vid}.{c}: tsv={tsv_val!r} db={db_val!r}")
    assert not mismatches, "voices mismatches:\n  " + "\n  ".join(mismatches)


# --- FTS5 senses_fts ---------------------------------------------------------


def test_fts_row_count(conn, tsv_rows):
    n = conn.execute("SELECT COUNT(*) FROM senses_fts").fetchone()[0]
    assert n == len(tsv_rows)


def test_fts_sense_id_set_parity(conn, tsv_rows):
    expected = {r["sense_id"] for r in tsv_rows}
    actual = {r[0] for r in conn.execute("SELECT sense_id FROM senses_fts")}
    assert expected == actual


def test_fts_searchable_pt(conn, tsv_rows):
    """A known PT word from 06-final.tsv must be findable via FTS."""
    sample = next(r for r in tsv_rows if r["pt"] == "casa")
    rows = list(conn.execute(
        "SELECT sense_id FROM senses_fts WHERE senses_fts MATCH 'casa'"
    ))
    sids = {r[0] for r in rows}
    assert sample["sense_id"] in sids


def test_fts_searchable_example(conn, tsv_rows):
    """A word appearing only in example_pt (not pt) is also indexed."""
    # 'mercado' is not a headword but appears in multiple examples.
    rows = list(conn.execute(
        "SELECT sense_id FROM senses_fts WHERE senses_fts MATCH 'mercado'"
    ))
    assert len(rows) >= 3


# --- views -------------------------------------------------------------------


def test_v_topic_counts_total_matches_topic_tags(conn):
    """v_topic_counts must aggregate all #topic-* tags in sense_tags."""
    expected = conn.execute(
        "SELECT COUNT(*) FROM sense_tags WHERE tag LIKE '#topic-%'"
    ).fetchone()[0]
    actual = conn.execute(
        "SELECT COALESCE(SUM(n), 0) FROM v_topic_counts"
    ).fetchone()[0]
    assert expected == actual


def test_v_senses_with_audio_row_count(conn, tsv_rows):
    """One row per sense_id, joining the three clip_types as LEFT JOINs."""
    n = conn.execute("SELECT COUNT(*) FROM v_senses_with_audio").fetchone()[0]
    assert n == len(tsv_rows)


# --- schema sanity -----------------------------------------------------------


def test_schema_expected_tables_exist(conn):
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }
    required = {
        "senses", "sense_tags", "sense_risk_flags", "sense_en_glosses",
        "audio_clips", "voices",
        "senses_fts",
        "v_topic_counts", "v_senses_with_audio",
    }
    missing = required - tables
    assert not missing, f"missing schema objects: {missing}"


def test_senses_column_count_matches_tsv(conn, tsv_rows):
    db_cols = [c[1] for c in conn.execute("PRAGMA table_info(senses)")]
    tsv_cols = list(tsv_rows[0].keys())
    assert db_cols == tsv_cols, (
        f"column drift between schema and 06-final.tsv:\n"
        f"  db:  {db_cols}\n"
        f"  tsv: {tsv_cols}"
    )
