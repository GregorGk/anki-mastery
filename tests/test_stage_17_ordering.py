"""Stage 17 tests — deterministic ordering + clean Anki export.

A session fixture rebuilds the SQLite via `16_0_build_sqlite.py` THEN
`17_0_build_ordering.py` (16_0 wipes the whole DB file, so 17_0 must
run after it). A function-scoped `conn` fixture defensively rebuilds if
`test_stage_16_sqlite.py`'s fixture re-wiped the DB between tests — so
this suite is correct regardless of pytest file-collection order.

Hard checks fail the build; soft warnings only `warnings.warn`.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
FINAL_TSV = DATA_DIR / "06-final.tsv"
DB_PATH = DATA_DIR / "06-final.sqlite"
ORDERING_TSV = DATA_DIR / "_ordering.tsv"
ANKI_TSV = DATA_DIR / "07-anki-listening-general.tsv"
MEDIA_TSV = DATA_DIR / "07-anki-media-manifest.tsv"

BUILD_16 = REPO_ROOT / "build" / "16_0_build_sqlite.py"
BUILD_17_0 = REPO_ROOT / "build" / "17_0_build_ordering.py"
BUILD_17_1 = REPO_ROOT / "build" / "17_1_export_anki.py"

from build.lib import order_rules  # noqa: E402
from build.lib.topic_tag_rules import ALLOWED_TOPIC_TAGS, ALLOWED_TOPIC_TAG_SET  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 17_0 / 17_1 have numeric filenames — load via importlib for their constants/fns.
stage17_0 = _load_module("stage17_0_build_ordering", BUILD_17_0)
stage17_1 = _load_module("stage17_1_export_anki", BUILD_17_1)
ANKI_EXPORT_FIELDS = stage17_1.ANKI_EXPORT_FIELDS
MEDIA_FIELDS = stage17_1.MEDIA_FIELDS

# Snapshot 06-final.tsv at import time — assert it is never mutated.
_FINAL_TSV_SHA = hashlib.sha256(FINAL_TSV.read_bytes()).hexdigest()

# Columns that must NOT leak into the clean Anki export.
FORBIDDEN_EXPORT_COLS = {
    "strict_topic_order", "topic_index", "topic_bucket", "topic_subrank",
    "polysemy_count_global", "polysemy_count_in_topic",
    "polysemy_round_in_topic", "spaced_topic_order",
    "audio_word_md5", "audio_example_md5", "audio_en_example_md5",
    "voice_id", "voice_id_en", "voice_gender", "source_line",
    "source_line_number", "notes", "bp_status", "pt_display", "tags",
    "source_pt", "pt_type", "sense_index", "normalization_action",
    "target_word_used",
}

_TAG_RE = re.compile(r"^[a-z]+(::[a-z0-9-]+)+$")
_SOUND_RE = re.compile(r"^\[sound:[^\]]+\.mp3\]$")


# --- fixtures ---------------------------------------------------------------


def _run(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )


@pytest.fixture(scope="session")
def ordering_db() -> Path:
    """Rebuild the full SQLite for stage-17 tests: 16_0 (base) then 17_0
    (ordering). Self-contained so it's correct under any collection order."""
    for script in (BUILD_16, BUILD_17_0):
        r = _run(script)
        assert r.returncode == 0, (
            f"{script.name} failed (exit {r.returncode}):\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert DB_PATH.exists()
    return DB_PATH


@pytest.fixture()
def conn(ordering_db: Path):
    """Function-scoped connection. If test_stage_16's fixture re-wiped the
    DB (16_0 unlinks the file), rebuild 16_0+17_0 before yielding."""
    c = sqlite3.connect(ordering_db)
    c.row_factory = sqlite3.Row
    have = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='anki_ordering'"
    ).fetchone()
    if have is None:
        c.close()
        for script in (BUILD_16, BUILD_17_0):
            r = _run(script)
            assert r.returncode == 0, f"{script.name} rebuild failed:\n{r.stderr}"
        c = sqlite3.connect(ordering_db)
        c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture(scope="session")
def anki_export(ordering_db: Path) -> tuple[list[dict], list[dict]]:
    """Run 17_1; return (anki_rows, media_rows) parsed from the TSVs."""
    r = _run(BUILD_17_1)
    assert r.returncode == 0, (
        f"17_1_export_anki.py failed (exit {r.returncode}):\n{r.stderr}")
    return _read_anki_tsv(), _read_plain_tsv(MEDIA_TSV)


def _read_anki_tsv() -> list[dict]:
    """Read the clean export, skipping the leading #directive lines."""
    lines = ANKI_TSV.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    reader = csv.DictReader(data_lines, dialect="excel-tab")
    return list(reader)


def _read_plain_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, dialect="excel-tab"))


def _anki_header() -> list[str]:
    lines = ANKI_TSV.read_text(encoding="utf-8").splitlines()
    for ln in lines:
        if not ln.startswith("#"):
            return ln.split("\t")
    return []


# --- anki_ordering table integrity -----------------------------------------


def test_anki_ordering_row_count(conn):
    n_order = conn.execute("SELECT COUNT(*) FROM anki_ordering").fetchone()[0]
    n_senses = conn.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    assert n_order == n_senses == 5725


def test_anki_ordering_sense_id_set_matches_senses(conn):
    order_ids = {r[0] for r in conn.execute("SELECT sense_id FROM anki_ordering")}
    sense_ids = {r[0] for r in conn.execute("SELECT sense_id FROM senses")}
    assert order_ids == sense_ids


def test_topic_primary_in_allowlist(conn):
    bad = [r[0] for r in conn.execute(
        "SELECT DISTINCT topic_primary FROM anki_ordering")
        if r[0] not in ALLOWED_TOPIC_TAG_SET]
    assert not bad, f"topic_primary values not in allowlist: {bad}"


def test_topic_primary_matches_sense_tag(conn):
    """Ordering never reassigned a topic — incl. via manual overrides."""
    rows = conn.execute(
        "SELECT o.sense_id, o.topic_primary, t.tag "
        "FROM anki_ordering o "
        "JOIN sense_tags t ON t.sense_id = o.sense_id AND t.tag LIKE '#topic-%'"
    ).fetchall()
    assert len(rows) == 5725
    mism = [(sid, op, tag) for sid, op, tag in rows if op != tag]
    assert not mism, f"topic_primary != #topic tag for {len(mism)} rows: {mism[:5]}"


def test_topic_index_matches_order_rules(conn):
    for tp, ti in conn.execute(
            "SELECT DISTINCT topic_primary, topic_index FROM anki_ordering"):
        assert ti == order_rules.topic_index_for(tp), f"{tp}: index {ti}"


def test_topic_order_table(conn):
    rows = conn.execute(
        "SELECT topic_primary, topic_index FROM topic_order "
        "ORDER BY topic_index").fetchall()
    assert [r[0] for r in rows] == list(ALLOWED_TOPIC_TAGS)
    assert [r[1] for r in rows] == list(range(len(ALLOWED_TOPIC_TAGS)))
    assert len(rows) == 50


def test_strict_topic_order_is_permutation(conn):
    vals = sorted(r[0] for r in conn.execute(
        "SELECT strict_topic_order FROM anki_ordering"))
    assert vals == list(range(1, len(vals) + 1))


def test_spaced_topic_order_is_permutation(conn):
    vals = sorted(r[0] for r in conn.execute(
        "SELECT spaced_topic_order FROM anki_ordering"))
    assert vals == list(range(1, len(vals) + 1))


def _topic_blocks_contiguous(conn, col: str) -> list[str]:
    bad = []
    for (tp,) in conn.execute("SELECT DISTINCT topic_primary FROM anki_ordering"):
        vals = [r[0] for r in conn.execute(
            f"SELECT {col} FROM anki_ordering WHERE topic_primary = ?", (tp,))]
        if max(vals) - min(vals) + 1 != len(vals):
            bad.append(tp)
    return bad


def test_topic_blocks_contiguous_strict(conn):
    assert not _topic_blocks_contiguous(conn, "strict_topic_order")


def test_topic_blocks_contiguous_spaced(conn):
    """Polysemy spacing reorders only WITHIN a topic — blocks never interleave."""
    assert not _topic_blocks_contiguous(conn, "spaced_topic_order")


def test_polysemy_round_contiguous_per_group(conn):
    rows = conn.execute(
        "SELECT o.topic_primary, s.pt, o.polysemy_round_in_topic, "
        "o.polysemy_count_in_topic "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id"
    ).fetchall()
    groups: dict[tuple, list] = {}
    counts: dict[tuple, int] = {}
    for tp, pt, rnd, cnt in rows:
        groups.setdefault((tp, pt), []).append(rnd)
        counts[(tp, pt)] = cnt
    for key, rounds in groups.items():
        assert sorted(rounds) == list(range(counts[key])), (
            f"{key}: rounds {sorted(rounds)} != 0..{counts[key]-1}")


def test_polysemy_count_global_matches_raw(conn):
    raw = {r[0]: r[1] for r in conn.execute(
        "SELECT pt, COUNT(*) FROM senses GROUP BY pt")}
    rows = conn.execute(
        "SELECT s.pt, o.polysemy_count_global "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id"
    ).fetchall()
    for pt, cnt in rows:
        assert cnt == raw[pt], f"{pt}: polysemy_count_global {cnt} != {raw[pt]}"


def test_polysemy_count_in_topic_matches_raw(conn):
    raw = {(r[0], r[1]): r[2] for r in conn.execute(
        "SELECT t.tag, s.pt, COUNT(*) "
        "FROM senses s JOIN sense_tags t ON t.sense_id = s.sense_id "
        "  AND t.tag LIKE '#topic-%' "
        "GROUP BY t.tag, s.pt")}
    rows = conn.execute(
        "SELECT o.topic_primary, s.pt, o.polysemy_count_in_topic "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id"
    ).fetchall()
    for tp, pt, cnt in rows:
        assert cnt == raw[(tp, pt)], f"({tp},{pt}): {cnt} != {raw[(tp,pt)]}"


def test_spaced_order_consistent_per_topic(conn):
    """Within each topic, spaced order == re-sort by
    (polysemy_round_in_topic, topic_bucket, topic_subrank, sense_id)."""
    for (tp,) in conn.execute("SELECT DISTINCT topic_primary FROM anki_ordering"):
        rows = conn.execute(
            "SELECT sense_id, polysemy_round_in_topic, topic_bucket, "
            "topic_subrank, spaced_topic_order "
            "FROM anki_ordering WHERE topic_primary = ?", (tp,)).fetchall()
        by_spaced = [r[0] for r in sorted(rows, key=lambda r: r[4])]
        by_tuple = [r[0] for r in sorted(rows, key=lambda r: (r[1], r[2], r[3], r[0]))]
        assert by_spaced == by_tuple, f"{tp}: spaced order inconsistent"


def test_strict_order_consistent_per_topic(conn):
    """Within each topic, strict order == re-sort by
    (topic_bucket, topic_subrank, sense_id)."""
    for (tp,) in conn.execute("SELECT DISTINCT topic_primary FROM anki_ordering"):
        rows = conn.execute(
            "SELECT sense_id, topic_bucket, topic_subrank, strict_topic_order "
            "FROM anki_ordering WHERE topic_primary = ?", (tp,)).fetchall()
        by_strict = [r[0] for r in sorted(rows, key=lambda r: r[3])]
        by_tuple = [r[0] for r in sorted(rows, key=lambda r: (r[1], r[2], r[0]))]
        assert by_strict == by_tuple, f"{tp}: strict order inconsistent"


# --- 06-final.tsv immutability + idempotency -------------------------------


def test_final_tsv_untouched(ordering_db, anki_export):
    """Stage 17 must NEVER mutate the canonical master file."""
    now = hashlib.sha256(FINAL_TSV.read_bytes()).hexdigest()
    assert now == _FINAL_TSV_SHA, "data/06-final.tsv was modified by Stage 17"


def test_17_0_idempotent(ordering_db, conn):
    """Re-running 17_0 (without 16_0) yields byte-identical tables + sidecar."""
    snap1 = [tuple(r) for r in conn.execute(
        "SELECT * FROM anki_ordering ORDER BY sense_id")]
    tsv1 = ORDERING_TSV.read_bytes()
    r = _run(BUILD_17_0)
    assert r.returncode == 0, r.stderr
    c2 = sqlite3.connect(DB_PATH)
    snap2 = [tuple(r) for r in c2.execute(
        "SELECT * FROM anki_ordering ORDER BY sense_id")]
    c2.close()
    assert snap1 == snap2, "17_0 not idempotent: anki_ordering differs"
    assert ORDERING_TSV.read_bytes() == tsv1, "17_0 not idempotent: _ordering.tsv differs"


def test_ordering_tsv_sidecar(conn):
    assert ORDERING_TSV.exists()
    rows = _read_plain_tsv(ORDERING_TSV)
    assert len(rows) == 5725
    assert list(rows[0].keys()) == stage17_0.ANKI_ORDERING_COLS
    # parity with the table
    db_rows = {
        r[0]: r for r in conn.execute(
            f"SELECT {', '.join(stage17_0.ANKI_ORDERING_COLS)} FROM anki_ordering")
    }
    for tr in rows:
        sid = tr["sense_id"]
        db = db_rows[sid]
        assert str(db[5]) == tr["strict_topic_order"]
        assert str(db[9]) == tr["spaced_topic_order"]


# --- _manual_ordering.tsv strict validation --------------------------------


def test_manual_ordering_bad_header_rejected(tmp_path):
    bad = tmp_path / "_manual_ordering.tsv"
    bad.write_text(
        "sense_id\ttopic_bucket\ttopic_subrank\tforce_after_sense_id\t"
        "topic_primary\treason\n",
        encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        stage17_0._load_manual_overrides(bad)
    assert "header must be exactly" in str(exc.value)


def test_manual_ordering_unknown_sense_id_rejected():
    """A manual row naming an unknown sense_id is a hard failure."""
    senses = [
        {"sense_id": "0001.00.01", "pt": "o", "pos": "art",
         "topic_primary": "#topic-grammar"},
    ]
    overrides = {"9999.99.99": {
        "topic_bucket": 0, "topic_subrank": 1.0,
        "force_after_sense_id": "", "reason": "x", "_row_index": 0}}
    with pytest.raises(SystemExit) as exc:
        stage17_0._compute_stage_rows(senses, overrides)
    assert "not found in" in str(exc.value)


def test_manual_ordering_force_after_cross_topic_rejected():
    senses = [
        {"sense_id": "0001.00.01", "pt": "o", "pos": "art",
         "topic_primary": "#topic-grammar"},
        {"sense_id": "0050.00.01", "pt": "dia", "pos": "noun",
         "topic_primary": "#topic-time-calendar"},
    ]
    overrides = {"0001.00.01": {
        "topic_bucket": None, "topic_subrank": None,
        "force_after_sense_id": "0050.00.01", "reason": "x", "_row_index": 0}}
    with pytest.raises(SystemExit) as exc:
        stage17_0._compute_stage_rows(senses, overrides)
    assert "across topic blocks" in str(exc.value)


def test_manual_ordering_force_after_missing_target_rejected():
    senses = [
        {"sense_id": "0001.00.01", "pt": "o", "pos": "art",
         "topic_primary": "#topic-grammar"},
    ]
    overrides = {"0001.00.01": {
        "topic_bucket": None, "topic_subrank": None,
        "force_after_sense_id": "9999.99.99", "reason": "x", "_row_index": 0}}
    with pytest.raises(SystemExit) as exc:
        stage17_0._compute_stage_rows(senses, overrides)
    assert "not found in senses" in str(exc.value)


# --- clean Anki export -----------------------------------------------------


def test_anki_export_header_is_canonical(anki_export):
    assert _anki_header() == ANKI_EXPORT_FIELDS
    assert len(ANKI_EXPORT_FIELDS) == 25


def test_anki_export_no_forbidden_columns(anki_export):
    header = set(_anki_header())
    leaked = header & FORBIDDEN_EXPORT_COLS
    assert not leaked, f"clean export leaked technical columns: {leaked}"


def test_anki_export_row_count_and_order(anki_export):
    anki_rows, _ = anki_export
    assert len(anki_rows) == 5725
    orders = [int(r["anki_order"]) for r in anki_rows]
    assert orders == list(range(1, 5726)), "anki_order not ascending 1..5725"


def test_anki_export_no_raw_urls(anki_export):
    anki_rows, _ = anki_export
    for r in anki_rows:
        for col, val in r.items():
            assert "http" not in (val or "").lower(), (
                f"{r['sense_id']}: column {col} contains a raw URL")


def test_anki_export_audio_fields_are_sound_tags(anki_export):
    anki_rows, _ = anki_export
    for r in anki_rows:
        for col in ("audio_word", "audio_example", "audio_en_example"):
            assert _SOUND_RE.match(r[col]), (
                f"{r['sense_id']}: {col}={r[col]!r} not a [sound:....mp3] tag")


def test_anki_export_anki_tags_well_formed(anki_export):
    anki_rows, _ = anki_export
    for r in anki_rows:
        for tag in (r["anki_tags"] or "").split():
            assert _TAG_RE.match(tag), (
                f"{r['sense_id']}: malformed anki tag {tag!r}")


def test_anki_export_directives_present():
    head = ANKI_TSV.read_text(encoding="utf-8").splitlines()[:2]
    assert head == ["#separator:tab", "#html:true"]


# --- media manifest --------------------------------------------------------


def test_media_manifest_row_count(anki_export, conn):
    _, media_rows = anki_export
    n_clips = conn.execute("SELECT COUNT(*) FROM audio_clips").fetchone()[0]
    assert len(media_rows) == n_clips == 17175
    assert list(media_rows[0].keys()) == MEDIA_FIELDS


def test_media_manifest_filename_strips_prefix(anki_export):
    _, media_rows = anki_export
    for r in media_rows:
        assert r["media_filename"] == r["object_key"].split("audio/", 1)[-1]
        assert "/" not in r["media_filename"]


def test_media_manifest_bidirectional_closure(anki_export):
    """Every [sound:...] ref resolves to a media file and vice versa."""
    anki_rows, media_rows = anki_export
    sound_basenames = set()
    for r in anki_rows:
        for col in ("audio_word", "audio_example", "audio_en_example"):
            sound_basenames.add(r[col][len("[sound:"):-1])
    media_filenames = {r["media_filename"] for r in media_rows}
    assert sound_basenames - media_filenames == set(), "dangling [sound:] refs"
    assert media_filenames - sound_basenames == set(), "orphan media files"


# --- order_rules unit tests ------------------------------------------------


def test_bucket_for_non_custom_topic():
    b, sr, reason = order_rules.bucket_for(
        "#topic-animals", "gato", "noun", "1234.00.01")
    assert b == 999 and reason == "topic_default"
    assert sr == float(order_rules.subrank_from_sense_id("1234.00.01"))


def test_bucket_for_custom_lemma_hit():
    b, sr, reason = order_rules.bucket_for(
        "#topic-numbers", "dois", "num", "0041.00.01")
    assert b == 0 and reason.startswith("bucket:") and sr == 1.0


def test_bucket_for_grammar_pos_rule():
    b, sr, reason = order_rules.bucket_for(
        "#topic-grammar", "de", "prep", "0002.00.01")
    assert b == 3 and reason == "grammar_pos:prep"


def test_bucket_for_custom_unmatched():
    b, sr, reason = order_rules.bucket_for(
        "#topic-numbers", "zzz-not-a-number", "noun", "9999.00.01")
    assert b == 999 and reason == "custom_unmatched"


def test_anki_tags_for_shape():
    tags = order_rules.anki_tags_for(
        topic_primary="#topic-money", pos="noun", risk_flags="false_friend",
        bp_validity="ep_leaning", tags="#top2000 #noun #topic-money")
    toks = tags.split()
    assert "topic::money" in toks
    assert "pos::noun" in toks
    assert "bp::ep-leaning" in toks
    assert "risk::false-friend" in toks
    assert "meta::top2000" in toks
    assert all(_TAG_RE.match(t) for t in toks)


def test_anki_tags_for_skips_standard_bp_and_none_risk():
    tags = order_rules.anki_tags_for(
        topic_primary="#topic-body", pos="noun", risk_flags="none",
        bp_validity="standard", tags="")
    toks = tags.split()
    assert "topic::body" in toks and "pos::noun" in toks
    assert not any(t.startswith("bp::") for t in toks)
    assert not any(t.startswith("risk::") for t in toks)


def test_order_rules_internal_consistency():
    """Unique bucket_ids per topic; no duplicate lemma within a topic."""
    for topic, buckets in order_rules.CUSTOM_TOPIC_BUCKETS.items():
        bids = [b for b, _, _ in buckets]
        assert len(bids) == len(set(bids)), f"{topic}: duplicate bucket_id"
        assert all(b < 999 for b in bids), f"{topic}: bucket_id >= 999"
        seen: dict[str, int] = {}
        for bid, _label, lemmas in buckets:
            for lemma in lemmas:
                assert lemma not in seen, (
                    f"{topic}: lemma {lemma!r} in buckets {seen[lemma]} and {bid}")
                seen[lemma] = bid


def test_topic_index_covers_all_50():
    assert len(order_rules.TOPIC_INDEX) == 50
    assert set(order_rules.TOPIC_INDEX) == ALLOWED_TOPIC_TAG_SET


# --- soft warnings (never fail) --------------------------------------------


def test_soft_warnings_bucket_coverage(conn):
    """Print bucket-999 coverage per custom topic — warn, never fail."""
    for topic in order_rules.CUSTOM_TOPICS:
        total = conn.execute(
            "SELECT COUNT(*) FROM anki_ordering WHERE topic_primary = ?",
            (topic,)).fetchone()[0]
        b999 = conn.execute(
            "SELECT COUNT(*) FROM anki_ordering "
            "WHERE topic_primary = ? AND topic_bucket = 999", (topic,)).fetchone()[0]
        pct = 100 * (total - b999) / total if total else 0.0
        if pct < 50:
            warnings.warn(f"{topic}: only {pct:.0f}% bucketed "
                          f"({b999}/{total} in bucket-999)")
        if b999:
            warnings.warn(f"{topic}: {b999} sense(s) in fallback bucket-999")


def test_soft_warnings_level_b_collisions(conn):
    """Count cross-topic same-lemma collisions <30 spaced rows apart."""
    rows = conn.execute(
        "SELECT s.pt, o.topic_primary, o.spaced_topic_order "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "WHERE s.pt IN (SELECT pt FROM senses GROUP BY pt HAVING COUNT(*) > 1) "
        "ORDER BY s.pt, o.spaced_topic_order").fetchall()
    by_pt: dict[str, list] = {}
    for pt, tp, sp in rows:
        by_pt.setdefault(pt, []).append((tp, sp))
    n = 0
    for pt, members in by_pt.items():
        for i in range(1, len(members)):
            (t0, s0), (t1, s1) = members[i - 1], members[i]
            if t0 != t1 and (s1 - s0) < 30:
                n += 1
    if n:
        warnings.warn(f"Level B: {n} cross-topic same-lemma collision(s) "
                      f"<30 rows apart (report-only)")
