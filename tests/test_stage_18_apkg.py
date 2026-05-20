"""Stage 18 — programmatic .apkg build quality gates.

Two tiers so a fresh clone works AND offline CI degrades gracefully:

  * Media-free logic — calls build_notes() directly on the pilot slice
    (no media, no network). Always runs.
  * Packaging + media — a session fixture runs 18_0 --pilot THEN
    18_1 --pilot via subprocess. If 18_0 fails (e.g. no network) the
    media tests pytest.skip with a clear message rather than erroring.

Constants / helpers are imported from the lib + the 18_1 script (via
importlib, since the numeric filename isn't a dotted module) so the
writer and the tests never drift.

Run:
    .venv/bin/python -m pytest tests/test_stage_18_apkg.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import anki_models as M  # noqa: E402
from build.lib import anki_pilot as P  # noqa: E402
from build.lib.anki_templates import CARD_CSS, TEMPLATES  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
ANKI_TSV = DATA / "07-anki-listening-general.tsv"
MANIFEST = DATA / "07-anki-media-manifest.tsv"
PILOT_APKG = REPO_ROOT / "dist" / "bp-listening-pilot.apkg"
BUILD_18_0 = REPO_ROOT / "build" / "18_0_fetch_anki_media.py"
BUILD_18_1 = REPO_ROOT / "build" / "18_1_build_apkg.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stage18_1 = _load("stage18_1_build_apkg", BUILD_18_1)


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def export_rows() -> list[dict]:
    if not ANKI_TSV.exists():
        pytest.skip("data/07-anki-listening-general.tsv missing — run 17_1 first.")
    return sorted(P.read_anki_export(ANKI_TSV),
                  key=lambda r: int(r.get("anki_order") or 0))


@pytest.fixture(scope="session")
def pilot_rows(export_rows) -> list[dict]:
    ids = set(P.select_pilot_sense_ids(export_rows))
    return [r for r in export_rows if r["sense_id"] in ids]


@pytest.fixture(scope="session")
def records(pilot_rows) -> list[dict]:
    return stage18_1.build_notes(pilot_rows)


@pytest.fixture(scope="session")
def pilot_apkg(pilot_rows):
    """Fetch pilot media THEN build the pilot apkg. Skip if offline."""
    r0 = subprocess.run([sys.executable, str(BUILD_18_0), "--pilot"],
                        capture_output=True, text=True)
    if r0.returncode != 0:
        pytest.skip("18_0 --pilot failed (needs network to fetch media):\n"
                    + (r0.stderr or r0.stdout)[-400:])
    r1 = subprocess.run([sys.executable, str(BUILD_18_1), "--pilot"],
                        capture_output=True, text=True)
    assert r1.returncode == 0, f"18_1 --pilot failed:\n{r1.stderr or r1.stdout}"
    assert PILOT_APKG.exists()
    return PILOT_APKG


# ── tier 1: media-free logic ─────────────────────────────────────────────────
def test_note_fields_shape():
    assert len(M.NOTE_FIELDS) == 24
    assert M.NOTE_FIELDS[0] == "sense_id"  # Anki sort/dedup field
    assert "anki_tags" not in M.NOTE_FIELDS  # → tags, not a field
    assert "issue_hint" in M.NOTE_FIELDS


def test_templates_present_with_footer():
    assert set(TEMPLATES) == set(M.CARD_TYPES)
    for ct in M.CARD_TYPES:
        t = TEMPLATES[ct]
        assert t["qfmt"] and t["afmt"]
        assert "Flag this card red" in t["afmt"]
        assert "{{sense_id}}" in t["afmt"]


def test_card_types_are_listening_recall():
    assert set(M.CARD_TYPES) == {"listening", "recall"}


def test_per_card_type_counts(records, pilot_rows):
    n = len(pilot_rows)
    by_ct = Counter(r["card_type"] for r in records)
    for ct in M.CARD_TYPES:
        assert by_ct[ct] == n
    assert len(records) == 2 * n


def test_template_shape_backs():
    # both backs: PT sentence audio + pt/en text + sentence IPA
    for ct in M.CARD_TYPES:
        a = TEMPLATES[ct]["afmt"]
        for need in ("{{audio_example}}", "{{example_pt}}",
                     "{{example_en}}", "{{ipa_example}}"):
            assert need in a, f"{ct} afmt missing {need}"


def test_listening_front_pt_audio_only():
    lq = TEMPLATES["listening"]["qfmt"]
    assert "{{audio_example}}" in lq
    for bad in ("{{example_pt}}", "{{example_en}}", "{{audio_word}}"):
        assert bad not in lq


def test_recall_front_english_cue_only():
    rq = TEMPLATES["recall"]["qfmt"]
    assert "{{example_en}}" in rq and "{{audio_en_example}}" in rq
    for bad in ("{{example_pt}}", "{{ipa_example}}",
                "{{audio_example}}", "{{audio_word}}"):
        assert bad not in rq, f"recall front leaks {bad}"


def test_word_audio_is_click_only():
    # audio_word never as [sound:]; audio_word_file only on the back, in <audio>
    for ct in M.CARD_TYPES:
        q, a = TEMPLATES[ct]["qfmt"], TEMPLATES[ct]["afmt"]
        assert "{{audio_word}}" not in q and "{{audio_word}}" not in a
        assert "{{audio_word_file}}" not in q
        assert "{{audio_word_file}}" in a
        pre = a[:a.find("{{audio_word_file}}")]
        assert pre.rfind("<audio") > pre.rfind("</audio>"), f"{ct}: not inside <audio>"


def test_no_stale_paste():
    # guards against old deck-list / run-plan text leaking into templates
    stale = ["::02 Word Listening", "::03 General Recognition",
             ".venv/bin/python", "verify_all.py"]
    blobs = [CARD_CSS]
    for t in TEMPLATES.values():
        blobs += [t["qfmt"], t["afmt"]]
    for b in blobs:
        for s in stale:
            assert s not in b, f"stale paste {s!r} found in a template/CSS"


def test_guids_unique(records):
    guids = [r["guid"] for r in records]
    assert len(set(guids)) == len(guids)


def test_guids_deterministic(pilot_rows):
    a = stage18_1.build_notes(pilot_rows)
    b = stage18_1.build_notes(pilot_rows)
    assert [r["guid"] for r in a] == [r["guid"] for r in b]


def test_sense_id_nonempty(records):
    assert all(r["sense_id"] for r in records)


def test_no_http_in_fields(records):
    for r in records:
        for k, v in r["field_map"].items():
            assert "http" not in str(v).lower(), f"{r['sense_id']}:{k}={v!r}"


def test_audio_en_example_keeps_sound_tag(records):
    # v2: en audio is rendered on the recall front → must stay [sound:...]
    for r in records:
        v = r["field_map"]["audio_en_example"]
        assert v.startswith("[sound:") and v.endswith(".mp3]"), f"{r['sense_id']}: {v!r}"


def test_audio_example_keeps_sound_tag(records):
    for r in records:
        v = r["field_map"]["audio_example"]
        assert v.startswith("[sound:") and v.endswith(".mp3]"), f"{r['sense_id']}"


def test_audio_word_file_is_safe_basename(records):
    rx = re.compile(r"^[A-Za-z0-9._-]+\.mp3$")
    for r in records:
        awf = r["field_map"]["audio_word_file"]
        assert "[sound:" not in awf
        assert rx.match(awf), f"{r['sense_id']}: unsafe audio_word_file {awf!r}"


def test_required_media_all_three_types(pilot_rows):
    needed = stage18_1.required_media(pilot_rows)
    assert needed, "expected some required media"
    assert all(b.endswith(".mp3") for b in needed)
    assert any("-en_ex-" in b for b in needed)                       # en_ex
    assert any("-ex-" in b and "-en_ex-" not in b for b in needed)   # example
    assert any("-word-" in b for b in needed)                        # word


def test_tags_include_card_stage_source(records):
    for r in records:
        tags = r["tags"]
        assert M.CARD_TAGS[r["card_type"]] in tags
        assert "stage::18" in tags
        assert "source::programmatic" in tags
        for t in tags:
            assert "http" not in t


def test_pilot_includes_all_spike_bads(pilot_rows):
    ids = {r["sense_id"] for r in pilot_rows}
    for sid in P.SPIKE_BAD_SENSE_IDS:
        assert sid in ids, f"spike-bad {sid} missing from pilot"


def test_pilot_size_reasonable(pilot_rows):
    # first-100 + 12 spike-bads + ~11 edge cases, deduped
    assert 100 <= len(pilot_rows) <= 140


# ── tier 2: packaging + media (needs network; skips if offline) ──────────────
def _open_apkg(apkg: Path):
    z = zipfile.ZipFile(apkg)
    db_name = "collection.anki21" if "collection.anki21" in z.namelist() else "collection.anki2"
    media = json.loads(z.read("media"))
    return z, db_name, media


def test_pilot_apkg_exists(pilot_apkg):
    assert pilot_apkg.exists() and pilot_apkg.stat().st_size > 0


def test_apkg_bundles_all_three_clip_types(pilot_apkg):
    _z, _db, media = _open_apkg(pilot_apkg)
    assert media, "no media bundled"
    assert all(v.endswith(".mp3") for v in media.values())
    vals = list(media.values())
    assert any("-en_ex-" in v for v in vals)                      # en_ex bundled
    assert any("-ex-" in v and "-en_ex-" not in v for v in vals)  # example bundled
    assert any("-word-" in v for v in vals)                       # word bundled


def test_apkg_media_matches_manifest_md5(pilot_apkg):
    z, _db, media = _open_apkg(pilot_apkg)
    man = {r["media_filename"]: (r.get("md5") or "").strip() for r in read_tsv(MANIFEST)}
    for idx, basename in media.items():
        want = man.get(basename, "")
        if not want:
            continue
        got = hashlib.md5(z.read(idx)).hexdigest()
        assert got == want, f"{basename}: md5 {got} != manifest {want}"


def test_apkg_referenced_media_all_bundled(pilot_apkg, pilot_rows):
    _z, _db, media = _open_apkg(pilot_apkg)
    bundled = set(media.values())
    needed = stage18_1.required_media(pilot_rows)
    missing = needed - bundled
    assert not missing, f"{len(missing)} referenced media not bundled: {sorted(missing)[:3]}"


def test_apkg_note_and_card_counts(pilot_apkg, pilot_rows):
    import sqlite3
    z, db_name, _media = _open_apkg(pilot_apkg)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / db_name
        p.write_bytes(z.read(db_name))
        con = sqlite3.connect(p)
        try:
            n_notes = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            n_cards = con.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            n_guids = con.execute("SELECT COUNT(DISTINCT guid) FROM notes").fetchone()[0]
        finally:
            con.close()
    n = len(pilot_rows)
    assert n_notes == 2 * n
    assert n_cards == 2 * n  # one card per note (one template per model)
    assert n_guids == 2 * n
