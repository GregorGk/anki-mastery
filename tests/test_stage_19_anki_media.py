"""Stage 19 — the click-only word clip must be a media reference Anki can see.

The bug: `audio_word_file` held a bare basename and the template built
`<audio controls preload="none" src="{{audio_word_file}}">`. Anki's
Tools > Check Media and the .apkg importer only count media referenced from
FIELDS (`[sound:…]` or `<img|audio|video|source src=…>`); a template-only
reference counts only for files starting with `_`. So every word clip was
"unused" and "Delete Unused" would delete them.

The fix: the field holds the whole `<audio controls preload="none"
src="X.mp3"></audio>` tag (field name unchanged) and the template renders
`{{audio_word_file}}` raw inside a `{{#audio_word_file}}` guard. Still
click-only: no `[sound:]` for the word, no autoplay.

Two tiers, both on synthetic rows (no data/, no dist/, no network):

  * pure — field builder, prepare_field_map, template shape, build gates.
  * anki pylib — builds tiny .apkg files into tmp_path with 18_1's own
    functions, imports them into a fresh collection, runs Check Media,
    renders the cards, and re-imports over the old shape. Skips unless the
    `anki` package is importable:
        uv run --with anki pytest tests/test_stage_19_anki_media.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import anki_models as M  # noqa: E402
from build.lib.anki_templates import (  # noqa: E402
    TEMPLATES,
    WORD_AUDIO_TAG,
    word_audio_field,
)

BUILD_18_1 = REPO_ROOT / "build" / "18_1_build_apkg.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("stage19_anki_media_18_1", BUILD_18_1)

WORD = "0001.00.01-word-eleven_v4-v1.mp3"
WORD_TAG = f'<audio controls preload="none" src="{WORD}"></audio>'
FULL_TAG_RE = re.compile(r'^<audio controls preload="none" src="([A-Za-z0-9._-]+\.mp3)"></audio>$')
# The pre-fix template shape (word clip referenced only from the template).
OLD_WORD_AUDIO = '<audio controls preload="none" src="{{audio_word_file}}"></audio>'


def _row(sid: str, order: int = 1, *, word: bool = True) -> dict:
    """A minimal Stage-17 export row (only what 18_1 reads)."""
    return {
        "sense_id": sid,
        "anki_order": str(order),
        "pt": "palavra",
        "pt_display_safe": "palavra",
        "en_primary": "word",
        "example_pt": "Uma palavra.",
        "example_en": "A word.",
        "ipa_example": "ˈuma paˈlavɾɐ",
        "audio_word": f"[sound:{sid}-word-eleven_v4-v1.mp3]" if word else "",
        "audio_example": f"[sound:{sid}-ex-eleven_v4-v1.mp3]",
        "audio_en_example": f"[sound:{sid}-en_ex-eleven_v4-v1.mp3]",
        "anki_tags": "topic::test",
    }


ROWS = [_row("0001.00.01", 1), _row("0002.00.01", 2), _row("0003.00.01", 3)]


@pytest.fixture
def media(tmp_path, monkeypatch) -> dict[str, str]:
    """Fake mp3s for ROWS in tmp_path; 18_1's MEDIA_DIR points there.

    Returns {basename: md5} (the manifest md5 map _assert_build checks).
    """
    d = tmp_path / "media"
    d.mkdir()
    man: dict[str, str] = {}
    for b in sorted(S.required_media(ROWS + [_row("0009.00.01", 9, word=False)])):
        data = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b.encode()
        (d / b).write_bytes(data)
        man[b] = hashlib.md5(data).hexdigest()
    monkeypatch.setattr(S, "MEDIA_DIR", d)
    return man


def _word_files(rows: list[dict]) -> set[str]:
    return {S._sound_basename(r["audio_word"]) for r in rows if r["audio_word"]}


def _set_old_shape(monkeypatch) -> None:
    """Rebuild the pre-fix shape: bare basename field + template <audio src>."""
    old_block = ('<div class="word-audio">\n'
                 '  <div class="small">Word audio, click only</div>\n'
                 f'  {OLD_WORD_AUDIO}\n'
                 '</div>')
    for ct in M.CARD_TYPES:
        a = TEMPLATES[ct]["afmt"]
        start = a.index("{{#audio_word_file}}")
        end = a.index("{{/audio_word_file}}") + len("{{/audio_word_file}}")
        monkeypatch.setitem(TEMPLATES[ct], "afmt", a[:start] + old_block + a[end:])
    monkeypatch.setattr(S, "word_audio_field", lambda basename: basename)


# ── tier 1: pure ─────────────────────────────────────────────────────────────
def test_word_audio_field_is_full_tag():
    assert word_audio_field(WORD) == WORD_TAG
    assert WORD_AUDIO_TAG.format(src=WORD) == WORD_TAG


def test_word_audio_field_empty_for_no_clip():
    assert word_audio_field("") == ""


def test_word_audio_field_escapes_src():
    # 18_1 rejects such names anyway; the builder must still not break the tag
    assert word_audio_field('a"b.mp3') == '<audio controls preload="none" src="a&quot;b.mp3"></audio>'


def test_field_name_unchanged():
    # keeps the note type's field list + the import mapping identical
    assert "audio_word_file" in M.NOTE_FIELDS
    assert "audio_word" not in M.NOTE_FIELDS
    assert len(M.NOTE_FIELDS) == 24


def test_prepare_field_map_word_clip_is_full_tag():
    fm = S.prepare_field_map(_row("0001.00.01"))
    awf = fm["audio_word_file"]
    assert awf == WORD_TAG
    assert FULL_TAG_RE.match(awf)
    assert "[sound:" not in awf
    assert "autoplay" not in awf.lower()


@pytest.mark.parametrize("value", ["", None, "missing", "0001.00.01-word-eleven_v4-v1.mp3",
                                   "[sound:]"])
def test_prepare_field_map_no_clip_is_empty(value):
    row = _row("0001.00.01")
    if value == "missing":
        del row["audio_word"]
    else:
        row["audio_word"] = value
    assert S.prepare_field_map(row)["audio_word_file"] == ""


def test_example_audio_fields_keep_sound_tags():
    fm = S.prepare_field_map(_row("0001.00.01"))
    assert fm["audio_example"] == "[sound:0001.00.01-ex-eleven_v4-v1.mp3]"
    assert fm["audio_en_example"] == "[sound:0001.00.01-en_ex-eleven_v4-v1.mp3]"


def test_word_clip_never_in_a_sound_tag():
    fm = S.prepare_field_map(_row("0001.00.01"))
    for k, v in fm.items():
        assert f"[sound:{WORD}]" not in v, f"{k} plays the word clip via [sound:]"
        if k != "audio_word_file":
            assert WORD not in v, f"{k} references the word clip"


def test_templates_render_word_field_raw():
    for ct in M.CARD_TYPES:
        q, a = TEMPLATES[ct]["qfmt"], TEMPLATES[ct]["afmt"]
        assert "audio_word_file" not in q
        assert a.count("{{audio_word_file}}") == 1
        # raw: not inside a template <audio …>, not in an attribute, no filter
        assert "<audio" not in a.lower()
        assert 'src="{{' not in a and "src='{{" not in a
        assert not re.search(r"\{\{[^}]*:\s*audio_word_file\s*\}\}", a)
        # guarded: an empty field renders nothing (not even the label)
        i = a.index("{{audio_word_file}}")
        assert a.rfind("{{#audio_word_file}}", 0, i) != -1
        assert a.find("{{/audio_word_file}}", i) != -1
        assert a.index("Word audio, click only") > a.rfind("{{#audio_word_file}}", 0, i)
        # no [sound:] for the word, no autoplay
        assert "{{audio_word}}" not in q + a
        assert "autoplay" not in (q + a).lower()


def test_assert_templates_accepts_current():
    S._assert_templates()


@pytest.mark.parametrize("bad", [
    OLD_WORD_AUDIO,                                                   # pre-fix shape
    '{{audio_word_file}}<audio controls src="{{audio_word_file}}"></audio>',
    "{{text:audio_word_file}}",
    "{{audio_word_file}}<audio autoplay></audio>",
    "{{audio_word}}{{audio_word_file}}",
])
def test_assert_templates_rejects_bad_word_audio(monkeypatch, bad):
    a = TEMPLATES["listening"]["afmt"].replace("{{audio_word_file}}", bad)
    monkeypatch.setitem(TEMPLATES["listening"], "afmt", a)
    with pytest.raises(S.ApkgError):
        S._assert_templates()


def test_assert_templates_rejects_word_on_front(monkeypatch):
    q = TEMPLATES["recall"]["qfmt"] + "{{audio_word_file}}"
    monkeypatch.setitem(TEMPLATES["recall"], "qfmt", q)
    with pytest.raises(S.ApkgError):
        S._assert_templates()


def test_assert_build_accepts_full_tag(media):
    records = S.build_notes(ROWS)
    S._assert_build(ROWS, records, S.required_media(ROWS), media)
    for r in records:
        m = FULL_TAG_RE.match(r["field_map"]["audio_word_file"])
        assert m and S._WORD_FILE_RE.match(m.group(1))


@pytest.mark.parametrize("bad", [
    WORD,                                                        # old bare basename
    f"[sound:{WORD}]",                                           # would autoplay
    "",                                                          # no word clip
    f'<audio controls preload="none" autoplay src="{WORD}"></audio>',
    '<audio controls preload="none" src="0001.00.01-word-not-in-manifest.mp3"></audio>',
    '<audio controls preload="none" src="../x.mp3"></audio>',
])
def test_assert_build_rejects_bad_word_field(media, bad):
    records = S.build_notes(ROWS)
    for r in records:
        if r["sense_id"] == "0001.00.01":
            r["field_map"]["audio_word_file"] = bad
    with pytest.raises(S.ApkgError):
        S._assert_build(ROWS, records, S.required_media(ROWS), media)


def test_required_media_still_bundles_word_clips():
    assert _word_files(ROWS) <= S.required_media(ROWS)


# ── tier 2: real Anki (pylib) — Check Media, render, re-import ──────────────
@pytest.fixture
def col(tmp_path):
    """A fresh, empty Anki collection in tmp_path (skips without the pylib)."""
    pytest.importorskip("anki", reason="anki pylib not installed "
                        "(uv run --with anki pytest tests/test_stage_19_anki_media.py)")
    from anki.collection import Collection

    d = tmp_path / "col"
    d.mkdir()
    c = Collection(str(d / "collection.anki2"))
    yield c
    c.close()


def _build(rows: list[dict], man: dict[str, str], out: Path, *,
           timestamp: float | None = None, gate: bool = True) -> Path:
    records = S.build_notes(rows)
    needed = S.required_media(rows)
    if gate:
        S._assert_build(rows, records, needed, man)
    S.write_apkg(records, needed, out, timestamp=timestamp)
    return out


def _import(col, apkg: Path):
    """Like the GUI import with "update existing notes" (if newer)."""
    from anki.collection import ImportAnkiPackageOptions, ImportAnkiPackageRequest
    from anki.import_export_pb2 import ImportAnkiPackageUpdateCondition

    if_newer = ImportAnkiPackageUpdateCondition.IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_IF_NEWER
    opts = ImportAnkiPackageOptions(
        merge_notetypes=False,
        update_notes=if_newer,
        update_notetypes=if_newer,
        with_scheduling=False,
        with_deck_configs=False,
    )
    req = ImportAnkiPackageRequest(package_path=str(apkg), options=opts)
    return col.import_anki_package(req).log


def _unused_or_absent(col, names: set[str]) -> set[str]:
    chk = col.media.check()
    media_dir = Path(col.media.dir())
    return {n for n in names if n in set(chk.unused) or not (media_dir / n).exists()}


def test_check_media_sees_word_clips(col, media, tmp_path):
    apkg = _build(ROWS, media, tmp_path / "new.apkg")
    log = _import(col, apkg)
    assert len(log.new) == len(ROWS) * len(M.CARD_TYPES)
    media_dir = Path(col.media.dir())
    for w in _word_files(ROWS):
        assert (media_dir / w).exists(), f"importer dropped word clip {w}"
    chk = col.media.check()
    assert list(chk.unused) == [], f"Check Media would delete: {list(chk.unused)}"
    assert list(chk.missing) == []


def test_old_shape_word_clips_are_unused_control(col, media, tmp_path, monkeypatch):
    # Negative control: proves the check above can detect the bug.
    _set_old_shape(monkeypatch)
    apkg = _build(ROWS, media, tmp_path / "old.apkg", gate=False)
    _import(col, apkg)
    assert _unused_or_absent(col, _word_files(ROWS)) == _word_files(ROWS)


def test_cards_render_click_only_word_audio(col, media, tmp_path):
    _import(col, _build(ROWS, media, tmp_path / "new.apkg"))
    cids = col.find_cards("")
    assert len(cids) == len(ROWS) * len(M.CARD_TYPES)
    for cid in cids:
        card = col.get_card(cid)
        sid = card.note()["sense_id"]
        word = f"{sid}-word-eleven_v4-v1.mp3"
        q, a = card.question(), card.answer()
        assert word not in q
        assert f'<audio controls preload="none" src="{word}"></audio>' in a
        assert "autoplay" not in (q + a).lower()
        played = {t.filename for t in card.answer_av_tags() if hasattr(t, "filename")}
        assert word not in played, "word clip must not be a [sound:] (auto)play tag"
        assert f"{sid}-ex-eleven_v4-v1.mp3" in played


def test_empty_word_clip_renders_nothing(col, media, tmp_path):
    rows = [_row("0009.00.01", 9, word=False)]
    _import(col, _build(rows, media, tmp_path / "noword.apkg", gate=False))
    for cid in col.find_cards(""):
        card = col.get_card(cid)
        assert card.note()["audio_word_file"] == ""
        a = card.answer()
        assert "<audio" not in a
        assert "Word audio, click only" not in a
        assert "0009.00.01-ex-eleven_v4-v1.mp3" in {
            t.filename for t in card.answer_av_tags() if hasattr(t, "filename")}


def test_reimport_over_old_shape_updates_in_place(col, media, tmp_path, monkeypatch):
    """The user's path: old deck already imported, then "update existing notes"."""
    now = time.time()
    with monkeypatch.context() as mp:
        _set_old_shape(mp)
        old = _build(ROWS, media, tmp_path / "old.apkg", timestamp=now - 3600, gate=False)
    _import(col, old)

    nids = sorted(col.find_notes(""))
    cids = sorted(col.find_cards(""))
    ntids = {col.get_note(n).mid for n in nids}
    # give one card review scheduling: it must survive the update
    col.sched.set_due_date([cids[0]], "7")
    before = col.get_card(cids[0])
    sched = (before.type, before.queue, before.ivl, before.due)
    assert before.type == 2  # review

    scm = col.db.scalar("select scm from col")
    log = _import(col, _build(ROWS, media, tmp_path / "new.apkg", timestamp=now))
    assert len(log.new) == 0
    assert len(log.updated) == len(nids)
    # same fields + template count → note type updated in place, no schema
    # change (so the import itself doesn't force a full sync)
    assert col.db.scalar("select scm from col") == scm

    assert sorted(col.find_notes("")) == nids          # same GUIDs → same notes
    assert sorted(col.find_cards("")) == cids          # no duplicate cards
    assert {col.get_note(n).mid for n in nids} == ntids  # same note types
    after = col.get_card(cids[0])
    assert (after.type, after.queue, after.ivl, after.due) == sched

    col.models._clear_cache()  # pylib caches note types; the import ran in the backend
    for mid in ntids:
        nt = col.models.get(mid)
        assert [f["name"] for f in nt["flds"]] == M.NOTE_FIELDS
        afmt = nt["tmpls"][0]["afmt"]
        assert "{{audio_word_file}}" in afmt and "<audio" not in afmt
    for n in nids:
        assert FULL_TAG_RE.match(col.get_note(n)["audio_word_file"])
    for cid in cids:  # rendered by the backend: one well-formed tag, not nested
        a = col.get_card(cid).answer()
        assert a.count("<audio") == 1 and 'src="<audio' not in a

    chk = col.media.check()
    assert list(chk.unused) == [] and list(chk.missing) == []
    media_dir = Path(col.media.dir())
    assert all((media_dir / w).exists() for w in _word_files(ROWS))
