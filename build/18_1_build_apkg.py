"""Stage 18 / Step 1 — Build the listening-first Anki `.apkg`.

Reads (read-only):  data/07-anki-listening-general.tsv   (clean learner TSV)
                    data/07-anki-media-manifest.tsv       (md5 verification)
                    data/anki_media/                      (bundled mp3s)
Writes:             dist/bp-listening-pilot.apkg  /  dist/bp-listening-general.apkg
                    reports/18_apkg_build.html

Builds two notes per sense (Sentence Listening / Sentence Recall), one per
genanki model, each added to its own subdeck. Deterministic GUIDs
(`anki_models.guid_seed`) make re-imports update rather than duplicate.
Notes are added in ascending `anki_order` (insertion order; no `due` set).

`audio_example` + `audio_en_example` are rendered as `[sound:…]` (autoplay
where placed). Word audio is click-only: rendered as a native HTML5
`<audio src="{{audio_word_file}}">` (bare basename), never `[sound:…]`.
All three clip types (word + example + en_ex) are bundled.

Run AFTER build/18_0_fetch_anki_media.py (it asserts media presence).

Usage:
    .venv/bin/python build/18_1_build_apkg.py            # full deck
    .venv/bin/python build/18_1_build_apkg.py --pilot    # pilot smoke-test
    .venv/bin/python build/18_1_build_apkg.py --limit 300
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
from collections import Counter
from pathlib import Path

import genanki

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.anki_models import (  # noqa: E402
    CARD_TYPES, ISSUE_HINT, NOTE_FIELDS, BPNote,
    build_decks, build_models, guid_seed, tags_for,
)
from build.lib.anki_templates import CARD_CSS, TEMPLATES  # noqa: E402
from build.lib.anki_pilot import (  # noqa: E402
    read_anki_export, select_pilot_sense_ids,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
ANKI_TSV = DATA / "07-anki-listening-general.tsv"
MANIFEST = DATA / "07-anki-media-manifest.tsv"
MEDIA_DIR = DATA / "anki_media"
DIST = REPO_ROOT / "dist"
REPORT = REPO_ROOT / "reports" / "18_apkg_build.html"

PILOT_APKG = DIST / "bp-listening-pilot.apkg"
FULL_APKG = DIST / "bp-listening-general.apkg"

# audio_word_file must be a safe bare basename for the <audio src> attribute.
_WORD_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.mp3$")


class ApkgError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (18_1_build_apkg): {msg}")


def _sound_basename(v: str | None) -> str:
    """'[sound:x.mp3]' -> 'x.mp3'; anything else -> ''."""
    v = v or ""
    if v.startswith("[sound:") and v.endswith("]"):
        return v[len("[sound:"):-1]
    return ""


def _md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def prepare_field_map(row: dict) -> dict:
    """Map a Stage-17 export row → the 24 NOTE_FIELDS values.

    `audio_example` / `audio_en_example` keep their `[sound:…]` tags
    (rendered, autoplay where placed). `audio_word_file` is the word clip's
    bare basename (for the click-only `<audio src>`, NEVER `[sound:…]`).
    `issue_hint` is the constant prompt.
    """
    fm: dict[str, str] = {}
    for f in NOTE_FIELDS:
        if f == "issue_hint":
            fm[f] = ISSUE_HINT
        elif f == "audio_word_file":
            # bare basename for the click-only <audio src> (never [sound:])
            fm[f] = _sound_basename(row.get("audio_word"))
        else:
            fm[f] = row.get(f, "") or ""
    return fm


def required_media(export_rows: list[dict]) -> set[str]:
    """Basenames of every bundled clip: word + example + en_ex.

    All three are stored as `[sound:…]` in the Stage-17 export; we extract
    each basename. (Word is rendered via `<audio src>` not `[sound:]`, but
    still bundled so the element resolves.)
    """
    needed: set[str] = set()
    for row in export_rows:
        for col in ("audio_word", "audio_example", "audio_en_example"):
            b = _sound_basename(row.get(col))
            if b:
                needed.add(b)
    return needed


def build_notes(export_rows: list[dict]) -> list[dict]:
    """Three note records per sense. Pure (no I/O, no packaging) → testable.

    Each record: card_type, sense_id, guid, fields (NOTE_FIELDS order),
    field_map, tags, note (the genanki.Note).
    """
    models = build_models()
    records: list[dict] = []
    for row in export_rows:
        fm = prepare_field_map(row)
        fields = [fm[f] for f in NOTE_FIELDS]
        sid = row["sense_id"]
        # No `due`/scheduling metadata — new-card order follows insertion
        # order (notes are added in ascending anki_order).
        for ct in CARD_TYPES:
            note = BPNote(
                model=models[ct],
                fields=list(fields),
                tags=tags_for(row, ct),
                guid_seed=guid_seed(ct, sid),
            )
            records.append({
                "card_type": ct, "sense_id": sid, "guid": note.guid,
                "fields": fields, "field_map": fm, "tags": note.tags,
                "note": note,
            })
    return records


def _assert_templates() -> None:
    """Lock the v2 sentence-first template shape + guard against stale paste."""
    stale = ["::02 Word Listening", "::03 General Recognition",
             ".venv/bin/python", "verify_all.py"]
    lq = TEMPLATES["listening"]["qfmt"]
    rq = TEMPLATES["recall"]["qfmt"]

    # listening front: PT sentence audio only
    if "{{audio_example}}" not in lq:
        raise ApkgError("listening qfmt must contain audio_example.")
    for bad in ("{{example_pt}}", "{{example_en}}", "{{audio_word}}"):
        if bad in lq:
            raise ApkgError(f"listening qfmt must NOT contain {bad}.")

    # recall front: English cue only (no Portuguese answer leak)
    for need in ("{{example_en}}", "{{audio_en_example}}"):
        if need not in rq:
            raise ApkgError(f"recall qfmt must contain {need}.")
    for bad in ("{{example_pt}}", "{{ipa_example}}", "{{audio_example}}", "{{audio_word}}"):
        if bad in rq:
            raise ApkgError(f"recall qfmt must NOT contain {bad} (answer leak).")

    # shared backs
    for ct in CARD_TYPES:
        a = TEMPLATES[ct]["afmt"]
        for need in ("{{audio_example}}", "{{example_pt}}", "{{example_en}}", "{{ipa_example}}"):
            if need not in a:
                raise ApkgError(f"{ct} afmt must contain {need}.")

    # word audio: never [sound:]; audio_word_file only on the back, inside <audio>
    for ct in CARD_TYPES:
        q, a = TEMPLATES[ct]["qfmt"], TEMPLATES[ct]["afmt"]
        if "{{audio_word}}" in q or "{{audio_word}}" in a:
            raise ApkgError(f"{ct}: audio_word must not appear; use audio_word_file.")
        if "{{audio_word_file}}" in q:
            raise ApkgError(f"{ct}: audio_word_file must not appear in qfmt.")
        if "{{audio_word_file}}" not in a:
            raise ApkgError(f"{ct}: audio_word_file must appear in afmt.")
        pre = a[:a.find("{{audio_word_file}}")]
        if pre.rfind("<audio") <= pre.rfind("</audio>"):
            raise ApkgError(f"{ct}: audio_word_file must be inside an <audio> element.")

    # anti-corruption
    blobs = [CARD_CSS] + [t[k] for t in TEMPLATES.values() for k in ("qfmt", "afmt")]
    for b in blobs:
        for s in stale:
            if s in b:
                raise ApkgError(f"stale-paste corruption in template/CSS: {s!r}")


def _assert_build(export_rows: list[dict], records: list[dict],
                  needed: set[str], man_md5: dict[str, str]) -> None:
    _assert_templates()
    n = len(export_rows)
    by_ct = Counter(r["card_type"] for r in records)
    for ct in CARD_TYPES:
        if by_ct[ct] != n:
            raise ApkgError(f"deck '{ct}' has {by_ct[ct]} notes, expected {n}.")
    if len(records) != len(CARD_TYPES) * n:
        raise ApkgError(f"{len(records)} notes built, expected {len(CARD_TYPES) * n}.")
    guids = [r["guid"] for r in records]
    if len(set(guids)) != len(guids):
        raise ApkgError("duplicate note GUIDs.")
    manifest_files = set(man_md5)
    for r in records:
        if not r["sense_id"]:
            raise ApkgError("a note has an empty sense_id.")
        fm = r["field_map"]
        awf = fm["audio_word_file"]
        if not _WORD_FILE_RE.match(awf):
            raise ApkgError(f"{r['sense_id']}: audio_word_file {awf!r} is not a safe "
                            f"basename (must match ^[A-Za-z0-9._-]+\\.mp3$).")
        if awf not in manifest_files:
            raise ApkgError(f"{r['sense_id']}: audio_word_file {awf!r} is not a "
                            f"manifest media_filename.")
        for col in ("audio_example", "audio_en_example"):
            v = fm[col]
            if not (v.startswith("[sound:") and v.endswith(".mp3]")):
                raise ApkgError(f"{r['sense_id']}: {col}={v!r} is not a [sound:…mp3] tag.")
        for k, v in fm.items():
            if "http" in str(v).lower():
                raise ApkgError(f"{r['sense_id']}: field {k} contains a raw URL: {v!r}")
    missing = sorted(b for b in needed if not (MEDIA_DIR / b).exists())
    if missing:
        raise ApkgError(f"{len(missing)} media files missing from {MEDIA_DIR} — run "
                        f"build/18_0_fetch_anki_media.py first. e.g. {missing[:3]}")
    bad_md5 = []
    for b in needed:
        want = man_md5.get(b, "")
        if want and _md5_file(MEDIA_DIR / b) != want:
            bad_md5.append(b)
    if bad_md5:
        raise ApkgError(f"{len(bad_md5)} bundled media files have md5 != manifest, "
                        f"e.g. {sorted(bad_md5)[:3]}")


def _write_report(out_path: Path, scope: str, n_senses: int,
                  records: list[dict], n_media: int) -> None:
    css = """
    :root{--bg:#0f1115;--panel:#181b22;--panel-2:#20242d;--border:#2c3140;
      --text:#e6e9ef;--muted:#9098a6;--accent:#5aa9ff;--ok:#4ade80;}
    *{box-sizing:border-box;}
    body{background:var(--bg);color:var(--text);margin:0;padding:24px;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
    .wrap{max-width:900px;margin:0 auto;}
    h1{font-size:21px;margin:0 0 4px;} .sub{color:var(--muted);margin:0 0 18px;}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 20px;}
    td,th{padding:7px 12px;text-align:left;border-bottom:1px solid var(--border);}
    th{color:var(--muted);font-weight:500;}
    td.num{text-align:right;font-family:ui-monospace,Menlo,monospace;}
    code{background:var(--panel-2);padding:1px 5px;border-radius:3px;font-size:12px;}
    """
    by_ct = Counter(r["card_type"] for r in records)
    stats = {
        "senses selected": n_senses,
        "notes built (3×)": len(records),
        "  Sentence Listening": by_ct["sentence"],
        "  Word Listening": by_ct["word"],
        "  General Recognition": by_ct["recognition"],
        "media files bundled": n_media,
    }
    rows_html = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v:,}</td></tr>"
        for k, v in stats.items()
    )
    doc = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Stage 18 — apkg build</title><style>{css}</style></head><body>"
        f"<div class='wrap'><h1>Stage 18 — apkg build</h1>"
        f"<p class='sub'>scope <code>{html.escape(scope)}</code> · output "
        f"<code>{html.escape(str(out_path.relative_to(REPO_ROOT)))}</code></p>"
        f"<table><tr><th>metric</th><th class='num'>count</th></tr>{rows_html}</table>"
        f"</div></body></html>"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only senses with anki_order <= N")
    args = ap.parse_args()

    if not ANKI_TSV.exists():
        raise ApkgError(f"{ANKI_TSV} missing — run build/17_1_export_anki.py first.")

    rows = sorted(read_anki_export(ANKI_TSV),
                  key=lambda r: int(r.get("anki_order") or 0))
    if not rows:
        raise ApkgError(f"{ANKI_TSV} has no data rows.")

    if args.pilot:
        ids = set(select_pilot_sense_ids(rows))
        sel = [r for r in rows if r["sense_id"] in ids]
        out_path, scope = PILOT_APKG, "pilot"
    elif args.limit:
        sel = [r for r in rows if int(r.get("anki_order") or 0) <= args.limit]
        out_path, scope = FULL_APKG, f"limit={args.limit}"
    else:
        sel = rows
        out_path, scope = FULL_APKG, "full"

    man_md5 = {r["media_filename"]: (r.get("md5") or "").strip()
               for r in read_tsv(MANIFEST)}

    records = build_notes(sel)
    needed = required_media(sel)
    _assert_build(sel, records, needed, man_md5)

    decks = build_decks()
    for rec in records:
        decks[rec["card_type"]].add_note(rec["note"])

    media_files = [str(MEDIA_DIR / b) for b in sorted(needed)]
    pkg = genanki.Package(list(decks.values()))
    pkg.media_files = media_files

    DIST.mkdir(parents=True, exist_ok=True)
    pkg.write_to_file(str(out_path))

    _write_report(out_path, scope, len(sel), records, len(media_files))

    print(f"=== Stage 18.1 — build apkg ({scope}) ===")
    print(f"  senses:        {len(sel):,}")
    print(f"  notes:         {len(records):,}  ({len(CARD_TYPES)} × {len(sel):,})")
    print(f"  media bundled: {len(media_files):,}")
    print(f"  output:        {out_path.relative_to(REPO_ROOT)}  "
          f"({out_path.stat().st_size/1024:.0f} KB)")
    print(f"  report:        {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
