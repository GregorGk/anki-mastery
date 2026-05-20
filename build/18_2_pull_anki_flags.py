"""Stage 18 / Step 2 — Pull red-flagged cards from Anki back into the repo.

Writes:  data/_learner_flags.tsv   (the repair queue; merges with any existing)

Two modes:

  --ankiconnect          Query a running Anki (AnkiConnect addon, port 8765)
                         for RED-flagged cards in the three Stage-18 decks.
                         Rich: deck → card_type → issue_scope, plus fields.

  --from-export PATH     Parse an Anki Browser TSV export (fallback when you
                         don't run AnkiConnect). Do: Browser → search
                         `flag:1` → select all → Notes → Export as "Notes in
                         Plain Text" WITH the tags column. We recover
                         sense_id (a field) + card_type (from the
                         `card::*` tag).

`flag:1` is RED specifically (not `-flag:0`, which is any non-zero color).
Re-pulling MERGES: existing `status`/`notes` for a (sense_id, card_type)
are preserved so your triage annotations survive a re-pull.

Stage 18 ONLY collects flags — it never regenerates audio/text. A later
stage consumes this TSV.

Usage:
    .venv/bin/python build/18_2_pull_anki_flags.py --ankiconnect
    .venv/bin/python build/18_2_pull_anki_flags.py --from-export flagged.tsv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.anki_pilot import read_anki_export  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
ANKI_TSV = DATA / "07-anki-listening-general.tsv"
FLAGS_TSV = DATA / "_learner_flags.tsv"

ANKICONNECT_URL = "http://localhost:8765"
DECK_GLOB = "Brazilian Portuguese Mastery::*"

LEARNER_FLAG_FIELDS = [
    "pulled_at", "sense_id", "deck", "card_type", "flag_color",
    "pt", "en_primary", "example_pt", "example_en",
    "issue_scope", "status", "notes",
]

SENSE_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}")

CARD_TYPE_BY_DECK_SUFFIX = {
    "01 Sentence Listening": "listening",
    "02 Sentence Recall": "recall",
}
CARD_TYPE_BY_TAG = {
    "card::sentence-listening": "listening",
    "card::sentence-recall": "recall",
}
ISSUE_SCOPE = {
    "listening": "sentence_or_example_audio",
    "recall": "sentence_text_or_audio",
    "": "",
}
FLAG_COLOR = {1: "red", 2: "orange", 3: "green", 4: "blue",
              5: "pink", 6: "turquoise", 7: "purple"}


class FlagError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (18_2_pull_anki_flags): {msg}")


def _anki_connect(action: str, **params):
    payload = json.dumps({"action": action, "version": 6,
                          "params": params}).encode("utf-8")
    req = urllib.request.Request(ANKICONNECT_URL, payload,
                                 {"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as exc:  # noqa: BLE001
        raise FlagError(
            f"cannot reach AnkiConnect at {ANKICONNECT_URL} ({exc}). "
            "Is Anki running with the AnkiConnect addon? Or use --from-export.")
    if resp.get("error"):
        raise FlagError(f"AnkiConnect: {resp['error']}")
    return resp.get("result")


def _enrichment() -> dict[str, dict]:
    """sense_id → {pt, en_primary, example_pt, example_en} from the export."""
    out: dict[str, dict] = {}
    for r in read_anki_export(ANKI_TSV) if ANKI_TSV.exists() else []:
        out[r["sense_id"]] = {
            "pt": r.get("pt_display_safe") or r.get("pt", ""),
            "en_primary": r.get("en_primary", ""),
            "example_pt": r.get("example_pt", ""),
            "example_en": r.get("example_en", ""),
        }
    return out


def _card_type_from_deck(deck: str) -> str:
    for suffix, ct in CARD_TYPE_BY_DECK_SUFFIX.items():
        if deck.endswith(suffix):
            return ct
    return ""


def _pull_ankiconnect() -> list[dict]:
    card_ids = _anki_connect("findCards", query=f'deck:"{DECK_GLOB}" flag:1')
    if not card_ids:
        return []
    infos = _anki_connect("cardsInfo", cards=card_ids)
    enrich = _enrichment()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict] = []
    for info in infos:
        deck = info.get("deckName", "")
        ct = _card_type_from_deck(deck)
        fields = info.get("fields", {})
        sense_id = (fields.get("sense_id", {}) or {}).get("value", "") or ""
        if not sense_id:
            # fall back to scanning all field values
            for fv in fields.values():
                m = SENSE_RE.search((fv or {}).get("value", "") or "")
                if m:
                    sense_id = m.group(0)
                    break
        flag = info.get("flags", 1)
        e = enrich.get(sense_id, {})
        rows.append({
            "pulled_at": now, "sense_id": sense_id, "deck": deck,
            "card_type": ct, "flag_color": FLAG_COLOR.get(flag, str(flag)),
            "pt": e.get("pt", ""), "en_primary": e.get("en_primary", ""),
            "example_pt": e.get("example_pt", ""),
            "example_en": e.get("example_en", ""),
            "issue_scope": ISSUE_SCOPE.get(ct, ""), "status": "open", "notes": "",
        })
    return rows


def _pull_from_export(path: Path) -> list[dict]:
    if not path.exists():
        raise FlagError(f"export file not found: {path}")
    enrich = _enrichment()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader((ln for ln in f if not ln.startswith("#")),
                            dialect="excel-tab")
        for cells in reader:
            blob = "\t".join(cells)
            m = SENSE_RE.search(blob)
            if not m:
                continue
            sense_id = m.group(0)
            ct = ""
            for tag, t in CARD_TYPE_BY_TAG.items():
                if tag in blob:
                    ct = t
                    break
            e = enrich.get(sense_id, {})
            rows.append({
                "pulled_at": now, "sense_id": sense_id, "deck": "",
                "card_type": ct, "flag_color": "red",
                "pt": e.get("pt", ""), "en_primary": e.get("en_primary", ""),
                "example_pt": e.get("example_pt", ""),
                "example_en": e.get("example_en", ""),
                "issue_scope": ISSUE_SCOPE.get(ct, ""), "status": "open",
                "notes": "",
            })
    return rows


def _merge_prior(rows: list[dict]) -> list[dict]:
    """Preserve existing status/notes for matching (sense_id, card_type)."""
    if not FLAGS_TSV.exists():
        return rows
    prior = {(r["sense_id"], r.get("card_type", "")): r
             for r in read_tsv(FLAGS_TSV)}
    for r in rows:
        key = (r["sense_id"], r.get("card_type", ""))
        if key in prior:
            if prior[key].get("status"):
                r["status"] = prior[key]["status"]
            if prior[key].get("notes"):
                r["notes"] = prior[key]["notes"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ankiconnect", action="store_true")
    g.add_argument("--from-export", type=Path, metavar="PATH")
    args = ap.parse_args()

    if args.ankiconnect:
        rows = _pull_ankiconnect()
        mode = "ankiconnect"
    else:
        rows = _pull_from_export(args.from_export)
        mode = f"export:{args.from_export.name}"

    rows.sort(key=lambda r: (r["sense_id"], r.get("card_type", "")))
    rows = _merge_prior(rows)
    n = write_tsv(FLAGS_TSV, rows, fieldnames=LEARNER_FLAG_FIELDS)

    from collections import Counter
    by_ct = Counter(r["card_type"] or "?" for r in rows)
    print(f"=== Stage 18.2 — pull learner flags ({mode}) ===")
    print(f"  flagged cards: {n}")
    for ct in ("listening", "recall", "?"):
        if by_ct.get(ct):
            print(f"    {ct:<12} {by_ct[ct]}")
    print(f"  wrote: {FLAGS_TSV.relative_to(REPO_ROOT)}")
    print("  next:  .venv/bin/python build/18_3_flags_html.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
