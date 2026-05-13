"""Stage 16 / Step 5 — Per-text EP-vs-BP listening cues via Claude.

For each unique text in `data/_audio_judge_ab.tsv`, ask Claude (Sonnet 4.6
via the existing AnthropicClient with prompt caching) to produce:
  - BP IPA target pronunciation
  - EP IPA alternative pronunciation
  - 2-4 specific listening cues distinguishing the two
  - one "headline marker" — the most obvious tell

Output to `data/_ep_cues.tsv`. The listening HTML (16_3) joins on `text`.

This helps a BP-fluent / EP-unfamiliar listener know what to listen for
on each clip — without giving away the dialect itself.

Cost: ~100 unique texts × ~$0.003 (cached Sonnet 4.6) ≈ **$0.30**.
Wall: ~30–60 s with concurrency.

Usage:
    .venv/bin/python build/16_5_ep_cues.py --dry-run
    .venv/bin/python build/16_5_ep_cues.py --yes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import AnthropicClient  # noqa: E402

DATA = REPO_ROOT / "data"
AB_TSV = DATA / "_audio_judge_ab.tsv"
OUT_TSV = DATA / "_ep_cues.tsv"
AUDIT_JSONL = REPO_ROOT / "audit" / "16_ep_cues.jsonl"

WORKERS = 8

CUES_TOOL_NAME = "ep_bp_listening_cues"
CUES_TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "bp_ipa": {
            "type": "string",
            "description": "Standard Brazilian Portuguese IPA transcription of the entire input (no slashes/brackets).",
        },
        "ep_ipa": {
            "type": "string",
            "description": "Standard European Portuguese IPA transcription of the same input.",
        },
        "headline_marker": {
            "type": "string",
            "description": "The single clearest acoustic tell. Plain English, ≤120 chars. Describe what BP does vs what EP does on the most distinguishing token in this specific text.",
        },
        "cues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 concrete listening cues, each one sentence. Format: '<token>: BP <phonetic> vs EP <phonetic> — <why distinct>'. Only include cues that are genuinely audible in a ~3s clip.",
            "minItems": 2,
            "maxItems": 4,
        },
        "difficulty": {
            "type": "string",
            "enum": ["obvious", "moderate", "subtle"],
            "description": "How easy it is for a casual BP-fluent listener (without EP exposure) to detect the dialect on this text. 'obvious' = clear final -l or final -e contrast; 'subtle' = mostly prosody.",
        },
    },
    "required": ["bp_ipa", "ep_ipa", "headline_marker", "cues", "difficulty"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a Brazilian Portuguese / European Portuguese phonetics tutor.

Your job: given one Portuguese text (a word or a short example sentence),
produce listening cues that help a Brazilian Portuguese speaker (fluent
in BP, NOT familiar with EP) recognize whether a spoken recording sounds
like BP or EP.

You DO NOT receive audio. You only get the text. Your output must be
phonetics-only — describe what the SAME text sounds like in each dialect.

Key BP-vs-EP markers (apply when present in the text):

  - Final -l → BP [w] (vocalized); EP [ɫ] (dark consonantal).
  - Final -de / -te → BP [dʒi] / [tʃi] (palatalized "ji" / "chi");
    EP [ð(ɨ)] / [t(ɨ)] (often near-silent, no palatalization).
  - Coda -s before consonant or pause → BP [s] or [z]; EP [ʃ] or [ʒ].
  - Final unstressed -e → BP [i] (clear "ee"); EP [ɨ] or silent.
  - Final unstressed -o → BP [u]; EP [u] (similar) but with weaker quality.
  - Initial r-, double rr → BP [h]/[χ] (back fricative); EP [ʁ] (uvular).
  - Pretonic -e- → BP [e]; EP [ɨ] (in some positions).
  - Stress on cognates (animal, hospital, social) → final syllable in both.
  - Initial h- → silent in both (no aspiration).
  - Nasal endings -ão, -ãe, -õe: similar.

Output rules:
  - `bp_ipa` and `ep_ipa`: faithful IPA for the whole input, no /…/ wrappers.
  - `headline_marker`: ONE plain-English sentence — what's the loudest tell?
    Bias toward final -l, final -e, coda -s, final -de/-te. Cite the actual
    token from the text.
  - `cues`: 2-4 concrete cues, EACH naming the specific token and giving
    the BP vs EP phonetic contrast for that token. NO meta-cues like
    "listen to prosody" — only audible token-level contrasts.
  - `difficulty`:
      * "obvious"  — at least one final -l, final -e/-de/-te, or coda -s
                     contrast is present that's audible in a normal-rate clip.
      * "moderate" — only one mild contrast (e.g. pretonic vowel quality).
      * "subtle"   — text lacks strong dialect markers (short, vowel-final,
                     no final -l, no -de/-te, no coda -s).
  - Stay neutral — do NOT say which dialect "sounds correct"; both are
    legitimate. You're just describing the contrast.
"""


def _user_message(text: str, clip_type: str) -> str:
    return (
        f"Portuguese text (clip_type = {clip_type}):\n"
        f"  {text}\n\n"
        f"Produce BP IPA, EP IPA, a headline marker, 2-4 concrete listening "
        f"cues, and a difficulty rating."
    )


def _read_unique_texts() -> list[tuple[str, str]]:
    """Return [(text, clip_type)] de-duplicated. BP and EP rows share text;
    we only need one query per (text, clip_type).

    Skips rows whose (text, clip_type) is already in OUT_TSV — so re-runs
    after the pool grows (16_6 expansion) only process the new texts.
    """
    cached: set[tuple[str, str]] = set()
    if OUT_TSV.exists():
        for r in csv.DictReader(OUT_TSV.open(encoding="utf-8"), dialect="excel-tab"):
            cached.add((r["text"], r["clip_type"]))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for r in csv.DictReader(AB_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        key = (r["text"], r["clip_type"])
        if key in seen or key in cached:
            continue
        seen.add(key)
        out.append(key)
    return out


def _run_one(client: AnthropicClient, text: str, clip_type: str) -> dict:
    res = client.call_tool(
        system=SYSTEM_PROMPT,
        user_message=_user_message(text, clip_type),
        tool_name=CUES_TOOL_NAME,
        tool_input_schema=CUES_TOOL_INPUT_SCHEMA,
        tool_description="Produce BP/EP listening cues for this Portuguese text.",
        max_tokens=600,
        stage="16_5_ep_cues",
        provenance_key=f"{clip_type}:{text}",
    )
    return res


FIELDS = ["text", "clip_type", "bp_ipa", "ep_ipa", "headline_marker",
          "cue_1", "cue_2", "cue_3", "cue_4", "difficulty"]


def _to_row(text: str, clip_type: str, payload: dict) -> dict:
    cues = list(payload.get("cues") or [])
    cues = (cues + ["", "", "", ""])[:4]
    return {
        "text": text,
        "clip_type": clip_type,
        "bp_ipa": payload.get("bp_ipa", ""),
        "ep_ipa": payload.get("ep_ipa", ""),
        "headline_marker": payload.get("headline_marker", ""),
        "cue_1": cues[0],
        "cue_2": cues[1],
        "cue_3": cues[2],
        "cue_4": cues[3],
        "difficulty": payload.get("difficulty", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not AB_TSV.exists():
        print(f"ERROR: {AB_TSV} missing; run 16_1 first.", file=sys.stderr)
        return 1

    items = _read_unique_texts()
    if args.limit and len(items) > args.limit:
        items = items[:args.limit]

    print(f"=== Stage 16.5 — EP/BP listening cues ===")
    print(f"  unique texts: {len(items)} ({sum(1 for _,ct in items if ct=='word')} word + "
          f"{sum(1 for _,ct in items if ct=='example')} example)")
    print(f"  cost (est):   ~${len(items) * 0.003:.2f}  (Claude Sonnet 4.6 with prompt cache)")
    print(f"  output:       {OUT_TSV}")
    print(f"  audit:        {AUDIT_JSONL}")
    print()

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to generate cues: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled."); return 0

    import os
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip())

    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    client = AnthropicClient(audit_path=AUDIT_JSONL)

    rows: list[dict] = []
    n_err = 0

    def _worker(item):
        text, ct = item
        try:
            payload = _run_one(client, text, ct)
            return text, ct, payload, None
        except Exception as e:  # noqa: BLE001
            return text, ct, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(_worker, it) for it in items]
        for i, fut in enumerate(as_completed(futs), start=1):
            text, ct, payload, err = fut.result()
            if err:
                n_err += 1
                print(f"  [{i:>3}/{len(items)}] FAIL {ct} {text[:50]!r}: {err[:80]}",
                      file=sys.stderr)
                continue
            row = _to_row(text, ct, payload)
            rows.append(row)
            print(f"  [{i:>3}/{len(items)}] OK   {ct:<7} "
                  f"diff={row['difficulty']:<9} {text[:60]}", flush=True)

    # Merge with any existing cues so we keep prior runs intact.
    existing = []
    if OUT_TSV.exists():
        existing = list(csv.DictReader(OUT_TSV.open(encoding="utf-8"),
                                       dialect="excel-tab"))
    by_key = {(r["text"], r["clip_type"]): r for r in existing}
    for r in rows:
        by_key[(r["text"], r["clip_type"])] = r
    merged = sorted(by_key.values(), key=lambda r: (r["clip_type"], r["text"]))
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in merged:
            w.writerow(r)

    print(f"\n  OK: {len(rows)} new + {len(existing)} kept = {len(merged)} total  FAIL: {n_err}")
    print(f"  wrote: {OUT_TSV} ({len(merged)} rows)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
