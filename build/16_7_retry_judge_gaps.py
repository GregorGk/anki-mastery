"""Stage 16 / Step 7 — Retry missing (row, judge) pairs serially.

After 16_2 saturated the Gemini Tier-1 quota window, ~156 rows have no
Gemini verdict and ~5 rows have no gpt-audio-1.5 verdict (the latter
from rare malformed tool-call JSON). This script:

  1. Loads the AB pool + per-judge audit JSONLs.
  2. Identifies (row, judge) pairs still missing.
  3. Retries them ONE AT A TIME with an explicit `--sleep-between` wait
     (defaults to 10 s for Gemini to keep us safely under the tier-1
     rolling-quota window; OpenAI judges go faster).
  4. Appends fresh records to the same per-judge audit JSONLs.

Usage:
    .venv/bin/python build/16_7_retry_judge_gaps.py --dry-run
    .venv/bin/python build/16_7_retry_judge_gaps.py --yes
    .venv/bin/python build/16_7_retry_judge_gaps.py --yes --judges gemini_31_pro --sleep-between 12
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_judge import AudioJudgeClient  # noqa: E402
from build.lib.gemini_audio_judge import GeminiAudioJudgeClient  # noqa: E402

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
INPUT_TSV = DATA / "_audio_judge_ab.tsv"

JUDGES = {
    "gpt4o": {
        "client": "openai", "model": "gpt-4o-audio-preview",
        "audit_jsonl": AUDIT / "16_judge_gpt4o.jsonl",
    },
    "gpt_audio_15": {
        "client": "openai", "model": "gpt-audio-1.5",
        "audit_jsonl": AUDIT / "16_judge_gpt_audio_15.jsonl",
    },
    "gemini_31_pro": {
        "client": "gemini", "model": "gemini-3.1-pro-preview",
        "audit_jsonl": AUDIT / "16_judge_gemini_31_pro.jsonl",
    },
}


def _fetch_audio(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _audited_keys(judge_key: str) -> set[tuple[str, str, str]]:
    """Return {(sense_id, clip_type, voice_id)} already in the audit JSONL."""
    path = JUDGES[judge_key]["audit_jsonl"]
    if not path.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.add((r.get("sense_id", ""), r.get("clip_type", ""), r.get("voice_id", "")))
    return out


def _judge_one(*, row: dict, judge_name: str, audio_bytes: bytes) -> None:
    """Call the judge — the wrappers append their own audit record."""
    cfg = JUDGES[judge_name]
    if cfg["client"] == "openai":
        client = AudioJudgeClient(model=cfg["model"], audit_path=cfg["audit_jsonl"])
    else:
        client = GeminiAudioJudgeClient(model=cfg["model"], audit_path=cfg["audit_jsonl"])
    client.judge(
        audio_bytes=audio_bytes, audio_format="mp3",
        pt=row.get("text", ""), ipa_word_final="",
        voice_id=row["voice_id"],
        sense_id=row["sense_id"], clip_type=row.get("clip_type", "word"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--judges", default="gemini_31_pro,gpt_audio_15",
                    help="Which judges to retry (comma-separated).")
    ap.add_argument("--sleep-between", type=float, default=8.0,
                    help="Seconds to sleep between calls (per-judge). Gemini tier-1 "
                         "quota recovers over ~1 min; 8 s gives ~7 RPM headroom.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N gap calls.")
    args = ap.parse_args()

    judges_to_run = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges_to_run:
        if j not in JUDGES:
            print(f"ERROR: unknown judge {j!r}; valid: {','.join(JUDGES)}", file=sys.stderr)
            return 1

    # Load .env credentials.
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip())

    # Build the gap list: (row, judge) where judge hasn't audited row yet.
    rows = list(csv.DictReader(INPUT_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    audited_per_judge = {j: _audited_keys(j) for j in judges_to_run}

    gaps: list[tuple[dict, str]] = []
    for j in judges_to_run:
        done = audited_per_judge[j]
        for row in rows:
            key = (row["sense_id"], row["clip_type"], row["voice_id"])
            if key not in done:
                gaps.append((row, j))

    if args.limit and len(gaps) > args.limit:
        gaps = gaps[:args.limit]

    print(f"=== Stage 16.7 retry gaps ===")
    print(f"  AB pool size:       {len(rows)}")
    print(f"  judges to retry:    {', '.join(judges_to_run)}")
    print(f"  per-judge done:     " + ", ".join(
        f"{j}={len(audited_per_judge[j])}" for j in judges_to_run))
    print(f"  gaps to retry:      {len(gaps)}")
    print(f"  sleep between:      {args.sleep_between} s")
    est_min = len(gaps) * (args.sleep_between + 9) / 60
    print(f"  est wall:           {est_min:.1f} min")
    print()

    if args.dry_run or not gaps:
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to retry the gaps: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled."); return 0

    n_ok = n_err = 0
    n_429 = 0
    t_start = time.time()
    for i, (row, j) in enumerate(gaps, start=1):
        # Sleep between calls to honor rolling quota window.
        if i > 1:
            time.sleep(args.sleep_between)
        try:
            audio_bytes = _fetch_audio(row["url"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:>3}/{len(gaps)}] FETCH-FAIL {row['row_id']} {j}: {exc!s:.80}",
                  file=sys.stderr)
            n_err += 1
            continue
        try:
            _judge_one(row=row, judge_name=j, audio_bytes=audio_bytes)
            n_ok += 1
            print(f"  [{i:>3}/{len(gaps)}] OK   {row['row_id']:<7} {j:<14}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:80]
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
                n_429 += 1
            n_err += 1
            print(f"  [{i:>3}/{len(gaps)}] FAIL {row['row_id']:<7} {j:<14} {msg}",
                  file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s — OK={n_ok}, FAIL={n_err} (429s: {n_429})")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
