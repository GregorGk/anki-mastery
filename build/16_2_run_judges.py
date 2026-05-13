"""Stage 16 / Step 2 — Run 3 audio judges over the AB pool.

Reads `data/_audio_judge_ab.tsv`, downloads each MP3 from R2, and runs:

  1. OpenAI `gpt-4o-audio-preview`   (legacy baseline)
  2. OpenAI `gpt-audio-1.5`           (latest GA audio model — successor to #1)
  3. Google `gemini-3.1-pro-preview`  (cross-provider competitor)

Each judge writes a per-row record to its own audit JSONL:

    audit/16_judge_gpt4o.jsonl
    audit/16_judge_gpt_audio_15.jsonl
    audit/16_judge_gemini_31_pro.jsonl

Same JSON tool-output schema as `build/lib/audio_judge.py`. The combined
result is merged into `audit/16_judge_verdicts.jsonl` for the HTML +
scorer to consume.

Cost (~$1.65 for 200 clips × 3 judges):
    gpt-4o-audio-preview      ~$1.00
    gpt-audio-1.5             ~$0.50  (est.)
    gemini-3.1-pro-preview    ~$0.06  (est.)

Usage:
    .venv/bin/python build/16_2_run_judges.py --dry-run
    .venv/bin/python build/16_2_run_judges.py --yes
    .venv/bin/python build/16_2_run_judges.py --yes --judges gpt4o,gpt_audio_15,gemini_31_pro
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_judge import AudioJudgeClient  # noqa: E402
from build.lib.gemini_audio_judge import GeminiAudioJudgeClient  # noqa: E402

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"

INPUT_TSV = DATA / "_audio_judge_ab.tsv"
MERGED_OUT = AUDIT / "16_judge_verdicts.jsonl"

JUDGES = {
    "gpt4o": {
        "client": "openai",
        "model": "gpt-4o-audio-preview",
        "audit_jsonl": AUDIT / "16_judge_gpt4o.jsonl",
        "label": "OpenAI gpt-4o-audio-preview",
    },
    "gpt_audio_15": {
        "client": "openai",
        "model": "gpt-audio-1.5",
        "audit_jsonl": AUDIT / "16_judge_gpt_audio_15.jsonl",
        "label": "OpenAI gpt-audio-1.5",
    },
    "gemini_31_pro": {
        "client": "gemini",
        "model": "gemini-3.1-pro-preview",
        "audit_jsonl": AUDIT / "16_judge_gemini_31_pro.jsonl",
        "label": "Google gemini-3.1-pro-preview",
    },
}

# Default outer concurrency. The post-2026-05 tier limits comfortably
# support 25 rows in flight, each fanning the 3 judges in parallel:
#   - Gemini 3.1 Pro:  1,000 RPM / 50,000 RPD
#   - gpt-audio-1.5:   3,000 RPM / 250k TPM
#   - gpt-4o-audio-pv: 10,000 RPM / 2M TPM
# 25 rows × 3 judges = 75 concurrent calls; at ~9s per Gemini call that's
# ~165 RPM per judge — well under the smallest cap (1k RPM Gemini).
DEFAULT_WORKERS = 25
DEFAULT_INNER_JUDGES_PARALLEL = True


def _fetch_audio(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _load_pool() -> list[dict]:
    rows = list(csv.DictReader(INPUT_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    if not rows:
        raise RuntimeError(f"{INPUT_TSV} is empty; run build/16_1_audio_judge_ab_prep.py first")
    return rows


def _already_judged(judges_to_run: list[str]) -> set[tuple[str, str, str, str]]:
    """Return {(sense_id, clip_type, voice_id, judge_name)} keys already
    in per-judge audit JSONLs. Used by --skip-done to make re-runs additive."""
    done: set[tuple[str, str, str, str]] = set()
    for j in judges_to_run:
        path = JUDGES[j]["audit_jsonl"]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sense_id")
            ct = rec.get("clip_type")
            vid = rec.get("voice_id")
            if sid and ct is not None and vid is not None:
                done.add((sid, ct, vid, j))
    return done


def _judge_one(*, row: dict, judge_name: str, judge_cfg: dict,
               audio_bytes: bytes) -> dict:
    """Call one judge on one clip; return a unified verdict dict."""
    if judge_cfg["client"] == "openai":
        client = AudioJudgeClient(
            model=judge_cfg["model"],
            audit_path=judge_cfg["audit_jsonl"],
        )
        r = client.judge(
            audio_bytes=audio_bytes, audio_format="mp3",
            pt=row.get("text", ""),
            ipa_word_final=row.get("ipa_word_final", ""),
            voice_id=row["voice_id"],
            sense_id=row["sense_id"],
            clip_type=row.get("clip_type", "word"),
        )
        return {
            "row_id": row["row_id"], "judge": judge_name,
            "model": r.model,
            "verdict": r.pronunciation_verdict, "drift": r.drift,
            "severity": r.severity, "confidence": r.confidence,
            "evidence": r.evidence,
            "cost_usd": r.cost_usd, "latency_ms": r.latency_ms,
        }
    elif judge_cfg["client"] == "gemini":
        client = GeminiAudioJudgeClient(
            model=judge_cfg["model"],
            audit_path=judge_cfg["audit_jsonl"],
        )
        r = client.judge(
            audio_bytes=audio_bytes, audio_format="mp3",
            pt=row.get("text", ""),
            ipa_word_final=row.get("ipa_word_final", ""),
            voice_id=row["voice_id"],
            sense_id=row["sense_id"],
            clip_type=row.get("clip_type", "word"),
        )
        return {
            "row_id": row["row_id"], "judge": judge_name,
            "model": r.model,
            "verdict": r.pronunciation_verdict, "drift": r.drift,
            "severity": r.severity, "confidence": r.confidence,
            "evidence": r.evidence,
            "cost_usd": r.cost_usd, "latency_ms": r.latency_ms,
        }
    raise ValueError(f"unknown judge client: {judge_cfg['client']}")


def _process_row(row: dict, judges_to_run: list[str],
                 already_done: set[tuple[str, str, str, str]] | None = None
                 ) -> list[dict]:
    """Fetch audio once, then fan out the 3 judges IN PARALLEL via an inner
    ThreadPool. Skip (row, judge) pairs listed in `already_done`.

    Inner parallel means each row's wall time = max(judge latencies)
    instead of sum(judge latencies). With Gemini at ~9 s and OpenAI at
    ~2 s, that's a ~30% per-row speedup vs sequential.
    """
    # Filter the judge list per-row.
    pending = [j for j in judges_to_run
               if already_done is None
               or (row["sense_id"], row["clip_type"], row["voice_id"], j) not in already_done]
    if not pending:
        return []
    try:
        audio_bytes = _fetch_audio(row["url"])
    except Exception as exc:  # noqa: BLE001
        return [{
            "row_id": row["row_id"], "judge": j, "error": f"fetch: {type(exc).__name__}: {exc}",
        } for j in pending]

    def _one(jname: str) -> dict:
        try:
            return _judge_one(row=row, judge_name=jname,
                              judge_cfg=JUDGES[jname], audio_bytes=audio_bytes)
        except Exception as exc:  # noqa: BLE001
            return {
                "row_id": row["row_id"], "judge": jname,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if DEFAULT_INNER_JUDGES_PARALLEL and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=min(3, len(pending))) as inner:
            return list(inner.map(_one, pending))
    else:
        return [_one(j) for j in pending]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--judges", default="gpt4o,gpt_audio_15,gemini_31_pro",
                    help="Comma-separated judge keys: " + ",".join(JUDGES))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only N rows (smoke test).")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip (row, judge) pairs already in per-judge audit JSONLs. "
                         "Use after a pool expansion (16_6) to re-run only new rows.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Outer ThreadPool size over rows (default {DEFAULT_WORKERS}). "
                         f"Each row also fans 3 judges in parallel inner workers.")
    args = ap.parse_args()

    judges_to_run = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges_to_run:
        if j not in JUDGES:
            print(f"ERROR: unknown judge {j!r}; valid: {','.join(JUDGES)}", file=sys.stderr)
            return 1

    # Load .env for credentials.
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip())

    # Sanity-check creds.
    for j in judges_to_run:
        if JUDGES[j]["client"] == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set", file=sys.stderr); return 1
        if JUDGES[j]["client"] == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); return 1

    pool = _load_pool()
    if args.limit and len(pool) > args.limit:
        pool = pool[:args.limit]

    already_done = _already_judged(judges_to_run) if args.skip_done else set()
    if args.skip_done:
        before = sum(len(judges_to_run) for _ in pool)
        skipped = sum(
            1 for r in pool for j in judges_to_run
            if (r["sense_id"], r["clip_type"], r["voice_id"], j) in already_done
        )
        print(f"  --skip-done active: {skipped:,} of {before:,} (row, judge) pairs already audited")

    print(f"=== Stage 16.2 audio-judge A/B run ===")
    print(f"  pool size:      {len(pool)} clips")
    print(f"  judges:         {', '.join(JUDGES[j]['label'] for j in judges_to_run)}")
    print(f"  concurrency:    {args.workers} outer × 3 inner (judges parallel)")
    print(f"  per-judge audit: " + ", ".join(
        str(JUDGES[j]['audit_jsonl'].name) for j in judges_to_run))
    print(f"  merged out:     {MERGED_OUT.name}")
    print()
    if args.dry_run:
        print("--dry-run: stopping.")
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to run the judges: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled."); return 0

    AUDIT.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    all_verdicts: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futs = [pool_exec.submit(_process_row, row, judges_to_run, already_done)
                for row in pool]
        for i, fut in enumerate(as_completed(futs), start=1):
            verdicts = fut.result()
            for v in verdicts:
                all_verdicts.append(v)
                if "error" in v:
                    n_err += 1
                    print(f"  [{i:>3}/{len(pool)}] FAIL {v['row_id']} {v['judge']:<14} {v['error'][:60]}",
                          file=sys.stderr)
                else:
                    n_ok += 1
                    print(f"  [{i:>3}/{len(pool)}] OK   {v['row_id']} {v['judge']:<14} "
                          f"{v['verdict']:<7} drift={v['drift']:<6} conf={v['confidence']}",
                          flush=True)

    with MERGED_OUT.open("w", encoding="utf-8") as f:
        for v in all_verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"\n  OK: {n_ok}  FAIL: {n_err}")
    print(f"  merged verdicts: {MERGED_OUT} ({len(all_verdicts)} records)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
