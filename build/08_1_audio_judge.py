"""Stage 8 / Step 1 — Audio LLM judge.

Calls `gpt-4o-audio-preview` via build.lib.audio_judge on word clips and
records a structured verdict per (sense_id, clip_type=word) row.

Default scope: **full corpus** (~6,250 word clips, ~$32 — user-locked).
Use `--bucket=risky` to limit to P0+P1+P2+P3 from
`data/_audio_risk_classification.tsv` (~2,088 clips, ~$10).

Smoke-test mode (`--smoke-test --n 3`) picks 3 representative clips
(one user-reported + one cognate + one clean control), runs them,
prints results, exits — used to verify the gpt-4o-audio-preview API
works as documented before the bulk run.

Outputs:
    audit/08_1_audio_judge.jsonl   (per-clip lifecycle events)
    data/_audio_judge_verdicts.tsv (one row per judged clip)

Idempotent: re-runs skip sense_ids already in the verdicts TSV unless
`--rejudge` is passed.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_judge import AudioJudgeClient, JudgeResult  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

RISK_CLASSIFICATION_PATH = DATA_DIR / "_audio_risk_classification.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
VERDICTS_PATH = DATA_DIR / "_audio_judge_verdicts.tsv"
AUDIT_JSONL = AUDIT_DIR / "08_1_audio_judge.jsonl"

VERDICT_COLUMNS = [
    "sense_id",
    "pt",
    "voice_id",
    "pronunciation_verdict",
    "drift",
    "severity",
    "confidence",
    "evidence",
    "model",
    "latency_ms",
    "cost_usd",
    "judged_at",
]


def _fetch_audio_bytes(sense_id: str, version: str, clip_type: str, url: str) -> bytes:
    """Read MP3 bytes for a clip. Local cache first, R2 URL fallback."""
    cache_path = AUDIO_CACHE / f"{sense_id}-{clip_type}-v{version}.mp3"
    if cache_path.exists():
        return cache_path.read_bytes()
    if not url:
        raise RuntimeError(f"no cached audio AND no url for {sense_id}/{clip_type}")
    # Fetch from R2 (read-only HTTP GET)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    # Cache for next call
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def _load_existing_verdicts() -> set[str]:
    """Return set of sense_ids already judged (for idempotent resume)."""
    if not VERDICTS_PATH.exists():
        return set()
    rows = read_tsv(VERDICTS_PATH)
    return {r["sense_id"] for r in rows if r.get("sense_id")}


def _append_verdict_row(row: dict, lock: Lock) -> None:
    """Append a single row to the verdicts TSV; create file with header if missing."""
    with lock:
        write_header = not VERDICTS_PATH.exists()
        with VERDICTS_PATH.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=VERDICT_COLUMNS, dialect="excel-tab",
                quoting=csv.QUOTE_MINIMAL, extrasaction="ignore",
            )
            if write_header:
                w.writeheader()
            w.writerow(row)


def _build_work_queue(
    *, bucket: str, smoke_test: bool, smoke_n: int, rejudge: bool
) -> list[dict]:
    """Build the list of word-clip rows to judge.

    Returns list of dicts with: sense_id, pt, voice_id, ipa_word_final,
    version, url, clip_type='word', priority.
    """
    risk_rows = read_tsv(RISK_CLASSIFICATION_PATH) if RISK_CLASSIFICATION_PATH.exists() else []
    manifest_rows = read_tsv(MANIFEST_PATH)
    ipa_rows = read_tsv(IPA_PATH)

    risk_by_sid = {r["sense_id"]: r for r in risk_rows}
    ipa_by_sid = {r["sense_id"]: r for r in ipa_rows}
    manifest_word = [
        r for r in manifest_rows
        if r.get("clip_type") == "word" and r.get("status") == "uploaded"
    ]

    # Build per-clip work items
    items: list[dict] = []
    for r in manifest_word:
        sid = r["sense_id"]
        risk = risk_by_sid.get(sid, {})
        ipa = ipa_by_sid.get(sid, {})
        items.append({
            "sense_id": sid,
            "pt": r.get("text_input") or risk.get("pt", ""),
            "voice_id": r.get("voice_id", ""),
            "ipa_word_final": ipa.get("ipa_word_final", ""),
            "version": r.get("version", "1"),
            "url": r.get("url", ""),
            "clip_type": "word",
            "priority": risk.get("priority", "—"),
            "risk_patterns": risk.get("risk_patterns", "none"),
        })

    # Filter by bucket
    if bucket == "risky":
        items = [i for i in items if i["priority"] in ("P0", "P1", "P2", "P3")]
    # bucket == "full" or "all": no filter

    # Filter out already-judged
    if not rejudge:
        existing = _load_existing_verdicts()
        items = [i for i in items if i["sense_id"] not in existing]

    if smoke_test:
        # Pick: 1 user-reported P0, 1 cognate (final_l_vocalization), 1 clean (no patterns)
        smoke: list[dict] = []
        seen_categories: set[str] = set()
        for i in items:
            if "user_reported" in i["risk_patterns"] and "user_reported" not in seen_categories:
                smoke.append(i); seen_categories.add("user_reported")
            elif "final_l_vocalization" in i["risk_patterns"] and "final_l_vocalization" not in seen_categories:
                smoke.append(i); seen_categories.add("final_l_vocalization")
            elif i["risk_patterns"] == "none" and "clean" not in seen_categories:
                smoke.append(i); seen_categories.add("clean")
            if len(smoke) >= smoke_n:
                break
        items = smoke[:smoke_n]

    return items


def _judge_one(client: AudioJudgeClient, item: dict, write_lock: Lock) -> tuple[str, JudgeResult | Exception]:
    sense_id = item["sense_id"]
    try:
        audio_bytes = _fetch_audio_bytes(
            sense_id, str(item["version"]), item["clip_type"], item["url"]
        )
        result = client.judge(
            audio_bytes=audio_bytes,
            audio_format="mp3",
            pt=item["pt"],
            ipa_word_final=item["ipa_word_final"],
            voice_id=item["voice_id"],
            sense_id=sense_id,
            clip_type=item["clip_type"],
        )
        # Append to verdicts TSV
        verdict_row = {
            "sense_id": sense_id,
            "pt": item["pt"],
            "voice_id": item["voice_id"],
            "pronunciation_verdict": result.pronunciation_verdict,
            "drift": result.drift,
            "severity": result.severity,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "cost_usd": round(result.cost_usd, 6),
            "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
        _append_verdict_row(verdict_row, write_lock)
        return sense_id, result
    except Exception as exc:  # noqa: BLE001
        return sense_id, exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bucket",
        choices=("full", "risky"),
        default="full",
        help="Default 'full' = all word clips (~6,250). 'risky' = P0+P1+P2+P3 (~2,088).",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Pick 3 representative clips, run, exit.")
    parser.add_argument("--n", type=int, default=3, help="Number of smoke-test clips.")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel judge calls.")
    parser.add_argument("--rejudge", action="store_true", help="Re-judge already-judged sense_ids.")
    parser.add_argument("--confirm", action="store_true", help="Required for non-smoke runs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many calls (0 = no limit). Useful for cost-bounded testing.",
    )
    args = parser.parse_args()

    if not args.smoke_test and not args.confirm:
        parser.error("non-smoke runs require --confirm. Re-run with --smoke-test or --confirm.")

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY not set in environment.")

    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    items = _build_work_queue(
        bucket=args.bucket,
        smoke_test=args.smoke_test,
        smoke_n=args.n,
        rejudge=args.rejudge,
    )
    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        print("Nothing to judge (empty work queue). Exiting.")
        return 0

    # Cost preview
    est_cost = len(items) * 0.012  # rough estimate per clip
    print(f"Audio judge — {len(items)} clip(s) queued (bucket={args.bucket}, smoke={args.smoke_test}).")
    print(f"  estimated cost (rough):  ~${est_cost:.2f}")
    print(f"  audit log:                {AUDIT_JSONL}")
    print(f"  verdicts file:            {VERDICTS_PATH}")
    print(f"  concurrency:              {args.concurrency}")
    print()

    client = AudioJudgeClient(audit_path=AUDIT_JSONL)
    write_lock = Lock()

    started_at = time.time()
    done = 0
    failed = 0
    verdict_dist: dict[str, int] = {"bp_ok": 0, "non_bp": 0, "unclear": 0}
    drift_dist: dict[str, int] = {}

    if args.concurrency == 1:
        for item in items:
            sense_id, result = _judge_one(client, item, write_lock)
            done += 1
            if isinstance(result, Exception):
                failed += 1
                print(f"  [FAIL] {sense_id}: {type(result).__name__}: {result}", file=sys.stderr)
            else:
                verdict_dist[result.pronunciation_verdict] = verdict_dist.get(result.pronunciation_verdict, 0) + 1
                drift_dist[result.drift] = drift_dist.get(result.drift, 0) + 1
                if args.smoke_test or done <= 5 or done % 50 == 0:
                    elapsed = time.time() - started_at
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  [{done:>4}/{len(items)}] {sense_id} '{item['pt']}' "
                        f"voice={item['voice_id'][:12]} → "
                        f"{result.pronunciation_verdict} (drift={result.drift}, "
                        f"sev={result.severity}, conf={result.confidence}) "
                        f"[{result.latency_ms}ms, ${result.cost_usd:.4f}]"
                    )
                    print(f"        evidence: {result.evidence}")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(_judge_one, client, i, write_lock): i for i in items}
            for fut in as_completed(futures):
                sense_id, result = fut.result()
                done += 1
                if isinstance(result, Exception):
                    failed += 1
                    print(f"  [FAIL] {sense_id}: {type(result).__name__}: {result}", file=sys.stderr)
                    continue
                verdict_dist[result.pronunciation_verdict] = verdict_dist.get(result.pronunciation_verdict, 0) + 1
                drift_dist[result.drift] = drift_dist.get(result.drift, 0) + 1
                if done <= 5 or done % 50 == 0:
                    elapsed = time.time() - started_at
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_s = (len(items) - done) / rate if rate > 0 else 0.0
                    s = client.stats
                    print(
                        f"  [{done:>4}/{len(items)} ({100*done/len(items):.1f}%)] "
                        f"rate={rate:.2f}/s ETA={eta_s/60:.1f}min "
                        f"spent=${s['total_cost_usd']:.2f}"
                    )

    elapsed = time.time() - started_at
    s = client.stats
    print()
    print(f"=== Audio judge summary ===")
    print(f"  processed:   {done}")
    print(f"  failed:      {failed}")
    print(f"  wall:        {elapsed/60:.1f} min")
    print(f"  cost actual: ${s['total_cost_usd']:.4f}")
    print(f"  audio tokens: {s['audio_tokens']:,}  cached_text: {s['cached_text_tokens']:,}  uncached_text: {s['uncached_text_tokens']:,}  output: {s['output_tokens']:,}")
    print()
    print("Verdict distribution:")
    for v in ("bp_ok", "non_bp", "unclear"):
        print(f"  {v:<10} {verdict_dist.get(v, 0):>5}")
    print()
    if drift_dist:
        print("Drift distribution (non_bp + unclear):")
        for d in ("EN", "EP", "ES", "FR", "other", "none"):
            if d in drift_dist:
                print(f"  {d:<6} {drift_dist[d]:>5}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
