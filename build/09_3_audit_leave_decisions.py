"""Stage 9 / Step 3 — Audio-judge the LEAVE decisions to confirm BP pronunciation.

The 77 clips the user marked LEAVE in the review queue had low ASR
similarity but the user (after listening) said the audio sounded BP.
Audio judge (gpt-4o-audio-preview) gives an independent verdict:
"is this clip's pronunciation actually BP, or is it drifted?".

For each LEAVE clip whose URL is current in the manifest:
  - Fetch the local cache audio (built from manifest URL)
  - Call AudioJudgeClient.judge()
  - If bp_ok → record in `_audio_asr_override.tsv` as
    "ASR-untranscribable, audio verified BP"
  - If non_bp/unclear → leave in queue for later iteration

Downstream tooling can filter the human-review queue against the
override file to count "actually problematic" clips (which should be
near zero).

Usage:
    .venv/bin/python build/09_3_audit_leave_decisions.py [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_judge import AudioJudgeClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

DECISIONS_PATH = DATA_DIR / "_audio_review_queue_decisions.tsv"
MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
OVERRIDE_PATH = DATA_DIR / "_audio_asr_override.tsv"
AUDIT_JSONL = AUDIT_DIR / "09_3_audit_leave_decisions.jsonl"

OVERRIDE_FIELDS = [
    "sense_id", "clip_type", "pt", "voice_id", "asr_status",
    "audio_judge_verdict", "audio_judge_drift", "audio_judge_severity",
    "audio_judge_confidence", "audio_judge_evidence", "judged_at",
]


def _split_sid_cliptype(merged: str) -> tuple[str, str]:
    """Parse '2043.00.01word' → ('2043.00.01', 'word')."""
    merged = merged.strip()
    if len(merged) < 10:
        return (merged, "")
    sid = merged[:10]
    rest = merged[10:]
    if rest in ("word", "example"):
        return (sid, rest)
    return (merged, "")


def _cache_path_from_url(url: str) -> Path:
    """Derive local cache path from manifest URL's trailing filename."""
    filename = url.rsplit("/", 1)[-1]
    return AUDIO_CACHE / filename


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N judge calls (0 = no limit).")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel audio-judge calls.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 1

    # Read LEAVE decisions
    decisions = read_tsv(DECISIONS_PATH)
    leave_keys: list[tuple[str, str]] = []
    for r in decisions:
        merged = r.get("sense_id", "")
        sid, ctype = _split_sid_cliptype(merged)
        if sid and ctype and r.get("decision") == "LEAVE":
            leave_keys.append((sid, ctype))
    print(f"Found {len(leave_keys)} LEAVE decisions to audit.")

    # Cross-reference with manifest to get current URLs + IPA
    manifest = read_tsv(MANIFEST_PATH)
    mf_idx = {(r["sense_id"], r["clip_type"]): r for r in manifest}
    ipa_idx = {r["sense_id"]: r for r in read_tsv(IPA_PATH)}

    to_audit: list[dict] = []
    for sid, ctype in leave_keys:
        m = mf_idx.get((sid, ctype))
        if not m:
            print(f"  WARN: ({sid}, {ctype}) not in manifest, skipping", file=sys.stderr)
            continue
        cache = _cache_path_from_url(m.get("url", ""))
        if not cache.exists():
            print(f"  WARN: cache missing for {sid} {ctype}: {cache}", file=sys.stderr)
            continue
        ipa_row = ipa_idx.get(sid, {})
        to_audit.append({
            "sense_id": sid, "clip_type": ctype,
            "pt": m.get("text_input", ""),
            "voice_id": m.get("voice_id", ""),
            "ipa_word_final": ipa_row.get("ipa_word_final", "") or ipa_row.get("ipa", ""),
            "cache_path": cache,
            "url": m.get("url", ""),
        })

    if args.limit and args.limit > 0:
        to_audit = to_audit[: args.limit]

    if not to_audit:
        print("Nothing to audit.")
        return 0

    print(f"Audio-judging {len(to_audit)} clips @ concurrency={args.concurrency}...")
    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    judge = AudioJudgeClient(audit_path=AUDIT_JSONL)

    results: dict[tuple[str, str], dict] = {}
    failures: list[tuple[str, str, str]] = []

    def _work(row: dict) -> tuple[tuple[str, str], dict | None, str | None]:
        key = (row["sense_id"], row["clip_type"])
        try:
            audio_bytes = row["cache_path"].read_bytes()
            res = judge.judge(
                audio_bytes=audio_bytes, audio_format="mp3",
                pt=row["pt"], ipa_word_final=row["ipa_word_final"],
                voice_id=row["voice_id"],
                sense_id=row["sense_id"], clip_type=row["clip_type"],
            )
            return (key, {
                "verdict": res.pronunciation_verdict, "drift": res.drift,
                "severity": res.severity, "confidence": res.confidence,
                "evidence": res.evidence,
            }, None)
        except Exception as exc:  # noqa: BLE001
            return (key, None, f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_work, r) for r in to_audit]
        done = 0
        for fut in as_completed(futures):
            key, verdict, err = fut.result()
            done += 1
            if err:
                failures.append((key[0], key[1], err))
                print(f"  [{done}/{len(to_audit)}] FAIL {key[0]} {key[1]}: {err}", file=sys.stderr)
            else:
                results[key] = verdict  # type: ignore[assignment]
                if done % 10 == 0 or done == len(to_audit):
                    print(f"  [{done}/{len(to_audit)}] judged; bp_ok so far: "
                          f"{sum(1 for v in results.values() if v['verdict'] == 'bp_ok')}")

    # Summarize
    n_bp_ok = sum(1 for v in results.values() if v["verdict"] == "bp_ok")
    n_non_bp = sum(1 for v in results.values() if v["verdict"] == "non_bp")
    n_unclear = sum(1 for v in results.values() if v["verdict"] == "unclear")
    print()
    print("=== Audit summary ===")
    print(f"  total audited:  {len(results)}")
    print(f"  bp_ok:          {n_bp_ok} ({n_bp_ok/max(1,len(results))*100:.1f}%)")
    print(f"  non_bp:         {n_non_bp}")
    print(f"  unclear:        {n_unclear}")
    print(f"  failed:         {len(failures)}")
    print(f"  judge cost:     ${judge.stats['total_cost_usd']:.4f}")

    # Write override TSV for bp_ok ones
    override_rows: list[dict] = []
    if OVERRIDE_PATH.exists():
        # Merge with existing override (preserve prior entries)
        existing = read_tsv(OVERRIDE_PATH)
        override_rows.extend(existing)
        existing_keys = {(r["sense_id"], r["clip_type"]) for r in existing}
    else:
        existing_keys = set()

    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    for row in to_audit:
        key = (row["sense_id"], row["clip_type"])
        verdict = results.get(key)
        if verdict is None or verdict["verdict"] != "bp_ok":
            continue
        if key in existing_keys:
            continue  # already in override
        override_rows.append({
            "sense_id": row["sense_id"], "clip_type": row["clip_type"],
            "pt": row["pt"], "voice_id": row["voice_id"],
            "asr_status": "untranscribable_audio_verified_bp",
            "audio_judge_verdict": verdict["verdict"],
            "audio_judge_drift": verdict["drift"],
            "audio_judge_severity": verdict["severity"],
            "audio_judge_confidence": verdict["confidence"],
            "audio_judge_evidence": verdict["evidence"],
            "judged_at": now,
        })

    write_tsv(OVERRIDE_PATH, override_rows, fieldnames=OVERRIDE_FIELDS)
    print()
    print(f"Wrote {OVERRIDE_PATH} ({len(override_rows)} total rows; "
          f"{n_bp_ok} new this run).")

    # Effective ASR-ok rate
    total = len(manifest)
    flagged_current = 0
    current_urls = {r["url"] for r in manifest if r.get("url")}
    for r in read_tsv(DATA_DIR / "_audio_human_review.tsv"):
        if "flash_v2" in r.get("url", "") and r.get("url") in current_urls:
            flagged_current += 1
    override_keys = {(r["sense_id"], r["clip_type"]) for r in override_rows}
    effective_flagged = flagged_current - sum(
        1 for r in read_tsv(DATA_DIR / "_audio_human_review.tsv")
        if "flash_v2" in r.get("url", "")
        and r.get("url") in current_urls
        and (r.get("sense_id"), r.get("clip_type")) in override_keys
    )
    print()
    print(f"=== Effective ASR-ok rate ===")
    print(f"  Total clips:                       {total}")
    print(f"  ASR-flagged (raw):                 {flagged_current} ({flagged_current/total*100:.2f}%)")
    print(f"  Verified BP via audio judge:       {len(override_keys)}")
    print(f"  Genuinely problematic (residual):  {effective_flagged} ({effective_flagged/total*100:.2f}%)")
    print(f"  Audio-verified BP rate:            {(total-effective_flagged)/total*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
