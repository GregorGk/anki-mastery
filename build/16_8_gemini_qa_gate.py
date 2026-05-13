"""Stage 16 / Step 8 — Gemini-only QA gate over all 11,449 BP-voiced v3 clips.

Runs gemini-3.1-pro-preview on every BP word + example clip in the
production deck, producing a deck-wide drift watchlist (sense_ids
where Gemini believes a BP-assigned voice sounds non-BP).

After the 800-clip A/B run + 40 human labels, Gemini hit:
  92.5 % agreement with the human, 100 % recall on non_bp, 96.7 % precision.
We treat it as the production audio-judge.

Pipeline per row:
  Fetch R2 audio → GeminiAudioJudgeClient.judge() → append to
  audit/16_8_gemini_qa_gate.jsonl (via the wrapper's audit_path).

After the run, walks the merged Gemini audit and emits
data/_gemini_drift_watchlist.tsv with every non_bp verdict.

Tier headroom (Gemini Pro paid Tier @ 2026-05): 1,000 RPM / 5 M TPM.
At ~9 s mean latency and ~1,200 tokens per call:
  148 outer workers → ~987 RPM peak  (just under the 1k cap)
  SlidingWindowRateLimiter capped at 990 RPM as a hard safety in case
  Gemini speeds up transiently.

Usage:
    .venv/bin/python build/16_8_gemini_qa_gate.py --dry-run
    .venv/bin/python build/16_8_gemini_qa_gate.py --yes
    .venv/bin/python build/16_8_gemini_qa_gate.py --yes --workers 100 --max-rpm 600
    .venv/bin/python build/16_8_gemini_qa_gate.py --watchlist-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx  # connection-pooled HTTP for R2 audio fetches

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.gemini_audio_judge import GeminiAudioJudgeClient  # noqa: E402
from build.lib.rate_limit import SlidingWindowRateLimiter  # noqa: E402

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
CONFIG = REPO_ROOT / "config"
# Local mp3 cache populated by build/16_8_prefetch_audio.py. Reading
# from disk eliminates the R2 DNS-storm failure mode we saw at 148
# workers (macOS mDNSResponder overwhelmed by concurrent lookups).
LOCAL_AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

MANIFEST = DATA / "_audio_manifest.tsv"
FINAL_TSV = DATA / "06-final.tsv"
VOICES_TSV = CONFIG / "voices.tsv"

# Existing AB-pool Gemini verdicts (~400 BP entries) — re-use, don't redo.
AB_AUDIT = AUDIT / "16_judge_gemini_31_pro.jsonl"
# This run's audit (append-only).
OUT_AUDIT = AUDIT / "16_8_gemini_qa_gate.jsonl"
OUT_WATCHLIST = DATA / "_gemini_drift_watchlist.tsv"

MODEL = "gemini-3.1-pro-preview"
DEFAULT_WORKERS = 148           # → ~987 RPM at ~9s mean latency
DEFAULT_MAX_RPM = 990           # hard ceiling (1k cap minus 1 % buffer)
BP_CLIP_TYPES = ("word", "example")
V3_MODEL_ID = "eleven_v3"


def _load_text_map() -> dict[tuple[str, str], str]:
    """(sense_id, clip_type) → text from 06-final.tsv. Same logic as 16_1."""
    out: dict[tuple[str, str], str] = {}
    for r in csv.DictReader(FINAL_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        word_text = (r.get("pt_display") or r.get("pt", "")).strip()
        example_pt = (r.get("example_pt") or "").strip()
        out[(r["sense_id"], "word")] = word_text
        out[(r["sense_id"], "example")] = example_pt
    return out


def _load_voice_names() -> dict[str, str]:
    return {r["voice_id"]: r.get("bp_name", "") for r in
            csv.DictReader(VOICES_TSV.open(encoding="utf-8"), dialect="excel-tab")}


def _load_active_bp_voices() -> set[str]:
    """voice_ids with status == 'active' in config/voices.tsv. Used by
    the watchlist deriver to filter out EP control clips (e.g. Nelson
    Silvestre from the AB pool) — those are correctly flagged non_bp
    but they're not production-deck drift."""
    return {r["voice_id"] for r in
            csv.DictReader(VOICES_TSV.open(encoding="utf-8"), dialect="excel-tab")
            if r.get("status", "") == "active"}


def _already_judged() -> set[tuple[str, str, str]]:
    """Union of (sense_id, clip_type, voice_id) keys judged successfully
    in either the AB-pool audit or this run's audit."""
    seen: set[tuple[str, str, str]] = set()
    for path in (AB_AUDIT, OUT_AUDIT):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") != "judged":
                continue
            seen.add((r.get("sense_id", ""), r.get("clip_type", ""),
                      r.get("voice_id", "")))
    return seen


def _candidate_rows(retry_failed: bool) -> list[dict]:
    """All BP word+example rows on eleven_v3 minus already-judged keys."""
    audit_skip = _already_judged()
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), dialect="excel-tab"))
    out: list[dict] = []
    for r in rows:
        if r["clip_type"] not in BP_CLIP_TYPES:
            continue
        if r["tts_model"] != V3_MODEL_ID:
            continue
        if r["status"] != "uploaded":
            continue
        key = (r["sense_id"], r["clip_type"], r["voice_id"])
        if key in audit_skip and not retry_failed:
            continue
        out.append(r)
    return out


# Shared connection-pooled HTTP client — single instance across all worker
# threads. Avoids the DNS storm we saw at 148 workers (macOS mDNSResponder
# overwhelmed by parallel hostname lookups) by reusing TCP connections.
_HTTP_CLIENT: httpx.Client | None = None


def _http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        # Keep ALL outbound connections alive so DNS resolution happens
        # at most once per host. R2 is a single CDN hostname; with
        # max_keepalive >= workers we never go back to mDNSResponder.
        _HTTP_CLIENT = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=300, max_keepalive_connections=300,
                                keepalive_expiry=120.0),
            transport=httpx.HTTPTransport(retries=3),  # network-level retries
            http2=False,  # HTTP/1.1; simpler pooling on macOS
        )
    return _HTTP_CLIENT


def _fetch_audio(url: str, max_attempts: int = 4) -> bytes:
    """Connection-pooled R2 GET with retries on transient network errors
    (DNS storms, RST, etc.). Returns raw bytes."""
    client = _http_client()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError, httpx.PoolTimeout,
                httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            last_exc = exc
            if attempt < max_attempts:
                # Tiny backoff with attempt-based growth: 0.5s, 1s, 2s.
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            break
    assert last_exc is not None
    raise last_exc


def _audio_bytes(row: dict) -> bytes:
    """Prefer local cache (populated by 16_8_prefetch_audio.py); fall
    back to R2 GET only when the local file is missing. This is the
    hot path: ~11k disk reads vs ~11k R2 fetches."""
    basename = Path(row["object_key"]).name
    local = LOCAL_AUDIO_CACHE / basename
    if local.exists() and local.stat().st_size > 0:
        return local.read_bytes()
    return _fetch_audio(row["url"])


def _judge_one(
    *,
    row: dict,
    text: str,
    client: GeminiAudioJudgeClient,
    limiter: SlidingWindowRateLimiter,
) -> tuple[dict, str, str]:
    """Returns (row, verdict-or-empty, error-or-empty)."""
    limiter.wait_before_call()
    try:
        audio = _audio_bytes(row)
    except Exception as exc:  # noqa: BLE001
        return row, "", f"fetch: {type(exc).__name__}: {exc}"
    try:
        result = client.judge(
            audio_bytes=audio, audio_format="mp3",
            pt=text, ipa_word_final="",
            voice_id=row["voice_id"],
            sense_id=row["sense_id"], clip_type=row["clip_type"],
        )
        return row, result.pronunciation_verdict, ""
    except Exception as exc:  # noqa: BLE001
        return row, "", f"judge: {type(exc).__name__}: {exc}"


def _build_watchlist(voice_names: dict[str, str]) -> int:
    """Walk both audit files; emit data/_gemini_drift_watchlist.tsv with
    every non_bp verdict on an ACTIVE production BP voice. Sorted by
    (severity DESC, confidence DESC, sense_id). Returns rows written.

    The AB-pool audit contains correct non_bp verdicts on EP control
    clips (e.g. Nelson Silvestre); those are filtered out so the
    watchlist only surfaces production-deck drift."""
    active_bp = _load_active_bp_voices()
    seen: set[tuple[str, str, str]] = set()
    drifts: list[dict] = []
    for path in (AB_AUDIT, OUT_AUDIT):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("verdict") != "non_bp":
                continue
            if r.get("clip_type") not in BP_CLIP_TYPES:
                continue
            if r.get("voice_id", "") not in active_bp:
                continue
            key = (r.get("sense_id", ""), r.get("clip_type", ""),
                   r.get("voice_id", ""))
            if key in seen:
                continue
            seen.add(key)
            drifts.append(r)
    sev_rank = {"high": 0, "medium": 1, "low": 2, "": 3}
    conf_rank = {"high": 0, "medium": 1, "low": 2, "": 3}
    drifts.sort(key=lambda r: (sev_rank.get(r.get("severity", ""), 3),
                               conf_rank.get(r.get("confidence", ""), 3),
                               r.get("sense_id", "")))
    OUT_WATCHLIST.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sense_id", "clip_type", "voice_id", "voice_name", "text",
            "verdict", "drift", "severity", "confidence", "evidence",
            "model", "judged_at"]
    with OUT_WATCHLIST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in drifts:
            vid = r.get("voice_id", "")
            w.writerow({
                "sense_id": r.get("sense_id", ""),
                "clip_type": r.get("clip_type", ""),
                "voice_id": vid,
                "voice_name": voice_names.get(vid, vid[:10]),
                "text": r.get("pt", ""),
                "verdict": r.get("verdict", ""),
                "drift": r.get("drift", ""),
                "severity": r.get("severity", ""),
                "confidence": r.get("confidence", ""),
                "evidence": (r.get("evidence", "") or "")[:300],
                "model": r.get("model", ""),
                "judged_at": r.get("judged_at", ""),
            })
    return len(drifts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Outer concurrency (default {DEFAULT_WORKERS}).")
    ap.add_argument("--max-rpm", type=int, default=DEFAULT_MAX_RPM,
                    help=f"Hard RPM cap via SlidingWindowRateLimiter "
                         f"(default {DEFAULT_MAX_RPM}; 0 disables).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N candidates (smoke test).")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-attempt rows even if a prior verdict exists.")
    ap.add_argument("--watchlist-only", action="store_true",
                    help="Skip rendering. Re-derive the watchlist from existing audits.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pre-flight only; no API calls.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the GO prompt.")
    args = ap.parse_args()

    # Load .env credentials.
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())

    voice_names = _load_voice_names()

    if args.watchlist_only:
        n = _build_watchlist(voice_names)
        print(f"watchlist-only: wrote {OUT_WATCHLIST} ({n} rows)")
        return 0

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    text_for = _load_text_map()
    candidates = _candidate_rows(args.retry_failed)
    if args.limit and len(candidates) > args.limit:
        candidates = candidates[:args.limit]

    skipped = len(_already_judged())

    print(f"=== Stage 16.8 — Gemini full-deck QA gate ===")
    print(f"  manifest rows (BP word+example on v3): {sum(1 for _ in MANIFEST.open()) - 1}")
    print(f"  already judged (skipped):              {skipped}")
    print(f"  to judge:                              {len(candidates):,}")
    n_chars = sum(len(text_for.get((r['sense_id'], r['clip_type']), '')) for r in candidates)
    cost_est = len(candidates) * 0.0027   # per-call cost from AB-pool average
    print(f"  total chars (informational):           {n_chars:,}")
    print(f"  cost (est at $0.0027/call):            ${cost_est:.2f}")
    print(f"  model:                                 {MODEL}")
    print(f"  workers:                               {args.workers}")
    print(f"  max RPM:                               {args.max_rpm} (hard cap)")
    rpm_at_9s = args.workers * (60.0 / 9.0)
    print(f"  RPM at ~9s mean latency:               {rpm_at_9s:.0f}")
    eta_min = len(candidates) / max(min(rpm_at_9s, args.max_rpm), 1) if args.max_rpm > 0 else \
              len(candidates) / max(rpm_at_9s, 1)
    print(f"  estimated wall:                        {eta_min:.1f} min")
    print(f"  audit JSONL:                           {OUT_AUDIT}")
    print(f"  watchlist out:                         {OUT_WATCHLIST}")
    print()

    if args.dry_run or not candidates:
        if args.dry_run:
            print("--dry-run: stopping.")
        else:
            print("nothing to do.")
        return 0

    if not args.yes:
        sys.stdout.write("Type GO to run: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    AUDIT.mkdir(parents=True, exist_ok=True)

    # NOTE: client is constructed once but is internally thread-safe via
    # its own locks. The wrapper appends to OUT_AUDIT for every call.
    client = GeminiAudioJudgeClient(model=MODEL, audit_path=OUT_AUDIT)
    limiter = SlidingWindowRateLimiter(max_rpm=args.max_rpm)

    t_start = time.time()
    n_ok = n_err = n_429 = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for row in candidates:
            text = text_for.get((row["sense_id"], row["clip_type"]), "")
            if not text:
                continue
            futs.append(pool.submit(_judge_one, row=row, text=text,
                                    client=client, limiter=limiter))
        for i, fut in enumerate(as_completed(futs), start=1):
            row, verdict, err = fut.result()
            if err:
                n_err += 1
                if "429" in err or "RESOURCE_EXHAUSTED" in err.upper():
                    n_429 += 1
                if i % 100 == 0 or n_err <= 5:
                    print(f"  [{i:>5}/{len(futs)}] FAIL {row['sense_id']:<12} "
                          f"{row['clip_type']:<7} {err[:90]}", file=sys.stderr)
            else:
                n_ok += 1
                if i % 500 == 0 or i == 1 or i == len(futs):
                    elapsed = time.time() - t_start
                    rpm_obs = n_ok / max(elapsed / 60, 0.01)
                    print(f"  [{i:>5}/{len(futs)}] OK   {row['sense_id']:<12} "
                          f"{row['clip_type']:<7} verdict={verdict:<7} "
                          f"obs_rpm={rpm_obs:.0f}", flush=True)

    elapsed = time.time() - t_start

    print()
    print(f"=== Stage 16.8 summary ===")
    print(f"  attempted:   {len(futs):,}")
    print(f"  OK:          {n_ok:,}")
    print(f"  FAIL:        {n_err:,}  (429s: {n_429})")
    print(f"  wall:        {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  obs RPM:     {n_ok / max(elapsed / 60, 0.01):.0f}")
    print(f"  audit:       {OUT_AUDIT}")

    # Always derive the watchlist at the end (succeeds even if some failed).
    n_drift = _build_watchlist(voice_names)
    print(f"  watchlist:   {OUT_WATCHLIST}  ({n_drift} rows)")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
