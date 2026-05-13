"""Stage 16 / Step 8 (prep) — Prefetch all BP v3 MP3s to a local cache.

The full-deck QA gate run (16_8) at 148 workers triggered macOS
mDNSResponder DNS storms when each worker independently resolved R2's
single hostname. Even with httpx connection pooling, throughput
collapsed mid-run.

Strategy change: download every BP v3 clip ONCE to disk, then run the
Gemini judge against local files. This removes network from the
judging hot path entirely and makes the judge run trivially
re-runnable.

Cache location:
    build/audio_cache/<basename>     # e.g. 0001.00.01-word-eleven_v3-v1.mp3

The basename comes from the manifest's object_key column
(`audio/<basename>`), so it's already deduped per (sense_id, clip_type,
voice_id, version) — the same key 16_8 uses to dedupe judgments.

Usage:
    .venv/bin/python build/16_8_prefetch_audio.py            # all 11,449
    .venv/bin/python build/16_8_prefetch_audio.py --workers 32
    .venv/bin/python build/16_8_prefetch_audio.py --limit 50 # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "build" / "audio_cache"
MANIFEST = DATA / "_audio_manifest.tsv"

BP_CLIP_TYPES = ("word", "example")
V3_MODEL_ID = "eleven_v3"
DEFAULT_WORKERS = 32  # well under DNS-storm threshold on macOS

_HTTP: httpx.Client | None = None


def _http() -> httpx.Client:
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(60.0, connect=15.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64,
                                keepalive_expiry=300.0),
            transport=httpx.HTTPTransport(retries=5),
            http2=False,
        )
    return _HTTP


def _fetch(url: str, dest: Path, max_attempts: int = 5) -> tuple[bool, str]:
    client = _http()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(resp.content)
            tmp.rename(dest)
            return True, ""
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError, httpx.PoolTimeout,
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    return False, f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"


def _candidate_rows() -> list[dict]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), dialect="excel-tab"))
    return [r for r in rows
            if r["clip_type"] in BP_CLIP_TYPES
            and r["tts_model"] == V3_MODEL_ID
            and r["status"] == "uploaded"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows = _candidate_rows()
    if args.limit:
        rows = rows[: args.limit]

    plan: list[tuple[dict, Path]] = []
    skipped_existing = 0
    for r in rows:
        basename = Path(r["object_key"]).name
        dest = CACHE_DIR / basename
        if dest.exists() and dest.stat().st_size > 0:
            skipped_existing += 1
            continue
        plan.append((r, dest))

    print(f"=== Stage 16.8 prefetch — BP v3 audio → {CACHE_DIR} ===")
    print(f"  manifest BP v3 uploaded rows: {len(rows):,}")
    print(f"  already cached locally:       {skipped_existing:,}")
    print(f"  to download:                  {len(plan):,}")
    print(f"  workers:                      {args.workers}")
    if not plan:
        print("  nothing to do.")
        return 0

    t_start = time.time()
    n_ok = n_err = 0
    err_samples: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_fetch, r["url"], dest): (r, dest) for r, dest in plan}
        for i, fut in enumerate(as_completed(futs), start=1):
            row, dest = futs[fut]
            ok, err = fut.result()
            if ok:
                n_ok += 1
            else:
                n_err += 1
                if len(err_samples) < 10:
                    err_samples.append(f"{row['sense_id']:<12} {row['clip_type']:<7} {err[:120]}")
            if i % 200 == 0 or i == 1 or i == len(plan):
                elapsed = time.time() - t_start
                rate = n_ok / max(elapsed, 0.01)
                eta = (len(plan) - i) / max(rate, 0.01)
                print(f"  [{i:>6}/{len(plan)}] ok={n_ok} err={n_err}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f} min", flush=True)

    elapsed = time.time() - t_start
    print()
    print(f"=== prefetch summary ===")
    print(f"  attempted: {len(plan):,}")
    print(f"  OK:        {n_ok:,}")
    print(f"  FAIL:      {n_err:,}")
    print(f"  wall:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if err_samples:
        print("  first failures:")
        for s in err_samples:
            print(f"    {s}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
