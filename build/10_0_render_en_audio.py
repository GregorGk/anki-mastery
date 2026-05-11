"""Stage 10 / Step 0 — Render English back-of-card audio.

For each row in `data/06-final.tsv` (5,725 senses), generate one EN audio
clip of the `example_en` sentence using the EN voice paired 1-to-1 with
the BP voice already assigned to that sense. The pairing lives in
`config/voices.tsv` as the `en_voice_id` column added in Stage 10.1.

Pipeline per clip:
    ElevenLabs PCM (Flash v2.5, language=en)
      -> ffmpeg loudnorm (-16 LUFS closed-loop)
        -> R2 upload (audio/{sense_id}-en_ex-eleven_flash_v2_5-v{N}.mp3)
          -> append manifest row (clip_type=en_ex, status=uploaded)

Concurrency: bounded ThreadPoolExecutor, default 20 (Pro/Flash ceiling).
Fail-fast on 429 — abort the batch so the operator can lower concurrency.

ASR roundtrip is intentionally skipped for EN (English Whisper accuracy
is high enough that the BP-style review queue is overkill; manual
spot-check via the pilot HTML is the QA gate).

Usage:
    .venv/bin/python build/10_0_render_en_audio.py --pilot 20
    .venv/bin/python build/10_0_render_en_audio.py --full --yes
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    STATUS_FAILED_TRANSIENT,
    STATUS_UPLOADED,
    STATUS_UPLOADING,
    init_row,
    read_manifest,
    write_manifest,
)
from build.lib.elevenlabs_client import ElevenLabsClient, RateLimitExceeded  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.progress import ProgressTracker, SummaryPrinter  # noqa: E402
from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE_DIR = REPO_ROOT / "build" / "audio_cache"
TMP_DIR = REPO_ROOT / "tmp"

FINAL_TSV = DATA_DIR / "06-final.tsv"
VOICES_TSV = CONFIG_DIR / "voices.tsv"
LOG_PATH = AUDIT_DIR / "10_0_render_en.log"
PROGRESS_JSONL = AUDIT_DIR / "10_0_render_en_progress.jsonl"

FLASH_MODEL_ID = "eleven_flash_v2_5"
LANGUAGE_CODE = "en"
DEFAULT_CONCURRENCY = 20
CLIP_TYPE = "en_ex"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _TeeStdout:
    """Mirror writes to both terminal and log file so `tail -f` works."""

    def __init__(self, *streams) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:  # noqa: BLE001
                    pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass


def _load_voice_pairing() -> dict[str, dict]:
    """Build {bp_voice_id: {en_voice_id, gender, pool_index}} from config/voices.tsv."""
    rows = read_tsv(VOICES_TSV)
    out: dict[str, dict] = {}
    for r in rows:
        bp = r["voice_id"]
        en = r.get("en_voice_id", "")
        if not en:
            raise RuntimeError(
                f"config/voices.tsv row for BP voice {bp!r} missing en_voice_id"
            )
        out[bp] = {
            "en_voice_id": en,
            "gender": r.get("gender", ""),
            "pool_index": r.get("pool_index", ""),
        }
    return out


def _load_senses(*, pilot_n: int | None) -> list[dict]:
    """Load (sense_id, example_en, bp_voice_id) from 06-final.tsv.

    Pilot mode: select up to `pilot_n` senses spread across all 8 BP voices.
    """
    rows = read_tsv(FINAL_TSV)
    senses = [
        {
            "sense_id": r["sense_id"],
            "rank": int(r.get("rank") or "0"),
            "example_en": r.get("example_en", ""),
            "bp_voice_id": r.get("voice_id", ""),
            "bp_voice_gender": r.get("voice_gender", ""),
        }
        for r in rows
        if r.get("example_en", "").strip()
    ]
    if pilot_n is None:
        return senses

    # Pilot: group by bp_voice_id, take lowest-rank senses per group until we hit pilot_n.
    by_voice: dict[str, list[dict]] = {}
    for s in senses:
        by_voice.setdefault(s["bp_voice_id"], []).append(s)
    for v in by_voice.values():
        v.sort(key=lambda x: x["rank"])

    voices_in_pool_order = list(by_voice.keys())
    picked: list[dict] = []
    indices = {v: 0 for v in voices_in_pool_order}
    while len(picked) < pilot_n and any(
        indices[v] < len(by_voice[v]) for v in voices_in_pool_order
    ):
        for v in voices_in_pool_order:
            if len(picked) >= pilot_n:
                break
            i = indices[v]
            if i < len(by_voice[v]):
                picked.append(by_voice[v][i])
                indices[v] = i + 1
    picked.sort(key=lambda x: x["sense_id"])
    return picked


@dataclass
class ClipResult:
    sense_id: str
    en_voice_id: str
    decision: str  # 'uploaded' | 'failed'
    chars: int
    final_lufs: float | None
    final_tp: float | None
    within_tolerance: bool
    applied_gain_db: float | None
    error: str = ""


def _process_one(
    *,
    sense: dict,
    pairing: dict[str, dict],
    el: ElevenLabsClient,
    r2: R2Client,
    public_base: str,
    manifest_lock: threading.Lock,
    cache_dir: Path,
    tracker: ProgressTracker | None = None,
) -> tuple[dict, ClipResult]:
    """TTS → loudnorm → R2 → return (manifest_row, ClipResult).

    Caller appends the manifest_row under manifest_lock.
    """
    sid = sense["sense_id"]
    text = sense["example_en"]
    bp_voice_id = sense["bp_voice_id"]
    pair = pairing[bp_voice_id]
    en_voice_id = pair["en_voice_id"]
    gender = pair["gender"]
    version = 1

    row = init_row(
        sense_id=sid,
        clip_type=CLIP_TYPE,
        voice_gender=gender,
        voice_id=en_voice_id,
        text_input=text,
        public_base=public_base,
        tts_provider="elevenlabs",
        tts_model=FLASH_MODEL_ID,
        version=version,
        filename_model_id=FLASH_MODEL_ID,
    )
    row["status"] = STATUS_UPLOADING

    key = sid  # ProgressTracker key — one task per sense
    if tracker is not None:
        tracker.started(key, sense_id=sid, voice_id=en_voice_id)

    try:
        tts = el.generate_pcm(
            text=text,
            voice_id=en_voice_id,
            sense_id=sid,
            clip_type=CLIP_TYPE,
            version=version,
        )
    except RateLimitExceeded:
        raise
    except Exception as exc:
        row["status"] = STATUS_FAILED_TRANSIENT
        row["notes"] = f"tts: {type(exc).__name__}: {str(exc)[:160]}"
        if tracker is not None:
            tracker.errored(
                key, error_type=type(exc).__name__, error_msg=str(exc),
                rendered=f"ERR  {sid} TTS {type(exc).__name__}",
            )
        return row, ClipResult(
            sense_id=sid, en_voice_id=en_voice_id, decision="failed",
            chars=len(text), final_lufs=None, final_tp=None,
            within_tolerance=False, applied_gain_db=None,
            error=f"tts: {type(exc).__name__}: {exc}",
        )

    try:
        vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
    except Exception as exc:
        row["status"] = STATUS_FAILED_TRANSIENT
        row["notes"] = f"loudnorm: {str(exc)[:160]}"
        if tracker is not None:
            tracker.errored(
                key, error_type=type(exc).__name__, error_msg=str(exc),
                rendered=f"ERR  {sid} loudnorm {type(exc).__name__}",
            )
        return row, ClipResult(
            sense_id=sid, en_voice_id=en_voice_id, decision="failed",
            chars=len(text), final_lufs=None, final_tp=None,
            within_tolerance=False, applied_gain_db=None,
            error=f"loudnorm: {type(exc).__name__}: {exc}",
        )

    mp3_bytes = vres.mp3_bytes

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sid}-en_ex-{FLASH_MODEL_ID}-v{version}.mp3"
    try:
        cache_path.write_bytes(mp3_bytes)
    except OSError:
        pass

    try:
        upload = r2.upload_bytes(
            mp3_bytes,
            row["object_key"],
            extra_metadata={
                "sense_id": sid,
                "clip_type": CLIP_TYPE,
                "voice_id": en_voice_id,
                "version": str(version),
                "language": LANGUAGE_CODE,
            },
        )
    except Exception as exc:
        row["status"] = STATUS_FAILED_TRANSIENT
        row["notes"] = f"r2: {str(exc)[:160]}"
        if tracker is not None:
            tracker.errored(
                key, error_type=type(exc).__name__, error_msg=str(exc),
                rendered=f"ERR  {sid} R2 {type(exc).__name__}",
            )
        return row, ClipResult(
            sense_id=sid, en_voice_id=en_voice_id, decision="failed",
            chars=len(text), final_lufs=vres.final_mp3_lufs, final_tp=vres.final_mp3_tp,
            within_tolerance=vres.within_tolerance,
            applied_gain_db=vres.applied_gain_db,
            error=f"r2: {type(exc).__name__}: {exc}",
        )

    with manifest_lock:
        row["url"] = upload.url
        row["md5"] = upload.content_md5
        row["applied_gain_db"] = f"{vres.applied_gain_db:.3f}"
        row["final_lufs"] = f"{vres.final_mp3_lufs:.3f}"
        row["final_tp"] = f"{vres.final_mp3_tp:.3f}"
        row["loudness_within_tolerance"] = "true" if vres.within_tolerance else "false"
        row["tp_limited"] = "true" if vres.tp_limited else "false"
        row["status"] = STATUS_UPLOADED
        row["generated_at"] = _now_iso()
        row["notes"] = "stage_10_en_render"

    if tracker is not None:
        tol_tag = "ok" if vres.within_tolerance else ("tp_lim" if vres.tp_limited else "drift")
        rendered = (
            f"OK   {sid}  {en_voice_id[:8]}.. "
            f"lufs={vres.final_mp3_lufs:.2f} {tol_tag}"
        )
        tracker.completed(key, decision="uploaded", rendered=rendered)

    return row, ClipResult(
        sense_id=sid, en_voice_id=en_voice_id, decision="uploaded",
        chars=len(text), final_lufs=vres.final_mp3_lufs, final_tp=vres.final_mp3_tp,
        within_tolerance=vres.within_tolerance,
        applied_gain_db=vres.applied_gain_db,
    )


def _print_preflight(*, n_senses: int, n_chars: int, concurrency: int, model: str) -> None:
    credits_est = int(n_chars * 0.5)  # Flash v2.5 = 0.5 credits/char
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  STAGE 10 — EN AUDIO RENDER PRE-FLIGHT                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Model:        {model:<55}│")
    print(f"│  Concurrency:  {concurrency:<55}│")
    print(f"│  Clips:        {n_senses:>6}  (clip_type=en_ex, version=1)              │")
    print(f"│  Chars:        {n_chars:>6,}                                              │")
    print(f"│  Credits (est): {credits_est:>5,}  (Flash v2.5 = 0.5 credits/char)         │")
    print(f"│  Language:     {LANGUAGE_CODE:<55}│")
    print(f"│  Filenames:    {{sid}}-en_ex-{model}-v{{N}}.mp3                │")
    print(f"│  Fail-fast:    429 → ABORT immediately                              │")
    print("└─────────────────────────────────────────────────────────────────────┘")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pilot", type=int, default=None,
                        help="Pilot run: render only this many clips, spread across voices.")
    parser.add_argument("--full", action="store_true",
                        help="Full run: all 5,725 senses.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive GO prompt (auto-mode safe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pre-flight and exit; no API calls.")
    args = parser.parse_args()

    if not args.pilot and not args.full:
        print("ERROR: pass --pilot N or --full", file=sys.stderr)
        return 2
    if args.pilot and args.full:
        print("ERROR: --pilot and --full are mutually exclusive", file=sys.stderr)
        return 2

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 1

    pairing = _load_voice_pairing()
    pilot_n = args.pilot if args.pilot else None
    senses = _load_senses(pilot_n=pilot_n)
    if not senses:
        print("ERROR: no senses to render", file=sys.stderr)
        return 1

    # Resume: skip senses that already have an uploaded en_ex row in the
    # manifest. Pilot mode keeps its deterministic top-N picks (rerunning
    # the pilot should be a no-op once seeded).
    existing = read_manifest()
    already_done = {
        r["sense_id"]
        for r in existing
        if r["clip_type"] == CLIP_TYPE and r["status"] == STATUS_UPLOADED
    }
    before = len(senses)
    senses = [s for s in senses if s["sense_id"] not in already_done]
    skipped = before - len(senses)
    if skipped:
        print(f"[resume] skipping {skipped} senses already uploaded "
              f"(clip_type={CLIP_TYPE}, status=uploaded)")
    if not senses:
        print("Nothing to do — all senses already rendered.")
        return 0

    n_chars = sum(len(s["example_en"]) for s in senses)
    _print_preflight(
        n_senses=len(senses), n_chars=n_chars,
        concurrency=args.concurrency, model=FLASH_MODEL_ID,
    )

    if args.dry_run:
        print("\n--dry-run: stopping.")
        return 0

    if not args.yes:
        sys.stdout.write("\nType GO to proceed (anything else cancels): ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled. No changes made.")
            return 0

    log_file = LOG_PATH.open("a", encoding="utf-8")
    log_file.write(f"\n=== stage 10 EN render started {_now_iso()} | "
                   f"n_senses={len(senses)} | pilot={pilot_n} ===\n")
    log_file.flush()

    # Tee stdout so the periodic SummaryPrinter lines land in the log too,
    # making `tail -f audit/10_0_render_en.log` show live %/ETA.
    original_stdout = sys.stdout
    sys.stdout = _TeeStdout(original_stdout, log_file)  # type: ignore[assignment]

    r2 = R2Client()
    public_base = r2.config.public_base
    el = ElevenLabsClient(
        model_id=FLASH_MODEL_ID,
        language_code=LANGUAGE_CODE,
        fail_fast_on_429=True,
    )

    # Write a run_started marker so progress JSONL is unambiguous about
    # which run is current (parallels stage_6.run()'s marker).
    PROGRESS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with PROGRESS_JSONL.open("a", encoding="utf-8") as _rf:
        _rf.write(_json.dumps({
            "event": "run_started",
            "stage": "Stage_10",
            "pilot_size": pilot_n,
            "concurrency": args.concurrency,
            "pending_total": len(senses),
            "started_at": _now_iso(),
        }, ensure_ascii=False) + "\n")

    tracker = ProgressTracker(
        progress_path=PROGRESS_JSONL,
        total=len(senses),
        decision_axes=("uploaded", "failed"),
        stage_label="Stage 10",
        ring_buffer_size=5,
    )
    summary_thread = SummaryPrinter(tracker, interval=30.0, stream=sys.stdout)
    summary_thread.start()

    manifest_lock = threading.Lock()
    results: list[ClipResult] = []
    new_rows: list[dict] = []
    rate_limit_hit = False
    t_start = time.time()

    def _worker(sense: dict) -> tuple[dict, ClipResult]:
        return _process_one(
            sense=sense,
            pairing=pairing,
            el=el,
            r2=r2,
            public_base=public_base,
            manifest_lock=manifest_lock,
            cache_dir=AUDIO_CACHE_DIR,
            tracker=tracker,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_worker, s): s for s in senses}
            for fut in as_completed(futures):
                try:
                    row, res = fut.result()
                    new_rows.append(row)
                    results.append(res)
                    # Per-clip line: only print failures inline (tracker
                    # logs successes via its 30-s summary + ring buffer).
                    if res.decision != "uploaded":
                        print(f"FAIL {res.sense_id} {res.error}", flush=True)
                except RateLimitExceeded as exc:
                    if not rate_limit_hit:
                        rate_limit_hit = True
                        print(
                            f"\n\nABORT: 429 at concurrency={args.concurrency}. "
                            f"{len(results)} clips done before stop. {exc}\n",
                            flush=True,
                        )
                    for f in futures:
                        f.cancel()
    finally:
        summary_thread.stop()
        # Print one final summary so the log captures the terminal state.
        print(tracker.format_summary(), flush=True)
        # Append new rows to the manifest (preserving all existing BP rows).
        if new_rows:
            existing = read_manifest()
            existing.extend(new_rows)
            write_manifest(existing)
            print(f"\nAppended {len(new_rows)} EN rows to {DEFAULT_MANIFEST_PATH}",
                  flush=True)
        sys.stdout = original_stdout  # type: ignore[assignment]
        log_file.close()

    # Summary
    uploaded = sum(1 for r in results if r.decision == "uploaded")
    failed = sum(1 for r in results if r.decision == "failed")
    within_tol = sum(1 for r in results if r.decision == "uploaded" and r.within_tolerance)
    elapsed = time.time() - t_start
    print()
    print("=== Stage 10 EN render summary ===")
    print(f"  Total:    {len(results):>4}")
    print(f"  Uploaded: {uploaded:>4}")
    print(f"  Failed:   {failed:>4}")
    print(f"  Within ±1 LU: {within_tol}/{uploaded} "
          f"({100*within_tol/max(uploaded,1):.0f}%)")
    print(f"  Wall:     {elapsed:.1f}s "
          f"({elapsed/max(len(results),1):.2f}s/clip)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
