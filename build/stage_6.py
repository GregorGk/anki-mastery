"""Stage 6/7 — Audio generation orchestrator.

Pipeline per clip:
  ElevenLabs PCM  ->  ffmpeg loudnorm + MP3 encode  ->  ASR roundtrip  ->  R2 upload
                      (pass-1 + pass-2 in pilot;
                       cached per-voice gain in full)

Concurrency: bounded ThreadPoolExecutor (default 4 workers — gentle on
ElevenLabs Pro tier rate limits and Whisper API).

Outputs:
  data/_audio_manifest.tsv          source of truth
  data/_voice_loudness_baselines.tsv  per-voice cached gain (Stage 7 fast path)
  data/_audio_human_review.tsv      twice-failed clips needing human listen
  audit/06_audio.jsonl              machine-readable lifecycle events
  audit/06_audio_transcript.log     human-readable, tail-friendly
  audit/06_audio_progress.jsonl     in-flight + stuck detection
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.audio_logger import (  # noqa: E402
    ERR,
    HUMAN,
    PASS,
    REGEN,
    AsrOutcome,
    AudioLogger,
)
from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    MANIFEST_FIELDS,
    STATUS_FAILED_PERMANENT,
    STATUS_FAILED_TRANSIENT,
    STATUS_UPLOADED,
    STATUS_UPLOADING,
    bump_version,
    find_pending,
    index_by_key,
    init_manifest_for_senses,
    read_manifest,
    summarize,
    write_manifest,
)
from build.lib.elevenlabs_client import ElevenLabsClient, RateLimitExceeded  # noqa: E402
from build.lib.loudness import (  # noqa: E402
    LoudnessMeasurement,
    VerifiedNormalizationResult,
    normalize_pcm_to_mp3_verified,
    verify_target,
    voice_baseline_from_offsets,
)
from build.lib.progress import ProgressTracker, SummaryPrinter  # noqa: E402
from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE_DIR = REPO_ROOT / "build" / "audio_cache"

IPA_PATH = DATA_DIR / "05-ipa.tsv"
SPEAKER_GENDER_PATH = DATA_DIR / "045-speaker_gender.tsv"
VOICE_BASELINES_PATH = DATA_DIR / "_voice_loudness_baselines.tsv"
HUMAN_REVIEW_PATH = DATA_DIR / "_audio_human_review.tsv"

AUDIO_JSONL_PATH = AUDIT_DIR / "06_audio.jsonl"
AUDIO_TRANSCRIPT_PATH = AUDIT_DIR / "06_audio_transcript.log"
AUDIO_PROGRESS_PATH = AUDIT_DIR / "06_audio_progress.jsonl"

VOICE_BASELINE_FIELDS = [
    "voice_id",
    "median_gain_db",
    "measured_n",
    "measured_at",
    "notes",
]
HUMAN_REVIEW_FIELDS = [
    "sense_id",
    "clip_type",
    "voice_id",
    "input_text",
    "asr_transcript",
    "similarity",
    "phonetic_distance",
    "url",
    "added_at",
    "notes",
]

DEFAULT_MAX_ATTEMPTS_PER_CLIP = 2  # TTS+ASR retries before HUMAN
DEFAULT_CONCURRENCY = 4
TOP_1000_RANK_THRESHOLD = 1000


# --- Inputs --------------------------------------------------------------- #


def _load_senses_with_voice(
    *,
    ipa_path: Path = IPA_PATH,
    voice_path: Path = SPEAKER_GENDER_PATH,
) -> list[dict]:
    """Join 05-ipa.tsv with 045-speaker_gender.tsv on sense_id.

    Returns rows containing all fields needed for clip generation:
        sense_id, rank, pt, example_pt, voice_id, voice_gender_assigned
    """
    if not ipa_path.exists():
        raise FileNotFoundError(f"missing {ipa_path}; run Stage 5 first")
    if not voice_path.exists():
        raise FileNotFoundError(f"missing {voice_path}; run Stage 4.5 first")
    ipa_rows = read_tsv(ipa_path)
    voice_rows = read_tsv(voice_path)
    voice_by_sid = {r["sense_id"]: r for r in voice_rows}
    out: list[dict] = []
    for r in ipa_rows:
        sid = r.get("sense_id", "")
        v = voice_by_sid.get(sid)
        if not v:
            raise RuntimeError(
                f"sense_id {sid!r} present in 05-ipa.tsv but missing from "
                f"045-speaker_gender.tsv — re-run Stage 4.5"
            )
        out.append(
            {
                "sense_id": sid,
                "rank": r.get("rank", ""),
                "pt": r.get("pt", ""),
                "example_pt": r.get("example_pt", ""),
                "voice_id": v.get("voice_id", ""),
                "voice_gender_assigned": v.get("voice_gender_assigned", ""),
            }
        )
    return out


def _load_voice_baselines(path: Path = VOICE_BASELINES_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = read_tsv(path)
    return {r["voice_id"]: float(r["median_gain_db"]) for r in rows if r.get("voice_id")}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Preflight cost report ------------------------------------------------ #


@dataclass
class PreflightReport:
    senses: int
    word_clips: int
    example_clips: int
    word_chars: int
    example_chars: int
    total_chars: int
    char_buffer_pct: float
    estimated_credits: int
    elevenlabs_pro_cap: int
    estimated_asr_cost_usd: float
    estimated_total_seconds: float

    def render(self) -> str:
        lines = [
            "=== Audio preflight cost report ===",
            f"  senses:                    {self.senses:,}",
            f"  word clips:                {self.word_clips:,}",
            f"  example clips:             {self.example_clips:,}",
            f"  word chars:                {self.word_chars:,}",
            f"  example chars:             {self.example_chars:,}",
            f"  total chars:               {self.total_chars:,}",
            f"  retry buffer:              +{self.char_buffer_pct:.0%}",
            f"  ElevenLabs credits (est):  {self.estimated_credits:,}  / Pro cap {self.elevenlabs_pro_cap:,}",
            f"  ASR (Whisper) cost (est):  ${self.estimated_asr_cost_usd:,.2f}",
            f"  approx audio seconds:      {self.estimated_total_seconds:,.0f}s",
            "==================================",
        ]
        return "\n".join(lines)


def preflight(
    senses: list[dict],
    *,
    char_buffer_pct: float = 0.30,
    elevenlabs_pro_cap: int = 600_000,
    avg_chars_per_second: float = 16.0,  # Portuguese speaking rate ≈ 16 chars/s
    whisper_price_per_minute: float = 0.006,
) -> PreflightReport:
    word_chars = sum(len((s.get("pt") or "")) for s in senses)
    example_chars = sum(len((s.get("example_pt") or "")) for s in senses)
    total = word_chars + example_chars
    credits = int(total * (1 + char_buffer_pct))
    audio_secs = total / max(1.0, avg_chars_per_second)
    asr_cost = (audio_secs / 60.0) * whisper_price_per_minute
    # ASR cost includes a second pass on every word clip (cross-check)
    asr_cost *= 1.5  # rough; example clips have 1 pass, word clips 1-2
    return PreflightReport(
        senses=len(senses),
        word_clips=len(senses),
        example_clips=len(senses),
        word_chars=word_chars,
        example_chars=example_chars,
        total_chars=total,
        char_buffer_pct=char_buffer_pct,
        estimated_credits=credits,
        elevenlabs_pro_cap=elevenlabs_pro_cap,
        estimated_asr_cost_usd=round(asr_cost, 2),
        estimated_total_seconds=audio_secs,
    )


# --- Per-clip worker ------------------------------------------------------ #


@dataclass
class ClipResult:
    sense_id: str
    clip_type: str
    final_decision: str  # 'pass' | 'human' | 'err'
    attempts: int
    similarity: float | None
    transcript: str
    cost_asr_usd: float
    voice_baseline_offset_db: float | None  # FINAL applied gain (post-correction)
    final_lufs: float | None = None
    final_tp: float | None = None
    within_tolerance: bool = False
    tp_limited: bool = False
    correction_iterations: int = 0
    error_type: str = ""
    error_msg: str = ""


def _normalize_pcm(
    pcm_bytes: bytes,
    *,
    voice_id: str,
    baselines: dict[str, float],
) -> tuple[bytes, VerifiedNormalizationResult]:
    """Always use closed-loop verified normalization.

    The earlier "cached-baseline fast path" (single-pass volume gain using
    a per-voice median) turned out to give terrible per-clip loudness:
    per-clip RMS varies by ~7 LU even within the same voice, so a single
    median gain produced a 22 LU range across the deck. The closed-loop
    pass 1 measurement is cheap (~50 ms/clip); use it always.

    The `baselines` argument is kept for backwards-compat but ignored for
    gain selection; the surgical_recovery / Stage 6 record voice baselines
    for diagnostic logging only.
    """
    r = normalize_pcm_to_mp3_verified(pcm_bytes)
    return r.mp3_bytes, r


def _process_one_clip(
    *,
    manifest_row: dict,
    el: ElevenLabsClient,
    asr: AsrClient,
    r2: R2Client,
    logger: AudioLogger,
    tracker: ProgressTracker,
    manifest_lock: threading.Lock,
    rows_by_key: dict[tuple[str, str], dict],
    baselines: dict[str, float],
    is_top_1000: bool,
    max_attempts: int,
    cache_dir: Path,
    filename_model_id: str = "",
) -> ClipResult:
    sid = manifest_row["sense_id"]
    ctype = manifest_row["clip_type"]
    voice_id = manifest_row["voice_id"]
    text = manifest_row["text_input"]
    key = f"{sid}|{ctype}"

    cache_dir.mkdir(parents=True, exist_ok=True)
    last_outcome: AsrOutcome | None = None
    pass1_offset: float | None = None
    final_lufs: float | None = None
    final_tp: float | None = None
    within_tolerance: bool = False
    tp_limited: bool = False
    correction_iterations: int = 0
    asr_cost_total = 0.0

    for attempt in range(1, max_attempts + 1):
        version = int(manifest_row["version"] or "1")
        tracker.started(key, attempt=attempt, sense_id=sid, clip_type=ctype, voice_id=voice_id)
        logger.started(sid, ctype, voice_id, text, attempt=attempt)

        # 1. ElevenLabs PCM
        try:
            tts = el.generate_pcm(
                text=text,
                voice_id=voice_id,
                sense_id=sid,
                clip_type=ctype,
                version=version,
            )
        except Exception as exc:
            logger.errored(sid, ctype, voice_id, attempt, type(exc).__name__, str(exc), stage="tts")
            tracker.errored(
                key,
                error_type=type(exc).__name__,
                error_msg=str(exc),
                attempt=attempt,
                rendered=f"{ERR}    {sid} {ctype}  TTS {type(exc).__name__}",
            )
            if attempt >= max_attempts:
                with manifest_lock:
                    manifest_row["status"] = STATUS_FAILED_TRANSIENT
                    manifest_row["notes"] = f"tts: {type(exc).__name__}: {str(exc)[:160]}"
                return ClipResult(
                    sense_id=sid,
                    clip_type=ctype,
                    final_decision="err",
                    attempts=attempt,
                    similarity=None,
                    transcript="",
                    cost_asr_usd=asr_cost_total,
                    voice_baseline_offset_db=None,
                    error_type=type(exc).__name__,
                    error_msg=str(exc)[:200],
                )
            continue

        # 2. Loudness norm + MP3 encode (closed-loop verified)
        try:
            mp3_bytes, vres = _normalize_pcm(
                tts.audio_pcm,
                voice_id=voice_id,
                baselines=baselines,
            )
            # Closed-loop normalization is always used now. vres is always present.
            pass1_offset = vres.applied_gain_db
            final_lufs = vres.final_mp3_lufs
            final_tp = vres.final_mp3_tp
            within_tolerance = vres.within_tolerance
            tp_limited = vres.tp_limited
            correction_iterations = vres.correction_iterations
        except Exception as exc:
            logger.errored(sid, ctype, voice_id, attempt, type(exc).__name__, str(exc), stage="loudnorm")
            tracker.errored(
                key,
                error_type=type(exc).__name__,
                error_msg=str(exc),
                attempt=attempt,
                rendered=f"{ERR}    {sid} {ctype}  loudnorm {type(exc).__name__}",
            )
            if attempt >= max_attempts:
                with manifest_lock:
                    manifest_row["status"] = STATUS_FAILED_TRANSIENT
                    manifest_row["notes"] = f"loudnorm: {str(exc)[:160]}"
                return ClipResult(
                    sense_id=sid,
                    clip_type=ctype,
                    final_decision="err",
                    attempts=attempt,
                    similarity=None,
                    transcript="",
                    cost_asr_usd=asr_cost_total,
                    voice_baseline_offset_db=pass1_offset,
                    error_type=type(exc).__name__,
                    error_msg=str(exc)[:200],
                )
            continue

        # Cache MP3 locally for inspection / re-encode. Filename mirrors
        # the manifest object_key (minus the leading "audio/"), so a
        # local cache file lines up 1:1 with the deployed R2 key.
        short = "word" if ctype == "word" else "ex"
        model_seg = f"-{filename_model_id}" if filename_model_id else ""
        cache_path = cache_dir / f"{sid}-{short}{model_seg}-v{version}.mp3"
        try:
            cache_path.write_bytes(mp3_bytes)
        except OSError:
            pass

        # 3. ASR roundtrip
        try:
            ar = asr_roundtrip(
                asr=asr,
                mp3_bytes=mp3_bytes,
                input_text=text,
                clip_type=ctype,
                sense_id=sid,
                is_top_1000=is_top_1000,
            )
            asr_cost_total += ar.cost_usd
        except Exception as exc:
            logger.errored(sid, ctype, voice_id, attempt, type(exc).__name__, str(exc), stage="asr")
            tracker.errored(
                key,
                error_type=type(exc).__name__,
                error_msg=str(exc),
                attempt=attempt,
                rendered=f"{ERR}    {sid} {ctype}  ASR {type(exc).__name__}",
            )
            if attempt >= max_attempts:
                with manifest_lock:
                    manifest_row["status"] = STATUS_FAILED_TRANSIENT
                    manifest_row["notes"] = f"asr: {str(exc)[:160]}"
                return ClipResult(
                    sense_id=sid,
                    clip_type=ctype,
                    final_decision="err",
                    attempts=attempt,
                    similarity=None,
                    transcript="",
                    cost_asr_usd=asr_cost_total,
                    voice_baseline_offset_db=pass1_offset,
                    error_type=type(exc).__name__,
                    error_msg=str(exc)[:200],
                )
            continue

        # Determine final decision tag for the transcript log
        is_last_attempt = attempt >= max_attempts
        if ar.decision == "pass":
            decision_tag = PASS
        elif ar.decision == "regen" and not is_last_attempt:
            decision_tag = REGEN
        else:
            decision_tag = HUMAN  # twice-failed -> human review queue

        # 4. Always upload to R2 on PASS or HUMAN (so the deck has audio).
        # On REGEN with attempts remaining, we skip upload, bump version, retry.
        upload_url = manifest_row["url"]
        upload_md5 = manifest_row["md5"]
        if decision_tag in (PASS, HUMAN):
            try:
                upload = r2.upload_bytes(
                    mp3_bytes,
                    manifest_row["object_key"],
                    extra_metadata={
                        "sense_id": sid,
                        "clip_type": ctype,
                        "voice_id": voice_id,
                        "version": str(version),
                        "asr_decision": ar.decision,
                    },
                )
                upload_url = upload.url
                upload_md5 = upload.content_md5
            except Exception as exc:
                logger.errored(sid, ctype, voice_id, attempt, type(exc).__name__, str(exc), stage="r2")
                tracker.errored(
                    key,
                    error_type=type(exc).__name__,
                    error_msg=str(exc),
                    attempt=attempt,
                    rendered=f"{ERR}    {sid} {ctype}  R2 {type(exc).__name__}",
                )
                with manifest_lock:
                    manifest_row["status"] = STATUS_FAILED_TRANSIENT
                    manifest_row["notes"] = f"r2: {str(exc)[:160]}"
                return ClipResult(
                    sense_id=sid,
                    clip_type=ctype,
                    final_decision="err",
                    attempts=attempt,
                    similarity=ar.text_similarity,
                    transcript=ar.transcript,
                    cost_asr_usd=asr_cost_total,
                    voice_baseline_offset_db=pass1_offset,
                    error_type=type(exc).__name__,
                    error_msg=str(exc)[:200],
                )

        # 5. Record outcome (always, for transcript log)
        outcome = AsrOutcome(
            sense_id=sid,
            clip_type=ctype,
            voice_id=voice_id,
            input_text=text,
            asr_transcript=ar.transcript,
            similarity=ar.text_similarity,
            phonetic_distance=ar.phonetic_distance,
            decision=decision_tag,
            attempt=attempt,
            latency_ms=0,  # filled by tracker
            cost_usd=ar.cost_usd,
            note=ar.notes,
        )
        rendered = logger.asr_completed(outcome)
        last_outcome = outcome

        # 6. PASS or HUMAN: finalize manifest + tracker, return.
        if decision_tag in (PASS, HUMAN):
            with manifest_lock:
                manifest_row["url"] = upload_url
                manifest_row["md5"] = upload_md5
                manifest_row["asr_transcript"] = ar.transcript
                manifest_row["asr_similarity"] = str(ar.text_similarity)
                manifest_row["asr_decision"] = ar.decision  # 'pass' or 'regen' (the failed one)
                # Loudness diagnostics
                if pass1_offset is not None:
                    manifest_row["applied_gain_db"] = f"{pass1_offset:.3f}"
                if final_lufs is not None:
                    manifest_row["final_lufs"] = f"{final_lufs:.3f}"
                if final_tp is not None:
                    manifest_row["final_tp"] = f"{final_tp:.3f}"
                manifest_row["loudness_within_tolerance"] = "true" if within_tolerance else "false"
                manifest_row["tp_limited"] = "true" if tp_limited else "false"
                manifest_row["status"] = STATUS_UPLOADED
                manifest_row["generated_at"] = _now_iso()
                if decision_tag == HUMAN:
                    manifest_row["notes"] = (
                        f"asr_decision=regen after {attempt} attempts; routed to human review"
                    )
            tracker.completed(
                key,
                decision="pass" if decision_tag == PASS else "human",
                attempt=attempt,
                cost_usd=ar.cost_usd,
                rendered=rendered,
            )
            return ClipResult(
                sense_id=sid,
                clip_type=ctype,
                final_decision="pass" if decision_tag == PASS else "human",
                attempts=attempt,
                similarity=ar.text_similarity,
                transcript=ar.transcript,
                cost_asr_usd=asr_cost_total,
                voice_baseline_offset_db=pass1_offset,
                final_lufs=final_lufs,
                final_tp=final_tp,
                within_tolerance=within_tolerance,
                tp_limited=tp_limited,
                correction_iterations=correction_iterations,
            )

        # REGEN with retries remaining: bump version, loop again
        with manifest_lock:
            bump_version(manifest_row, public_base=r2.config.public_base, model_id=filename_model_id)
        logger.regenerated(sid, ctype, voice_id, attempt + 1, ar.transcript)

    # If we fall through (all attempts exhausted as REGEN, no PASS/HUMAN
    # branch hit) — shouldn't happen because the last-attempt REGEN is
    # promoted to HUMAN above, but be safe.
    if last_outcome is not None:
        tracker.completed(
            key,
            decision="human",
            attempt=max_attempts,
            cost_usd=0.0,
            rendered=logger.render_outcome(last_outcome),
        )
    return ClipResult(
        sense_id=sid,
        clip_type=ctype,
        final_decision="human",
        attempts=max_attempts,
        similarity=last_outcome.similarity if last_outcome else None,
        transcript=last_outcome.asr_transcript if last_outcome else "",
        cost_asr_usd=asr_cost_total,
        voice_baseline_offset_db=pass1_offset,
        final_lufs=final_lufs,
        final_tp=final_tp,
        within_tolerance=within_tolerance,
        tp_limited=tp_limited,
        correction_iterations=correction_iterations,
    )


# --- Main run ------------------------------------------------------------- #


def run(
    *,
    pilot_size: int | None = 500,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_attempts_per_clip: int = DEFAULT_MAX_ATTEMPTS_PER_CLIP,
    dry_run: bool = False,
    confirm: bool = False,
    sense_id_filter: set[str] | None = None,
    cache_dir: Path = AUDIO_CACHE_DIR,
    pronunciation_dict_locators: list[dict] | None = None,
    model_id: str | None = None,
    fail_fast_on_429: bool = False,
) -> dict:
    """Run Stage 6 (pilot, default pilot_size=500) or Stage 7 (pilot_size=None).

    Resume-safe: rows with status=uploaded are skipped.
    """
    senses = _load_senses_with_voice()

    if sense_id_filter is not None:
        senses = [s for s in senses if s["sense_id"] in sense_id_filter]
    if pilot_size is not None:
        senses = senses[:pilot_size]

    if not senses:
        print("[stage_6] no senses to process; exiting", file=sys.stderr)
        return {"senses": 0}

    # Preflight
    report = preflight(senses)
    print(report.render(), file=sys.stderr)
    if dry_run:
        print("[stage_6] dry-run: no API calls made.", file=sys.stderr)
        return {"senses": len(senses), "dry_run": True}
    if not confirm:
        raise RuntimeError(
            "Refusing to spend without --confirm flag. "
            "Re-run with confirm=True / --confirm after reviewing the cost report."
        )

    # Load / init manifest
    r2 = R2Client()
    public_base = r2.config.public_base

    existing_rows = read_manifest()
    if existing_rows:
        rows_by_key = index_by_key(existing_rows)
        # Merge in any new senses missing from the manifest
        for s in senses:
            for ctype, text in (("word", s["pt"]), ("example", s["example_pt"])):
                if (s["sense_id"], ctype) not in rows_by_key:
                    new_rows = init_manifest_for_senses([s], public_base=public_base)
                    for r in new_rows:
                        existing_rows.append(r)
                        rows_by_key[(r["sense_id"], r["clip_type"])] = r
        manifest_rows = existing_rows
    else:
        manifest_rows = init_manifest_for_senses(senses, public_base=public_base)
        rows_by_key = index_by_key(manifest_rows)

    pending = find_pending(manifest_rows)
    pending = [
        r
        for r in pending
        if r["sense_id"] in {s["sense_id"] for s in senses}
    ]
    if not pending:
        print(f"[stage_6] manifest complete; {summarize(manifest_rows).fmt()}", file=sys.stderr)
        return {"senses": len(senses), "pending": 0}

    # Write a run_started marker into the audit JSONL so audio_status.py can
    # cleanly distinguish this run from any prior runs in the same log.
    AUDIO_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_started_at = _now_iso()
    with AUDIO_JSONL_PATH.open("a", encoding="utf-8") as _rf:
        import json as _json
        _rf.write(_json.dumps({
            "event": "run_started",
            "stage": f"Stage_{'6' if pilot_size else '7'}",
            "pilot_size": pilot_size,
            "concurrency": concurrency,
            "pending_total": len(pending),
            "started_at": run_started_at,
        }, ensure_ascii=False) + "\n")

    print(f"[stage_6] manifest: {summarize(manifest_rows).fmt()}", file=sys.stderr)
    print(f"[stage_6] pending clips: {len(pending):,}", file=sys.stderr)

    # Top-1000 lookup
    rank_by_sid = {s["sense_id"]: int(s["rank"]) for s in senses if s.get("rank", "").strip().isdigit()}

    # Clients
    el_kwargs: dict = {"pronunciation_dict_locators": pronunciation_dict_locators,
                       "fail_fast_on_429": fail_fast_on_429}
    if model_id:
        el_kwargs["model_id"] = model_id
    el = ElevenLabsClient(**el_kwargs)
    # The model_id segment baked into filenames — only set when caller
    # explicitly wants provenance in the path (Stage 9 migration).
    filename_model_id = model_id if model_id else ""
    asr = AsrClient()
    logger = AudioLogger(AUDIO_JSONL_PATH, AUDIO_TRANSCRIPT_PATH)
    tracker = ProgressTracker(
        progress_path=AUDIO_PROGRESS_PATH,
        total=len(pending),
        decision_axes=("pass", "regen", "human", "err"),
        stage_label=f"Stage {'6' if pilot_size else '7'}",
        ring_buffer_size=5,
    )
    summary_thread = SummaryPrinter(tracker, interval=30.0)
    summary_thread.start()

    # Cached per-voice baselines (Stage 7 fast path)
    baselines = _load_voice_baselines()
    if baselines:
        print(
            f"[stage_6] using cached voice baselines for {len(baselines)} voices",
            file=sys.stderr,
        )
    else:
        print("[stage_6] no voice baselines yet; using two-pass loudnorm per clip", file=sys.stderr)

    # Per-voice pass-1 measurements collected during the run (for baselines TSV)
    voice_offsets: dict[str, list[float]] = {}
    voice_offsets_lock = threading.Lock()
    manifest_lock = threading.Lock()

    def _worker(row: dict) -> ClipResult:
        rank = rank_by_sid.get(row["sense_id"], 0)
        is_top = rank > 0 and rank <= TOP_1000_RANK_THRESHOLD
        # Mark uploading (no further LLM calls happen on cancel; but visible on inspection)
        with manifest_lock:
            row["status"] = STATUS_UPLOADING
        result = _process_one_clip(
            manifest_row=row,
            el=el,
            asr=asr,
            r2=r2,
            logger=logger,
            tracker=tracker,
            manifest_lock=manifest_lock,
            rows_by_key=rows_by_key,
            baselines=baselines,
            is_top_1000=is_top,
            max_attempts=max_attempts_per_clip,
            cache_dir=cache_dir,
            filename_model_id=filename_model_id,
        )
        if result.voice_baseline_offset_db is not None:
            with voice_offsets_lock:
                voice_offsets.setdefault(row["voice_id"], []).append(
                    result.voice_baseline_offset_db
                )
        return result

    results: list[ClipResult] = []
    rate_limit_hit = False
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker, row) for row in pending]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except RateLimitExceeded as exc:
                    # Fail-fast on 429: cancel pending tasks, abort the run.
                    if not rate_limit_hit:
                        rate_limit_hit = True
                        print(
                            f"\n\n[stage_6] ABORT: rate limit (429) at concurrency={concurrency}. "
                            f"{len(results)} clips already done before stop. "
                            f"Lower --concurrency and rerun; resume-mode skips completed clips.\n"
                            f"  details: {exc}\n",
                            file=sys.stderr,
                            flush=True,
                        )
                    for f in futures:
                        f.cancel()
                except Exception as exc:  # noqa: BLE001 — capture for summary
                    print(f"[stage_6] worker raised: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        summary_thread.stop()
        # Persist manifest no matter what.
        write_manifest(manifest_rows)

    # Final summary
    print(tracker.format_summary(), file=sys.stderr, flush=True)
    print(f"[stage_6] manifest: {summarize(manifest_rows).fmt()}", file=sys.stderr)

    # Voice baselines TSV (pilot only — when we measured pass-1 ourselves)
    if voice_offsets and not baselines:
        baseline_rows = []
        now_iso = _now_iso()
        for voice_id, offsets in voice_offsets.items():
            b = voice_baseline_from_offsets(voice_id, offsets, notes="pilot")
            baseline_rows.append(
                {
                    "voice_id": b.voice_id,
                    "median_gain_db": str(b.median_gain_db),
                    "measured_n": str(b.measured_n),
                    "measured_at": now_iso,
                    "notes": b.notes,
                }
            )
        write_tsv(VOICE_BASELINES_PATH, baseline_rows, fieldnames=VOICE_BASELINE_FIELDS)
        print(
            f"[stage_6] wrote voice baselines for {len(baseline_rows)} voices -> "
            f"{VOICE_BASELINES_PATH}",
            file=sys.stderr,
        )

    # Human review queue
    human_rows = [r for r in results if r.final_decision == "human"]
    if human_rows:
        existing_review = read_tsv(HUMAN_REVIEW_PATH) if HUMAN_REVIEW_PATH.exists() else []
        existing_keys = {(r.get("sense_id"), r.get("clip_type")) for r in existing_review}
        appended = list(existing_review)
        for r in human_rows:
            if (r.sense_id, r.clip_type) in existing_keys:
                continue
            mr = rows_by_key.get((r.sense_id, r.clip_type), {})
            appended.append(
                {
                    "sense_id": r.sense_id,
                    "clip_type": r.clip_type,
                    "voice_id": mr.get("voice_id", ""),
                    "input_text": mr.get("text_input", ""),
                    "asr_transcript": r.transcript,
                    "similarity": str(r.similarity) if r.similarity is not None else "",
                    "phonetic_distance": "",
                    "url": mr.get("url", ""),
                    "added_at": _now_iso(),
                    "notes": "twice-failed ASR roundtrip",
                }
            )
        write_tsv(HUMAN_REVIEW_PATH, appended, fieldnames=HUMAN_REVIEW_FIELDS)
        print(
            f"[stage_6] {len(human_rows)} clip(s) routed to human review -> {HUMAN_REVIEW_PATH}",
            file=sys.stderr,
        )

    # Loudness verification spot-check (pilot only) — sample ~10 random PASS clips
    # per voice from cached MP3s and confirm |LUFS - target| ≤ 1.
    # (Full implementation deferred to follow-up; the per-clip post-encode
    # verification already happens implicitly via the loudnorm pass-2 output.)

    return {
        "senses": len(senses),
        "pending_at_start": len(pending),
        "results": len(results),
        "passed": sum(1 for r in results if r.final_decision == "pass"),
        "human": sum(1 for r in results if r.final_decision == "human"),
        "errored": sum(1 for r in results if r.final_decision == "err"),
        "asr_cost_usd": round(sum(r.cost_asr_usd for r in results), 4),
    }


__all__ = ["run", "preflight", "PreflightReport"]
