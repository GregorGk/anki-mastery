"""Stage 16 / Step 9 — Surgical re-render of the 222 Gemini-flagged drifts.

Walks `data/_gemini_drift_watchlist.tsv` and per drift row attempts:

  Phase A — same-voice retries (max 2): bump version, re-render, judge.
  Phase B — voice swap: try active voices of the same gender ordered
            by deck-wide drift rate ascending (cleanest first).
  Phase C — hard case: revert to original (voice_id, v1) + log to
            data/_audio_known_drifts.tsv.

The first attempt that returns `bp_ok` becomes the published row in
`data/_audio_manifest.tsv`. The manifest is backed up to
`data/_audio_manifest.tsv.bak_pre_stage16_9` before the first mutation.

Reuses (no new client code):
  ElevenLabsClient.generate_pcm    build/lib/elevenlabs_client.py
  normalize_pcm_to_mp3_verified    build/lib/loudness.py
  asr_roundtrip                    build/lib/asr.py
  R2Client.upload_bytes            build/lib/r2_client.py
  bump_version / write_manifest    build/lib/audio_manifest.py
  GeminiAudioJudgeClient.judge     build/lib/gemini_audio_judge.py
  load_voices                      build/lib/voices.py

Usage:
    .venv/bin/python build/16_9_surgical_rerender.py --dry-run
    .venv/bin/python build/16_9_surgical_rerender.py --limit 5 --yes
    .venv/bin/python build/16_9_surgical_rerender.py --yes
    .venv/bin/python build/16_9_surgical_rerender.py --yes --only-voice "José Paulo"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    STATUS_FAILED_PERMANENT,
    STATUS_FAILED_TRANSIENT,
    STATUS_UPLOADED,
    bump_version,
    object_key_for,
    read_manifest,
    text_hash,
    url_for,
    write_manifest,
)
from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.gemini_audio_judge import GeminiAudioJudgeClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client, R2Config  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402
from build.lib.voices import load_voices, split_by_gender  # noqa: E402

# ─── Paths ──────────────────────────────────────────────────────────
DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
CONFIG = REPO_ROOT / "config"

WATCHLIST = DATA / "_gemini_drift_watchlist.tsv"
MANIFEST_BACKUP = DATA / "_audio_manifest.tsv.bak_pre_stage16_9"
FINAL_TSV = DATA / "06-final.tsv"
KNOWN_DRIFTS = DATA / "_audio_known_drifts.tsv"
AUDIT_JSONL = AUDIT / "16_9_surgical_rerenders.jsonl"
GEMINI_AUDIT_JSONL = AUDIT / "16_9_gemini_judge.jsonl"

# ─── Constants ──────────────────────────────────────────────────────
V3_MODEL_ID = "eleven_v3"
LANGUAGE_PT = "pt"
DEFAULT_WORKERS = 8
DEFAULT_MAX_SAME_VOICE_RETRIES = 2
MAX_TOTAL_ATTEMPTS_PER_ROW = 8

# Same v6 pronunciation dictionary as Stage 11 v3 regen.
PT_DICT_LOCATOR = {
    "pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
    "version_id":                  "th9qzGumY1q3fkvV3Fi3",
}

# Cost estimates (per-call, USD, marginal). ElevenLabs is covered by
# the subscription's character credit allowance, so TTS is treated as
# free — only ASR (Whisper) and Gemini judge cost real money.
COST_TTS_PER_ATTEMPT = 0.0
COST_ASR_PER_ATTEMPT = 0.001
COST_GEMINI_PER_ATTEMPT = 0.003


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Attempt:
    phase: str               # "A" (same-voice retry) or "B" (swap)
    attempt_idx: int         # 1-indexed within phase
    voice_id: str
    version: int             # post-bump version on the row
    object_key: str
    url: str
    verdict: str = ""        # bp_ok / non_bp / unclear / error
    severity: str = ""
    confidence: str = ""
    evidence: str = ""
    md5: str = ""
    asr_decision: str = ""
    final_lufs: float = 0.0
    error: str = ""
    elapsed_ms: int = 0


@dataclass
class RowState:
    sense_id: str
    clip_type: str
    text: str
    rank: int
    original_voice_id: str
    original_voice_name: str
    swap_pool: list[str] = field(default_factory=list)   # voice_ids in try-order
    attempts: list[Attempt] = field(default_factory=list)
    resolved: bool = False
    resolved_phase: str = ""           # "A" or "B" or "" (unresolved)
    resolved_voice_id: str = ""


# ─── State load helpers ─────────────────────────────────────────────
def _load_text_for(sense_ids: set[str]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in read_tsv(FINAL_TSV):
        if r["sense_id"] not in sense_ids:
            continue
        out[(r["sense_id"], "word")] = (r.get("pt_display") or r.get("pt", "")).strip()
        out[(r["sense_id"], "example")] = (r.get("example_pt") or "").strip()
    return out


def _load_rank_for(sense_ids: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in read_tsv(FINAL_TSV):
        if r["sense_id"] not in sense_ids:
            continue
        try:
            out[r["sense_id"]] = int(r.get("rank") or "0")
        except ValueError:
            out[r["sense_id"]] = 0
    return out


def _load_voice_meta() -> tuple[dict[str, str], dict[str, str], list[str], list[str]]:
    """Returns (voice_id→bp_name, voice_id→gender, female_voice_ids,
    male_voice_ids). Only active voices are included."""
    voices = load_voices()
    vid_name: dict[str, str] = {}
    vid_gender: dict[str, str] = {}
    for r in read_tsv(CONFIG / "voices.tsv"):
        if (r.get("status") or "active") != "active":
            continue
        vid_name[r["voice_id"]] = r.get("bp_name", "")
        vid_gender[r["voice_id"]] = r["gender"]
    female_ids, male_ids = split_by_gender(voices)
    return vid_name, vid_gender, female_ids, male_ids


def _deck_drift_rate(watchlist_rows: list[dict],
                     active_voice_ids: set[str]) -> dict[str, float]:
    """Compute drift rate per voice_id from the watchlist + manifest totals
    (caller passes the active set so we only score active voices)."""
    # Per-voice drift count from the watchlist.
    drift = {v: 0 for v in active_voice_ids}
    for r in watchlist_rows:
        vid = r["voice_id"]
        if vid in drift:
            drift[vid] += 1
    # Per-voice total BP v3 word+example uploaded clips.
    total = {v: 0 for v in active_voice_ids}
    for r in read_tsv(DATA / "_audio_manifest.tsv"):
        if (r["clip_type"] in ("word", "example")
                and r["tts_model"] == V3_MODEL_ID
                and r["status"] == "uploaded"
                and r["voice_id"] in total):
            total[r["voice_id"]] += 1
    return {v: (drift[v] / total[v]) if total[v] else 1.0 for v in active_voice_ids}


def _load_resolved_keys() -> set[tuple[str, str]]:
    """Walk audit/16_9_surgical_rerenders.jsonl for completed rows
    (resolved or logged as known-drift). Skip those on subsequent runs."""
    seen: set[tuple[str, str]] = set()
    if not AUDIT_JSONL.exists():
        return seen
    for line in AUDIT_JSONL.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") in ("row_resolved", "row_unresolved"):
            seen.add((rec.get("sense_id", ""), rec.get("clip_type", "")))
    return seen


# ─── Audit + manifest helpers ───────────────────────────────────────
_audit_lock = threading.Lock()


def _append_audit(rec: dict) -> None:
    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        with AUDIT_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _backup_manifest_if_needed() -> None:
    if not MANIFEST_BACKUP.exists() and DEFAULT_MANIFEST_PATH.exists():
        shutil.copy2(DEFAULT_MANIFEST_PATH, MANIFEST_BACKUP)
        print(f"  backed up manifest → {MANIFEST_BACKUP.name}")


def _write_known_drift_header_if_missing() -> None:
    if KNOWN_DRIFTS.exists():
        return
    cols = ["sense_id", "clip_type", "voice_id", "voice_name", "text",
            "attempts_total", "attempts_summary", "first_logged_at"]
    with KNOWN_DRIFTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()


def _append_known_drift(state: RowState, voice_names: dict[str, str]) -> None:
    _write_known_drift_header_if_missing()
    summary_parts = []
    for a in state.attempts:
        summary_parts.append(f"{a.phase}{a.attempt_idx}:{a.verdict or 'err'}"
                             f"({a.severity[:1] or '?'})")
    cols = ["sense_id", "clip_type", "voice_id", "voice_name", "text",
            "attempts_total", "attempts_summary", "first_logged_at"]
    row = {
        "sense_id": state.sense_id,
        "clip_type": state.clip_type,
        "voice_id": state.original_voice_id,
        "voice_name": state.original_voice_name,
        "text": state.text,
        "attempts_total": str(len(state.attempts)),
        "attempts_summary": " ".join(summary_parts),
        "first_logged_at": _now_iso(),
    }
    with _audit_lock:
        with KNOWN_DRIFTS.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, dialect="excel-tab",
                               quoting=csv.QUOTE_MINIMAL)
            w.writerow(row)


# ─── Render-one-attempt ─────────────────────────────────────────────
@dataclass
class RenderUploadResult:
    ok: bool
    object_key: str = ""
    url: str = ""
    md5: str = ""
    asr_decision: str = ""
    final_lufs: float = 0.0
    final_tp: float = 0.0
    applied_gain_db: float = 0.0
    asr_transcript: str = ""
    asr_similarity: float = 0.0
    error: str = ""


def _render_and_upload(
    *,
    sense_id: str, clip_type: str, voice_id: str, text: str, version: int,
    rank: int,
    el: ElevenLabsClient, asr: AsrClient, r2: R2Client,
) -> RenderUploadResult:
    """TTS → loudnorm → ASR → R2 for ONE attempt at the given version."""
    object_key = object_key_for(sense_id, clip_type, version, V3_MODEL_ID)
    out = RenderUploadResult(ok=False, object_key=object_key)

    try:
        tts = el.generate_pcm(text=text, voice_id=voice_id,
                              sense_id=sense_id, clip_type=clip_type,
                              version=version)
    except Exception as exc:  # noqa: BLE001
        out.error = f"tts: {type(exc).__name__}: {exc}"
        return out

    try:
        vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
    except Exception as exc:  # noqa: BLE001
        out.error = f"loudnorm: {type(exc).__name__}: {exc}"
        return out

    mp3 = vres.mp3_bytes
    out.final_lufs = vres.final_mp3_lufs
    out.final_tp = vres.final_mp3_tp
    out.applied_gain_db = vres.applied_gain_db

    try:
        ar = asr_roundtrip(
            asr=asr, mp3_bytes=mp3, input_text=text,
            clip_type=clip_type, sense_id=sense_id,
            is_top_1000=(0 < rank <= 1000),
        )
        out.asr_transcript = ar.transcript
        out.asr_similarity = ar.text_similarity
        out.asr_decision = ar.decision
    except Exception as exc:  # noqa: BLE001
        # ASR fail doesn't block upload — we still want the audio.
        out.error = f"asr: {type(exc).__name__}: {exc}"

    try:
        up = r2.upload_bytes(
            mp3, object_key,
            extra_metadata={
                "sense_id": sense_id, "clip_type": clip_type,
                "voice_id": voice_id, "model": V3_MODEL_ID,
                "stage": "16_9_surgical_rerender",
            },
        )
        out.url = up.url
        out.md5 = up.content_md5
    except Exception as exc:  # noqa: BLE001
        out.error = f"r2: {type(exc).__name__}: {exc}"
        return out

    # All steps OK (ASR error is soft — we still have a valid mp3 + url).
    out.ok = True
    return out


# ─── Manifest mutation ──────────────────────────────────────────────
def _apply_resolution_to_manifest_row(
    *,
    row: dict, voice_id: str, voice_gender: str, text: str, version: int,
    ru: RenderUploadResult, public_base: str,
) -> None:
    """Mutate the manifest row to point at the winning render."""
    row["tts_provider"] = "elevenlabs"
    row["tts_model"] = V3_MODEL_ID
    row["voice_id"] = voice_id
    row["voice_gender"] = voice_gender or row.get("voice_gender", "")
    row["text_input"] = text
    row["text_hash"] = text_hash(text) if text else ""
    row["object_key"] = ru.object_key
    row["url"] = ru.url or url_for(public_base, row["sense_id"],
                                    row["clip_type"], version, V3_MODEL_ID)
    row["version"] = str(version)
    row["md5"] = ru.md5
    row["asr_transcript"] = ru.asr_transcript
    row["asr_similarity"] = f"{ru.asr_similarity:.4f}" if ru.asr_similarity else ""
    row["asr_decision"] = ru.asr_decision
    row["applied_gain_db"] = f"{ru.applied_gain_db:.3f}"
    row["final_lufs"] = f"{ru.final_lufs:.3f}"
    row["final_tp"] = f"{ru.final_tp:.3f}"
    row["status"] = STATUS_UPLOADED
    row["generated_at"] = _now_iso()
    row["notes"] = "stage_16_9_rerender"


# ─── Per-row state machine ──────────────────────────────────────────
def _process_row(
    *,
    state: RowState, row: dict,
    el: ElevenLabsClient, asr: AsrClient, r2: R2Client,
    judge: GeminiAudioJudgeClient,
    public_base: str,
    max_same_voice_retries: int,
    vid_gender: dict[str, str],
    voice_names: dict[str, str],
) -> None:
    """Walk the Phase A → Phase B → Phase C state machine for ONE drift row.
    Mutates `row` in-place if a resolution is found."""
    starting_version = int(row.get("version", "1") or "1")

    def _attempt(phase: str, attempt_idx: int, voice_id: str) -> bool:
        """Returns True if the attempt resolved (bp_ok)."""
        nonlocal row
        # 1. Bump version on the row (this updates object_key + url too).
        bump_version(row, public_base, model_id=V3_MODEL_ID)
        new_version = int(row["version"])
        # If we're swapping voice, also update voice_id BEFORE rendering.
        if voice_id != row.get("voice_id", "") or phase == "B":
            row["voice_id"] = voice_id
            row["voice_gender"] = vid_gender.get(voice_id, row.get("voice_gender", ""))

        attempt = Attempt(
            phase=phase, attempt_idx=attempt_idx,
            voice_id=voice_id, version=new_version,
            object_key=row["object_key"], url=row["url"],
        )

        t0 = time.time()
        ru = _render_and_upload(
            sense_id=state.sense_id, clip_type=state.clip_type,
            voice_id=voice_id, text=state.text, version=new_version,
            rank=state.rank, el=el, asr=asr, r2=r2,
        )
        if not ru.ok:
            attempt.error = ru.error
            attempt.elapsed_ms = int((time.time() - t0) * 1000)
            state.attempts.append(attempt)
            _append_audit({
                "event": "attempt",
                "ts": _now_iso(),
                "sense_id": state.sense_id, "clip_type": state.clip_type,
                "phase": phase, "attempt_idx": attempt_idx,
                "voice_id": voice_id, "version": new_version,
                "verdict": "", "error": ru.error,
                "elapsed_ms": attempt.elapsed_ms,
            })
            return False

        attempt.object_key = ru.object_key
        attempt.url = ru.url
        attempt.md5 = ru.md5
        attempt.asr_decision = ru.asr_decision
        attempt.final_lufs = ru.final_lufs

        # 2. Judge the freshly-uploaded clip.
        try:
            jresult = judge.judge(
                audio_bytes=open(_local_or_none(ru.url), "rb").read()
                            if _local_or_none(ru.url) else _download(ru.url),
                audio_format="mp3", pt=state.text, ipa_word_final="",
                voice_id=voice_id, sense_id=state.sense_id,
                clip_type=state.clip_type,
            )
            attempt.verdict = jresult.pronunciation_verdict
            attempt.severity = jresult.severity
            attempt.confidence = jresult.confidence
            attempt.evidence = jresult.evidence
        except Exception as exc:  # noqa: BLE001
            attempt.verdict = "error"
            attempt.error = f"judge: {type(exc).__name__}: {exc}"

        attempt.elapsed_ms = int((time.time() - t0) * 1000)
        state.attempts.append(attempt)

        _append_audit({
            "event": "attempt", "ts": _now_iso(),
            "sense_id": state.sense_id, "clip_type": state.clip_type,
            "phase": phase, "attempt_idx": attempt_idx,
            "voice_id": voice_id, "version": new_version,
            "verdict": attempt.verdict,
            "severity": attempt.severity,
            "confidence": attempt.confidence,
            "evidence": attempt.evidence,
            "error": attempt.error,
            "object_key": ru.object_key,
            "elapsed_ms": attempt.elapsed_ms,
        })

        if attempt.verdict == "bp_ok":
            # Apply final manifest state for this resolved row.
            _apply_resolution_to_manifest_row(
                row=row, voice_id=voice_id,
                voice_gender=vid_gender.get(voice_id, ""),
                text=state.text, version=new_version,
                ru=ru, public_base=public_base,
            )
            state.resolved = True
            state.resolved_phase = phase
            state.resolved_voice_id = voice_id
            return True
        return False

    # Phase A — same-voice retries
    for i in range(1, max_same_voice_retries + 1):
        if len(state.attempts) >= MAX_TOTAL_ATTEMPTS_PER_ROW:
            break
        if _attempt("A", i, state.original_voice_id):
            return

    # Phase B — voice swap
    for j, swap_voice in enumerate(state.swap_pool, start=1):
        if len(state.attempts) >= MAX_TOTAL_ATTEMPTS_PER_ROW:
            break
        if _attempt("B", j, swap_voice):
            return

    # Phase C — hard case; revert manifest row to original v1
    _revert_to_original(row=row, state=state, public_base=public_base,
                        vid_gender=vid_gender)
    _append_known_drift(state, voice_names)


def _revert_to_original(*, row: dict, state: RowState, public_base: str,
                        vid_gender: dict[str, str]) -> None:
    """Restore the row to the pre-16.9 (original_voice_id, v1) state so the
    deck keeps using the original clip. The original v1 mp3 is still on R2."""
    row["voice_id"] = state.original_voice_id
    row["voice_gender"] = vid_gender.get(state.original_voice_id,
                                          row.get("voice_gender", ""))
    row["version"] = "1"
    row["object_key"] = object_key_for(state.sense_id, state.clip_type, 1, V3_MODEL_ID)
    row["url"] = url_for(public_base, state.sense_id, state.clip_type, 1, V3_MODEL_ID)
    row["status"] = STATUS_UPLOADED
    row["notes"] = "stage_16_9_known_drift"


# ─── Audio bytes for judge — prefer local cache, fall back to HTTP ──
def _local_or_none(url: str) -> str:
    """If the URL's basename matches a file in build/audio_cache/, return
    that path. Else return empty string (caller falls back to HTTP).
    The local cache only helps when the FRESH render path happens to match
    a previously-cached file — usually it won't, so most attempts fetch
    from R2."""
    basename = url.rsplit("/", 1)[-1]
    p = REPO_ROOT / "build" / "audio_cache" / basename
    if p.exists() and p.stat().st_size > 0:
        return str(p)
    return ""


_HTTP = None
_HTTP_LOCK = threading.Lock()


def _http():
    global _HTTP
    if _HTTP is None:
        with _HTTP_LOCK:
            if _HTTP is None:
                import httpx
                _HTTP = httpx.Client(
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=httpx.Timeout(60.0, connect=15.0),
                    limits=httpx.Limits(max_connections=64,
                                         max_keepalive_connections=64,
                                         keepalive_expiry=300.0),
                    transport=httpx.HTTPTransport(retries=5),
                    http2=False,
                )
    return _HTTP


def _download(url: str, max_attempts: int = 5) -> bytes:
    """Download the freshly-uploaded MP3 from R2 (with retries)."""
    import httpx
    client = _http()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            break
    raise last_exc  # type: ignore[misc]


# ─── Main ───────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--watchlist", type=Path, default=WATCHLIST)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-same-voice-retries", type=int,
                    default=DEFAULT_MAX_SAME_VOICE_RETRIES)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N drifts (smoke test).")
    ap.add_argument("--only-voice", type=str, default="",
                    help="Limit to drifts on a specific voice (by bp_name).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    # Load .env.
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())

    for key in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    if not args.watchlist.exists():
        print(f"ERROR: {args.watchlist} missing", file=sys.stderr)
        return 1

    watch = list(csv.DictReader(args.watchlist.open(encoding="utf-8"),
                                 dialect="excel-tab"))
    if args.only_voice:
        watch = [r for r in watch if r["voice_name"] == args.only_voice]
    skip_keys = _load_resolved_keys()
    watch = [r for r in watch
             if (r["sense_id"], r["clip_type"]) not in skip_keys]
    if args.limit and len(watch) > args.limit:
        watch = watch[: args.limit]

    if not watch:
        print("No drift rows to process. Done.")
        return 0

    # ── Manifest + side tables ──────────────────────────────────────
    manifest = read_manifest()
    rows_by_key: dict[tuple[str, str], dict] = {(r["sense_id"], r["clip_type"]): r
                                                  for r in manifest}
    sense_ids = {r["sense_id"] for r in watch}
    text_for = _load_text_for(sense_ids)
    rank_for = _load_rank_for(sense_ids)
    voice_names, vid_gender, female_ids, male_ids = _load_voice_meta()
    active_voice_ids = set(voice_names.keys())
    drift_rates = _deck_drift_rate(
        list(csv.DictReader(WATCHLIST.open(encoding="utf-8"), dialect="excel-tab")),
        active_voice_ids,
    )

    # ── Build per-row state ─────────────────────────────────────────
    states: list[RowState] = []
    skipped_no_manifest_row = 0
    skipped_no_text = 0
    for r in watch:
        sid = r["sense_id"]; ct = r["clip_type"]; vid = r["voice_id"]
        if (sid, ct) not in rows_by_key:
            skipped_no_manifest_row += 1
            continue
        text = text_for.get((sid, ct), "").strip()
        if not text:
            skipped_no_text += 1
            continue
        # Build swap pool: active voices of same gender, cleanest first,
        # excluding the current voice.
        gender = vid_gender.get(vid, "")
        pool = female_ids if gender == "female" else male_ids
        pool = [v for v in pool if v != vid]
        pool.sort(key=lambda v: drift_rates.get(v, 1.0))
        state = RowState(
            sense_id=sid, clip_type=ct, text=text,
            rank=rank_for.get(sid, 0),
            original_voice_id=vid,
            original_voice_name=r.get("voice_name", ""),
            swap_pool=pool,
        )
        states.append(state)

    # ── Pre-flight ──────────────────────────────────────────────────
    max_attempts_per_row = (args.max_same_voice_retries
                             + max((len(s.swap_pool) for s in states), default=0))
    worst_total = len(states) * max_attempts_per_row
    expected_total = int(len(states) * 1.5)  # rough guess (most resolve in 1-2 tries)
    cost_per_attempt = (COST_TTS_PER_ATTEMPT + COST_ASR_PER_ATTEMPT
                         + COST_GEMINI_PER_ATTEMPT)
    n_chars = sum(len(s.text) for s in states)

    print()
    print(f"=== Stage 16.9 surgical re-render — pre-flight ===")
    print(f"  watchlist rows:              {len(watch):,}")
    if skipped_no_manifest_row:
        print(f"  skipped (no manifest row):   {skipped_no_manifest_row}")
    if skipped_no_text:
        print(f"  skipped (no text):           {skipped_no_text}")
    if skip_keys:
        print(f"  already resolved (audit):    {len(skip_keys)} (resuming)")
    print(f"  drifts to process:           {len(states):,}")
    print(f"  workers:                     {args.workers}")
    print(f"  same-voice retries (max):    {args.max_same_voice_retries}")
    print(f"  swap pool sizes by gender:   F={len(female_ids)-1} "
          f"M={len(male_ids)-1}  (excluding current voice)")
    print(f"  total chars:                 {n_chars:,}")
    print(f"  cost per attempt (est):      ${cost_per_attempt:.4f}")
    print(f"  expected attempts (~1.5×):   {expected_total:,}  "
          f"→ ${expected_total * cost_per_attempt:.2f}")
    print(f"  worst-case attempts:         {worst_total:,}  "
          f"→ ${worst_total * cost_per_attempt:.2f}")
    print(f"  manifest:                    {DEFAULT_MANIFEST_PATH.name}")
    print(f"  manifest backup:             {MANIFEST_BACKUP.name}")
    print(f"  audit JSONL:                 {AUDIT_JSONL.name}")
    print(f"  known drifts:                {KNOWN_DRIFTS.name}")
    print()
    print(f"  drift-aware swap order:")
    for vid in female_ids:
        rate = drift_rates.get(vid, 1.0)
        print(f"    F {voice_names.get(vid,vid[:8])}: drift_rate={rate*100:.2f}%")
    for vid in male_ids:
        rate = drift_rates.get(vid, 1.0)
        print(f"    M {voice_names.get(vid,vid[:8])}: drift_rate={rate*100:.2f}%")
    print()

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0

    if not args.yes:
        sys.stdout.write("Type GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    _backup_manifest_if_needed()

    # ── Clients ─────────────────────────────────────────────────────
    r2_config = R2Config.from_env()
    public_base = r2_config.public_base.rstrip("/")
    r2 = R2Client(r2_config)
    el = ElevenLabsClient(
        model_id=V3_MODEL_ID, language_code=LANGUAGE_PT,
        pronunciation_dict_locators=[PT_DICT_LOCATOR],
    )
    asr = AsrClient()
    judge = GeminiAudioJudgeClient(model="gemini-3.1-pro-preview",
                                    audit_path=GEMINI_AUDIT_JSONL)

    # ── Worker loop ─────────────────────────────────────────────────
    manifest_lock = threading.Lock()
    n_done = 0
    n_resolved_A = 0
    n_resolved_B = 0
    n_unresolved = 0
    t_start = time.time()

    def _worker(state: RowState) -> RowState:
        row = rows_by_key[(state.sense_id, state.clip_type)]
        with manifest_lock:
            # Snapshot a fresh row dict so we can mutate freely; we apply
            # back to the shared `row` only on success or finalization.
            row_local = dict(row)
        _process_row(
            state=state, row=row_local,
            el=el, asr=asr, r2=r2, judge=judge,
            public_base=public_base,
            max_same_voice_retries=args.max_same_voice_retries,
            vid_gender=vid_gender, voice_names=voice_names,
        )
        with manifest_lock:
            # Copy the mutated fields back to the canonical manifest row.
            row.update(row_local)
        return state

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, s) for s in states]
        for fut in as_completed(futs):
            state = fut.result()
            n_done += 1
            # CRITICAL: flush manifest to disk BEFORE writing the
            # row_{resolved,unresolved} audit event. The skip-on-resume
            # logic uses those events as ground truth — if we logged
            # the event but failed to persist the manifest mutation
            # (crash, kill), a future run would skip this row while
            # the on-disk manifest still pointed to the drifting v1
            # audio. Flushing first means a crash leaves us in a
            # consistent "not yet resolved" state.
            with manifest_lock:
                write_manifest(manifest)

            if state.resolved and state.resolved_phase == "A":
                n_resolved_A += 1
                _append_audit({
                    "event": "row_resolved", "ts": _now_iso(),
                    "sense_id": state.sense_id, "clip_type": state.clip_type,
                    "phase": "A", "voice_id": state.resolved_voice_id,
                    "attempts": len(state.attempts),
                })
            elif state.resolved and state.resolved_phase == "B":
                n_resolved_B += 1
                _append_audit({
                    "event": "row_resolved", "ts": _now_iso(),
                    "sense_id": state.sense_id, "clip_type": state.clip_type,
                    "phase": "B", "voice_id": state.resolved_voice_id,
                    "attempts": len(state.attempts),
                })
            else:
                n_unresolved += 1
                _append_audit({
                    "event": "row_unresolved", "ts": _now_iso(),
                    "sense_id": state.sense_id, "clip_type": state.clip_type,
                    "attempts": len(state.attempts),
                })

            if n_done % 10 == 0 or n_done == len(states):
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 0.01)
                eta = (len(states) - n_done) / max(rate, 0.001)
                print(f"  [{n_done:>4}/{len(states)}] "
                      f"A={n_resolved_A} B={n_resolved_B} hard={n_unresolved}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f} min", flush=True)

    # Final flush.
    with manifest_lock:
        write_manifest(manifest)

    elapsed = time.time() - t_start
    total_attempts = sum(len(s.attempts) for s in states)

    print()
    print(f"=== Stage 16.9 summary ===")
    print(f"  drifts processed:    {n_done:,}")
    print(f"  resolved (phase A):  {n_resolved_A:,}  (same-voice retry)")
    print(f"  resolved (phase B):  {n_resolved_B:,}  (voice swap)")
    print(f"  unresolved:          {n_unresolved:,}  (logged → {KNOWN_DRIFTS.name})")
    print(f"  total attempts:      {total_attempts:,}")
    print(f"  est cost:            ${total_attempts * cost_per_attempt:.2f}")
    print(f"  wall:                {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  audit:               {AUDIT_JSONL.name}")
    print(f"  manifest:            updated in place; backup at {MANIFEST_BACKUP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
