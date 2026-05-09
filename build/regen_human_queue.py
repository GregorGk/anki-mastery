"""Single-attempt regen pass on the residual HUMAN review queue.

After Stage 6 / Stage 7 finishes, some clips end up routed to
`data/_audio_human_review.tsv` because they failed ASR roundtrip after the
default 2 attempts. This script gives each of those clips ONE more chance
with fresh ElevenLabs audio (bumped version → new deterministic seed →
different audio bytes).

Per-clip flow:
  1. Read the manifest row by (sense_id, clip_type).
  2. Bump version (e.g. v2 → v3) → new R2 object key.
  3. Generate fresh ElevenLabs PCM with the new seed.
  4. Closed-loop loudness normalization.
  5. ASR roundtrip.
  6. If PASS: upload to R2, update manifest, remove from human-review TSV.
     If REGEN/HUMAN: leave manifest as-is (clip stays in human-review queue
                     with the new attempt's transcript appended to notes).

Idempotent: a second run on the same input retries again with another
version bump. Costs ~$0.0002/clip in ASR + a handful of ElevenLabs credits.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.audio_logger import AsrOutcome, AudioLogger, ERR, HUMAN, PASS  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    STATUS_UPLOADED,
    bump_version,
    index_by_key,
    read_manifest,
    write_manifest,
)
from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE_DIR = REPO_ROOT / "build" / "audio_cache"
HUMAN_REVIEW_PATH = DATA_DIR / "_audio_human_review.tsv"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
LOG_JSONL = AUDIT_DIR / "06_audio.jsonl"
LOG_TRANSCRIPT = AUDIT_DIR / "06_audio_transcript.log"

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
TOP_1000_RANK_THRESHOLD = 1000


@dataclass
class RegenResult:
    sense_id: str
    clip_type: str
    decision: str       # 'pass' | 'human' | 'err'
    new_version: int
    transcript: str
    similarity: float
    notes: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_one(
    *,
    sid: str,
    ctype: str,
    rank_by_sid: dict[str, int],
    rows_by_key: dict[tuple[str, str], dict],
    el: ElevenLabsClient,
    asr: AsrClient,
    r2: R2Client,
    logger: AudioLogger,
) -> RegenResult:
    key = (sid, ctype)
    row = rows_by_key.get(key)
    if row is None:
        return RegenResult(sid, ctype, "err", 0, "", 0.0, "manifest row missing")

    voice_id = row["voice_id"]
    text = row["text_input"]
    rank = rank_by_sid.get(sid, 0)
    is_top = 0 < rank <= TOP_1000_RANK_THRESHOLD

    # Bump version BEFORE generation so the new seed differs from prior attempts.
    bump_version(row, public_base=r2.config.public_base)
    version = int(row["version"])
    logger.started(sid, ctype, voice_id, text, attempt=99)  # 99 marks regen-pass

    # 1. ElevenLabs PCM (fresh seed because version bumped)
    try:
        tts = el.generate_pcm(
            text=text,
            voice_id=voice_id,
            sense_id=sid,
            clip_type=ctype,
            version=version,
        )
    except Exception as exc:
        logger.errored(sid, ctype, voice_id, 99, type(exc).__name__, str(exc), stage="tts")
        return RegenResult(sid, ctype, "err", version, "", 0.0, f"tts: {type(exc).__name__}")

    # 2. Closed-loop loudness norm
    try:
        vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
        mp3_bytes = vres.mp3_bytes
    except Exception as exc:
        logger.errored(sid, ctype, voice_id, 99, type(exc).__name__, str(exc), stage="loudnorm")
        return RegenResult(sid, ctype, "err", version, "", 0.0, f"loudnorm: {str(exc)[:60]}")

    # Cache MP3 locally
    cache = AUDIO_CACHE_DIR / f"{sid}-{ctype}-v{version}.mp3"
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(mp3_bytes)
    except OSError:
        pass

    # 3. ASR roundtrip (single shot — no retry inside this script)
    try:
        ar = asr_roundtrip(
            asr=asr,
            mp3_bytes=mp3_bytes,
            input_text=text,
            clip_type=ctype,
            sense_id=sid,
            is_top_1000=is_top,
        )
    except Exception as exc:
        logger.errored(sid, ctype, voice_id, 99, type(exc).__name__, str(exc), stage="asr")
        return RegenResult(sid, ctype, "err", version, "", 0.0, f"asr: {str(exc)[:60]}")

    # 4. Always upload (ASR pass or fail) so deck has audio at this version.
    try:
        upload = r2.upload_bytes(
            mp3_bytes,
            row["object_key"],
            extra_metadata={
                "sense_id": sid,
                "clip_type": ctype,
                "voice_id": voice_id,
                "version": str(version),
                "asr_decision": ar.decision,
                "regen_pass": "true",
            },
        )
    except Exception as exc:
        logger.errored(sid, ctype, voice_id, 99, type(exc).__name__, str(exc), stage="r2")
        return RegenResult(sid, ctype, "err", version, "", 0.0, f"r2: {str(exc)[:60]}")

    # 5. Update manifest
    decision_tag = PASS if ar.decision == "pass" else HUMAN
    row["url"] = upload.url
    row["md5"] = upload.content_md5
    row["asr_transcript"] = ar.transcript
    row["asr_similarity"] = str(ar.text_similarity)
    row["asr_decision"] = ar.decision
    row["applied_gain_db"] = f"{vres.applied_gain_db:.3f}"
    row["final_lufs"] = f"{vres.final_mp3_lufs:.3f}"
    row["final_tp"] = f"{vres.final_mp3_tp:.3f}"
    row["loudness_within_tolerance"] = "true" if vres.within_tolerance else "false"
    row["tp_limited"] = "true" if vres.tp_limited else "false"
    row["status"] = STATUS_UPLOADED
    row["generated_at"] = _now_iso()
    if ar.decision == "pass":
        row["notes"] = f"recovered via regen-human-queue v{version}"
    else:
        row["notes"] = f"regen-human-queue v{version} still failed: '{ar.transcript[:50]}' (sim {ar.text_similarity:.2f})"

    outcome = AsrOutcome(
        sense_id=sid,
        clip_type=ctype,
        voice_id=voice_id,
        input_text=text,
        asr_transcript=ar.transcript,
        similarity=ar.text_similarity,
        phonetic_distance=ar.phonetic_distance,
        decision=decision_tag,
        attempt=99,
        latency_ms=0,
        cost_usd=ar.cost_usd,
        note=f"regen-human-queue v{version}",
    )
    logger.asr_completed(outcome)

    return RegenResult(
        sid, ctype, "pass" if ar.decision == "pass" else "human",
        version, ar.transcript, ar.text_similarity,
        f"v{version} {ar.decision}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if not HUMAN_REVIEW_PATH.exists():
        print("[regen] no human review queue; nothing to do.", file=sys.stderr)
        return 0
    queue = read_tsv(HUMAN_REVIEW_PATH)
    if not queue:
        print("[regen] human review queue empty.", file=sys.stderr)
        return 0

    manifest = read_manifest()
    rows_by_key = index_by_key(manifest)

    # Need rank for top-1000 ASR threshold
    ipa_rows = read_tsv(IPA_PATH) if IPA_PATH.exists() else []
    rank_by_sid = {r["sense_id"]: int(r.get("rank", 0) or 0) for r in ipa_rows}

    el = ElevenLabsClient()
    asr = AsrClient()
    r2 = R2Client()
    logger = AudioLogger(LOG_JSONL, LOG_TRANSCRIPT)

    print(f"[regen] processing {len(queue)} clips from human review queue", file=sys.stderr)

    results: list[RegenResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _process_one,
                sid=q["sense_id"],
                ctype=q["clip_type"],
                rank_by_sid=rank_by_sid,
                rows_by_key=rows_by_key,
                el=el,
                asr=asr,
                r2=r2,
                logger=logger,
            )
            for q in queue
        ]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            tag = {"pass": "PASS ", "human": "HUMAN", "err": "ERR  "}.get(r.decision, "?")
            print(f"  {tag}  {r.sense_id} {r.clip_type:<7}  v{r.new_version}  -> {r.transcript[:40]!r}", file=sys.stderr)

    # Persist updated manifest
    write_manifest(manifest)

    # Re-write human review queue with only the still-failing clips
    still_failing = []
    for q in queue:
        sid, ctype = q["sense_id"], q["clip_type"]
        rs = [r for r in results if r.sense_id == sid and r.clip_type == ctype]
        if not rs:
            still_failing.append(q)
            continue
        r = rs[0]
        if r.decision == "pass":
            continue  # recovered → drop from queue
        # human or err: update the queue entry with latest attempt info
        new_q = dict(q)
        new_q["asr_transcript"] = r.transcript
        new_q["similarity"] = str(r.similarity)
        new_q["url"] = rows_by_key[(sid, ctype)]["url"]
        new_q["added_at"] = _now_iso()
        new_q["notes"] = f"still failing after regen-human-queue v{r.new_version}: {r.notes}"
        still_failing.append(new_q)
    write_tsv(HUMAN_REVIEW_PATH, still_failing, fieldnames=HUMAN_REVIEW_FIELDS)

    # Summary
    passed = sum(1 for r in results if r.decision == "pass")
    human = sum(1 for r in results if r.decision == "human")
    err = sum(1 for r in results if r.decision == "err")
    print(file=sys.stderr)
    print(f"=== regen-human-queue summary ===", file=sys.stderr)
    print(f"  total processed:   {len(results)}", file=sys.stderr)
    print(f"  recovered (PASS):  {passed}", file=sys.stderr)
    print(f"  still HUMAN:       {human}", file=sys.stderr)
    print(f"  errored:           {err}", file=sys.stderr)
    print(f"  remaining queue:   {len(still_failing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
