"""Stage 11 / Step 1 — Full deck re-render on eleven_v3.

Walks `data/_audio_manifest.tsv` (17,175 rows = 5,725 senses × {word,
example, en_ex}) and re-renders every clip that is not yet on eleven_v3.
The BP voice for `word`/`example` comes from `data/045-speaker_gender.tsv`
(post-Lair-swap state). The EN voice for `en_ex` is read from the
existing manifest row and PRESERVED across the regen — Michael C. Vincent
keeps voicing the ex-Lair senses.

Pipeline per clip:
    ElevenLabs PCM (eleven_v3, language=pt, latest pron-dict locator)
      -> ffmpeg loudnorm (-16 LUFS, closed-loop)
        -> Whisper ASR roundtrip (Stage 6 length-aware policy)
          -> R2 upload (audio/{sid}-{word|ex|en_ex}-eleven_v3-v1.mp3)
            -> manifest row updated in-place (status=uploaded)
              -> JSONL audit record appended

Resume: rows already at `tts_model=eleven_v3` + `status=uploaded` are
skipped. Manifest is flushed every 50 successful renders, so kills lose
at most ~50 rows of progress.

Pro tier concurrency cap on non-Flash models is 10 (verified 2026-05-13
via ElevenLabs docs `overview/models`); we cap at that.

Usage:
    .venv/bin/python build/11_1_full_regen_v3.py --dry-run
    .venv/bin/python build/11_1_full_regen_v3.py --limit 100 --yes
    .venv/bin/python build/11_1_full_regen_v3.py --yes
"""
from __future__ import annotations

import argparse
import csv
import json
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

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    MANIFEST_FIELDS,
    STATUS_FAILED_PERMANENT,
    STATUS_FAILED_TRANSIENT,
    STATUS_UPLOADED,
    object_key_for,
    read_manifest,
    text_hash,
    url_for,
    write_manifest,
)
from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client, R2Config  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
AUDIT_DIR = REPO_ROOT / "audit"

FINAL_TSV = DATA_DIR / "06-final.tsv"
VOICES_TSV = CONFIG_DIR / "voices.tsv"
SPEAKER_GENDER_TSV = DATA_DIR / "045-speaker_gender.tsv"

AUDIT_JSONL = AUDIT_DIR / "11_1_full_regen_v3.jsonl"
LOG_PATH = AUDIT_DIR / "11_1_full_regen_v3.log"

V3_MODEL_ID = "eleven_v3"
LANGUAGE_PT = "pt"
LANGUAGE_EN = "en"
DEFAULT_CONCURRENCY = 10            # Pro tier non-Flash cap (verified 2026-05-13)
MAX_CONCURRENCY = 10                # hard cap — exceeding stalls in queue
MANIFEST_FLUSH_EVERY = 50           # rows; lower => safer, higher => faster

# Hard-pinned pronunciation dictionary for PT clips. Do NOT use
# _latest_dict_locator() — that would pick v7 (Stage 9 review-queue
# rules), which the user explicitly rejected for this regen. v6 has 1
# rule (`hospital → ospitau`, the BP family-7 template). EN clips
# (`en_ex`) render with NO dictionary.
PT_DICT_LOCATOR = {
    "pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",   # anki-bp-stage-8-v6
    "version_id":                  "th9qzGumY1q3fkvV3Fi3",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_bp_voice_assignments() -> dict[str, tuple[str, str]]:
    """sense_id -> (bp_voice_id, voice_gender) from 045-speaker_gender.tsv."""
    out: dict[str, tuple[str, str]] = {}
    for r in read_tsv(SPEAKER_GENDER_TSV):
        out[r["sense_id"]] = (r["voice_id"], r.get("voice_gender_assigned", ""))
    return out


def _load_text_per_clip(sense_ids: set[str]) -> dict[tuple[str, str], str]:
    """{(sense_id, clip_type): text_input} from 06-final.tsv.

    clip_type ∈ {word, example, en_ex}.
    """
    out: dict[tuple[str, str], str] = {}
    rows = read_tsv(FINAL_TSV)
    for r in rows:
        sid = r["sense_id"]
        if sid not in sense_ids:
            continue
        # Use pt_display when present (handles articles + idiom forms);
        # fall back to pt for safety.
        word_text = (r.get("pt_display") or r.get("pt", "")).strip()
        example_pt = (r.get("example_pt") or "").strip()
        example_en = (r.get("example_en") or "").strip()
        out[(sid, "word")] = word_text
        out[(sid, "example")] = example_pt
        out[(sid, "en_ex")] = example_en
    return out


@dataclass
class RenderResult:
    sense_id: str
    clip_type: str
    voice_id: str
    text: str
    status: str           # 'uploaded' | 'failed_transient' | 'failed_permanent'
    language_code: str = ""   # 'pt' for word/example, 'en' for en_ex
    dictionary_id: str = ""   # PT v6 id for word/example, '' for en_ex
    url: str = ""
    md5: str = ""
    object_key: str = ""
    asr_transcript: str = ""
    asr_similarity: float = 0.0
    asr_decision: str = ""
    asr_attempts: int = 0
    asr_cost_usd: float = 0.0
    asr_notes: str = ""
    applied_gain_db: float = 0.0
    final_lufs: float = 0.0
    final_tp: float = 0.0
    loudness_within_tolerance: bool = False
    tp_limited: bool = False
    error: str = ""


def _render_one_clip(
    *,
    sense_id: str,
    clip_type: str,
    voice_id: str,
    text: str,
    rank: int,
    el: ElevenLabsClient,
    asr: AsrClient,
    r2: R2Client,
    public_base: str,
) -> RenderResult:
    """End-to-end: TTS → loudnorm → ASR → R2. Returns rich result.

    The caller selects `el` (el_pt for word/example, el_en for en_ex) so
    `language_code` and `pronunciation_dict_locators` are correct for the
    clip type. We record both in the result so the audit JSONL can be
    spot-checked post-hoc.
    """
    object_key = object_key_for(sense_id, clip_type, version=1, model_id=V3_MODEL_ID)
    dict_id = ""
    if el.pronunciation_dict_locators:
        dict_id = el.pronunciation_dict_locators[0].get(
            "pronunciation_dictionary_id", "")
    result = RenderResult(
        sense_id=sense_id, clip_type=clip_type, voice_id=voice_id,
        text=text, status=STATUS_FAILED_TRANSIENT, object_key=object_key,
        language_code=el.language_code, dictionary_id=dict_id,
    )

    if not text:
        result.status = STATUS_FAILED_PERMANENT
        result.error = "empty text"
        return result

    try:
        tts = el.generate_pcm(
            text=text, voice_id=voice_id,
            sense_id=sense_id, clip_type=clip_type, version=1,
        )
    except Exception as exc:  # noqa: BLE001
        result.error = f"tts: {type(exc).__name__}: {exc}"
        return result

    try:
        vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
    except Exception as exc:  # noqa: BLE001
        result.error = f"loudnorm: {type(exc).__name__}: {exc}"
        return result

    mp3 = vres.mp3_bytes
    result.applied_gain_db = vres.applied_gain_db
    result.final_lufs = vres.final_mp3_lufs
    result.final_tp = vres.final_mp3_tp
    result.loudness_within_tolerance = vres.within_tolerance
    result.tp_limited = vres.tp_limited

    # ASR roundtrip
    try:
        ar = asr_roundtrip(
            asr=asr, mp3_bytes=mp3, input_text=text,
            clip_type=clip_type, sense_id=sense_id,
            is_top_1000=(0 < rank <= 1000),
        )
        result.asr_transcript = ar.transcript
        result.asr_similarity = ar.text_similarity
        result.asr_decision = ar.decision
        result.asr_attempts = ar.attempts
        result.asr_cost_usd = ar.cost_usd
        result.asr_notes = ar.notes
    except Exception as exc:  # noqa: BLE001
        # ASR fail doesn't block upload — we still want the audio bytes.
        result.error = f"asr: {type(exc).__name__}: {exc}"

    try:
        upload = r2.upload_bytes(
            mp3, object_key,
            extra_metadata={
                "sense_id": sense_id, "clip_type": clip_type,
                "voice_id": voice_id, "model": V3_MODEL_ID,
                "stage": "11_1_full_regen",
            },
        )
        result.url = upload.url
        result.md5 = upload.content_md5
        result.status = STATUS_UPLOADED
    except Exception as exc:  # noqa: BLE001
        result.error = f"r2: {type(exc).__name__}: {exc}"

    return result


def _apply_result_to_row(row: dict, result: RenderResult, public_base: str) -> None:
    """Mutate manifest row in-place with the regen outcome."""
    row["tts_provider"] = "elevenlabs"
    row["tts_model"] = V3_MODEL_ID
    row["voice_id"] = result.voice_id
    row["text_input"] = result.text
    row["text_hash"] = text_hash(result.text) if result.text else ""
    row["object_key"] = result.object_key
    row["url"] = result.url
    row["version"] = "1"
    row["md5"] = result.md5
    row["asr_transcript"] = result.asr_transcript
    row["asr_similarity"] = f"{result.asr_similarity:.4f}" if result.asr_similarity else ""
    row["asr_decision"] = result.asr_decision
    row["applied_gain_db"] = f"{result.applied_gain_db:.3f}" if result.status == STATUS_UPLOADED else ""
    row["final_lufs"] = f"{result.final_lufs:.3f}" if result.status == STATUS_UPLOADED else ""
    row["final_tp"] = f"{result.final_tp:.3f}" if result.status == STATUS_UPLOADED else ""
    row["loudness_within_tolerance"] = "true" if result.loudness_within_tolerance else "false"
    row["tp_limited"] = "true" if result.tp_limited else "false"
    row["generated_at"] = _now_iso() if result.status == STATUS_UPLOADED else row.get("generated_at", "")
    row["status"] = result.status
    row["notes"] = result.error or ""


def _append_audit(rec: dict) -> None:
    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class _TeeStdout:
    def __init__(self, *streams) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for s in self.streams:
                try:
                    s.write(data); s.flush()
                except Exception:  # noqa: BLE001
                    pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try: s.flush()
            except Exception: pass  # noqa: BLE001


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"Worker count (default {DEFAULT_CONCURRENCY}; "
                         f"Pro tier non-Flash cap is {MAX_CONCURRENCY})")
    ap.add_argument("--limit", type=int, default=0,
                    help="Render only the first N candidates (smoke test). 0 = all.")
    ap.add_argument("--yes", action="store_true", help="Skip GO prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pre-flight only: count candidates, estimate cost.")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Also re-render rows already on eleven_v3 if "
                         "status != uploaded.")
    args = ap.parse_args()

    if args.concurrency > MAX_CONCURRENCY:
        print(f"WARN: clamping concurrency to {MAX_CONCURRENCY} (Pro tier cap)")
        args.concurrency = MAX_CONCURRENCY

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (needed for ASR)", file=sys.stderr)
        return 1

    # 1. Load manifest + voice assignments
    manifest = read_manifest()
    if not manifest:
        print(f"ERROR: manifest empty at {DEFAULT_MANIFEST_PATH}", file=sys.stderr)
        return 1
    rows_by_key = {(r["sense_id"], r["clip_type"]): r for r in manifest}
    bp_assign = _load_bp_voice_assignments()

    # 2. Identify candidates
    candidates: list[tuple[dict, str, str, str]] = []  # (row, voice_id, text, voice_gender)
    sense_ids = {r["sense_id"] for r in manifest}
    text_for = _load_text_per_clip(sense_ids)
    rank_for: dict[str, int] = {}
    for r in read_tsv(FINAL_TSV):
        try:
            rank_for[r["sense_id"]] = int(r.get("rank") or "0")
        except ValueError:
            rank_for[r["sense_id"]] = 0

    done = transient = pending = 0
    for row in manifest:
        is_v3 = row.get("tts_model") == V3_MODEL_ID
        is_uploaded = row.get("status") == STATUS_UPLOADED
        if is_v3 and is_uploaded:
            done += 1
            continue
        if is_v3 and not is_uploaded and not args.retry_failed:
            transient += 1
            continue

        sid = row["sense_id"]
        ct = row["clip_type"]
        if ct in ("word", "example"):
            voice_id, voice_gender = bp_assign.get(sid, ("", ""))
            if not voice_id:
                print(f"  WARN: no BP voice in 045 for {sid}; skipping {ct}",
                      file=sys.stderr)
                continue
        elif ct == "en_ex":
            # PRESERVE the existing en_ex voice (Michael C. Vincent for ex-Lair).
            voice_id = row.get("voice_id", "").strip()
            voice_gender = row.get("voice_gender", "")
            if not voice_id:
                print(f"  WARN: no EN voice in manifest for {sid} en_ex; skipping",
                      file=sys.stderr)
                continue
        else:
            print(f"  WARN: unknown clip_type {ct!r} for {sid}; skipping",
                  file=sys.stderr)
            continue

        text = text_for.get((sid, ct), "").strip()
        if not text:
            print(f"  WARN: no text for {sid} {ct}; skipping", file=sys.stderr)
            continue

        candidates.append((row, voice_id, text, voice_gender))
        pending += 1

    if args.limit and len(candidates) > args.limit:
        candidates = candidates[:args.limit]
        print(f"  --limit {args.limit}: rendering first {len(candidates)} only")

    # 3. Pre-flight summary
    n_chars = sum(len(text) for _, _, text, _ in candidates)
    by_ct: dict[str, int] = {}
    for row, _, _, _ in candidates:
        by_ct[row["clip_type"]] = by_ct.get(row["clip_type"], 0) + 1

    print()
    print(f"=== Stage 11.1 full v3 regen — pre-flight ===")
    print(f"  Manifest rows:    {len(manifest):,}")
    print(f"  Already on v3:    {done:,}  (skipped)")
    print(f"  Failed transient: {transient:,}  ({'re-render' if args.retry_failed else 'skipped'})")
    print(f"  To render:        {len(candidates):,}")
    print(f"    by clip_type:   word={by_ct.get('word',0):,}  "
          f"example={by_ct.get('example',0):,}  en_ex={by_ct.get('en_ex',0):,}")
    print(f"  Total chars:      {n_chars:,}")
    print(f"  TTS cost (est):   ${n_chars * 22.50e-6:.2f}  (v3 ~$22.50/1M chars)")
    print(f"  ASR cost (est):   ${len(candidates) * 0.00030:.2f}  "
          f"(Whisper-1 ~$0.006/min × ~3s/clip)")
    print(f"  PT clients:       lang=pt, dict={PT_DICT_LOCATOR['pronunciation_dictionary_id']} "
          f"(anki-bp-stage-8-v6)")
    print(f"  EN clients:       lang=en, dict=(none)")
    print(f"  Concurrency:      {args.concurrency}")
    print(f"  Manifest:         {DEFAULT_MANIFEST_PATH.name}")
    print(f"  Audit JSONL:      {AUDIT_JSONL.name}")
    print()

    if not candidates:
        print("Nothing to render. Done.")
        return 0

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0

    if not args.yes:
        sys.stdout.write("Type GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # 4. Tee stdout to log
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("a", encoding="utf-8")
    log.write(f"\n=== stage 11.1 full v3 regen started {_now_iso()} | "
              f"n={len(candidates)} concurrency={args.concurrency} ===\n")
    log.flush()
    original_stdout = sys.stdout
    sys.stdout = _TeeStdout(original_stdout, log)  # type: ignore[assignment]

    # 5. Clients — two ElevenLabsClient instances, one per language.
    #    PT clips (word/example): language_code=pt + v6 pronunciation dict.
    #    EN clips (en_ex):        language_code=en + NO dictionary.
    r2_config = R2Config.from_env()
    public_base = r2_config.public_base.rstrip("/")
    r2 = R2Client(r2_config)
    el_pt = ElevenLabsClient(
        model_id=V3_MODEL_ID,
        language_code=LANGUAGE_PT,
        pronunciation_dict_locators=[PT_DICT_LOCATOR],
    )
    el_en = ElevenLabsClient(
        model_id=V3_MODEL_ID,
        language_code=LANGUAGE_EN,
        pronunciation_dict_locators=None,
    )
    asr = AsrClient()

    # 6. Worker
    manifest_lock = threading.Lock()
    completed_since_flush = [0]
    n_uploaded = [0]
    n_failed = [0]
    n_total_cost = [0.0]
    t_start = time.time()

    def _worker(item):
        row, voice_id, text, voice_gender = item
        sid = row["sense_id"]
        ct = row["clip_type"]
        rank = rank_for.get(sid, 0)
        # Per-clip-type client selection. en_ex → English client (no dict);
        # word/example → Portuguese client (v6 dict).
        el = el_en if ct == "en_ex" else el_pt
        return row, _render_one_clip(
            sense_id=sid, clip_type=ct,
            voice_id=voice_id, text=text, rank=rank,
            el=el, asr=asr, r2=r2, public_base=public_base,
        )

    def _flush_manifest_locked() -> None:
        """Caller MUST hold manifest_lock."""
        write_manifest(manifest)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(_worker, c) for c in candidates]
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                row, res = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i:>5}/{len(candidates)}] WORKER-EXC "
                      f"{type(exc).__name__}: {exc}", flush=True)
                n_failed[0] += 1
                continue

            with manifest_lock:
                _apply_result_to_row(row, res, public_base)
                completed_since_flush[0] += 1
                if completed_since_flush[0] >= MANIFEST_FLUSH_EVERY:
                    _flush_manifest_locked()
                    completed_since_flush[0] = 0

            _append_audit({
                "ts": _now_iso(),
                "sense_id": res.sense_id, "clip_type": res.clip_type,
                "voice_id": res.voice_id, "model": V3_MODEL_ID,
                "language_code": res.language_code,
                "dictionary_id": res.dictionary_id,
                "status": res.status, "url": res.url, "md5": res.md5,
                "text_chars": len(res.text),
                "asr_decision": res.asr_decision,
                "asr_similarity": res.asr_similarity,
                "final_lufs": res.final_lufs,
                "loudness_within_tolerance": res.loudness_within_tolerance,
                "tp_limited": res.tp_limited,
                "asr_cost_usd": res.asr_cost_usd,
                "error": res.error,
            })

            if res.status == STATUS_UPLOADED:
                n_uploaded[0] += 1
            else:
                n_failed[0] += 1
            n_total_cost[0] += res.asr_cost_usd

            tag = "OK  " if res.status == STATUS_UPLOADED else "FAIL"
            err_short = (" err=" + res.error[:60]) if res.error else ""
            print(f"  [{i:>5}/{len(candidates)}] {tag} {res.sense_id} "
                  f"{res.clip_type:<7} v={res.voice_id[:8]}.. "
                  f"lufs={res.final_lufs:+.2f} asr={res.asr_decision or '-'} "
                  f"sim={res.asr_similarity:.2f}{err_short}",
                  flush=True)

    # 7. Final manifest flush
    with manifest_lock:
        _flush_manifest_locked()

    elapsed = time.time() - t_start

    # 8. Summary
    print()
    print("=== Stage 11.1 summary ===")
    print(f"  Rendered:     {len(candidates):,}")
    print(f"  Uploaded:     {n_uploaded[0]:,}")
    print(f"  Failed:       {n_failed[0]:,}")
    print(f"  ASR cost:     ${n_total_cost[0]:.4f}")
    print(f"  Wall time:    {elapsed:.0f}s "
          f"({elapsed/max(len(candidates),1):.2f}s/clip)")
    print(f"  Manifest:     written to {DEFAULT_MANIFEST_PATH}")
    print(f"  Audit JSONL:  {AUDIT_JSONL}")

    sys.stdout = original_stdout  # type: ignore[assignment]
    log.close()
    return 0 if n_failed[0] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
