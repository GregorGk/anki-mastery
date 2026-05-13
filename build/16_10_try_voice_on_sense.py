"""Stage 16 / Step 10 — Targeted hard-case override: try a specific
voice on a specific (sense_id, clip_type) and accept the result if
Gemini agrees it sounds BP.

Use this for the residual drifts left over after Stage 16.9 — words
where every active voice in the same gender pool failed. The script
renders ONE clip, judges it with Gemini, and:

  - on bp_ok:  bumps the manifest row's version, swaps voice_id,
                uploads new mp3 to R2, removes the sense from
                data/_audio_known_drifts.tsv, appends an audit record
  - on non_bp: prints Gemini's evidence, leaves manifest untouched,
                leaves the known-drift row in place

Unlike Stage 16.9 this script does NOT consult voices.tsv's `status`
column — it accepts any voice_id you pass, including `active_escape_hatch`
voices that are documented but excluded from round-robin assignment.

Usage:
    .venv/bin/python build/16_10_try_voice_on_sense.py \
        --sense-id 2687.00.01 --clip-type word \
        --voice-id PznTnBc8X6pvixs9UkQm \
        --voice-name "Dani"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
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

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"

FINAL_TSV = DATA / "06-final.tsv"
KNOWN_DRIFTS = DATA / "_audio_known_drifts.tsv"
AUDIT_JSONL = AUDIT / "16_10_voice_overrides.jsonl"
GEMINI_AUDIT_JSONL = AUDIT / "16_10_gemini_judge.jsonl"

V3_MODEL_ID = "eleven_v3"
PT_DICT_LOCATOR = {
    "pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
    "version_id":                  "th9qzGumY1q3fkvV3Fi3",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def _text_for(sense_id: str, clip_type: str) -> str:
    for r in read_tsv(FINAL_TSV):
        if r["sense_id"] != sense_id:
            continue
        if clip_type == "word":
            return (r.get("pt_display") or r.get("pt", "")).strip()
        if clip_type == "example":
            return (r.get("example_pt") or "").strip()
        return ""
    return ""


def _rank_for(sense_id: str) -> int:
    for r in read_tsv(FINAL_TSV):
        if r["sense_id"] == sense_id:
            try:
                return int(r.get("rank") or "0")
            except ValueError:
                return 0
    return 0


def _remove_from_known_drifts(sense_id: str, clip_type: str) -> bool:
    if not KNOWN_DRIFTS.exists():
        return False
    rows = list(csv.DictReader(KNOWN_DRIFTS.open(encoding="utf-8"),
                                dialect="excel-tab"))
    kept = [r for r in rows
            if not (r["sense_id"] == sense_id and r["clip_type"] == clip_type)]
    if len(kept) == len(rows):
        return False
    cols = list(rows[0].keys()) if rows else ["sense_id", "clip_type", "voice_id",
                                               "voice_name", "text",
                                               "attempts_total",
                                               "attempts_summary",
                                               "first_logged_at"]
    with KNOWN_DRIFTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in kept:
            w.writerow(r)
    return True


def _append_audit(rec: dict) -> None:
    AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sense-id", required=True)
    ap.add_argument("--clip-type", choices=("word", "example"), required=True)
    ap.add_argument("--voice-id", required=True,
                    help="ElevenLabs voice_id (escape-hatch voices accepted).")
    ap.add_argument("--voice-name", default="",
                    help="Display name (audit only).")
    ap.add_argument("--voice-gender", default="female",
                    choices=("female", "male"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _load_env()
    for key in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set", file=sys.stderr)
            return 1

    text = _text_for(args.sense_id, args.clip_type)
    if not text:
        print(f"ERROR: no text for {args.sense_id} {args.clip_type} in {FINAL_TSV.name}",
              file=sys.stderr)
        return 1
    rank = _rank_for(args.sense_id)

    manifest = read_manifest()
    rows_by_key = {(r["sense_id"], r["clip_type"]): r for r in manifest}
    row = rows_by_key.get((args.sense_id, args.clip_type))
    if not row:
        print(f"ERROR: no manifest row for {args.sense_id} {args.clip_type}",
              file=sys.stderr)
        return 1

    print(f"=== Stage 16.10 voice override ===")
    print(f"  sense:       {args.sense_id} {args.clip_type}")
    print(f"  text:        {text!r}")
    print(f"  current row: voice={row['voice_id'][:10]} version={row['version']} "
          f"url=...{row['url'][-40:]}")
    print(f"  trying:      voice={args.voice_id[:10]} ({args.voice_name})")
    print()

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to proceed: "); sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # Clients (same setup as 16_9 — pt + v6 dict).
    r2_config = R2Config.from_env()
    public_base = r2_config.public_base.rstrip("/")
    r2 = R2Client(r2_config)
    el = ElevenLabsClient(
        model_id=V3_MODEL_ID, language_code="pt",
        pronunciation_dict_locators=[PT_DICT_LOCATOR],
    )
    asr = AsrClient()
    judge = GeminiAudioJudgeClient(model="gemini-3.1-pro-preview",
                                    audit_path=GEMINI_AUDIT_JSONL)

    # Mutate row: bump version + swap voice_id.
    bump_version(row, public_base, model_id=V3_MODEL_ID)
    row["voice_id"] = args.voice_id
    row["voice_gender"] = args.voice_gender
    new_version = int(row["version"])
    object_key = row["object_key"]
    url = row["url"]

    print(f"  → bumped to v{new_version}, object_key={object_key}")
    print(f"  → rendering...", flush=True)

    t0 = time.time()
    tts = el.generate_pcm(text=text, voice_id=args.voice_id,
                          sense_id=args.sense_id, clip_type=args.clip_type,
                          version=new_version)
    vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
    mp3 = vres.mp3_bytes
    print(f"  → rendered + loudnormed ({time.time()-t0:.1f}s, "
          f"LUFS={vres.final_mp3_lufs:.1f}, TP={vres.final_mp3_tp:.1f})")

    print(f"  → ASR roundtrip...", flush=True)
    ar = asr_roundtrip(asr=asr, mp3_bytes=mp3, input_text=text,
                       clip_type=args.clip_type, sense_id=args.sense_id,
                       is_top_1000=(0 < rank <= 1000))
    print(f"  → ASR: {ar.decision} (sim={ar.text_similarity:.3f}, "
          f"transcript={ar.transcript!r})")

    print(f"  → uploading to R2...", flush=True)
    up = r2.upload_bytes(
        mp3, object_key,
        extra_metadata={"sense_id": args.sense_id, "clip_type": args.clip_type,
                         "voice_id": args.voice_id, "model": V3_MODEL_ID,
                         "stage": "16_10_voice_override"},
    )
    print(f"  → uploaded: {up.url}")

    print(f"  → judging with Gemini...", flush=True)
    jr = judge.judge(
        audio_bytes=mp3, audio_format="mp3", pt=text,
        ipa_word_final="",
        voice_id=args.voice_id, sense_id=args.sense_id,
        clip_type=args.clip_type,
    )
    print()
    print(f"  Gemini verdict: {jr.pronunciation_verdict}")
    print(f"    severity:   {jr.severity}")
    print(f"    confidence: {jr.confidence}")
    print(f"    evidence:   {jr.evidence}")
    print()

    accepted = jr.pronunciation_verdict == "bp_ok"
    if accepted:
        # Apply final manifest mutation
        row["tts_provider"] = "elevenlabs"
        row["tts_model"] = V3_MODEL_ID
        row["text_input"] = text
        row["text_hash"] = text_hash(text)
        row["url"] = up.url
        row["md5"] = up.content_md5
        row["asr_transcript"] = ar.transcript
        row["asr_similarity"] = f"{ar.text_similarity:.4f}"
        row["asr_decision"] = ar.decision
        row["applied_gain_db"] = f"{vres.applied_gain_db:.3f}"
        row["final_lufs"] = f"{vres.final_mp3_lufs:.3f}"
        row["final_tp"] = f"{vres.final_mp3_tp:.3f}"
        row["loudness_within_tolerance"] = "true" if vres.within_tolerance else "false"
        row["tp_limited"] = "true" if vres.tp_limited else "false"
        row["status"] = STATUS_UPLOADED
        row["generated_at"] = _now_iso()
        row["notes"] = "stage_16_10_voice_override"
        write_manifest(manifest)
        removed = _remove_from_known_drifts(args.sense_id, args.clip_type)
        print(f"  ACCEPTED → manifest updated, "
              f"{'removed from' if removed else 'not in'} known_drifts")
    else:
        # Revert row so it points back at the original v1.
        row["version"] = "1"
        row["voice_id"] = row.get("voice_id", "")  # leave current swap for inspection
        row["object_key"] = object_key_for(args.sense_id, args.clip_type, 1,
                                            V3_MODEL_ID)
        row["url"] = url_for(public_base, args.sense_id, args.clip_type, 1,
                              V3_MODEL_ID)
        row["status"] = STATUS_UPLOADED
        # Leave notes as the previous known_drift mark; don't touch.
        write_manifest(manifest)
        print(f"  REJECTED → manifest reverted to v1, known_drift retained")

    _append_audit({
        "event": "voice_override_attempt", "ts": _now_iso(),
        "sense_id": args.sense_id, "clip_type": args.clip_type,
        "voice_id": args.voice_id, "voice_name": args.voice_name,
        "version_tried": new_version, "object_key": object_key,
        "url": up.url,
        "verdict": jr.pronunciation_verdict,
        "severity": jr.severity, "confidence": jr.confidence,
        "evidence": jr.evidence,
        "asr_decision": ar.decision, "asr_similarity": ar.text_similarity,
        "final_lufs": vres.final_mp3_lufs,
        "accepted": accepted,
    })
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
