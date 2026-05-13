"""Stage 16 / Step 1 — Build the AudioJudge A/B test pool.

Samples 100 senses from `data/06-final.tsv` (mix of clip_types and BP
voices) and produces a side TSV that pairs each sense's existing BP
v3 audio with a freshly-rendered European Portuguese version using
Nelson Silvestre (`hOLl3246BMBsdy0qtYLb`, ElevenLabs "Lisbon accent"
narrator).

Output:
    data/_audio_judge_ab.tsv      — 200 rows (100 BP + 100 EP), one per clip
    audio_judge_ab/{sid}-{ct}-eleven_v3-ep.mp3   — new EP renders on R2
    (BP clips already exist at audio/{sid}-{ct}-eleven_v3-v1.mp3 from the
     full regen.)

Usage:
    .venv/bin/python build/16_1_audio_judge_ab_prep.py --dry-run
    .venv/bin/python build/16_1_audio_judge_ab_prep.py --yes

Cost: 100 EP clips × ~30 chars × $22.50/1M ≈ $0.07. Wall ~5–10 min.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client, R2Config  # noqa: E402

DATA = REPO_ROOT / "data"
FINAL_TSV = DATA / "06-final.tsv"
OUT_TSV = DATA / "_audio_judge_ab.tsv"

V3_MODEL_ID = "eleven_v3"
EP_VOICE_ID = "hOLl3246BMBsdy0qtYLb"
EP_VOICE_NAME = "Nelson Silvestre"
EP_VOICE_DESC = "Calm, Lisbon accent (European Portuguese narrator)"
R2_PREFIX = "audio_judge_ab"
SEED = 16
N_SAMPLES = 100
WORKERS = 4   # leave headroom for the full regen still consuming concurrency=10

FIELDS = [
    "row_id", "sense_id", "clip_type", "dialect", "text",
    "voice_id", "voice_name",
    "bp_url_existing", "url",
]


def _voices_map() -> dict[str, str]:
    """voice_id → bp_name for label rendering. Active voices only."""
    with (REPO_ROOT / "config" / "voices.tsv").open(encoding="utf-8") as f:
        return {r["voice_id"]: r["bp_name"]
                for r in csv.DictReader(f, dialect="excel-tab")
                if (r.get("status") or "active") == "active"}


def _current_bp_voice() -> dict[str, str]:
    """sense_id → current BP voice_id, sourced from 045-speaker_gender.tsv
    (the post-swap state)."""
    p = REPO_ROOT / "data" / "045-speaker_gender.tsv"
    with p.open(encoding="utf-8") as f:
        return {r["sense_id"]: r["voice_id"]
                for r in csv.DictReader(f, dialect="excel-tab")}


def _sample_senses(rng: random.Random) -> list[dict]:
    """Pick exactly N_SAMPLES clips (N_SAMPLES/2 word + N_SAMPLES/2 example),
    balanced across the 7 active BP voices (post-Lair-retirement).

    Voice attribution uses the LIVE 045-speaker_gender.tsv assignment, not
    06-final.tsv's stale voice_id field (which still says Lair for the 476
    ex-Lair senses until derive_final.py reruns).
    """
    rows = list(csv.DictReader(FINAL_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    current_voice = _current_bp_voice()
    active_voices = set(_voices_map())   # 7 active

    # Rewrite voice_id to the current live assignment.
    eligible = []
    for r in rows:
        if not ((r.get("pt_display") or r.get("pt")) and r.get("example_pt")):
            continue
        live_v = current_voice.get(r["sense_id"], r["voice_id"])
        if live_v not in active_voices:
            continue
        r = {**r, "voice_id": live_v}
        eligible.append(r)

    voices = sorted(active_voices)
    half = N_SAMPLES // 2

    # Stratified sampling per voice for word + example separately so we land
    # at exactly half/half. Distribute as evenly as the pool allows.
    def _pick(clip_type: str) -> list[dict]:
        per_voice = half // len(voices)             # base allotment per voice
        remainder = half - per_voice * len(voices)  # extra to distribute round-robin
        out: list[dict] = []
        for i, v in enumerate(voices):
            quota = per_voice + (1 if i < remainder else 0)
            v_rows = [r for r in eligible if r["voice_id"] == v]
            rng.shuffle(v_rows)
            for r in v_rows[:quota]:
                out.append({**r, "_ct": clip_type})
        return out

    return _pick("word") + _pick("example")


@dataclass
class EPJob:
    row_id: str
    sense_id: str
    clip_type: str
    text: str
    bp_voice_id: str
    bp_voice_name: str
    bp_url_existing: str

    @property
    def ep_object_key(self) -> str:
        short = "word" if self.clip_type == "word" else "ex"
        return f"{R2_PREFIX}/{self.sense_id}-{short}-{V3_MODEL_ID}-ep.mp3"


def _render_one_ep(job: EPJob, el: ElevenLabsClient, r2: R2Client) -> tuple[EPJob, str, str]:
    try:
        tts = el.generate_pcm(
            text=job.text, voice_id=EP_VOICE_ID,
            sense_id=job.sense_id, clip_type=job.clip_type, version=1,
        )
        norm = normalize_pcm_to_mp3_verified(tts.audio_pcm)
        up = r2.upload_bytes(
            norm.mp3_bytes, job.ep_object_key,
            extra_metadata={
                "sense_id": job.sense_id, "clip_type": job.clip_type,
                "voice_id": EP_VOICE_ID, "model": V3_MODEL_ID,
                "stage": "16_1_audio_judge_ab",
            },
        )
        return job, up.url, ""
    except Exception as exc:  # noqa: BLE001
        return job, "", f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done, no API.")
    ap.add_argument("--yes", action="store_true", help="Skip GO prompt.")
    args = ap.parse_args()

    voices = _voices_map()
    rng = random.Random(SEED)
    sample = _sample_senses(rng)
    print(f"  sampled: {len(sample)} senses ({sum(1 for s in sample if s['_ct']=='word')} word + "
          f"{sum(1 for s in sample if s['_ct']=='example')} example)")
    voice_counts: dict[str, int] = {}
    for s in sample:
        voice_counts[s["voice_id"]] = voice_counts.get(s["voice_id"], 0) + 1
    print("  by BP voice:")
    for vid, n in sorted(voice_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {voices.get(vid, vid[:10]):<30} {n:>3}")

    # Build the EP job list.
    jobs: list[EPJob] = []
    rows_out: list[dict] = []
    for i, s in enumerate(sample, start=1):
        ct = s["_ct"]
        text = s["pt_display"] if ct == "word" else s["example_pt"]
        if not text:
            continue
        bp_url_field = "audio_word" if ct == "word" else "audio_example"
        bp_url = s.get(bp_url_field, "")
        if not bp_url:
            print(f"  WARN: no {bp_url_field} for {s['sense_id']}; skipping", file=sys.stderr)
            continue
        # Force the BP URL to point at the v3 file pattern so that even if
        # 06-final.tsv hasn't been regenerated yet, we hit the new clips.
        v3_short = "word" if ct == "word" else "ex"
        v3_bp_url = (
            f"https://pub-155fa287ae724688aaa870d99135311a.r2.dev/audio/"
            f"{s['sense_id']}-{v3_short}-{V3_MODEL_ID}-v1.mp3"
        )
        row_id_bp = f"{i:03d}_bp"
        row_id_ep = f"{i:03d}_ep"
        jobs.append(EPJob(
            row_id=row_id_ep, sense_id=s["sense_id"], clip_type=ct, text=text,
            bp_voice_id=s["voice_id"],
            bp_voice_name=voices.get(s["voice_id"], s["voice_id"][:10]),
            bp_url_existing=v3_bp_url,
        ))
        # BP row pre-filled (no API call needed — URL already known).
        rows_out.append({
            "row_id": row_id_bp,
            "sense_id": s["sense_id"], "clip_type": ct, "dialect": "BP",
            "text": text,
            "voice_id": s["voice_id"],
            "voice_name": voices.get(s["voice_id"], s["voice_id"][:10]),
            "bp_url_existing": v3_bp_url,
            "url": v3_bp_url,
        })

    print(f"  EP jobs to render: {len(jobs)}")
    total_chars = sum(len(j.text) for j in jobs)
    print(f"  total chars: {total_chars:,}")
    print(f"  EP TTS cost (est): ${total_chars * 22.50e-6:.4f}  (v3 ~$22.50/1M chars)")
    print(f"  EP voice: {EP_VOICE_NAME} ({EP_VOICE_ID})  — {EP_VOICE_DESC}")
    print(f"  R2 prefix: {R2_PREFIX}/")
    print(f"  output TSV: {OUT_TSV}")
    print()

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to render the EP clips: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    r2 = R2Client(R2Config.from_env())
    el = ElevenLabsClient(
        model_id=V3_MODEL_ID,
        language_code="pt",       # EP also language_code='pt' (Portuguese)
        pronunciation_dict_locators=None,
    )

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(_render_one_ep, j, el, r2) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            job, url, err = fut.result()
            if err:
                print(f"  [{i:>3}/{len(jobs)}] FAIL {job.sense_id} {job.clip_type:<7} {err[:80]}",
                      file=sys.stderr)
                continue
            n_ok += 1
            rows_out.append({
                "row_id": job.row_id,
                "sense_id": job.sense_id, "clip_type": job.clip_type, "dialect": "EP",
                "text": job.text,
                "voice_id": EP_VOICE_ID, "voice_name": EP_VOICE_NAME,
                "bp_url_existing": job.bp_url_existing,
                "url": url,
            })
            print(f"  [{i:>3}/{len(jobs)}] OK   {job.sense_id} {job.clip_type:<7} → {url}",
                  flush=True)

    print(f"\n  rendered: {n_ok}/{len(jobs)} EP clips OK")

    rows_out.sort(key=lambda r: (r["row_id"]))
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"  wrote: {OUT_TSV} ({len(rows_out)} rows)")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
