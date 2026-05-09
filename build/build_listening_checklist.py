"""Generate a comprehensive listening checklist for the user.

Sections:
  1. Voice quality samples (3 per voice = 33 URLs)
  2. HUMAN review queue (clips that still failed after regen pass)
  3. TP-limited clips (couldn't reach -16 LUFS without distortion)
  4. Loudness outliers post-correction (>2 LU off target, excluding TP-limited
     duplicates from section 3)

Each entry includes URL + expected text + (where applicable) what Whisper
heard. Output to audit/listening_checklist.txt and stdout.
"""
from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

OUT_PATH = AUDIT_DIR / "listening_checklist.txt"


def load_manifest():
    rows = []
    with open(DATA_DIR / "_audio_manifest.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, dialect="excel-tab"):
            if r.get("status") == "uploaded":
                rows.append(r)
    return rows


def load_voices():
    out = {}
    with open(REPO_ROOT / "config" / "voices.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, dialect="excel-tab"):
            out[r["voice_id"]] = (r["gender"], int(r["pool_index"]))
    return out


def load_human_queue():
    p = DATA_DIR / "_audio_human_review.tsv"
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8"), dialect="excel-tab"))


def load_baselines():
    p = DATA_DIR / "_voice_loudness_baselines.tsv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, dialect="excel-tab"):
            out[r["voice_id"]] = (float(r["median_gain_db"]), int(r["measured_n"]))
    return out


def main():
    manifest = load_manifest()
    voices = load_voices()
    human_queue = load_human_queue()
    baselines = load_baselines()

    by_key = {(r["sense_id"], r["clip_type"]): r for r in manifest}
    by_voice = defaultdict(list)
    for r in manifest:
        by_voice[r["voice_id"]].append(r)

    out_lines: list[str] = []

    def w(s: str = ""):
        out_lines.append(s)

    w("=" * 100)
    w("COMPREHENSIVE LISTENING CHECKLIST")
    w(f"Stage 6 pilot — 500 senses / 1,000 clips total")
    w("=" * 100)
    w()
    w("How to use:")
    w("  - Open URLs in browser (each one plays directly).")
    w("  - For voice quality: confirm each voice sounds right.")
    w("  - For HUMAN review: PASS = audio is fine, REGEN = audio is wrong.")
    w("  - For TP-limited / loudness outliers: confirm volume is acceptable")
    w("    (these are the clips loudness norm couldn't pull all the way to target).")
    w()

    # ─────────── 1. VOICE QUALITY SAMPLES ─────────── #
    w("=" * 100)
    w("§1.  VOICE QUALITY SAMPLES — 3 clips per voice (= 33 URLs)")
    w("     Pick: 1 word + 2 examples per voice. Confirm tone / accent / gender match.")
    w("=" * 100)

    rng = random.Random(2026)
    voice_list = sorted(by_voice.keys(), key=lambda v: (voices.get(v, ("", 999))[0], voices.get(v, ("", 999))[1]))

    for vid in voice_list:
        gender, idx = voices.get(vid, ("?", 0))
        gain, n = baselines.get(vid, (0.0, 0))
        rows = by_voice[vid]
        words = [r for r in rows if r["clip_type"] == "word"]
        exs = [r for r in rows if r["clip_type"] == "example"]
        sample_w = rng.sample(words, 1) if words else []
        sample_e = rng.sample(exs, 2) if len(exs) >= 2 else exs
        w()
        w(f"--- {gender.upper()} #{idx}: {vid}")
        w(f"    baseline gain: {gain:+.2f} dB | n={n} measurements")
        for r in sample_w:
            w(f"  WORD     {r['url']}")
            w(f"           expected: \"{r['text_input']}\"")
        for r in sample_e:
            w(f"  EXAMPLE  {r['url']}")
            w(f"           expected: \"{r['text_input']}\"")
    w()

    # ─────────── 2. HUMAN REVIEW QUEUE ─────────── #
    w("=" * 100)
    w(f"§2.  HUMAN REVIEW QUEUE — {len(human_queue)} clips (failed ASR after 3 attempts)")
    w("     For each, decide PASS (audio is correct, ASR was wrong) or REGEN (audio is wrong).")
    w("=" * 100)

    if not human_queue:
        w()
        w("  (empty — every clip recovered.)")
    for i, q in enumerate(human_queue, 1):
        sid, ctype = q["sense_id"], q["clip_type"]
        gender_idx = voices.get(q["voice_id"], ("?", 0))
        w()
        w(f"[{i:>2}/{len(human_queue)}]  {sid}  {ctype:<7}  voice={gender_idx[0].upper()} #{gender_idx[1]} ({q['voice_id'][:8]}…)")
        w(f"         expected:       \"{q['input_text']}\"")
        w(f"         Whisper heard:  \"{q['asr_transcript']}\"  (sim={q['similarity']})")
        w(f"         URL:            {q['url']}")

    # ─────────── 3. TP-LIMITED CLIPS ─────────── #
    tp_limited = [r for r in manifest if r.get("tp_limited") == "true"]
    w()
    w("=" * 100)
    w(f"§3.  TP-LIMITED CLIPS — {len(tp_limited)} clips")
    w("     These are clips where the ElevenLabs source PCM had a high crest factor")
    w("     (peaks too close to 0 dB), so gain-only normalization couldn't push them")
    w("     to -16 LUFS without exceeding the -1.5 dB true-peak ceiling. They're")
    w("     usable but quieter than other clips. Listen to a sample to confirm")
    w("     audibility — if too quiet, we can add a true-peak limiter (industry")
    w("     standard for podcasts) which pushes ~95% of clips into ±1 LU at the")
    w("     cost of clipping rare peaks (inaudible on speech).")
    w("=" * 100)

    if tp_limited:
        # Sort by final_lufs ascending (worst quietest first), show first 25 for review
        tp_limited_sorted = sorted(
            tp_limited,
            key=lambda r: float(r.get("final_lufs", "0") or "0"),
        )
        w(f"\n     Showing 25 worst (quietest) of {len(tp_limited)} TP-limited clips:")
        for i, r in enumerate(tp_limited_sorted[:25], 1):
            gender_idx = voices.get(r["voice_id"], ("?", 0))
            w()
            w(f"[{i:>2}/25]  {r['sense_id']}  {r['clip_type']:<7}  voice={gender_idx[0].upper()} #{gender_idx[1]}")
            w(f"         final loudness: {r['final_lufs']} LUFS  (target -16, applied gain {r['applied_gain_db']} dB)")
            w(f"         expected:       \"{r['text_input']}\"")
            w(f"         URL:            {r['url']}")

    # ─────────── 4. LOUDNESS OUTLIERS POST-CORRECTION ─────────── #
    # Outliers = >2 LU off target, NOT in tp_limited (because section 3 already covers those)
    tp_limited_keys = {(r["sense_id"], r["clip_type"]) for r in tp_limited}
    outliers = []
    for r in manifest:
        try:
            lufs = float(r.get("final_lufs", "0") or "0")
        except ValueError:
            continue
        if abs(lufs + 16) > 2 and (r["sense_id"], r["clip_type"]) not in tp_limited_keys:
            outliers.append((r, lufs))

    w()
    w("=" * 100)
    w(f"§4.  NON-TP-LIMITED LOUDNESS OUTLIERS — {len(outliers)} clips")
    w("     Clips off-target by >2 LU NOT due to TP ceiling — these are unexpected.")
    w("=" * 100)

    if not outliers:
        w()
        w("  (none — all non-TP-limited clips are within ±2 LU.)")
    else:
        outliers.sort(key=lambda x: abs(x[1] + 16), reverse=True)
        w(f"\n     Showing all {len(outliers)} outliers, worst first:")
        for i, (r, lufs) in enumerate(outliers[:30], 1):
            gender_idx = voices.get(r["voice_id"], ("?", 0))
            w()
            w(f"[{i:>2}/{len(outliers)}]  {r['sense_id']}  {r['clip_type']:<7}  voice={gender_idx[0].upper()} #{gender_idx[1]}")
            w(f"         final loudness: {lufs:+.2f} LUFS  (off by {lufs + 16:+.2f} LU; gain {r['applied_gain_db']} dB)")
            w(f"         expected:       \"{r['text_input']}\"")
            w(f"         URL:            {r['url']}")

    # Footer
    w()
    w("=" * 100)
    w("END OF CHECKLIST")
    w("=" * 100)

    text = "\n".join(out_lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[checklist] saved to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
