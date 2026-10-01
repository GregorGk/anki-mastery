#!/usr/bin/env python3
"""
Spike: phoneme-level ASR as a Brazilian-Portuguese defect detector.

The make-or-break question
--------------------------
The verde run showed wav2vec2-espeak *diverges* on a bad clip. But it only
tested the bad side. The open question: does the recognizer reproduce the
BP-defining features (`dʒ`/`tʃ` palatalization, final-`i` raising) when a
clip is GENUINELY GOOD? If not, it's just European-Portuguese-biased and
useless as a BP gate. If it does, the markers are a real defect detector.

The labeled batch
-----------------
21 word clips, all already in `build/audio_cache/`, in four groups:

  legacy_human_OK   8  calibration-OK lemmas, multilingual-era file
                       (`{sid}-word-v1.mp3`). A human listened and said OK.
                       These have a single legacy version, so v1 is
                       unambiguously the labeled file.  -> the GOOD anchor.
  legacy_human_BAD  6  calibration-MISPRONOUNCED lemmas, same file pattern.
                       Human-confirmed bad.            -> the BAD anchor.
  v3_gemini_BAD     6  Stage-16.8 `non_bp` clips on the shipped eleven_v3
                       deck (`{sid}-word-eleven_v3-v1.mp3`). Includes verde.
  v3_contested      1  verde v2 -- Gemini passed it, the user hears it wrong.

Every clip's reference IPA contains at least one diagnostic feature
(`dʒ`, `tʃ`, or final-`i`). For each clip we check: did the recognizer
reproduce the feature(s) the reference has? Per-group mean = the verdict.

  GOOD high + BAD low  -> discriminates -> real gate signal.
  GOOD low             -> EP-biased     -> drop it.
  everything high      -> can't see defects -> drop it.

References are read live from `data/06-final.tsv` (never mutated).
No `phonemizer` dependency (we bypass the bundled tokenizer and greedy-decode
the CTC logits by hand against the raw `vocab.json`). No API calls.

Run:
    .venv/bin/python build/phoneme_asr_spike.py        # terminal scorecard
    .venv/bin/python build/phoneme_asr_spike_html.py   # inspectable HTML report
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO_CACHE = Path(__file__).resolve().parent / "audio_cache"
FINAL_TSV = ROOT / "data" / "06-final.tsv"

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16_000
_SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>"}
# stress, length, syllable dot, affricate tie-bar -- the recognizer omits these
_MARKS = str.maketrans("", "", "ˈˌːˑ.͡")

# (sense_id, filename in build/audio_cache/, group, source label)
BATCH = [
    # --- legacy human-confirmed GOOD (calibration OK, single legacy version) ---
    ("0259.00.01", "0259.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("0719.00.01", "0719.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("2632.00.01", "2632.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("4125.00.01", "4125.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("3234.00.01", "3234.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("2363.00.01", "2363.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("1430.00.01", "1430.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    ("3341.00.01", "3341.00.01-word-v1.mp3", "legacy_human_OK", "human:OK"),
    # --- legacy human-confirmed BAD (calibration MISPRONOUNCED, single version) ---
    ("0937.00.01", "0937.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    ("3009.00.01", "3009.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    ("1175.00.01", "1175.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    ("3536.00.01", "3536.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    ("3013.00.01", "3013.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    ("4064.00.01", "4064.00.01-word-v1.mp3", "legacy_human_BAD", "human:MISPRONOUNCED"),
    # --- v3 production deck, Gemini-flagged non_bp in Stage 16.8 ---
    ("0534.00.01", "0534.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    ("0002.00.01", "0002.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    ("0304.00.01", "0304.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    ("0913.00.01", "0913.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    ("1454.00.01", "1454.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    ("0580.00.01", "0580.00.01-word-eleven_v3-v1.mp3", "v3_gemini_BAD", "gemini:non_bp"),
    # --- v3 production deck, contested (Gemini passed, user rejects) ---
    ("0534.00.01", "0534.00.01-word-eleven_v3-v2.mp3", "v3_contested", "gemini:bp_ok/user:bad"),
]

GROUP_ORDER = ["legacy_human_OK", "legacy_human_BAD", "v3_gemini_BAD", "v3_contested"]
GROUP_DESC = {
    "legacy_human_OK": "human-confirmed GOOD   (calibration OK; multilingual-era file)",
    "legacy_human_BAD": "human-confirmed BAD    (calibration MISPRONOUNCED; same era)",
    "v3_gemini_BAD": "Gemini-flagged BAD     (16.8 non_bp on the shipped eleven_v3 deck)",
    "v3_contested": "contested              (Gemini passed; the user hears it wrong)",
}


def strip_marks(s: str) -> str:
    return s.translate(_MARKS)


def decode_mp3(path: Path) -> np.ndarray:
    """MP3 -> 16 kHz mono float32 in [-1, 1] via ffmpeg (no extra deps)."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(path),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "-acodec", "pcm_s16le", "-",
        ],
        capture_output=True,
        check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def ctc_greedy_decode(pred_ids: list[int], id_to_token: dict[int, str]) -> str:
    """Standard CTC greedy decode: collapse consecutive repeats, drop blanks."""
    out: list[str] = []
    prev: int | None = None
    for i in pred_ids:
        if i == prev:  # collapse repeats (on raw ids, blanks included)
            continue
        prev = i
        tok = id_to_token.get(i, "")
        if not tok or tok in _SPECIAL_TOKENS:  # drop blank / specials
            continue
        out.append(tok)
    return " ".join(t for t in out if t != "|").strip()


def bp_markers(ref_ipa: str, recognized: str) -> list[tuple[str, bool]]:
    """For each BP-defining feature present in the reference, did the
    recognizer reproduce it? Returns [(feature_name, reproduced), ...]."""
    ref = strip_marks(ref_ipa)
    rec = strip_marks(recognized).replace(" ", "")
    feats: list[tuple[str, bool]] = []
    if "dʒ" in ref:
        feats.append(("dʒ", "dʒ" in rec))
    if "tʃ" in ref:
        feats.append(("tʃ", "tʃ" in rec))
    if ref and ref[-1] in "iɪ":
        feats.append(("final-i", bool(rec) and rec[-1] in "iɪ"))
    return feats


def run_batch(verbose: bool = True) -> list[dict]:
    """Load wav2vec2-espeak, run every clip in BATCH, return result dicts.

    Each result dict: sense_id, pt, en, group, source, ref_ipa, recognized,
    feats [(name, reproduced)], score, sim, filename. Shared by the terminal
    report (`main`) and the HTML report (`phoneme_asr_spike_html.py`) -- one
    source of truth for the recognition logic.

    Raises ImportError if torch / transformers are not installed.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    from transformers import logging as hf_logging
    from rapidfuzz.distance import Levenshtein

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    # references, read live from the canonical master (never mutated)
    with open(FINAL_TSV, encoding="utf-8") as fh:
        final = {r["sense_id"]: r for r in csv.DictReader(fh, dialect="excel-tab")}

    if verbose:
        print(f"loading {MODEL_ID} (cached after first run)")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    with open(hf_hub_download(MODEL_ID, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    if verbose:
        print(f"model ready ({len(vocab)} phoneme vocab entries).\n")

    results: list[dict] = []
    for sense_id, filename, group, source in BATCH:
        path = AUDIO_CACHE / filename
        row = final.get(sense_id, {})
        ref_ipa = row.get("ipa_word", "")
        pt = row.get("pt", "?")
        en = row.get("en_primary", "")
        if not path.exists():
            print(f"!! missing audio: {path}", file=sys.stderr)
            continue

        audio = decode_mp3(path)
        inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        recognized = ctc_greedy_decode(pred_ids, id_to_token)

        feats = bp_markers(ref_ipa, recognized)
        score = sum(1 for _, ok in feats if ok) / len(feats) if feats else float("nan")
        sim = Levenshtein.normalized_similarity(
            strip_marks(ref_ipa).replace(" ", ""),
            strip_marks(recognized).replace(" ", ""),
        )
        results.append(
            dict(sense_id=sense_id, pt=pt, en=en, group=group, source=source,
                 ref_ipa=ref_ipa, recognized=recognized, feats=feats,
                 score=score, sim=sim, filename=filename)
        )
    return results


def group_means(results: list[dict]) -> dict[str, dict]:
    """Per-group {n, markers, sim} aggregates (skips score==NaN clips)."""
    out: dict[str, dict] = {}
    for group in GROUP_ORDER:
        grp = [r for r in results if r["group"] == group and r["score"] == r["score"]]
        if not grp:
            continue
        out[group] = dict(
            n=len(grp),
            markers=sum(r["score"] for r in grp) / len(grp),
            sim=sum(r["sim"] for r in grp) / len(grp),
        )
    return out


def main() -> int:
    try:
        results = run_batch(verbose=True)
    except ImportError as exc:  # pragma: no cover - spike
        print(f"missing dependency: {exc}", file=sys.stderr)
        print("  .venv/bin/pip install torch transformers", file=sys.stderr)
        return 1

    # ---- per-clip detail, grouped ----
    for group in GROUP_ORDER:
        grp = [r for r in results if r["group"] == group]
        if not grp:
            continue
        print("=" * 78)
        print(f"  {group}   --   {GROUP_DESC[group]}")
        print("=" * 78)
        for r in grp:
            marks = "  ".join(f"{name}{'✓' if ok else '✗'}" for name, ok in r["feats"])
            n_ok = sum(1 for _, ok in r["feats"] if ok)
            print(f"  {r['sense_id']}  {r['pt']:<15} [{n_ok}/{len(r['feats'])}]  {marks}")
            print(f"      ref  {r['ref_ipa']}")
            print(f"      rec  {r['recognized']}      (sim {r['sim']:.2f})")
        print()

    # ---- aggregate scorecard ----
    print("=" * 78)
    print("  SCORECARD  --  mean BP-marker reproduction rate per group")
    print("=" * 78)
    means = group_means(results)
    for group in GROUP_ORDER:
        if group not in means:
            continue
        m = means[group]
        bar = "█" * round(m["markers"] * 30)
        print(f"  {group:<18} n={m['n']:<2}  markers {m['markers']:4.0%}   sim {m['sim']:.2f}  {bar}")
    print()

    # ---- verdict ----
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    good = means.get("legacy_human_OK", {}).get("markers")
    legbad = means.get("legacy_human_BAD", {}).get("markers")
    v3bad = means.get("v3_gemini_BAD", {}).get("markers")
    if good is None or legbad is None:
        print("  inconclusive: missing a labeled group.")
    else:
        gap = good - legbad
        print("  controlled test -- same era, same label source (human), only label differs:")
        print(f"    human-GOOD legacy   {good:.0%}")
        print(f"    human-BAD  legacy   {legbad:.0%}")
        print(f"    separation gap      {gap:+.0%}")
        if v3bad is not None:
            print("  cross-era reference (confounded by TTS model, read with caution):")
            print(f"    Gemini-flagged v3   {v3bad:.0%}")
        print()
        if good >= 0.70 and gap >= 0.25:
            print("  DISCRIMINATES.  Good clips keep their BP markers, bad clips lose them,")
            print("  with a clean gap. Real defect signal -- worth building into a gate.")
        elif gap < 0.15:
            print("  NO DISCRIMINATION.  Human-GOOD and human-BAD legacy clips are")
            print("  statistically indistinguishable on these markers. wav2vec2-espeak does")
            print("  not separate good BP from bad BP on same-era audio. Either the")
            print("  recognizer is too weak at BP palatalization, or the multilingual-era")
            print("  'good' clips were never textbook BP -- this batch cannot tell which.")
            print("  Not gate-worthy as-is.")
        elif good < 0.50:
            print("  WEAK.  Even human-confirmed-GOOD clips reproduce only a minority of")
            print("  their BP markers. The recognizer is unreliable at BP palatalization.")
        else:
            print("  AMBIGUOUS.  Some separation, but not clean enough for a gate.")
    print()
    print("  (verde v2, the contested clip, is in group v3_contested above --")
    print("   compare its score against the GOOD-group baseline.)")
    print("  HTML companion: .venv/bin/python build/phoneme_asr_spike_html.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
