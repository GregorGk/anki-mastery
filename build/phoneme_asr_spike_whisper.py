#!/usr/bin/env python3
"""
Spike v2: Whisper-large-v3-turbo + espeak-ng pt-br G2P.

The make-or-break question, restated
------------------------------------
The xlsr-53 wav2vec2-espeak run produced IPA the human review flagged
`model_wrong` on 6/21 clips and `unsure` on 4 more (see
`phoneme_asr_spike_review.tsv`). On confirmed-GOOD BP audio
(`oportunidade`, `saudade`, `geralmente`, `ansiedade`) the recognizer
stripped the BP palatalization (`dʒ`→`d`, `tʃ`→`t`) and the final-i
raising (`i`→`ɨ`) -- the wav2vec2-espeak family is EP-biased at its
core.

This spike trades a different way:

  1. **Whisper-large-v3-turbo** transcribes audio -> Brazilian-Portuguese
     orthographic text. (~5% WER on BP -- the strongest open ASR.)
  2. **phonemizer / espeak-ng pt-br** turns that text into the textbook
     BP IPA -- including `dʒi` / `tʃi`, the retracted-r, final-i.

Two-stage means high transcription fidelity on good audio: a clip that
clearly says `oportunidade` produces `dʒi` because espeak knows BP, not
because the acoustic model heard it. The trade-off is the inverse of
xlsr-53's: Whisper is *word*-aware, so on a mispronounced clip it can
"rescue" the intended word and produce canonical IPA anyway -- masking
the very defect a gate would want to see. The smoke run already shows
both sides of this: Whisper mistranscribes `0937.00.01-sede` to "Cd"
and `0002.00.01-de` to "Dio" (the bad pronunciations confused it -- a
defect signal!), but it transcribes the contested verde v2 cleanly to
"verde" (so the IPA matches ref, defect hidden).

  Useful for: "did the audio match what should be said?"
  NOT useful for: a true acoustic-defect gate.

Reuses BATCH, decode_mp3, strip_marks, bp_markers, group_means,
GROUP_ORDER, GROUP_DESC, AUDIO_CACHE, FINAL_TSV from
`phoneme_asr_spike` -- one labeled batch, one scoring rule.

Run:
    .venv/bin/python build/phoneme_asr_spike_whisper.py
    .venv/bin/python build/phoneme_asr_spike_whisper_html.py
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

# espeak-ng dylib lives in homebrew on this machine; phonemizer needs the
# explicit path because its default search doesn't cover /opt/homebrew.
os.environ.setdefault(
    "PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoneme_asr_spike import (  # noqa: E402
    AUDIO_CACHE,
    BATCH,
    FINAL_TSV,
    GROUP_DESC,
    GROUP_ORDER,
    SAMPLE_RATE,
    bp_markers,
    decode_mp3,
    group_means,
    strip_marks,
)

MODEL_ID = "openai/whisper-large-v3-turbo"
G2P_VOICE = "pt-br"


def espeak_normalize(s: str) -> str:
    """Map espeak-ng pt-br output into the IPA convention Stage 5 uses.

    espeak emits a handful of conventions that drift from the reference
    IPA without changing the phonetic intent:

      - ZWJ (`‍`) between affricate halves -> strip
      - final unstressed `y` (espeak shorthand for raised i) -> `i`
      - velar/uvular r rendered as `x` -> `ʁ` (the Stage-5 convention)
      - `lj` digraph (espeak) -> `ʎ` (BP convention)
      - `nj` digraph (espeak) -> `ɲ` (BP convention)
      - `ʊ` at word end -> `u` (Stage-5 doesn't mark word-final laxing)
      - `ŋ` immediately after a nasalized vowel -> drop (espeak adds it
        for syllabification; Stage 5 doesn't)

    Schwa epenthesis (espeak inserts `ə` between consonant clusters that
    BP just runs together) is intentionally NOT stripped here -- it
    shows up in `recognized` so the difference is visible to the reader,
    but `strip_marks` already removes `.` and stress marks so the
    Levenshtein-similarity score isn't punished much for it.
    """
    s = s.replace("‍", "")
    s = s.replace("y", "i")
    s = s.replace("x", "ʁ")
    s = s.replace("lj", "ʎ").replace("nj", "ɲ")
    # word-final ʊ -> u (only when truly terminal, not mid-string)
    if s.endswith("ʊ"):
        s = s[:-1] + "u"
    # ŋ right after a nasalized vowel character (combining tilde or
    # already-nasalized vowel) -- strip it.
    out = []
    nasal_vowels = {"ɐ̃", "ẽ", "ĩ", "õ", "ũ"}
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "ŋ" and out and (
            out[-1] in nasal_vowels
            or (len(out) >= 2 and out[-2] + out[-1] in nasal_vowels)
            or (i > 0 and s[i - 1] == "̃")  # combining tilde
        ):
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def asr_text(audio, processor, model, device):
    """Run Whisper-large-v3-turbo to transcribe one clip to BP text."""
    import torch

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    feats = inputs.input_features.to(device=device, dtype=model.dtype)
    with torch.no_grad():
        out = model.generate(
            feats,
            language="portuguese",
            task="transcribe",
            max_new_tokens=40,
            num_beams=1,
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def g2p_ipa(text: str, phonemize_fn) -> str:
    """text -> espeak pt-br IPA, normalized to Stage-5 convention."""
    text = text.strip().strip(".,!?;:").strip()
    if not text:
        return ""
    ipa = phonemize_fn(
        [text], language=G2P_VOICE, backend="espeak",
        strip=True, with_stress=True,
    )[0]
    return espeak_normalize(ipa)


def run_batch(verbose: bool = True) -> list[dict]:
    """Whisper + espeak-pt-br over BATCH. Returns the same dict shape as
    `phoneme_asr_spike.run_batch` plus an extra `whisper_text` field.

    Each result: sense_id, pt, en, group, source, ref_ipa, whisper_text,
    recognized, feats [(name, reproduced)], score, sim, filename.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers import logging as hf_logging
    from rapidfuzz.distance import Levenshtein
    from phonemizer import phonemize

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    # references read live from the canonical master (never mutated)
    with open(FINAL_TSV, encoding="utf-8") as fh:
        final = {r["sense_id"]: r for r in csv.DictReader(fh, dialect="excel-tab")}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    if verbose:
        print(f"loading {MODEL_ID} on {device}/{str(dtype).split('.')[-1]} "
              "(cached after first run)")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    if verbose:
        print("model ready.\n")

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
        whisper_text = asr_text(audio, processor, model, device)
        recognized = g2p_ipa(whisper_text, phonemize)

        feats = bp_markers(ref_ipa, recognized)
        score = sum(1 for _, ok in feats if ok) / len(feats) if feats else float("nan")
        sim = Levenshtein.normalized_similarity(
            strip_marks(ref_ipa).replace(" ", ""),
            strip_marks(recognized).replace(" ", ""),
        )
        results.append(
            dict(sense_id=sense_id, pt=pt, en=en, group=group, source=source,
                 ref_ipa=ref_ipa, whisper_text=whisper_text, recognized=recognized,
                 feats=feats, score=score, sim=sim, filename=filename)
        )
        if verbose:
            print(f"  {sense_id}  {pt:<15} asr={whisper_text!r:<22} ipa={recognized}")
    return results


def main() -> int:
    try:
        results = run_batch(verbose=True)
    except ImportError as exc:  # pragma: no cover - spike
        print(f"missing dependency: {exc}", file=sys.stderr)
        print("  .venv/bin/pip install torch transformers phonemizer", file=sys.stderr)
        print("  brew install espeak-ng", file=sys.stderr)
        return 1

    print()
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
            print(f"      asr  {r['whisper_text']!r}")
            print(f"      ref  {r['ref_ipa']}")
            print(f"      rec  {r['recognized']}      (sim {r['sim']:.2f})")
        print()

    # ---- aggregate scorecard ----
    print("=" * 78)
    print("  SCORECARD  --  mean BP-marker reproduction rate per group (Whisper+espeak)")
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
            print("  cross-era reference (TTS model differs, read with caution):")
            print(f"    Gemini-flagged v3   {v3bad:.0%}")
        print()
        if good >= 0.85:
            print(f"  TRANSCRIPTION QUALITY OK.  GOOD clips reproduce {good:.0%} of their BP markers.")
            print("  But remember: Whisper is word-aware, so it can mask phoneme-level")
            print("  mispronunciations by 'rescuing' the intended word. Look at the BAD")
            print("  group: high % there = defect masked, low % = Whisper struggled with the")
            print("  audio (which itself is a defect signal).")
        elif good >= 0.70:
            print(f"  PARTIAL.  GOOD clips reproduce {good:.0%} of BP markers -- some")
            print("  espeak normalization still off, or Whisper struggled on a few.")
        else:
            print(f"  TRANSCRIPTION WEAK.  Only {good:.0%} on confirmed-GOOD clips.")
    print()
    print("  HTML companion: .venv/bin/python build/phoneme_asr_spike_whisper_html.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
