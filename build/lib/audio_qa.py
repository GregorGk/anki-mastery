"""Stage 19 — per-take audio QA: PCM sanity, loudness, relaxed ASR, judge gate.

A take is accepted only if ALL hold:
  * TTS returned audio
  * PCM sanity: even byte count, not silent, plausible duration (absolute and
    relative to the v3 clip of the same sense), ≤ 0.7 s silence at each edge
  * loudnorm succeeded with a finite LUFS and true peak ≤ −1.0 dBTP
    (±1 LU tolerance / TP-limited flags are recorded, not gated — 34 % of the
    v3 word clips sit outside ±1 LU)
  * relaxed ASR pass against the plain display text (never the TTS string)
  * every configured judge says bp_ok
"""
from __future__ import annotations

import math
import re
import unicodedata
from array import array
from dataclasses import dataclass, field

from build.lib.asr import (
    THRESHOLD_LONGTAIL,
    THRESHOLD_TOP1000,
    normalize_text,
    text_similarity,
)

SAMPLE_RATE = 44100
SILENCE_DBFS = -40.0
MAX_EDGE_SILENCE_S = 0.7
WORD_ABS_S = (0.25, 3.0)
WORD_REL = (0.5, 2.0)   # only the upper bound is applied to words (see pcm_sanity)
EXAMPLE_REL = (0.6, 1.7)
EXAMPLE_MAX_S = 12.0
MAX_TRUE_PEAK = -1.0

ARTICLES = {"o", "a", "os", "as", "um", "uma", "uns", "umas"}
LEAK_TOKENS = ("barra", "aspas", "slash", "sotaque", "paulistano", "pronuncia", "pronúncia",
               "clear", "careful", "pronunciation")


@dataclass
class Sanity:
    ok: bool
    duration_s: float
    lead_silence_s: float
    trail_silence_s: float
    issues: list = field(default_factory=list)


def pcm_sanity(pcm: bytes, clip_type: str, v3_duration_s: float | None = None) -> Sanity:
    issues: list[str] = []
    if not pcm or len(pcm) % 2:
        return Sanity(False, 0.0, 0.0, 0.0, ["pcm:empty_or_odd"])
    samples = array("h")
    samples.frombytes(pcm)
    n = len(samples)
    dur = n / SAMPLE_RATE
    thr = int(32768 * 10 ** (SILENCE_DBFS / 20))
    loud = [i for i in range(0, n, 32) if abs(samples[i]) > thr]
    if not loud:
        return Sanity(False, dur, dur, dur, ["pcm:silent"])
    lead = loud[0] / SAMPLE_RATE
    trail = (n - loud[-1]) / SAMPLE_RATE
    if lead > MAX_EDGE_SILENCE_S:
        issues.append(f"edge:lead_{lead:.2f}s")
    if trail > MAX_EDGE_SILENCE_S:
        issues.append(f"edge:trail_{trail:.2f}s")
    if clip_type == "word":
        if not WORD_ABS_S[0] <= dur <= WORD_ABS_S[1]:
            issues.append(f"dur:word_{dur:.2f}s")
        # Only the UPPER relative bound for words (runaway renders): v3 word
        # clips carry variable padding, so v4/v3 ratios measured 0.34–1.2× on
        # perfectly good takes (pilot, 232 takes). Truncation is caught by the
        # absolute bound and the ASR check.
        if v3_duration_s and dur / v3_duration_s > WORD_REL[1]:
            issues.append(f"dur:vs_v3_{dur / v3_duration_s:.2f}x")
    else:
        if dur > EXAMPLE_MAX_S:
            issues.append(f"dur:example_{dur:.1f}s")
        if v3_duration_s and not EXAMPLE_REL[0] <= dur / v3_duration_s <= EXAMPLE_REL[1]:
            issues.append(f"dur:vs_v3_{dur / v3_duration_s:.2f}x")
    return Sanity(not issues, dur, lead, trail, issues)


def loudness_ok(final_lufs: float, final_tp: float) -> tuple[bool, list[str]]:
    issues = []
    if final_lufs is None or not math.isfinite(final_lufs):
        issues.append("loud:lufs_not_finite")
    if final_tp is None or not math.isfinite(final_tp) or final_tp > MAX_TRUE_PEAK:
        issues.append(f"loud:true_peak_{final_tp}")
    return not issues, issues


def _joined_forms(s: str) -> set[str]:
    """Space-insensitive forms: all tokens joined, and joined without leading
    articles ("a corda" ≈ "Acorda", "o altar" ≈ "altar")."""
    toks = normalize_text(s).split()
    forms = {"".join(toks)}
    while toks and toks[0] in ARTICLES:
        toks = toks[1:]
        forms.add("".join(toks))
    return {f for f in forms if f}


def leaked(transcript: str, reference: str = "") -> bool:
    """A tag / slash word spoken aloud — unless the reference itself contains
    that word ("Ela fala português sem sotaque.")."""
    t = unicodedata.normalize("NFC", (transcript or "").lower())
    ref = set(normalize_text(reference).split())
    return any(re.search(rf"\b{re.escape(x)}\b", t) for x in LEAK_TOKENS
               if normalize_text(x) not in ref)


_NUMBER_WORDS = set("""zero um uma dois duas tres quatro cinco seis sete oito nove dez onze doze
treze catorze quatorze quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta
cinquenta sessenta setenta oitenta noventa cem cento duzentos duzentas trezentos trezentas
quatrocentos quinhentos seiscentos setecentos oitocentos novecentos mil milhao milhoes
bilhao bilhoes e primeiro segundo terceiro""".split())
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
_UPPER_ROMAN_RE = re.compile(r"\b[IVXLCDM]{2,}\b")


def _strip_numbers(s: str) -> str:
    """Drop digit tokens, Roman numerals, number words and abbreviation
    tokens (km/h, kg, %) so '40 km/h' ≈ 'quarenta quilômetros por hora'."""
    raw = unicodedata.normalize("NFC", s.lower())
    raw = re.sub(r"\b\d[\d.,:/%]*\b|%|\bkm/h\b|\bkm\b|\bkg\b|\bm²\b|\bh\b", " ", raw)
    toks = [t for t in normalize_text(raw).split()
            if t not in _NUMBER_WORDS and not (len(t) <= 4 and _ROMAN_RE.match(t))
            and t not in ("quilometros", "quilometro", "por", "hora", "horas", "quilos",
                          "quilo", "metros", "metro", "porcento", "cento", "reais", "real")]
    return " ".join(toks)




def relaxed_asr_pass(
    *,
    transcripts: list[str],
    reference: str,
    clip_type: str,
    is_top_1000: bool,
    production_decision: str,
    spoken_reference: str = "",
    manual_pass: bool = False,
) -> tuple[bool, str, float]:
    """(pass, mode, best_similarity). `transcripts` = every transcript the
    production roundtrip produced (biased / unbiased). Leaks always fail.

    Relaxations, in order: space/article-insensitive match; the spelled-out
    reference for sentences with numbers; a number-insensitive match (digits,
    Roman numerals, number words and unit abbreviations removed on both sides).
    There is deliberately NO phonetic-distance relaxation: it passed dropped
    final -r (resumir → 'Resumi') in the pilot. A dropped final -r stays a
    failure; oddly-spelled transcripts of correct audio are simply re-rolled."""
    if any(leaked(t, reference) for t in transcripts if t):
        return False, "leak", 0.0
    thr = THRESHOLD_TOP1000 if is_top_1000 else THRESHOLD_LONGTAIL
    best = max((text_similarity(t, reference) for t in transcripts if t), default=0.0)
    if production_decision == "pass":
        return True, "production", best
    if manual_pass:
        return True, "manual_override", best
    from rapidfuzz.distance import Levenshtein
    ref_forms = _joined_forms(reference)
    for t in transcripts:
        if not t:
            continue
        if any(Levenshtein.normalized_similarity(a, b) >= thr
               for a in _joined_forms(t) for b in ref_forms):
            return True, "relaxed", best
        if spoken_reference and text_similarity(t, spoken_reference) >= thr:
            return True, "spoken", best
        if re.search(r"\d", reference + spoken_reference + t) or _UPPER_ROMAN_RE.search(t):
            a, b = _strip_numbers(t), _strip_numbers(reference)
            if a and b and Levenshtein.normalized_similarity(a, b) >= thr:
                return True, "numbers", best
    return False, "fail", best


def gate_pass(judge_verdicts: dict[str, str]) -> bool:
    """AND over the configured judges: every verdict must be bp_ok."""
    return bool(judge_verdicts) and all(v == "bp_ok" for v in judge_verdicts.values())
