"""Stage 19 — Brazilian-Portuguese (São Paulo) broad IPA: normalization + validators.

Pure functions, no I/O except the cached espeak-ng stress oracle. Used by
build/19_1_ipa_v2.py to turn the Stage-05 IPA into one consistent convention
and to flag entries that need adjudication.

Convention ("IPA v2", São Paulo broad transcription)
  * primary stress ˈ before the stressed syllable's onset; no secondary
    stress, no syllable dots, no brackets/slashes
  * strong r (word-initial, rr, after n/l/s/z or a nasal vowel) → h;
    coda r (before a consonant or word-final) → ɾ; onset clusters → ɾ
  * t/d before i (or j) → tʃ/dʒ, written without tie bars
  * word-final unstressed e/o → i/u (also before final s); final a → ɐ
  * coda s: ʃ → s before voiceless / word-final, → z before voiced
  * final l → w; nasal diphthongs ɐ̃w̃ ẽj̃ õj̃ ɐ̃j̃ ũj̃
  * eSpeak artefacts dropped: ə, ŋ after a nasal vowel, ɪ/ʊ → i/u, æ/y → ɐ/i,
    ASCII g → ɡ

Validators
  * symbol whitelist
  * exactly one primary stress for polysyllables
  * stress position vs the espeak-ng pt-br oracle (counted from the end and
    from the start, glide-insensitive — either may match)
  * written accent ⇒ stressed-vowel quality (é→ɛ, ê→e, ó→ɔ, ô→o …)
  * MFA dictionary as a soft cross-check of consonants + stressed vowel
"""
from __future__ import annotations

import functools
import re
import shutil
import subprocess
import unicodedata

TILDE = "̃"
STRESS = "ˈ"
SECONDARY = "ˌ"

VOWEL_BASES = set("aeiouɐɛɔ")
GLIDE_BASES = set("jw")
CONSONANTS = {"p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ",
              "l", "ʎ", "ɾ", "h", "tʃ", "dʒ"}
OBSTRUENTS = set("pbtdkɡfv")
VOICED_CONSONANTS = {"b", "d", "ɡ", "v", "z", "ʒ", "m", "n", "ɲ", "l", "ʎ", "ɾ", "dʒ", "h"}
STRONG_RHOTICS = {"ʁ", "x", "χ", "h", "R", "ʀ"}
ALLOWED_CHARS = set("pbtdkɡfvszʃʒmnɲlʎɾhjwaeiouɐɛɔ") | {TILDE, STRESS}

# Clitics / function words: São Paulo weak (unstressed) forms used in sentences
# and as the displayed headword IPA for function-word senses.
WEAK_FORMS: dict[str, str] = {
    "o": "u", "os": "us", "a": "a", "as": "as", "à": "a", "às": "as",
    "e": "i", "é": "ˈɛ", "de": "dʒi", "do": "du", "dos": "dus", "da": "dɐ", "das": "dɐs",
    "em": "ẽj̃", "no": "nu", "nos": "nus", "na": "nɐ", "nas": "nɐs",
    "um": "ũ", "uns": "ũs", "uma": "ˈumɐ", "umas": "ˈumɐs",
    "que": "ki", "se": "si", "me": "mi", "te": "tʃi", "lhe": "ʎi", "lhes": "ʎis",
    "nos.": "nus", "com": "kõ", "sem": "sẽj̃", "por": "poɾ", "pelo": "ˈpelu", "pela": "ˈpelɐ",
    "pelos": "ˈpelus", "pelas": "ˈpelɐs", "para": "ˈpaɾɐ", "pra": "pɾa", "ao": "aw", "aos": "aws",
    "num": "nũ", "numa": "ˈnumɐ", "mas": "mas", "ou": "ow", "nem": "nẽj̃", "porque": "poɾˈke",
    "quando": "ˈkwɐ̃du", "como": "ˈkomu", "se.": "si", "sobre": "ˈsobɾi",
}
WEAK_FORMS = {k.rstrip("."): v for k, v in WEAK_FORMS.items()}

# Spellings whose stressed e/o quality depends on part of speech / meaning
# (noun vs verb, homograph pairs). Resolved per occurrence with context.
HETEROPHONES = frozenset({
    "gosto", "jogo", "jogos", "olho", "olhos", "colher", "almoço", "almoços", "começo",
    "acordo", "esforço", "esforços", "forma", "formas", "molho", "sede", "seca", "governo",
    "cerca", "torre", "posto", "rego", "selo", "apelo", "conserto", "concerto", "gelo",
    "modelo", "erro", "erros", "porto", "transtorno", "socorro", "consolo", "toco",
    "acerto", "aceito", "seco", "meta", "colheres", "nojo", "choro", "rolo", "bolo",
    "piloto", "fecho",
})

ACCENT_VOWEL_CLASS = {   # written accent → allowed stressed nucleus bases
    "á": {"a", "ɐ"}, "é": {"ɛ"}, "í": {"i"}, "ó": {"ɔ"}, "ú": {"u"},
    "â": {"ɐ", "a"}, "ê": {"e"}, "ô": {"o"},
}


# ── segmentation ─────────────────────────────────────────────────────────────
def _clean_chars(ipa: str) -> str:
    s = unicodedata.normalize("NFD", ipa or "")
    s = s.replace("͡", "").replace("͜", "")          # tie bars
    s = s.replace("'", STRESS).replace("ˈˈ", STRESS)
    s = re.sub(r"[\[\]/.()‿͜|]", "", s)
    s = s.replace("g", "ɡ").replace("ʀ", "ʁ")
    s = s.replace("æ", "ɐ").replace("y", "i").replace("ɪ", "i").replace("ʊ", "u")
    s = s.replace("ɫ", "w").replace("ə", "").replace("ɚ", "ɾ")
    s = s.replace("ɹ", "ɾ").replace("ɻ", "ɾ")
    s = s.replace("ε", "ɛ").replace("ɨ", "i").replace("ʉ", "u")      # Greek ε, EP ɨ
    for junk in ("̯", "´", "́", "ⁿ", "-", "ː", "ˑ", "ʼ"):
        s = s.replace(junk, "")                                       # ̯ ´ ́ ⁿ - ː
    s = s.replace(SECONDARY, "")
    return unicodedata.normalize("NFC", s.strip())


def segments(ipa: str) -> list[str]:
    """Split IPA into segments: base char + combining marks; tʃ/dʒ joined;
    the stress mark is its own segment."""
    s = unicodedata.normalize("NFD", ipa)
    out: list[str] = []
    for ch in s:
        if unicodedata.combining(ch):
            if out:
                out[-1] += ch
            continue
        if ch in ("ʃ", "ʒ") and out and out[-1] in ("t", "d"):
            out[-1] += ch
            continue
        if ch.isspace():
            continue
        out.append(ch)
    return [unicodedata.normalize("NFC", x) for x in out]


def base(seg: str) -> str:
    return unicodedata.normalize("NFD", seg)[0] if seg else ""


def is_nasal(seg: str) -> bool:
    return TILDE in unicodedata.normalize("NFD", seg)


def is_vowel(seg: str) -> bool:
    return base(seg) in VOWEL_BASES


def is_glide(seg: str) -> bool:
    return base(seg) in GLIDE_BASES


def is_consonant(seg: str) -> bool:
    return seg not in (STRESS,) and not is_vowel(seg) and not is_glide(seg)


def strip_stress(segs: list[str]) -> list[str]:
    return [s for s in segs if s != STRESS]


def nuclei(segs: list[str]) -> list[int]:
    """Indices (into segs w/o stress marks) of syllable nuclei."""
    return [i for i, s in enumerate(segs) if is_vowel(s)]


def glide_insensitive_nuclei(segs: list[str], stressed_seg: int | None) -> list[int]:
    """Nuclei after treating unstressed high vowels next to another vowel as
    glides — makes espeak / LLM / MFA syllable counts comparable."""
    idx = nuclei(segs)
    keep = []
    for i in idx:
        b = base(segs[i])
        if i != stressed_seg and b in ("i", "u"):
            prev_v = i > 0 and is_vowel(segs[i - 1])
            next_v = i + 1 < len(segs) and is_vowel(segs[i + 1])
            if prev_v or next_v:
                continue
        keep.append(i)
    return keep


def stressed_segment(segs_with_marks: list[str]) -> int | None:
    """Index (into the stress-free segment list) of the first vowel after the
    (first) primary stress mark."""
    pos = 0
    seen_mark = False
    for s in segs_with_marks:
        if s == STRESS:
            seen_mark = True
            continue
        if seen_mark and is_vowel(s):
            return pos
        pos += 1
    return None


# ── onset legality / stress placement ────────────────────────────────────────
def _legal_onset(cluster: list[str]) -> bool:
    if len(cluster) <= 1:
        return True
    bases = [base(s) if len(s) == 1 or s in ("tʃ", "dʒ") else base(s) for s in cluster]
    if len(cluster) == 2:
        a, b = cluster
        if base(a) in OBSTRUENTS and b in ("ɾ", "l"):
            return True
        if is_consonant(a) and is_glide(b):
            return True
        return False
    if len(cluster) == 3:
        a, b, c = cluster
        return base(a) in OBSTRUENTS and b in ("ɾ", "l") and is_glide(c)
    del bases
    return False


def onset_start(segs: list[str], nucleus: int) -> int:
    """Leftmost index of the legal onset preceding `nucleus` (max onset)."""
    start = nucleus
    while start > 0:
        prev = segs[start - 1]
        if is_vowel(prev):
            break
        if not _legal_onset(segs[start - 1:nucleus]):
            break
        start -= 1
    return start


def place_stress(segs: list[str], nucleus: int | None) -> list[str]:
    segs = strip_stress(segs)
    if nucleus is None:
        return segs
    st = onset_start(segs, nucleus)
    return segs[:st] + [STRESS] + segs[st:]


# ── SP rewrite rules ─────────────────────────────────────────────────────────
def _rhotics(segs: list[str]) -> list[str]:
    out = list(segs)
    for i, s in enumerate(out):
        b = base(s)
        if b not in STRONG_RHOTICS and b not in ("ɾ", "r"):
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        prv = out[i - 1] if i > 0 else None
        before_vowel = nxt is not None and (is_vowel(nxt) or is_glide(nxt))
        if not before_vowel:
            out[i] = "ɾ"                                   # coda → SP tap
            continue
        if prv is not None and base(prv) in OBSTRUENTS:
            out[i] = "ɾ"                                   # cluster (br, pr, tr…)
            continue
        strong_ctx = (prv is None or base(prv) in ("n", "l", "s", "z") or is_nasal(prv))
        if b in STRONG_RHOTICS or b == "r" and strong_ctx or (b == "ɾ" and strong_ctx):
            out[i] = "h"
        else:
            out[i] = "ɾ"
    return out


def _palatalize(segs: list[str]) -> list[str]:
    out = list(segs)
    for i in range(len(out) - 1):
        if out[i] in ("t", "d") and base(out[i + 1]) in ("i", "j"):
            out[i] = "tʃ" if out[i] == "t" else "dʒ"
    return out


def _drop_ng(segs: list[str]) -> list[str]:
    out = []
    for i, s in enumerate(segs):
        if s == "ŋ":
            if out and is_nasal(out[-1]):
                continue
            s = "n" if i + 1 < len(segs) and is_vowel(segs[i + 1]) else "ŋ"
            if s == "ŋ":
                continue
        out.append(s)
    return out


def _final_reduction(segs: list[str], stressed: int | None) -> list[str]:
    out = list(segs)
    n = len(out)
    j = n - 1
    if j >= 0 and base(out[j]) in ("s", "z", "ʃ", "ʒ"):
        j -= 1
    if j >= 0 and is_vowel(out[j]) and j != stressed and len(nuclei(out)) > 1:
        b = base(out[j])
        tilde = TILDE if is_nasal(out[j]) else ""
        if b == "e" and not tilde:
            out[j] = "i"
        elif b == "o" and not tilde:
            out[j] = "u"
        elif b == "a" and not tilde:
            out[j] = "ɐ"
    return out


def _coda_s(segs: list[str]) -> list[str]:
    out = list(segs)
    for i, s in enumerate(out):
        if base(s) not in ("s", "z", "ʃ", "ʒ"):
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        if nxt is None:
            out[i] = "s"
        elif is_consonant(nxt):
            out[i] = "z" if nxt in VOICED_CONSONANTS else "s"
    return out


def _nasal_glides(segs: list[str]) -> list[str]:
    """Nasal diphthongs: a glide after a nasal vowel is nasal (ɐ̃w̃, ẽj̃ …)."""
    out = list(segs)
    for i in range(1, len(out)):
        if is_glide(out[i]) and is_nasal(out[i - 1]) and not is_nasal(out[i]):
            out[i] = out[i] + TILDE
            out[i] = unicodedata.normalize("NFC", out[i])
    return out


def to_sp(ipa: str, *, stressed_override: int | None = None) -> tuple[str, list[str]]:
    """Normalize one word's IPA to the São Paulo convention.

    Returns (ipa_sp, issues). `issues` lists structural problems found in the
    INPUT (e.g. 'stress:none', 'stress:multiple') — normalization never
    invents a stress position; `stressed_override` (index into the stress-free
    segments) lets callers move it explicitly.
    """
    issues: list[str] = []
    s = _clean_chars(ipa)
    segs_m = segments(s)
    n_marks = sum(1 for x in segs_m if x == STRESS)
    if n_marks > 1:
        issues.append("stress:multiple")
    segs = strip_stress(segs_m)
    stressed = stressed_override if stressed_override is not None else stressed_segment(segs_m)
    # Rewrites keep segment count stable except _drop_ng — track the stressed
    # nucleus by its ordinal among vowels instead of the raw index.
    ordinal = nuclei(segs).index(stressed) if stressed in nuclei(segs) else None
    segs = _drop_ng(segs)
    segs = _rhotics(segs)
    segs = _palatalize(segs)
    segs = _coda_s(segs)
    segs = _nasal_glides(segs)
    nuc = nuclei(segs)
    stressed = nuc[ordinal] if ordinal is not None and ordinal < len(nuc) else None
    segs = _final_reduction(segs, stressed)
    segs = _palatalize(segs)                     # finals may have created new 'ti'/'di'
    if stressed is None and len(nuc) > 1:
        issues.append("stress:none")
    out = place_stress(segs, stressed if len(nuc) > 1 or n_marks else None)
    return "".join(out), issues


# ── validators ───────────────────────────────────────────────────────────────
def whitelist_issues(ipa: str) -> list[str]:
    bad = sorted({c for c in unicodedata.normalize("NFD", ipa) if c not in ALLOWED_CHARS
                  and not c.isspace()})
    return [f"symbol:{''.join(bad)}"] if bad else []


def stress_positions(ipa: str) -> tuple[int | None, int | None, int]:
    """(from_end, from_start, n_nuclei) of the primary stress, glide-insensitive.
    1-based. None when the word has no primary stress mark."""
    segs_m = segments(_clean_chars(ipa))
    segs = strip_stress(segs_m)
    st = stressed_segment(segs_m)
    nuc = glide_insensitive_nuclei(segs, st)
    if st is None or st not in nuc:
        return None, None, len(nuc)
    k = nuc.index(st)
    return len(nuc) - k, k + 1, len(nuc)


def stressed_vowel(ipa: str) -> str:
    segs_m = segments(_clean_chars(ipa))
    st = stressed_segment(segs_m)
    if st is None:
        return ""
    return strip_stress(segs_m)[st]


@functools.lru_cache(maxsize=None)
def espeak_ipa(word: str) -> str:
    exe = shutil.which("espeak-ng")
    if not exe or not word:
        return ""
    try:
        r = subprocess.run([exe, "-v", "pt-br", "--ipa", "-q", word], capture_output=True,
                           text=True, timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return r.stdout.strip()


_VOWEL_CLASS = {"a": "a", "ɐ": "a", "e": "e", "ɛ": "e", "i": "i", "o": "o", "ɔ": "o", "u": "u"}


def _stressed_class(ipa: str) -> str:
    sv = stressed_vowel(ipa)
    return _VOWEL_CLASS.get(base(sv), "") if sv else ""


def stress_matches_oracle(word: str, ipa: str) -> bool | None:
    """True/False vs the espeak-ng pt-br stress oracle; None if unknown.

    The stressed nucleus must be the same vowel class (a/e/i/o/u, ignoring
    open/closed and nasality) AND sit at the same position counted from the
    end or from the start (glide counting differs between sources)."""
    oracle = espeak_ipa(word)
    if not oracle or " " in oracle:
        return None
    o_end, o_start, o_n = stress_positions(oracle)
    c_end, c_start, c_n = stress_positions(ipa)
    if o_end is None:
        return None
    if c_end is None:
        return c_n <= 1
    oc, cc = _stressed_class(oracle), _stressed_class(ipa)
    if oc and cc and oc != cc:
        return False
    return c_end == o_end or c_start == o_start


_FINAL_VOWEL = {"e": "i", "o": "u", "a": "ɐ"}


def final_vowel_fix(word: str, ipa: str) -> str | None:
    """Orthographic final unstressed -e/-o/-a (optionally + s) ⇒ IPA final
    nucleus i/u/ɐ. Returns the fixed IPA when the final nucleus disagrees,
    else None (aquece 'aˈkesɐ' → 'aˈkesi')."""
    w = unicodedata.normalize("NFC", word.lower())
    w = w[:-1] if w.endswith("s") and len(w) > 2 else w
    if not w or w[-1] not in _FINAL_VOWEL or len(w) < 2 or w[-2] in "aeiouáéíóúâêôãõ":
        return None
    want = _FINAL_VOWEL[w[-1]]
    segs_m = segments(ipa)
    segs = strip_stress(segs_m)
    nuc = nuclei(segs)
    if len(nuc) < 2:
        return None
    last = nuc[-1]
    if last == stressed_segment(segs_m) or is_nasal(segs[last]) or base(segs[last]) == want:
        return None
    if base(segs[last]) not in ("i", "u", "ɐ", "a", "e", "o", "ɛ", "ɔ"):
        return None
    out, k = [], 0
    for s in segs_m:
        if s != STRESS:
            if k == last:
                s = want
            k += 1
        out.append(s)
    return _palatal_fix("".join(out))


def _palatal_fix(ipa: str) -> str:
    segs = segments(ipa)
    for i in range(len(segs) - 1):
        nxt = next((s for s in segs[i + 1:] if s != STRESS), "")
        if segs[i] in ("t", "d") and base(nxt) in ("i", "j"):
            segs[i] = "tʃ" if segs[i] == "t" else "dʒ"
    return "".join(segs)


def oracle_stress_target(word: str, ipa: str) -> int | None:
    """Stress-free segment index in `ipa` that the oracle says is stressed —
    only when counting from the end and from the start agree."""
    oracle = espeak_ipa(word)
    if not oracle:
        return None
    o_end, o_start, o_n = stress_positions(oracle)
    segs_m = segments(_clean_chars(ipa))
    segs = strip_stress(segs_m)
    nuc = glide_insensitive_nuclei(segs, None)
    if o_end is None or len(nuc) != o_n:
        return None
    k_end = len(nuc) - o_end
    k_start = o_start - 1
    if k_end != k_start or not (0 <= k_end < len(nuc)):
        return None
    return nuc[k_end]


def accent_issue(word: str, ipa: str) -> str | None:
    """Written accent ⇒ stressed-vowel quality; returns the required base
    vowel when violated (e.g. 'e' for ê), else None."""
    letters = [c for c in unicodedata.normalize("NFC", word.lower()) if c in ACCENT_VOWEL_CLASS]
    if len(letters) != 1:
        return None
    allowed = ACCENT_VOWEL_CLASS[letters[0]]
    sv = stressed_vowel(ipa)
    if not sv or is_nasal(sv):
        # Nasal vowels have no open/closed contrast (também, parabéns, câmera).
        return None
    if base(sv) in allowed:
        return None
    if letters[0] in ("ê", "ô", "é", "ó"):
        return next(iter(allowed))
    return None


def accent_stress_mismatch(word: str, ipa: str) -> bool:
    """True when the word has exactly one written accent but the stressed
    nucleus is not a vowel of that accent's class at all (wrong syllable
    stressed, e.g. suicídio 'ˈsujsidʒiu'). Open/closed swaps (é↔ê, ó↔ô) are
    NOT reported here — accent_issue() fixes those deterministically."""
    letters = [c for c in unicodedata.normalize("NFC", word.lower()) if c in ACCENT_VOWEL_CLASS]
    if len(letters) != 1:
        return False
    sv = stressed_vowel(ipa)
    if not sv:
        return True
    b = base(sv)
    family = {"á": "aɐ", "â": "aɐ", "é": "eɛ", "ê": "eɛ", "í": "i", "ó": "oɔ", "ô": "oɔ",
              "ú": "u"}[letters[0]]
    return b not in family


def fix_stressed_vowel(ipa: str, required_base: str) -> str:
    segs_m = segments(ipa)
    out, seen = [], False
    for s in segs_m:
        if s == STRESS:
            seen = True
            out.append(s)
            continue
        if seen and is_vowel(s):
            s = required_base + (TILDE if is_nasal(s) else "")
            s = unicodedata.normalize("NFC", s)
            seen = False
        out.append(s)
    return "".join(out)


# ── MFA soft cross-check ─────────────────────────────────────────────────────
_MFA_MAP = {"x": "R", "c": "k", "ɟ": "ɡ", "ɾ": "R", "h": "R", "w̃": "w", "j̃": "j"}


def _skeleton(segs: list[str]) -> str:
    """Consonant skeleton for the MFA soft check. Collapses distinctions MFA
    draws differently from us: rhotic classes, coda sibilant voicing, t/d vs
    tʃ/dʒ, l vs ʎ (MFA writes l before i as ʎ), and nasal consonants (MFA
    writes nh as a nasal glide j̃)."""
    out = []
    for s in segs:
        if s == STRESS:
            continue
        b = base(s)
        if s in ("tʃ", "dʒ"):
            out.append(s[0])                       # palatalization is allophonic
        elif b in ("ɾ", "h", "x", "ʁ", "χ", "R"):
            out.append("R")
        elif b in ("s", "z", "ʃ", "ʒ"):
            out.append("S")
        elif b in ("l", "ʎ"):
            out.append("L")
        elif b in ("m", "n", "ɲ", "ŋ"):
            continue
        elif is_consonant(s):
            out.append(b)
    return "".join(out)


def mfa_consonants(phones: list[str]) -> str:
    return _skeleton([_MFA_MAP.get(p, p) for p in phones])


def consonant_skeleton(ipa: str) -> str:
    return _skeleton(segments(_clean_chars(ipa)))


def mfa_stressed_vowels(phones: list[str], ipa: str) -> set[str]:
    """Bases of the MFA vowel aligned with `ipa`'s stressed nucleus (by
    position from the end, then from the start)."""
    segs = [_MFA_MAP.get(p, p) for p in phones]
    nuc = glide_insensitive_nuclei(segs, None)
    end, start, _ = stress_positions(ipa)
    out = set()
    if end is not None and 0 < end <= len(nuc):
        out.add(base(segs[nuc[len(nuc) - end]]))
    if start is not None and 0 < start <= len(nuc):
        out.add(base(segs[nuc[start - 1]]))
    return out
