"""Stage 19 — audio-judge prompt versions shared by the Gemini and OpenAI judges.

  J1   the production prompt, byte-identical (keeps v3 verdicts comparable)
  J1p  J1 + one checklist line: São Paulo coda r ([ɾ]) and [h]/[x]/[χ] are
       both standard BP — never flag either. (Our IPA v2 target is
       São Paulo, so a judge that dislikes a tapped coda r would reject
       exactly the pronunciation we asked for.)
  J2   J1p + an expected-pronunciation block: the caller passes the
       validated São Paulo IPA of the target and the judge must flag a
       wrong stressed syllable, a wrong open/closed stressed vowel, a wrong
       consonant or a missing/extra syllable — while ignoring free
       variation (coda-r variant, nasalization degree, pretonic raising).

`build_system_prompt(base, version)` derives J1p/J2 from each client's own
J1 text by inserting at fixed anchors (asserted, so a prompt edit can't
silently skip the insertion).
"""
from __future__ import annotations

import hashlib

PROMPT_VERSIONS = ("J1", "J1p", "J2")

_CHECKLIST_ANCHOR = (
    "- Silent initial h (hora, hospital, hotel) → no aspiration. "
    "NOT English aspirated [h].\n"
)
_DRIFT_ANCHOR = "Common drift directions:"

J1P_CODA_R_LINE = (
    "- Coda r (before a consonant or at the end of a word) → tap [ɾ] "
    "(São Paulo) or back fricative [h]/[x]/[χ]. Both are standard Brazilian "
    "Portuguese — never flag either.\n"
)

J2_EXPECTED_BLOCK = """Expected pronunciation check:
The message gives the EXPECTED São Paulo pronunciation of the target in IPA
(primary stress mark ˈ before the stressed syllable). Compare the audio with
it. Return `non_bp` (drift `other` unless a language drift is evident) if you
hear ANY of:
- primary stress on a different syllable than the expected ˈ;
- the wrong quality of the STRESSED vowel: open [ɛ]/[ɔ] vs closed [e]/[o];
- a wrong consonant, e.g. [t]/[d] instead of [tʃ]/[dʒ] before [i], a
  consonantal [l] instead of [w] in a coda, [ʃ] for a coda s;
- a missing or an extra syllable.
Do NOT flag: coda-r variants ([ɾ] [h] [x] [χ]), the degree of nasalization,
unstressed pretonic raising or reduction (e→i, o→u, a→ɐ), or a very short
epenthetic vowel.

"""


def build_system_prompt(base: str, version: str) -> str:
    """Return the system prompt for `version`, derived from the J1 `base`."""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt_version must be one of {PROMPT_VERSIONS}, got {version!r}")
    if version == "J1":
        return base
    if _CHECKLIST_ANCHOR not in base or _DRIFT_ANCHOR not in base:
        raise ValueError("judge prompt anchors missing — J1p/J2 can't be derived")
    out = base.replace(_CHECKLIST_ANCHOR, _CHECKLIST_ANCHOR + J1P_CODA_R_LINE, 1)
    if version == "J2":
        out = out.replace(_DRIFT_ANCHOR, J2_EXPECTED_BLOCK + _DRIFT_ANCHOR, 1)
    return out


def user_text(*, pt: str, ipa: str, clip_type: str, version: str,
              json_only_suffix: str = "") -> str:
    """User message. J1/J1p keep the production wording; J2 labels the IPA
    as the expected São Paulo pronunciation the judge must check against."""
    if version == "J2":
        text = (
            f"Word: {pt}\n"
            f"Expected São Paulo pronunciation (IPA): /{ipa}/\n"
            f"Clip type: {clip_type}"
        )
    else:
        text = (
            f"Word: {pt}\n"
            f"Target IPA (Brazilian Portuguese): {ipa}\n"
            f"Clip type: {clip_type}"
        )
    return text + json_only_suffix


def prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
