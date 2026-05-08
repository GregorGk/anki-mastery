"""eSpeak-NG IPA helpers for Stage 5 baseline transcription.

Provides:
- transcribe_word(text): single-word IPA via eSpeak-NG (`-v pt-br --ipa -q`)
- transcribe_tokens(text): tokenize `text` (preserving hyphens/apostrophes
  per the validate.py rule), call eSpeak per token, return space-separated
  isolated forms.

Why isolated forms? The plan calls per-word IPA "isolated form, space-separated".
eSpeak's connected-speech output applies sandhi (vowel reductions, /s/-linking,
contractions) which differs from word-level pronunciation. The audio in
Stage 6/7 will show natural sandhi; the IPA column is per-word reference.
"""
from __future__ import annotations

import shutil
import subprocess

from build.lib.validate import tokenize

ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak")


class EspeakNotFound(RuntimeError):
    pass


def _ensure_espeak() -> str:
    if ESPEAK_BIN is None:
        raise EspeakNotFound(
            "espeak-ng not found in PATH. Install with `brew install espeak-ng` "
            "(macOS) or your platform's package manager."
        )
    return ESPEAK_BIN


def transcribe_word(text: str, *, voice: str = "pt-br", timeout: float = 5.0) -> str:
    """Run eSpeak-NG once on `text`, return IPA string.

    Empty input -> empty string. Whitespace stripped from output.
    Uses `--ipa -q` to get IPA only with no audio output.

    Note: eSpeak-NG produces a single connected utterance even for multi-word
    input. For per-token isolated forms, use transcribe_tokens().
    """
    text = text.strip()
    if not text:
        return ""
    bin_ = _ensure_espeak()
    try:
        result = subprocess.run(
            [bin_, "-v", voice, "--ipa", "-q", text],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"espeak-ng failed on input {text!r}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def transcribe_tokens(
    text: str, *, voice: str = "pt-br", timeout: float = 5.0
) -> str:
    """Tokenize `text`, transcribe each token in isolation, join with spaces.

    Token boundary rules from build/lib/validate.py: hyphens and apostrophes
    are kept (so `primeiro-ministro` is one token, `dizer-lhe` is one token).

    Returns space-separated IPA tokens. Token count matches tokenize(text).
    """
    text = text.strip()
    if not text:
        return ""
    tokens = tokenize(text)
    if not tokens:
        return ""
    ipa_tokens: list[str] = []
    for tok in tokens:
        ipa_tokens.append(transcribe_word(tok, voice=voice, timeout=timeout))
    return " ".join(ipa_tokens)


def token_count(ipa: str) -> int:
    """Count whitespace-delimited IPA tokens — used for invariant checking."""
    return len(ipa.split())
