"""Validation helpers for Stage 4 + 5 outputs.

`token_in_sentence` — checks whether a target word appears as a complete
token in an example sentence, handling BP-specific edge cases:

- Hyphens preserved (so `primeiro-ministro`, `segunda-feira`, enclitic
  pronouns like `dizer-lhe` aren't split)
- Apostrophes preserved (so `n'água` doesn't break)
- Other punctuation stripped (period, comma, semicolon, exclamation, etc.)
- Case-insensitive comparison
- Substring traps avoided (e.g., `por` inside `porque` is NOT a match;
  `casa` inside `casamento` is NOT a match)

Per docs/plan.md §"Stage 4 — Example sentences": Python's `\b` regex
treats hyphens as non-word characters which would split `primeiro-ministro`
into two boundaries. Token-list comparison avoids this.
"""
from __future__ import annotations

# Punctuation to strip (keep hyphens and apostrophes as part of tokens)
_PUNCT = '.,;:!?¿¡«»"\'()[]{}…—–'
_PUNCT_TRANS = str.maketrans('', '', _PUNCT)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Hyphens and apostrophes are kept; all other punctuation stripped.
    Whitespace boundaries split tokens.
    """
    return text.lower().translate(_PUNCT_TRANS).split()


def token_in_sentence(target: str, sentence: str) -> bool:
    """Return True iff target appears as a complete token (or contiguous
    multi-token phrase) in sentence.

    Single-word targets: exact token match.
    Multi-word targets: contiguous token sequence match.

    Both target and sentence are lowercased and stripped of punctuation
    (except hyphens and apostrophes) before comparison.

    Examples:
        token_in_sentence("por", "Por que você foi?")     # True
        token_in_sentence("por", "Eu não sei porque.")    # False — substring trap
        token_in_sentence("casa", "Casa nova.")           # True
        token_in_sentence("casa", "Casamento bonito.")    # False — substring trap
        token_in_sentence("primeiro-ministro", "O primeiro-ministro falou.")  # True
        token_in_sentence("dizer-lhe", "Vou dizer-lhe agora.")  # True
        token_in_sentence("à medida que", "À medida que cresço, aprendo.")  # True
        token_in_sentence("em vigor", "A lei está em vigor desde janeiro.")  # True
        token_in_sentence("em vigor", "Em vigência total agora.")            # False
    """
    target_tokens = tokenize(target)
    if not target_tokens:
        return False

    sentence_tokens = tokenize(sentence)

    if len(target_tokens) == 1:
        return target_tokens[0] in sentence_tokens

    # Multi-word target: sliding window over sentence tokens
    n = len(target_tokens)
    if n > len(sentence_tokens):
        return False
    for i in range(len(sentence_tokens) - n + 1):
        if sentence_tokens[i : i + n] == target_tokens:
            return True
    return False


def word_count(text: str) -> int:
    """Count whitespace-delimited tokens. Useful for the ≤15 word rule."""
    return len(text.split())


def example_within_word_limit(text: str, limit: int = 15) -> bool:
    return word_count(text) <= limit
