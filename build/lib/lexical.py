"""Stage 1c helpers: lexical EP→BP replacement + collision merge.

`_lexical_bp_replacements.tsv` has whole-word swaps like `comboio` → `trem`.
When the BP target already exists as another row's `pt`, we have a collision.
This module decides merge vs. manual_review based on English gloss overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .tsv import read_tsv


@dataclass
class LexicalRule:
    source_pt: str
    bp_replacement: str
    action: str
    confidence: str
    note: str = ""


def load_lexical_replacements(path: str | Path) -> dict[str, LexicalRule]:
    """Read `_lexical_bp_replacements.tsv` and key by source_pt."""
    out: dict[str, LexicalRule] = {}
    for row in read_tsv(path):
        rule = LexicalRule(
            source_pt=row["source_pt"],
            bp_replacement=row["bp_replacement"],
            action=row.get("action", "replace_or_merge"),
            confidence=row.get("confidence", ""),
            note=row.get("note", ""),
        )
        out[rule.source_pt] = rule
    return out


# --- Gloss similarity --------------------------------------------------------

# Tokens we DO NOT count as overlap (English stopwords + dictionary scaffolding).
_GLOSS_STOPWORDS: frozenset[str] = frozenset(
    {
        # English articles, prepositions, copulas
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "be",
        "or",
        "and",
        "as",
        "into",
        "out",
        # Dictionary scaffolding
        "bp",
        "ep",
    }
)

_GLOSS_SPLIT_RE = re.compile(r"[\s/,;()\[\]]+")
_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


def gloss_tokens(en_all: str) -> set[str]:
    """Tokenize an English gloss into a set of meaningful content words.

    Lowercases, splits on whitespace + slash + punctuation, strips edge punctuation,
    drops stopwords and tokens shorter than 3 characters.
    """
    if not en_all:
        return set()
    tokens = _GLOSS_SPLIT_RE.split(en_all.lower())
    out: set[str] = set()
    for tok in tokens:
        tok = _PUNCT_STRIP_RE.sub("", tok)
        if not tok:
            continue
        if len(tok) < 3:
            continue
        if tok in _GLOSS_STOPWORDS:
            continue
        out.add(tok)
    return out


@dataclass
class GlossSimilarity:
    overlap: set[str]
    score: float  # 0.0 .. 1.0; jaccard
    is_similar: bool


def gloss_similarity(en_a: str, en_b: str, *, min_overlap_tokens: int = 1) -> GlossSimilarity:
    """Decide whether two English glosses describe the same concept.

    Heuristic: at least `min_overlap_tokens` content-word in common AND
    Jaccard ≥ 0.10. The second condition rejects spurious matches like both
    glosses containing 'small' or 'large' but otherwise unrelated.
    """
    tokens_a = gloss_tokens(en_a)
    tokens_b = gloss_tokens(en_b)
    if not tokens_a or not tokens_b:
        return GlossSimilarity(set(), 0.0, False)

    overlap = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(overlap) / len(union) if union else 0.0

    is_similar = len(overlap) >= min_overlap_tokens and jaccard >= 0.10
    return GlossSimilarity(overlap=overlap, score=jaccard, is_similar=is_similar)
