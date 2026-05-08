"""Deterministic pre-classifier for Stage 2 sense splitting.

Routes each input row to one of:
    - SINGLE_SENSE_NO_SPLIT     en_all has no '/'; one sense, en_primary = en_all
    - IDIOM_EXPANSION           expansion_index >= 1; needs en_primary from LLM
    - FORCED_GENDER_SPLIT       en_all has (M ... / F ...) pattern; 2 senses
    - FUNCTION_WORD_POLYSEMY    pt is in PREMIUM_FUNCTION_WORDS; LLM premium tier
    - LEXICAL_POLYSEMY          everything else with '/'; LLM default tier

Each category has a fixed routing decision (LLM tier or no LLM at all).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Categories
CAT_SINGLE_SENSE_NO_SPLIT = "single_sense_no_split"
CAT_IDIOM_EXPANSION = "idiom_expansion"
CAT_FORCED_GENDER_SPLIT = "forced_gender_split"
CAT_FUNCTION_WORD_POLYSEMY = "function_word_polysemy"
CAT_LEXICAL_POLYSEMY = "lexical_polysemy"

# Curated list of headwords whose polysemy is grammatical and warrants premium-tier
# (Opus) treatment. Keeping this list small (~30) keeps premium-tier cost under $1.
# Picked from the top frequency band; only words whose senses are decided by
# grammatical role (article vs pronoun, copula vs auxiliary, etc.) are included.
PREMIUM_FUNCTION_WORDS = frozenset(
    {
        # Articles + object pronouns + demonstratives (high polysemy)
        "o",
        "a",
        "os",
        "as",
        # Prepositions with multiple senses
        "de",
        "em",
        "por",
        "para",
        "com",
        "sobre",
        # Conjunctions
        "e",
        "que",
        "mas",
        "se",
        "ou",
        # Subject and object pronouns
        "eu",
        "tu",
        "ele",
        "ela",
        "nós",
        "vocês",
        "me",
        "te",
        "lhe",
        "nos",
        # Demonstratives
        "isso",
        "isto",
        "aquilo",
        # Possessives
        "meu",
        "seu",
        "nosso",
        # Auxiliaries / copulas
        "ser",
        "estar",
        "ter",
        "haver",
        # Question words
        "quem",
        "qual",
        "quando",
        "onde",
        # Indefinites
        "um",
        "uma",
    }
)

# Pattern for "(M ... / F ...)" or "(F ... / M ...)" within a parenthetical
_FORCED_GENDER_INLINE = re.compile(
    r"\((?:M\s+[^)]+/\s*F\s|F\s+[^)]+/\s*M\s)",
    re.IGNORECASE,
)

# Pattern for "(M)" / "(F)" markers per sense (e.g. "goat (F) / guy (M)")
_FORCED_GENDER_PER_SENSE = re.compile(
    r"\(M\)[^)]*\(F\)|\(F\)[^)]*\(M\)",
    re.IGNORECASE,
)


@dataclass
class Categorization:
    category: str
    needs_llm: bool
    llm_tier: str  # "default" | "premium" | "" (when no LLM)
    reason: str = ""


def has_forced_gender_split(en_all: str) -> bool:
    """Return True if en_all has an explicit M/F gender split pattern."""
    if _FORCED_GENDER_INLINE.search(en_all):
        return True
    if _FORCED_GENDER_PER_SENSE.search(en_all):
        return True
    return False


def categorize(row: dict) -> Categorization:
    """Decide how to handle a row in Stage 2.

    The row dict must have at least: pt, en_all, expansion_index.
    """
    pt = (row.get("pt") or "").strip()
    en_all = row.get("en_all") or ""
    expansion_index = int(row.get("expansion_index") or 0)

    # Idiom expansions from Stage 1a are already separate rows with their own
    # pt; they need an en_primary that fits the idiom (not the parent's en_all).
    if expansion_index > 0:
        return Categorization(
            category=CAT_IDIOM_EXPANSION,
            needs_llm=True,
            llm_tier="default",
            reason="idiom expansion needs en_primary",
        )

    # Forced gender splits: 8 entries with (M / F) markers in en_all.
    if has_forced_gender_split(en_all):
        return Categorization(
            category=CAT_FORCED_GENDER_SPLIT,
            needs_llm=True,
            llm_tier="premium",
            reason="M/F gender split",
        )

    # Function-word polysemy: top-frequency grammar words.
    if pt in PREMIUM_FUNCTION_WORDS:
        return Categorization(
            category=CAT_FUNCTION_WORD_POLYSEMY,
            needs_llm=True,
            llm_tier="premium",
            reason="function-word polysemy is grammatical, not lexical",
        )

    # No slash -> single sense, no LLM needed.
    if "/" not in en_all:
        return Categorization(
            category=CAT_SINGLE_SENSE_NO_SPLIT,
            needs_llm=False,
            llm_tier="",
            reason="no slash in en_all",
        )

    # Default: lexical polysemy candidate -> LLM default tier.
    return Categorization(
        category=CAT_LEXICAL_POLYSEMY,
        needs_llm=True,
        llm_tier="default",
        reason="lexical polysemy candidate",
    )
