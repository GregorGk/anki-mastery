"""Stage 3 helpers: deterministic shortcuts for gender / PoS, plus tag derivation.

The Stage 3 LLM call returns gender + pos + is_cognate_en in a single
Tool Use response. Tags are computed deterministically from the row's
existing fields plus the LLM's pos/cognate output.
"""
from __future__ import annotations

# --- Function-word PoS table ------------------------------------------------

# When pt is in this map, PoS is deterministic. Used to skip LLM PoS work for
# the most common grammatical words. Mirrors the PREMIUM_FUNCTION_WORDS list
# from sense_split.py with PoS labels.
FUNCTION_WORD_POS: dict[str, str] = {
    # Articles
    "o": "art",
    "a": "art",
    "os": "art",
    "as": "art",
    "um": "art",
    "uma": "art",
    "uns": "art",
    "umas": "art",
    # Prepositions
    "de": "prep",
    "em": "prep",
    "por": "prep",
    "para": "prep",
    "com": "prep",
    "sobre": "prep",
    "sem": "prep",
    "até": "prep",
    "entre": "prep",
    "desde": "prep",
    # Conjunctions
    "e": "conj",
    "que": "conj",
    "mas": "conj",
    "se": "conj",
    "ou": "conj",
    "porque": "conj",
    "embora": "conj",
    "quando": "conj",
    # Subject pronouns
    "eu": "pron",
    "tu": "pron",
    "ele": "pron",
    "ela": "pron",
    "nós": "pron",
    "vocês": "pron",
    "eles": "pron",
    "elas": "pron",
    "você": "pron",
    # Object pronouns
    "me": "pron",
    "te": "pron",
    "lhe": "pron",
    "lhes": "pron",
    "nos": "pron",
    "vos": "pron",
    # Demonstrative pronouns
    "isto": "pron",
    "isso": "pron",
    "aquilo": "pron",
    "este": "pron",
    "esta": "pron",
    "esse": "pron",
    "essa": "pron",
    "aquele": "pron",
    "aquela": "pron",
    # Possessives
    "meu": "pron",
    "minha": "pron",
    "teu": "pron",
    "tua": "pron",
    "seu": "pron",
    "sua": "pron",
    "nosso": "pron",
    "nossa": "pron",
    # Question / relative
    "quem": "pron",
    "qual": "pron",
    "onde": "adv",
}

NUMERALS: frozenset[str] = frozenset(
    {
        "um",
        "uma",
        "dois",
        "duas",
        "três",
        "quatro",
        "cinco",
        "seis",
        "sete",
        "oito",
        "nove",
        "dez",
        "onze",
        "doze",
        "treze",
        "catorze",
        "quatorze",
        "quinze",
        "dezesseis",
        "dezessete",
        "dezoito",
        "dezenove",
        "vinte",
        "trinta",
        "quarenta",
        "cinquenta",
        "sessenta",
        "setenta",
        "oitenta",
        "noventa",
        "cem",
        "cento",
        "mil",
        "milhão",
        "bilhão",
        "primeiro",
        "segundo",
        "terceiro",
        "quarto",
        "quinto",
        "sexto",
        "sétimo",
        "oitavo",
        "nono",
        "décimo",
    }
)


# --- Deterministic shortcuts ------------------------------------------------


def deterministic_pos(row: dict) -> str:
    """Return PoS string when 100% certain, else empty string."""
    pt = (row.get("pt") or "").strip().lower()
    en_primary = (row.get("en_primary") or "").strip().lower()
    annotation = (row.get("annotation") or "").lower()
    expansion_index = int(row.get("expansion_index") or 0)
    split_category = row.get("split_category") or ""

    # Idiom expansion rows
    if expansion_index > 0:
        return "idiom"
    if split_category == "idiom_expansion":
        return "idiom"

    # Function words from the table
    if pt in FUNCTION_WORD_POS:
        # But only if the en_primary doesn't suggest noun usage (e.g. `o` as object pronoun)
        return FUNCTION_WORD_POS[pt]

    # Numerals
    if pt in NUMERALS:
        return "num"

    # Reflexive verbs (annotation marker from Stage 1a)
    if "reflexive" in annotation:
        return "verb"

    # "to X" pattern in en_primary
    if en_primary.startswith("to "):
        return "verb"

    # Bare adjective heuristic: en_primary is exactly one or two short words
    # without "to" prefix and without a / -- hard to be 100% certain. Skip.

    return ""


def deterministic_gender(row: dict) -> str:
    """Return gender if already populated upstream (forced gender splits), else empty."""
    gender = (row.get("gender") or "").strip()
    if gender in ("o", "a", "o/a"):
        return gender
    return ""


# --- Display + tags ---------------------------------------------------------


def compose_pt_display(pt: str, gender: str, pt_type: str) -> str:
    """Produce the article-prefixed display form for nouns; bare otherwise.

    BP convention: o caminho (M), a casa (F), o/a estudante (both).
    Idioms and space-compounds are shown as-is (no leading article).
    """
    if pt_type in ("idiom", "space_compound", "abbreviation_expansion"):
        return pt
    if gender == "o":
        return f"o {pt}"
    if gender == "a":
        return f"a {pt}"
    if gender == "o/a":
        return f"o/a {pt}"
    return pt


# Locked regional list. Per plan pronoun policy, `tu` is the canonical regional
# BP form. Extend this list if additional regional markers surface in QA.
REGIONAL_PT: frozenset[str] = frozenset({"tu"})


def derive_tags(row: dict, *, gender: str, pos: str, cognate_en: bool) -> str:
    """Compose space-separated tags from existing row fields + LLM output.

    Tag taxonomy matches docs/plan.md §"Tag taxonomy" exactly:
      - Morphology / pt_type: #single-word #hyphenated-compound #space-compound
        #idiom #abbreviation-expansion #reflexive #gendered-meaning
      - Regional: #bp-rare #regional
      - Special: #nsfw #false-friend #cognate-en #function-word #pronoun
    """
    tags: list[str] = []

    # Frequency tier
    rank = int(row.get("rank") or 0)
    if rank <= 500:
        tags.append("#top500")
    elif rank <= 1000:
        tags.append("#top1000")
    elif rank <= 2000:
        tags.append("#top2000")
    elif rank <= 3000:
        tags.append("#top3000")
    else:
        tags.append("#top5000")

    # PoS
    if pos:
        tags.append(f"#{pos}")

    # Morphology / pt_type — exact tag names per plan
    pt_type = row.get("pt_type") or ""
    if pt_type == "single_word":
        tags.append("#single-word")
    elif pt_type == "hyphenated_compound":
        tags.append("#hyphenated-compound")
    elif pt_type == "space_compound":
        tags.append("#space-compound")
    elif pt_type == "abbreviation_expansion":
        tags.append("#abbreviation-expansion")

    # Idiom expansion (any row whose expansion_index > 0)
    if int(row.get("expansion_index") or 0) > 0:
        tags.append("#idiom")

    # Reflexive: detect via source_line containing the +se marker (more
    # reliable than the annotation field, which is often empty after Stage 2
    # sense splits drop the marker from en_primary).
    source_line = row.get("source_line") or ""
    annotation = row.get("annotation") or ""
    if "+se" in source_line or "reflexive" in annotation.lower():
        tags.append("#reflexive")

    # Forced gender split + function word
    split_category = row.get("split_category") or ""
    if split_category == "forced_gender_split":
        tags.append("#gendered-meaning")
    if split_category == "function_word_polysemy":
        tags.append("#function-word")
    if pos == "pron":
        tags.append("#pronoun")

    # Regional
    pt_lower = (row.get("pt") or "").strip().lower()
    if pt_lower in REGIONAL_PT:
        tags.append("#regional")

    # Status
    bp_status = row.get("bp_status") or ""
    if bp_status == "uncommon":
        tags.append("#bp-rare")
    if bp_status == "nsfw":
        tags.append("#nsfw")
    if bp_status == "false_friend":
        tags.append("#false-friend")

    # Cognate
    if cognate_en:
        tags.append("#cognate-en")

    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)
