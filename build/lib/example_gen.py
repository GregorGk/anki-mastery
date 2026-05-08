"""Stage 4 helpers: example policy lookup, prompt formatting, deterministic
validators.

Stage 2.5 (sensitive-term screen) is not built yet, so this module provides
a deterministic fallback that derives `example_policy` directives from
`bp_status` + `tags` + keyword screen on the headword. Once Stage 2.5
exists, replace `_default_example_policy` with a `_sensitive_terms.tsv`
lookup.
"""
from __future__ import annotations

# Sensitive keyword screen — same list as tests/build_smoke_sample_1000.py
# Intent: catch terms that need careful example framing even when not
# flagged via bp_status.
_SENSITIVE_KEYWORDS = {
    "gozar",
    "mulato",
    "índio",
    "aborto",
    "arma",
    "bala",
    "faca",
    "morrer",
    "matar",
    "droga",
    "cigano",
    "raça",
    "homossexual",
    "violar",
    "suicidar",
    "espingarda",
    "revólver",
    "religião",
    "corno",
    "puta",
    "puto",
    "merda",
    "porra",
    "caralho",
    "buceta",
    "bicha",
    "viado",
    "rapariga",
    "preto",
    "negro",
}

# Policy directives keyed on category. Generator prompt receives one of these
# strings directly. Keep them short and prescriptive.
_POLICY_BY_CATEGORY = {
    "nsfw_sexual": (
        "Use a clinical, news, or educational register. No sexual detail. "
        "No graphic body description."
    ),
    "nsfw_violence": (
        "Use a news, scholarly, or hypothetical register. No graphic violence. "
        "No first-person violence."
    ),
    "slur_or_identity": (
        "Use a scholarly, historical, or descriptive register. Frame the term "
        "neutrally — never affectively negative or insulting."
    ),
    "false_friend": (
        "Make the BP sense unambiguous in context. Avoid contexts where the "
        "EP cognate's meaning would also fit, since that's where learners "
        "get confused."
    ),
    "weapon": (
        "Use a hypothetical, news, or hunting/sport register. No first-person "
        "violence. No description of injury."
    ),
    "drug_or_substance": (
        "Use a clinical or news register. No glamorization. No first-person "
        "consumption."
    ),
}


def example_policy_for_row(row: dict) -> str:
    """Return a short policy directive for sensitive rows; empty string for safe rows.

    Priority order:
      1. bp_status='nsfw' + sexual keyword → nsfw_sexual
      2. bp_status='nsfw' + violence keyword → nsfw_violence
      3. bp_status='nsfw' (catch-all) → slur_or_identity
      4. bp_status='false_friend' → false_friend
      5. weapon keyword (faca, arma, espingarda, revólver, bala) → weapon
      6. drug keyword (droga) → drug_or_substance
      7. violence keyword (matar, morrer, suicidar, violar, aborto) → nsfw_violence
      8. slur keyword (mulato, índio, cigano, etc.) → slur_or_identity
      9. otherwise empty string (no policy)
    """
    pt = (row.get("pt") or "").strip().lower()
    bp_status = (row.get("bp_status") or "").strip().lower()

    sexual_keywords = {
        "gozar",
        "puta",
        "puto",
        "buceta",
        "caralho",
        "rapariga",
        "homossexual",
    }
    violence_keywords = {
        "matar",
        "morrer",
        "suicidar",
        "violar",
        "aborto",
    }
    weapon_keywords = {"faca", "arma", "espingarda", "revólver", "bala"}
    drug_keywords = {"droga"}
    slur_keywords = {
        "mulato",
        "índio",
        "cigano",
        "preto",
        "negro",
        "raça",
        "viado",
        "bicha",
        "corno",
    }

    if bp_status == "nsfw":
        if pt in sexual_keywords:
            return _POLICY_BY_CATEGORY["nsfw_sexual"]
        if pt in violence_keywords:
            return _POLICY_BY_CATEGORY["nsfw_violence"]
        return _POLICY_BY_CATEGORY["slur_or_identity"]
    if bp_status == "false_friend":
        return _POLICY_BY_CATEGORY["false_friend"]
    if pt in weapon_keywords:
        return _POLICY_BY_CATEGORY["weapon"]
    if pt in drug_keywords:
        return _POLICY_BY_CATEGORY["drug_or_substance"]
    if pt in violence_keywords:
        return _POLICY_BY_CATEGORY["nsfw_violence"]
    if pt in slur_keywords:
        return _POLICY_BY_CATEGORY["slur_or_identity"]
    return ""


def build_generator_user_message(row: dict) -> str:
    """Format the per-row user message for the example generator."""
    policy = example_policy_for_row(row)
    parts = [
        f"pt: {row.get('pt', '')}",
        f"pt_display: {row.get('pt_display', '')}",
        f"pos: {row.get('pos', '')}",
        f"gender: {row.get('gender', '')}",
        f"en_primary: {row.get('en_primary', '')}",
        f"en_all: {row.get('en_all', '')}",
        f"pt_type: {row.get('pt_type', '')}",
        f"bp_status: {row.get('bp_status', 'standard')}",
        f"tags: {row.get('tags', '')}",
    ]
    if policy:
        parts.append(f"example_policy: {policy}")
    return "\n".join(parts)


def build_validator_user_message(
    row: dict,
    *,
    example_pt: str,
    example_en: str,
    target_word_used: str,
) -> str:
    """Format the per-row user message for the cross-family validator."""
    policy = example_policy_for_row(row)
    parts = [
        f"pt: {row.get('pt', '')}",
        f"en_primary: {row.get('en_primary', '')}",
        f"en_all: {row.get('en_all', '')}",
        f"pos: {row.get('pos', '')}",
        f"tags: {row.get('tags', '')}",
        f"example_pt: {example_pt}",
        f"example_en: {example_en}",
        f"target_word_used: {target_word_used}",
    ]
    if policy:
        parts.append(f"example_policy: {policy}")
    return "\n".join(parts)
