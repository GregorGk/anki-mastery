"""Deterministic post-LLM rules for usage_hint sidecar entries.

The classifier (build/12_0_usage_hint_classifier.py) makes a per-row LLM
call and gets back {usage_hint, hint_priority, confidence, reason}. This
module then applies hard rules on top:

  1. TIER_1_OVERRIDES — a tiny (6-lemma) safety net for high-frequency
     active-production traps the user wants surfaced even when the row's
     example does NOT exercise them. v2 strict Rule 1 correctly omits
     these; the override re-installs the hint at essential priority.

  2. ESSENTIAL_TO_USEFUL — specific lemmas where the user said the LLM's
     "essential" tag is over-generous. Downgrades them to "useful". The
     downstream export filter (in derive_final.py) hides "useful" from
     learner-facing cards unless rank <= 1000 or row has a risk tag.

  3. Parenthetical strip — removes trailing clarifications like
     "(do = de + o)" or "(na = em + a)" that the example already teaches.

  4. RISK_NOTE_OVERRIDES — for register/NSFW warnings (currently `gozar`),
     move the warning out of usage_hint and into a separate risk_note
     field so card layout can render it differently.

All four passes are idempotent: re-applying them yields the same output.
"""
from __future__ import annotations

import re


TIER_1_OVERRIDES: dict[str, str] = {
    "ajudar":    "ajudar + person + a + infinitive",
    "ensinar":   "ensinar + person + a + infinitive",
    "aprender":  "aprender a + infinitive",
    "parar":     "parar de + infinitive = stop doing",
    "acreditar": "acreditar em + noun; acreditar que + clause",
    "esquecer":  "esquecer + noun; esquecer-se de + noun",
}

ESSENTIAL_TO_USEFUL_LEMMAS: set[str] = {
    "ficar", "passar", "levar", "voltar", "tomar", "continuar", "achar",
    "contar", "marcar", "atender", "investir", "provir", "romper",
}

# Lemma -> (usage_hint_replacement, risk_note). usage_hint goes empty when
# risk_note is the right surface; the row keeps a "useful" priority so the
# audit trail records that the LLM saw the row, but the hint moves field.
RISK_NOTE_OVERRIDES: dict[str, tuple[str, str]] = {
    "gozar": ("", "BP warning: strong sexual slang; avoid for neutral 'enjoy'"),
}

# Strips a trailing parenthetical aside like " (do = de + o)" or
# " (na = em + a)" — but NOT internal parens like "...em + a/para + ...".
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def apply_post_process(row: dict) -> dict:
    """Mutate `row` in place with the four deterministic passes.

    Expected keys (writes them, creates missing ones with defaults):
      pt, usage_hint, hint_priority, risk_note

    `confidence` and `reason` are not touched. Returns the same dict for
    convenience.
    """
    lemma = row.get("pt", "")
    hint = (row.get("usage_hint") or "").strip()
    priority = row.get("hint_priority") or "omit"
    row.setdefault("risk_note", "")

    # 1. Tier-1 override (lemma-anchored re-install).
    if lemma in TIER_1_OVERRIDES:
        hint = TIER_1_OVERRIDES[lemma]
        priority = "essential"

    # 2. Essential -> useful downgrade.
    if priority == "essential" and lemma in ESSENTIAL_TO_USEFUL_LEMMAS:
        priority = "useful"

    # 3. Strip trailing parenthetical (visual noise).
    hint = _TRAILING_PAREN_RE.sub("", hint).strip()

    # 4. Risk-note rerouting.
    if lemma in RISK_NOTE_OVERRIDES:
        new_hint, new_risk = RISK_NOTE_OVERRIDES[lemma]
        hint = new_hint
        row["risk_note"] = new_risk
        # Keep priority where the LLM put it; the row is informational, not
        # a construction trap. If the LLM marked it essential, downgrade.
        if priority == "essential":
            priority = "useful"

    # Normalize: priority=omit must mean empty hint (and vice-versa).
    if priority == "omit":
        hint = ""
    elif not hint and not row["risk_note"]:
        priority = "omit"

    row["usage_hint"] = hint
    row["hint_priority"] = priority
    return row
