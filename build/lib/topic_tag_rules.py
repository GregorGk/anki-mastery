"""Stage 14 — Primary topic tag (`#topic-*`) taxonomy + deterministic POS rules.

This module owns the controlled vocabulary that gates every row in
`data/06-final.tsv`:

  1. `ALLOWED_TOPIC_TAGS`        — the frozen 50-tag enum.
  2. `POS_TOPIC_PRERULES`        — deterministic POS → tag map (155 rows).
  3. `apply_post_process(row)`   — idempotent per-row hygiene pass (enum
                                    validation, prefix normalization,
                                    confidence default).

Both the classifier (build/14_0_topic_classifier.py) and the verifier
(build/verify_all.py::_verify_topic_tags) import from here so the
allowlist is single-sourced.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Frozen 50-tag taxonomy
# --------------------------------------------------------------------------- #

# Order is taxonomy-stable (matches docs/plan.md §"Taxonomy") so anyone
# diffing this list against the plan can scan visually.
ALLOWED_TOPIC_TAGS: tuple[str, ...] = (
    "#topic-grammar",
    "#topic-person-identity",
    "#topic-family-relations",
    "#topic-character-qualities",     # renamed from #topic-character-behavior
    "#topic-feelings",
    "#topic-body",
    "#topic-health-care",
    "#topic-hygiene",
    "#topic-appearance",
    "#topic-food-drink",
    "#topic-home-household",
    "#topic-daily-routines",
    "#topic-clothing-style",
    "#topic-shopping-services",
    "#topic-work-jobs",
    "#topic-money",
    "#topic-economy",
    "#topic-law-rules",
    "#topic-government-admin",
    "#topic-politics",
    "#topic-social-life",
    "#topic-social-problems",
    "#topic-speech-language",
    "#topic-opinion-belief",
    "#topic-learning-education",
    "#topic-science",
    "#topic-media-technology",
    "#topic-culture-art",
    "#topic-religion-belief",
    "#topic-time-calendar",
    "#topic-numbers",
    "#topic-measurement",
    "#topic-colors",
    "#topic-shapes",
    "#topic-sound",
    "#topic-movement-position",
    "#topic-transport",
    "#topic-travel",
    "#topic-city-places",
    "#topic-countryside",
    "#topic-countries-languages",
    "#topic-nature-landscape",
    "#topic-plants",
    "#topic-animals",
    "#topic-weather",
    "#topic-environment",
    "#topic-materials-substances",
    "#topic-objects-tools",
    "#topic-sport-leisure",
    "#topic-danger-disaster",
)

# Set for O(1) membership checks.
ALLOWED_TOPIC_TAG_SET: frozenset[str] = frozenset(ALLOWED_TOPIC_TAGS)

# Tool-schema enum order: sorted for stable JSON output across runs.
ALLOWED_TOPIC_TAGS_SORTED: tuple[str, ...] = tuple(sorted(ALLOWED_TOPIC_TAGS))


# --------------------------------------------------------------------------- #
# Deterministic POS pre-rules — no LLM call
# --------------------------------------------------------------------------- #

# Locked decision from plan: only these POS values skip the LLM.
# interj + idiom go to the LLM (locked Q&A, content-bearing).
POS_TOPIC_PRERULES: dict[str, str] = {
    "prep": "#topic-grammar",
    "conj": "#topic-grammar",
    "pron": "#topic-grammar",
    "art":  "#topic-grammar",
    "num":  "#topic-numbers",
}

# Pilot-v1 surfaced four rows where the source dictionary tagged `pos=num`
# but the SENSE isn't numeric. Without these overrides, `quarto = bedroom`
# would ship `#topic-numbers`. Tied by sense_id (not lemma) because some
# lemmas have a numeric sense AND a non-numeric sense.
NUM_TOPIC_OVERRIDES: dict[str, str] = {
    "0107.00.02": "#topic-grammar",          # segundo = "according to" (prep-like)
    "0503.00.01": "#topic-home-household",   # quarto = room
    "0503.00.02": "#topic-home-household",   # quarto = bedroom
    "0659.00.01": "#topic-measurement",      # cento (por c. = percent)
}

# Manual topic overrides keyed by sense_id — applied BEFORE the LLM call so
# the override carries source="manual" provenance. Use when the LLM's
# classification is genuinely off due to a known example-contamination
# pattern and the right tag is unambiguous.
#
# Discovered overrides:
#   0360.00.01 ocorrer (to occur)     — pilot v2: example "O acidente
#     ocorreu ontem" pushed the LLM to #topic-danger-disaster, but the
#     lemma is a generic event verb. The model's own reason admitted the
#     example was incidental.
#   1845.00.01 fundação (founding of a city)  — full run: LLM tried to
#     return #topic-history (not in allowlist) for historical/civic senses.
#   3688.00.01 decadência (decline of the Roman Empire) — same: LLM wanted
#     #topic-history; closest in-taxonomy match is #topic-social-problems.
#   4692.00.01 legião (Roman legion, military unit) — same: closest in-
#     taxonomy match is #topic-government-admin (military/government unit).
MANUAL_TOPIC_OVERRIDES: dict[str, str] = {
    "0360.00.01": "#topic-daily-routines",       # ocorrer = to occur (generic event verb)
    "1845.00.01": "#topic-city-places",          # fundação = founding (of a city)
    "3688.00.01": "#topic-social-problems",      # decadência = decline (of empire/society)
    "4692.00.01": "#topic-government-admin",     # legião = legion (military unit)
}

# Content POS values that go to the LLM.
LLM_POS: frozenset[str] = frozenset(("noun", "verb", "adj", "adv", "idiom", "interj"))


def deterministic_topic_for_row(sense_id: str, pos: str) -> str | None:
    """Return the deterministic topic tag for this row, honoring per-sense
    overrides. Returns None when the row requires LLM classification.

    Order:
      1. MANUAL_TOPIC_OVERRIDES (highest precedence — explicit user override)
      2. NUM_TOPIC_OVERRIDES   (sense-level fix on top of pos=num rule)
      3. POS_TOPIC_PRERULES    (the bulk rule for function words + numerals)
      4. LLM (everything else)
    """
    if sense_id in MANUAL_TOPIC_OVERRIDES:
        return MANUAL_TOPIC_OVERRIDES[sense_id]
    if sense_id in NUM_TOPIC_OVERRIDES:
        return NUM_TOPIC_OVERRIDES[sense_id]
    return POS_TOPIC_PRERULES.get(pos)


# Back-compat shim for callers that only have the POS.
def deterministic_topic(pos: str) -> str | None:
    """Return the deterministic topic tag for the given POS only.

    Does NOT consult NUM_TOPIC_OVERRIDES. Use `deterministic_topic_for_row`
    when you have the sense_id available.
    """
    return POS_TOPIC_PRERULES.get(pos)


# --------------------------------------------------------------------------- #
# Per-row post-process (idempotent)
# --------------------------------------------------------------------------- #


def apply_post_process(row: dict) -> dict:
    """Mutate `row` in place with the per-row deterministic passes.

    Expected keys (writes them, creates missing with defaults):
      topic_primary, confidence, reason

    Passes (all idempotent):
      1. Strip whitespace on `topic_primary`.
      2. Validate enum: if `topic_primary ∉ ALLOWED_TOPIC_TAGS`, clear it
         and tag `reason` with `"invalid_tag_rejected: <orig>"`. Defence-
         in-depth — the tool schema enum should already block this.
      3. Normalize prefix: must start with `#topic-`; otherwise reject.
      4. Confidence default: missing / empty → `"low"`.

    Returns the same dict for convenience.
    """
    row.setdefault("confidence", "low")
    row.setdefault("reason", "")

    raw = (row.get("topic_primary") or "").strip()

    if not raw:
        row["topic_primary"] = ""
        row["confidence"] = (row.get("confidence") or "low").strip().lower() or "low"
        return row

    if not raw.startswith("#topic-"):
        row["topic_primary"] = ""
        row["reason"] = (
            f"missing_topic_prefix: {raw!r}; " + (row.get("reason") or "")
        ).strip()
        row["confidence"] = "low"
        return row

    if raw not in ALLOWED_TOPIC_TAG_SET:
        row["topic_primary"] = ""
        row["reason"] = (
            f"invalid_tag_rejected: {raw!r}; " + (row.get("reason") or "")
        ).strip()
        row["confidence"] = "low"
        return row

    row["topic_primary"] = raw
    conf = (row.get("confidence") or "").strip().lower()
    row["confidence"] = conf if conf in {"high", "medium", "low"} else "low"
    return row
