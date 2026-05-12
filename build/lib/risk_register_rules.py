"""Stage 15 — BP validity / register / risk_flags classifier rules.

Single source of truth for:
  - ALLOWED_BP_VALIDITY      (7 enum values)
  - ALLOWED_REGISTER         (10 enum values)
  - ALLOWED_RISK_FLAGS       (20 enum values incl. "none")
  - MANUAL_REVIEW_SEED_LEMMAS (lemma-level seed list — feeds the candidate
                               filter, never the final values)
  - CANDIDATE_TAG_PATTERNS   (which tag substrings flag a row as a candidate)
  - CANDIDATE_USAGE_HINT_TRIGGERS (case-insensitive phrases in Stage-12
                                    usage_hint that flag a row)
  - is_candidate(row)        (the deterministic filter)
  - apply_post_process(row)  (idempotent enum / pipe-list / risk_note hygiene)

Imported by both the classifier and the verifier so the enums are single-
sourced.
"""
from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Enums (locked from plan)
# --------------------------------------------------------------------------- #

ALLOWED_BP_VALIDITY: tuple[str, ...] = (
    "standard",
    "rare_in_bp",
    "ep_leaning",
    "ep_only",
    "regional_br",
    "nonstandard",
    "uncertain",
)
ALLOWED_BP_VALIDITY_SET: frozenset[str] = frozenset(ALLOWED_BP_VALIDITY)

ALLOWED_REGISTER: tuple[str, ...] = (
    "neutral",
    "informal",
    "formal",
    "technical",
    "literary",
    "archaic",
    "slang",
    "vulgar",
    "taboo",
    "uncertain",
)
ALLOWED_REGISTER_SET: frozenset[str] = frozenset(ALLOWED_REGISTER)

ALLOWED_RISK_FLAGS: tuple[str, ...] = (
    "none",
    "false_friend",
    "ep_misleading",
    "regional_misuse",
    "vulgar",
    "sexual",
    "offensive",
    "slur",
    "racial_sensitive",
    "gender_sensitive",
    "outdated",
    "childish",
    "profanity",
    "violence",
    "drug_related",
    "religious_sensitive",
    "political_sensitive",
    "medical_sensitive",
    "legal_sensitive",
    "ambiguous_translation",
)
ALLOWED_RISK_FLAGS_SET: frozenset[str] = frozenset(ALLOWED_RISK_FLAGS)


# Defaults (what the deterministic path writes for non-candidate rows).
DEFAULT_BP_VALIDITY = "standard"
DEFAULT_REGISTER = "neutral"
DEFAULT_RISK_FLAGS = "none"

# Hard limits enforced by the verifier and the post-process.
RISK_NOTE_MAX_CHARS = 180


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #

# Lemmas surfaced in the user's Stage 15 spec as known-risky. Membership
# only adds the row to the LLM candidate set — final values still come
# from data/_manual_risk_register.tsv (sense_id keyed). A lemma can have
# one risky sense and one safe sense; we don't want lemma-level overrides
# in production.
MANUAL_REVIEW_SEED_LEMMAS: frozenset[str] = frozenset((
    "gozar", "rapariga", "mulato",
    "eventualmente", "demitir",
    "camisola", "gelado",
    # Common false-friend / sensitive lemmas pre-emptively included:
    "pretender", "assistir", "atender", "fazenda",
))

# Tag substrings that flag a row as a candidate. Substring match against
# whitespace-split `tags`. (Stage 14 added many #topic-* tags that are NOT
# in this list — those don't flag.)
CANDIDATE_TAG_PATTERNS: tuple[str, ...] = (
    "#false-friend",
    "#nsfw",
    "#bp-rare",
    "#regional",
    "#archaic",
    "#sensitive-reviewed",
    "#vulgar",
    "#slang",
)

# Case-insensitive substrings of Stage-12 usage_hint that flag the row.
CANDIDATE_USAGE_HINT_TRIGGERS: tuple[str, ...] = (
    "false friend",
    "avoid",
    "warning",
    "bp warning",
    "ep ",
)

CANDIDATE_BP_STATUS_VALUES: frozenset[str] = frozenset((
    "uncommon", "false_friend", "nsfw",
))


def is_candidate(
    row: dict,
    *,
    usage_hint: str = "",
    manual_override_sense_ids: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Return True if this row should be sent to the LLM (Stage 15).

    A row is a candidate unless it's already manually overridden AND has at
    least one risk signal. Manual overrides win (no LLM call needed); pure
    deterministic-default rows are skipped.

    Signals (any one is enough):
      - bp_status ∈ {uncommon, false_friend, nsfw}
      - non-empty risk_note (Stage 12)
      - non-empty annotation (col 12)
      - any tag matches CANDIDATE_TAG_PATTERNS
      - usage_hint contains any CANDIDATE_USAGE_HINT_TRIGGERS
      - pt is in MANUAL_REVIEW_SEED_LEMMAS
    """
    sense_id = (row.get("sense_id") or "").strip()
    if sense_id in manual_override_sense_ids:
        return False
    bp_status = (row.get("bp_status") or "").strip()
    if bp_status in CANDIDATE_BP_STATUS_VALUES:
        return True
    if (row.get("risk_note") or "").strip():
        return True
    if (row.get("annotation") or "").strip():
        return True
    tags = (row.get("tags") or "")
    if any(p in tags for p in CANDIDATE_TAG_PATTERNS):
        return True
    if usage_hint:
        lower = usage_hint.lower()
        if any(t in lower for t in CANDIDATE_USAGE_HINT_TRIGGERS):
            return True
    pt = (row.get("pt") or "").strip()
    if pt in MANUAL_REVIEW_SEED_LEMMAS:
        return True
    return False


# --------------------------------------------------------------------------- #
# Post-process (idempotent)
# --------------------------------------------------------------------------- #


_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_WHITESPACE_NORMALIZE_RE = re.compile(r"[\t\n\r]+")


def _normalize_risk_flags(raw: str) -> tuple[str, list[str]]:
    """Normalize risk_flags to a canonical pipe-string.

    Returns (normalized_string, dropped_flags). Rules:
      - Empty / "none" → "none".
      - Drop empty tokens after split.
      - Drop any flag not in ALLOWED_RISK_FLAGS (return them via dropped).
      - If "none" appears with other flags, drop "none".
      - Dedupe while preserving order.
      - If everything was dropped, return "none".
    """
    if not raw or raw.strip() in {"", "none"}:
        return "none", []
    parts = [p.strip() for p in _PIPE_SPLIT_RE.split(raw) if p.strip()]
    seen: set[str] = set()
    keep: list[str] = []
    dropped: list[str] = []
    for p in parts:
        if p == "none":
            # absorbed below if other flags exist
            continue
        if p not in ALLOWED_RISK_FLAGS_SET:
            dropped.append(p)
            continue
        if p in seen:
            continue
        seen.add(p)
        keep.append(p)
    if not keep:
        return "none", dropped
    return "|".join(keep), dropped


def apply_post_process(row: dict) -> dict:
    """Mutate `row` in place with the per-row deterministic hygiene passes.

    Writes / normalizes these keys (creates with defaults if missing):
      bp_validity, register, risk_flags, risk_note

    Returns the same dict for convenience.
    """
    # 1. bp_validity — strip + enum check; fallback to "uncertain"
    bv = (row.get("bp_validity") or "").strip()
    if bv not in ALLOWED_BP_VALIDITY_SET:
        row.setdefault("reason", "")
        row["reason"] = (
            f"invalid_bp_validity={bv!r}; " + (row.get("reason") or "")
        ).strip()
        bv = "uncertain"
    row["bp_validity"] = bv

    # 2. register — same shape
    reg = (row.get("register") or "").strip()
    if reg not in ALLOWED_REGISTER_SET:
        row.setdefault("reason", "")
        row["reason"] = (
            f"invalid_register={reg!r}; " + (row.get("reason") or "")
        ).strip()
        reg = "uncertain"
    row["register"] = reg

    # 3. risk_flags — normalize pipe list, drop invalid, enforce "none" rule
    flags_raw = (row.get("risk_flags") or "").strip()
    flags_norm, dropped = _normalize_risk_flags(flags_raw)
    if dropped:
        row.setdefault("reason", "")
        row["reason"] = (
            f"dropped_invalid_flags={dropped}; " + (row.get("reason") or "")
        ).strip()
    row["risk_flags"] = flags_norm

    # 4. risk_note — strip whitespace artifacts, length cap
    note = (row.get("risk_note") or "").strip()
    note = _WHITESPACE_NORMALIZE_RE.sub(" ", note)
    if len(note) > RISK_NOTE_MAX_CHARS:
        note = note[:RISK_NOTE_MAX_CHARS].rstrip()
    row["risk_note"] = note

    # 5. confidence default
    conf = (row.get("confidence") or "").strip().lower()
    if conf not in {"high", "medium", "low"}:
        conf = "low" if (bv == "uncertain" or reg == "uncertain") else "high"
    row["confidence"] = conf

    return row
