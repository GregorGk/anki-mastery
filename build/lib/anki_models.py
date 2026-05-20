"""Stage 18 — genanki models, fields, deterministic IDs, note builder, tags.

Three models (one per card type), three decks, three notes per sense.
Separate notes (not one note with three templates) so the decks have no
sibling-burying and can be studied independently in the listening-first
order.

All IDs are HARD-CODED so rebuilds are byte-stable and re-imports update
existing notes instead of duplicating them. Note GUIDs are derived from a
stable seed `"<prefix>::<sense_id>"` via `genanki.guid_for`, so a sense's
identity never changes across rebuilds.

Imported by `build/18_1_build_apkg.py` and `tests/test_stage_18_apkg.py`.
"""
from __future__ import annotations

import genanki

from build.lib.anki_templates import CARD_CSS, TEMPLATES

# ── card types ───────────────────────────────────────────────────────────────
CARD_TYPES = ("listening", "recall")

# ── the 24 note fields (shared by all three models) ──────────────────────────
# `sense_id` first → it's the Anki sort field + dedup key within a note type.
# `rank` and `anki_tags` from the 25 Stage-17 export fields are intentionally
# dropped here (`anki_tags` → note tags; `rank` unused). `issue_hint` added.
NOTE_FIELDS = [
    "sense_id",
    "anki_order",
    "topic_primary",
    "pt",
    "pt_display_safe",
    "pos",
    "gender",
    "en_primary",
    "en_all",
    "annotation",
    "example_pt",
    "example_en",
    "ipa_word",
    "ipa_example",
    "audio_word_file",  # bare basename for click-only HTML5 <audio src>
    "audio_example",
    "audio_en_example",
    "usage_hint",
    "risk_note",
    "bp_validity",
    "register",
    "risk_flags",
    "family_root",
    "issue_hint",
]

ISSUE_HINT = "Flag red if audio, sentence, or meaning seems wrong."

# ── deterministic IDs (hard-coded; 1 << 30 < id < 1 << 31) ───────────────────
MODEL_IDS = {
    "listening": 1810000001,  # reuses the old "sentence" model id (idempotent)
    "recall":    1810000004,  # new
}
DECK_IDS = {
    "listening": 1810000101,  # reuses the old "sentence" deck id
    "recall":    1810000104,  # new
}
MODEL_NAMES = {
    "listening": "BP Sentence Listening",
    "recall":    "BP Sentence Recall",
}
DECK_NAMES = {
    "listening": "Brazilian Portuguese Mastery::01 Sentence Listening",
    "recall":    "Brazilian Portuguese Mastery::02 Sentence Recall",
}
GUID_PREFIX = {
    "listening": "bp-sentence-listen",
    "recall":    "bp-sentence-recall",
}
CARD_TAGS = {
    "listening": "card::sentence-listening",
    "recall":    "card::sentence-recall",
}
STAGE_TAGS = ["stage::18", "source::programmatic"]


class BPNote(genanki.Note):
    """Note with a deterministic GUID derived from a stable seed.

    Notes are added to each deck in ascending `anki_order`; new-card order
    follows insertion order on import (no `due`/scheduling metadata is set).
    `guid` is overridden (getter only — genanki's __init__ assigns
    `self._guid` directly, never via the setter, so a read-only override is
    safe) so re-imports update rather than duplicate.
    """

    def __init__(self, *, model, fields, tags, guid_seed):
        super().__init__(model=model, fields=fields, tags=tags)
        self._guid_seed = guid_seed

    @property
    def guid(self):
        return genanki.guid_for(self._guid_seed)


def build_models() -> dict[str, genanki.Model]:
    """One genanki.Model per card type, sharing NOTE_FIELDS + CARD_CSS."""
    models: dict[str, genanki.Model] = {}
    for ct in CARD_TYPES:
        tmpl = TEMPLATES[ct]
        models[ct] = genanki.Model(
            MODEL_IDS[ct],
            MODEL_NAMES[ct],
            fields=[{"name": f} for f in NOTE_FIELDS],
            templates=[{
                "name": tmpl["name"],
                "qfmt": tmpl["qfmt"],
                "afmt": tmpl["afmt"],
            }],
            css=CARD_CSS,
        )
    return models


def build_decks() -> dict[str, genanki.Deck]:
    """One genanki.Deck per card type (subdecks via the `::` names)."""
    return {ct: genanki.Deck(DECK_IDS[ct], DECK_NAMES[ct]) for ct in CARD_TYPES}


def guid_seed(card_type: str, sense_id: str) -> str:
    return f"{GUID_PREFIX[card_type]}::{sense_id}"


def tags_for(row: dict, card_type: str) -> list[str]:
    """Stage-17 `anki_tags` (space-joined) + card-type + stage/source tags.

    All tokens already match `^[a-z]+(::[a-z0-9-]+)+$`.
    """
    tags = [t for t in (row.get("anki_tags") or "").split() if t]
    tags.append(CARD_TAGS[card_type])
    tags.extend(STAGE_TAGS)
    return tags
