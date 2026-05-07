"""Source ledger and sense<->source many-to-many provenance map.

`_source_ledger.tsv` is the backbone: one row per source.txt line, mutated
across stages by changing `action` and adding stage-specific columns.

`_sense_source_map.tsv` is the authoritative many-to-many provenance:
edges from source_line_number → sense_id with provenance_type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .tsv import read_tsv, write_tsv

LEDGER_FIELDS = [
    "source_line_number",
    "rank",
    "source_raw",
    "source_pt",
    "source_en_all",
    "action",
    "normalization_action",
    "bp_replacement",
    "merge_target_rank",
    "merge_target_pt",
    "output_sense_ids",
    "drop_reason",
    "manual_review_status",
    "stage_decided",
    "notes",
]

SENSE_SOURCE_MAP_FIELDS = [
    "sense_id",
    "source_line_number",
    "provenance_type",
    "notes",
]

# Provenance types (per plan § Source ledger)
PROVENANCE_ORIGINAL = "original"
PROVENANCE_NORMALIZED = "normalized"
PROVENANCE_LEXICAL_REPLACEMENT = "lexical_replacement"
PROVENANCE_MERGED = "merged"
PROVENANCE_IDIOM_EXPANSION = "idiom_expansion"

# Action states (per plan § Stage 1c)
ACTION_KEEP = "keep"
ACTION_NORMALIZE_SPELLING = "normalize_spelling"
ACTION_REPLACE_WITH_BP = "replace_with_bp_equivalent"
ACTION_MERGE_INTO_BP_ROW = "merge_into_existing_bp_row"
ACTION_KEEP_TAG_RARE = "keep_tag_rare"
ACTION_MANUAL_REVIEW = "manual_review"
ACTION_DROP_EP_ONLY = "drop_ep_only"
ACTION_EXPAND_IDIOM_ADDED = "expand_idiom_added"


@dataclass
class LedgerRow:
    source_line_number: int
    rank: int
    source_raw: str
    source_pt: str = ""
    source_en_all: str = ""
    action: str = ACTION_KEEP
    normalization_action: str = "none"
    bp_replacement: str = ""
    merge_target_rank: str = ""
    merge_target_pt: str = ""
    output_sense_ids: str = ""
    drop_reason: str = ""
    manual_review_status: str = "not_required"
    stage_decided: str = "1a"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "source_line_number": self.source_line_number,
            "rank": self.rank,
            "source_raw": self.source_raw,
            "source_pt": self.source_pt,
            "source_en_all": self.source_en_all,
            "action": self.action,
            "normalization_action": self.normalization_action,
            "bp_replacement": self.bp_replacement,
            "merge_target_rank": self.merge_target_rank,
            "merge_target_pt": self.merge_target_pt,
            "output_sense_ids": self.output_sense_ids,
            "drop_reason": self.drop_reason,
            "manual_review_status": self.manual_review_status,
            "stage_decided": self.stage_decided,
            "notes": self.notes,
        }


@dataclass
class SenseSourceEdge:
    sense_id: str
    source_line_number: int
    provenance_type: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "sense_id": self.sense_id,
            "source_line_number": self.source_line_number,
            "provenance_type": self.provenance_type,
            "notes": self.notes,
        }


def write_ledger(path: str | Path, rows: Iterable[LedgerRow]) -> int:
    return write_tsv(
        path, (r.to_dict() for r in rows), fieldnames=LEDGER_FIELDS
    )


def read_ledger(path: str | Path) -> list[dict[str, str]]:
    return read_tsv(path)


def write_sense_source_map(
    path: str | Path, edges: Iterable[SenseSourceEdge]
) -> int:
    return write_tsv(
        path, (e.to_dict() for e in edges), fieldnames=SENSE_SOURCE_MAP_FIELDS
    )


def read_sense_source_map(path: str | Path) -> list[dict[str, str]]:
    return read_tsv(path)
