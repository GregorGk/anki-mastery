#!/usr/bin/env python
"""Surgical patch: apply _manual_sense_splits.tsv to Stage 2/3 outputs.

Reads:
    data/_manual_sense_splits.tsv     (function-word splits + reflexive sub-senses)
    data/_manual_gender.tsv           (pos/gender overrides for new sense_ids)
    data/02-senses.tsv                (Stage 2 output)
    data/03-enriched.tsv              (Stage 3 output)
    data/_sense_source_map.tsv        (provenance edges)
    data/_source_ledger.tsv           (output_sense_ids per source line)

Writes the same four files in place.

This script exists because re-running Stage 2 + Stage 3 from scratch costs
~$5 in API calls, but the manual overrides bypass the LLM entirely. It
applies the overrides directly to the existing artifacts.

Idempotent: re-running it produces the same output. No API costs.

Run:
    python3 build/patch_function_words.py

After running, re-run rederive_tags.py to refresh tags on the new senses.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.enrich import compose_pt_display, derive_tags  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OVERRIDES_PATH = DATA_DIR / "_manual_sense_splits.tsv"
GENDER_OVERRIDES_PATH = DATA_DIR / "_manual_gender.tsv"
SENSES_PATH = DATA_DIR / "02-senses.tsv"
ENRICHED_PATH = DATA_DIR / "03-enriched.tsv"
SENSE_MAP_PATH = DATA_DIR / "_sense_source_map.tsv"
LEDGER_PATH = DATA_DIR / "_source_ledger.tsv"


def _make_sense_id(rank: int, expansion_index: int, sense_index: int) -> str:
    return f"{rank:04d}.{expansion_index:02d}.{sense_index:02d}"


def _load_sense_overrides() -> dict[tuple[int, int], list[dict]]:
    """Map (source_line_number, expansion_index) -> list of sense override dicts."""
    rows = read_tsv(OVERRIDES_PATH)
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        ln = (r.get("source_line_number") or "").strip()
        if not ln:
            continue
        exp = int(r.get("expansion_index") or "0")
        sense_index = int(r.get("sense_index") or "0")
        en_primary = (r.get("en_primary") or "").strip()
        if not en_primary:
            continue
        grouped[(int(ln), exp)].append(
            {
                "sense_index": sense_index,
                "en_primary": en_primary,
                "gender": (r.get("gender") or "").strip(),
                "pt_display": (r.get("pt_display") or "").strip(),
                "annotation_json": (r.get("annotation_json") or "").strip(),
                "notes": (r.get("notes") or "").strip(),
            }
        )
    for k in grouped:
        grouped[k].sort(key=lambda s: s["sense_index"])
    return grouped


def _load_gender_overrides() -> dict[str, dict[str, str]]:
    """Map sense_id -> {gender, pos, is_cognate_en} dict."""
    rows = read_tsv(GENDER_OVERRIDES_PATH)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "gender": (r.get("gender") or "").strip(),
            "pos": (r.get("pos") or "").strip(),
            "is_cognate_en": (r.get("is_cognate_en") or "false").strip().lower(),
        }
    return out


def main() -> int:
    overrides = _load_sense_overrides()
    if not overrides:
        print("No overrides in _manual_sense_splits.tsv; nothing to do.")
        return 0

    gender_overrides = _load_gender_overrides()

    # --- Load existing artifacts ---------------------------------------------

    senses = read_tsv(SENSES_PATH)
    enriched = read_tsv(ENRICHED_PATH)
    sense_map = read_tsv(SENSE_MAP_PATH)
    ledger = read_tsv(LEDGER_PATH)

    senses_fields = list(senses[0].keys()) if senses else []
    enriched_fields = list(enriched[0].keys()) if enriched else []
    sense_map_fields = list(sense_map[0].keys()) if sense_map else []
    ledger_fields = list(ledger[0].keys()) if ledger else []

    # Index existing rows by (source_line_number, expansion_index)
    senses_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in senses:
        ln = int(r.get("source_line_number") or 0)
        exp = int(r.get("expansion_index") or 0)
        senses_by_key[(ln, exp)].append(r)

    enriched_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in enriched:
        ln = int(r.get("source_line_number") or 0)
        exp = int(r.get("expansion_index") or 0)
        enriched_by_key[(ln, exp)].append(r)

    # --- Apply each override -------------------------------------------------

    new_sense_ids_by_key: dict[tuple[int, int], list[str]] = {}
    summary_per_key: list[tuple[tuple[int, int], int, int]] = []

    for key, ov_senses in overrides.items():
        ln, exp = key
        existing_senses = senses_by_key.get(key, [])
        existing_enriched = enriched_by_key.get(key, [])

        if not existing_senses:
            print(
                f"[warn] override for ({ln}, {exp}) has no existing rows; skipping",
                file=sys.stderr,
            )
            continue

        # Use the first existing row as the template (carries pt, source_pt, source_line, etc.)
        template_sense = existing_senses[0]
        template_enriched = existing_enriched[0] if existing_enriched else None
        rank = int(template_sense["rank"])

        # Build new sense_ids list and replace existing rows
        new_sense_rows: list[dict] = []
        new_enriched_rows: list[dict] = []
        new_ids: list[str] = []

        for ov in ov_senses:
            sense_index = ov["sense_index"]
            sid = _make_sense_id(rank, exp, sense_index)
            new_ids.append(sid)

            # Pull gender/pos/is_cognate_en from gender overrides (mandatory for these
            # patches — the manual gender file should have been extended in tandem)
            gov = gender_overrides.get(sid, {})
            pos_override = gov.get("pos", "")
            gender_override = gov.get("gender") or ov.get("gender", "")
            cognate_str = gov.get("is_cognate_en", "false")
            cognate_bool = cognate_str in ("true", "1", "yes")

            # 02-senses.tsv row
            new_sense_row = dict(template_sense)
            new_sense_row.update(
                {
                    "sense_id": sid,
                    "sense_index": sense_index,
                    "gender": gender_override,
                    "pos": pos_override or template_sense.get("pos", ""),
                    "en_primary": ov["en_primary"],
                    "annotation": ov.get("annotation_json", ""),
                    "split_method": "manual_override",
                    "split_reason": ov.get("notes", "manual override"),
                    "split_confidence": "high",
                    "manual_review_required": "no",
                }
            )
            new_sense_rows.append(new_sense_row)

            # 03-enriched.tsv row
            new_pt_display = compose_pt_display(
                template_sense.get("pt", ""),
                gender_override,
                template_sense.get("pt_type", ""),
            )

            new_enriched_row = dict(template_enriched or template_sense)
            new_enriched_row.update(
                {
                    "sense_id": sid,
                    "sense_index": sense_index,
                    "gender": gender_override,
                    "pos": pos_override or new_enriched_row.get("pos", ""),
                    "en_primary": ov["en_primary"],
                    "annotation": ov.get("annotation_json", ""),
                    "split_method": "manual_override",
                    "split_confidence": "high",
                    "is_cognate_en": "true" if cognate_bool else "false",
                    "enrich_method": "manual_override",
                    "enrich_confidence": "high",
                    "enrich_reason": "manual override (function-word patch)",
                    "pt_display": new_pt_display,
                }
            )
            new_enriched_row["tags"] = derive_tags(
                new_enriched_row,
                gender=gender_override,
                pos=pos_override or new_enriched_row.get("pos", ""),
                cognate_en=cognate_bool,
            )
            new_enriched_rows.append(new_enriched_row)

        # Replace senses for this key
        senses_by_key[key] = new_sense_rows
        enriched_by_key[key] = new_enriched_rows
        new_sense_ids_by_key[key] = new_ids

        summary_per_key.append((key, len(existing_senses), len(new_ids)))

    # --- Sweep all enriched rows for _manual_gender.tsv overrides ----------
    # This catches existing rows whose pos/gender differs from the override
    # (e.g., 0001.00.02 = "him" was pos=art from deterministic shortcut; should
    # be pron). Idempotent — re-running produces same output.

    sweep_changed = 0
    for key, rows_for_key in enriched_by_key.items():
        for r in rows_for_key:
            sid = r.get("sense_id", "")
            gov = gender_overrides.get(sid)
            if not gov:
                continue
            new_gender = gov.get("gender", "")
            new_pos = gov.get("pos", "") or r.get("pos", "")
            cognate_str = gov.get("is_cognate_en", "")
            cognate_bool = cognate_str in ("true", "1", "yes")
            old_gender = r.get("gender", "")
            old_pos = r.get("pos", "")
            old_cognate = (r.get("is_cognate_en", "") or "").lower() in ("true", "1", "yes")
            if (
                new_gender != old_gender
                or new_pos != old_pos
                or (cognate_str and cognate_bool != old_cognate)
            ):
                r["gender"] = new_gender
                r["pos"] = new_pos
                if cognate_str:
                    r["is_cognate_en"] = "true" if cognate_bool else "false"
                r["enrich_method"] = "manual_override"
                r["pt_display"] = compose_pt_display(
                    r.get("pt", ""),
                    new_gender,
                    r.get("pt_type", ""),
                )
                r["tags"] = derive_tags(
                    r,
                    gender=new_gender,
                    pos=new_pos,
                    cognate_en=cognate_bool if cognate_str else old_cognate,
                )
                sweep_changed += 1

    # Mirror gender/pos changes back into 02-senses.tsv (which doesn't carry
    # tags/cognate but has gender/pos columns)
    for key, rows_for_key in senses_by_key.items():
        for r in rows_for_key:
            sid = r.get("sense_id", "")
            gov = gender_overrides.get(sid)
            if not gov:
                continue
            new_gender = gov.get("gender", "")
            new_pos = gov.get("pos", "") or r.get("pos", "")
            if new_gender != r.get("gender", "") or new_pos != r.get("pos", ""):
                r["gender"] = new_gender
                r["pos"] = new_pos

    # --- Rebuild flat lists -------------------------------------------------

    # Original ordering: by (rank, expansion_index, sense_index). Re-sort.
    flat_senses: list[dict] = []
    for key in sorted(senses_by_key.keys()):
        flat_senses.extend(senses_by_key[key])
    flat_senses.sort(
        key=lambda r: (
            int(r["rank"]),
            int(r.get("expansion_index") or 0),
            int(r.get("sense_index") or 0),
        )
    )

    flat_enriched: list[dict] = []
    for key in sorted(enriched_by_key.keys()):
        flat_enriched.extend(enriched_by_key[key])
    flat_enriched.sort(
        key=lambda r: (
            int(r["rank"]),
            int(r.get("expansion_index") or 0),
            int(r.get("sense_index") or 0),
        )
    )

    # --- Update sense_source_map -------------------------------------------

    # For each (source_line_number, expansion_index) we patched, remove old
    # edges with provenance_type ∈ {original, idiom_expansion} and add new ones.
    affected_lines = {ln for (ln, _exp) in overrides.keys()}

    new_sense_map: list[dict] = []
    for r in sense_map:
        sln = int(r.get("source_line_number") or 0)
        ptype = (r.get("provenance_type") or "").strip()
        if sln in affected_lines and ptype in ("original", "idiom_expansion"):
            continue  # to be replaced
        new_sense_map.append(r)

    for key, new_ids in new_sense_ids_by_key.items():
        ln, exp = key
        provenance = "idiom_expansion" if exp > 0 else "original"
        for sid in new_ids:
            new_sense_map.append(
                {
                    "sense_id": sid,
                    "source_line_number": ln,
                    "provenance_type": provenance,
                    "notes": "split_method=manual_override",
                }
            )

    new_sense_map.sort(
        key=lambda e: (e.get("sense_id", ""), int(e.get("source_line_number") or 0))
    )

    # --- Update ledger output_sense_ids ------------------------------------

    # For each affected source_line, set output_sense_ids = comma-joined new sense_ids
    # (collected across all expansion_indexes for that line)
    new_ids_by_line: dict[int, list[str]] = defaultdict(list)
    # First, gather ALL sense_ids for affected lines (existing + new). The manual
    # patch only touched some (source_line_number, expansion_index) keys, but a
    # source line may have other expansion_indexes that should also remain in
    # output_sense_ids.
    all_senses_by_line: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for r in flat_senses:
        ln = int(r["source_line_number"])
        exp = int(r.get("expansion_index") or 0)
        si = int(r.get("sense_index") or 0)
        all_senses_by_line[ln].append((exp, si, r["sense_id"]))

    for ln in affected_lines:
        senses_for_line = sorted(all_senses_by_line.get(ln, []))
        new_ids_by_line[ln] = [sid for (_e, _s, sid) in senses_for_line]

    for r in ledger:
        ln = int(r.get("source_line_number") or 0)
        if ln in new_ids_by_line:
            r["output_sense_ids"] = ",".join(new_ids_by_line[ln])

    # --- Write back --------------------------------------------------------

    n_senses = write_tsv(SENSES_PATH, flat_senses, fieldnames=senses_fields)
    n_enriched = write_tsv(ENRICHED_PATH, flat_enriched, fieldnames=enriched_fields)
    n_map = write_tsv(SENSE_MAP_PATH, new_sense_map, fieldnames=sense_map_fields)
    n_ledger = write_tsv(LEDGER_PATH, ledger, fieldnames=ledger_fields)

    # --- Summary -----------------------------------------------------------

    print(f"\nApplied {len(overrides)} sense-split overrides:")
    for (key, before, after) in summary_per_key:
        print(f"  source_line {key[0]:>4d} exp={key[1]}: {before} sense → {after} sense(s)")
    print(f"\nApplied {sweep_changed} _manual_gender.tsv pos/gender overrides to existing rows.")
    print(f"\nWrote:")
    print(f"  {str(SENSES_PATH.relative_to(REPO_ROOT)):40s} {n_senses:5d} rows")
    print(f"  {str(ENRICHED_PATH.relative_to(REPO_ROOT)):40s} {n_enriched:5d} rows")
    print(f"  {str(SENSE_MAP_PATH.relative_to(REPO_ROOT)):40s} {n_map:5d} rows")
    print(f"  {str(LEDGER_PATH.relative_to(REPO_ROOT)):40s} {n_ledger:5d} rows")

    # --- Verify invariants -------------------------------------------------

    seen_ids: set[str] = set()
    for r in flat_senses:
        sid = r["sense_id"]
        if sid in seen_ids:
            print(f"\nERROR: duplicate sense_id in 02-senses.tsv: {sid}")
            return 1
        seen_ids.add(sid)
        assert r["en_primary"], f"empty en_primary at {sid}"

    seen_ids.clear()
    for r in flat_enriched:
        sid = r["sense_id"]
        if sid in seen_ids:
            print(f"\nERROR: duplicate sense_id in 03-enriched.tsv: {sid}")
            return 1
        seen_ids.add(sid)

    print("\nAll invariants pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
