"""Stage 2 — Sense split.

Reads:
    data/018-deduped.tsv                  (Stage 1.8 output; 4972 active rows)
    data/_manual_sense_splits.tsv         (manual overrides; wins over LLM)
    data/_source_ledger.tsv               (mutated in place: output_sense_ids)
    data/_sense_source_map.tsv            (rewritten with real sense_ids)

Writes:
    data/02-senses.tsv                    (one row per sense)
    data/_source_ledger.tsv               (output_sense_ids populated)
    data/_sense_source_map.tsv            (placeholder RRRR.EE.00 → real RRRR.EE.SS)
    audit/02_sense_split.jsonl            (LLM provenance)

Behaviour:
    1. Categorize each row via lib/sense_split.py (deterministic).
    2. Single-sense rows (no slash) skip the LLM entirely; en_primary = en_all.
    3. Forced gender splits (8 entries with M/F markers) → premium tier LLM.
    4. Function-word polysemy (top ~30 grammar words) → premium tier LLM.
    5. Idiom-expansion rows → default tier LLM (compute en_primary for the phrase).
    6. Lexical polysemy candidates → default tier LLM.
    7. Manual overrides (in _manual_sense_splits.tsv) override LLM output.
    8. Assign sense_id = {RRRR.EE.SS} per the locked format.
    9. Update _sense_source_map.tsv to use real sense_ids (was placeholder
       RRRR.EE.00 from Stage 1a; now RRRR.EE.SS for each emitted sense).
   10. Update ledger output_sense_ids with comma-joined sense_ids.

Hard invariants (per § Verification):
    - sense_id uniqueness
    - every emitted row has a non-empty en_primary
    - every kept ledger row has output_sense_ids populated
    - {RRRR.EE.SS} format respected
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.ledger import (  # noqa: E402
    LEDGER_FIELDS,
    SENSE_SOURCE_MAP_FIELDS,
    LedgerRow,
    SenseSourceEdge,
    write_ledger,
    write_sense_source_map,
)
from build.lib.llm import TIER_DEFAULT, TIER_PREMIUM, AnthropicClient  # noqa: E402
from build.lib.sense_split import (  # noqa: E402
    CAT_FORCED_GENDER_SPLIT,
    CAT_FUNCTION_WORD_POLYSEMY,
    CAT_IDIOM_EXPANSION,
    CAT_LEXICAL_POLYSEMY,
    CAT_SINGLE_SENSE_NO_SPLIT,
    Categorization,
    categorize,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "018-deduped.tsv"
LEDGER_PATH = DATA_DIR / "_source_ledger.tsv"
SENSE_MAP_PATH = DATA_DIR / "_sense_source_map.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_sense_splits.tsv"
OUTPUT_PATH = DATA_DIR / "02-senses.tsv"
AUDIT_PATH = AUDIT_DIR / "02_sense_split.jsonl"

OUTPUT_FIELDS = [
    "sense_id",
    "source_line_number",
    "rank",
    "expansion_index",
    "sense_index",
    "source_pt",
    "pt",
    "pt_type",
    "gender",
    "pos",
    "en_primary",
    "en_all",
    "annotation",
    "normalization_action",
    "bp_status",
    "bp_status_confidence",
    "split_category",
    "split_confidence",
    "split_method",  # 'llm' | 'deterministic' | 'manual_override'
    "split_reason",
    "manual_review_required",
    "source_line",
]

DEFAULT_CONCURRENCY = 16

# Tool Use schema for sense splitter
SENSE_SPLIT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "is_polysemy": {
            "type": "boolean",
            "description": "True if senses array contains more than one element",
        },
        "senses": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,  # safety cap; very few rows need >5 senses
            "items": {
                "type": "object",
                "properties": {
                    "en_primary": {
                        "type": "string",
                        "description": "Single English phrase for this sense; no qualifiers",
                    },
                    "gender": {
                        "type": "string",
                        "enum": ["o", "a", "o/a", ""],
                        "description": "Optional grammatical gender; required for forced_gender_split",
                    },
                    "annotation": {
                        "type": "string",
                        "description": "Optional notes (M/F marker, register, etc.)",
                    },
                },
                "required": ["en_primary"],
            },
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "string",
            "description": "Required if confidence != high",
        },
    },
    "required": ["is_polysemy", "senses", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "sense_split.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Split a Brazilian-Portuguese headword into senses based on whether "
        "different example sentences would be needed. Use Tool Use."
    )


# --- Manual overrides --------------------------------------------------------


def _load_manual_overrides(path: Path) -> dict[tuple[int, int], list[dict]]:
    """Map (source_line_number, expansion_index) -> list of sense dicts."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        ln = (r.get("source_line_number") or "").strip()
        if not ln:
            continue
        exp = int((r.get("expansion_index") or "0").strip() or "0")
        sense_index = int((r.get("sense_index") or "0").strip() or "0")
        en_primary = (r.get("en_primary") or "").strip()
        if not en_primary:
            continue
        grouped[(int(ln), exp)].append(
            {
                "sense_index": sense_index,
                "en_primary": en_primary,
                "gender": (r.get("gender") or "").strip(),
                "pt_display": (r.get("pt_display") or "").strip(),
                "annotation": (r.get("annotation_json") or "").strip(),
            }
        )
    # Sort each group by sense_index
    for k in grouped:
        grouped[k].sort(key=lambda s: s["sense_index"])
    return grouped


# --- LLM call ----------------------------------------------------------------


@dataclass
class SplitResult:
    is_polysemy: bool
    senses: list[dict]
    confidence: str
    reason: str
    method: str  # 'llm' | 'deterministic' | 'manual_override' | 'fallback'


def _build_user_message(row: dict, cat: Categorization) -> str:
    return (
        f"category: {cat.category}\n"
        f"pt: {row.get('pt', '')}\n"
        f"source_pt: {row.get('source_pt', '')}\n"
        f"expansion_index: {row.get('expansion_index', 0)}\n"
        f"en_all: {row.get('en_all', '')}\n"
        f"bp_status: {row.get('bp_status', 'standard')}\n"
    )


def _call_llm(
    row: dict,
    cat: Categorization,
    *,
    client: AnthropicClient,
    system_prompt: str,
) -> SplitResult:
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=_build_user_message(row, cat),
            tool_name="split_senses",
            tool_description=(
                "Split a Brazilian-Portuguese headword into distinct senses. "
                "Two items are distinct senses iff different example sentences "
                "would be required."
            ),
            tool_input_schema=SENSE_SPLIT_TOOL_SCHEMA,
            stage="2",
            provenance_key=f"line_{row.get('source_line_number')}_exp{row.get('expansion_index')}",
            tier=cat.llm_tier or TIER_DEFAULT,
        )
    except Exception as exc:
        # Fall back to single sense from the en_all so the pipeline doesn't stall.
        print(
            f"[warn] line {row.get('source_line_number')} exp={row.get('expansion_index')}: "
            f"LLM error {type(exc).__name__}: {exc}; defaulting to single sense",
            file=sys.stderr,
        )
        return SplitResult(
            is_polysemy=False,
            senses=[{"en_primary": (row.get("en_all") or "").strip()[:200], "gender": "", "annotation": ""}],
            confidence="low",
            reason=f"LLM error fallback: {type(exc).__name__}",
            method="fallback",
        )

    senses = decision.get("senses") or []
    # Defensive: ensure at least one sense
    if not senses:
        senses = [
            {"en_primary": (row.get("en_all") or "").strip()[:200], "gender": "", "annotation": ""}
        ]
    # Coerce sense entries to dicts with defaults
    cleaned: list[dict] = []
    for s in senses:
        if isinstance(s, dict):
            cleaned.append(
                {
                    "en_primary": (s.get("en_primary") or "").strip(),
                    "gender": (s.get("gender") or "").strip(),
                    "annotation": (s.get("annotation") or "").strip(),
                }
            )
    if not cleaned:
        cleaned = [{"en_primary": (row.get("en_all") or "").strip()[:200], "gender": "", "annotation": ""}]
    return SplitResult(
        is_polysemy=bool(decision.get("is_polysemy", len(cleaned) > 1)),
        senses=cleaned,
        confidence=decision.get("confidence", "medium"),
        reason=decision.get("reason", ""),
        method="llm",
    )


# --- sense_id assignment ----------------------------------------------------


def _make_sense_id(rank: int, expansion_index: int, sense_index: int) -> str:
    return f"{rank:04d}.{expansion_index:02d}.{sense_index:02d}"


# --- Main pipeline ----------------------------------------------------------


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    ledger_path: Path = LEDGER_PATH,
    sense_map_path: Path = SENSE_MAP_PATH,
    output_path: Path = OUTPUT_PATH,
    audit_path: Path = AUDIT_PATH,
    concurrency: int = DEFAULT_CONCURRENCY,
    update_ledger: bool = True,
    update_sense_map: bool = True,
    limit: int | None = None,
    client: AnthropicClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 1.8 first."
        )

    rows = read_tsv(input_path)
    if limit is not None:
        rows = rows[:limit]

    overrides = _load_manual_overrides(overrides_path)
    system_prompt = _load_prompt()

    # Categorize all rows up front (deterministic)
    categorizations: list[Categorization] = [categorize(r) for r in rows]

    # Initialize results map
    results: dict[int, SplitResult] = {}
    deterministic_count = 0
    manual_override_count = 0

    # Apply manual overrides first
    for i, row in enumerate(rows):
        key = (int(row["source_line_number"]), int(row.get("expansion_index") or 0))
        if key in overrides:
            ov_senses = overrides[key]
            results[i] = SplitResult(
                is_polysemy=len(ov_senses) > 1,
                senses=[
                    {
                        "en_primary": s["en_primary"],
                        "gender": s.get("gender", ""),
                        "annotation": s.get("annotation", ""),
                    }
                    for s in ov_senses
                ],
                confidence="high",
                reason="manual override",
                method="manual_override",
            )
            manual_override_count += 1

    # Deterministic single-sense rows: no LLM
    for i, (row, cat) in enumerate(zip(rows, categorizations)):
        if i in results:
            continue
        if cat.category == CAT_SINGLE_SENSE_NO_SPLIT:
            en_primary = (row.get("en_all") or "").strip()
            results[i] = SplitResult(
                is_polysemy=False,
                senses=[{"en_primary": en_primary, "gender": "", "annotation": ""}],
                confidence="high",
                reason="no slash in en_all",
                method="deterministic",
            )
            deterministic_count += 1

    # LLM-eligible rows
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Stage 2 requires an Anthropic API key."
            )
        client = AnthropicClient(audit_path=audit_path)

    llm_indices = [
        i
        for i, (row, cat) in enumerate(zip(rows, categorizations))
        if i not in results and cat.needs_llm
    ]
    print(
        f"[info] Stage 2: input={len(rows)} rows; "
        f"deterministic={deterministic_count}, manual_override={manual_override_count}, "
        f"to LLM={len(llm_indices)}",
        file=sys.stderr,
    )

    # Concurrent LLM calls
    if llm_indices:
        completed = 0
        progress_step = max(50, len(llm_indices) // 20)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    _call_llm,
                    rows[i],
                    categorizations[i],
                    client=client,
                    system_prompt=system_prompt,
                ): i
                for i in llm_indices
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                results[i] = fut.result()
                completed += 1
                if completed % progress_step == 0:
                    print(
                        f"[info] Stage 2: {completed}/{len(llm_indices)} LLM calls done",
                        file=sys.stderr,
                    )

    # Build output rows + sense_source_map edges
    output_rows: list[dict] = []
    new_edges: list[SenseSourceEdge] = []
    counts_by_category: dict[str, int] = defaultdict(int)
    counts_by_method: dict[str, int] = defaultdict(int)
    polysemy_rows = 0
    sense_id_per_input: dict[int, list[str]] = defaultdict(list)

    for i, row in enumerate(rows):
        cat = categorizations[i]
        result = results[i]
        counts_by_category[cat.category] += 1
        counts_by_method[result.method] += 1
        if result.is_polysemy:
            polysemy_rows += 1

        rank = int(row["rank"])
        exp = int(row.get("expansion_index") or 0)

        for sense_index, sense in enumerate(result.senses, start=1):
            sense_id = _make_sense_id(rank, exp, sense_index)
            out = {
                "sense_id": sense_id,
                "source_line_number": row["source_line_number"],
                "rank": row["rank"],
                "expansion_index": exp,
                "sense_index": sense_index,
                "source_pt": row.get("source_pt", ""),
                "pt": row.get("pt", ""),
                "pt_type": row.get("pt_type", ""),
                "gender": sense.get("gender", "") or row.get("gender", ""),
                "pos": row.get("pos", ""),
                "en_primary": sense["en_primary"],
                "en_all": row.get("en_all", ""),
                "annotation": sense.get("annotation", ""),
                "normalization_action": row.get("normalization_action", ""),
                "bp_status": row.get("bp_status", ""),
                "bp_status_confidence": row.get("bp_status_confidence", ""),
                "split_category": cat.category,
                "split_confidence": result.confidence,
                "split_method": result.method,
                "split_reason": result.reason,
                "manual_review_required": "yes" if result.confidence == "low" else "no",
                "source_line": row.get("source_line", ""),
            }
            output_rows.append(out)
            sense_id_per_input[i].append(sense_id)

            # New provenance edge: maps sense_id back to source_line_number.
            # The provenance_type is determined by the source row's history;
            # for Stage 2 we mark all as "original" or "idiom_expansion" based
            # on expansion_index. (Lexical_replacement and merged edges were
            # added in Stages 1c; we keep those untouched.)
            edge_provenance = (
                "idiom_expansion" if exp > 0 else "original"
            )
            new_edges.append(
                SenseSourceEdge(
                    sense_id=sense_id,
                    source_line_number=int(row["source_line_number"]),
                    provenance_type=edge_provenance,
                    notes=f"split_method={result.method}",
                )
            )

    # Write outputs
    out_count = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)

    # Update _sense_source_map.tsv: replace placeholder original/idiom edges
    # with real sense_ids; preserve lexical_replacement/merged edges from Stage 1c.
    if update_sense_map:
        existing_edges = read_tsv(sense_map_path)
        # Keep only the non-placeholder edges (lexical_replacement, merged).
        # Placeholder edges are those with sense_id ending in ".00" or with
        # provenance_type ∈ {original, idiom_expansion}.
        preserved: list[SenseSourceEdge] = []
        for e in existing_edges:
            ptype = e.get("provenance_type", "")
            if ptype in {"original", "idiom_expansion"}:
                continue  # replaced by Stage 2's real sense_ids below
            preserved.append(
                SenseSourceEdge(
                    sense_id=e.get("sense_id", ""),
                    source_line_number=int(e.get("source_line_number", 0)),
                    provenance_type=ptype,
                    notes=e.get("notes", ""),
                )
            )
        # Combine preserved + new
        all_edges = preserved + new_edges
        # Sort for deterministic output: by sense_id then source_line_number
        all_edges.sort(key=lambda e: (e.sense_id, e.source_line_number))
        edge_count = write_sense_source_map(sense_map_path, all_edges)
    else:
        edge_count = 0

    # Update ledger output_sense_ids
    if update_ledger:
        ledger_rows = read_tsv(ledger_path)
        # Build ledger key index: (source_line_number, expansion_index from enumeration)
        from build.stage_1b import _enumerate_expansion_indices

        ledger_exp = _enumerate_expansion_indices(ledger_rows)
        ledger_by_key: dict[tuple[int, int], int] = {
            (int(r["source_line_number"]), ledger_exp[idx]): idx
            for idx, r in enumerate(ledger_rows)
        }

        # For each input row, map (source_line_number, expansion_index) → list of sense_ids
        for i, row in enumerate(rows):
            key = (int(row["source_line_number"]), int(row.get("expansion_index") or 0))
            ledger_idx = ledger_by_key.get(key)
            if ledger_idx is None:
                continue
            ledger_rows[ledger_idx]["output_sense_ids"] = ",".join(
                sense_id_per_input[i]
            )

        # Coerce and rewrite ledger
        coerced = [_dict_to_ledger(r) for r in ledger_rows]
        write_ledger(ledger_path, coerced)

    # --- Hard invariants ---
    seen_ids: set[str] = set()
    for r in output_rows:
        sid = r["sense_id"]
        assert sid not in seen_ids, f"duplicate sense_id: {sid}"
        seen_ids.add(sid)
        assert r["en_primary"], f"empty en_primary at {sid}"
        # Verify sense_id format
        parts = sid.split(".")
        assert len(parts) == 3, f"bad sense_id format: {sid!r}"
        assert all(len(p) >= 2 and p.isdigit() for p in parts), f"bad sense_id format: {sid!r}"

    summary = {
        "input_rows": len(rows),
        "output_senses": out_count,
        "polysemy_input_rows": polysemy_rows,
        "manual_overrides_used": manual_override_count,
        "deterministic_decisions": deterministic_count,
        "llm_calls": counts_by_method.get("llm", 0),
        "fallback_calls": counts_by_method.get("fallback", 0),
        "categories": dict(counts_by_category),
        "methods": dict(counts_by_method),
        "sense_source_edges": edge_count,
    }
    if client is not None:
        summary["cache_stats"] = client.cache_stats
    return summary


def _dict_to_ledger(r: dict) -> LedgerRow:
    return LedgerRow(
        source_line_number=int(r["source_line_number"]),
        rank=int(r["rank"]),
        source_raw=r.get("source_raw", ""),
        source_pt=r.get("source_pt", ""),
        source_en_all=r.get("source_en_all", ""),
        action=r.get("action", "keep"),
        normalization_action=r.get("normalization_action", "none"),
        bp_replacement=r.get("bp_replacement", ""),
        merge_target_rank=r.get("merge_target_rank", ""),
        merge_target_pt=r.get("merge_target_pt", ""),
        output_sense_ids=r.get("output_sense_ids", ""),
        drop_reason=r.get("drop_reason", ""),
        manual_review_status=r.get("manual_review_status", "not_required"),
        stage_decided=r.get("stage_decided", "1a"),
        notes=r.get("notes", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 — sense split")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--sense-map", type=Path, default=SENSE_MAP_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows of input (smoke test)",
    )
    parser.add_argument("--no-ledger-update", action="store_true")
    parser.add_argument("--no-sense-map-update", action="store_true")
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        ledger_path=args.ledger,
        sense_map_path=args.sense_map,
        output_path=args.output,
        audit_path=args.audit,
        concurrency=args.concurrency,
        limit=args.limit,
        update_ledger=not args.no_ledger_update,
        update_sense_map=not args.no_sense_map_update,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
