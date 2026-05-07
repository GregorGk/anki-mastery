"""Stage 1.5 — BP-status classification.

Reads:
    data/012-lexical_replaced.tsv         (Stage 1c output)
    data/_manual_bp_status.tsv            (manual overrides; wins over LLM)

Writes:
    data/015-bp_status.tsv                (active rows + bp_status; ep_only filtered out)
    data/_ep_drops.tsv                    (audit: rows dropped as ep_only)
    data/_source_ledger.tsv               (in-place: action=drop_ep_only for dropped rows)
    audit/015_bp_status.jsonl             (LLM provenance)

Behaviour:
    1. Load lexical-replaced rows.
    2. For each row:
       - If covered by manual override → use override decision (no LLM call).
       - Else → call Anthropic Tool Use (Sonnet 4.5 default; configurable).
    3. Append `bp_status` + `bp_status_confidence` + `bp_status_method` to
       each row. Filter rows classified `ep_only` from the next-stage TSV
       and append them to `_ep_drops.tsv` with action=drop_ep_only in the
       ledger.

Design notes:
    - Single-juror today (Anthropic). Plan v3 specifies a 3-model jury for
      bp_status decisions, but the user has only the Anthropic key configured.
      The JuryConfig dataclass below is structured so adding more jurors is
      a config change.
    - Concurrency: ThreadPoolExecutor with bounded workers (default 8). The
      Anthropic SDK's HTTP client is thread-safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.ledger import (  # noqa: E402
    ACTION_DROP_EP_ONLY,
    LEDGER_FIELDS,
    LedgerRow,
    write_ledger,
)
from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "012-lexical_replaced.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_bp_status.tsv"
OUTPUT_PATH = DATA_DIR / "015-bp_status.tsv"
EP_DROPS_PATH = DATA_DIR / "_ep_drops.tsv"
LEDGER_PATH = DATA_DIR / "_source_ledger.tsv"
AUDIT_PATH = AUDIT_DIR / "015_bp_status.jsonl"

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_CONCURRENCY = 8

OUTPUT_FIELDS = [
    "source_line_number",
    "rank",
    "expansion_index",
    "source_pt",
    "pt",
    "pt_type",
    "gender",
    "pos",
    "en_all",
    "annotation",
    "normalization_action",
    "bp_replacement",
    "merge_target_rank",
    "merge_target_pt",
    "merged_from_source_lines",
    "source_variants",
    "bp_status",
    "bp_status_confidence",
    "bp_status_method",
    "bp_status_reason",
    "source_line",
]

EP_DROPS_FIELDS = [
    "source_line_number",
    "rank",
    "source_pt",
    "pt",
    "en_all",
    "drop_reason",
    "drop_stage",
    "classifier_confidence",
    "classifier_reason",
    "notes",
]

VALID_BP_STATUS = {"standard", "uncommon", "false_friend", "nsfw", "ep_only"}

BP_STATUS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "bp_status": {
            "type": "string",
            "enum": ["standard", "uncommon", "false_friend", "nsfw", "ep_only"],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "string",
            "description": "Required if confidence != high. One short phrase.",
        },
    },
    "required": ["bp_status", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "bp_status.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Classify the BP-status of a headword. Categories: standard, "
        "uncommon, false_friend, nsfw, ep_only. Use Tool Use."
    )


def _load_manual_overrides(path: Path) -> dict[int, dict]:
    """Map source_line_number -> override dict. Empty if file missing or empty."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[int, dict] = {}
    for r in rows:
        ln = r.get("source_line_number", "").strip()
        status = r.get("bp_status", "").strip()
        if not ln or not status:
            continue
        if status not in VALID_BP_STATUS:
            print(
                f"[warn] manual override line {ln} has invalid bp_status={status!r}; skipping",
                file=sys.stderr,
            )
            continue
        out[int(ln)] = {
            "bp_status": status,
            "confidence": r.get("confidence", "high"),
            "reason": r.get("reason", ""),
        }
    return out


@dataclass
class ClassifyResult:
    bp_status: str
    confidence: str
    reason: str
    method: str  # 'manual_override' | 'llm' | 'fallback_standard'


def classify_one(
    row: dict,
    *,
    client: AnthropicClient,
    system_prompt: str,
) -> ClassifyResult:
    """Call the LLM to classify a single row. Falls back to 'standard' on
    repeated LLM failures so the pipeline never stalls."""
    user_msg = (
        f"pt: {row.get('pt', '')}\n"
        f"en_all: {row.get('en_all', '')}\n"
        f"source_pt: {row.get('source_pt', '')}\n"
    )
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="classify_bp_status",
            tool_description=(
                "Classify the BP-status of a headword as standard, uncommon, "
                "false_friend, nsfw, or ep_only."
            ),
            tool_input_schema=BP_STATUS_TOOL_SCHEMA,
            stage="1.5",
            provenance_key=f"line_{row.get('source_line_number', '?')}_{row.get('pt', '?')}",
        )
    except Exception as exc:
        # Fall back to 'standard' so the pipeline continues; log the issue.
        print(
            f"[warn] line {row.get('source_line_number')}: LLM error "
            f"{type(exc).__name__}: {exc}; defaulting to 'standard'",
            file=sys.stderr,
        )
        return ClassifyResult(
            bp_status="standard",
            confidence="low",
            reason=f"LLM error: {type(exc).__name__}",
            method="fallback_standard",
        )

    bp_status = decision.get("bp_status", "standard")
    if bp_status not in VALID_BP_STATUS:
        # Tool Use enforced the enum but defensive code is cheap.
        print(
            f"[warn] line {row.get('source_line_number')}: invalid bp_status "
            f"{bp_status!r} returned; defaulting to 'standard'",
            file=sys.stderr,
        )
        bp_status = "standard"

    return ClassifyResult(
        bp_status=bp_status,
        confidence=decision.get("confidence", "medium"),
        reason=decision.get("reason", ""),
        method="llm",
    )


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    output_path: Path = OUTPUT_PATH,
    ep_drops_path: Path = EP_DROPS_PATH,
    ledger_path: Path = LEDGER_PATH,
    audit_path: Path = AUDIT_PATH,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    update_ledger: bool = True,
    limit: int | None = None,
    client: AnthropicClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 1c first."
        )
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Stage 1.5 requires an Anthropic "
                "API key (Tool Use enforces output schema)."
            )
        client = AnthropicClient(model=model, audit_path=audit_path)

    rows = read_tsv(input_path)
    if limit is not None:
        rows = rows[:limit]

    overrides = _load_manual_overrides(overrides_path)
    system_prompt = _load_prompt()

    # Build to_classify list (rows without manual override)
    to_classify_indices = [
        i
        for i, r in enumerate(rows)
        if int(r["source_line_number"]) not in overrides
    ]
    results: dict[int, ClassifyResult] = {}

    # Manual overrides first
    for i, r in enumerate(rows):
        ln = int(r["source_line_number"])
        if ln in overrides:
            ov = overrides[ln]
            results[i] = ClassifyResult(
                bp_status=ov["bp_status"],
                confidence=ov["confidence"],
                reason=ov["reason"],
                method="manual_override",
            )

    # Concurrent LLM calls
    if to_classify_indices:
        print(
            f"[info] Stage 1.5: classifying {len(to_classify_indices)} rows "
            f"({len(overrides)} manual overrides) via {model} "
            f"with concurrency={concurrency}",
            file=sys.stderr,
        )
        completed = 0
        progress_step = max(50, len(to_classify_indices) // 20)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    classify_one,
                    rows[i],
                    client=client,
                    system_prompt=system_prompt,
                ): i
                for i in to_classify_indices
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                results[i] = fut.result()
                completed += 1
                if completed % progress_step == 0:
                    print(
                        f"[info] Stage 1.5: {completed}/{len(to_classify_indices)} done",
                        file=sys.stderr,
                    )

    # Build outputs
    output_rows: list[dict] = []
    ep_drop_rows: list[dict] = []
    counts: dict[str, int] = {s: 0 for s in VALID_BP_STATUS}
    method_counts: dict[str, int] = {"llm": 0, "manual_override": 0, "fallback_standard": 0}

    for i, row in enumerate(rows):
        result = results[i]
        counts[result.bp_status] = counts.get(result.bp_status, 0) + 1
        method_counts[result.method] = method_counts.get(result.method, 0) + 1

        if result.bp_status == "ep_only":
            ep_drop_rows.append(
                {
                    "source_line_number": row["source_line_number"],
                    "rank": row["rank"],
                    "source_pt": row.get("source_pt", ""),
                    "pt": row["pt"],
                    "en_all": row.get("en_all", ""),
                    "drop_reason": "Stage 1.5 LLM bp_status=ep_only",
                    "drop_stage": "1.5",
                    "classifier_confidence": result.confidence,
                    "classifier_reason": result.reason,
                    "notes": "",
                }
            )
            # Excluded from 015-bp_status.tsv
            continue

        out = dict(row)
        out["bp_status"] = result.bp_status
        out["bp_status_confidence"] = result.confidence
        out["bp_status_method"] = result.method
        out["bp_status_reason"] = result.reason
        # Ensure all expected fields exist (some carry-through fields may be empty)
        for f in OUTPUT_FIELDS:
            out.setdefault(f, "")
        output_rows.append(out)

    out_count = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)
    drop_count = write_tsv(ep_drops_path, ep_drop_rows, fieldnames=EP_DROPS_FIELDS)

    # Update ledger: rows now classified ep_only get action=drop_ep_only.
    if update_ledger and ep_drop_rows:
        ledger_rows = read_tsv(ledger_path)
        # Index by (source_line_number, expansion_index) using same enumeration as 1b.
        from build.stage_1b import _enumerate_expansion_indices

        ledger_exp = _enumerate_expansion_indices(ledger_rows)
        ledger_by_key: dict[tuple[int, int], int] = {
            (int(r["source_line_number"]), ledger_exp[i]): i
            for i, r in enumerate(ledger_rows)
        }

        # We need to map ep-drop rows to their (source_line_number, expansion_index).
        # The 012-lexical_replaced.tsv carries expansion_index; ep_drop_rows came from
        # rows[] where we have it.
        for i, row in enumerate(rows):
            if results[i].bp_status != "ep_only":
                continue
            key = (
                int(row["source_line_number"]),
                int(row.get("expansion_index", 0)),
            )
            idx = ledger_by_key.get(key)
            if idx is None:
                continue
            ledger_rows[idx]["action"] = ACTION_DROP_EP_ONLY
            ledger_rows[idx]["drop_reason"] = "Stage 1.5 LLM bp_status=ep_only"
            ledger_rows[idx]["stage_decided"] = "1.5"
            existing_notes = ledger_rows[idx].get("notes", "")
            extra = (
                f"classifier_confidence={results[i].confidence}; "
                f"reason={results[i].reason}"
            )
            ledger_rows[idx]["notes"] = (
                f"{existing_notes}; {extra}".strip("; ") if existing_notes else extra
            )

        coerced = [_dict_to_ledger(r) for r in ledger_rows]
        write_ledger(ledger_path, coerced)

    # --- Hard invariants ----------------------------------------------------
    for r in output_rows:
        assert r["bp_status"] in VALID_BP_STATUS - {"ep_only"}, (
            f"line {r['source_line_number']}: invalid bp_status {r['bp_status']!r} "
            f"in active output (ep_only should be filtered)"
        )

    return {
        "input_rows": len(rows),
        "output_rows": out_count,
        "ep_drops": drop_count,
        "manual_overrides_used": method_counts["manual_override"],
        "llm_calls": method_counts["llm"],
        "fallback_standard_count": method_counts["fallback_standard"],
        "bp_status_counts": counts,
    }


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
    parser = argparse.ArgumentParser(
        description="Stage 1.5 — BP-status classification (Tool Use)"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--ep-drops", type=Path, default=EP_DROPS_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (for smoke tests)",
    )
    parser.add_argument("--no-ledger-update", action="store_true")
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        output_path=args.output,
        ep_drops_path=args.ep_drops,
        ledger_path=args.ledger,
        audit_path=args.audit,
        model=args.model,
        concurrency=args.concurrency,
        limit=args.limit,
        update_ledger=not args.no_ledger_update,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
