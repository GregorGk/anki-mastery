"""Stage 1a — Parse source.txt and initialize the ledger.

Reads:
    data/source.txt                       (immutable)

Writes:
    data/_source_ledger.tsv               (one row per source.txt line)
    data/_sense_source_map.tsv            (initial provenance edges)
    data/_idioms_expanded.tsv             (audit: candidates + LLM expansions)
    data/_flags.tsv                       (initial structured flags)
    audit/01a_parse.jsonl                 (LLM provenance for idiom expansion)

Behaviour:
    1. Parse each line: split on FIRST ` = ` only (handles 3 embedded-= cases).
    2. Detect parentheticals; classify markers (BP/EP/+se/M-F-split).
    3. Detect idiom-expansion candidates; resolve via Anthropic Tool Use.
       (Bypasses LLM if ANTHROPIC_API_KEY missing — falls back to heuristic.)
    4. Initialise ledger (action=keep) and provenance map (provenance=original).
    5. Add expansion rows for resolved idioms (action=expand_idiom_added).

This stage performs no orthographic or lexical replacement. That happens in 1b
and 1c respectively. `_flags.tsv` is initialized here from (BP)/(EP) markers
and gets enriched in later stages.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE checking environment so the API key is visible. override=True
# is intentional: shell may have ANTHROPIC_API_KEY=""(empty) which otherwise
# wins and silently disables the LLM.
load_dotenv(override=True)

# Allow running as `python build/01a_parse.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.ledger import (
    ACTION_EXPAND_IDIOM_ADDED,
    ACTION_KEEP,
    LEDGER_FIELDS,
    PROVENANCE_IDIOM_EXPANSION,
    PROVENANCE_ORIGINAL,
    LedgerRow,
    SenseSourceEdge,
    write_ledger,
    write_sense_source_map,
)
from build.lib.parse import ParsedLine, parse_line
from build.lib.tsv import write_tsv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

SOURCE_FILE = DATA_DIR / "source.txt"
LEDGER_OUT = DATA_DIR / "_source_ledger.tsv"
SENSE_MAP_OUT = DATA_DIR / "_sense_source_map.tsv"
IDIOMS_OUT = DATA_DIR / "_idioms_expanded.tsv"
FLAGS_OUT = DATA_DIR / "_flags.tsv"
AUDIT_OUT = AUDIT_DIR / "01a_parse.jsonl"

IDIOMS_FIELDS = [
    "source_line_number",
    "headword",
    "parenthetical",
    "expansions",
    "expansion_count",
    "method",
    "confidence",
    "notes",
]

FLAGS_FIELDS = [
    "source_line_number",
    "rank",
    "source_pt",
    "pt",
    "source_line",
    "flag_type",
    "flag_reason",
    "manual_review_required",
    "manual_status",
    "approved_by",
    "approved_at",
    "notes",
]


# --- Idiom expansion ---------------------------------------------------------

IDIOM_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "expansions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Full idiom phrase(s) in Brazilian Portuguese, with crase / "
                "contractions applied. Empty array if the abbreviation cannot "
                "be confidently expanded."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "string",
            "description": "Required if confidence is medium or low.",
        },
    },
    "required": ["expansions", "confidence"],
}


IDIOM_PROMPT_PATH = REPO_ROOT / "build" / "prompts" / "idiom_expand.md"


def _idiom_system_prompt() -> str:
    if IDIOM_PROMPT_PATH.exists():
        return IDIOM_PROMPT_PATH.read_text(encoding="utf-8")
    # Fallback minimal prompt
    return (
        "You are a Brazilian-Portuguese lexicographer. Expand the parenthetical "
        "abbreviation by substituting the headword for the period-marked initial. "
        "Apply BP crase and contractions. Return via the expand_idiom tool."
    )


@dataclass
class IdiomExpansion:
    expansions: list[str]
    confidence: str
    reason: str
    method: str  # 'llm' | 'heuristic' | 'skipped'


def _heuristic_expand(headword: str, parenthetical: str) -> list[str]:
    """Best-effort deterministic expansion when LLM is unavailable.

    Substitutes `<letter>.` (matching the headword's initial) with the headword.
    Handles `em / ao r.` style alternatives. If the parenthetical contains an
    embedded `=` (e.g., `a m. que = to the degree / extent that`), only the
    portion BEFORE the first ` = ` is treated as the idiom phrase — the rest
    is an English gloss.

    Note: this does NOT apply BP crase (`a medida que` → `à medida que`).
    LLM expansion handles crase; the heuristic is intentionally conservative.
    """
    import re

    head_initial = headword[0].lower() if headword else ""
    if not head_initial:
        return []

    text = parenthetical.strip()
    # If the parenthetical contains ` = `, the idiom phrase ends there.
    if " = " in text:
        text = text.split(" = ", 1)[0].strip()

    pattern = re.compile(rf"\b([{head_initial.upper()}{head_initial}])\.")
    if not pattern.search(text):
        return []

    # Handle `em / ao r.` style: split on " / " when the pattern is present.
    if " / " in text:
        prefix_match = re.match(
            r"^(.*?)\s+/\s+(\S+)\s+([a-zà-ÿ])\.\s*(.*)$",
            text,
            re.IGNORECASE,
        )
        if prefix_match:
            left, right, abbrev, tail = prefix_match.groups()
            if abbrev.lower() == head_initial:
                tail_part = f" {tail}".rstrip() if tail.strip() else ""
                return [
                    f"{left} {headword}{tail_part}".strip(),
                    f"{right} {headword}{tail_part}".strip(),
                ]

    expanded = pattern.sub(headword, text)
    return [expanded.strip()]


def expand_idiom(
    parsed: ParsedLine,
    parenthetical: str,
    *,
    llm_client,
) -> IdiomExpansion:
    """Resolve a single idiom candidate. Uses LLM if available; heuristic otherwise."""
    if llm_client is None:
        exps = _heuristic_expand(parsed.pt, parenthetical)
        return IdiomExpansion(
            expansions=exps,
            confidence="medium" if exps else "low",
            reason="heuristic fallback (no LLM)",
            method="heuristic",
        )

    user_msg = (
        f"headword: {parsed.pt}\n"
        f"parenthetical: {parenthetical}\n"
        f"en_all: {parsed.en_all}\n"
    )
    try:
        decision = llm_client.call_tool(
            system=_idiom_system_prompt(),
            user_message=user_msg,
            tool_name="expand_idiom",
            tool_description="Expand a Brazilian-Portuguese idiom abbreviation against its headword.",
            tool_input_schema=IDIOM_TOOL_SCHEMA,
            stage="1a",
            provenance_key=f"line_{parsed.source_line_number}_{parenthetical[:32]}",
        )
    except Exception as exc:
        # Fall back to heuristic — never crash Stage 1a on a transient LLM failure.
        exps = _heuristic_expand(parsed.pt, parenthetical)
        return IdiomExpansion(
            expansions=exps,
            confidence="low",
            reason=f"LLM error: {type(exc).__name__}: {exc}; fell back to heuristic",
            method="heuristic",
        )

    return IdiomExpansion(
        expansions=list(decision.get("expansions", [])),
        confidence=decision.get("confidence", "low"),
        reason=decision.get("reason", ""),
        method="llm",
    )


# --- Main pipeline -----------------------------------------------------------


def run(
    *,
    source_path: Path = SOURCE_FILE,
    ledger_out: Path = LEDGER_OUT,
    sense_map_out: Path = SENSE_MAP_OUT,
    idioms_out: Path = IDIOMS_OUT,
    flags_out: Path = FLAGS_OUT,
    audit_out: Path = AUDIT_OUT,
    use_llm: bool = True,
    limit: int | None = None,
) -> dict:
    """Run Stage 1a end-to-end. Returns a summary dict."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    # Optionally instantiate the LLM client.
    llm_client = None
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from build.lib.llm import AnthropicClient

            llm_client = AnthropicClient(audit_path=audit_out)
        except Exception as exc:
            print(f"[warn] LLM init failed ({exc}); falling back to heuristic.", file=sys.stderr)
            llm_client = None
    elif use_llm:
        print("[warn] ANTHROPIC_API_KEY not set; idiom expansions use heuristic.", file=sys.stderr)

    ledger_rows: list[LedgerRow] = []
    sense_edges: list[SenseSourceEdge] = []
    idiom_audit_rows: list[dict] = []
    flag_rows: list[dict] = []

    parse_failures: list[tuple[int, str]] = []

    with source_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            if limit is not None and line_num > limit:
                break
            stripped = raw.rstrip("\n")
            if not stripped.strip():
                # Empty line: log to ledger as-is for completeness.
                ledger_rows.append(
                    LedgerRow(
                        source_line_number=line_num,
                        rank=line_num,
                        source_raw=stripped,
                        action="drop_ep_only",  # treat as non-data; not really EP-only
                        drop_reason="empty line in source",
                    )
                )
                continue

            parsed = parse_line(line_num, raw)
            if parsed is None:
                parse_failures.append((line_num, stripped))
                ledger_rows.append(
                    LedgerRow(
                        source_line_number=line_num,
                        rank=line_num,
                        source_raw=stripped,
                        action="manual_review",
                        manual_review_status="pending",
                        notes="parse failure: no ` = ` separator",
                    )
                )
                continue

            # Original ledger row
            placeholder_sense = f"{line_num:04d}.00.00"  # populated by Stage 2
            ledger_rows.append(
                LedgerRow(
                    source_line_number=line_num,
                    rank=line_num,
                    source_raw=stripped,
                    source_pt=parsed.pt,
                    source_en_all=parsed.en_all,
                    action=ACTION_KEEP,
                    notes=f"pt_type={parsed.pt_type}",
                )
            )

            # Provenance edge for the original row (sense_id will be assigned in Stage 2;
            # for now we record source_line_number → placeholder so the verify_all
            # round-trip works post-Stage-2).
            sense_edges.append(
                SenseSourceEdge(
                    sense_id=placeholder_sense,
                    source_line_number=line_num,
                    provenance_type=PROVENANCE_ORIGINAL,
                    notes=f"pt={parsed.pt}",
                )
            )

            # (BP)/(EP)/mainly-EP flags
            if parsed.has_bp_marker:
                flag_rows.append(
                    {
                        "source_line_number": line_num,
                        "rank": line_num,
                        "source_pt": parsed.pt,
                        "pt": parsed.pt,
                        "source_line": stripped,
                        "flag_type": "bp_marker",
                        "flag_reason": "RHS contains (BP)",
                        "manual_review_required": "yes",
                        "manual_status": "pending",
                    }
                )
            if parsed.has_ep_marker:
                flag_rows.append(
                    {
                        "source_line_number": line_num,
                        "rank": line_num,
                        "source_pt": parsed.pt,
                        "pt": parsed.pt,
                        "source_line": stripped,
                        "flag_type": "ep_marker",
                        "flag_reason": "RHS contains (EP)",
                        "manual_review_required": "yes",
                        "manual_status": "pending",
                    }
                )
            if parsed.has_mainly_ep_marker:
                flag_rows.append(
                    {
                        "source_line_number": line_num,
                        "rank": line_num,
                        "source_pt": parsed.pt,
                        "pt": parsed.pt,
                        "source_line": stripped,
                        "flag_type": "ep_marker",
                        "flag_reason": "RHS contains 'mainly EP'",
                        "manual_review_required": "yes",
                        "manual_status": "pending",
                    }
                )

            # Idiom expansion
            for parenthetical in parsed.idiom_candidates:
                exp = expand_idiom(parsed, parenthetical, llm_client=llm_client)
                idiom_audit_rows.append(
                    {
                        "source_line_number": line_num,
                        "headword": parsed.pt,
                        "parenthetical": parenthetical,
                        "expansions": json.dumps(exp.expansions, ensure_ascii=False),
                        "expansion_count": len(exp.expansions),
                        "method": exp.method,
                        "confidence": exp.confidence,
                        "notes": exp.reason,
                    }
                )
                # Add ledger rows + provenance edges for each expansion (1+).
                for idx, phrase in enumerate(exp.expansions, start=1):
                    expanded_sense = f"{line_num:04d}.{idx:02d}.00"
                    ledger_rows.append(
                        LedgerRow(
                            source_line_number=line_num,
                            rank=line_num,
                            source_raw=stripped,
                            source_pt=phrase,  # the expanded phrase becomes the pt for this row
                            source_en_all=parsed.en_all,
                            action=ACTION_EXPAND_IDIOM_ADDED,
                            normalization_action="idiom_expansion",
                            notes=(
                                f"expanded from parenthetical '{parenthetical}' "
                                f"(method={exp.method}, confidence={exp.confidence})"
                            ),
                            stage_decided="1a",
                        )
                    )
                    sense_edges.append(
                        SenseSourceEdge(
                            sense_id=expanded_sense,
                            source_line_number=line_num,
                            provenance_type=PROVENANCE_IDIOM_EXPANSION,
                            notes=f"phrase={phrase}",
                        )
                    )

    # Write outputs
    ledger_count = write_ledger(ledger_out, ledger_rows)
    edge_count = write_sense_source_map(sense_map_out, sense_edges)
    idiom_count = write_tsv(idioms_out, idiom_audit_rows, fieldnames=IDIOMS_FIELDS)
    flag_count = write_tsv(flags_out, flag_rows, fieldnames=FLAGS_FIELDS)

    # --- Hard invariants ----------------------------------------------------
    # Every source line appears exactly once in the ledger as `original` (action != expand_idiom_added)
    original_lines = {
        r.source_line_number
        for r in ledger_rows
        if r.action != ACTION_EXPAND_IDIOM_ADDED
    }
    expected_max = limit or sum(1 for _ in source_path.open("r", encoding="utf-8"))
    if len(original_lines) != expected_max:
        raise AssertionError(
            f"Ledger original-rows count {len(original_lines)} != source line count {expected_max}"
        )
    # No `=` in any pt of a kept row
    for r in ledger_rows:
        if r.action == "manual_review":
            continue
        if "=" in r.source_pt:
            raise AssertionError(
                f"Line {r.source_line_number}: source_pt contains '=': {r.source_pt!r}"
            )

    summary = {
        "source_lines_processed": expected_max,
        "ledger_rows_written": ledger_count,
        "ledger_original_rows": len(original_lines),
        "ledger_idiom_expansion_rows": ledger_count - len(original_lines)
        - sum(1 for r in ledger_rows if r.action == "drop_ep_only"),
        "sense_source_edges": edge_count,
        "idiom_candidates_resolved": idiom_count,
        "flags_logged": flag_count,
        "parse_failures": len(parse_failures),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1a — parse + ledger init")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N lines of source.txt (for smoke tests)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Anthropic LLM idiom expansion; use deterministic heuristic",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_FILE,
        help="Path to source.txt",
    )
    parser.add_argument(
        "--ledger-out", type=Path, default=LEDGER_OUT, help="Output: _source_ledger.tsv"
    )
    parser.add_argument(
        "--sense-map-out",
        type=Path,
        default=SENSE_MAP_OUT,
        help="Output: _sense_source_map.tsv",
    )
    parser.add_argument(
        "--idioms-out", type=Path, default=IDIOMS_OUT, help="Output: _idioms_expanded.tsv"
    )
    parser.add_argument(
        "--flags-out", type=Path, default=FLAGS_OUT, help="Output: _flags.tsv"
    )
    parser.add_argument(
        "--audit-out", type=Path, default=AUDIT_OUT, help="Output: audit/01a_parse.jsonl"
    )
    args = parser.parse_args()

    summary = run(
        source_path=args.source,
        ledger_out=args.ledger_out,
        sense_map_out=args.sense_map_out,
        idioms_out=args.idioms_out,
        flags_out=args.flags_out,
        audit_out=args.audit_out,
        use_llm=not args.no_llm,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
