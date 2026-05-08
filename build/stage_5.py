"""Stage 5 — IPA (eSpeak-NG baseline + Anthropic Sonnet correction).

Reads:
    data/04-examples.tsv               (Stage 4 output)
    data/_manual_ipa.tsv               (manual overrides; wins over LLM)

Writes:
    data/05-ipa.tsv                    (per-sense IPA fields)
    audit/05_ipa.jsonl                 (LLM provenance)

Pipeline:
    Phase 1 (deterministic, no API): eSpeak-NG transcribes:
      - the headword (or full idiom for idiom rows) -> ipa_word_machine
      - example_pt token-by-token (isolated form) -> ipa_example_machine
    Phase 2 (LLM): Anthropic Sonnet corrects the machine baseline for known
      BP-paulistano issues (closed-vowel realizations, /ʁ/ vs /ɾ/, palatalization
      of /t/ /d/ before /i/, nasal handling). Output: ipa_word_final,
      ipa_example_final, confidence, notes.

The final TSV exposes ipa_word and ipa_example (= *_final).

Hard invariants:
    - Every Stage-4 sense_id has exactly one row in 05-ipa.tsv.
    - ipa_word_final non-empty for every row.
    - ipa_example_final whitespace-token count == example_pt whitespace-token count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.ipa import (  # noqa: E402
    EspeakNotFound,
    token_count,
    transcribe_tokens,
    transcribe_word,
)
from build.lib.llm import TIER_DEFAULT, AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.validate import tokenize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "04-examples.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_ipa.tsv"
OUTPUT_PATH = DATA_DIR / "05-ipa.tsv"
AUDIT_PATH = AUDIT_DIR / "05_ipa.jsonl"

OUTPUT_FIELDS = [
    "sense_id",
    "rank",
    "expansion_index",
    "sense_index",
    "pt",
    "pt_display",
    "pt_type",
    "gender",
    "pos",
    "en_primary",
    "en_all",
    "tags",
    "bp_status",
    "example_pt",
    "example_en",
    "target_word_used",
    "ipa_word_machine",
    "ipa_word_final",
    "ipa_example_machine",
    "ipa_example_final",
    "ipa_source",        # 'machine' | 'corrected' | 'manual_override' | 'fallback'
    "ipa_confidence",    # 'high' | 'medium' | 'low'
    "ipa_notes",         # ≤80 chars; what changed
]

DEFAULT_CONCURRENCY = 16

CORRECT_IPA_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "ipa_word_final": {
            "type": "string",
            "description": "Corrected IPA for the headword (or full idiom phrase).",
        },
        "ipa_example_final": {
            "type": "string",
            "description": (
                "Corrected per-token IPA for example_pt, space-separated. "
                "Token count must equal whitespace-token count of example_pt."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "notes": {
            "type": "string",
            "description": "≤80 chars summary of changes (or 'no changes').",
        },
    },
    "required": ["ipa_word_final", "ipa_example_final", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "ipa_correct.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Correct eSpeak-NG IPA output for Brazilian-Portuguese paulistano accent. "
        "Use Tool Use."
    )


def _load_overrides(path: Path) -> dict[str, dict[str, str]]:
    """Map sense_id -> {ipa_word, ipa_example}."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "ipa_word": (r.get("ipa_word_override") or "").strip(),
            "ipa_example": (r.get("ipa_example_override") or "").strip(),
            "notes": (r.get("notes") or "").strip(),
        }
    return out


@dataclass
class IPAResult:
    sense_id: str
    ipa_word_final: str
    ipa_example_final: str
    source: str  # 'machine' | 'corrected' | 'manual_override' | 'fallback'
    confidence: str
    notes: str


def _build_user_message(
    row: dict, ipa_word_machine: str, ipa_example_machine: str
) -> str:
    return (
        f"pt: {row.get('pt', '')}\n"
        f"example_pt: {row.get('example_pt', '')}\n"
        f"ipa_word_machine: {ipa_word_machine}\n"
        f"ipa_example_machine: {ipa_example_machine}"
    )


def _correct_one(
    row: dict,
    ipa_word_machine: str,
    ipa_example_machine: str,
    *,
    client: AnthropicClient,
    system_prompt: str,
) -> IPAResult:
    sid = row["sense_id"]
    user_msg = _build_user_message(row, ipa_word_machine, ipa_example_machine)
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="correct_ipa",
            tool_description=(
                "Correct eSpeak-NG IPA output for Brazilian-Portuguese "
                "paulistano accent. Light-touch corrections only; preserve "
                "correct phonemes verbatim."
            ),
            tool_input_schema=CORRECT_IPA_TOOL_SCHEMA,
            stage="5",
            provenance_key=sid,
            tier=TIER_DEFAULT,
            max_tokens=512,
        )
    except Exception as exc:
        # Fall back to machine baseline; pipeline doesn't stall
        return IPAResult(
            sense_id=sid,
            ipa_word_final=ipa_word_machine,
            ipa_example_final=ipa_example_machine,
            source="fallback",
            confidence="low",
            notes=f"LLM error fallback: {type(exc).__name__}",
        )

    ipa_word_final = (decision.get("ipa_word_final") or "").strip()
    ipa_example_final = (decision.get("ipa_example_final") or "").strip()

    # Defensive: if LLM returned empty, fall back
    if not ipa_word_final:
        ipa_word_final = ipa_word_machine
    if not ipa_example_final:
        ipa_example_final = ipa_example_machine

    return IPAResult(
        sense_id=sid,
        ipa_word_final=ipa_word_final,
        ipa_example_final=ipa_example_final,
        source="corrected",
        confidence=(decision.get("confidence") or "medium").strip().lower(),
        notes=(decision.get("notes") or "").strip()[:200],
    )


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    output_path: Path = OUTPUT_PATH,
    audit_path: Path = AUDIT_PATH,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    sense_id_filter: set[str] | None = None,
    skip_correction: bool = False,
    anthropic_client: AnthropicClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 4 first."
        )

    rows = read_tsv(input_path)
    if sense_id_filter is not None:
        rows = [r for r in rows if r["sense_id"] in sense_id_filter]
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("[info] no rows to process; exiting", file=sys.stderr)
        return {"input_rows": 0, "output_rows": 0}

    overrides = _load_overrides(overrides_path)
    system_prompt = _load_prompt()

    # --- Phase 1: machine baseline (eSpeak) ----------------------------------

    print(f"[info] Stage 5: input={len(rows)}, running eSpeak baseline...", file=sys.stderr)
    machine: dict[str, tuple[str, str]] = {}
    espeak_failures: list[str] = []
    for i, row in enumerate(rows, start=1):
        sid = row["sense_id"]
        try:
            ipa_word = transcribe_word(row.get("pt") or "")
            ipa_example = transcribe_tokens(row.get("example_pt") or "")
        except EspeakNotFound:
            raise
        except Exception as exc:
            espeak_failures.append(sid)
            ipa_word = ""
            ipa_example = ""
            print(
                f"[warn] eSpeak failed on {sid}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        machine[sid] = (ipa_word, ipa_example)
        if i % 500 == 0:
            print(f"[info] eSpeak: {i}/{len(rows)}", file=sys.stderr)

    # --- Phase 2: LLM correction ---------------------------------------------

    results: dict[str, IPAResult] = {}
    rows_to_correct: list[dict] = []
    manual_count = 0

    for row in rows:
        sid = row["sense_id"]
        ipa_word_machine, ipa_example_machine = machine[sid]
        ov = overrides.get(sid, {})
        if ov.get("ipa_word") or ov.get("ipa_example"):
            results[sid] = IPAResult(
                sense_id=sid,
                ipa_word_final=ov.get("ipa_word") or ipa_word_machine,
                ipa_example_final=ov.get("ipa_example") or ipa_example_machine,
                source="manual_override",
                confidence="high",
                notes=ov.get("notes", "manual override"),
            )
            manual_count += 1
        else:
            rows_to_correct.append(row)

    if skip_correction:
        # Use machine baseline as final output
        for row in rows_to_correct:
            sid = row["sense_id"]
            ipa_word_machine, ipa_example_machine = machine[sid]
            results[sid] = IPAResult(
                sense_id=sid,
                ipa_word_final=ipa_word_machine,
                ipa_example_final=ipa_example_machine,
                source="machine",
                confidence="medium",
                notes="skip_correction=True",
            )
    elif rows_to_correct:
        if anthropic_client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            anthropic_client = AnthropicClient(audit_path=audit_path)

        print(
            f"[info] Stage 5: manual_overrides={manual_count}, "
            f"to_correct={len(rows_to_correct)}",
            file=sys.stderr,
        )
        completed = 0
        progress_step = max(50, len(rows_to_correct) // 25)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_sid = {
                pool.submit(
                    _correct_one,
                    r,
                    machine[r["sense_id"]][0],
                    machine[r["sense_id"]][1],
                    client=anthropic_client,
                    system_prompt=system_prompt,
                ): r["sense_id"]
                for r in rows_to_correct
            }
            for fut in as_completed(future_to_sid):
                res = fut.result()
                results[res.sense_id] = res
                completed += 1
                if completed % progress_step == 0:
                    print(
                        f"[info] Stage 5 correction: {completed}/{len(rows_to_correct)}",
                        file=sys.stderr,
                    )

    # --- Build output --------------------------------------------------------

    output_rows: list[dict] = []
    token_mismatch_count = 0
    empty_word_count = 0

    for row in rows:
        sid = row["sense_id"]
        ipa_word_machine, ipa_example_machine = machine[sid]
        res = results[sid]

        # Token-count invariant check (soft warning)
        ex_tok_count = token_count(row.get("example_pt") or "")
        ipa_tok_count = token_count(res.ipa_example_final)
        if ex_tok_count != ipa_tok_count and ex_tok_count > 0:
            token_mismatch_count += 1

        if not res.ipa_word_final:
            empty_word_count += 1

        out = {
            "sense_id": sid,
            "rank": row["rank"],
            "expansion_index": row.get("expansion_index", ""),
            "sense_index": row.get("sense_index", ""),
            "pt": row.get("pt", ""),
            "pt_display": row.get("pt_display", ""),
            "pt_type": row.get("pt_type", ""),
            "gender": row.get("gender", ""),
            "pos": row.get("pos", ""),
            "en_primary": row.get("en_primary", ""),
            "en_all": row.get("en_all", ""),
            "tags": row.get("tags", ""),
            "bp_status": row.get("bp_status", ""),
            "example_pt": row.get("example_pt", ""),
            "example_en": row.get("example_en", ""),
            "target_word_used": row.get("target_word_used", ""),
            "ipa_word_machine": ipa_word_machine,
            "ipa_word_final": res.ipa_word_final,
            "ipa_example_machine": ipa_example_machine,
            "ipa_example_final": res.ipa_example_final,
            "ipa_source": res.source,
            "ipa_confidence": res.confidence,
            "ipa_notes": res.notes,
        }
        output_rows.append(out)

    n_out = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)

    # --- Summary -------------------------------------------------------------

    from collections import Counter

    source_counts = Counter(r["ipa_source"] for r in output_rows)
    confidence_counts = Counter(r["ipa_confidence"] for r in output_rows)

    summary = {
        "input_rows": len(rows),
        "output_rows": n_out,
        "manual_overrides": manual_count,
        "espeak_failures": len(espeak_failures),
        "token_mismatch_count": token_mismatch_count,
        "empty_word_count": empty_word_count,
        "ipa_source_counts": dict(source_counts),
        "ipa_confidence_counts": dict(confidence_counts),
    }
    if anthropic_client is not None and not skip_correction:
        summary["anthropic_cache_stats"] = anthropic_client.cache_stats
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 — IPA")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-correction", action="store_true",
                        help="Use eSpeak baseline as final output (no LLM call)")
    parser.add_argument(
        "--smoke-sample",
        type=Path,
        default=None,
        help="Path to a smoke-sample TSV; restrict input to those sense_ids.",
    )
    args = parser.parse_args()

    sense_id_filter: set[str] | None = None
    if args.smoke_sample:
        sample = read_tsv(args.smoke_sample)
        sense_id_filter = {r["sense_id"] for r in sample}

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        output_path=args.output,
        audit_path=args.audit,
        concurrency=args.concurrency,
        limit=args.limit,
        sense_id_filter=sense_id_filter,
        skip_correction=args.skip_correction,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
