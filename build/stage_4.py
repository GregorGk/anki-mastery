"""Stage 4 — Example sentence generation + cross-family validation.

Reads:
    data/03-enriched.tsv                (Stage 3 output)
    data/_manual_examples.tsv           (manual overrides; wins over LLM)

Writes:
    data/04-examples.tsv                (one row per sense + example fields)
    data/_example_fixes.tsv             (validator failures for re-generation)
    audit/04_examples.jsonl             (Anthropic generator provenance)
    audit/04_validate.jsonl             (OpenAI validator provenance)

Pipeline:
    1. Apply manual overrides first.
    2. Generate example sentences via Anthropic (Sonnet) with prompt caching.
       Submit as Anthropic Message Batch (50% discount) when row count is large
       enough; else sync.
    3. Deterministic validation: token_in_sentence, word count.
    4. Cross-family validation: OpenAI mini-tier flags semantic / BP /
       sense-mismatch / sensitive-policy issues.
    5. Failed rows route to _example_fixes.tsv with the defect noted.

Hard invariants:
    - sense_id uniqueness preserved
    - every output row has example_pt and example_en
    - target_word_used appears as a complete token in example_pt (when
      generation succeeds; failures are flagged but written so the pipeline
      doesn't stall)
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

from build.lib.example_gen import (  # noqa: E402
    build_generator_user_message,
    build_validator_user_message,
    example_policy_for_row,
)
from build.lib.llm import (  # noqa: E402
    TIER_DEFAULT,
    AnthropicClient,
    OpenAIClient,
)
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.validate import (  # noqa: E402
    example_within_word_limit,
    token_in_sentence,
    word_count,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "03-enriched.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_examples.tsv"
OUTPUT_PATH = DATA_DIR / "04-examples.tsv"
FIXES_PATH = DATA_DIR / "_example_fixes.tsv"
GENERATE_AUDIT = AUDIT_DIR / "04_examples.jsonl"
VALIDATE_AUDIT = AUDIT_DIR / "04_validate.jsonl"

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
    "example_method",  # 'manual_override' | 'llm' | 'fallback'
    "example_token_match",  # 'pass' | 'fail'
    "example_word_count_ok",  # 'pass' | 'fail'
    "example_validation_status",  # 'pass' | 'fail' | 'borderline' | 'skipped'
    "example_validation_reason",
    "example_policy",
]

FIXES_FIELDS = [
    "sense_id",
    "rank",
    "pt",
    "en_primary",
    "example_pt",
    "example_en",
    "target_word_used",
    "failure_reason",
    "failure_axes",
]

DEFAULT_CONCURRENCY = 16

GENERATOR_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "example_pt": {
            "type": "string",
            "description": "Brazilian-Portuguese example sentence, ≤15 words, A1/A2.",
        },
        "example_en": {
            "type": "string",
            "description": "Faithful English translation.",
        },
        "target_word_used": {
            "type": "string",
            "description": "Exact surface form of the target word as it appears in example_pt (lowercase).",
        },
    },
    "required": ["example_pt", "example_en", "target_word_used"],
}

VALIDATOR_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "uses_intended_sense": {"type": "boolean"},
        "is_bp": {"type": "boolean"},
        "is_natural": {"type": "boolean"},
        "translation_matches": {"type": "boolean"},
        "fails_neutral_example_policy": {"type": "boolean"},
        "validation_status": {
            "type": "string",
            "enum": ["pass", "fail", "borderline"],
        },
        "validation_reason": {"type": "string"},
    },
    "required": ["validation_status"],
}


def _load_prompt(filename: str) -> str:
    p = REPO_ROOT / "build" / "prompts" / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _load_manual_overrides(path: Path) -> dict[str, dict]:
    """Map sense_id -> {example_pt, example_en, target_word_used}."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "example_pt": (r.get("example_pt") or "").strip(),
            "example_en": (r.get("example_en") or "").strip(),
            "target_word_used": (r.get("target_word_used") or "").strip(),
        }
    return out


@dataclass
class GenerationResult:
    sense_id: str
    example_pt: str
    example_en: str
    target_word_used: str
    method: str  # 'manual_override' | 'llm' | 'fallback'
    error: str = ""


@dataclass
class ValidationResult:
    sense_id: str
    status: str  # 'pass' | 'fail' | 'borderline' | 'skipped'
    reason: str = ""
    failure_axes: list[str] | None = None


def _check_deterministic(
    target_word_used: str, example_pt: str
) -> tuple[bool, bool]:
    """Return (token_match, word_count_ok)."""
    token_match = bool(target_word_used) and token_in_sentence(
        target_word_used, example_pt
    )
    wc_ok = example_within_word_limit(example_pt, limit=15)
    return token_match, wc_ok


def _generate_one(
    row: dict,
    *,
    client: AnthropicClient,
    system_prompt: str,
) -> GenerationResult:
    sid = row["sense_id"]
    user_msg = build_generator_user_message(row)
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="generate_example",
            tool_description=(
                "Generate one BP example sentence (≤15 words, A1/A2) for "
                "the given sense, with English translation and the literal "
                "surface form of the target word."
            ),
            tool_input_schema=GENERATOR_TOOL_SCHEMA,
            stage="4",
            provenance_key=sid,
            tier=TIER_DEFAULT,
            max_tokens=512,
        )
    except Exception as exc:
        return GenerationResult(
            sense_id=sid,
            example_pt="",
            example_en="",
            target_word_used="",
            method="fallback",
            error=f"{type(exc).__name__}: {exc}",
        )

    return GenerationResult(
        sense_id=sid,
        example_pt=(decision.get("example_pt") or "").strip(),
        example_en=(decision.get("example_en") or "").strip(),
        target_word_used=(decision.get("target_word_used") or "").strip().lower(),
        method="llm",
    )


def _validate_one(
    row: dict,
    *,
    gen: GenerationResult,
    client: OpenAIClient,
    system_prompt: str,
) -> ValidationResult:
    sid = gen.sense_id
    if not gen.example_pt or not gen.example_en:
        return ValidationResult(
            sense_id=sid,
            status="skipped",
            reason="no example generated",
            failure_axes=["generation_failed"],
        )

    user_msg = build_validator_user_message(
        row,
        example_pt=gen.example_pt,
        example_en=gen.example_en,
        target_word_used=gen.target_word_used,
    )
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="validate_example",
            tool_description=(
                "Validate a generated BP example sentence on intended-sense, "
                "BP-purity, naturalness, translation match, and sensitive policy."
            ),
            tool_input_schema=VALIDATOR_TOOL_SCHEMA,
            stage="4_validate",
            provenance_key=sid,
            max_tokens=512,
        )
    except Exception as exc:
        return ValidationResult(
            sense_id=sid,
            status="skipped",
            reason=f"validator error: {type(exc).__name__}: {exc}",
            failure_axes=["validator_error"],
        )

    status = decision.get("validation_status") or "fail"
    reason = decision.get("validation_reason") or ""

    failure_axes: list[str] = []
    for axis in (
        "uses_intended_sense",
        "is_bp",
        "is_natural",
        "translation_matches",
    ):
        v = decision.get(axis)
        if v is False:
            failure_axes.append(axis)
    if decision.get("fails_neutral_example_policy") is True:
        failure_axes.append("sensitive_policy")

    return ValidationResult(
        sense_id=sid,
        status=status,
        reason=reason,
        failure_axes=failure_axes or None,
    )


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    output_path: Path = OUTPUT_PATH,
    fixes_path: Path = FIXES_PATH,
    generate_audit: Path = GENERATE_AUDIT,
    validate_audit: Path = VALIDATE_AUDIT,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    sense_id_filter: set[str] | None = None,
    skip_validator: bool = False,
    anthropic_client: AnthropicClient | None = None,
    openai_client: OpenAIClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 3 first."
        )

    rows = read_tsv(input_path)
    if sense_id_filter is not None:
        rows = [r for r in rows if r["sense_id"] in sense_id_filter]
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("[info] no rows to process; exiting", file=sys.stderr)
        return {"input_rows": 0, "output_rows": 0}

    overrides = _load_manual_overrides(overrides_path)
    generate_prompt = _load_prompt("example_generate.md")
    validate_prompt = _load_prompt("example_validate.md")

    if anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        anthropic_client = AnthropicClient(audit_path=generate_audit)
    if not skip_validator and openai_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        openai_client = OpenAIClient(audit_path=validate_audit)

    # --- Phase 1: Generation -------------------------------------------------

    generations: dict[str, GenerationResult] = {}
    rows_to_generate: list[dict] = []
    manual_count = 0

    for row in rows:
        sid = row["sense_id"]
        if sid in overrides:
            ov = overrides[sid]
            generations[sid] = GenerationResult(
                sense_id=sid,
                example_pt=ov["example_pt"],
                example_en=ov["example_en"],
                target_word_used=ov["target_word_used"] or row["pt"].lower(),
                method="manual_override",
            )
            manual_count += 1
        else:
            rows_to_generate.append(row)

    print(
        f"[info] Stage 4: input={len(rows)}, manual_override={manual_count}, "
        f"to_generate={len(rows_to_generate)}",
        file=sys.stderr,
    )

    if rows_to_generate:
        completed = 0
        progress_step = max(20, len(rows_to_generate) // 25)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_sid = {
                pool.submit(
                    _generate_one,
                    r,
                    client=anthropic_client,
                    system_prompt=generate_prompt,
                ): r["sense_id"]
                for r in rows_to_generate
            }
            for fut in as_completed(future_to_sid):
                gen = fut.result()
                generations[gen.sense_id] = gen
                completed += 1
                if completed % progress_step == 0:
                    print(
                        f"[info] Stage 4 generation: {completed}/{len(rows_to_generate)}",
                        file=sys.stderr,
                    )

    # --- Phase 2: Deterministic check ---------------------------------------

    det_results: dict[str, tuple[bool, bool]] = {}
    for row in rows:
        sid = row["sense_id"]
        gen = generations[sid]
        det_results[sid] = _check_deterministic(gen.target_word_used, gen.example_pt)

    # --- Phase 3: Cross-family validation -----------------------------------

    validations: dict[str, ValidationResult] = {}
    if skip_validator or openai_client is None:
        for row in rows:
            sid = row["sense_id"]
            validations[sid] = ValidationResult(
                sense_id=sid, status="skipped", reason="validator skipped"
            )
    else:
        rows_by_sid = {r["sense_id"]: r for r in rows}
        validate_targets = [
            (rows_by_sid[sid], generations[sid])
            for sid in generations
            if generations[sid].method != "manual_override"
            and generations[sid].example_pt
        ]
        print(
            f"[info] Stage 4 validator: {len(validate_targets)} examples to validate",
            file=sys.stderr,
        )
        if validate_targets:
            completed = 0
            progress_step = max(20, len(validate_targets) // 25)
            with ThreadPoolExecutor(max_workers=concurrency // 2 or 1) as pool:
                future_to_sid = {
                    pool.submit(
                        _validate_one,
                        row,
                        gen=gen,
                        client=openai_client,
                        system_prompt=validate_prompt,
                    ): row["sense_id"]
                    for (row, gen) in validate_targets
                }
                for fut in as_completed(future_to_sid):
                    val = fut.result()
                    validations[val.sense_id] = val
                    completed += 1
                    if completed % progress_step == 0:
                        print(
                            f"[info] Stage 4 validator: {completed}/{len(validate_targets)}",
                            file=sys.stderr,
                        )
        # Manual-override rows skip validator
        for row in rows:
            sid = row["sense_id"]
            if sid not in validations:
                validations[sid] = ValidationResult(
                    sense_id=sid,
                    status="pass",
                    reason="manual override (validator skipped)",
                )

    # --- Phase 4: Build output rows + fixes queue ---------------------------

    output_rows: list[dict] = []
    fixes_rows: list[dict] = []

    pass_count = 0
    fail_count = 0
    borderline_count = 0
    skipped_count = 0
    token_match_fail = 0
    word_count_fail = 0

    for row in rows:
        sid = row["sense_id"]
        gen = generations[sid]
        val = validations[sid]
        token_match, wc_ok = det_results[sid]

        if not token_match:
            token_match_fail += 1
        if not wc_ok:
            word_count_fail += 1

        # Track validator status counts
        if val.status == "pass":
            pass_count += 1
        elif val.status == "fail":
            fail_count += 1
        elif val.status == "borderline":
            borderline_count += 1
        else:
            skipped_count += 1

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
            "example_pt": gen.example_pt,
            "example_en": gen.example_en,
            "target_word_used": gen.target_word_used,
            "example_method": gen.method,
            "example_token_match": "pass" if token_match else "fail",
            "example_word_count_ok": "pass" if wc_ok else "fail",
            "example_validation_status": val.status,
            "example_validation_reason": val.reason,
            "example_policy": example_policy_for_row(row),
        }
        output_rows.append(out)

        # Fixes queue: any row that failed deterministic OR validator
        needs_fix = (
            (not token_match)
            or (not wc_ok)
            or (val.status == "fail")
            or (val.status == "skipped" and gen.method != "manual_override")
        )
        if needs_fix:
            failure_axes = list(val.failure_axes or [])
            if not token_match:
                failure_axes.append("token_match")
            if not wc_ok:
                failure_axes.append("word_count")
            if gen.error:
                failure_axes.append("generation_error")
            fixes_rows.append(
                {
                    "sense_id": sid,
                    "rank": row["rank"],
                    "pt": row["pt"],
                    "en_primary": row["en_primary"],
                    "example_pt": gen.example_pt,
                    "example_en": gen.example_en,
                    "target_word_used": gen.target_word_used,
                    "failure_reason": val.reason or gen.error or "deterministic check failed",
                    "failure_axes": ",".join(failure_axes),
                }
            )

    # --- Write outputs -------------------------------------------------------

    n_out = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)
    n_fix = write_tsv(fixes_path, fixes_rows, fieldnames=FIXES_FIELDS)

    summary = {
        "input_rows": len(rows),
        "output_rows": n_out,
        "fixes_rows": n_fix,
        "manual_overrides": manual_count,
        "generated_via_llm": sum(
            1 for g in generations.values() if g.method == "llm"
        ),
        "generation_failures": sum(
            1 for g in generations.values() if g.method == "fallback"
        ),
        "validator_pass": pass_count,
        "validator_fail": fail_count,
        "validator_borderline": borderline_count,
        "validator_skipped": skipped_count,
        "deterministic_token_match_fail": token_match_fail,
        "deterministic_word_count_fail": word_count_fail,
    }
    if anthropic_client is not None:
        summary["anthropic_cache_stats"] = anthropic_client.cache_stats
    if openai_client is not None and not skip_validator:
        summary["openai_stats"] = openai_client.stats

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 — example sentences")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--fixes", type=Path, default=FIXES_PATH)
    parser.add_argument(
        "--smoke-sample",
        type=Path,
        default=None,
        help="Path to a smoke-sample TSV; if given, restrict input to those sense_ids.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--skip-validator", action="store_true")
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="Combined with --smoke-sample, take first N sense_ids only (for quick checks).",
    )
    args = parser.parse_args()

    sense_id_filter: set[str] | None = None
    if args.smoke_sample:
        sample = read_tsv(args.smoke_sample)
        sense_id_filter = {r["sense_id"] for r in sample}
        if args.smoke_limit:
            ordered = [r["sense_id"] for r in sample][: args.smoke_limit]
            sense_id_filter = set(ordered)
        print(
            f"[info] smoke mode: filtering to {len(sense_id_filter)} sense_ids "
            f"from {args.smoke_sample.name}",
            file=sys.stderr,
        )

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        output_path=args.output,
        fixes_path=args.fixes,
        concurrency=args.concurrency,
        limit=args.limit,
        sense_id_filter=sense_id_filter,
        skip_validator=args.skip_validator,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
