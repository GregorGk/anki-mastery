"""Stage 4.5 — Speaker gender classifier + voice assignment.

Reads:
    data/04-examples.tsv                  (Stage 4 output)
    config/voices.tsv                     (4F + 7M voice pool, committed)
    data/_manual_speaker_gender.tsv       (manual overrides; wins over LLM)

Writes:
    data/045-speaker_gender.tsv           (per-sense gender + voice_id)
    audit/045_speaker_gender.jsonl        (LLM provenance)

Pipeline:
    Phase 1: For each row in 04-examples.tsv, OpenAI gpt-4o-mini Tool Use
             returns {speaker_gender, evidence, confidence}.
    Phase 2a: For neutrals, balanced shuffle (seed=44) → voice_gender_assigned.
    Phase 2b: Within each gender pool, round-robin assignment (seeds 45/46)
              → voice_id from config/voices.tsv.

Hard invariants:
    - Every Stage-4 sense_id has exactly one row in 045-speaker_gender.tsv.
    - voice_gender_assigned ∈ {male, female} (no neutrals after Phase 2a).
    - voice_id is non-empty and resolves in config/voices.tsv.
    - Per-voice usage is off-by-at-most-1 within each gender pool.
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

from build.lib.llm import OpenAIClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.lib.voices import (  # noqa: E402
    SEED_FEMALE_POOL,
    SEED_MALE_POOL,
    SEED_NEUTRAL_BALANCE,
    Voice,
    assign_voice_ids,
    load_voices,
    resolve_neutrals,
    split_by_gender,
    usage_counts,
    verify_balanced_within_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
CONFIG_DIR = REPO_ROOT / "config"

INPUT_PATH = DATA_DIR / "04-examples.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_speaker_gender.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"
OUTPUT_PATH = DATA_DIR / "045-speaker_gender.tsv"
AUDIT_PATH = AUDIT_DIR / "045_speaker_gender.jsonl"

OUTPUT_FIELDS = [
    "sense_id",
    "speaker_gender",          # raw classifier output: male/female/neutral
    "evidence",                # one-phrase cue
    "confidence",              # high/medium/low
    "voice_gender_assigned",   # post-Phase-2a: male/female
    "voice_id",                # post-Phase-2b: specific voice from pool
    "assignment_method",       # 'llm' | 'manual_override' | 'fallback'
]

DEFAULT_CONCURRENCY = 8

CLASSIFIER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker_gender": {
            "type": "string",
            "enum": ["male", "female", "neutral"],
            "description": "The most likely speaker gender for this sentence.",
        },
        "evidence": {
            "type": "string",
            "description": "One short phrase (≤80 chars) citing the cue.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["speaker_gender", "evidence", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "speaker_gender.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Classify the most likely speaker gender of a Brazilian-Portuguese "
        "example sentence. Use Tool Use."
    )


def _load_overrides(path: Path) -> dict[str, dict[str, str]]:
    """Map sense_id -> override dict.

    Override columns: speaker_gender_override, voice_gender_override, voice_id_override.
    Any non-empty field wins over the LLM/Phase 2 logic.
    """
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "speaker_gender": (r.get("speaker_gender_override") or "").strip().lower(),
            "voice_gender": (r.get("voice_gender_override") or "").strip().lower(),
            "voice_id": (r.get("voice_id_override") or "").strip(),
            "notes": (r.get("notes") or "").strip(),
        }
    return out


def _build_user_message(row: dict) -> str:
    return (
        f"pt: {row.get('pt', '')}\n"
        f"en_primary: {row.get('en_primary', '')}\n"
        f"example_pt: {row.get('example_pt', '')}\n"
        f"example_en: {row.get('example_en', '')}"
    )


@dataclass
class ClassificationResult:
    sense_id: str
    speaker_gender: str  # male/female/neutral
    evidence: str
    confidence: str
    method: str  # 'llm' | 'manual_override' | 'fallback'


def _classify_one(
    row: dict, *, client: OpenAIClient, system_prompt: str
) -> ClassificationResult:
    sid = row["sense_id"]
    user_msg = _build_user_message(row)
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="classify_speaker_gender",
            tool_description=(
                "Classify the most likely speaker gender of a Brazilian-Portuguese "
                "example sentence as male, female, or neutral, citing evidence."
            ),
            tool_input_schema=CLASSIFIER_TOOL_SCHEMA,
            stage="4_5",
            provenance_key=sid,
            max_tokens=256,
        )
    except Exception as exc:
        return ClassificationResult(
            sense_id=sid,
            speaker_gender="neutral",
            evidence=f"classifier error: {type(exc).__name__}",
            confidence="low",
            method="fallback",
        )

    sg = (decision.get("speaker_gender") or "neutral").strip().lower()
    if sg not in ("male", "female", "neutral"):
        sg = "neutral"
    return ClassificationResult(
        sense_id=sid,
        speaker_gender=sg,
        evidence=(decision.get("evidence") or "").strip()[:200],
        confidence=(decision.get("confidence") or "medium").strip().lower(),
        method="llm",
    )


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    voices_path: Path = VOICES_PATH,
    output_path: Path = OUTPUT_PATH,
    audit_path: Path = AUDIT_PATH,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    sense_id_filter: set[str] | None = None,
    openai_client: OpenAIClient | None = None,
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

    voices = load_voices(voices_path)
    female_voices, male_voices = split_by_gender(voices)
    overrides = _load_overrides(overrides_path)
    system_prompt = _load_prompt()

    if openai_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        openai_client = OpenAIClient(audit_path=audit_path)

    # --- Phase 1: classify ---------------------------------------------------

    classifications: dict[str, ClassificationResult] = {}
    rows_to_classify: list[dict] = []
    manual_count = 0

    for row in rows:
        sid = row["sense_id"]
        ov = overrides.get(sid, {})
        if ov.get("speaker_gender"):
            classifications[sid] = ClassificationResult(
                sense_id=sid,
                speaker_gender=ov["speaker_gender"],
                evidence="manual override",
                confidence="high",
                method="manual_override",
            )
            manual_count += 1
        else:
            rows_to_classify.append(row)

    print(
        f"[info] Stage 4.5: input={len(rows)}, manual_override={manual_count}, "
        f"to_classify={len(rows_to_classify)}",
        file=sys.stderr,
    )

    if rows_to_classify:
        completed = 0
        progress_step = max(50, len(rows_to_classify) // 25)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_sid = {
                pool.submit(
                    _classify_one,
                    r,
                    client=openai_client,
                    system_prompt=system_prompt,
                ): r["sense_id"]
                for r in rows_to_classify
            }
            for fut in as_completed(future_to_sid):
                cls = fut.result()
                classifications[cls.sense_id] = cls
                completed += 1
                if completed % progress_step == 0:
                    print(
                        f"[info] Stage 4.5 classification: {completed}/{len(rows_to_classify)}",
                        file=sys.stderr,
                    )

    # --- Phase 2a: balanced gender resolution --------------------------------

    raw_gender = {sid: cls.speaker_gender for sid, cls in classifications.items()}

    # Apply explicit voice_gender overrides (skip Phase 2a balancing for these)
    for sid, ov in overrides.items():
        if sid in raw_gender and ov.get("voice_gender"):
            raw_gender[sid] = ov["voice_gender"]

    voice_gender_assigned = resolve_neutrals(raw_gender, seed=SEED_NEUTRAL_BALANCE)

    # --- Phase 2b: per-voice round-robin -------------------------------------

    voice_id_for = assign_voice_ids(
        voice_gender_assigned,
        voices,
        female_seed=SEED_FEMALE_POOL,
        male_seed=SEED_MALE_POOL,
    )

    # Apply explicit voice_id overrides last (full bypass)
    for sid, ov in overrides.items():
        if sid in voice_id_for and ov.get("voice_id"):
            voice_id_for[sid] = ov["voice_id"]

    # --- Build output rows ---------------------------------------------------

    output_rows: list[dict] = []
    for row in rows:
        sid = row["sense_id"]
        cls = classifications[sid]
        out = {
            "sense_id": sid,
            "speaker_gender": cls.speaker_gender,
            "evidence": cls.evidence,
            "confidence": cls.confidence,
            "voice_gender_assigned": voice_gender_assigned[sid],
            "voice_id": voice_id_for[sid],
            "assignment_method": cls.method,
        }
        output_rows.append(out)

    # --- Write + invariants --------------------------------------------------

    n_out = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)

    # Verify invariants
    valid_voice_ids = {v.voice_id for v in voices}
    missing_voice_ids = [
        r["sense_id"] for r in output_rows if r["voice_id"] not in valid_voice_ids
    ]
    if missing_voice_ids:
        raise AssertionError(
            f"{len(missing_voice_ids)} rows have voice_id not in pool: "
            f"{missing_voice_ids[:5]}"
        )

    bad_gender = [
        r["sense_id"]
        for r in output_rows
        if r["voice_gender_assigned"] not in ("male", "female")
    ]
    if bad_gender:
        raise AssertionError(
            f"{len(bad_gender)} rows have invalid voice_gender_assigned: "
            f"{bad_gender[:5]}"
        )

    # Per-voice usage balance
    is_balanced, balance_msg = verify_balanced_within_pool(voice_id_for, voices)
    if not is_balanced:
        print(
            f"[warn] per-voice usage NOT balanced: {balance_msg}", file=sys.stderr
        )

    # --- Summary -------------------------------------------------------------

    from collections import Counter

    raw_gender_counts = Counter(r["speaker_gender"] for r in output_rows)
    assigned_gender_counts = Counter(r["voice_gender_assigned"] for r in output_rows)
    confidence_counts = Counter(r["confidence"] for r in output_rows)
    voice_usage = usage_counts(voice_id_for)

    summary = {
        "input_rows": len(rows),
        "output_rows": n_out,
        "manual_overrides": manual_count,
        "raw_gender_counts": dict(raw_gender_counts),
        "voice_gender_assigned_counts": dict(assigned_gender_counts),
        "confidence_counts": dict(confidence_counts),
        "voice_pool_size": {"female": len(female_voices), "male": len(male_voices)},
        "voice_usage": voice_usage,
        "balanced": is_balanced,
        "balance_msg": balance_msg,
    }
    if openai_client is not None:
        summary["openai_stats"] = openai_client.stats
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4.5 — speaker gender classifier + voice assignment"
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--voices", type=Path, default=VOICES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--smoke-sample",
        type=Path,
        default=None,
        help="Path to a smoke-sample TSV; if given, restrict input to those sense_ids.",
    )
    args = parser.parse_args()

    sense_id_filter: set[str] | None = None
    if args.smoke_sample:
        sample = read_tsv(args.smoke_sample)
        sense_id_filter = {r["sense_id"] for r in sample}
        print(
            f"[info] smoke mode: filtering to {len(sense_id_filter)} sense_ids "
            f"from {args.smoke_sample.name}",
            file=sys.stderr,
        )

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        voices_path=args.voices,
        output_path=args.output,
        audit_path=args.audit,
        concurrency=args.concurrency,
        limit=args.limit,
        sense_id_filter=sense_id_filter,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
