"""Stage 3 — Enrichment (gender + PoS + cognate flag + tags + pt_display).

Reads:
    data/02-senses.tsv                    (Stage 2 output; 5720 senses)
    data/_manual_gender.tsv               (manual override; wins over LLM)

Writes:
    data/03-enriched.tsv                  (full enriched senses, ready for Stage 4)
    audit/03_enrich.jsonl                 (LLM provenance)

Behaviour (locked decisions in plan v3):
    1. Apply deterministic shortcuts first (gender from forced splits, PoS
       from "to X" / function-word table / numerals / reflexives).
    2. Single LLM call per row to fill gender + pos + is_cognate_en. Hints
       passed in to bias the model toward already-resolved values.
    3. Compose pt_display from pt + gender (article-prefixed for nouns).
    4. Derive tags deterministically from existing fields + LLM output.
    5. Family root: schema column kept but left empty (deferred per plan v3).

Tier: default (Sonnet 4.5) only. No premium tier on this stage.
Concurrency: ThreadPoolExecutor, 16 workers (matches Stage 2).
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

from build.lib.enrich import (  # noqa: E402
    compose_pt_display,
    derive_tags,
    deterministic_gender,
    deterministic_pos,
)
from build.lib.llm import TIER_DEFAULT, AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "02-senses.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_gender.tsv"
OUTPUT_PATH = DATA_DIR / "03-enriched.tsv"
AUDIT_PATH = AUDIT_DIR / "03_enrich.jsonl"

DEFAULT_CONCURRENCY = 16

OUTPUT_FIELDS = [
    "sense_id",
    "source_line_number",
    "rank",
    "expansion_index",
    "sense_index",
    "source_pt",
    "pt",
    "pt_display",
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
    "split_method",
    "is_cognate_en",
    "family_root",  # left empty in this run (deferred)
    "enrich_method",  # 'llm' | 'deterministic_only' | 'manual_override' | 'fallback'
    "enrich_confidence",
    "enrich_reason",
    "tags",
    "source_line",
]

VALID_GENDERS = {"o", "a", "o/a", ""}
VALID_POS = {
    "noun",
    "verb",
    "adj",
    "adv",
    "prep",
    "conj",
    "pron",
    "art",
    "num",
    "interj",
    "idiom",
    "",
}

ENRICH_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "gender": {
            "type": "string",
            "enum": ["o", "a", "o/a", ""],
        },
        "pos": {
            "type": "string",
            "enum": [
                "noun",
                "verb",
                "adj",
                "adv",
                "prep",
                "conj",
                "pron",
                "art",
                "num",
                "interj",
                "",
            ],
        },
        "is_cognate_en": {"type": "boolean"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {"type": "string"},
    },
    "required": ["gender", "pos", "is_cognate_en", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "enrich.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Classify gender, pos, and cognate flag for a BP headword. "
        "Use Tool Use exclusively."
    )


def _load_manual_overrides(path: Path) -> dict[str, dict]:
    """Map sense_id -> {gender, pos, is_cognate_en} dict."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        gender = (r.get("gender") or "").strip()
        pos = (r.get("pos") or "").strip()
        cognate = (r.get("is_cognate_en") or "").strip().lower()
        if gender and gender not in VALID_GENDERS:
            print(
                f"[warn] manual override {sid}: invalid gender {gender!r}; skipping",
                file=sys.stderr,
            )
            continue
        out[sid] = {
            "gender": gender,
            "pos": pos if pos in VALID_POS else "",
            "is_cognate_en": cognate in ("true", "1", "yes"),
        }
    return out


@dataclass
class EnrichResult:
    gender: str
    pos: str
    is_cognate_en: bool
    confidence: str
    reason: str
    method: str  # 'llm' | 'deterministic_only' | 'manual_override' | 'fallback'


def _build_user_message(row: dict, hint_gender: str, hint_pos: str) -> str:
    return (
        f"pt: {row.get('pt', '')}\n"
        f"en_primary: {row.get('en_primary', '')}\n"
        f"en_all: {row.get('en_all', '')}\n"
        f"pt_type: {row.get('pt_type', '')}\n"
        f"bp_status: {row.get('bp_status', '')}\n"
        f"hint_gender: {hint_gender}\n"
        f"hint_pos: {hint_pos}\n"
    )


def _llm_enrich(
    row: dict,
    hint_gender: str,
    hint_pos: str,
    *,
    client: AnthropicClient,
    system_prompt: str,
) -> EnrichResult:
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=_build_user_message(row, hint_gender, hint_pos),
            tool_name="enrich_sense",
            tool_description=(
                "Classify gender, part of speech, and English-cognate flag "
                "for a Brazilian-Portuguese sense."
            ),
            tool_input_schema=ENRICH_TOOL_SCHEMA,
            stage="3",
            provenance_key=row.get("sense_id", ""),
            tier=TIER_DEFAULT,
        )
    except Exception as exc:
        print(
            f"[warn] {row.get('sense_id')}: LLM error "
            f"{type(exc).__name__}: {exc}; falling back to hints",
            file=sys.stderr,
        )
        return EnrichResult(
            gender=hint_gender,
            pos=hint_pos,
            is_cognate_en=False,
            confidence="low",
            reason=f"LLM error fallback: {type(exc).__name__}",
            method="fallback",
        )

    gender = decision.get("gender", "")
    pos = decision.get("pos", "")
    if gender not in VALID_GENDERS:
        gender = hint_gender
    if pos not in VALID_POS:
        pos = hint_pos
    return EnrichResult(
        gender=gender,
        pos=pos,
        is_cognate_en=bool(decision.get("is_cognate_en", False)),
        confidence=decision.get("confidence", "medium"),
        reason=decision.get("reason", ""),
        method="llm",
    )


def run(
    *,
    input_path: Path = INPUT_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    output_path: Path = OUTPUT_PATH,
    audit_path: Path = AUDIT_PATH,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    client: AnthropicClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 2 first."
        )

    rows = read_tsv(input_path)
    if limit is not None:
        rows = rows[:limit]

    overrides = _load_manual_overrides(overrides_path)
    system_prompt = _load_prompt()

    # Pre-compute deterministic hints
    hints: list[tuple[str, str]] = []
    for r in rows:
        hints.append((deterministic_gender(r), deterministic_pos(r)))

    # Pre-allocate results
    results: dict[int, EnrichResult] = {}

    # Apply manual overrides
    for i, r in enumerate(rows):
        sid = r.get("sense_id", "")
        if sid in overrides:
            ov = overrides[sid]
            results[i] = EnrichResult(
                gender=ov["gender"],
                pos=ov["pos"],
                is_cognate_en=ov["is_cognate_en"],
                confidence="high",
                reason="manual override",
                method="manual_override",
            )

    # Set up LLM client if needed
    llm_indices = [i for i in range(len(rows)) if i not in results]
    if llm_indices:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Stage 3 requires an Anthropic API key."
                )
            client = AnthropicClient(audit_path=audit_path)

        print(
            f"[info] Stage 3: enriching {len(llm_indices)} rows via LLM "
            f"({len(overrides)} manual overrides) with concurrency={concurrency}",
            file=sys.stderr,
        )

        completed = 0
        progress_step = max(50, len(llm_indices) // 20)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(
                    _llm_enrich,
                    rows[i],
                    hints[i][0],
                    hints[i][1],
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
                        f"[info] Stage 3: {completed}/{len(llm_indices)} done",
                        file=sys.stderr,
                    )

    # Build output rows
    output_rows: list[dict] = []
    method_counts: dict[str, int] = {"llm": 0, "manual_override": 0, "fallback": 0}
    pos_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    cognate_count = 0

    for i, row in enumerate(rows):
        result = results[i]
        method_counts[result.method] = method_counts.get(result.method, 0) + 1
        pos_counts[result.pos] = pos_counts.get(result.pos, 0) + 1
        gender_counts[result.gender] = gender_counts.get(result.gender, 0) + 1
        if result.is_cognate_en:
            cognate_count += 1

        gender = result.gender
        pos = result.pos
        pt = row.get("pt", "")
        pt_type = row.get("pt_type", "")
        pt_display = compose_pt_display(pt, gender, pt_type)

        # Tags merged from row + LLM output
        # First, any prior tags from upstream stages (none in 02-senses.tsv schema currently)
        tags = derive_tags(row, gender=gender, pos=pos, cognate_en=result.is_cognate_en)

        out = dict(row)  # carry through everything from Stage 2
        out["pt_display"] = pt_display
        out["gender"] = gender
        out["pos"] = pos
        out["is_cognate_en"] = "true" if result.is_cognate_en else "false"
        out["family_root"] = ""  # deferred per plan
        out["enrich_method"] = result.method
        out["enrich_confidence"] = result.confidence
        out["enrich_reason"] = result.reason
        out["tags"] = tags
        # Ensure all OUTPUT_FIELDS present
        for f in OUTPUT_FIELDS:
            out.setdefault(f, "")
        output_rows.append(out)

    # Write
    out_count = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)

    # --- Hard invariants ---
    seen_ids: set[str] = set()
    nouns_missing_gender: list[str] = []
    for r in output_rows:
        sid = r["sense_id"]
        assert sid not in seen_ids, f"duplicate sense_id {sid}"
        seen_ids.add(sid)
        assert r["pt_display"], f"empty pt_display at {sid}"
        # Soft check: nouns SHOULD have gender, but if the LLM returned
        # pos="noun" with empty gender, log it for manual review and continue.
        # These rows can be patched via _manual_gender.tsv on a re-run.
        if r["pos"] == "noun" and r["gender"] not in {"o", "a", "o/a"}:
            nouns_missing_gender.append(f"{sid} ({r['pt']})")
    if nouns_missing_gender:
        print(
            f"[warn] {len(nouns_missing_gender)} nouns have empty gender "
            f"(eligible for _manual_gender.tsv override). First 10: "
            f"{nouns_missing_gender[:10]}",
            file=sys.stderr,
        )

    summary = {
        "input_rows": len(rows),
        "output_rows": out_count,
        "manual_overrides_used": method_counts.get("manual_override", 0),
        "llm_calls": method_counts.get("llm", 0),
        "fallback_calls": method_counts.get("fallback", 0),
        "pos_counts": pos_counts,
        "gender_counts": gender_counts,
        "cognate_count": cognate_count,
    }
    if client is not None:
        summary["cache_stats"] = client.cache_stats
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 — enrichment")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (smoke test)",
    )
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        overrides_path=args.overrides,
        output_path=args.output,
        audit_path=args.audit,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
