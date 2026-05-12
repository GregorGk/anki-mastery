"""Stage 13 / Step 0 — Classify `family_root` for content-word rows.

For each row in `data/06-final.tsv` whose `pos` is in {noun, verb, adj, adv},
generate a list of candidate roots from the deck (via deterministic
suffix/prefix/seed heuristics), then ask Claude Sonnet 4.6 (via Anthropic
Tool Use + prompt caching) to pick one root or return empty.

The output is a sidecar TSV (`data/_family_roots.tsv`) that
`build/derive_final.py` joins at TSV-build time to populate the existing
empty `family_root` column.

Modes:
    --dry-run   Run candidate generation locally, print the report, exit.
                Includes a gate: aborts the (would-be) LLM run if
                candidate-rows > 3,000 or p95 candidate-set size > 25.
    (default)   Run the full classifier. Resumes from existing sidecar.

Usage:
    .venv/bin/python build/13_0_family_root_classifier.py --dry-run
    .venv/bin/python build/13_0_family_root_classifier.py --yes
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.family_root_rules import (  # noqa: E402
    CONTENT_POS,
    PtInfo,
    apply_post_process,
    build_pt_indexes,
    generate_candidates,
)
from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
PROMPTS_DIR = REPO_ROOT / "build" / "prompts"

FINAL_TSV = DATA_DIR / "06-final.tsv"
SIDECAR_TSV = DATA_DIR / "_family_roots.tsv"
PROMPT_PATH = PROMPTS_DIR / "family_root.md"
AUDIT_JSONL = AUDIT_DIR / "13_family_root.jsonl"
DRY_RUN_REPORT = AUDIT_DIR / "13_family_root_dryrun.txt"

MODEL_ID = "claude-sonnet-4-6"
DEFAULT_CONCURRENCY = 10

# Gate thresholds — if exceeded, the dry-run aborts the LLM run.
GATE_CANDIDATE_ROW_CEILING = 3000
GATE_P95_CANDIDATE_CEILING = 25

SIDECAR_FIELDS = [
    "sense_id", "pos", "pt", "en_primary",
    "family_root", "family_relation", "confidence", "reason",
    "source", "model_id", "generated_at",
]


TOOL_NAME = "record_family_root"
TOOL_DESCRIPTION = (
    "Record the family_root assignment for one BP sense. Return "
    "family_root='' whenever the candidate list doesn't contain a clear "
    "derivational head whose sense matches the target. False positives are "
    "expensive; default to empty when unsure."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "family_root": {
            "type": "string",
            "description": (
                "One pt from candidate_roots, or '' if no candidate fits. "
                "MUST be one of the listed lemmas; never invent."
            ),
        },
        "family_relation": {
            "type": "string",
            "enum": [
                "self", "inflected_or_participle", "action_noun", "agent_noun",
                "quality_noun", "adjective", "negative_form",
                "other_derivation", "uncertain", "unrelated",
            ],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": (
                "Confidence in the assignment itself. Only `high` ships to "
                "the final TSV; medium/low are dropped by the export filter."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence justifying the decision.",
        },
    },
    "required": ["family_root", "family_relation", "confidence", "reason"],
}


@dataclass
class Row:
    sense_id: str
    rank: int
    pt: str
    pos: str
    en_primary: str
    en_all: str
    annotation: str
    example_pt: str
    tags: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows() -> list[Row]:
    out: list[Row] = []
    for r in read_tsv(FINAL_TSV):
        try:
            rank = int(r.get("rank") or "0")
        except ValueError:
            rank = 0
        out.append(Row(
            sense_id=r["sense_id"],
            rank=rank,
            pt=(r.get("pt") or "").strip(),
            pos=(r.get("pos") or "").strip(),
            en_primary=(r.get("en_primary") or "").strip(),
            en_all=(r.get("en_all") or "").strip(),
            annotation=(r.get("annotation") or "").strip(),
            example_pt=(r.get("example_pt") or "").strip(),
            tags=(r.get("tags") or "").strip(),
        ))
    return out


def _read_existing_sidecar() -> dict[str, dict]:
    if not SIDECAR_TSV.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(SIDECAR_TSV)}


def _is_eligible(r: Row) -> bool:
    return r.pos in CONTENT_POS and bool(r.pt)


def _build_user_message(r: Row, candidates: list[PtInfo]) -> str:
    """Render the row + candidate list as a single user-message payload.

    Format keeps things terse and cache-friendly (system prompt is the
    cacheable bulk, user message is per-row tail)."""
    lines = [
        f"sense_id: {r.sense_id}",
        f"pt: {r.pt}",
        f"pos: {r.pos}",
        f"rank: {r.rank}",
        f"en_primary: {r.en_primary}",
        f"en_all: {r.en_all}",
        f"annotation: {r.annotation}",
        f"example_pt: {r.example_pt}",
        "candidate_roots:",
    ]
    for c in candidates:
        lines.append(
            f"  - {{pt: {c.pt}, pos: {c.pos}, rank: {c.rank}, "
            f"en_primary: {c.en_primary}}}"
        )
    return "\n".join(lines)


def _classify_one(
    *,
    client: AnthropicClient,
    system: str,
    row: Row,
    candidates: list[PtInfo],
    all_pts: set[str],
) -> dict:
    """One LLM call + idempotent post-processing. Returns dict ready for
    the sidecar."""
    user_msg = _build_user_message(row, candidates)
    decision = client.call_tool(
        system=system,
        user_message=user_msg,
        tool_name=TOOL_NAME,
        tool_input_schema=TOOL_INPUT_SCHEMA,
        tool_description=TOOL_DESCRIPTION,
        max_tokens=256,
        stage="13_0_family_root",
        provenance_key=row.sense_id,
    )
    payload = {
        "sense_id":        row.sense_id,
        "pt":              row.pt,
        "pos":              row.pos,
        "en_primary":       row.en_primary,
        "family_root":     (decision.get("family_root") or "").strip(),
        "family_relation": (decision.get("family_relation") or "").strip(),
        "confidence":      (decision.get("confidence") or "low").strip().lower(),
        "reason":          (decision.get("reason") or "").strip().replace("\n", " "),
        "source":          "llm",
    }
    return apply_post_process(payload, all_pts)


def _write_sidecar_row(row_out: dict, fh, writer_lock: threading.Lock) -> None:
    with writer_lock:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        # Coerce all values to strings (csv chokes on None).
        sanitized = {k: ("" if v is None else str(v)) for k, v in row_out.items()}
        w.writerow(sanitized)
        fh.flush()


def _open_sidecar_appender(existing: dict[str, dict]):
    SIDECAR_TSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not SIDECAR_TSV.exists() or SIDECAR_TSV.stat().st_size == 0
    fh = SIDECAR_TSV.open("a", encoding="utf-8", newline="")
    if is_new:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        w.writeheader()
        fh.flush()
    return fh


# --------------------------------------------------------------------------- #
# Dry-run report
# --------------------------------------------------------------------------- #


def _format_dry_run_report(
    rows: list[Row],
    eligible: list[Row],
    per_row_candidates: dict[str, list[PtInfo]],
) -> tuple[str, dict]:
    """Build the human-readable dry-run report. Returns (text, stats_dict)."""
    candidate_rows = [r for r in eligible if per_row_candidates.get(r.sense_id)]
    skipped_rows = [r for r in eligible if not per_row_candidates.get(r.sense_id)]

    cand_sizes = [len(per_row_candidates[r.sense_id]) for r in candidate_rows]
    if cand_sizes:
        median = statistics.median(cand_sizes)
        p75 = statistics.quantiles(cand_sizes, n=4)[2] if len(cand_sizes) >= 4 else max(cand_sizes)
        p95_idx = max(0, int(round(len(cand_sizes) * 0.95)) - 1)
        sorted_sizes = sorted(cand_sizes)
        p95 = sorted_sizes[min(p95_idx, len(sorted_sizes) - 1)]
    else:
        median = p75 = p95 = 0

    by_pos: dict[str, int] = {}
    for r in eligible:
        by_pos[r.pos] = by_pos.get(r.pos, 0) + 1

    lines = [
        "================================================================",
        " STAGE 13 — FAMILY ROOT CLASSIFIER DRY RUN",
        "================================================================",
        f"  total rows in 06-final.tsv: {len(rows)}",
        f"  content-word rows (eligible):   {len(eligible)}",
        f"    by pos: " + ", ".join(f"{p}={n}" for p, n in sorted(by_pos.items())),
        f"  rows with ≥1 candidate:         {len(candidate_rows)}",
        f"  rows skipped (no candidates):   {len(skipped_rows)}",
        "",
        f"  candidate-set size: median={median}  p75={p75}  p95={p95}",
        f"  GATE: candidate-rows ceiling = {GATE_CANDIDATE_ROW_CEILING}",
        f"  GATE: p95 ceiling = {GATE_P95_CANDIDATE_CEILING}",
    ]

    # Top 50 largest candidate sets.
    by_size = sorted(
        ((len(per_row_candidates[r.sense_id]), r) for r in candidate_rows),
        key=lambda x: (-x[0], x[1].sense_id),
    )
    lines.append("")
    lines.append("  --- top 50 largest candidate sets ---")
    for n, r in by_size[:50]:
        cand_pts = [c.pt for c in per_row_candidates[r.sense_id]]
        sample = ", ".join(cand_pts[:8])
        more = f", +{len(cand_pts) - 8} more" if len(cand_pts) > 8 else ""
        lines.append(f"    [{n:>2}] {r.sense_id} {r.pt:<24} [{r.pos}] → {sample}{more}")

    # 50 random samples.
    rng = random.Random(42)
    sample = rng.sample(candidate_rows, min(50, len(candidate_rows)))
    sample.sort(key=lambda r: r.sense_id)
    lines.append("")
    lines.append("  --- 50 random candidate sets ---")
    for r in sample:
        cand_pts = [c.pt for c in per_row_candidates[r.sense_id]]
        sample_str = ", ".join(cand_pts[:6])
        more = f", +{len(cand_pts) - 6} more" if len(cand_pts) > 6 else ""
        lines.append(f"    {r.sense_id} {r.pt:<24} [{r.pos}] → {sample_str}{more}")

    stats = {
        "total_rows": len(rows),
        "eligible": len(eligible),
        "candidate_rows": len(candidate_rows),
        "skipped_rows": len(skipped_rows),
        "median": median,
        "p75": p75,
        "p95": p95,
    }
    return "\n".join(lines), stats


def _gate_decision(stats: dict) -> tuple[bool, list[str]]:
    """Apply the dry-run gate. Returns (ok, reasons)."""
    reasons: list[str] = []
    if stats["candidate_rows"] > GATE_CANDIDATE_ROW_CEILING:
        reasons.append(
            f"candidate-rows {stats['candidate_rows']} > ceiling "
            f"{GATE_CANDIDATE_ROW_CEILING} — tighten the generator"
        )
    if stats["p95"] > GATE_P95_CANDIDATE_CEILING:
        reasons.append(
            f"p95 candidate-set size {stats['p95']} > ceiling "
            f"{GATE_P95_CANDIDATE_CEILING} — tighten the generator"
        )
    return (not reasons, reasons)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="Run candidate generation + report; no LLM calls.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--yes", action="store_true", help="Skip GO prompt")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only classify this many rows (for piloting).")
    args = p.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    if not PROMPT_PATH.exists():
        print(f"ERROR: prompt missing at {PROMPT_PATH}", file=sys.stderr)
        return 1
    system = PROMPT_PATH.read_text(encoding="utf-8")

    # Load every row (for the pt index), then filter to eligible content-words.
    all_rows_raw = list(read_tsv(FINAL_TSV))
    pt_to_info, norm_to_pts = build_pt_indexes(all_rows_raw)
    rows = _load_rows()
    eligible = [r for r in rows if _is_eligible(r)]

    # Candidate generation for every eligible row.
    per_row_candidates: dict[str, list[PtInfo]] = {}
    for r in eligible:
        cand_pts = generate_candidates(r.pt, r.pos, pt_to_info, norm_to_pts)
        cand_infos = [pt_to_info[pt] for pt in cand_pts if pt in pt_to_info]
        if cand_infos:
            per_row_candidates[r.sense_id] = cand_infos

    # Dry-run report (also printed in non-dry-run mode as pre-flight info).
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report, stats = _format_dry_run_report(rows, eligible, per_row_candidates)
    DRY_RUN_REPORT.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nReport written to: {DRY_RUN_REPORT}")

    gate_ok, gate_reasons = _gate_decision(stats)
    if not gate_ok:
        print("\n!!! GATE FAILED !!!")
        for reason in gate_reasons:
            print(f"  - {reason}")
        if not args.dry_run:
            print("\nAborting LLM run. Tighten the candidate generator and re-try.")
            return 1
    else:
        print("\n✓ Gate check passed.")

    if args.dry_run:
        return 0 if gate_ok else 1

    existing = _read_existing_sidecar()
    candidate_rows = [
        r for r in eligible
        if r.sense_id in per_row_candidates and r.sense_id not in existing
    ]
    if args.limit:
        candidate_rows = candidate_rows[: args.limit]

    n = len(candidate_rows)
    skip_pre = len(per_row_candidates) - n - (
        len(per_row_candidates) - len(candidate_rows) - (len(existing) & 0)
    )
    print()
    print(f"  Already in sidecar: {len(existing)}")
    print(f"  To classify:        {n}")
    print(f"  Concurrency:        {args.concurrency}")
    print(f"  Model:              {args.model}")

    if n == 0:
        print("\nNothing to do — all eligible senses already in sidecar.")
        return 0

    # Cost estimate (mirrors Stage 12 calculation)
    cache_create_in = 3500
    cache_read_in = max(0, n - 1) * 3500
    user_in = n * 300  # slightly larger than stage 12 because of candidate list
    out_t = n * 80
    est_cost = (
        cache_create_in / 1_000_000 * 3.75
        + cache_read_in / 1_000_000 * 0.30
        + user_in / 1_000_000 * 3.0
        + out_t / 1_000_000 * 15.0
    )
    print(f"  Cost (est):         ${est_cost:.2f}")
    print(f"  Sidecar:            {SIDECAR_TSV}")
    print(f"  Audit:              {AUDIT_JSONL}")

    if not args.yes:
        sys.stdout.write("\nType GO to proceed with the LLM run: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # Run.
    all_pts: set[str] = set(pt_to_info.keys())
    client = AnthropicClient(
        model=args.model,
        audit_path=AUDIT_JSONL,
        enable_caching=True,
    )
    fh = _open_sidecar_appender(existing)
    writer_lock = threading.Lock()

    def _worker(r: Row) -> tuple[Row, dict | None, str]:
        try:
            d = _classify_one(
                client=client, system=system, row=r,
                candidates=per_row_candidates[r.sense_id], all_pts=all_pts,
            )
            return r, d, ""
        except Exception as exc:  # noqa: BLE001
            return r, None, f"{type(exc).__name__}: {exc}"

    t_start = time.time()
    n_done = 0
    n_root = 0
    n_empty = 0
    n_failed = 0
    print()
    print(f"Processing {n} rows at concurrency={args.concurrency}…")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_worker, r): r for r in candidate_rows}
            for fut in as_completed(futs):
                row, decision, err = fut.result()
                n_done += 1
                if err:
                    n_failed += 1
                    print(f"[{n_done:>4}/{n}] FAIL {row.sense_id} {row.pt}  {err}",
                          flush=True)
                    continue
                root = decision.get("family_root") or ""
                conf = decision.get("confidence") or ""
                relation = decision.get("family_relation") or ""
                tag = " ROOT" if root else "     "
                print(f"[{n_done:>4}/{n}]{tag} {row.sense_id} {row.pt:<20} → "
                      f"{root or '(empty)':<18} [{conf}/{relation}]",
                      flush=True)
                if root:
                    n_root += 1
                else:
                    n_empty += 1
                _write_sidecar_row({
                    "sense_id":        row.sense_id,
                    "pos":             row.pos,
                    "pt":              row.pt,
                    "en_primary":      row.en_primary,
                    "family_root":     decision.get("family_root", ""),
                    "family_relation": decision.get("family_relation", ""),
                    "confidence":      decision.get("confidence", ""),
                    "reason":          decision.get("reason", ""),
                    "source":          decision.get("source", "llm"),
                    "model_id":        args.model,
                    "generated_at":    _now_iso(),
                }, fh, writer_lock)
    finally:
        fh.close()

    elapsed = time.time() - t_start
    stats_cache = client.cache_stats

    print()
    print("=== Stage 13.0 summary ===")
    print(f"  Done:              {n_done}")
    print(f"  With root:         {n_root} ({100*n_root/max(n_done,1):.0f}%)")
    print(f"  Empty:             {n_empty}")
    print(f"  Failed:            {n_failed}")
    print(f"  Wall:              {elapsed:.1f}s ({elapsed/max(n_done,1):.2f}s/row)")
    print(f"  Cache hit:         {stats_cache['cache_hit_ratio']:.0%}  "
          f"(create={stats_cache['cache_creation_tokens']}, "
          f"read={stats_cache['cache_read_tokens']})")
    print(f"  Sidecar:           {SIDECAR_TSV}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
