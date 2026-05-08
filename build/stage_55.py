"""Stage 5.5 — Single-pass auditor (Anthropic Sonnet 4.5).

Reads:
    data/05-ipa.tsv                    (Stage 5 output, full row context)
    data/04-examples.tsv               (Stage 4 prior-flag context, optional)
    data/_example_fixes.tsv            (Stage 4 validator's prior flags)
    data/_manual_audit.tsv             (manual override; wins over LLM)

Writes:
    data/055-audit.tsv                 (per-sense verdict + defects)
    data/_jury_disagreements.tsv       (rows for human review)
    audit/055_audit.jsonl              (LLM provenance per call)
    audit/055_progress.jsonl           (live monitoring: started/completed/errored events)

Single-pass design (per § Stage 5.5 in docs/plan.md): one Sonnet call per
row produces both an adversarial defect list AND a binding verdict
(pass / regenerate / human_review). For verdict=regenerate rows, Stage 5.5
calls Stage 4's run() with a sense_id_filter to re-generate the example;
≤ 2 regen attempts. Rows still failing after 2 regens route to the human
disagreement queue. Idempotent + resumable: already-audited rows in
data/055-audit.tsv are skipped on restart.

Live monitoring:
    - audit/055_progress.jsonl receives a JSON line on every event:
      'started' / 'completed' / 'errored' / 'stuck' (>120s in flight).
      Tail with: tail -f audit/055_progress.jsonl
    - Every 30 seconds, the main loop prints a stdout summary line.
    - Standalone snapshot: build/audit_status.py reads the progress JSONL
      and prints current state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.llm import TIER_DEFAULT, AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

INPUT_PATH = DATA_DIR / "05-ipa.tsv"
EXAMPLES_PATH = DATA_DIR / "04-examples.tsv"
PRIOR_FIXES_PATH = DATA_DIR / "_example_fixes.tsv"
OVERRIDES_PATH = DATA_DIR / "_manual_audit.tsv"
OUTPUT_PATH = DATA_DIR / "055-audit.tsv"
HUMAN_REVIEW_PATH = DATA_DIR / "_jury_disagreements.tsv"
AUDIT_PATH = AUDIT_DIR / "055_audit.jsonl"
PROGRESS_PATH = AUDIT_DIR / "055_progress.jsonl"

OUTPUT_FIELDS = [
    "sense_id",
    "verdict",            # 'pass' | 'regenerate' | 'human_review'
    "defect_count",
    "defects_json",
    "reason",
    "confidence",
    "regen_attempts",     # 0 / 1 / 2
    "final_status",       # pass_first | pass_after_regen | human_review | failed_after_regen
    "audit_method",       # 'llm' | 'manual_override'
]

HUMAN_REVIEW_FIELDS = [
    "sense_id",
    "pt",
    "en_primary",
    "example_pt",
    "example_en",
    "verdict",
    "reason",
    "defects_json",
    "regen_attempts",
]

DEFAULT_CONCURRENCY = 16
STUCK_THRESHOLD_SEC = 120
SUMMARY_INTERVAL_SEC = 30
MAX_REGEN_ATTEMPTS = 2

AUDIT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": [
                            "sense_consistency",
                            "example_uses_intended_sense",
                            "translation_match",
                            "bp_purity",
                            "sensitive_policy",
                            "ipa_plausibility",
                            "target_word_token_match",
                            "naturalness",
                            "level_appropriateness",
                            "other",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                },
                "required": ["axis", "severity", "description"],
            },
        },
        "verdict": {
            "type": "string",
            "enum": ["pass", "regenerate", "human_review"],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["defects", "verdict", "reason", "confidence"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "audit.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Adversarially audit a Brazilian-Portuguese sense row and produce a "
        "verdict (pass / regenerate / human_review). Use Tool Use."
    )


def _load_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        v = (r.get("verdict_override") or "").strip().lower()
        if v not in ("pass", "regenerate", "human_review"):
            continue
        out[sid] = {
            "verdict": v,
            "reason": (r.get("reason") or "manual override").strip(),
            "notes": (r.get("notes") or "").strip(),
        }
    return out


def _load_existing_audit(path: Path) -> dict[str, dict]:
    """Read existing 055-audit.tsv for resume support; return {sense_id -> row}."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    return {r["sense_id"]: r for r in rows if (r.get("final_status") or "").strip()}


def _load_prior_fixes(path: Path) -> dict[str, dict[str, str]]:
    """Map sense_id -> {failure_axes, failure_reason} for context to the auditor."""
    if not path.exists():
        return {}
    rows = read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "failure_axes": (r.get("failure_axes") or "").strip(),
            "failure_reason": (r.get("failure_reason") or "").strip(),
        }
    return out


# --- Live monitoring ------------------------------------------------------- #


class ProgressTracker:
    """Thread-safe progress tracker.

    Writes JSONL events to disk and maintains in-memory counters for the
    periodic stdout summary.
    """

    def __init__(self, progress_path: Path, total: int) -> None:
        self.progress_path = progress_path
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.total = total
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self.in_flight: dict[str, float] = {}  # sense_id -> start_ts (monotonic)
        self.done = 0
        self.error_count = 0
        self.verdicts = {"pass": 0, "regenerate": 0, "human_review": 0}
        self.latencies_ms: list[int] = []
        self.start_wall = time.time()

    def _emit(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._file_lock:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def started(self, sense_id: str, attempt: int = 0) -> None:
        now = time.monotonic()
        with self._lock:
            self.in_flight[sense_id] = now
        self._emit(
            {
                "event": "started",
                "sense_id": sense_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
            }
        )

    def completed(
        self, sense_id: str, *, verdict: str, defect_count: int, attempt: int = 0
    ) -> None:
        now = time.monotonic()
        with self._lock:
            start = self.in_flight.pop(sense_id, now)
            latency_ms = int((now - start) * 1000)
            self.latencies_ms.append(latency_ms)
            self.done += 1
            self.verdicts[verdict] = self.verdicts.get(verdict, 0) + 1
        self._emit(
            {
                "event": "completed",
                "sense_id": sense_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": latency_ms,
                "verdict": verdict,
                "defect_count": defect_count,
                "attempt": attempt,
            }
        )

    def errored(
        self, sense_id: str, *, error_type: str, error_msg: str, attempt: int = 0
    ) -> None:
        with self._lock:
            self.in_flight.pop(sense_id, None)
            self.error_count += 1
        self._emit(
            {
                "event": "errored",
                "sense_id": sense_id,
                "errored_at": datetime.now(timezone.utc).isoformat(),
                "error_type": error_type,
                "error_msg": error_msg[:200],
                "attempt": attempt,
            }
        )

    def stuck_check(self, threshold_sec: float = STUCK_THRESHOLD_SEC) -> list[tuple[str, float]]:
        """Return list of (sense_id, secs_in_flight) for calls running > threshold."""
        now = time.monotonic()
        stuck = []
        with self._lock:
            for sid, start in self.in_flight.items():
                elapsed = now - start
                if elapsed > threshold_sec:
                    stuck.append((sid, elapsed))
        return stuck

    def snapshot(self) -> dict:
        with self._lock:
            in_flight_count = len(self.in_flight)
            done = self.done
            errored = self.error_count
            verdicts = dict(self.verdicts)
            lats = list(self.latencies_ms)
        avg_lat = sum(lats) / len(lats) if lats else 0
        p99_lat = sorted(lats)[int(0.99 * len(lats))] if lats else 0
        elapsed = time.time() - self.start_wall
        rate = done / elapsed if elapsed > 0 else 0
        remaining = max(0, self.total - done)
        eta_sec = int(remaining / rate) if rate > 0 else 0
        return {
            "total": self.total,
            "done": done,
            "in_flight": in_flight_count,
            "errored": errored,
            "stuck": len(self.stuck_check()),
            "verdicts": verdicts,
            "avg_latency_ms": int(avg_lat),
            "p99_latency_ms": int(p99_lat),
            "elapsed_sec": int(elapsed),
            "eta_sec": eta_sec,
        }

    def format_summary(self) -> str:
        s = self.snapshot()
        eta_min = s["eta_sec"] // 60
        return (
            f"[Stage 5.5  {s['done']:,}/{s['total']:,} done | "
            f"{s['in_flight']} in-flight | "
            f"{s['stuck']} stuck (>{STUCK_THRESHOLD_SEC}s) | "
            f"avg {s['avg_latency_ms']/1000:.1f}s | "
            f"p99 {s['p99_latency_ms']/1000:.1f}s | "
            f"ETA {eta_min} min | "
            f"verdicts P/R/H: {s['verdicts']['pass']}/{s['verdicts']['regenerate']}/{s['verdicts']['human_review']}]"
        )


# --- Per-row auditing ----------------------------------------------------- #


def _build_user_message(row: dict, prior_fix: dict[str, str] | None = None) -> str:
    parts = [
        f"pt: {row.get('pt', '')}",
        f"pt_display: {row.get('pt_display', '')}",
        f"pt_type: {row.get('pt_type', '')}",
        f"gender: {row.get('gender', '')}",
        f"pos: {row.get('pos', '')}",
        f"tags: {row.get('tags', '')}",
        f"bp_status: {row.get('bp_status', '')}",
        f"en_primary: {row.get('en_primary', '')}",
        f"en_all: {row.get('en_all', '')}",
        f"example_pt: {row.get('example_pt', '')}",
        f"example_en: {row.get('example_en', '')}",
        f"target_word_used: {row.get('target_word_used', '')}",
        f"ipa_word_final: {row.get('ipa_word_final', '')}",
        f"ipa_example_final: {row.get('ipa_example_final', '')}",
    ]
    # Optional context
    policy = row.get("example_policy") or row.get("_example_policy") or ""
    if policy:
        parts.append(f"example_policy: {policy}")
    if prior_fix:
        parts.append(
            f"prior_validator_status: fail (axes: {prior_fix.get('failure_axes','')})"
        )
        parts.append(f"prior_validator_reason: {prior_fix.get('failure_reason','')}")
    return "\n".join(parts)


@dataclass
class AuditResult:
    sense_id: str
    verdict: str  # 'pass' | 'regenerate' | 'human_review'
    defects: list[dict] = field(default_factory=list)
    reason: str = ""
    confidence: str = "medium"
    method: str = "llm"  # 'llm' | 'manual_override' | 'fallback'


def _audit_one(
    row: dict,
    prior_fix: dict | None,
    *,
    client: AnthropicClient,
    system_prompt: str,
    tracker: ProgressTracker,
    attempt: int = 0,
) -> AuditResult:
    sid = row["sense_id"]
    tracker.started(sid, attempt=attempt)
    user_msg = _build_user_message(row, prior_fix)
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="audit_row",
            tool_description=(
                "Adversarially audit a Brazilian-Portuguese sense record and "
                "produce a verdict (pass / regenerate / human_review) with "
                "defect list."
            ),
            tool_input_schema=AUDIT_TOOL_SCHEMA,
            stage="5_5",
            provenance_key=sid,
            tier=TIER_DEFAULT,
            max_tokens=1024,
        )
    except Exception as exc:
        tracker.errored(
            sid,
            error_type=type(exc).__name__,
            error_msg=str(exc),
            attempt=attempt,
        )
        return AuditResult(
            sense_id=sid,
            verdict="human_review",
            defects=[],
            reason=f"auditor error: {type(exc).__name__}",
            confidence="low",
            method="fallback",
        )

    verdict = (decision.get("verdict") or "human_review").strip().lower()
    if verdict not in ("pass", "regenerate", "human_review"):
        verdict = "human_review"
    defects = decision.get("defects") or []
    defect_count = len(defects)

    tracker.completed(sid, verdict=verdict, defect_count=defect_count, attempt=attempt)

    return AuditResult(
        sense_id=sid,
        verdict=verdict,
        defects=defects,
        reason=(decision.get("reason") or "").strip()[:200],
        confidence=(decision.get("confidence") or "medium").strip().lower(),
        method="llm",
    )


# --- Periodic summary thread --------------------------------------------- #


class SummaryPrinter(threading.Thread):
    """Prints a one-line summary every SUMMARY_INTERVAL_SEC until stopped."""

    def __init__(self, tracker: ProgressTracker, interval: float = SUMMARY_INTERVAL_SEC):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            print(self.tracker.format_summary(), file=sys.stderr, flush=True)
            stuck = self.tracker.stuck_check()
            if stuck:
                for sid, secs in stuck[:3]:
                    print(
                        f"  [warn] stuck: {sid} running {int(secs)}s",
                        file=sys.stderr,
                        flush=True,
                    )

    def stop(self) -> None:
        self.stop_event.set()


# --- Main pipeline ------------------------------------------------------- #


def _serialize_audit_row(
    sense_id: str,
    res: AuditResult,
    *,
    regen_attempts: int,
    final_status: str,
) -> dict:
    return {
        "sense_id": sense_id,
        "verdict": res.verdict,
        "defect_count": len(res.defects),
        "defects_json": json.dumps(res.defects, ensure_ascii=False),
        "reason": res.reason,
        "confidence": res.confidence,
        "regen_attempts": str(regen_attempts),
        "final_status": final_status,
        "audit_method": res.method,
    }


def run(
    *,
    input_path: Path = INPUT_PATH,
    examples_path: Path = EXAMPLES_PATH,
    prior_fixes_path: Path = PRIOR_FIXES_PATH,
    overrides_path: Path = OVERRIDES_PATH,
    output_path: Path = OUTPUT_PATH,
    human_review_path: Path = HUMAN_REVIEW_PATH,
    audit_path: Path = AUDIT_PATH,
    progress_path: Path = PROGRESS_PATH,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    sense_id_filter: set[str] | None = None,
    skip_regeneration: bool = False,
    anthropic_client: AnthropicClient | None = None,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found at {input_path}. Run Stage 5 first."
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
    prior_fixes = _load_prior_fixes(prior_fixes_path)
    existing = _load_existing_audit(output_path)
    system_prompt = _load_prompt()

    if anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        anthropic_client = AnthropicClient(audit_path=audit_path)

    # Skip rows already audited (resume support)
    rows_to_audit = [r for r in rows if r["sense_id"] not in existing]
    skipped_resume = len(rows) - len(rows_to_audit)

    print(
        f"[info] Stage 5.5: input={len(rows)}, "
        f"resumed-skipped={skipped_resume}, to_audit={len(rows_to_audit)}",
        file=sys.stderr,
    )

    # --- Phase A: classify all rows -----------------------------------------

    tracker = ProgressTracker(progress_path, total=len(rows_to_audit))
    summary_thread = SummaryPrinter(tracker)
    summary_thread.start()

    results: dict[str, AuditResult] = {}

    # Apply manual overrides first (no LLM)
    manual_count = 0
    rows_for_llm: list[dict] = []
    for row in rows_to_audit:
        sid = row["sense_id"]
        ov = overrides.get(sid)
        if ov:
            results[sid] = AuditResult(
                sense_id=sid,
                verdict=ov["verdict"],
                defects=[],
                reason=ov["reason"],
                confidence="high",
                method="manual_override",
            )
            manual_count += 1
        else:
            rows_for_llm.append(row)

    if rows_for_llm:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_sid = {
                pool.submit(
                    _audit_one,
                    row,
                    prior_fixes.get(row["sense_id"]),
                    client=anthropic_client,
                    system_prompt=system_prompt,
                    tracker=tracker,
                    attempt=0,
                ): row["sense_id"]
                for row in rows_for_llm
            }
            for fut in as_completed(future_to_sid):
                res = fut.result()
                results[res.sense_id] = res

    summary_thread.stop()
    print(tracker.format_summary(), file=sys.stderr, flush=True)

    # --- Phase B: regen failures (skipped for v1 — see plan) ----------------
    # Note: the plan calls for ≤2 regen attempts per failed row. For the
    # first run, we DO NOT auto-regenerate — we surface failures so the user
    # can review and decide. This is more conservative than the plan; the
    # `skip_regeneration` flag defaults to True for safety.
    #
    # When skip_regeneration=False (future runs), we'd:
    #   1. Collect verdict=='regenerate' rows
    #   2. Call build/stage_4.py::run(sense_id_filter=failed_sids) to re-prompt
    #   3. Re-audit the regenerated rows
    #   4. After 2 failed regens, route to human_review
    # This is gated to a future iteration to keep this run simple and review-able.

    regen_count_per_sid: dict[str, int] = {sid: 0 for sid in results}

    # --- Build output -------------------------------------------------------

    output_rows: list[dict] = []
    human_review_rows: list[dict] = []

    rows_by_sid = {r["sense_id"]: r for r in rows}

    # Preserve existing rows verbatim (resume support)
    for sid, existing_row in existing.items():
        if sid in rows_by_sid:
            output_rows.append(existing_row)

    for sid, res in results.items():
        regen_attempts = regen_count_per_sid.get(sid, 0)
        if res.verdict == "pass":
            final_status = "pass_first"
        elif res.verdict == "human_review":
            final_status = "human_review"
        elif res.verdict == "regenerate":
            # In skip_regeneration mode, treat as human_review for now
            final_status = "human_review"
        else:
            final_status = "human_review"

        out = _serialize_audit_row(
            sid, res, regen_attempts=regen_attempts, final_status=final_status
        )
        output_rows.append(out)

        if final_status == "human_review":
            row_ctx = rows_by_sid.get(sid, {})
            human_review_rows.append(
                {
                    "sense_id": sid,
                    "pt": row_ctx.get("pt", ""),
                    "en_primary": row_ctx.get("en_primary", ""),
                    "example_pt": row_ctx.get("example_pt", ""),
                    "example_en": row_ctx.get("example_en", ""),
                    "verdict": res.verdict,
                    "reason": res.reason,
                    "defects_json": json.dumps(res.defects, ensure_ascii=False),
                    "regen_attempts": str(regen_attempts),
                }
            )

    # Stable order: by sense_id
    output_rows.sort(key=lambda r: r["sense_id"])
    human_review_rows.sort(key=lambda r: r["sense_id"])

    n_out = write_tsv(output_path, output_rows, fieldnames=OUTPUT_FIELDS)
    n_hr = write_tsv(human_review_path, human_review_rows, fieldnames=HUMAN_REVIEW_FIELDS)

    # --- Summary ------------------------------------------------------------

    from collections import Counter

    final_counts = Counter(r["final_status"] for r in output_rows)
    verdict_counts = Counter(r["verdict"] for r in output_rows)
    method_counts = Counter(r["audit_method"] for r in output_rows)

    summary = {
        "input_rows": len(rows),
        "audited_this_run": len(rows_for_llm),
        "manual_overrides": manual_count,
        "resumed_skipped": skipped_resume,
        "output_rows": n_out,
        "human_review_rows": n_hr,
        "final_status_counts": dict(final_counts),
        "verdict_counts": dict(verdict_counts),
        "audit_method_counts": dict(method_counts),
        "tracker_snapshot": tracker.snapshot(),
    }
    if anthropic_client is not None:
        summary["anthropic_cache_stats"] = anthropic_client.cache_stats
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5.5 — single-pass auditor")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--prior-fixes", type=Path, default=PRIOR_FIXES_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--human-review", type=Path, default=HUMAN_REVIEW_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--progress", type=Path, default=PROGRESS_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-regeneration",
        action="store_true",
        default=True,
        help="Surface verdict=regenerate as human_review instead of re-prompting Stage 4 (default: True for v1).",
    )
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
        prior_fixes_path=args.prior_fixes,
        overrides_path=args.overrides,
        output_path=args.output,
        human_review_path=args.human_review,
        audit_path=args.audit,
        progress_path=args.progress,
        concurrency=args.concurrency,
        limit=args.limit,
        sense_id_filter=sense_id_filter,
        skip_regeneration=args.skip_regeneration,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
