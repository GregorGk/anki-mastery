"""Stage 14 / Step 0 — Classify primary topic tag (`#topic-*`) for every row.

For each row in `data/06-final.tsv`:
  * If `pos ∈ {prep, conj, pron, art}` → write `#topic-grammar` to the sidecar
    deterministically (no LLM call).
  * If `pos == num` → write `#topic-numbers` deterministically.
  * Otherwise → ask Claude Sonnet 4.6 (Anthropic Tool Use + prompt caching)
    to pick one tag from the 50-item enum.

Output:
  - full mode:  data/_topic_tags.tsv + audit/14_topic_tags.jsonl
  - pilot mode: data/_topic_tags.pilot.tsv + audit/14_topic_tags.pilot.jsonl
                + reports/14_topic_tags_pilot.html (via build/14_1_pilot_html.py)

Modes:
  --dry-run             preflight only — prompt hash, system size, cost est.
  --pilot N --seed S    stratified N-row pilot (default 500, seed 14)
  default               full run; resumes from existing sidecar

CLI flags (env fallback in parens):
  --max-workers N       (ANTHROPIC_MAX_WORKERS)    default 75 full / 25 pilot
  --max-rpm N           (ANTHROPIC_MAX_RPM)        default 3200 (0 = unbounded)
  --max-retries N       (ANTHROPIC_MAX_RETRIES)    default 8
  --yes                 skip GO prompt

Usage:
  .venv/bin/python build/14_0_topic_classifier.py --dry-run
  ANTHROPIC_MAX_WORKERS=25 ANTHROPIC_MAX_RPM=3200 \\
    .venv/bin/python build/14_0_topic_classifier.py --pilot 500 --seed 14
  ANTHROPIC_MAX_WORKERS=75 ANTHROPIC_MAX_RPM=3200 \\
    .venv/bin/python build/14_0_topic_classifier.py --yes
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.rate_limit import SlidingWindowRateLimiter  # noqa: E402
from build.lib.topic_tag_rules import (  # noqa: E402
    ALLOWED_TOPIC_TAGS_SORTED,
    LLM_POS,
    MANUAL_TOPIC_OVERRIDES,
    NUM_TOPIC_OVERRIDES,
    POS_TOPIC_PRERULES,
    apply_post_process,
    deterministic_topic_for_row,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
REPORTS_DIR = REPO_ROOT / "reports"
PROMPTS_DIR = REPO_ROOT / "build" / "prompts"

FINAL_TSV = DATA_DIR / "06-final.tsv"
SIDECAR_TSV_FULL = DATA_DIR / "_topic_tags.tsv"
AUDIT_JSONL_FULL = AUDIT_DIR / "14_topic_tags.jsonl"
PROMPT_PATH = PROMPTS_DIR / "topic_tags.md"


def _pilot_paths(suffix: str) -> tuple[Path, Path, Path]:
    """Compute pilot file paths from a suffix.

    Empty suffix    → data/_topic_tags.pilot.tsv (v1)
    suffix='v2'     → data/_topic_tags.pilot.v2.tsv (v2)
    suffix='v3'     → data/_topic_tags.pilot.v3.tsv (v3)

    Each pilot iteration changes the prompt/rules; reusing the same sidecar
    would resume-skip rows and not actually test the new logic. The suffix
    lets the user fork pilot artifacts without touching v1's outputs.
    """
    if not suffix:
        return (
            DATA_DIR / "_topic_tags.pilot.tsv",
            AUDIT_DIR / "14_topic_tags.pilot.jsonl",
            REPORTS_DIR / "14_topic_tags_pilot.html",
        )
    return (
        DATA_DIR / f"_topic_tags.pilot.{suffix}.tsv",
        AUDIT_DIR / f"14_topic_tags.pilot.{suffix}.jsonl",
        REPORTS_DIR / f"14_topic_tags_pilot_{suffix}.html",
    )

MODEL_ID = "claude-sonnet-4-6"
DEFAULT_MAX_WORKERS_FULL = 75
DEFAULT_MAX_WORKERS_PILOT = 25
DEFAULT_MAX_RPM = 3200
DEFAULT_MAX_RETRIES = 8

# Cache-hit gate — dynamic theoretical-ceiling rule (replaces fixed 0.90/0.95).
# The asymptotic ceiling is `cached_block_size / (cached_block_size +
# avg_uncached_per_call)`. Pilot v1 measured 0.864 with this prompt shape
# (3,491 cached tokens + ~550 uncached per call). Fixed 0.90 was unreachable.
#
# Hard halt diagnostics:
#   - cache_read_tokens == 0 (the cache never fired — prompt is mutating)
#   - cache_hit_ratio < CACHE_HARD_HALT_FLOOR (catastrophic miss)
# Warn diagnostic:
#   - cache_hit_ratio < theoretical_ceiling - CACHE_CEILING_TOLERANCE
CACHE_HARD_HALT_FLOOR = 0.75
CACHE_CEILING_TOLERANCE = 0.03
CACHE_GATE_AFTER_N_CALLS = 100

# Sidecar schema (production + pilot use the same columns).
SIDECAR_FIELDS = [
    "sense_id", "rank", "pt", "pos", "en_primary",
    "example_pt", "example_en",
    "topic_primary", "confidence", "reason",
    "source", "model_id", "generated_at",
]

TOOL_NAME = "record_topic_tag"
TOOL_DESCRIPTION = (
    "Record the primary topic tag for one BP sense. Must be one of the "
    "50 allowed #topic-* strings. Classify by the target sense first, "
    "then use the example only as context. Default to the broadest honest "
    "topic when uncertain; set confidence accordingly."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_primary": {
            "type": "string",
            "enum": list(ALLOWED_TOPIC_TAGS_SORTED),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "string",
            "description": "One short sentence justifying the choice.",
        },
    },
    "required": ["topic_primary", "confidence", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Data shape
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    sense_id: str
    rank: int
    pt: str
    pt_display_safe: str
    pos: str
    pt_type: str
    en_primary: str
    en_all: str
    family_root: str
    bp_status: str
    tags: str
    usage_hint: str
    risk_note: str
    example_pt: str
    example_en: str


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
            pt_display_safe=(r.get("pt_display_safe") or "").strip(),
            pos=(r.get("pos") or "").strip(),
            pt_type=(r.get("pt_type") or "").strip(),
            en_primary=(r.get("en_primary") or "").strip(),
            en_all=(r.get("en_all") or "").strip(),
            family_root=(r.get("family_root") or "").strip(),
            bp_status=(r.get("bp_status") or "").strip(),
            tags=(r.get("tags") or "").strip(),
            usage_hint=(r.get("usage_hint") or "").strip(),
            risk_note=(r.get("risk_note") or "").strip(),
            example_pt=(r.get("example_pt") or "").strip(),
            example_en=(r.get("example_en") or "").strip(),
        ))
    return out


def _read_existing_sidecar(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(path)}


# --------------------------------------------------------------------------- #
# User-message builder (sense-first; example fields LAST per plan).
# --------------------------------------------------------------------------- #


def _build_user_message(r: Row) -> str:
    return (
        f"sense_id: {r.sense_id}\n"
        f"rank: {r.rank}\n"
        f"pt: {r.pt}\n"
        f"pt_display_safe: {r.pt_display_safe}\n"
        f"pos: {r.pos}\n"
        f"pt_type: {r.pt_type}\n"
        f"en_primary: {r.en_primary}\n"
        f"en_all: {r.en_all}\n"
        f"family_root: {r.family_root}\n"
        f"bp_status: {r.bp_status}\n"
        f"tags: {r.tags}\n"
        f"usage_hint: {r.usage_hint}\n"
        f"risk_note: {r.risk_note}\n"
        f"example_pt: {r.example_pt}\n"
        f"example_en: {r.example_en}"
    )


# --------------------------------------------------------------------------- #
# Pilot stratification
# --------------------------------------------------------------------------- #


def _stratify_pilot(
    rows: list[Row], n: int, seed: int
) -> list[Row]:
    """Return a stratified n-row pilot subset.

    Composition (locked from plan §"Stratified composition"):
      - all deterministic-eligible rows (prep/conj/pron/art/num)
      - all idiom + interj rows (LLM-eligible content-bearing)
      - up to 20 rows with usage_hint
      - all rows with risk_note (1 today: gozar)
      - up to 30 multi-sense lemmas (≥ 2 senses for the same pt)
      - remainder filled by stratified-by-rank-tier random sample
        across the LLM-eligible content-word pool
    """
    rng = random.Random(seed)
    picked: dict[str, Row] = {}

    def add(r: Row) -> None:
        picked.setdefault(r.sense_id, r)

    # 1. All deterministic rows.
    det = [r for r in rows if r.pos in POS_TOPIC_PRERULES]
    for r in det:
        add(r)

    # 2. All idiom + interj rows (10 rows total today).
    for r in rows:
        if r.pos in ("idiom", "interj"):
            add(r)

    # 3. usage_hint rows — sample up to 20.
    hint_rows = [r for r in rows if r.usage_hint and r.sense_id not in picked]
    if hint_rows:
        for r in rng.sample(hint_rows, min(20, len(hint_rows))):
            add(r)

    # 4. risk_note rows — include all (only 1 today).
    for r in rows:
        if r.risk_note and r.sense_id not in picked:
            add(r)

    # 5. Multi-sense lemmas — sample up to 30 rows from pts with ≥ 2 senses.
    by_pt: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.pt:
            by_pt[r.pt].append(r)
    multi = [r for pt, group in by_pt.items() if len(group) >= 2
             for r in group if r.sense_id not in picked
             and r.pos in LLM_POS]
    if multi:
        for r in rng.sample(multi, min(30, len(multi))):
            add(r)

    # 6. Fill remainder via stratified-by-rank-tier random over the LLM pool.
    tiers = (
        ("top-1-500",    1,    500),
        ("top-501-2000",  501,  2000),
        ("top-2001-3500", 2001, 3500),
        ("top-3501-plus", 3501, 10**9),
    )
    remaining = n - len(picked)
    if remaining > 0:
        per_tier = max(1, remaining // len(tiers))
        for _, lo, hi in tiers:
            if len(picked) >= n:
                break
            pool = [r for r in rows
                    if lo <= r.rank <= hi
                    and r.pos in LLM_POS
                    and r.sense_id not in picked]
            if not pool:
                continue
            quota = min(per_tier, len(pool), n - len(picked))
            for r in rng.sample(pool, quota):
                add(r)

    return sorted(picked.values(), key=lambda r: r.sense_id)


# --------------------------------------------------------------------------- #
# Sidecar I/O (append-safe, threadsafe)
# --------------------------------------------------------------------------- #


def _open_sidecar_appender(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    fh = path.open("a", encoding="utf-8", newline="")
    if is_new:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        w.writeheader()
        fh.flush()
    return fh


def _write_sidecar_row(row_out: dict, fh, writer_lock: threading.Lock) -> None:
    with writer_lock:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL,
                           extrasaction="ignore")
        sanitized = {k: ("" if row_out.get(k) is None else str(row_out.get(k, "")))
                     for k in SIDECAR_FIELDS}
        w.writerow(sanitized)
        fh.flush()


# --------------------------------------------------------------------------- #
# Progress tracker — realized-RPM rolling window + cache stats + 429 count
# --------------------------------------------------------------------------- #


class ProgressTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.rate_limit_429 = 0
        self.completed_ts: deque[float] = deque()  # rolling window of 100
        self.start_ts = time.monotonic()

    def mark_done(self) -> None:
        with self.lock:
            self.done += 1
            now = time.monotonic()
            self.completed_ts.append(now)
            if len(self.completed_ts) > 100:
                self.completed_ts.popleft()

    def mark_fail(self, exc_type_name: str) -> None:
        with self.lock:
            self.failed += 1
            if "RateLimit" in exc_type_name:
                self.rate_limit_429 += 1

    def snapshot(self) -> dict:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.start_ts
            rpm_last_100 = 0.0
            if len(self.completed_ts) >= 2:
                span = max(self.completed_ts[-1] - self.completed_ts[0], 1e-3)
                rpm_last_100 = (len(self.completed_ts) - 1) / span * 60.0
            rpm_total = self.done / elapsed * 60.0 if elapsed > 0 else 0.0
            return {
                "done": self.done,
                "failed": self.failed,
                "elapsed_s": elapsed,
                "rpm_last_100": rpm_last_100,
                "rpm_total": rpm_total,
                "rate_limit_429": self.rate_limit_429,
            }


# --------------------------------------------------------------------------- #
# Cache-hit gate
# --------------------------------------------------------------------------- #


class CacheGate:
    """Tracks cache hit ratio; halts/warns per the dynamic-ceiling policy.

    The fixed 0.90 / 0.95 thresholds from earlier plan iterations are
    mathematically unreachable for prompts where the per-row user message
    is non-trivial compared to the cached system block. The ceiling is:

        ceiling ≈ cached_block_size / (cached_block_size + avg_uncached_per_call)

    Acceptance: actual_ratio >= ceiling - CACHE_CEILING_TOLERANCE.
    Hard halt: cache_read_tokens == 0 (cache never fired) OR
               ratio < CACHE_HARD_HALT_FLOOR (catastrophic).
    """

    def __init__(self, client: AnthropicClient,
                 hard_halt_floor: float = CACHE_HARD_HALT_FLOOR,
                 ceiling_tolerance: float = CACHE_CEILING_TOLERANCE,
                 after_calls: int = CACHE_GATE_AFTER_N_CALLS) -> None:
        self.client = client
        self.hard_halt_floor = hard_halt_floor
        self.ceiling_tolerance = ceiling_tolerance
        self.after_calls = after_calls
        self.baseline_cache_size = 0          # set by record_baseline_*

    def record_baseline_after_warmup(self) -> int:
        """Snapshot the cached-block size after the sequential warmup call.

        Returns the captured size for logging. Call this BEFORE the worker
        pool starts so the ceiling computation has the right C value.
        """
        stats = self.client.cache_stats
        # After 1 sequential warmup call, cache_creation_tokens equals the
        # cached system-prompt block size (no races, single creator).
        self.baseline_cache_size = stats.get("cache_creation_tokens", 0)
        return self.baseline_cache_size

    def evaluate(self) -> tuple[str, dict]:
        """Return ("ok" | "warn" | "halt", augmented_stats_dict)."""
        stats = dict(self.client.cache_stats)
        ratio = stats.get("cache_hit_ratio", 0.0)
        calls = stats.get("calls", 0)
        if calls < self.after_calls:
            return "ok", stats

        cache_read = stats.get("cache_read_tokens", 0)

        # Hard halt 1: cache never fired (system prompt is mutating).
        if cache_read == 0:
            stats["reason"] = "no_cache_reads"
            return "halt", stats

        # Hard halt 2: catastrophic miss below the diagnostic floor.
        if ratio < self.hard_halt_floor:
            stats["reason"] = f"ratio_below_floor_{self.hard_halt_floor}"
            return "halt", stats

        # Theoretical ceiling check.
        if self.baseline_cache_size > 0:
            avg_uncached = stats.get("uncached_input_tokens", 0) / max(calls, 1)
            ceiling = (
                self.baseline_cache_size
                / (self.baseline_cache_size + avg_uncached)
            )
            stats["theoretical_ceiling"] = ceiling
            stats["baseline_cache_size"] = self.baseline_cache_size
            stats["avg_uncached_per_call"] = avg_uncached
            if ratio < ceiling - self.ceiling_tolerance:
                stats["reason"] = (
                    f"below_ceiling: actual={ratio:.3f} ceiling={ceiling:.3f}"
                )
                return "warn", stats

        return "ok", stats


# --------------------------------------------------------------------------- #
# One-row classification
# --------------------------------------------------------------------------- #


def _classify_one(
    *,
    client: AnthropicClient,
    system: str,
    row: Row,
    limiter: SlidingWindowRateLimiter,
    stage_tag: str,
) -> dict:
    """One LLM call (rate-limited + idempotent post-process). Returns the
    dict ready to write to the sidecar."""
    limiter.wait_before_call()
    user_msg = _build_user_message(row)
    decision = client.call_tool(
        system=system,
        user_message=user_msg,
        tool_name=TOOL_NAME,
        tool_input_schema=TOOL_INPUT_SCHEMA,
        tool_description=TOOL_DESCRIPTION,
        max_tokens=128,
        stage=stage_tag,
        provenance_key=row.sense_id,
    )
    payload = {
        "topic_primary": (decision.get("topic_primary") or "").strip(),
        "confidence":    (decision.get("confidence") or "low").strip().lower(),
        "reason":        (decision.get("reason") or "").strip().replace("\n", " "),
    }
    return apply_post_process(payload)


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _print_preflight(
    *, mode: str, total_rows: int, det_rows: int, llm_rows: int,
    max_workers: int, max_rpm: int, max_retries: int,
    sidecar: Path, audit: Path, system_prompt: str,
) -> None:
    sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    # Cost estimate (Sonnet 4.6 + caching): ~$0.003 / call.
    est_cost = llm_rows * 0.003
    border = "┌" + "─" * 72 + "┐"
    bottom = "└" + "─" * 72 + "┘"
    sep    = "├" + "─" * 72 + "┤"
    print(border)
    print(f"│  STAGE 14 — TOPIC TAG CLASSIFIER PREFLIGHT" + " " * 30 + "│")
    print(sep)
    print(f"│  Mode:                  {mode:<48}│")
    print(f"│  Model:                  {MODEL_ID:<47}│")
    print(f"│  System prompt path:    {str(PROMPT_PATH.relative_to(REPO_ROOT)):<48}│")
    print(f"│  System prompt chars:   {len(system_prompt):<48}│")
    print(f"│  System prompt hash:    {sys_hash[:48]:<48}│")
    print(f"│  Total rows:            {total_rows:<48}│")
    print(f"│  Deterministic rows:    {det_rows:<48}│")
    print(f"│  LLM rows (to classify):{llm_rows:<48}│")
    print(f"│  --max-workers:         {max_workers:<48}│")
    print(f"│  --max-rpm:             {max_rpm:<48}│")
    print(f"│  --max-retries:         {max_retries:<48}│")
    print(f"│  Cost (est):            ${est_cost:<47.2f}│")
    print(f"│  Sidecar:               {str(sidecar.relative_to(REPO_ROOT)):<48}│")
    print(f"│  Audit:                 {str(audit.relative_to(REPO_ROOT)):<48}│")
    print(bottom)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="Preflight only — no LLM calls, no sidecar writes.")
    p.add_argument("--pilot", type=int, default=0, metavar="N",
                   help="Run a stratified N-row pilot (separate sidecar/audit/HTML).")
    p.add_argument("--seed", type=int, default=14,
                   help="Pilot stratification seed (default 14).")
    p.add_argument("--pilot-suffix", type=str, default="",
                   help="Pilot version suffix — e.g. 'v2' writes "
                        "data/_topic_tags.pilot.v2.tsv. Empty = v1 paths. "
                        "Use when prompt/rules change between pilots so "
                        "the new pilot doesn't resume-skip old rows.")
    p.add_argument("--max-workers", type=int, default=0,
                   help="Override default workers (full=75, pilot=25). "
                        "Env fallback: ANTHROPIC_MAX_WORKERS.")
    p.add_argument("--max-rpm", type=int, default=-1,
                   help=f"RPM cap (default {DEFAULT_MAX_RPM}; 0 = unbounded). "
                        "Env fallback: ANTHROPIC_MAX_RPM.")
    p.add_argument("--max-retries", type=int, default=0,
                   help=f"Per-call retry cap (default {DEFAULT_MAX_RETRIES}). "
                        "Env fallback: ANTHROPIC_MAX_RETRIES.")
    p.add_argument("--yes", action="store_true", help="Skip GO prompt")
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    if not PROMPT_PATH.exists():
        print(f"ERROR: prompt missing at {PROMPT_PATH}", file=sys.stderr)
        return 1
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    is_pilot = args.pilot > 0
    suffix_label = f" suffix={args.pilot_suffix!r}" if args.pilot_suffix else ""
    mode = (f"PILOT (n={args.pilot}, seed={args.seed}{suffix_label})"
            if is_pilot else "FULL RUN")
    if is_pilot:
        sidecar_path, audit_path, _pilot_html_path = _pilot_paths(args.pilot_suffix)
    else:
        sidecar_path = SIDECAR_TSV_FULL
        audit_path   = AUDIT_JSONL_FULL

    # Resolve worker count / RPM with env fallbacks.
    default_workers = DEFAULT_MAX_WORKERS_PILOT if is_pilot else DEFAULT_MAX_WORKERS_FULL
    max_workers = args.max_workers or _env_int("ANTHROPIC_MAX_WORKERS", default_workers)
    max_rpm = args.max_rpm if args.max_rpm >= 0 else _env_int("ANTHROPIC_MAX_RPM", DEFAULT_MAX_RPM)
    max_retries = args.max_retries or _env_int("ANTHROPIC_MAX_RETRIES", DEFAULT_MAX_RETRIES)

    # Pilot mode forces a sane worker ceiling so the cache-stat signal is
    # readable even if the user passes a bigger value via env.
    if is_pilot and max_workers > DEFAULT_MAX_WORKERS_PILOT * 2:
        print(f"[pilot] forcing --max-workers to {DEFAULT_MAX_WORKERS_PILOT} "
              f"(was {max_workers}); cache-stat signal needs to be readable")
        max_workers = DEFAULT_MAX_WORKERS_PILOT

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and partition rows.
    rows = _load_rows()
    if is_pilot:
        scope = _stratify_pilot(rows, n=args.pilot, seed=args.seed)
    else:
        scope = rows

    # Partition rows via deterministic_topic_for_row (honors per-sense
    # NUM_TOPIC_OVERRIDES — e.g. quarto/segundo/cento don't go to numbers).
    det_in_scope: list[tuple[Row, str]] = []
    llm_in_scope: list[Row] = []
    for r in scope:
        topic = deterministic_topic_for_row(r.sense_id, r.pos)
        if topic is not None:
            det_in_scope.append((r, topic))
        elif r.pos in LLM_POS:
            llm_in_scope.append(r)

    # Resume support — read existing sidecar, skip processed sense_ids.
    existing = _read_existing_sidecar(sidecar_path)
    det_todo = [(r, t) for r, t in det_in_scope if r.sense_id not in existing]
    llm_todo = [r for r in llm_in_scope if r.sense_id not in existing]

    _print_preflight(
        mode=mode, total_rows=len(scope),
        det_rows=len(det_todo), llm_rows=len(llm_todo),
        max_workers=max_workers, max_rpm=max_rpm, max_retries=max_retries,
        sidecar=sidecar_path, audit=audit_path,
        system_prompt=system_prompt,
    )

    if args.dry_run:
        # Show stratification breakdown so the user can sanity-check pilot mix.
        if is_pilot:
            print()
            print("Pilot stratification (after dedupe):")
            by_pos: dict[str, int] = defaultdict(int)
            for r in scope:
                by_pos[r.pos] += 1
            for pos_v, n in sorted(by_pos.items(), key=lambda x: -x[1]):
                print(f"  pos={pos_v:<8} {n}")
        print("\n--dry-run: stopping (no LLM calls, no sidecar writes).")
        return 0

    if not llm_todo and not det_todo:
        print("\nNothing to do — every row in scope already in sidecar.")
        return 0

    if not args.yes:
        sys.stdout.write("\nType GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # Open sidecar appender.
    fh = _open_sidecar_appender(sidecar_path)
    writer_lock = threading.Lock()

    # --- Deterministic prelude: zero API calls. -----------------------------
    if det_todo:
        ts = _now_iso()
        for r, topic in det_todo:
            if r.sense_id in MANUAL_TOPIC_OVERRIDES:
                reason = f"MANUAL_TOPIC_OVERRIDE: sense_id={r.sense_id} -> {topic}"
                source_tag = "manual"
            elif r.sense_id in NUM_TOPIC_OVERRIDES:
                reason = f"NUM_TOPIC_OVERRIDE: sense_id={r.sense_id} -> {topic}"
                source_tag = "deterministic"
            else:
                reason = f"POS pre-rule: pos={r.pos} -> {topic}"
                source_tag = "deterministic"
            _write_sidecar_row({
                "sense_id":      r.sense_id,
                "rank":          r.rank,
                "pt":            r.pt,
                "pos":           r.pos,
                "en_primary":    r.en_primary,
                "example_pt":    r.example_pt,
                "example_en":    r.example_en,
                "topic_primary": topic,
                "confidence":    "high",
                "reason":        reason,
                "source":        source_tag,
                "model_id":      "",
                "generated_at":  ts,
            }, fh, writer_lock)
        n_manual = sum(1 for r, _ in det_todo if r.sense_id in MANUAL_TOPIC_OVERRIDES)
        n_num_override = sum(1 for r, _ in det_todo if r.sense_id in NUM_TOPIC_OVERRIDES)
        n_pos = len(det_todo) - n_manual - n_num_override
        print(f"\nWrote {len(det_todo)} deterministic rows "
              f"({n_manual} via MANUAL_TOPIC_OVERRIDES, "
              f"{n_num_override} via NUM_TOPIC_OVERRIDES, "
              f"{n_pos} via POS pre-rules). No API calls.")

    # --- LLM run -----------------------------------------------------------
    if not llm_todo:
        fh.close()
        print("\nNo LLM rows to classify; deterministic prelude complete.")
        return 0

    client = AnthropicClient(
        model=args.model,
        audit_path=audit_path,
        enable_caching=True,
        max_retries=max_retries,
    )
    limiter = SlidingWindowRateLimiter(max_rpm=max_rpm)
    tracker = ProgressTracker()
    cache_gate = CacheGate(client)
    halt_event = threading.Event()
    stage_tag = "14_topic_tag" + ("_pilot" if is_pilot else "")

    print(f"\nLLM run: {len(llm_todo)} rows, "
          f"--max-workers={max_workers}, --max-rpm={max_rpm}\n")

    # ----- Cache-warmup pass (sequential, 1 call) -----
    # Without this, N concurrent workers all race a cold cache on startup,
    # each becoming an independent cache_creator. After the race, only the
    # late callers benefit. The cache_hit_ratio post-warmup looks artificially
    # bad and the cache gate triggers a false halt. Fix: classify exactly
    # one row sequentially so the server-side cache is seeded BEFORE the
    # pool fans out. All subsequent calls within the 5-min TTL hit the warm
    # cache cleanly.
    warm_row = llm_todo[0]
    print(f"  Cache warmup (1 sequential call to seed the server cache)...")
    try:
        warm_decision = _classify_one(
            client=client, system=system_prompt, row=warm_row,
            limiter=limiter, stage_tag=stage_tag,
        )
        _write_sidecar_row({
            "sense_id":      warm_row.sense_id,
            "rank":          warm_row.rank,
            "pt":            warm_row.pt,
            "pos":           warm_row.pos,
            "en_primary":    warm_row.en_primary,
            "example_pt":    warm_row.example_pt,
            "example_en":    warm_row.example_en,
            "topic_primary": warm_decision["topic_primary"],
            "confidence":    warm_decision["confidence"],
            "reason":        warm_decision["reason"],
            "source":        "llm",
            "model_id":      args.model,
            "generated_at":  _now_iso(),
        }, fh, writer_lock)
        warm_stats = client.cache_stats
        baseline = cache_gate.record_baseline_after_warmup()
        print(f"  Warmup done: cache_creation={warm_stats.get('cache_creation_tokens', 0)} "
              f"cache_read={warm_stats.get('cache_read_tokens', 0)}")
        print(f"  Baseline cached-block size: {baseline} tokens "
              f"(theoretical-ceiling gate uses this as C)")
        llm_todo = llm_todo[1:]
    except Exception as exc:  # noqa: BLE001
        print(f"  Warmup FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        # Don't abort — the warmup is an optimization, not a hard requirement.
        # The cache will still seed on the first pool call (just with a race).

    n_failed_total = 0
    last_log_at = 0

    def _worker(r: Row) -> tuple[Row, dict | None, str]:
        if halt_event.is_set():
            return r, None, "halted"
        try:
            d = _classify_one(
                client=client, system=system_prompt, row=r,
                limiter=limiter, stage_tag=stage_tag,
            )
            return r, d, ""
        except Exception as exc:  # noqa: BLE001
            return r, None, f"{type(exc).__name__}: {exc}"

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_worker, r): r for r in llm_todo}
            for fut in as_completed(futs):
                if halt_event.is_set():
                    continue
                r, decision, err = fut.result()
                if err == "halted":
                    continue
                if err:
                    tracker.mark_fail(err.split(":")[0])
                    n_failed_total += 1
                    print(f"  FAIL {r.sense_id} {r.pt}  {err}", flush=True)
                    continue

                tracker.mark_done()
                _write_sidecar_row({
                    "sense_id":      r.sense_id,
                    "rank":          r.rank,
                    "pt":            r.pt,
                    "pos":           r.pos,
                    "en_primary":    r.en_primary,
                    "example_pt":    r.example_pt,
                    "example_en":    r.example_en,
                    "topic_primary": decision["topic_primary"],
                    "confidence":    decision["confidence"],
                    "reason":        decision["reason"],
                    "source":        "llm",
                    "model_id":      args.model,
                    "generated_at":  _now_iso(),
                }, fh, writer_lock)

                # Progress + cache check every 100 calls.
                snap = tracker.snapshot()
                if snap["done"] >= last_log_at + 100:
                    last_log_at = (snap["done"] // 100) * 100
                    cache_state, cache_stats = cache_gate.evaluate()
                    ratio = cache_stats.get("cache_hit_ratio", 0.0)
                    ceiling = cache_stats.get("theoretical_ceiling")
                    ceiling_str = f"ceiling={ceiling:.2f}" if ceiling is not None else "ceiling=?"
                    print(
                        f"  [{snap['done']:>4}/{len(llm_todo)}] "
                        f"elapsed={snap['elapsed_s']:>5.1f}s "
                        f"rpm_last_100={snap['rpm_last_100']:.0f} "
                        f"rpm_total={snap['rpm_total']:.0f} "
                        f"workers={max_workers} max_rpm={max_rpm} "
                        f"429={snap['rate_limit_429']} "
                        f"cache_hit={ratio:.2f} {ceiling_str}",
                        flush=True,
                    )
                    if cache_state == "halt":
                        print(
                            f"\n!!! HARD HALT — cache_hit_ratio={ratio:.2f}; "
                            f"reason={cache_stats.get('reason', '?')} after "
                            f"{snap['done']} calls. Aborting.",
                            file=sys.stderr,
                        )
                        halt_event.set()
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    elif cache_state == "warn":
                        gap = (ceiling - ratio) if ceiling is not None else 0
                        print(
                            f"  ⚠ cache_hit_ratio={ratio:.3f} is below "
                            f"theoretical_ceiling={ceiling:.3f} by "
                            f"{gap:.3f} (tolerance "
                            f"{CACHE_CEILING_TOLERANCE:.2f}). Run continues.",
                            flush=True,
                        )
    finally:
        fh.close()

    if halt_event.is_set():
        print("\nRun aborted due to cache-hit gate.", file=sys.stderr)
        return 2

    snap = tracker.snapshot()
    final_cache = client.cache_stats
    print()
    print("=== Stage 14.0 summary ===")
    print(f"  Mode:                {mode}")
    print(f"  Deterministic rows:  {len(det_todo)}")
    print(f"  LLM done:            {snap['done']}")
    print(f"  LLM failed:          {snap['failed']}  (429: {snap['rate_limit_429']})")
    print(f"  Wall:                {snap['elapsed_s']:.1f}s")
    print(f"  Realized RPM total:  {snap['rpm_total']:.0f}")
    print(f"  Cache hit ratio:     {final_cache.get('cache_hit_ratio', 0):.2%}")
    print(f"    cache_creation:    {final_cache.get('cache_creation_tokens', 0)}")
    print(f"    cache_read:        {final_cache.get('cache_read_tokens', 0)}")
    print(f"    uncached_input:    {final_cache.get('uncached_input_tokens', 0)}")
    print(f"  Sidecar:             {sidecar_path}")
    if is_pilot:
        print()
        print(f"  Run pilot HTML generator:")
        print(f"    .venv/bin/python build/14_1_pilot_html.py")
    return 0 if n_failed_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
