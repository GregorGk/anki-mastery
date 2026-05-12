"""Stage 15 / Step 0 — BP-validity / register / risk_flags classifier.

For every row in `data/06-final.tsv`, write a sidecar row with:
  - bp_validity (7-value enum)
  - register    (10-value enum)
  - risk_flags  (pipe-separated, 20-value allowlist)
  - risk_note   (≤180 chars; Stage-12 carry-forward when LLM omits)

Per-row precedence (locked from plan):
  1. MANUAL  — data/_manual_risk_register.tsv (sense_id keyed). Wins.
  2. LLM     — candidate filter (is_candidate). ~317 rows.
  3. DEFAULT — deterministic standard/neutral/none + stage12 risk_note carry.

Modes:
  --dry-run                     preflight — no LLM calls, no sidecar writes
  --pilot --seed N              stratified pilot: all candidates + 100 random
                                "audit-only" defaults; separate sidecar paths
  default                       full run; resumes from existing sidecar

CLI flags (env fallback):
  --max-workers N      (ANTHROPIC_MAX_WORKERS)    default 25
  --max-rpm N          (ANTHROPIC_MAX_RPM)        default 3200 (0 = unbounded)
  --max-retries N      (ANTHROPIC_MAX_RETRIES)    default 8
  --yes                skip GO prompt
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.rate_limit import SlidingWindowRateLimiter  # noqa: E402
from build.lib.risk_register_rules import (  # noqa: E402
    ALLOWED_BP_VALIDITY,
    ALLOWED_REGISTER,
    ALLOWED_RISK_FLAGS,
    DEFAULT_BP_VALIDITY,
    DEFAULT_REGISTER,
    DEFAULT_RISK_FLAGS,
    RISK_NOTE_MAX_CHARS,
    apply_post_process,
    is_candidate,
)
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
REPORTS_DIR = REPO_ROOT / "reports"
PROMPTS_DIR = REPO_ROOT / "build" / "prompts"

FINAL_TSV = DATA_DIR / "06-final.tsv"
USAGE_HINTS_TSV = DATA_DIR / "_usage_hints.tsv"
MANUAL_TSV = DATA_DIR / "_manual_risk_register.tsv"
SIDECAR_TSV_FULL = DATA_DIR / "_risk_register.tsv"
SIDECAR_TSV_PILOT = DATA_DIR / "_risk_register.pilot.tsv"
AUDIT_JSONL_FULL = AUDIT_DIR / "15_risk_register.jsonl"
AUDIT_JSONL_PILOT = AUDIT_DIR / "15_risk_register.pilot.jsonl"
PROMPT_PATH = PROMPTS_DIR / "risk_register.md"

MODEL_ID = "claude-sonnet-4-6"
DEFAULT_MAX_WORKERS = 25
DEFAULT_MAX_RPM = 3200
DEFAULT_MAX_RETRIES = 8

# Cache-hit gate (same theoretical-ceiling pattern as Stage 14).
CACHE_HARD_HALT_FLOOR = 0.75
CACHE_CEILING_TOLERANCE = 0.03
CACHE_GATE_AFTER_N_CALLS = 100

PILOT_RANDOM_NORMAL_COUNT = 100

# Sidecar schema (19 cols — keeps example_pt/en for self-contained review).
SIDECAR_FIELDS = [
    "sense_id", "rank", "pt", "pos", "en_primary",
    "example_pt", "example_en",
    "bp_status", "tags", "old_risk_note",
    "bp_validity", "register", "risk_flags", "risk_note",
    "confidence", "reason",
    "source", "model_id", "generated_at",
]

# Manual override file schema.
MANUAL_FIELDS = [
    "sense_id", "bp_validity", "register", "risk_flags", "risk_note", "reason",
]

TOOL_NAME = "record_risk_register"
TOOL_DESCRIPTION = (
    "Record bp_validity / register / risk_flags / risk_note for one BP "
    "sense. Default to standard / neutral / none / \"\" when no learner "
    "safety issue applies. Only warn when the learner needs to change "
    "behavior. Classify the specific sense, not just the lemma."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "bp_validity": {"type": "string", "enum": list(ALLOWED_BP_VALIDITY)},
        "register":    {"type": "string", "enum": list(ALLOWED_REGISTER)},
        "risk_flags":  {
            "type": "string",
            "description": (
                "'none' OR pipe-separated allowed flags. Allowlist: "
                + ", ".join(ALLOWED_RISK_FLAGS)
            ),
        },
        "risk_note": {
            "type": "string",
            "description": f"≤{RISK_NOTE_MAX_CHARS} chars; empty when risk_flags=none.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason":     {"type": "string"},
    },
    "required": ["bp_validity", "register", "risk_flags", "risk_note",
                 "confidence", "reason"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Data classes
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
    annotation: str
    bp_status: str
    tags: str
    usage_hint: str
    old_risk_note: str
    example_pt: str
    example_en: str
    is_audit_only: bool = False  # set True for pilot random-normals subset


@dataclass
class ManualOverride:
    sense_id: str
    bp_validity: str
    register: str
    risk_flags: str
    risk_note: str
    reason: str


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
            annotation=(r.get("annotation") or "").strip(),
            bp_status=(r.get("bp_status") or "").strip(),
            tags=(r.get("tags") or "").strip(),
            usage_hint=(r.get("usage_hint") or "").strip(),
            old_risk_note=(r.get("risk_note") or "").strip(),
            example_pt=(r.get("example_pt") or "").strip(),
            example_en=(r.get("example_en") or "").strip(),
        ))
    return out


def _load_manual_overrides() -> dict[str, ManualOverride]:
    if not MANUAL_TSV.exists():
        return {}
    out: dict[str, ManualOverride] = {}
    for r in read_tsv(MANUAL_TSV):
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            continue
        out[sid] = ManualOverride(
            sense_id=sid,
            bp_validity=(r.get("bp_validity") or "").strip(),
            register=(r.get("register") or "").strip(),
            risk_flags=(r.get("risk_flags") or "").strip(),
            risk_note=(r.get("risk_note") or "").strip(),
            reason=(r.get("reason") or "").strip(),
        )
    return out


def _load_usage_hints() -> dict[str, str]:
    """Return {sense_id: usage_hint_text} from data/_usage_hints.tsv."""
    if not USAGE_HINTS_TSV.exists():
        return {}
    out: dict[str, str] = {}
    for r in read_tsv(USAGE_HINTS_TSV):
        sid = (r.get("sense_id") or "").strip()
        hint = (r.get("usage_hint") or "").strip()
        if sid and hint:
            out[sid] = hint
    return out


def _read_existing_sidecar(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(path)}


# --------------------------------------------------------------------------- #
# User-message builder
# --------------------------------------------------------------------------- #


def _build_user_message(r: Row) -> str:
    """Sense-first; example fields last (Stage 14 pattern)."""
    return (
        f"sense_id: {r.sense_id}\n"
        f"rank: {r.rank}\n"
        f"pt: {r.pt}\n"
        f"pt_display_safe: {r.pt_display_safe}\n"
        f"pos: {r.pos}\n"
        f"pt_type: {r.pt_type}\n"
        f"en_primary: {r.en_primary}\n"
        f"en_all: {r.en_all}\n"
        f"annotation: {r.annotation}\n"
        f"bp_status: {r.bp_status}\n"
        f"tags: {r.tags}\n"
        f"usage_hint: {r.usage_hint}\n"
        f"old_risk_note: {r.old_risk_note}\n"
        f"example_pt: {r.example_pt}\n"
        f"example_en: {r.example_en}"
    )


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
# Progress tracker + Cache gate (lifted from Stage 14)
# --------------------------------------------------------------------------- #


class ProgressTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.rate_limit_429 = 0
        self.completed_ts: deque[float] = deque()
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
                "done": self.done, "failed": self.failed,
                "elapsed_s": elapsed,
                "rpm_last_100": rpm_last_100, "rpm_total": rpm_total,
                "rate_limit_429": self.rate_limit_429,
            }


class CacheGate:
    """Theoretical-ceiling cache-hit gate (Stage 14 pattern)."""

    def __init__(self, client: AnthropicClient) -> None:
        self.client = client
        self.baseline_cache_size = 0

    def record_baseline_after_warmup(self) -> int:
        stats = self.client.cache_stats
        self.baseline_cache_size = stats.get("cache_creation_tokens", 0)
        return self.baseline_cache_size

    def evaluate(self) -> tuple[str, dict]:
        stats = dict(self.client.cache_stats)
        ratio = stats.get("cache_hit_ratio", 0.0)
        calls = stats.get("calls", 0)
        if calls < CACHE_GATE_AFTER_N_CALLS:
            return "ok", stats
        cache_read = stats.get("cache_read_tokens", 0)
        if cache_read == 0:
            stats["reason"] = "no_cache_reads"
            return "halt", stats
        if ratio < CACHE_HARD_HALT_FLOOR:
            stats["reason"] = f"ratio_below_floor_{CACHE_HARD_HALT_FLOOR}"
            return "halt", stats
        if self.baseline_cache_size > 0:
            avg_uncached = stats.get("uncached_input_tokens", 0) / max(calls, 1)
            ceiling = (
                self.baseline_cache_size
                / (self.baseline_cache_size + avg_uncached)
            )
            stats["theoretical_ceiling"] = ceiling
            stats["baseline_cache_size"] = self.baseline_cache_size
            stats["avg_uncached_per_call"] = avg_uncached
            if ratio < ceiling - CACHE_CEILING_TOLERANCE:
                stats["reason"] = (
                    f"below_ceiling: actual={ratio:.3f} ceiling={ceiling:.3f}"
                )
                return "warn", stats
        return "ok", stats


# --------------------------------------------------------------------------- #
# One-row classification (with rate-limit + post-process)
# --------------------------------------------------------------------------- #


def _classify_one(
    *,
    client: AnthropicClient,
    system: str,
    row: Row,
    limiter: SlidingWindowRateLimiter,
    stage_tag: str,
) -> dict:
    limiter.wait_before_call()
    user_msg = _build_user_message(row)
    decision = client.call_tool(
        system=system,
        user_message=user_msg,
        tool_name=TOOL_NAME,
        tool_input_schema=TOOL_INPUT_SCHEMA,
        tool_description=TOOL_DESCRIPTION,
        max_tokens=256,
        stage=stage_tag,
        provenance_key=row.sense_id,
    )
    payload = {
        "bp_validity": (decision.get("bp_validity") or "").strip(),
        "register":    (decision.get("register") or "").strip(),
        "risk_flags":  (decision.get("risk_flags") or "").strip(),
        "risk_note":   (decision.get("risk_note") or "").strip(),
        "confidence":  (decision.get("confidence") or "").strip().lower(),
        "reason":      (decision.get("reason") or "").strip().replace("\n", " "),
    }
    apply_post_process(payload)
    # If LLM returned empty risk_note BUT Stage-12 had one, carry it forward.
    if not payload["risk_note"] and row.old_risk_note:
        payload["risk_note"] = row.old_risk_note
    return payload


# --------------------------------------------------------------------------- #
# Pilot stratification (audit-only random normals)
# --------------------------------------------------------------------------- #


def _select_pilot_audit_normals(
    default_rows: list[Row], seed: int, n: int = PILOT_RANDOM_NORMAL_COUNT,
) -> list[Row]:
    """Sample n random default rows and mark them is_audit_only=True."""
    if not default_rows or n <= 0:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(default_rows, min(n, len(default_rows)))
    for r in chosen:
        r.is_audit_only = True
    return chosen


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def _print_preflight(
    *, mode: str, total_rows: int, n_manual: int, n_default: int,
    n_llm: int, max_workers: int, max_rpm: int, max_retries: int,
    sidecar: Path, audit: Path, system_prompt: str, n_audit_only: int = 0,
) -> None:
    sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    est_cost = n_llm * 0.003
    border = "┌" + "─" * 72 + "┐"
    bottom = "└" + "─" * 72 + "┘"
    sep    = "├" + "─" * 72 + "┤"
    print(border)
    print(f"│  STAGE 15 — RISK / REGISTER CLASSIFIER PREFLIGHT" + " " * 24 + "│")
    print(sep)
    print(f"│  Mode:                  {mode:<48}│")
    print(f"│  Model:                  {MODEL_ID:<47}│")
    print(f"│  System prompt path:    {str(PROMPT_PATH.relative_to(REPO_ROOT)):<48}│")
    print(f"│  System prompt chars:   {len(system_prompt):<48}│")
    print(f"│  System prompt hash:    {sys_hash[:48]:<48}│")
    print(f"│  Total rows:            {total_rows:<48}│")
    print(f"│  Manual overrides:      {n_manual:<48}│")
    print(f"│  Deterministic default: {n_default:<48}│")
    print(f"│  LLM candidates:        {n_llm:<48}│")
    if n_audit_only:
        print(f"│    (incl. audit-only normals): {n_audit_only:<41}│")
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
    p.add_argument("--pilot", action="store_true",
                   help="Pilot mode: candidates + 100 audit-only normals; "
                        "writes to data/_risk_register.pilot.tsv.")
    p.add_argument("--seed", type=int, default=15,
                   help="Random seed for pilot audit-only sample (default 15).")
    p.add_argument("--max-workers", type=int, default=0,
                   help="Override default workers. Env: ANTHROPIC_MAX_WORKERS.")
    p.add_argument("--max-rpm", type=int, default=-1,
                   help=f"RPM cap (default {DEFAULT_MAX_RPM}; 0 = unbounded). "
                        "Env: ANTHROPIC_MAX_RPM.")
    p.add_argument("--max-retries", type=int, default=0,
                   help=f"Per-call retry cap (default {DEFAULT_MAX_RETRIES}). "
                        "Env: ANTHROPIC_MAX_RETRIES.")
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

    is_pilot = args.pilot
    mode = f"PILOT (seed={args.seed})" if is_pilot else "FULL RUN"
    sidecar_path = SIDECAR_TSV_PILOT if is_pilot else SIDECAR_TSV_FULL
    audit_path   = AUDIT_JSONL_PILOT if is_pilot else AUDIT_JSONL_FULL

    max_workers = args.max_workers or _env_int("ANTHROPIC_MAX_WORKERS", DEFAULT_MAX_WORKERS)
    max_rpm = args.max_rpm if args.max_rpm >= 0 else _env_int("ANTHROPIC_MAX_RPM", DEFAULT_MAX_RPM)
    max_retries = args.max_retries or _env_int("ANTHROPIC_MAX_RETRIES", DEFAULT_MAX_RETRIES)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load inputs -----
    rows = _load_rows()
    manual_overrides = _load_manual_overrides()
    manual_sids = frozenset(manual_overrides.keys())
    usage_hints = _load_usage_hints()

    # ----- Partition: manual / candidate / default -----
    manual_rows: list[Row] = []
    candidate_rows: list[Row] = []
    default_rows: list[Row] = []
    for r in rows:
        if r.sense_id in manual_sids:
            manual_rows.append(r)
            continue
        # Build a row dict for is_candidate
        if is_candidate({
            "sense_id":   r.sense_id,
            "bp_status":  r.bp_status,
            "annotation": r.annotation,
            "tags":       r.tags,
            "pt":         r.pt,
            "risk_note":  r.old_risk_note,
        }, usage_hint=usage_hints.get(r.sense_id, ""), manual_override_sense_ids=manual_sids):
            candidate_rows.append(r)
        else:
            default_rows.append(r)

    audit_only_rows: list[Row] = []
    if is_pilot:
        audit_only_rows = _select_pilot_audit_normals(
            default_rows, seed=args.seed, n=PILOT_RANDOM_NORMAL_COUNT,
        )

    # Resume support — read existing sidecar, skip processed sense_ids.
    existing = _read_existing_sidecar(sidecar_path)
    manual_todo = [r for r in manual_rows if r.sense_id not in existing]
    candidate_todo = [r for r in candidate_rows if r.sense_id not in existing]
    audit_only_todo = [r for r in audit_only_rows if r.sense_id not in existing]
    default_todo = [
        r for r in default_rows
        if r.sense_id not in existing and r.sense_id not in {x.sense_id for x in audit_only_todo}
    ]

    if is_pilot:
        n_llm_pilot = len(candidate_todo) + len(audit_only_todo)
        _print_preflight(
            mode=mode, total_rows=len(rows),
            n_manual=len(manual_todo),
            n_default=len(default_todo),
            n_llm=n_llm_pilot,
            n_audit_only=len(audit_only_todo),
            max_workers=max_workers, max_rpm=max_rpm, max_retries=max_retries,
            sidecar=sidecar_path, audit=audit_path,
            system_prompt=system_prompt,
        )
    else:
        _print_preflight(
            mode=mode, total_rows=len(rows),
            n_manual=len(manual_todo),
            n_default=len(default_todo),
            n_llm=len(candidate_todo),
            max_workers=max_workers, max_rpm=max_rpm, max_retries=max_retries,
            sidecar=sidecar_path, audit=audit_path,
            system_prompt=system_prompt,
        )

    if args.dry_run:
        print("\n--dry-run: stopping (no LLM calls, no sidecar writes).")
        return 0

    llm_todo = candidate_todo + audit_only_todo if is_pilot else candidate_todo
    if not llm_todo and not manual_todo and not default_todo:
        print("\nNothing to do — every row already in sidecar.")
        return 0

    if not args.yes and llm_todo:
        sys.stdout.write("\nType GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    fh = _open_sidecar_appender(sidecar_path)
    writer_lock = threading.Lock()
    ts0 = _now_iso()

    # ----- Manual override writes ------------------------------------------
    for r in manual_todo:
        mv = manual_overrides[r.sense_id]
        payload = {
            "bp_validity": mv.bp_validity,
            "register":    mv.register,
            "risk_flags":  mv.risk_flags,
            "risk_note":   mv.risk_note,
            "confidence":  "high",
            "reason":      mv.reason or "manual override",
        }
        apply_post_process(payload)
        _write_sidecar_row({
            "sense_id":      r.sense_id,
            "rank":          r.rank,
            "pt":            r.pt,
            "pos":           r.pos,
            "en_primary":    r.en_primary,
            "example_pt":    r.example_pt,
            "example_en":    r.example_en,
            "bp_status":     r.bp_status,
            "tags":          r.tags,
            "old_risk_note": r.old_risk_note,
            "bp_validity":   payload["bp_validity"],
            "register":      payload["register"],
            "risk_flags":    payload["risk_flags"],
            "risk_note":     payload["risk_note"],
            "confidence":    payload["confidence"],
            "reason":        payload["reason"],
            "source":        "manual",
            "model_id":      "",
            "generated_at":  ts0,
        }, fh, writer_lock)
    if manual_todo:
        print(f"\nWrote {len(manual_todo)} manual-override rows (no API calls).")

    # ----- Deterministic-default writes ------------------------------------
    if not is_pilot:
        # Full-run: write every default row immediately.
        n_carried = 0
        for r in default_todo:
            risk_note = r.old_risk_note     # carry forward Stage 12 note
            if risk_note:
                n_carried += 1
            _write_sidecar_row({
                "sense_id":      r.sense_id,
                "rank":          r.rank,
                "pt":            r.pt,
                "pos":           r.pos,
                "en_primary":    r.en_primary,
                "example_pt":    r.example_pt,
                "example_en":    r.example_en,
                "bp_status":     r.bp_status,
                "tags":          r.tags,
                "old_risk_note": r.old_risk_note,
                "bp_validity":   DEFAULT_BP_VALIDITY,
                "register":      DEFAULT_REGISTER,
                "risk_flags":    DEFAULT_RISK_FLAGS,
                "risk_note":     risk_note,
                "confidence":    "high",
                "reason":        "no risk signal",
                "source":        "deterministic_default",
                "model_id":      "",
                "generated_at":  ts0,
            }, fh, writer_lock)
        if default_todo:
            print(f"Wrote {len(default_todo)} deterministic-default rows "
                  f"(carried {n_carried} Stage-12 risk_note(s) forward).")
    else:
        # Pilot: do NOT write default rows to the pilot sidecar (the pilot
        # focuses on LLM behavior, not default-coverage). The audit-only
        # subset of 100 normals IS sent to the LLM and recorded.
        pass

    # ----- LLM classification ----------------------------------------------
    if not llm_todo:
        fh.close()
        print("\nNo LLM rows to classify.")
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
    stage_tag = "15_risk_register" + ("_pilot" if is_pilot else "")

    print(f"\nLLM run: {len(llm_todo)} rows ({len(audit_only_todo)} audit-only), "
          f"--max-workers={max_workers}, --max-rpm={max_rpm}\n")

    # Cache warmup (sequential 1-call) — same fix from Stage 14.
    warm_row = llm_todo[0]
    print("  Cache warmup (1 sequential call to seed the server cache)...")
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
            "bp_status":     warm_row.bp_status,
            "tags":          warm_row.tags,
            "old_risk_note": warm_row.old_risk_note,
            "bp_validity":   warm_decision["bp_validity"],
            "register":      warm_decision["register"],
            "risk_flags":    warm_decision["risk_flags"],
            "risk_note":     warm_decision["risk_note"],
            "confidence":    warm_decision["confidence"],
            "reason":        warm_decision["reason"],
            "source":        "llm_audit_only" if warm_row.is_audit_only else "llm",
            "model_id":      args.model,
            "generated_at":  _now_iso(),
        }, fh, writer_lock)
        baseline = cache_gate.record_baseline_after_warmup()
        warm_stats = client.cache_stats
        print(f"  Warmup done: cache_creation={warm_stats.get('cache_creation_tokens', 0)} "
              f"cache_read={warm_stats.get('cache_read_tokens', 0)}")
        print(f"  Baseline cached-block size: {baseline} tokens "
              f"(theoretical-ceiling gate uses this as C)")
        llm_todo = llm_todo[1:]
    except Exception as exc:  # noqa: BLE001
        print(f"  Warmup FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)

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

    n_failed_total = 0
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
                    "bp_status":     r.bp_status,
                    "tags":          r.tags,
                    "old_risk_note": r.old_risk_note,
                    "bp_validity":   decision["bp_validity"],
                    "register":      decision["register"],
                    "risk_flags":    decision["risk_flags"],
                    "risk_note":     decision["risk_note"],
                    "confidence":    decision["confidence"],
                    "reason":        decision["reason"],
                    "source":        "llm_audit_only" if r.is_audit_only else "llm",
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
                        print(
                            f"  ⚠ below ceiling; reason={cache_stats.get('reason', '?')}",
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
    print("=== Stage 15.0 summary ===")
    print(f"  Mode:                {mode}")
    print(f"  Manual rows:         {len(manual_todo)}")
    print(f"  Deterministic rows:  {len(default_todo) if not is_pilot else 0}")
    print(f"  LLM done:            {snap['done']}  (incl. warmup)")
    if is_pilot:
        print(f"    audit-only normals: {len(audit_only_todo)}")
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
        print(f"    .venv/bin/python build/15_1_pilot_html.py")
    return 0 if n_failed_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
