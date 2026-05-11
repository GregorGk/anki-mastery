"""Stage 12 / Step 0 — Classify `usage_hint` for verb (and verb-like idiom) rows.

For each verb/idiom row in `data/06-final.tsv`, ask Claude Sonnet 4.6
(via Anthropic Tool Use, with prompt caching) whether the row contains
a learner trap worth flagging in a short `usage_hint` string.
Most rows return an empty hint — only ~10–25% should get one.

The output is a sidecar TSV (`data/_usage_hints.tsv`) that `derive_final.py`
joins at TSV-build time to append the new `usage_hint` column at the end
of `06-final.tsv`. This script is the only place that talks to the LLM.

Modes:
    --pilot         render ~100 representative rows (known-trap verbs +
                    known-non-trap verbs + low-rank random sample)
    --full          process every verb/idiom row not already in the sidecar
    --dry-run       print preflight (token estimate + cost) and exit

Usage:
    .venv/bin/python build/12_0_usage_hint_classifier.py --pilot
    .venv/bin/python build/12_0_usage_hint_classifier.py --full --yes
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402
from build.lib.usage_hint_rules import apply_post_process  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
PROMPTS_DIR = REPO_ROOT / "build" / "prompts"

FINAL_TSV = DATA_DIR / "06-final.tsv"
SIDECAR_TSV = DATA_DIR / "_usage_hints.tsv"
PROMPT_PATH = PROMPTS_DIR / "usage_hint.md"
AUDIT_JSONL = AUDIT_DIR / "12_usage_hints.jsonl"
LOG_PATH = AUDIT_DIR / "12_0_usage_hint.log"

MODEL_ID = "claude-sonnet-4-6"
DEFAULT_CONCURRENCY = 10
ELIGIBLE_POS = ("verb", "idiom")

SIDECAR_FIELDS = [
    "sense_id", "pos", "pt", "usage_hint", "hint_priority", "confidence",
    "reason", "risk_note", "model_id", "generated_at",
]

# Curated lemma allowlist. Only rows whose `pt` is in this set (or whose
# `bp_status` / `tags` flag them) are sent to the LLM. The classifier is
# strict — it may still return `usage_hint=""` for any of these if the
# row's example doesn't actually exercise the trap. This is the precision
# screen the user added in v2 (pilot v1 had no screen and returned 67% hits).
USAGE_HINT_ALLOWLIST = {
    # Tier 1 — required preposition / verb + a + infinitive connector
    "gostar", "precisar", "depender", "concordar", "discordar", "lidar",
    "participar", "pensar", "acreditar", "desistir", "cuidar",
    "começar", "parar", "aprender", "ensinar", "ajudar", "obrigar",
    # Tier 1 — false-friend / reflexive contrast
    "demitir", "lembrar", "esquecer", "pretender", "assistir", "atender",
    # Tier 1 — high-frequency two-verb traps
    "haver", "ser", "estar", "saber", "conhecer", "ficar", "tornar-se",
    # Tier 1 — preposition-of-place
    "entrar", "sair",
    # Tier 2 — keep if example supports
    "tomar", "marcar", "fechar", "assinar", "passar", "acabar", "voltar",
    "sentir", "contar", "tratar", "negociar", "investir",
    "deixar", "levar", "encontrar", "partir", "achar", "dever", "continuar",
    "procurar",
}

# Pre-screen tags: rows tagged this way are interesting even if the lemma
# isn't on the allowlist (handful of false-friend / reflexive verbs the
# pipeline already flagged in Stage 1 enrichment).
INTERESTING_TAGS = ("#false-friend", "#reflexive")
INTERESTING_BP_STATUS = {"false_friend", "uncommon", "nsfw"}


TOOL_NAME = "record_usage_hint"
TOOL_DESCRIPTION = (
    "Record the optional usage hint for one verb/idiom row. Leave usage_hint "
    "empty (hint_priority='omit') unless this row's example_pt directly "
    "exercises a genuine BP-vs-English construction trap. Be ruthless — the "
    "default is omit."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "usage_hint": {
            "type": "string",
            "description": (
                "Compact one-line hint in English (<= 70 chars when possible). "
                "Empty string when hint_priority='omit'. No quotes, no markdown, "
                "no parenthetical asides."
            ),
        },
        "hint_priority": {
            "type": "string",
            "enum": ["essential", "useful", "omit"],
            "description": (
                "essential = this row teaches a high-frequency English-speaker "
                "trap the bare lemma+example would not prevent. "
                "useful = collocation directly exercised by THIS row's example. "
                "omit = no hint needed (empty usage_hint)."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": (
                "Confidence in the decision itself, NOT 'is hint needed'. "
                "If you're sure this row should be empty, return omit + high."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One short sentence justifying the decision.",
        },
    },
    "required": ["usage_hint", "hint_priority", "confidence", "reason"],
}


@dataclass
class Row:
    sense_id: str
    rank: int
    pt: str
    pt_display: str
    pos: str
    pt_type: str
    gender: str
    en_primary: str
    en_all: str
    annotation: str
    bp_status: str
    tags: str
    example_pt: str
    example_en: str
    target_word_used: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_eligible_rows() -> list[Row]:
    out: list[Row] = []
    for r in read_tsv(FINAL_TSV):
        if r.get("pos", "") not in ELIGIBLE_POS:
            continue
        out.append(Row(
            sense_id=r["sense_id"],
            rank=int(r.get("rank") or "0"),
            pt=r.get("pt", ""),
            pt_display=r.get("pt_display", ""),
            pos=r.get("pos", ""),
            pt_type=r.get("pt_type", ""),
            gender=r.get("gender", ""),
            en_primary=r.get("en_primary", ""),
            en_all=r.get("en_all", ""),
            annotation=r.get("annotation", ""),
            bp_status=r.get("bp_status", ""),
            tags=r.get("tags", ""),
            example_pt=r.get("example_pt", ""),
            example_en=r.get("example_en", ""),
            target_word_used=r.get("target_word_used", ""),
        ))
    return out


def _read_existing_sidecar() -> dict[str, dict]:
    if not SIDECAR_TSV.exists():
        return {}
    return {r["sense_id"]: r for r in read_tsv(SIDECAR_TSV)}


def _is_candidate(r: Row) -> bool:
    """Precision filter: only send rows that look likely to have a real trap.

    The LLM is the final-quality check on this set; it can still return
    `hint_priority="omit"` for any row whose example doesn't exercise the
    trap (Rule 1 — example-anchored).
    """
    if r.pos not in ELIGIBLE_POS:
        return False
    if r.pt in USAGE_HINT_ALLOWLIST:
        return True
    if r.bp_status in INTERESTING_BP_STATUS:
        return True
    if any(t in r.tags for t in INTERESTING_TAGS):
        return True
    return False


def _select_candidates(rows: list[Row]) -> list[Row]:
    """Apply the precision filter; return rows ordered by sense_id for
    deterministic, audit-friendly processing."""
    picked = [r for r in rows if _is_candidate(r)]
    picked.sort(key=lambda r: r.sense_id)
    return picked


def _build_user_message(r: Row) -> str:
    return (
        f"sense_id: {r.sense_id}\n"
        f"rank: {r.rank}\n"
        f"pt: {r.pt}\n"
        f"pt_display: {r.pt_display}\n"
        f"pos: {r.pos}\n"
        f"pt_type: {r.pt_type}\n"
        f"gender: {r.gender}\n"
        f"en_primary: {r.en_primary}\n"
        f"en_all: {r.en_all}\n"
        f"annotation: {r.annotation}\n"
        f"bp_status: {r.bp_status}\n"
        f"tags: {r.tags}\n"
        f"example_pt: {r.example_pt}\n"
        f"example_en: {r.example_en}\n"
        f"target_word_used: {r.target_word_used}"
    )


def _classify_one(*, client: AnthropicClient, system: str, row: Row) -> dict:
    """One LLM call + deterministic post-processing. Returns dict ready for
    the sidecar (usage_hint, hint_priority, confidence, reason, risk_note)."""
    user_msg = _build_user_message(row)
    decision = client.call_tool(
        system=system,
        user_message=user_msg,
        tool_name=TOOL_NAME,
        tool_input_schema=TOOL_INPUT_SCHEMA,
        tool_description=TOOL_DESCRIPTION,
        max_tokens=256,
        stage="12_0_usage_hint",
        provenance_key=row.sense_id,
    )
    hint = (decision.get("usage_hint") or "").strip().replace("\n", " ")
    priority = decision.get("hint_priority") or "omit"
    # Hard rule: priority=omit ⇒ hint must be empty.
    if priority == "omit":
        hint = ""
    # Build dict for the rules module to consume + mutate.
    payload = {
        "pt": row.pt,
        "usage_hint": hint,
        "hint_priority": priority,
        "confidence": decision.get("confidence") or "medium",
        "reason": (decision.get("reason") or "").strip().replace("\n", " "),
        "risk_note": "",
    }
    return apply_post_process(payload)


def _write_sidecar_row(row_out: dict, fh, writer_lock: threading.Lock) -> None:
    with writer_lock:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL)
        w.writerow(row_out)
        fh.flush()


def _open_sidecar_appender(existing: dict[str, dict]):
    """Open _usage_hints.tsv for appending, writing header if it's new."""
    SIDECAR_TSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not SIDECAR_TSV.exists() or SIDECAR_TSV.stat().st_size == 0
    fh = SIDECAR_TSV.open("a", encoding="utf-8", newline="")
    if is_new:
        w = csv.DictWriter(fh, fieldnames=SIDECAR_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        fh.flush()
    return fh


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="Pre-flight only; no LLM calls")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--yes", action="store_true", help="Skip GO prompt")
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    if not PROMPT_PATH.exists():
        print(f"ERROR: prompt missing at {PROMPT_PATH}", file=sys.stderr)
        return 1
    system = PROMPT_PATH.read_text(encoding="utf-8")

    eligible = _load_eligible_rows()
    by_pos: dict[str, int] = {}
    for r in eligible:
        by_pos[r.pos] = by_pos.get(r.pos, 0) + 1

    existing = _read_existing_sidecar()

    # Precision filter — see _is_candidate(). All rows passing the filter
    # are sent to the LLM; nothing is sampled or sliced. The classifier is
    # strict and will return hint_priority='omit' for off-example cases.
    filtered = _select_candidates(eligible)
    # Skip rows already in the sidecar (resume support).
    candidates = [r for r in filtered if r.sense_id not in existing]
    pre_skip = len(filtered) - len(candidates)
    if pre_skip:
        print(f"[resume] skipping {pre_skip} already in sidecar")
    scope_label = "filtered candidates"

    # Pre-flight
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    # Token estimate: system ~3500 input tokens (one-time cache create, then
    # cache reads at 10% rate); per-call user msg ~200 input tokens + ~50
    # output. Sonnet 4.6 list: $3/M input, $15/M output, cache_read $0.30/M.
    n = len(candidates)
    cache_create_in = 3500
    cache_read_in = max(0, n - 1) * 3500
    user_in = n * 200
    out_t = n * 50
    cost = (
        cache_create_in / 1_000_000 * 3.75   # cache write (~25% surcharge)
        + cache_read_in / 1_000_000 * 0.30
        + user_in / 1_000_000 * 3.0
        + out_t / 1_000_000 * 15.0
    )
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  STAGE 12 — USAGE HINT CLASSIFIER PRE-FLIGHT                        │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Mode:         {scope_label:<53}│")
    print(f"│  Model:        {args.model:<53}│")
    print(f"│  Eligible:     {len(eligible):>4}   (verbs={by_pos.get('verb',0)}, idioms={by_pos.get('idiom',0)}) "
          f"                {'':<10}│")
    print(f"│  Already done: {len(existing):>4}                                                    │")
    print(f"│  To classify:  {n:>4}                                                    │")
    print(f"│  Concurrency:  {args.concurrency:<53}│")
    print(f"│  Cost (est):   ${cost:>5.2f}                                              │")
    print(f"│  Sidecar:      {SIDECAR_TSV.name:<53}│")
    print(f"│  Audit log:    {AUDIT_JSONL.name:<53}│")
    print("└─────────────────────────────────────────────────────────────────────┘")

    if args.dry_run:
        print("\n--dry-run: stopping.")
        return 0
    if n == 0:
        print("Nothing to do — all eligible senses already in sidecar.")
        return 0

    if not args.yes:
        sys.stdout.write("Type GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # Run
    client = AnthropicClient(
        model=args.model,
        audit_path=AUDIT_JSONL,
        enable_caching=True,
    )
    fh = _open_sidecar_appender(existing)
    writer_lock = threading.Lock()

    def _worker(r: Row) -> tuple[Row, dict | None, str]:
        try:
            d = _classify_one(client=client, system=system, row=r)
            return r, d, ""
        except Exception as exc:  # noqa: BLE001
            return r, None, f"{type(exc).__name__}: {exc}"

    t_start = time.time()
    n_done = 0
    n_hint = 0
    n_empty = 0
    n_failed = 0
    print()
    print(f"Processing {n} rows at concurrency={args.concurrency}…")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_worker, r): r for r in candidates}
            for fut in as_completed(futs):
                row, decision, err = fut.result()
                n_done += 1
                if err:
                    n_failed += 1
                    print(f"[{n_done:>4}/{n}] FAIL {row.sense_id} {row.pt}  {err}",
                          flush=True)
                    continue
                hint = decision["usage_hint"]
                conf = decision["confidence"]
                priority = decision["hint_priority"]
                pri_tag = {"essential": "ESS ", "useful": "USE ", "omit": "    "}.get(priority, "    ")
                print(f"[{n_done:>4}/{n}] {pri_tag} {row.sense_id} {row.pt:<14} "
                      f"[{conf}] {hint}", flush=True)
                if hint:
                    n_hint += 1
                else:
                    n_empty += 1
                _write_sidecar_row({
                    "sense_id": row.sense_id,
                    "pos": row.pos,
                    "pt": row.pt,
                    "usage_hint": hint,
                    "hint_priority": decision["hint_priority"],
                    "confidence": conf,
                    "reason": decision["reason"],
                    "risk_note": decision.get("risk_note", ""),
                    "model_id": args.model,
                    "generated_at": _now_iso(),
                }, fh, writer_lock)
    finally:
        fh.close()

    elapsed = time.time() - t_start
    stats = client.cache_stats
    # Re-read sidecar to get priority breakdown
    pri_counts: dict[str, int] = {"essential": 0, "useful": 0, "omit": 0}
    if SIDECAR_TSV.exists():
        for r in read_tsv(SIDECAR_TSV):
            p = r.get("hint_priority", "")
            if p in pri_counts:
                pri_counts[p] += 1
    print()
    print("=== Stage 12.0 summary ===")
    print(f"  Done:              {n_done}")
    print(f"  With hint:         {n_hint} ({100*n_hint/max(n_done,1):.0f}%)")
    print(f"  Empty:             {n_empty}")
    print(f"  Priority (cum):    essential={pri_counts['essential']}  "
          f"useful={pri_counts['useful']}  omit={pri_counts['omit']}")
    print(f"  Failed:            {n_failed}")
    print(f"  Wall:              {elapsed:.1f}s ({elapsed/max(n_done,1):.2f}s/clip)")
    print(f"  Cache hit:         {stats['cache_hit_ratio']:.0%} "
          f"(create={stats['cache_creation_tokens']}, "
          f"read={stats['cache_read_tokens']})")
    print(f"  Sidecar:           {SIDECAR_TSV}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
