"""Stage 17 / Step 0 — Build deterministic ordering tables inside 06-final.sqlite.

Reads:  data/06-final.sqlite        (senses, sense_tags — built by 16_0)
        data/_manual_ordering.tsv   (optional per-sense bucket overrides)
Writes (into the same sqlite): topic_order, anki_ordering
        data/_ordering.tsv          (full technical ordering sidecar)

Ordering = topic_index → topic_bucket → topic_subrank → sense_id, then a
polysemy-spacing pass (Level A: same-topic same-lemma senses pushed onto
later "rounds"). All ranking math is SQL window functions; bucket
assignment is pure Python (build/lib/order_rules.py).

NO LLM. Pure deterministic SQL + Python. Idempotent: drops & recreates
its tables every run. Pipeline order is strict — run
build/16_0_build_sqlite.py FIRST (it WIPES the whole .sqlite file).

Usage:
    .venv/bin/python build/17_0_build_ordering.py
    .venv/bin/python build/17_0_build_ordering.py --db path/to/other.sqlite
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import order_rules  # noqa: E402
from build.lib.topic_tag_rules import ALLOWED_TOPIC_TAGS  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
DB_PATH = DATA / "06-final.sqlite"
MANUAL_ORDERING = DATA / "_manual_ordering.tsv"
ORDERING_TSV = DATA / "_ordering.tsv"

MANUAL_FIELDS = ["sense_id", "topic_bucket", "topic_subrank",
                 "force_after_sense_id", "reason"]

ANKI_ORDERING_COLS = [
    "sense_id", "topic_primary", "topic_index", "topic_bucket",
    "topic_subrank", "strict_topic_order", "polysemy_count_global",
    "polysemy_count_in_topic", "polysemy_round_in_topic",
    "spaced_topic_order", "spacing_action", "order_reason",
]

DDL = """
DROP TABLE IF EXISTS anki_ordering;
DROP TABLE IF EXISTS topic_order;
DROP TABLE IF EXISTS _ordering_stage;

CREATE TABLE topic_order (
  topic_primary TEXT PRIMARY KEY,
  topic_index   INTEGER NOT NULL
);

CREATE TABLE _ordering_stage (
  sense_id       TEXT PRIMARY KEY,
  topic_primary  TEXT NOT NULL,
  topic_index    INTEGER NOT NULL,
  topic_bucket   INTEGER NOT NULL,
  topic_subrank  REAL NOT NULL,
  pt             TEXT NOT NULL,
  spacing_action TEXT NOT NULL,
  order_reason   TEXT NOT NULL
);

CREATE TABLE anki_ordering (
  sense_id                TEXT PRIMARY KEY REFERENCES senses(sense_id),
  topic_primary           TEXT NOT NULL,
  topic_index             INTEGER NOT NULL,
  topic_bucket            INTEGER NOT NULL,
  topic_subrank           REAL NOT NULL,
  strict_topic_order      INTEGER NOT NULL,
  polysemy_count_global   INTEGER NOT NULL,
  polysemy_count_in_topic INTEGER NOT NULL,
  polysemy_round_in_topic INTEGER NOT NULL,
  spaced_topic_order      INTEGER NOT NULL,
  spacing_action          TEXT NOT NULL,
  order_reason            TEXT NOT NULL
);
CREATE INDEX idx_anki_ordering_spaced ON anki_ordering(spaced_topic_order);
CREATE INDEX idx_anki_ordering_strict ON anki_ordering(strict_topic_order);
CREATE INDEX idx_anki_ordering_topic  ON anki_ordering(topic_primary);
"""

WINDOW_INSERT = """
INSERT INTO anki_ordering (
  sense_id, topic_primary, topic_index, topic_bucket, topic_subrank,
  strict_topic_order, polysemy_count_global, polysemy_count_in_topic,
  polysemy_round_in_topic, spaced_topic_order, spacing_action, order_reason
)
WITH ranked AS (
  SELECT sense_id, topic_primary, topic_index, topic_bucket, topic_subrank, pt,
         spacing_action, order_reason,
         ROW_NUMBER() OVER (
           ORDER BY topic_index, topic_bucket, topic_subrank, sense_id
         ) AS strict_topic_order,
         COUNT(*) OVER (PARTITION BY pt) AS polysemy_count_global,
         COUNT(*) OVER (PARTITION BY topic_primary, pt) AS polysemy_count_in_topic,
         ROW_NUMBER() OVER (
           PARTITION BY topic_primary, pt
           ORDER BY topic_bucket, topic_subrank, sense_id
         ) - 1 AS polysemy_round_in_topic
  FROM _ordering_stage
)
SELECT sense_id, topic_primary, topic_index, topic_bucket, topic_subrank,
       strict_topic_order, polysemy_count_global, polysemy_count_in_topic,
       polysemy_round_in_topic,
       ROW_NUMBER() OVER (
         ORDER BY topic_index, polysemy_round_in_topic,
                  topic_bucket, topic_subrank, sense_id
       ) AS spaced_topic_order,
       spacing_action, order_reason
FROM ranked;
"""


class BuildError(SystemExit):
    """Raised for any hard failure — message goes to stderr, exit code 1."""
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (17_0_build_ordering): {msg}")


def _require_base_tables(conn: sqlite3.Connection) -> None:
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for needed in ("senses", "sense_tags", "audio_clips"):
        if needed not in have:
            raise BuildError(
                f"base table '{needed}' missing from {DB_PATH.name}. "
                f"Run build/16_0_build_sqlite.py first.")


def _load_senses(conn: sqlite3.Connection) -> list[dict]:
    """One row per sense: sense_id, pt, pos, topic_primary.
    Asserts the 'exactly one #topic-* tag per sense' invariant."""
    rows = [
        {"sense_id": r[0], "pt": r[1], "pos": r[2] or "", "topic_primary": r[3]}
        for r in conn.execute(
            "SELECT s.sense_id, s.pt, s.pos, t.tag "
            "FROM senses s "
            "JOIN sense_tags t ON t.sense_id = s.sense_id "
            "  AND t.tag LIKE '#topic-%'"
        )
    ]
    n_senses = conn.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
    if len(rows) != n_senses:
        raise BuildError(
            f"topic-tag invariant broken: {len(rows)} sense×topic rows but "
            f"{n_senses} senses — every sense must have exactly one #topic-* tag.")
    return rows


def _load_manual_overrides(path: Path = MANUAL_ORDERING) -> dict[str, dict]:
    """Load + STRICTLY validate data/_manual_ordering.tsv.

    The header must be exactly MANUAL_FIELDS. Any extra/missing/renamed
    column (especially topic_primary) is a hard build failure. Returns
    {sense_id: {topic_bucket, topic_subrank, force_after_sense_id, reason,
    _row_index}} preserving file order via _row_index.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, dialect="excel-tab")
        try:
            header = next(reader)
        except StopIteration:
            return {}
    if header != MANUAL_FIELDS:
        raise BuildError(
            f"{path.name} header must be exactly "
            f"{MANUAL_FIELDS}, got {header}. "
            f"A manual row may never carry/alter topic_primary.")

    out: dict[str, dict] = {}
    for i, r in enumerate(read_tsv(path)):
        sid = (r.get("sense_id") or "").strip()
        if not sid:
            raise BuildError(f"{path.name} row {i + 2}: empty sense_id.")
        if sid in out:
            raise BuildError(f"{path.name}: duplicate sense_id {sid!r}.")
        tb = (r.get("topic_bucket") or "").strip()
        ts = (r.get("topic_subrank") or "").strip()
        out[sid] = {
            "topic_bucket": int(tb) if tb else None,
            "topic_subrank": float(ts) if ts else None,
            "force_after_sense_id": (r.get("force_after_sense_id") or "").strip(),
            "reason": (r.get("reason") or "").strip(),
            "_row_index": i,
        }
    return out


def _compute_stage_rows(
    senses: list[dict], overrides: dict[str, dict]
) -> list[tuple]:
    """Compute (sense_id, topic_primary, topic_index, topic_bucket,
    topic_subrank, pt, spacing_action, order_reason) per sense, applying
    order_rules + manual overrides + force_after epsilon resolution."""
    by_id = {s["sense_id"]: s for s in senses}

    # 1. base bucket assignment from order_rules
    base: dict[str, dict] = {}
    for s in senses:
        bucket, subrank, reason = order_rules.bucket_for(
            s["topic_primary"], s["pt"], s["pos"], s["sense_id"])
        base[s["sense_id"]] = {
            "topic_primary": s["topic_primary"],
            "topic_index": order_rules.topic_index_for(s["topic_primary"]),
            "topic_bucket": bucket,
            "topic_subrank": subrank,
            "pt": s["pt"],
            "spacing_action": "none",
            "order_reason": reason,
        }

    # 2. hard-check every manual sense_id exists
    for sid in overrides:
        if sid not in by_id:
            raise BuildError(
                f"{MANUAL_ORDERING.name}: sense_id {sid!r} not found in "
                f"senses — manual files must be precise.")

    # 3. direct bucket/subrank overrides (non-force-after rows first)
    for sid, ov in overrides.items():
        if ov["force_after_sense_id"]:
            continue
        rec = base[sid]
        if ov["topic_bucket"] is not None:
            rec["topic_bucket"] = ov["topic_bucket"]
        if ov["topic_subrank"] is not None:
            rec["topic_subrank"] = ov["topic_subrank"]
        rec["order_reason"] = f"manual:{ov['reason'] or 'override'}"

    # 4. force_after resolution — epsilon offset after the target
    forced = sorted(
        ((sid, ov) for sid, ov in overrides.items() if ov["force_after_sense_id"]),
        key=lambda kv: kv[1]["_row_index"],
    )
    per_target: Counter = Counter()
    for sid, ov in forced:
        target = ov["force_after_sense_id"]
        if target not in by_id:
            raise BuildError(
                f"{MANUAL_ORDERING.name}: force_after_sense_id {target!r} "
                f"(for {sid!r}) not found in senses.")
        if by_id[target]["topic_primary"] != by_id[sid]["topic_primary"]:
            raise BuildError(
                f"{MANUAL_ORDERING.name}: force_after target {target!r} is in "
                f"topic {by_id[target]['topic_primary']!r} but {sid!r} is in "
                f"{by_id[sid]['topic_primary']!r} — a manual row can never "
                f"push a sense across topic blocks.")
        per_target[target] += 1
        k = per_target[target]
        tgt = base[target]
        rec = base[sid]
        rec["topic_bucket"] = tgt["topic_bucket"]
        rec["topic_subrank"] = tgt["topic_subrank"] + 0.001 * k
        rec["spacing_action"] = f"forced_after:{target}"
        rec["order_reason"] = f"manual:{ov['reason'] or 'force_after'}"

    return [
        (sid, r["topic_primary"], r["topic_index"], r["topic_bucket"],
         r["topic_subrank"], r["pt"], r["spacing_action"], r["order_reason"])
        for sid, r in base.items()
    ]


def _build(conn: sqlite3.Connection, stage_rows: list[tuple]) -> None:
    conn.executescript(DDL)
    conn.executemany(
        "INSERT INTO _ordering_stage (sense_id, topic_primary, topic_index, "
        "topic_bucket, topic_subrank, pt, spacing_action, order_reason) "
        "VALUES (?,?,?,?,?,?,?,?)",
        stage_rows,
    )
    conn.execute(WINDOW_INSERT)
    conn.execute("DROP TABLE _ordering_stage")
    conn.executemany(
        "INSERT INTO topic_order (topic_primary, topic_index) VALUES (?,?)",
        [(topic, idx) for idx, topic in enumerate(ALLOWED_TOPIC_TAGS)],
    )
    conn.execute("ANALYZE")
    conn.commit()


def _dump_ordering_tsv(conn: sqlite3.Connection) -> int:
    rows = [
        dict(zip(ANKI_ORDERING_COLS, r))
        for r in conn.execute(
            f"SELECT {', '.join(ANKI_ORDERING_COLS)} FROM anki_ordering "
            f"ORDER BY spaced_topic_order")
    ]
    return write_tsv(ORDERING_TSV, rows, fieldnames=ANKI_ORDERING_COLS)


def _print_summary(conn: sqlite3.Connection, n_overrides: int) -> None:
    n = conn.execute("SELECT COUNT(*) FROM anki_ordering").fetchone()[0]
    n_topics = conn.execute("SELECT COUNT(*) FROM topic_order").fetchone()[0]
    diff = conn.execute(
        "SELECT COUNT(*) FROM anki_ordering "
        "WHERE strict_topic_order != spaced_topic_order").fetchone()[0]
    print(f"=== Stage 17.0 — ordering tables built ===")
    print(f"  senses ordered:        {n:,}")
    print(f"  topics:                {n_topics}")
    print(f"  manual overrides:      {n_overrides}")
    print(f"  strict≠spaced rows:    {diff:,}  (moved by polysemy spacing)")
    print(f"  per-custom-topic bucket coverage:")
    for topic in order_rules.CUSTOM_TOPICS:
        total = conn.execute(
            "SELECT COUNT(*) FROM anki_ordering WHERE topic_primary = ?",
            (topic,)).fetchone()[0]
        b999 = conn.execute(
            "SELECT COUNT(*) FROM anki_ordering "
            "WHERE topic_primary = ? AND topic_bucket = 999",
            (topic,)).fetchone()[0]
        curated = total - b999
        pct = 100 * curated / total if total else 0.0
        flag = "  <-- low coverage" if pct < 50 else ""
        print(f"    {topic:<28} {curated:>4}/{total:<4} curated "
              f"({pct:.0f}%), {b999} in bucket-999{flag}")
    print(f"  wrote {ORDERING_TSV.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DB_PATH,
                    help="SQLite path (default data/06-final.sqlite).")
    args = ap.parse_args()

    if not args.db.exists():
        raise BuildError(f"{args.db} does not exist. "
                         f"Run build/16_0_build_sqlite.py first.")

    conn = sqlite3.connect(args.db)
    try:
        _require_base_tables(conn)
        senses = _load_senses(conn)
        overrides = _load_manual_overrides()
        stage_rows = _compute_stage_rows(senses, overrides)
        _build(conn, stage_rows)
        _dump_ordering_tsv(conn)
        _print_summary(conn, len(overrides))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
