#!/usr/bin/env python
"""Generate tests/smoke_sample_1000.tsv deterministically.

Two-tier smoke sample for Stage 4:
  - Tier A: 115 rows preserved verbatim from tests/smoke_sample_100.tsv
  - Tier B-edge-cases: 141 hand-picked rows covering every Stage 4 quality risk
  - Tier B-random: 744 stratified random rows via random.seed(43)

Total: 1000 rows.

The seed-43 random selection is disjoint from seed-42 (used for Tier A) — all
Tier A and hand-picked sense_ids are excluded from the random pool.

Reads:
  tests/smoke_sample_100.tsv         (preserved verbatim)
  data/03-enriched.tsv               (canonical sense source for sense_id lookup)
  data/_idioms_expanded.tsv          (idiom expansion source list)
  data/_flags.tsv                    (NSFW / false-friend / BP-EP flag list)
  data/source.txt                    (for +se reflexive detection)

Writes:
  tests/smoke_sample_1000.tsv

Idempotent: same inputs -> same output. No LLM calls, no API costs.

Run via: uv run python tests/build_smoke_sample_1000.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

# --- paths -------------------------------------------------------------------

TIER_A_PATH = REPO_ROOT / "tests" / "smoke_sample_100.tsv"
ENRICHED_PATH = REPO_ROOT / "data" / "03-enriched.tsv"
IDIOMS_PATH = REPO_ROOT / "data" / "_idioms_expanded.tsv"
FLAGS_PATH = REPO_ROOT / "data" / "_flags.tsv"
SOURCE_PATH = REPO_ROOT / "data" / "source.txt"
OUT_PATH = REPO_ROOT / "tests" / "smoke_sample_1000.tsv"

# --- output schema (matches tests/smoke_sample_100.tsv) ----------------------

OUT_COLUMNS = [
    "sense_id",
    "rank",
    "pt",
    "pos",
    "gender",
    "en_primary",
    "bp_status",
    "split_category",
    "sample_source",
]

# --- target counts (sum = 1000) ----------------------------------------------

TIER_A_COUNT = 115
EDGE_CASE_TARGET = 141
RANDOM_TARGET = 744
TOTAL_TARGET = TIER_A_COUNT + EDGE_CASE_TARGET + RANDOM_TARGET  # 1000

# --- hand-picked allowlists (locked sense_ids and headwords) -----------------

# All 10 remaining idiom expansions (Tier A has 3: 0314.01.01, 2224.01.01, 2224.02.01)
IDIOM_PT_ALLOWLIST = {
    "em diante",
    "por cento",
    "em seguida",
    "em vigor",
    "de repente",
    "ao invés",
    "em contrapartida",
    "não obstante",
    "à mercê de",
    "à tona",
}

# All 12 remaining forced gender splits (Tier A has 4: capital M/F, cabra M/F)
# Need polícia, rádio, corte, cura, grama, banana × M/F = 12 rows
GENDER_SPLIT_PT_ALLOWLIST = {"polícia", "rádio", "corte", "cura", "grama", "banana"}

# Sensitive bucket — keyword screen ON TOP OF bp_status ∈ {nsfw, false_friend} ∪ _flags.tsv.
# These are deterministic keyword screens for terms the future Stage 2.5 would catch.
SENSITIVE_KEYWORDS = {
    "gozar",
    "mulato",
    "índio",
    "aborto",
    "arma",
    "bala",
    "faca",
    "morrer",
    "matar",
    "droga",
    "cigano",
    "raça",
    "homossexual",
    "violar",
    "suicidar",
    "espingarda",
    "revólver",
    "religião",
}
SENSITIVE_TARGET_COUNT = 25

# High-frequency irregular verbs (Tier A has ser, ter, poder)
IRREGULAR_VERBS_PT = ["ir", "fazer", "estar", "dizer", "ver", "vir", "dar", "querer"]

# Function-word headwords with sense_index >= 2 in corpus (only these have polysemy)
FUNCTION_WORD_HEADS = {"o", "que", "se", "de", "em", "para", "com", "a", "por"}

# Cognate target: stratified across frequency tiers (2 each from top500/1000/2000/3000/5000)
# Plus 2 more for total 12. Hand-picked cognates that are A1-recognizable.
COGNATE_HAND_PICKS = [
    "importante",
    "hospital",
    "televisão",
    "informação",
    "chocolate",
    "telefone",
    "atenção",
    "situação",
    "possível",
    "problema",
    "futuro",
    "natural",
]

# Comparatives that aren't in the regular pos=adj/adv flow
COMPARATIVES_PT = ["melhor", "pior"]

# Per-bucket targets (sum = 141)
# Note: `function_word` was 7 before the function-word polysemy patch
# (build/patch_function_words.py) added 5 new secondary senses for se/de/em/a.
# Bumped to 12 to cover all secondary senses; offset by reducing top100_padding.
BUCKET_TARGETS = {
    "idioms": 10,
    "gender_splits": 12,
    "sensitive": 25,
    "reflexive": 10,
    "hyphenated": 15,
    "space_compound": 5,
    "function_word": 12,
    "irregular_verbs": 8,
    "cognates": 12,
    "numerals_interj_comp": 12,
    "top100_padding": 8,
    "bp_rare": 12,
}
assert sum(BUCKET_TARGETS.values()) == EDGE_CASE_TARGET, sum(BUCKET_TARGETS.values())


# --- helpers -----------------------------------------------------------------


def project_row(enriched_row: dict[str, str], sample_source: str) -> dict[str, str]:
    """Project an enriched-corpus row into the smoke-sample column schema."""
    return {
        "sense_id": enriched_row["sense_id"],
        "rank": enriched_row["rank"],
        "pt": enriched_row["pt"],
        "pos": enriched_row.get("pos", ""),
        "gender": enriched_row.get("gender", ""),
        "en_primary": enriched_row["en_primary"],
        "bp_status": enriched_row.get("bp_status", ""),
        "split_category": enriched_row.get("split_category", ""),
        "sample_source": sample_source,
    }


def get_freq_tier(rank: int) -> str:
    if rank <= 500:
        return "top500"
    if rank <= 1000:
        return "top1000"
    if rank <= 2000:
        return "top2000"
    if rank <= 3000:
        return "top3000"
    return "top5000"


def parse_reflexive_source_line_numbers(source_path: Path) -> list[int]:
    """Find source.txt line numbers containing the +se reflexive marker."""
    line_numbers = []
    with source_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if "+se" in line:
                line_numbers.append(i)
    return line_numbers


# --- bucket pickers (each returns list of sense_ids from enriched corpus) ----


def pick_idioms(
    enriched: list[dict[str, str]], excluded: set[str]
) -> list[dict[str, str]]:
    """Pick all idiom-expansion rows whose pt matches IDIOM_PT_ALLOWLIST.

    Idiom rows have expansion_index >= 1 (per sense_id format RRRR.EE.SS).
    """
    picks = []
    seen = set()
    for r in enriched:
        if r["sense_id"] in excluded:
            continue
        if int(r.get("expansion_index", "0")) < 1:
            continue
        if r["pt"] in IDIOM_PT_ALLOWLIST and r["pt"] not in seen:
            picks.append(r)
            seen.add(r["pt"])
    return picks


def pick_gender_splits(
    enriched: list[dict[str, str]], excluded: set[str]
) -> list[dict[str, str]]:
    """Pick rows tagged #gendered-meaning whose pt is in the allowlist."""
    picks = []
    for r in enriched:
        if r["sense_id"] in excluded:
            continue
        if "#gendered-meaning" not in r.get("tags", ""):
            continue
        if r["pt"] in GENDER_SPLIT_PT_ALLOWLIST:
            picks.append(r)
    return picks


def pick_sensitive(
    enriched: list[dict[str, str]],
    flags_pts: set[str],
    excluded: set[str],
    target: int,
) -> list[dict[str, str]]:
    """Pick rows from sensitive sources, deduplicated.

    Priority order:
      1. All bp_status='nsfw' rows
      2. All bp_status='false_friend' rows
      3. _flags.tsv rows (BP/EP markers)
      4. Keyword screen on SENSITIVE_KEYWORDS

    Caps at `target` rows. Stable order — same inputs produce same output.
    """
    picks: list[dict[str, str]] = []
    picked_ids: set[str] = set()

    def add(r: dict[str, str]) -> None:
        if r["sense_id"] in excluded or r["sense_id"] in picked_ids:
            return
        picks.append(r)
        picked_ids.add(r["sense_id"])

    # 1. NSFW
    for r in enriched:
        if len(picks) >= target:
            break
        if r.get("bp_status") == "nsfw":
            add(r)

    # 2. false_friend
    for r in enriched:
        if len(picks) >= target:
            break
        if r.get("bp_status") == "false_friend":
            add(r)

    # 3. _flags.tsv pts
    for r in enriched:
        if len(picks) >= target:
            break
        if r["pt"] in flags_pts:
            add(r)

    # 4. keyword screen
    for r in enriched:
        if len(picks) >= target:
            break
        if r["pt"] in SENSITIVE_KEYWORDS:
            add(r)

    return picks[:target]


def pick_reflexive(
    enriched: list[dict[str, str]],
    reflexive_source_lines: set[int],
    excluded: set[str],
) -> list[dict[str, str]]:
    """Pick rows whose source_line_number indicates a +se reflexive marker.

    For each source_line, take the first sense_index (the canonical sense).
    """
    picks = []
    seen_lines = set()
    for r in enriched:
        if r["sense_id"] in excluded:
            continue
        sln = int(r["source_line_number"])
        if sln in reflexive_source_lines and sln not in seen_lines:
            # Only the first sense per source line, to avoid duplicating semantics
            if int(r.get("sense_index", "1")) == 1:
                picks.append(r)
                seen_lines.add(sln)
    return picks


def pick_hyphenated(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """All pt_type=hyphenated_compound rows minus excluded, capped at target.

    Stratified by rank to ensure spread across frequency tiers.
    """
    candidates = [
        r
        for r in enriched
        if r.get("pt_type") == "hyphenated_compound" and r["sense_id"] not in excluded
    ]
    # Sort by rank ascending; with 18 total and 3 already in Tier A, we have 15 candidates.
    candidates.sort(key=lambda r: int(r["rank"]))
    return candidates[:target]


def pick_space_compound(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """All pt_type=space_compound rows that are NOT idiom expansions.

    Idiom-expansion rows have expansion_index >= 1; the non-idiom space compounds
    are the multi-word headwords like 'no entanto', 'em torno', 'dia a dia',
    'mão de obra'.
    """
    candidates = [
        r
        for r in enriched
        if r.get("pt_type") == "space_compound"
        and int(r.get("expansion_index", "0")) == 0
        and r["sense_id"] not in excluded
    ]
    # Sort by rank for stability
    candidates.sort(key=lambda r: (int(r["rank"]), int(r["sense_index"])))
    return candidates[:target]


def pick_function_word_polysemy(
    enriched: list[dict[str, str]], excluded: set[str]
) -> list[dict[str, str]]:
    """Function-word secondary senses (sense_index >= 2 for FUNCTION_WORD_HEADS)."""
    picks = []
    for r in enriched:
        if r["sense_id"] in excluded:
            continue
        if r["pt"] not in FUNCTION_WORD_HEADS:
            continue
        if int(r.get("sense_index", "1")) < 2:
            continue
        picks.append(r)
    return picks


def pick_irregular_verbs(
    enriched: list[dict[str, str]], excluded: set[str]
) -> list[dict[str, str]]:
    """Top senses (sense_index=1) of irregular verbs in IRREGULAR_VERBS_PT."""
    picks = []
    seen = set()
    for r in enriched:
        if r["sense_id"] in excluded:
            continue
        if r.get("pos") != "verb":
            continue
        if r["pt"] in IRREGULAR_VERBS_PT and r["pt"] not in seen:
            if int(r.get("sense_index", "1")) == 1:
                picks.append(r)
                seen.add(r["pt"])
    return picks


def pick_cognates(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """Cognate rows. First fill via COGNATE_HAND_PICKS allowlist, then top up
    via stratified random across frequency tiers (deterministic).
    """
    # Phase 1: hand-picked cognates by exact pt match
    picks = []
    seen = set()
    for r in enriched:
        if r["sense_id"] in excluded or r["sense_id"] in seen:
            continue
        if r["pt"] in COGNATE_HAND_PICKS and "#cognate-en" in r.get("tags", ""):
            if int(r.get("sense_index", "1")) == 1:
                picks.append(r)
                seen.add(r["sense_id"])
                if len(picks) >= target:
                    break
    return picks[:target]


def pick_numerals_interj_comparatives(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """Numerals (pos=num), interjections (pos=interj), and comparatives.

    Mix: ~5 num, ~3 interj, ~2 comparatives, ~2 padding from num/interj.
    """
    picks: list[dict[str, str]] = []
    picked_ids: set[str] = set()

    def add(r: dict[str, str]) -> None:
        if r["sense_id"] in excluded or r["sense_id"] in picked_ids:
            return
        picks.append(r)
        picked_ids.add(r["sense_id"])

    # Comparatives
    for r in enriched:
        if r["pt"] in COMPARATIVES_PT:
            add(r)
            if sum(1 for p in picks if p["pt"] in COMPARATIVES_PT) >= 2:
                break

    # Interjections (up to 3)
    interj_count = 0
    for r in enriched:
        if r.get("pos") == "interj" and interj_count < 3:
            add(r)
            interj_count += 1

    # Numerals (fill remainder, sorted by rank for stability)
    nums = sorted(
        [r for r in enriched if r.get("pos") == "num"],
        key=lambda r: int(r["rank"]),
    )
    for r in nums:
        if len(picks) >= target:
            break
        add(r)

    return picks[:target]


def pick_top100_padding(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """Senses from rank <= 100 not yet picked. Sorted by rank for stability."""
    candidates = [
        r
        for r in enriched
        if int(r["rank"]) <= 100 and r["sense_id"] not in excluded
    ]
    candidates.sort(key=lambda r: (int(r["rank"]), int(r["sense_index"])))
    return candidates[:target]


def pick_bp_rare(
    enriched: list[dict[str, str]], excluded: set[str], target: int
) -> list[dict[str, str]]:
    """Rows tagged #bp-rare. Sorted by rank."""
    candidates = [
        r
        for r in enriched
        if "#bp-rare" in r.get("tags", "") and r["sense_id"] not in excluded
    ]
    candidates.sort(key=lambda r: (int(r["rank"]), int(r["sense_index"])))
    return candidates[:target]


# --- stratified random -------------------------------------------------------


def stratified_random_sample(
    available: list[dict[str, str]], target: int, seed: int
) -> list[dict[str, str]]:
    """Stratified random sample by PoS, proportional to corpus.

    Stratification: noun, verb, adj, adv, other (everything else).
    Targets are computed proportionally to the available pool.
    """
    rng = random.Random(seed)

    by_pos: dict[str, list[dict[str, str]]] = {
        "noun": [],
        "verb": [],
        "adj": [],
        "adv": [],
        "other": [],
    }
    for r in available:
        pos = r.get("pos", "")
        if pos in ("noun", "verb", "adj", "adv"):
            by_pos[pos].append(r)
        else:
            by_pos["other"].append(r)

    total_available = sum(len(v) for v in by_pos.values())
    # Proportional targets (round, then fix to sum exactly to target)
    raw = {
        pos: target * len(rows) / total_available for pos, rows in by_pos.items()
    }
    int_targets = {pos: int(v) for pos, v in raw.items()}
    deficit = target - sum(int_targets.values())
    # Distribute remainder by largest fractional part
    fractions = sorted(
        ((pos, raw[pos] - int_targets[pos]) for pos in raw),
        key=lambda x: -x[1],
    )
    for i in range(deficit):
        pos = fractions[i % len(fractions)][0]
        int_targets[pos] += 1

    assert sum(int_targets.values()) == target, int_targets

    # Sort each stratum by sense_id for deterministic shuffling
    picks: list[dict[str, str]] = []
    for pos, rows in by_pos.items():
        rows_sorted = sorted(rows, key=lambda r: r["sense_id"])
        rng.shuffle(rows_sorted)
        picks.extend(rows_sorted[: int_targets[pos]])

    return picks


# --- main --------------------------------------------------------------------


def main() -> int:
    print(f"Reading Tier A from {TIER_A_PATH.relative_to(REPO_ROOT)}...")
    tier_a = read_tsv(TIER_A_PATH)
    if len(tier_a) != TIER_A_COUNT:
        print(f"  WARNING: Tier A has {len(tier_a)} rows, expected {TIER_A_COUNT}")
    tier_a_ids = {r["sense_id"] for r in tier_a}

    print(f"Reading enriched corpus from {ENRICHED_PATH.relative_to(REPO_ROOT)}...")
    enriched = read_tsv(ENRICHED_PATH)
    enriched_by_id = {r["sense_id"]: r for r in enriched}
    print(f"  loaded {len(enriched)} enriched rows")

    # Verify all Tier A sense_ids resolve in enriched
    missing = tier_a_ids - set(enriched_by_id.keys())
    if missing:
        print(f"  ERROR: {len(missing)} Tier A sense_ids missing from enriched: {sorted(missing)[:5]}")
        return 1

    print(f"Reading flags from {FLAGS_PATH.relative_to(REPO_ROOT)}...")
    flags = read_tsv(FLAGS_PATH)
    flags_pts = {r["pt"] for r in flags}

    print(f"Detecting +se reflexive lines from {SOURCE_PATH.relative_to(REPO_ROOT)}...")
    reflexive_source_lines = set(parse_reflexive_source_line_numbers(SOURCE_PATH))
    print(f"  found {len(reflexive_source_lines)} +se source lines")

    # --- bucket assembly -----------------------------------------------------

    excluded = set(tier_a_ids)
    edge_case_rows: list[dict[str, str]] = []
    bucket_counts: dict[str, int] = {}

    def add_bucket(name: str, rows: list[dict[str, str]]) -> None:
        target = BUCKET_TARGETS[name]
        if len(rows) > target:
            rows = rows[:target]
        edge_case_rows.extend(rows)
        excluded.update(r["sense_id"] for r in rows)
        bucket_counts[name] = len(rows)
        print(f"  {name:30s} target={target:3d}  picked={len(rows):3d}")

    print("\nBuilding hand-picked edge cases...")

    add_bucket("idioms", pick_idioms(enriched, excluded))
    add_bucket("gender_splits", pick_gender_splits(enriched, excluded))
    add_bucket(
        "sensitive",
        pick_sensitive(enriched, flags_pts, excluded, BUCKET_TARGETS["sensitive"]),
    )
    add_bucket("reflexive", pick_reflexive(enriched, reflexive_source_lines, excluded))
    add_bucket(
        "hyphenated", pick_hyphenated(enriched, excluded, BUCKET_TARGETS["hyphenated"])
    )
    add_bucket(
        "space_compound",
        pick_space_compound(enriched, excluded, BUCKET_TARGETS["space_compound"]),
    )
    add_bucket("function_word", pick_function_word_polysemy(enriched, excluded))
    add_bucket("irregular_verbs", pick_irregular_verbs(enriched, excluded))
    add_bucket(
        "cognates", pick_cognates(enriched, excluded, BUCKET_TARGETS["cognates"])
    )
    add_bucket(
        "numerals_interj_comp",
        pick_numerals_interj_comparatives(
            enriched, excluded, BUCKET_TARGETS["numerals_interj_comp"]
        ),
    )
    add_bucket(
        "top100_padding",
        pick_top100_padding(enriched, excluded, BUCKET_TARGETS["top100_padding"]),
    )
    add_bucket("bp_rare", pick_bp_rare(enriched, excluded, BUCKET_TARGETS["bp_rare"]))

    print(f"\nEdge cases total: {len(edge_case_rows)} (target {EDGE_CASE_TARGET})")

    # If a bucket fell short, redistribute the deficit to "padding" — pick more
    # top-frequency rows (rank <= 200) not yet picked.
    deficit = EDGE_CASE_TARGET - len(edge_case_rows)
    if deficit > 0:
        print(f"  Edge-case deficit {deficit}; padding from top-200 frequency rows")
        candidates = sorted(
            [
                r
                for r in enriched
                if int(r["rank"]) <= 200 and r["sense_id"] not in excluded
            ],
            key=lambda r: (int(r["rank"]), int(r["sense_index"])),
        )
        pad = candidates[:deficit]
        edge_case_rows.extend(pad)
        excluded.update(r["sense_id"] for r in pad)
        print(f"  padded {len(pad)} rows from top-200")

    if len(edge_case_rows) != EDGE_CASE_TARGET:
        print(
            f"  ERROR: edge-case count {len(edge_case_rows)} != target {EDGE_CASE_TARGET}"
        )
        return 1

    # --- stratified random ---------------------------------------------------

    print("\nBuilding stratified random sample (seed=43)...")
    available = [r for r in enriched if r["sense_id"] not in excluded]
    print(f"  available pool: {len(available)} (universe {len(enriched)} − excluded {len(excluded)})")

    random_rows = stratified_random_sample(available, RANDOM_TARGET, seed=43)
    print(f"  sampled: {len(random_rows)}")
    if len(random_rows) != RANDOM_TARGET:
        print(f"  ERROR: random count {len(random_rows)} != target {RANDOM_TARGET}")
        return 1

    # --- assemble output -----------------------------------------------------

    out_rows: list[dict[str, str]] = []

    # Tier A: preserve verbatim (already in smoke-sample column schema)
    for r in tier_a:
        out_rows.append(r)

    # Edge cases: project from enriched schema
    for r in edge_case_rows:
        out_rows.append(project_row(r, sample_source="edge_case_v2"))

    # Random: project
    for r in random_rows:
        out_rows.append(project_row(r, sample_source="random_v2"))

    # Sanity: total count
    if len(out_rows) != TOTAL_TARGET:
        print(f"\nERROR: total {len(out_rows)} != target {TOTAL_TARGET}")
        return 1

    # Sanity: no duplicate sense_ids
    seen = set()
    dupes = []
    for r in out_rows:
        sid = r["sense_id"]
        if sid in seen:
            dupes.append(sid)
        seen.add(sid)
    if dupes:
        print(f"\nERROR: {len(dupes)} duplicate sense_ids: {dupes[:5]}")
        return 1

    # --- write ---------------------------------------------------------------

    written = write_tsv(OUT_PATH, out_rows, fieldnames=OUT_COLUMNS)
    print(f"\nWrote {written} rows to {OUT_PATH.relative_to(REPO_ROOT)}")

    # --- summary -------------------------------------------------------------

    print("\n=== Sample composition ===")
    print(f"  Tier A (preserved):        {len(tier_a)}")
    print(f"  Edge cases (v2):           {len(edge_case_rows)}")
    print(f"  Stratified random (v2):    {len(random_rows)}")
    print(f"  TOTAL:                     {len(out_rows)}")
    print("\n=== Edge-case bucket composition ===")
    for name, count in bucket_counts.items():
        print(f"  {name:30s} {count:3d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
