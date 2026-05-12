"""Verify `data/06-final.tsv` against the invariants in docs/plan.md.

Exits with non-zero on any HARD invariant failure. SOFT warnings print
but don't fail.

Invariants (HARD):
  - row_count == 5,725
  - column count == 37 and header matches FINAL_FIELDS exactly
  - sense_id is unique
  - sense_id matches the RRRR.EE.SS pattern
  - audio_word + audio_example both non-empty
  - audio URLs contain `eleven_flash_v2_5` (Stage 9 lock)
  - voice_id is non-empty and resolves in config/voices.tsv
  - voice_gender ∈ {m, f} and matches the voice_id's gender in config
  - audio_word_md5 + audio_example_md5 populated (≥10 hex chars each)
  - Stage 12: usage_hint_priority ∈ {essential, useful, ""}; risk-note-only
    rows ship "" (gozar rule); empty usage_hint ⇒ empty priority
  - Stage 12: no embedded newline / tab in usage_hint or risk_note
  - Stage 12: usage_hint length ≤ 120 chars
  - Stage 12: every shipped usage_hint or risk_note has a sidecar row
  - Stage 13: every non-empty family_root exists as a `pt` in the deck
  - Stage 13: no family_root ends with `-` or contains a digit (no bare stems)
  - Stage 13: no low-confidence sidecar row ships (except postprocess_self_root)
  - Stage 13: every shipped family_root has a sidecar entry
  - Stage 13: cluster size ≥ 2 distinct `pt` lemmas per shipped family_root
  - Stage 13: family_root empty on every row where pos ∉ {noun, verb, adj, adv}
  - random sample of N audio URLs returns HTTP 200 (controlled by --http-sample)

Soft warnings (DON'T fail):
  - source_line blank
  - example_pt blank
  - usage_hint longer than 80 chars (target ceiling, hard fail at 120)
  - Stage 13: content-word coverage < 20% or > 70%
  - Stage 13: content-word coverage < 30% (target floor)
  - Stage 13: same `pt` maps to different family_root values across senses
  - Stage 13: family_root contains whitespace (rare; legitimate only if BP MWE)
  - Stage 13: chosen root row is bp_status ∈ {false_friend, uncommon, nsfw}
  - Stage 13: cluster size > 20 (over-grouping)
  - Stage 13: family_root value whose pt row is itself a function word

Usage:
    .venv/bin/python build/verify_all.py [--http-sample 50]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv  # noqa: E402
from build.lib.topic_tag_rules import (  # noqa: E402
    ALLOWED_TOPIC_TAG_SET,
    MANUAL_TOPIC_OVERRIDES,
    NUM_TOPIC_OVERRIDES,
)
from build.lib.risk_register_rules import (  # noqa: E402
    ALLOWED_BP_VALIDITY_SET,
    ALLOWED_REGISTER_SET,
    ALLOWED_RISK_FLAGS_SET,
    RISK_NOTE_MAX_CHARS,
)

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

FINAL_PATH = DATA_DIR / "06-final.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"
USAGE_HINTS_PATH = DATA_DIR / "_usage_hints.tsv"
FAMILY_ROOTS_PATH = DATA_DIR / "_family_roots.tsv"
TOPIC_TAGS_PATH = DATA_DIR / "_topic_tags.tsv"
RISK_REGISTER_PATH = DATA_DIR / "_risk_register.tsv"
MANUAL_RISK_REGISTER_PATH = DATA_DIR / "_manual_risk_register.tsv"

EXPECTED_ROW_COUNT = 5725
SENSE_ID_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
MD5_MIN_LEN = 10
FLASH_MODEL_MARKER = "eleven_flash_v2_5"

# Stage 12 column expectations
VALID_USAGE_HINT_PRIORITIES = {"essential", "useful", ""}
USAGE_HINT_SOFT_MAX = 80
USAGE_HINT_HARD_MAX = 120

# Stage 13 column expectations
FAMILY_ROOT_CONTENT_POS = {"noun", "verb", "adj", "adv"}
FAMILY_ROOT_DETERMINISTIC_SOURCES = {"postprocess_self_root", "manual"}
FAMILY_ROOT_NON_STANDARD_BP_STATUS = {"false_friend", "uncommon", "nsfw"}
FAMILY_ROOT_COVERAGE_FLOOR = 0.30
FAMILY_ROOT_COVERAGE_LOW = 0.20
FAMILY_ROOT_COVERAGE_HIGH = 0.70
FAMILY_ROOT_CLUSTER_MAX_SOFT = 20
FAMILY_ROOT_BARE_STEM_RE = re.compile(r"-$|\d")

# Stage 14 column expectations
TOPIC_TAG_DETERMINISTIC_POS_GRAMMAR = {"prep", "conj", "pron", "art"}
TOPIC_TAG_DETERMINISTIC_POS_NUMBERS = {"num"}
TOPIC_COVERAGE_ANY_CAP = 0.25
TOPIC_COVERAGE_DAILY_ROUTINES_CAP = 0.15
TOPIC_COVERAGE_OBJECTS_TOOLS_CAP = 0.15
TOPIC_COVERAGE_WORK_JOBS_CAP = 0.15
TOPIC_COVERAGE_CHARACTER_QUALITIES_CAP = 0.15
TOPIC_COVERAGE_MEASUREMENT_CAP = 0.12
TOPIC_LOW_CONF_CAP = 0.05

# Full expected header (40 cols, Stage 15). Used by _verify_header().
EXPECTED_HEADER = [
    "sense_id", "rank", "source_pt", "pt", "pt_type", "gender",
    "pt_display", "pos", "sense_index", "en_primary", "en_all",
    "annotation", "bp_status", "normalization_action", "ipa_word",
    "example_pt", "example_en", "target_word_used", "ipa_example",
    "audio_word", "audio_example", "audio_word_md5", "audio_example_md5",
    "audio_en_example", "audio_en_example_md5",
    "voice_id", "voice_id_en", "voice_gender", "family_root", "tags",
    "source_line", "source_line_number", "notes",
    "pt_display_safe",
    "usage_hint", "usage_hint_priority", "risk_note",
    "bp_validity", "register", "risk_flags",
]

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")


class Verifier:
    def __init__(self) -> None:
        self.hard_failures: list[str] = []
        self.soft_warnings: list[str] = []

    def hard_fail(self, msg: str) -> None:
        self.hard_failures.append(msg)
        print(f"  ✗ HARD: {msg}", file=sys.stderr)

    def soft_warn(self, msg: str) -> None:
        self.soft_warnings.append(msg)
        print(f"  ⚠ SOFT: {msg}", file=sys.stderr)

    def passed(self) -> bool:
        return not self.hard_failures


def _verify_rows(rows: list[dict], voice_gender: dict[str, str], v: Verifier) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        v.hard_fail(f"row count {len(rows)} != expected {EXPECTED_ROW_COUNT}")
    else:
        print(f"  ✓ row_count == {EXPECTED_ROW_COUNT}")

    sids = [r["sense_id"] for r in rows]
    if len(set(sids)) != len(sids):
        dups = [s for s in sids if sids.count(s) > 1]
        v.hard_fail(f"duplicate sense_ids: {dups[:5]}")
    else:
        print(f"  ✓ sense_id uniqueness")

    bad_pattern = [s for s in sids if not SENSE_ID_PATTERN.match(s)]
    if bad_pattern:
        v.hard_fail(f"{len(bad_pattern)} sense_ids fail RRRR.EE.SS pattern; e.g., {bad_pattern[:3]}")
    else:
        print(f"  ✓ sense_id pattern (RRRR.EE.SS)")

    missing_audio = [r["sense_id"] for r in rows
                     if not r["audio_word"] or not r["audio_example"]]
    if missing_audio:
        v.hard_fail(f"{len(missing_audio)} rows missing audio_word or audio_example; e.g., {missing_audio[:3]}")
    else:
        print(f"  ✓ audio_word + audio_example populated on all rows")

    non_flash = [r["sense_id"] for r in rows
                 if FLASH_MODEL_MARKER not in r["audio_word"]
                 or FLASH_MODEL_MARKER not in r["audio_example"]]
    if non_flash:
        v.hard_fail(f"{len(non_flash)} rows have non-Flash audio URL; e.g., {non_flash[:3]}")
    else:
        print(f"  ✓ all audio URLs contain '{FLASH_MODEL_MARKER}'")

    missing_md5 = [r["sense_id"] for r in rows
                   if len(r.get("audio_word_md5", "")) < MD5_MIN_LEN
                   or len(r.get("audio_example_md5", "")) < MD5_MIN_LEN]
    if missing_md5:
        v.hard_fail(f"{len(missing_md5)} rows have short/missing md5; e.g., {missing_md5[:3]}")
    else:
        print(f"  ✓ md5 fields populated on all rows")

    bad_voice = []
    bad_gender = []
    for r in rows:
        vid = r.get("voice_id", "")
        if not vid or vid not in voice_gender:
            bad_voice.append(r["sense_id"])
            continue
        expected_gender = "m" if voice_gender[vid] == "male" else "f"
        if r.get("voice_gender", "") != expected_gender:
            bad_gender.append((r["sense_id"], r.get("voice_gender", ""), expected_gender))
    if bad_voice:
        v.hard_fail(f"{len(bad_voice)} rows have bad/unknown voice_id; e.g., {bad_voice[:3]}")
    else:
        print(f"  ✓ voice_id resolves in config/voices.tsv (all {len(rows)})")
    if bad_gender:
        v.hard_fail(f"{len(bad_gender)} rows have voice_gender mismatch; e.g., {bad_gender[:3]}")
    else:
        print(f"  ✓ voice_gender matches config gender")

    # Soft warnings — note: family_root coverage is reported in
    # _verify_family_roots(), which understands the content-word denominator.
    n_no_srcline = sum(1 for r in rows if not r.get("source_line", ""))
    if n_no_srcline:
        v.soft_warn(f"source_line blank on {n_no_srcline}/{len(rows)} rows")
    n_no_example = sum(1 for r in rows if not r.get("example_pt", ""))
    if n_no_example:
        v.soft_warn(f"example_pt blank on {n_no_example}/{len(rows)} rows")


def _verify_header(v: Verifier) -> None:
    """Read the 06-final.tsv header line and confirm column order + count."""
    with FINAL_PATH.open(encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
    header = header_line.split("\t")
    if len(header) != len(EXPECTED_HEADER):
        v.hard_fail(f"column count {len(header)} != expected {len(EXPECTED_HEADER)}")
        return
    if header != EXPECTED_HEADER:
        diff = [(i, h, e) for i, (h, e) in enumerate(zip(header, EXPECTED_HEADER), 1) if h != e]
        v.hard_fail(f"header mismatch in {len(diff)} positions; first: "
                    f"col {diff[0][0]} got {diff[0][1]!r} expected {diff[0][2]!r}")
        return
    print(f"  ✓ header has {len(EXPECTED_HEADER)} cols, exact match")


def _verify_usage_hints(rows: list[dict], v: Verifier) -> None:
    """Stage 12: priority enum, length, no-newline, sidecar coverage."""
    bad_priority: list[tuple[str, str]] = []
    embedded_ws: list[tuple[str, str]] = []
    too_long: list[tuple[str, int]] = []
    soft_long: list[tuple[str, int]] = []
    risk_only_with_priority: list[str] = []
    priority_without_hint: list[str] = []
    hint_without_sidecar: list[str] = []

    sidecar_sids: set[str] = set()
    if USAGE_HINTS_PATH.exists():
        sidecar_sids = {r["sense_id"] for r in read_tsv(USAGE_HINTS_PATH)}

    # Stage 15 added a second + third risk_note source: data/_risk_register.tsv
    # (LLM / deterministic) and data/_manual_risk_register.tsv (manual).
    # Provenance for risk_note now flows from any of these three files.
    risk_register_sids: set[str] = set()
    if RISK_REGISTER_PATH.exists():
        risk_register_sids = {r["sense_id"] for r in read_tsv(RISK_REGISTER_PATH)}
    manual_risk_sids: set[str] = set()
    if MANUAL_RISK_REGISTER_PATH.exists():
        manual_risk_sids = {r["sense_id"] for r in read_tsv(MANUAL_RISK_REGISTER_PATH)}
    risk_note_provenance_sids = sidecar_sids | risk_register_sids | manual_risk_sids

    for r in rows:
        sid = r["sense_id"]
        hint = r.get("usage_hint", "")
        priority = r.get("usage_hint_priority", "")
        risk = r.get("risk_note", "")

        if priority not in VALID_USAGE_HINT_PRIORITIES:
            bad_priority.append((sid, priority))

        for field_name, value in (("usage_hint", hint), ("risk_note", risk)):
            if any(ch in value for ch in ("\n", "\r", "\t")):
                embedded_ws.append((sid, field_name))

        if len(hint) > USAGE_HINT_HARD_MAX:
            too_long.append((sid, len(hint)))
        elif len(hint) > USAGE_HINT_SOFT_MAX:
            soft_long.append((sid, len(hint)))

        # Gozar rule: risk-note-only rows must have empty priority.
        if risk and not hint and priority:
            risk_only_with_priority.append(sid)

        # Consistency: empty hint ⇒ empty priority (unless the row has neither).
        if priority and not hint:
            priority_without_hint.append(sid)

        # Provenance:
        #   - shipped usage_hint must have an entry in _usage_hints.tsv (Stage 12)
        #   - shipped risk_note may come from Stage 12 OR Stage 15 sources
        if hint and sid not in sidecar_sids:
            hint_without_sidecar.append(sid)
        elif risk and sid not in risk_note_provenance_sids:
            hint_without_sidecar.append(sid)

    n_hints = sum(1 for r in rows if r.get("usage_hint", ""))
    n_risk = sum(1 for r in rows if r.get("risk_note", ""))
    n_priority = sum(1 for r in rows if r.get("usage_hint_priority", ""))
    print(f"  Stage 12: {n_hints} usage_hints, {n_risk} risk_notes, "
          f"{n_priority} priority chips")

    if bad_priority:
        v.hard_fail(f"{len(bad_priority)} rows have invalid usage_hint_priority "
                    f"(not in {sorted(VALID_USAGE_HINT_PRIORITIES)}); "
                    f"e.g., {bad_priority[:3]}")
    else:
        print(f"  ✓ usage_hint_priority ∈ {{essential, useful, ''}} on all rows")

    if embedded_ws:
        v.hard_fail(f"{len(embedded_ws)} usage_hint/risk_note values contain "
                    f"newline/tab; e.g., {embedded_ws[:3]}")
    else:
        print(f"  ✓ no embedded newline/tab in usage_hint or risk_note")

    if too_long:
        v.hard_fail(f"{len(too_long)} usage_hint values exceed {USAGE_HINT_HARD_MAX} "
                    f"chars; e.g., {too_long[:3]}")
    else:
        print(f"  ✓ usage_hint ≤ {USAGE_HINT_HARD_MAX} chars on all rows")

    if risk_only_with_priority:
        v.hard_fail(f"{len(risk_only_with_priority)} risk-note-only rows still "
                    f"carry a usage_hint_priority; e.g., {risk_only_with_priority[:3]}")
    else:
        print(f"  ✓ risk-note-only rows ship without usage_hint_priority")

    if priority_without_hint:
        v.hard_fail(f"{len(priority_without_hint)} rows have priority but empty "
                    f"usage_hint; e.g., {priority_without_hint[:3]}")
    else:
        print(f"  ✓ usage_hint and usage_hint_priority are co-empty")

    if hint_without_sidecar:
        v.hard_fail(f"{len(hint_without_sidecar)} rows ship a hint/risk_note "
                    f"with no matching sidecar entry; e.g., {hint_without_sidecar[:3]}")
    else:
        print(f"  ✓ every shipped hint/risk_note has a sidecar row")

    if soft_long:
        v.soft_warn(f"{len(soft_long)} usage_hint values exceed soft target "
                    f"{USAGE_HINT_SOFT_MAX} chars (still ≤ {USAGE_HINT_HARD_MAX})")


def _verify_family_roots(rows: list[dict], v: Verifier) -> None:
    """Stage 13: validate family_root assignments against sidecar + invariants."""
    all_pts: set[str] = {r["pt"] for r in rows if r.get("pt")}
    pt_to_pos: dict[str, str] = {}
    pt_to_bp_status: dict[str, str] = {}
    for r in rows:
        pt = r.get("pt") or ""
        if pt and pt not in pt_to_pos:
            pt_to_pos[pt] = r.get("pos", "")
            pt_to_bp_status[pt] = r.get("bp_status", "")

    sidecar_index: dict[str, dict] = {}
    if FAMILY_ROOTS_PATH.exists():
        for sc in read_tsv(FAMILY_ROOTS_PATH):
            sidecar_index[sc["sense_id"]] = sc

    # Collect issue sets.
    not_in_pts: list[tuple[str, str]] = []
    bare_stem: list[tuple[str, str]] = []
    contains_whitespace: list[tuple[str, str]] = []
    low_conf_shipped: list[tuple[str, str]] = []
    missing_sidecar: list[str] = []
    function_word_leak: list[tuple[str, str, str]] = []
    function_word_root: list[tuple[str, str]] = []
    nonstandard_root: list[tuple[str, str, str]] = []

    # Per-pt → roots seen, per-root → distinct pts seen.
    pt_to_roots: dict[str, set[str]] = {}
    root_to_pts: dict[str, set[str]] = {}

    n_content = 0
    n_content_filled = 0
    for r in rows:
        sid = r["sense_id"]
        pos = r.get("pos", "")
        pt = r.get("pt", "")
        root = (r.get("family_root") or "").strip()

        if pos in FAMILY_ROOT_CONTENT_POS:
            n_content += 1
            if root:
                n_content_filled += 1

        if not root:
            continue

        # H6: must be empty for non-content-word rows.
        if pos not in FAMILY_ROOT_CONTENT_POS:
            function_word_leak.append((sid, pos, root))

        # H1: must exist as a pt in the deck.
        if root not in all_pts:
            not_in_pts.append((sid, root))

        # H2: no bare stem / digit.
        if FAMILY_ROOT_BARE_STEM_RE.search(root):
            bare_stem.append((sid, root))

        # S4: whitespace.
        if any(ch.isspace() for ch in root):
            contains_whitespace.append((sid, root))

        # S7: chosen root's pt row is a function word.
        root_pos = pt_to_pos.get(root, "")
        if root_pos and root_pos not in FAMILY_ROOT_CONTENT_POS:
            function_word_root.append((sid, root))

        # S5: chosen root row is non-standard while standard candidates exist.
        if pt_to_bp_status.get(root, "") in FAMILY_ROOT_NON_STANDARD_BP_STATUS:
            nonstandard_root.append((sid, root, pt_to_bp_status.get(root, "")))

        # H3, H4: cross-check sidecar provenance.
        sc = sidecar_index.get(sid)
        if sc is None:
            missing_sidecar.append(sid)
        else:
            conf = (sc.get("confidence") or "").strip().lower()
            source = (sc.get("source") or "llm").strip()
            if (conf != "high"
                    and source not in FAMILY_ROOT_DETERMINISTIC_SOURCES):
                low_conf_shipped.append((sid, conf))

        # Track cluster membership: count distinct pts per root.
        root_to_pts.setdefault(root, set()).add(pt)
        if pt:
            pt_to_roots.setdefault(pt, set()).add(root)

    # H5: cluster size ≥ 2 distinct pts (root's own pt counts if it's in deck).
    cluster_violations: list[tuple[str, int]] = []
    oversized_clusters: list[tuple[str, int]] = []
    for root, pts in root_to_pts.items():
        members = set(pts)
        if root in all_pts:
            members.add(root)
        if len(members) < 2:
            cluster_violations.append((root, len(members)))
        if len(members) > FAMILY_ROOT_CLUSTER_MAX_SOFT:
            oversized_clusters.append((root, len(members)))

    polyroot_pts = {
        pt: sorted(roots) for pt, roots in pt_to_roots.items() if len(roots) > 1
    }

    coverage = n_content_filled / n_content if n_content else 0.0
    print(f"  Stage 13: {n_content_filled}/{n_content} content-word rows have "
          f"family_root ({coverage*100:.1f}% coverage); "
          f"{len(root_to_pts)} distinct roots")

    # --- Hard checks ---
    if not_in_pts:
        v.hard_fail(f"{len(not_in_pts)} family_root values not present as pt; "
                    f"e.g., {not_in_pts[:3]}")
    else:
        print(f"  ✓ every family_root is a pt in the deck")

    if bare_stem:
        v.hard_fail(f"{len(bare_stem)} family_root values end with `-` or "
                    f"contain a digit; e.g., {bare_stem[:3]}")
    else:
        print(f"  ✓ no bare stems / digit-containing roots")

    if low_conf_shipped:
        v.hard_fail(f"{len(low_conf_shipped)} sidecar rows shipped a root with "
                    f"non-high confidence and non-deterministic source; "
                    f"e.g., {low_conf_shipped[:3]}")
    else:
        print(f"  ✓ only high-confidence (or deterministic) roots shipped")

    if missing_sidecar:
        v.hard_fail(f"{len(missing_sidecar)} rows ship a family_root with no "
                    f"matching sidecar entry; e.g., {missing_sidecar[:3]}")
    else:
        print(f"  ✓ every shipped family_root has a sidecar row")

    if cluster_violations:
        v.hard_fail(f"{len(cluster_violations)} family_root values have cluster "
                    f"size < 2 distinct pts; e.g., {cluster_violations[:3]}")
    else:
        print(f"  ✓ every cluster has ≥ 2 distinct pt lemmas")

    if function_word_leak:
        v.hard_fail(f"{len(function_word_leak)} rows ship a family_root on a "
                    f"non-content-word pos; e.g., {function_word_leak[:3]}")
    else:
        print(f"  ✓ family_root empty on every function-word row")

    # --- Soft warnings ---
    if coverage < FAMILY_ROOT_COVERAGE_LOW or coverage > FAMILY_ROOT_COVERAGE_HIGH:
        v.soft_warn(f"content-word coverage {coverage*100:.1f}% is outside "
                    f"healthy range "
                    f"[{FAMILY_ROOT_COVERAGE_LOW*100:.0f}%, "
                    f"{FAMILY_ROOT_COVERAGE_HIGH*100:.0f}%]")
    elif coverage < FAMILY_ROOT_COVERAGE_FLOOR:
        v.soft_warn(f"content-word coverage {coverage*100:.1f}% is below the "
                    f"target floor "
                    f"{FAMILY_ROOT_COVERAGE_FLOOR*100:.0f}%")

    if polyroot_pts:
        sample = list(polyroot_pts.items())[:3]
        v.soft_warn(f"{len(polyroot_pts)} pts map to different family_roots "
                    f"across senses (legitimate for sense splits); e.g., {sample}")

    if contains_whitespace:
        v.soft_warn(f"{len(contains_whitespace)} family_root values contain "
                    f"whitespace; e.g., {contains_whitespace[:3]}")

    if nonstandard_root:
        v.soft_warn(f"{len(nonstandard_root)} rows chose a non-standard "
                    f"(false_friend/uncommon/nsfw) root; e.g., {nonstandard_root[:3]}")

    if oversized_clusters:
        oversized_clusters.sort(key=lambda x: -x[1])
        v.soft_warn(f"{len(oversized_clusters)} clusters exceed soft cap "
                    f"{FAMILY_ROOT_CLUSTER_MAX_SOFT}; e.g., {oversized_clusters[:3]}")

    if function_word_root:
        v.soft_warn(f"{len(function_word_root)} rows point to a function-word "
                    f"pt as root; e.g., {function_word_root[:3]}")


def _verify_topic_tags(rows: list[dict], v: Verifier) -> None:
    """Stage 14: every row carries exactly one #topic-* tag from the 50-item
    allowlist; deterministic invariants hold; no legacy #cat-* leakage.
    """
    sidecar_index: dict[str, dict] = {}
    if TOPIC_TAGS_PATH.exists():
        sidecar_index = {r["sense_id"]: r for r in read_tsv(TOPIC_TAGS_PATH)}

    no_topic: list[tuple[str, str]] = []
    multi_topic: list[tuple[str, list[str]]] = []
    invalid_topic: list[tuple[str, str]] = []
    duplicate_topic: list[tuple[str, str]] = []
    missing_sidecar: list[str] = []
    grammar_violation: list[tuple[str, str, str]] = []
    numbers_violation: list[tuple[str, str, str]] = []
    legacy_cat: list[tuple[str, str]] = []

    topic_counts: dict[str, int] = {}
    by_conf: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "": 0}
    by_source: dict[str, int] = {"deterministic": 0, "llm": 0, "manual": 0, "": 0}

    for r in rows:
        sid = r["sense_id"]
        pos = r.get("pos", "")
        tags = (r.get("tags") or "").split()

        topic_hits = [t for t in tags if t.startswith("#topic-")]
        cat_hits = [t for t in tags if t.startswith("#cat-")]

        # H7: legacy #cat-* regression
        if cat_hits:
            legacy_cat.append((sid, " ".join(cat_hits)))

        # H1: exactly one #topic-*
        if not topic_hits:
            no_topic.append((sid, pos))
            continue
        if len(topic_hits) > 1:
            # H3: duplicates first (could be same tag repeated)
            if len(set(topic_hits)) == 1:
                duplicate_topic.append((sid, topic_hits[0]))
            else:
                multi_topic.append((sid, topic_hits))
            # Use the first for further checks
            t = topic_hits[0]
        else:
            t = topic_hits[0]

        # H2: allowlist enum
        if t not in ALLOWED_TOPIC_TAG_SET:
            invalid_topic.append((sid, t))

        topic_counts[t] = topic_counts.get(t, 0) + 1

        # H5: deterministic grammar invariant — except per-sense MANUAL
        # overrides (e.g. a future override could re-route a function-word
        # row away from #topic-grammar, but only via explicit manual entry).
        if pos in TOPIC_TAG_DETERMINISTIC_POS_GRAMMAR:
            expected = MANUAL_TOPIC_OVERRIDES.get(sid, "#topic-grammar")
            if t != expected:
                grammar_violation.append((sid, pos, t))

        # H6: deterministic numbers invariant — except per-sense overrides.
        # NUM_TOPIC_OVERRIDES handles pos=num rows whose sense isn't numeric
        # (segundo/quarto/cento). MANUAL_TOPIC_OVERRIDES wins if both apply.
        if pos in TOPIC_TAG_DETERMINISTIC_POS_NUMBERS:
            if sid in MANUAL_TOPIC_OVERRIDES:
                expected = MANUAL_TOPIC_OVERRIDES[sid]
            elif sid in NUM_TOPIC_OVERRIDES:
                expected = NUM_TOPIC_OVERRIDES[sid]
            else:
                expected = "#topic-numbers"
            if t != expected:
                numbers_violation.append((sid, pos, t))

        # H4: sidecar provenance
        sc = sidecar_index.get(sid)
        if sc is None:
            missing_sidecar.append(sid)
        else:
            by_conf[(sc.get("confidence") or "").strip().lower() or ""] = (
                by_conf.get((sc.get("confidence") or "").strip().lower() or "", 0) + 1
            )
            by_source[(sc.get("source") or "").strip() or ""] = (
                by_source.get((sc.get("source") or "").strip() or "", 0) + 1
            )

    n_rows = len(rows)
    n_topics = sum(topic_counts.values())
    n_distinct = len(topic_counts)
    print(f"  Stage 14: {n_topics}/{n_rows} rows tagged across "
          f"{n_distinct}/{len(ALLOWED_TOPIC_TAG_SET)} topics; "
          f"sources: {by_source}; conf: {by_conf}")

    # --- Hard checks ---
    if no_topic:
        v.hard_fail(f"{len(no_topic)} rows have NO #topic-* tag; e.g., {no_topic[:3]}")
    else:
        print(f"  ✓ every row has at least one #topic-* tag")

    if multi_topic:
        v.hard_fail(f"{len(multi_topic)} rows have multiple distinct #topic-* "
                    f"tags; e.g., {multi_topic[:3]}")
    else:
        print(f"  ✓ no row has multiple distinct #topic-* tags")

    if duplicate_topic:
        v.hard_fail(f"{len(duplicate_topic)} rows have duplicate #topic-* "
                    f"entries (same tag repeated); e.g., {duplicate_topic[:3]}")
    else:
        print(f"  ✓ no duplicate #topic-* within a single row's tags")

    if invalid_topic:
        v.hard_fail(f"{len(invalid_topic)} rows have an invalid #topic-* tag "
                    f"(not in the 50-item allowlist); e.g., {invalid_topic[:3]}")
    else:
        print(f"  ✓ every #topic-* tag is in the 50-item allowlist")

    if missing_sidecar:
        v.hard_fail(f"{len(missing_sidecar)} rows ship a #topic-* tag with no "
                    f"matching sidecar entry; e.g., {missing_sidecar[:3]}")
    else:
        print(f"  ✓ every shipped #topic-* tag has a sidecar row")

    if grammar_violation:
        v.hard_fail(f"{len(grammar_violation)} function-word rows missing "
                    f"#topic-grammar; e.g., {grammar_violation[:3]}")
    else:
        print(f"  ✓ pos∈{{prep,conj,pron,art}} → #topic-grammar invariant")

    if numbers_violation:
        v.hard_fail(f"{len(numbers_violation)} numeral rows missing "
                    f"#topic-numbers; e.g., {numbers_violation[:3]}")
    else:
        print(f"  ✓ pos=num → #topic-numbers invariant")

    if legacy_cat:
        v.hard_fail(f"{len(legacy_cat)} rows contain legacy #cat-* tags; "
                    f"e.g., {legacy_cat[:3]}")
    else:
        print(f"  ✓ no legacy #cat-* tags present")

    # --- Soft warnings ---
    zero_coverage = sorted(ALLOWED_TOPIC_TAG_SET - set(topic_counts.keys()))
    if zero_coverage:
        v.soft_warn(f"{len(zero_coverage)} topics have zero coverage: "
                    f"{zero_coverage[:5]}...")

    n_total = max(n_rows, 1)
    over_25 = [(t, c) for t, c in topic_counts.items()
               if c / n_total > TOPIC_COVERAGE_ANY_CAP]
    if over_25:
        v.soft_warn(f"{len(over_25)} topics exceed 25% of the deck "
                    f"(anti-collapse): {over_25[:3]}")

    soft_caps = (
        ("#topic-daily-routines", TOPIC_COVERAGE_DAILY_ROUTINES_CAP, "verb catchall"),
        ("#topic-objects-tools",  TOPIC_COVERAGE_OBJECTS_TOOLS_CAP, "noun catchall"),
        ("#topic-work-jobs",      TOPIC_COVERAGE_WORK_JOBS_CAP, "business-skew"),
        ("#topic-character-qualities", TOPIC_COVERAGE_CHARACTER_QUALITIES_CAP,
         "generic-adj catchall"),
        ("#topic-measurement",    TOPIC_COVERAGE_MEASUREMENT_CAP,
         "numbers/measurement confusion"),
    )
    for tag, cap, label in soft_caps:
        c = topic_counts.get(tag, 0)
        if c / n_total > cap:
            v.soft_warn(
                f"{tag} = {c}/{n_total} ({c/n_total*100:.1f}%) > "
                f"{cap*100:.0f}% soft cap ({label})"
            )

    # Only `low` is the "guessing" signal — `medium` is honest uncertainty
    # (the prompt's "broadest honest topic" fallback firing on generic
    # verbs/adjectives). Calibrated against pilot v1 which had 0% low and
    # 31% medium, all rows correctly using the catchall fallback.
    n_low = by_conf.get("low", 0)
    if n_low / n_total > TOPIC_LOW_CONF_CAP:
        v.soft_warn(f"low-confidence rows {n_low}/{n_total} "
                    f"({n_low/n_total*100:.1f}%) > "
                    f"{TOPIC_LOW_CONF_CAP*100:.0f}% soft cap")
    n_medium = by_conf.get("medium", 0)
    if n_medium > 0:
        print(f"  Stage 14: {n_medium}/{n_total} medium-confidence rows "
              f"({n_medium/n_total*100:.1f}%) — informational, not failing")


def _verify_risk_register(rows: list[dict], v: Verifier) -> None:
    """Stage 15: bp_validity / register / risk_flags / risk_note invariants.

    Hard checks (H1-H11) + soft warnings (S1-S10) per plan.
    """
    sidecar_index: dict[str, dict] = {}
    if RISK_REGISTER_PATH.exists():
        sidecar_index = {r["sense_id"]: r for r in read_tsv(RISK_REGISTER_PATH)}
    manual_index: dict[str, dict] = {}
    if MANUAL_RISK_REGISTER_PATH.exists():
        manual_index = {r["sense_id"]: r for r in read_tsv(MANUAL_RISK_REGISTER_PATH)}

    # Hard check buckets
    bad_bv: list[tuple[str, str]] = []
    bad_reg: list[tuple[str, str]] = []
    bad_flags: list[tuple[str, str]] = []
    none_mix: list[tuple[str, str]] = []
    bad_note_len: list[tuple[str, int]] = []
    bad_note_ws: list[tuple[str, str]] = []
    h6_violations: list[tuple[str, str, str]] = []   # ep_only/nonstandard + flags=none
    h7_violations: list[tuple[str, str, str]] = []   # vulgar/taboo + flags=none
    h8_violations: list[tuple[str, str]] = []         # note+none
    missing_sidecar: list[str] = []

    # Stage-12 carry-forward checks (H9/H10)
    # Build set of sense_ids that had non-empty risk_note in the prior
    # 06-final.tsv. Since we're verifying AFTER regen, we can't read the
    # "old" file directly. Instead, derive_final's audit JSONL records the
    # Stage-12 carry — but for a single-shot verifier we approximate by
    # looking at the LIVE risk_register sidecar's `old_risk_note` column
    # (which was populated from the prior 06-final.tsv at classifier time).
    stage12_carry_sids: set[str] = set()
    for sc in sidecar_index.values():
        if (sc.get("old_risk_note") or "").strip():
            stage12_carry_sids.add(sc["sense_id"])
    stage12_dropped: list[str] = []
    stage12_no_flag: list[tuple[str, str]] = []

    by_bv: dict[str, int] = {}
    by_reg: dict[str, int] = {}
    by_source: dict[str, int] = {"manual": 0, "llm": 0, "deterministic_default": 0,
                                 "llm_audit_only": 0, "": 0}
    n_rows = len(rows)
    n_flags_nonzero = 0
    n_note_nonempty = 0
    n_bp_status_ff_missing_ff_flag = 0
    n_bp_status_nsfw_missing_sensitivity = 0
    n_bp_status_uncommon_but_standard = 0
    n_reg_vulgar_no_note = 0
    n_ep_only_no_note = 0
    n_regional_br_no_flag = 0
    pt_to_validity: dict[str, set[str]] = {}     # ignore - reserved for future

    NSFW_FLAG_SET = {"sexual", "vulgar", "offensive", "slur", "racial_sensitive"}

    for r in rows:
        sid = r["sense_id"]
        bv = (r.get("bp_validity") or "").strip()
        reg = (r.get("register") or "").strip()
        flags_raw = (r.get("risk_flags") or "").strip()
        note = (r.get("risk_note") or "").strip()
        bp_status = (r.get("bp_status") or "").strip()

        # H1: bp_validity enum
        if bv not in ALLOWED_BP_VALIDITY_SET:
            bad_bv.append((sid, bv))
        # H2: register enum
        if reg not in ALLOWED_REGISTER_SET:
            bad_reg.append((sid, reg))

        # H3 + H4: risk_flags pipe-list + none-mix
        parts = [p.strip() for p in flags_raw.split("|") if p.strip()] if flags_raw else []
        invalid_in_flags = [p for p in parts if p not in ALLOWED_RISK_FLAGS_SET]
        if invalid_in_flags or not parts:
            bad_flags.append((sid, flags_raw))
        elif "none" in parts and len(parts) > 1:
            none_mix.append((sid, flags_raw))

        # H5: risk_note length + whitespace
        if len(note) > RISK_NOTE_MAX_CHARS:
            bad_note_len.append((sid, len(note)))
        if any(ch in note for ch in ("\t", "\n", "\r")):
            bad_note_ws.append((sid, note[:60]))

        # H6: bp_validity ∈ {ep_only, nonstandard} ⇒ flags != none
        if bv in {"ep_only", "nonstandard"} and flags_raw == "none":
            # Allow when manual override explicitly set it (defensive).
            if sid not in manual_index:
                h6_violations.append((sid, bv, flags_raw))

        # H7: register ∈ {vulgar, taboo} ⇒ flags != none
        if reg in {"vulgar", "taboo"} and flags_raw == "none":
            if sid not in manual_index:
                h7_violations.append((sid, reg, flags_raw))

        # H8: note non-empty ⇒ flags != none
        if note and flags_raw == "none":
            h8_violations.append((sid, note[:60]))

        # H9 + H10: Stage-12 carry-forward
        if sid in stage12_carry_sids:
            if not note:
                stage12_dropped.append(sid)
            if flags_raw == "none":
                stage12_no_flag.append((sid, note[:60]))

        # H11: shipped non-default rows must have a sidecar entry
        # The default source is "deterministic_default"; manual and llm rows
        # always come from sidecar. So we just need to check rows whose
        # sidecar source isn't deterministic_default OR which have non-default
        # values — every such row must be in the sidecar.
        sc = sidecar_index.get(sid)
        if sc is None and (bv != "standard" or reg != "neutral" or flags_raw != "none" or note):
            missing_sidecar.append(sid)

        # Distribution + soft-warning aggregations
        by_bv[bv] = by_bv.get(bv, 0) + 1
        by_reg[reg] = by_reg.get(reg, 0) + 1
        src = (sc.get("source") if sc else "").strip()
        by_source[src] = by_source.get(src, 0) + 1
        if flags_raw != "none":
            n_flags_nonzero += 1
        if note:
            n_note_nonempty += 1
        if bp_status == "false_friend" and "false_friend" not in parts:
            n_bp_status_ff_missing_ff_flag += 1
        if bp_status == "nsfw" and not (set(parts) & NSFW_FLAG_SET):
            n_bp_status_nsfw_missing_sensitivity += 1
        if bp_status == "uncommon" and bv == "standard":
            n_bp_status_uncommon_but_standard += 1
        if reg in {"vulgar", "taboo"} and not note:
            n_reg_vulgar_no_note += 1
        if bv == "ep_only" and not note:
            n_ep_only_no_note += 1
        if bv == "regional_br" and flags_raw == "none":
            n_regional_br_no_flag += 1

    uncertain_bv_pct = (by_bv.get("uncertain", 0) / n_rows) if n_rows else 0
    uncertain_reg_pct = (by_reg.get("uncertain", 0) / n_rows) if n_rows else 0
    flags_nonzero_pct = (n_flags_nonzero / n_rows) if n_rows else 0
    note_nonempty_pct = (n_note_nonempty / n_rows) if n_rows else 0

    print(f"  Stage 15: {n_rows} rows; "
          f"sources: manual={by_source.get('manual',0)} "
          f"llm={by_source.get('llm',0)} "
          f"default={by_source.get('deterministic_default',0)}; "
          f"flags!=none={n_flags_nonzero} ({flags_nonzero_pct*100:.1f}%); "
          f"risk_note non-empty={n_note_nonempty} ({note_nonempty_pct*100:.1f}%)")

    # ----- Hard checks -----
    if bad_bv:
        v.hard_fail(f"{len(bad_bv)} rows have invalid bp_validity; e.g., {bad_bv[:3]}")
    else:
        print(f"  ✓ every bp_validity in allowed enum (H1)")
    if bad_reg:
        v.hard_fail(f"{len(bad_reg)} rows have invalid register; e.g., {bad_reg[:3]}")
    else:
        print(f"  ✓ every register in allowed enum (H2)")
    if bad_flags:
        v.hard_fail(f"{len(bad_flags)} rows have invalid risk_flags; e.g., {bad_flags[:3]}")
    else:
        print(f"  ✓ every risk_flags in allowed pipe-list (H3)")
    if none_mix:
        v.hard_fail(f"{len(none_mix)} rows mix 'none' with other flags; e.g., {none_mix[:3]}")
    else:
        print(f"  ✓ 'none' never combined with other flags (H4)")
    if bad_note_len or bad_note_ws:
        v.hard_fail(f"{len(bad_note_len) + len(bad_note_ws)} risk_note format violations "
                    f"(too long or whitespace artifacts)")
    else:
        print(f"  ✓ risk_note ≤ {RISK_NOTE_MAX_CHARS} chars + no embedded tab/newline (H5)")
    if h6_violations:
        v.hard_fail(f"{len(h6_violations)} bp_validity∈{{ep_only,nonstandard}} rows "
                    f"with risk_flags=none; e.g., {h6_violations[:3]}")
    else:
        print(f"  ✓ ep_only/nonstandard ⇒ risk_flags!=none (H6)")
    if h7_violations:
        v.hard_fail(f"{len(h7_violations)} vulgar/taboo register rows with "
                    f"risk_flags=none; e.g., {h7_violations[:3]}")
    else:
        print(f"  ✓ vulgar/taboo ⇒ risk_flags!=none (H7)")
    if h8_violations:
        v.hard_fail(f"{len(h8_violations)} rows have non-empty risk_note but "
                    f"risk_flags=none; e.g., {h8_violations[:3]}")
    else:
        print(f"  ✓ non-empty risk_note ⇒ risk_flags!=none (H8)")
    if stage12_dropped:
        v.hard_fail(f"{len(stage12_dropped)} Stage-12 risk_note(s) silently dropped; "
                    f"e.g., {stage12_dropped[:3]}")
    else:
        print(f"  ✓ no Stage-12 risk_note dropped (H9)")
    if stage12_no_flag:
        v.hard_fail(f"{len(stage12_no_flag)} Stage-12 risk_note rows ship with "
                    f"risk_flags=none; e.g., {stage12_no_flag[:3]}")
    else:
        print(f"  ✓ Stage-12 carry-forward rows have risk_flags!=none (H10)")
    if missing_sidecar:
        v.hard_fail(f"{len(missing_sidecar)} non-default rows missing sidecar entry; "
                    f"e.g., {missing_sidecar[:3]}")
    else:
        print(f"  ✓ every non-default shipped row has a sidecar entry (H11)")

    # ----- Soft warnings -----
    if uncertain_bv_pct > 0.01:
        v.soft_warn(f"bp_validity=uncertain on {uncertain_bv_pct*100:.1f}% of rows (cap 1%)")
    if uncertain_reg_pct > 0.01:
        v.soft_warn(f"register=uncertain on {uncertain_reg_pct*100:.1f}% of rows (cap 1%)")
    if flags_nonzero_pct > 0.10:
        v.soft_warn(f"risk_flags!=none on {flags_nonzero_pct*100:.1f}% of rows (cap 10%)")
    if note_nonempty_pct > 0.03:
        v.soft_warn(f"risk_note non-empty on {note_nonempty_pct*100:.1f}% of rows (cap 3%)")
    if n_bp_status_ff_missing_ff_flag:
        v.soft_warn(f"{n_bp_status_ff_missing_ff_flag} bp_status=false_friend rows "
                    f"lack the false_friend flag (S5)")
    if n_bp_status_nsfw_missing_sensitivity:
        v.soft_warn(f"{n_bp_status_nsfw_missing_sensitivity} bp_status=nsfw rows "
                    f"lack a sensitivity flag (S6)")
    if n_bp_status_uncommon_but_standard:
        v.soft_warn(f"{n_bp_status_uncommon_but_standard} bp_status=uncommon rows "
                    f"shipped bp_validity=standard (S7)")
    if n_reg_vulgar_no_note:
        v.soft_warn(f"{n_reg_vulgar_no_note} vulgar/taboo rows have empty risk_note (S8)")
    if n_ep_only_no_note:
        v.soft_warn(f"{n_ep_only_no_note} ep_only rows have empty risk_note (S9)")
    if n_regional_br_no_flag:
        v.soft_warn(f"{n_regional_br_no_flag} regional_br rows ship risk_flags=none "
                    f"(S10 — verify each is genuinely harmless)")


def _http_sample(rows: list[dict], n: int, v: Verifier) -> None:
    print(f"\n[HTTP-sample] checking {n} random URLs return 200...")
    rng = random.Random(42)
    sampled = rng.sample(rows, min(n, len(rows)))
    failures: list[tuple[str, int]] = []
    for i, r in enumerate(sampled, start=1):
        for col in ("audio_word", "audio_example"):
            url = r[col]
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    code = resp.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:
                print(f"  {r['sense_id']} {col}: error {e}", file=sys.stderr)
                code = -1
            if code != 200:
                failures.append((url, code))
        if i % 10 == 0:
            print(f"  checked {i}/{len(sampled)} senses ({i*2} URLs)")
    if failures:
        v.hard_fail(f"{len(failures)} URLs not HTTP 200; e.g., {failures[:3]}")
    else:
        print(f"  ✓ all {len(sampled)*2} sampled URLs return HTTP 200")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--http-sample", type=int, default=50,
                        help="How many random rows to HTTP-check (0 = skip).")
    args = parser.parse_args()

    if not FINAL_PATH.exists():
        print(f"ERROR: {FINAL_PATH} not found. Run build/derive_final.py first.",
              file=sys.stderr)
        return 1

    rows = read_tsv(FINAL_PATH)
    voice_gender = {r["voice_id"]: r.get("gender", "")
                    for r in read_tsv(VOICES_PATH)}

    v = Verifier()
    print("[verify_all] checking invariants on 06-final.tsv...")
    _verify_header(v)
    _verify_rows(rows, voice_gender, v)
    _verify_usage_hints(rows, v)
    _verify_family_roots(rows, v)
    _verify_topic_tags(rows, v)
    _verify_risk_register(rows, v)
    if args.http_sample > 0:
        _http_sample(rows, args.http_sample, v)

    print()
    if v.passed():
        print(f"✓ verify_all PASSED  (hard=0, soft={len(v.soft_warnings)})")
        return 0
    else:
        print(f"✗ verify_all FAILED  (hard={len(v.hard_failures)}, soft={len(v.soft_warnings)})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
