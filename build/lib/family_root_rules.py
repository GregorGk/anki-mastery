"""Stage 13 — Family-root candidate generation + idempotent post-LLM rules.

The classifier (build/13_0_family_root_classifier.py) deterministically
generates candidate root lemmas from the deck, asks Claude to pick one, then
this module applies hard rules on top:

  1. Strip whitespace; reject empty / hallucinated / bare-stem roots.
  2. Require confidence='high' for a root to ship (the export filter in
     derive_final.py re-checks this, but we also enforce it at sidecar-write
     time so the audit trail is clean).
  3. Deck-level cluster-size enforcement (≥ 2 distinct `pt` lemmas).
  4. Self-root propagation: any `pt` that survives as a cluster head also
     gets `family_root = pt` on every row where it appears (so the head
     belongs to its own family for Anki sibling-spacing logic).

Candidate generation lives here too because both the classifier and any
reruns / tests need to call the same deterministic logic.

All passes are idempotent — re-applying them yields the same output.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONTENT_POS = ("noun", "verb", "adj", "adv")


# Forward suffix-strip: turn the target into a possible base form. Each entry
# maps a target-suffix to a set of base-suffix candidates. Example: a row whose
# pt ends in "ção" might derive from a verb in -ar/-er/-ir (decisão → decidir).
SUFFIX_HINTS: dict[str, tuple[str, ...]] = {
    "ção":   ("ar", "er", "ir"),         # decisão → decidir
    "são":   ("der", "dir", "ter", "tir"),
    "mento": ("ar", "er", "ir"),         # movimento → mover
    "dade":  ("", "o", "a"),             # felicidade → feliz
    "dor":   ("ar", "er", "ir"),         # trabalhador → trabalhar
    "dora":  ("ar", "er", "ir"),
    "nte":   ("ar", "er", "ir"),         # estudante → estudar
    "vel":   ("ar", "er", "ir"),         # amável → amar
    "ivo":   ("ar", "ir"),
    "iva":   ("ar", "ir"),
    "ado":   ("ar",),                    # participles
    "ada":   ("ar",),
    "ido":   ("er", "ir"),
    "ida":   ("er", "ir"),
    "mente": ("", "o"),                  # felizmente → feliz
}


# Negation / iteration prefixes. One-hop only — the LLM bridges multi-hop chains.
NEGATION_PREFIXES: tuple[str, ...] = ("in", "im", "des", "anti", "re")


# Explicit recall booster for highest-value business/abstract families. Used
# ONLY to guarantee candidate inclusion; the LLM still decides. Not
# authoritative truth.
SEED_FAMILIES: dict[str, tuple[str, ...]] = {
    "decidir":   ("decisão", "decidido", "decisivo", "indeciso"),
    "produzir":  ("produto", "produção", "produtivo", "produtividade", "produtor"),
    "negociar":  ("negociação", "negociador", "negociável"),
    "criar":     ("criação", "criador", "criativo"),
    "organizar": ("organização", "organizado", "organizador"),
    "feliz":     ("felicidade", "infeliz", "infelizmente"),
}


# Caps on the noisy prefix-sibling generator. The combined candidate set per
# row is also capped (see _cap_candidates). These thresholds were tightened
# after the dry-run showed common Latin prefixes (inter-, trans-, cons-,
# esp-, contr-, corr-) producing 10–12 noisy hits each via the 5-char
# fallback. The 7-char preferred / 6-char fallback combination kills most of
# that noise without losing legit suffix-strip matches (those come from the
# (a)/(b) generators).
PREFIX_SIBLING_MIN_LEN = 6
PREFIX_SIBLING_PREFER_LEN = 7
PREFIX_SIBLING_RANK_CEILING = 3000
PREFIX_SIBLING_MAX = 6
TOTAL_CANDIDATES_MAX = 15


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def normalize(s: str) -> str:
    """Strip accents, lowercase, drop hyphens and whitespace. For matching only.

    The original accented form is preserved everywhere else.
    """
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return (
        no_marks
        .lower()
        .replace("-", "")
        .replace(" ", "")
    )


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PtInfo:
    """Minimal per-pt info used during candidate generation."""
    pt: str
    pos: str
    rank: int
    en_primary: str


def _suffix_strip_candidates(
    target_norm: str, norm_to_pts: dict[str, list[PtInfo]]
) -> set[str]:
    """(a) Forward suffix-strip — try every (suffix, replacement) pair."""
    out: set[str] = set()
    for suffix, replacements in SUFFIX_HINTS.items():
        if not target_norm.endswith(suffix):
            continue
        stem = target_norm[: -len(suffix)] if suffix else target_norm
        for repl in replacements:
            candidate = stem + repl
            for info in norm_to_pts.get(candidate, []):
                out.add(info.pt)
    return out


def _reverse_suffix_candidates(
    target_norm: str, norm_to_pts: dict[str, list[PtInfo]]
) -> set[str]:
    """(b) Reverse suffix-strip — find derivatives of the target.

    Iterate over every known `pt`; if applying the forward rule to that pt
    lands at the target's normalized form, then that pt is a derivative.
    """
    out: set[str] = set()
    for cand_norm, infos in norm_to_pts.items():
        for suffix, replacements in SUFFIX_HINTS.items():
            if not cand_norm.endswith(suffix):
                continue
            stem = cand_norm[: -len(suffix)] if suffix else cand_norm
            for repl in replacements:
                if stem + repl == target_norm:
                    for info in infos:
                        out.add(info.pt)
    return out


def _negation_prefix_candidates(
    target_norm: str, norm_to_pts: dict[str, list[PtInfo]]
) -> set[str]:
    """(c) Strip a one-hop negation prefix; look up in the dataset."""
    out: set[str] = set()
    for prefix in NEGATION_PREFIXES:
        if not target_norm.startswith(prefix) or len(target_norm) <= len(prefix) + 1:
            continue
        bare = target_norm[len(prefix):]
        for info in norm_to_pts.get(bare, []):
            out.add(info.pt)
    return out


def _prefix_sibling_candidates(
    target_pt: str,
    target_norm: str,
    norm_to_pts: dict[str, list[PtInfo]],
    pt_to_info: dict[str, PtInfo],
) -> list[str]:
    """(d) Normalized-prefix sibling match, capped and ranked.

    Try 6-char prefix first; only fall back to 5 if 6-char yielded < 2 hits.
    Order: content-words first, then rank ASC (preferring rank <= 3000), then
    normalized-edit-distance via prefix-length tie-break.
    """
    if len(target_norm) < PREFIX_SIBLING_MIN_LEN:
        return []

    def collect(prefix_len: int) -> list[PtInfo]:
        if len(target_norm) < prefix_len:
            return []
        prefix = target_norm[:prefix_len]
        seen: set[str] = set()
        bucket: list[PtInfo] = []
        for norm, infos in norm_to_pts.items():
            if not norm.startswith(prefix):
                continue
            for info in infos:
                if info.pt == target_pt or info.pt in seen:
                    continue
                seen.add(info.pt)
                bucket.append(info)
        return bucket

    bucket = collect(PREFIX_SIBLING_PREFER_LEN)
    if len(bucket) < 2 and len(target_norm) >= PREFIX_SIBLING_MIN_LEN:
        bucket = collect(PREFIX_SIBLING_MIN_LEN)

    def sort_key(info: PtInfo) -> tuple[int, int, int, str]:
        is_content = 0 if info.pos in CONTENT_POS else 1
        rank_bucket = 0 if (info.rank and info.rank <= PREFIX_SIBLING_RANK_CEILING) else 1
        # Stable tie-break: shorter norm distance from target_norm via
        # length difference, then alphabetical.
        len_dist = abs(len(normalize(info.pt)) - len(target_norm))
        return (is_content, rank_bucket, len_dist, info.pt)

    bucket.sort(key=sort_key)
    return [info.pt for info in bucket[:PREFIX_SIBLING_MAX]]


def _seed_family_candidates(target_pt: str, all_pts: set[str]) -> set[str]:
    """(e) SEED_FAMILIES recall booster — guarantee coverage of high-value
    business/abstract families when the suffix tables miss them."""
    out: set[str] = set()
    for head, members in SEED_FAMILIES.items():
        family = {head, *members}
        if target_pt not in family:
            continue
        # Only add entries that are actually in the deck.
        for m in family:
            if m == target_pt:
                continue
            if m in all_pts:
                out.add(m)
    return out


def _cap_candidates(target_pt: str, ordered: list[str]) -> list[str]:
    """Cap the merged candidate list, preserving order of first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for pt in ordered:
        if pt == target_pt or pt in seen:
            continue
        seen.add(pt)
        out.append(pt)
        if len(out) >= TOTAL_CANDIDATES_MAX:
            break
    return out


def generate_candidates(
    target_pt: str,
    target_pos: str,
    pt_to_info: dict[str, PtInfo],
    norm_to_pts: dict[str, list[PtInfo]],
) -> list[str]:
    """Return ordered candidate root lemmas for the target row.

    Order: suffix-strip (a) > reverse-suffix (b) > negation-prefix (c) >
    seed-families (e) > prefix-sibling (d). (d) is the noisy generator
    capped to PREFIX_SIBLING_MAX; the total list is capped at
    TOTAL_CANDIDATES_MAX.
    """
    if target_pos not in CONTENT_POS:
        return []
    target_norm = normalize(target_pt)
    if not target_norm:
        return []

    all_pts = set(pt_to_info.keys())

    sa = _suffix_strip_candidates(target_norm, norm_to_pts)
    sb = _reverse_suffix_candidates(target_norm, norm_to_pts)
    sc = _negation_prefix_candidates(target_norm, norm_to_pts)
    seeds = _seed_family_candidates(target_pt, all_pts)
    sd = _prefix_sibling_candidates(target_pt, target_norm, norm_to_pts, pt_to_info)

    # Merge order: high-precision generators first.
    ordered: list[str] = []
    for source in (sa, sb, sc, seeds):
        for pt in sorted(source):
            if pt != target_pt and pt not in ordered:
                ordered.append(pt)
    # Then the prefix-sibling list, preserving its rank-order.
    for pt in sd:
        if pt != target_pt and pt not in ordered:
            ordered.append(pt)

    return _cap_candidates(target_pt, ordered)


def build_pt_indexes(rows: Iterable[dict]) -> tuple[dict[str, PtInfo], dict[str, list[PtInfo]]]:
    """Build the two indexes used by candidate generation.

    Returns (pt_to_info, norm_to_pts):
      - pt_to_info[pt]: PtInfo for the lowest-sense_id row with that pt
      - norm_to_pts[normalize(pt)]: list of PtInfo for every pt that
        normalizes to that key (handles homographs)
    """
    by_pt: dict[str, PtInfo] = {}
    for r in rows:
        pt = (r.get("pt") or "").strip()
        if not pt:
            continue
        sense_id = r.get("sense_id") or ""
        try:
            rank = int(r.get("rank") or "0")
        except ValueError:
            rank = 0
        info = PtInfo(
            pt=pt,
            pos=(r.get("pos") or "").strip(),
            rank=rank,
            en_primary=(r.get("en_primary") or "").strip(),
        )
        existing = by_pt.get(pt)
        if existing is None:
            by_pt[pt] = info
            continue
        # Keep the lower-sense_id (= lower rank usually) entry. The list
        # we're iterating isn't guaranteed sorted, so compare explicitly.
        prev_rank = existing.rank or 10**9
        new_rank = info.rank or 10**9
        if new_rank < prev_rank:
            by_pt[pt] = info

    norm_to_pts: dict[str, list[PtInfo]] = defaultdict(list)
    for info in by_pt.values():
        norm_to_pts[normalize(info.pt)].append(info)
    return by_pt, dict(norm_to_pts)


# --------------------------------------------------------------------------- #
# Per-row post-process (idempotent)
# --------------------------------------------------------------------------- #


_TRAILING_HYPHEN_RE = re.compile(r"-$")
_DIGIT_RE = re.compile(r"\d")


def apply_post_process(row: dict, all_pts: set[str]) -> dict:
    """Mutate `row` in place with the per-row deterministic passes.

    Expected keys (writes them, creates missing ones with defaults):
      family_root, family_relation, confidence, reason, source

    Returns the same dict.
    """
    row.setdefault("family_relation", "")
    row.setdefault("confidence", "low")
    row.setdefault("reason", "")
    row.setdefault("source", "llm")

    root = (row.get("family_root") or "").strip()
    confidence = (row.get("confidence") or "").strip().lower() or "low"

    # 1. Reject empty.
    if not root:
        row["family_root"] = ""
        row["confidence"] = confidence
        return row

    # 2. Reject hallucinated roots not in the dataset.
    if root not in all_pts:
        row["family_root"] = ""
        row["reason"] = (
            f"hallucinated_root (LLM picked {root!r}, not in dataset); "
            + (row.get("reason") or "")
        ).strip()
        row["confidence"] = confidence
        return row

    # 3. Reject bare stems / digit-containing strings.
    if _TRAILING_HYPHEN_RE.search(root) or _DIGIT_RE.search(root):
        row["family_root"] = ""
        row["reason"] = (
            f"bare_stem_rejected ({root!r}); " + (row.get("reason") or "")
        ).strip()
        row["confidence"] = confidence
        return row

    # 4. Drop low-confidence roots from shipping, but keep audit fields.
    # Exception: postprocess_self_root provenance is deterministic and ships
    # at high confidence regardless of LLM confidence value.
    if (row.get("source") or "llm") != "postprocess_self_root":
        if confidence != "high":
            row["family_root"] = ""
            # leave reason / family_relation alone for audit
            row["confidence"] = confidence
            return row

    row["family_root"] = root
    row["confidence"] = confidence
    return row


# --------------------------------------------------------------------------- #
# Deck-level passes
# --------------------------------------------------------------------------- #


def cluster_pt_counts(
    rows: list[dict],
    *,
    all_rows_pts: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """For each non-empty `family_root` value Y, return the set of distinct
    `pt` lemmas that belong to Y's family.

    Membership: a `pt` belongs to Y's family if either (a) one of its rows
    has `family_root == Y`, or (b) the lemma `pt` itself equals Y (the head
    counts as a member).

    `rows` is the sidecar (or final TSV view). Each row must have `pt` and
    `family_root`. Optionally pass `all_rows_pts` to also fold in heads that
    only appear in the final TSV but not in the sidecar — useful for the
    self-root propagation pass.
    """
    out: dict[str, set[str]] = defaultdict(set)
    pts_seen: set[str] = set()
    for r in rows:
        pt = (r.get("pt") or "").strip()
        root = (r.get("family_root") or "").strip()
        if pt:
            pts_seen.add(pt)
        if root and pt:
            out[root].add(pt)
    # Heads that exist as a pt in the (sidecar) dataset count as a member.
    for root in list(out.keys()):
        if root in pts_seen:
            out[root].add(root)
    if all_rows_pts is not None:
        for root in list(out.keys()):
            if root in all_rows_pts:
                out[root].add(root)
    return dict(out)


def enforce_cluster_size(rows: list[dict], *, min_distinct_pts: int = 2) -> int:
    """Clear `family_root` on rows whose root's cluster has < min distinct pts.

    Mutates `rows` in place. Returns the count of rows cleared.
    """
    clusters = cluster_pt_counts(rows)
    bad_roots = {r for r, pts in clusters.items() if len(pts) < min_distinct_pts}
    if not bad_roots:
        return 0
    cleared = 0
    for r in rows:
        if (r.get("family_root") or "").strip() in bad_roots:
            r["family_root"] = ""
            # Demote reason / family_relation provenance so the audit trail
            # records the cluster-eviction without losing the LLM's call log.
            existing_reason = r.get("reason") or ""
            r["reason"] = ("cluster_size_evicted; " + existing_reason).strip()
            r["family_relation"] = ""
            cleared += 1
    return cleared


def propagate_self_root(
    sidecar_rows: list[dict],
    *,
    deck_rows_by_pt: dict[str, list[dict]],
    timestamp_iso: str,
    model_id: str,
) -> list[dict]:
    """For every cluster head that survived the cluster-size cut, ensure each
    `pt == head` row in the deck has a sidecar entry with
    `family_root = head`, provenance `postprocess_self_root`.

    Returns a list of NEW sidecar rows to append (one per missing head sense).
    The caller is responsible for merging these into the sidecar file.
    """
    # Collect surviving roots.
    clusters = cluster_pt_counts(sidecar_rows)
    surviving_roots = {r for r, pts in clusters.items() if len(pts) >= 2}

    # Existing sidecar entries by sense_id (so we don't double-write).
    by_sense: dict[str, dict] = {
        r["sense_id"]: r for r in sidecar_rows if r.get("sense_id")
    }
    new_rows: list[dict] = []
    for root in sorted(surviving_roots):
        deck_rows = deck_rows_by_pt.get(root, [])
        for deck_row in deck_rows:
            sid = deck_row.get("sense_id") or ""
            pos = deck_row.get("pos") or ""
            if not sid or pos not in CONTENT_POS:
                continue
            existing = by_sense.get(sid)
            if existing and (existing.get("family_root") or "").strip():
                # Already set (probably the head sense was processed by the LLM).
                continue
            if existing:
                # Existing sidecar row but with empty family_root — update in place.
                existing["family_root"] = root
                existing["family_relation"] = "self"
                existing["confidence"] = "high"
                existing["reason"] = "postprocess_self_root"
                existing["source"] = "postprocess_self_root"
                existing.setdefault("pos", pos)
                existing.setdefault("pt", root)
                existing.setdefault("en_primary", deck_row.get("en_primary", ""))
                existing.setdefault("model_id", model_id)
                existing.setdefault("generated_at", timestamp_iso)
                continue
            # No sidecar row exists — append a new one.
            new_rows.append({
                "sense_id":        sid,
                "pt":              root,
                "pos":             pos,
                "en_primary":      deck_row.get("en_primary", ""),
                "family_root":     root,
                "family_relation": "self",
                "confidence":      "high",
                "reason":          "postprocess_self_root",
                "source":          "postprocess_self_root",
                "model_id":        model_id,
                "generated_at":    timestamp_iso,
            })
    return new_rows
