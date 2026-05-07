"""Parsing primitives for Stage 1a.

Responsibilities:
- Split a source.txt line on the FIRST ` = ` only (handles 3 embedded-= cases).
- Extract parentheticals from the RHS into a structured `annotation` dict.
- Detect false-positives that must NOT be misinterpreted as gender markers
  or idiom abbreviations: `M. Gerais`, `OBJ = me/vos`, `e.g.`, `i.e.`.
- Detect idiom-expansion candidates: parentheticals of the form
  `(<preposition> <abbrev>.)` where the abbreviation initial matches the headword.
- Classify the headword's `pt_type`: single_word | hyphenated_compound |
  space_compound | idiom | abbreviation_expansion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --- Constants ---------------------------------------------------------------

# False-positive patterns: parentheticals that look like markers but aren't.
# Each entry is a regex matched against parenthetical INNER text (no parens).
FALSE_POSITIVE_PATTERNS = [
    re.compile(r"^M\. Gerais"),  # mina = mine (M. Gerais: state in B)
    re.compile(r"^OBJ\s*=", re.IGNORECASE),  # eu = I (OBJ = me); vós (OBJ = vos)
    re.compile(r"^e\.g\.", re.IGNORECASE),  # `\b` after `.` doesn't anchor as expected
    re.compile(r"^i\.e\.", re.IGNORECASE),
]

# Pattern for idiom abbreviation candidates inside parentheticals.
# Matches things like:
#   (em d.)
#   (a m. que)
#   (em / ao r.)
#   (à t.)
#   (de r.)
#   (não o.)
#   (a m. de)
#   (por c.)
#   (ao i.)
# Must contain at least one single-letter-with-period token.
IDIOM_ABBREV_PATTERN = re.compile(
    r"\b([a-zà-ÿ])\.(?:\s|$|\)|,)",
    re.IGNORECASE | re.UNICODE,
)

# Hyphenation detection (informational; the actual rule application is in bp_rules.py).
HYPHEN_RE = re.compile(r"-")
WHITESPACE_RE = re.compile(r"\s+")


# --- Data classes ------------------------------------------------------------


@dataclass
class ParsedLine:
    """Result of parsing one source.txt line."""

    source_line_number: int
    source_raw: str
    pt: str  # Headword (LHS of first ` = `)
    en_all: str  # Full RHS as-is
    pt_type: str  # single_word | hyphenated_compound | space_compound | idiom
    annotation: dict = field(default_factory=dict)
    parentheticals: list[str] = field(default_factory=list)
    idiom_candidates: list[str] = field(default_factory=list)
    has_bp_marker: bool = False
    has_ep_marker: bool = False
    has_mainly_ep_marker: bool = False
    has_reflexive_se: bool = False
    has_gender_split: bool = False  # (M ... / F ...) or (F ... / M ...)


# --- Public API --------------------------------------------------------------


def split_on_first_equals(line: str) -> tuple[str, str] | None:
    """Split on FIRST ` = ` only. Returns (pt, en_all) or None if malformed.

    Handles the 3 embedded-= cases (lines 31, 314, 3372 in source.txt).
    """
    line = line.rstrip("\n").rstrip()
    if not line:
        return None
    idx = line.find(" = ")
    if idx == -1:
        return None
    pt = line[:idx].strip()
    en_all = line[idx + 3 :].strip()
    if not pt:
        return None
    return pt, en_all


def extract_parentheticals(rhs: str) -> list[str]:
    """Extract top-level parenthetical groups from the RHS.

    Returns inner text only (no parens). Skips nested parens by tracking depth.
    """
    out: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(rhs):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(rhs[start:i])
                start = -1
            elif depth < 0:
                # Unbalanced — reset
                depth = 0
                start = -1
    return out


def is_false_positive(parenthetical_inner: str) -> bool:
    """Return True if this parenthetical should NOT be treated as a marker/idiom."""
    text = parenthetical_inner.strip()
    return any(p.search(text) for p in FALSE_POSITIVE_PATTERNS)


def detect_idiom_candidate(
    parenthetical_inner: str, headword: str
) -> Optional[str]:
    """Return the parenthetical text if it looks like a headword-abbreviation idiom.

    Rule: the parenthetical contains at least one single-letter-period token,
    AND that letter (case-insensitive) matches the headword's first letter,
    AND it's NOT a false-positive pattern.
    """
    if is_false_positive(parenthetical_inner):
        return None
    if not headword:
        return None
    head_initial = headword[0].lower()
    matches = IDIOM_ABBREV_PATTERN.findall(parenthetical_inner)
    if not matches:
        return None
    # Initial must match the headword's
    if any(m.lower() == head_initial for m in matches):
        return parenthetical_inner.strip()
    return None


def detect_pt_type(pt: str) -> str:
    """Classify the headword's surface type."""
    if HYPHEN_RE.search(pt):
        return "hyphenated_compound"
    if WHITESPACE_RE.search(pt):
        return "space_compound"
    return "single_word"


def detect_markers(rhs: str) -> dict:
    """Detect (BP), (EP), (mainly EP), (+se), and M/F gender split markers in RHS."""
    out = {
        "has_bp_marker": bool(re.search(r"\(BP\)", rhs)),
        "has_ep_marker": bool(re.search(r"\(EP\)", rhs)),
        "has_mainly_ep_marker": bool(re.search(r"mainly\s+EP", rhs, re.IGNORECASE)),
        "has_reflexive_se": bool(re.search(r"\(\+se\)", rhs)),
        # M/F gender split: "(M ... / F ...)" or "(F ... / M ...)"
        # Must NOT match `M. Gerais`. We require the M/F to be standalone tokens.
        "has_gender_split": bool(
            re.search(r"\((?:M\s+[^/]+/\s*F\s|F\s+[^/]+/\s*M\s)", rhs)
        )
        or bool(
            re.search(r"\([^)]*\(M\)\s*/[^/]*\(F\)|\([^)]*\(F\)\s*/[^/]*\(M\)", rhs)
        )
        or bool(
            # Pattern like: "cabra = goat (F) / guy (M)"  (gender markers per sense)
            re.search(r"\(M\)[^)]*\(F\)|\(F\)[^)]*\(M\)", rhs)
        ),
    }
    return out


def parse_line(source_line_number: int, raw: str) -> ParsedLine | None:
    """Parse one source.txt line into a ParsedLine. Returns None on malformed input."""
    split = split_on_first_equals(raw)
    if split is None:
        return None
    pt, en_all = split

    parens = extract_parentheticals(en_all)
    idiom_candidates: list[str] = []
    for inner in parens:
        cand = detect_idiom_candidate(inner, pt)
        if cand:
            idiom_candidates.append(cand)

    markers = detect_markers(en_all)
    annotation: dict = {}
    if markers["has_bp_marker"]:
        annotation["bp_marker"] = True
    if markers["has_ep_marker"]:
        annotation["ep_marker"] = True
    if markers["has_mainly_ep_marker"]:
        annotation["mainly_ep_marker"] = True
    if markers["has_reflexive_se"]:
        annotation["reflexive"] = True
    if markers["has_gender_split"]:
        annotation["gender_split"] = True
    if parens:
        annotation["parentheticals"] = parens

    return ParsedLine(
        source_line_number=source_line_number,
        source_raw=raw.rstrip("\n"),
        pt=pt,
        en_all=en_all,
        pt_type=detect_pt_type(pt),
        annotation=annotation,
        parentheticals=parens,
        idiom_candidates=idiom_candidates,
        **markers,
    )
