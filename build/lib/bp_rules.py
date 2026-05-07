"""Stage 1b helpers: orthographic spelling map + post-1990 hyphen rules.

These are deterministic, table-driven transformations:

- `_ep_spelling_map.tsv`: whole-word replacements for known EP forms.
- `_hyphen_rules.tsv`: literal `dehyphenate_to_spaces` and regex `preserve`.

Lexical replacement (`comboio` → `trem`) lives in Stage 1c, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .tsv import read_tsv


@dataclass
class SpellingRule:
    source_form: str
    bp_form: str
    rule_type: str
    confidence: str
    note: str = ""


@dataclass
class HyphenRule:
    pattern: str
    pattern_type: str  # 'literal' or 'regex'
    action: str  # 'dehyphenate_to_spaces' or 'preserve'
    rule_name: str
    note: str = ""
    _compiled: re.Pattern | None = None

    def matches(self, pt: str) -> bool:
        if self.pattern_type == "literal":
            return pt == self.pattern
        if self.pattern_type == "regex":
            if self._compiled is None:
                self._compiled = re.compile(self.pattern, re.IGNORECASE)
            return bool(self._compiled.search(pt))
        return False


# --- Loaders -----------------------------------------------------------------


def load_spelling_map(path: str | Path) -> dict[str, SpellingRule]:
    """Read `_ep_spelling_map.tsv` and key by source_form."""
    out: dict[str, SpellingRule] = {}
    for row in read_tsv(path):
        rule = SpellingRule(
            source_form=row["source_form"],
            bp_form=row["bp_form"],
            rule_type=row.get("rule_type", ""),
            confidence=row.get("confidence", ""),
            note=row.get("note", ""),
        )
        out[rule.source_form] = rule
    return out


def load_hyphen_rules(path: str | Path) -> list[HyphenRule]:
    """Read `_hyphen_rules.tsv` in table order. Order matters for application."""
    out: list[HyphenRule] = []
    for row in read_tsv(path):
        out.append(
            HyphenRule(
                pattern=row["pattern"],
                pattern_type=row.get("pattern_type", "literal"),
                action=row.get("action", "preserve"),
                rule_name=row.get("rule_name", ""),
                note=row.get("note", ""),
            )
        )
    return out


# --- Rule application --------------------------------------------------------


def apply_spelling_rule(pt: str, spelling_map: dict[str, SpellingRule]) -> tuple[str, str | None]:
    """Whole-word EP→BP normalization.

    Returns (normalized_pt, rule_type_applied | None).
    """
    rule = spelling_map.get(pt)
    if rule is None:
        return pt, None
    return rule.bp_form, rule.rule_type


def apply_hyphen_rule(pt: str, rules: list[HyphenRule]) -> tuple[str, str | None]:
    """Apply post-1990 hyphen rules.

    Order:
      1. If no hyphen in pt → no-op, return (pt, None).
      2. Literal `dehyphenate_to_spaces` rules (most specific).
      3. Regex `preserve` rules (a positive marker that the hyphen is intended).
      4. Default: keep as-is.
    """
    if "-" not in pt:
        return pt, None

    # 1. Literal dehyphenate first.
    for rule in rules:
        if (
            rule.pattern_type == "literal"
            and rule.action == "dehyphenate_to_spaces"
            and rule.matches(pt)
        ):
            return pt.replace("-", " "), rule.rule_name

    # 2. Regex `preserve` rules — explicit "this hyphen is correct" marker.
    for rule in rules:
        if rule.action == "preserve" and rule.matches(pt):
            return pt, f"preserve:{rule.rule_name}"

    # 3. Default: keep hyphen (conservative).
    return pt, None


def normalize_pt(
    pt: str,
    spelling_map: dict[str, SpellingRule],
    hyphen_rules: list[HyphenRule],
) -> tuple[str, list[str]]:
    """Apply spelling then hyphen rules. Return (normalized_pt, applied_actions).

    `applied_actions` is a list with up to two strings:
      - "spelling" if any spelling rule fired
      - "hyphen" if any hyphen `dehyphenate_to_spaces` rule fired
        (preserve actions are not reported because they're no-ops on output)
    """
    actions: list[str] = []

    # 1. Spelling.
    pt_after_spelling, spelling_rule = apply_spelling_rule(pt, spelling_map)
    if spelling_rule is not None:
        actions.append("spelling")

    # 2. Hyphen.
    pt_final, hyphen_rule = apply_hyphen_rule(pt_after_spelling, hyphen_rules)
    if hyphen_rule and not hyphen_rule.startswith("preserve:"):
        actions.append("hyphen")

    return pt_final, actions
