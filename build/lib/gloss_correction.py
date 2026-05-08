"""Gloss correction helper — Stage 5.5 remediation Phase B.

Calls Sonnet 4.5 with the gloss_correct.md prompt for sense_consistency-
flagged rows. Returns (is_changed, corrected_gloss, reasoning).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.llm import TIER_DEFAULT, AnthropicClient  # noqa: E402

GLOSS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_en_primary": {
            "type": "string",
            "description": "The corrected gloss, or the original if no change needed.",
        },
        "is_changed": {
            "type": "boolean",
            "description": "True iff corrected_en_primary differs from input en_primary.",
        },
        "reasoning": {
            "type": "string",
            "description": "≤80 chars: why or why not.",
        },
    },
    "required": ["corrected_en_primary", "is_changed", "reasoning"],
}


def _load_prompt() -> str:
    p = REPO_ROOT / "build" / "prompts" / "gloss_correct.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Correct an English gloss for a Brazilian-Portuguese sense if needed. "
        "Use Tool Use."
    )


def build_user_message(
    *,
    pt: str,
    en_primary: str,
    en_all: str,
    example_pt: str,
    example_en: str,
    auditor_defect: str,
) -> str:
    return (
        f"pt: {pt}\n"
        f"en_primary: {en_primary}\n"
        f"en_all: {en_all}\n"
        f"example_pt: {example_pt}\n"
        f"example_en: {example_en}\n"
        f"auditor_defect: {auditor_defect}"
    )


def correct_gloss(
    *,
    pt: str,
    en_primary: str,
    en_all: str,
    example_pt: str,
    example_en: str,
    auditor_defect: str,
    client: AnthropicClient,
    sense_id: str = "",
) -> tuple[bool, str, str]:
    """Return (is_changed, corrected_en_primary, reasoning).

    On error, returns (False, en_primary, error_msg).
    """
    user_msg = build_user_message(
        pt=pt,
        en_primary=en_primary,
        en_all=en_all,
        example_pt=example_pt,
        example_en=example_en,
        auditor_defect=auditor_defect,
    )
    system_prompt = _load_prompt()
    try:
        decision = client.call_tool(
            system=system_prompt,
            user_message=user_msg,
            tool_name="correct_gloss",
            tool_description=(
                "Correct the English gloss for a BP sense if it's wrong, "
                "or confirm it's already correct. Light-touch corrections only."
            ),
            tool_input_schema=GLOSS_TOOL_SCHEMA,
            stage="5_5_gloss",
            provenance_key=sense_id,
            tier=TIER_DEFAULT,
            max_tokens=256,
        )
    except Exception as exc:
        return (False, en_primary, f"error: {type(exc).__name__}")

    is_changed = bool(decision.get("is_changed", False))
    corrected = (decision.get("corrected_en_primary") or en_primary).strip()
    reasoning = (decision.get("reasoning") or "").strip()[:200]

    # Defensive: if LLM said is_changed but didn't change the text, treat as not changed
    if is_changed and corrected == en_primary:
        is_changed = False
    # Conversely if it changed the text but said no change, trust the text
    if not is_changed and corrected != en_primary:
        is_changed = True

    return (is_changed, corrected, reasoning)
