"""Targeted gloss correction for rows where Stage 4 regen left a sense/gloss mismatch.

These are the 15 cataloged "wrong-sense-in-example" rows (whose BP example was
actually correct — gloss was wrong) plus a few additional Stage 2 errors spotted
during post-regen review.

Reads:
    data/055-audit.tsv
    data/04-examples.tsv
    data/05-ipa.tsv

Writes (in place):
    data/04-examples.tsv     (corrected en_primary)
    data/05-ipa.tsv          (corrected en_primary)
    data/055-audit.tsv       (final_status -> pass_after_gloss_fix for changed rows)
    audit/055_remediation.jsonl  (append events)

Run:
    python3 build/fix_remaining_glosses.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.lib.gloss_correction import correct_gloss  # noqa: E402
from build.lib.llm import AnthropicClient  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402
from build.stage_55 import OUTPUT_FIELDS as AUDIT_FIELDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"

AUDIT_PATH = DATA_DIR / "055-audit.tsv"
EXAMPLES_PATH = DATA_DIR / "04-examples.tsv"
IPA_PATH = DATA_DIR / "05-ipa.tsv"
GLOSS_AUDIT = AUDIT_DIR / "055_gloss_corrections.jsonl"
REMEDIATION_AUDIT = AUDIT_DIR / "055_remediation.jsonl"

# 15 cataloged + roxo — rows where after Stage 4 regen the example illustrates
# the natural BP sense and the gloss should adapt.
TARGET_SIDS = {
    # Cataloged wrong-sense (re-categorized as gloss-correction candidates)
    "1510.00.01",  # japonês
    "2193.00.01",  # argentino
    "1284.00.02",  # espera
    "1880.00.01",  # sentença
    "1410.00.02",  # corda
    "1277.00.01",  # reserva
    "2802.00.03",  # roteiro
    "3911.00.01",  # cova
    "3711.00.02",  # ficha (card)
    "3711.00.03",  # ficha (slip)
    "1684.00.01",  # interpretação
    "0164.00.02",  # ponto
    "2596.00.02",  # limpeza
    "4218.00.01",  # caseiro
    "4799.00.01",  # ingresso
    # Additional spotted post-regen
    "4463.00.01",  # roxo (means purple, was glossed "dark red")
}


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    audit = read_tsv(AUDIT_PATH)
    audit_by_sid = {r["sense_id"]: r for r in audit}
    examples = read_tsv(EXAMPLES_PATH)
    examples_by_sid = {r["sense_id"]: r for r in examples}
    ipa = read_tsv(IPA_PATH)
    ipa_by_sid = {r["sense_id"]: r for r in ipa}

    client = AnthropicClient(audit_path=GLOSS_AUDIT)

    print(f"[fix] Targeting {len(TARGET_SIDS)} rows for re-gloss-correction", file=sys.stderr)

    def _correct_one(sid: str) -> tuple[str, bool, str, str, str]:
        ex = examples_by_sid.get(sid, {})
        audit_row = audit_by_sid.get(sid, {})
        # Use the audit's verdict reason as the auditor_defect signal
        auditor_defect = audit_row.get("reason", "")
        # If we can find a sense_consistency defect in defects_json, use that instead
        try:
            defects = json.loads(audit_row.get("defects_json", "[]"))
            for d in defects:
                if d.get("axis") == "sense_consistency":
                    auditor_defect = d.get("description", auditor_defect)
                    break
        except json.JSONDecodeError:
            pass

        old_gloss = ex.get("en_primary", "")
        is_changed, new_gloss, reasoning = correct_gloss(
            pt=ex.get("pt", ""),
            en_primary=old_gloss,
            en_all=ex.get("en_all", ""),
            example_pt=ex.get("example_pt", ""),
            example_en=ex.get("example_en", ""),
            auditor_defect=auditor_defect,
            client=client,
            sense_id=sid,
        )
        return (sid, is_changed, old_gloss, new_gloss, reasoning)

    results: dict[str, tuple[bool, str, str, str]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_correct_one, sid) for sid in TARGET_SIDS]
        for fut in as_completed(futures):
            sid, is_changed, old, new, reasoning = fut.result()
            results[sid] = (is_changed, old, new, reasoning)

    # Apply changes
    changed = []
    kept = []
    now = datetime.now(timezone.utc).isoformat()
    REMEDIATION_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with REMEDIATION_AUDIT.open("a", encoding="utf-8") as f:
        for sid, (is_changed, old, new, reasoning) in results.items():
            if is_changed:
                examples_by_sid[sid]["en_primary"] = new
                if sid in ipa_by_sid:
                    ipa_by_sid[sid]["en_primary"] = new
                if sid in audit_by_sid:
                    audit_by_sid[sid]["final_status"] = "pass_after_gloss_fix"
                changed.append((sid, old, new, reasoning))
                f.write(json.dumps({
                    "phase": "B2_followup",
                    "sense_id": sid,
                    "action": "gloss_changed",
                    "old_en_primary": old,
                    "new_en_primary": new,
                    "reasoning": reasoning,
                    "ts": now,
                }, ensure_ascii=False) + "\n")
            else:
                kept.append((sid, old, reasoning))
                f.write(json.dumps({
                    "phase": "B2_followup",
                    "sense_id": sid,
                    "action": "gloss_kept",
                    "reasoning": reasoning,
                    "ts": now,
                }, ensure_ascii=False) + "\n")

    # Write back files (preserving order)
    examples_fields = list(examples[0].keys())
    write_tsv(EXAMPLES_PATH, examples, fieldnames=examples_fields)
    ipa_fields = list(ipa[0].keys())
    write_tsv(IPA_PATH, ipa, fieldnames=ipa_fields)
    write_tsv(AUDIT_PATH, audit, fieldnames=AUDIT_FIELDS)

    print(f"\n=== Changed: {len(changed)} ===", file=sys.stderr)
    for sid, old, new, reasoning in changed:
        print(f"  [{sid}]  {old!r}  ->  {new!r}", file=sys.stderr)
        print(f"          why: {reasoning[:90]}", file=sys.stderr)
    print(f"\n=== Kept: {len(kept)} ===", file=sys.stderr)
    for sid, old, reasoning in kept:
        print(f"  [{sid}]  kept {old!r}  ({reasoning[:80]})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
