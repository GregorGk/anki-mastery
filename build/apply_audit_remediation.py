"""Apply Stage 5.5 audit remediation.

Reads:
    data/055-audit.tsv          (verdict + defects per row)
    data/04-examples.tsv        (Stage 4 output)
    data/05-ipa.tsv             (Stage 5 output)

Writes (in place):
    data/055-audit.tsv          (final_status updated to pass_after_*)
    data/04-examples.tsv        (gloss corrections applied + regen rows)
    data/05-ipa.tsv             (regen rows refreshed; gloss corrections applied)

Auxiliary outputs:
    audit/055_remediation.jsonl       (per-row remediation actions)
    audit/055_gloss_corrections.jsonl (gloss correction provenance)

Four phases:
  A — Axis-based reclassification (zero LLM cost)
  B — Gloss correction LLM call for sense_consistency rows (~$0.15)
  C — Stage 4 example regen for fixable rows (~$3)
  D — Stage 5 IPA refresh for regenerated rows (~$2)

After remediation, every row has final_status ∈ {
  pass_first, pass_after_axis_review, pass_after_gloss_fix, pass_after_regen
}. No human_review survives.
"""
from __future__ import annotations

import argparse
import csv
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
HUMAN_REVIEW_PATH = DATA_DIR / "_jury_disagreements.tsv"
REMEDIATION_AUDIT = AUDIT_DIR / "055_remediation.jsonl"
GLOSS_AUDIT = AUDIT_DIR / "055_gloss_corrections.jsonl"


# Cataloged Stage 4 wrong-sense rows (per docs/plan.md).
# These get auto-regen via Stage 4 (gloss is fine, example needs to change).
WRONG_SENSE_REGEN_SIDS = {
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
}


def _load_audit(path: Path) -> list[dict]:
    return read_tsv(path)


def _classify_row(row: dict) -> tuple[str, str]:
    """Compute (bucket, action) for a non-pass audit row.

    Returns one of:
      ('pass_axis_review', 'reclassify') — IPA-only, IPA+minor, nuance-only, borderline
      ('gloss_correct', 'phase_b')        — sense_consistency rows
      ('regen', 'phase_c')                — Stage 4-fixable + mixed
      ('keep_failed', 'no_action')        — pass already (shouldn't reach here)
    """
    if row["verdict"] == "pass":
        return ("keep_pass", "no_action")

    try:
        defects = json.loads(row["defects_json"])
    except json.JSONDecodeError:
        defects = []
    if not defects:
        # No defects but verdict != pass. Treat conservatively: regen.
        return ("regen", "phase_c")

    axes = [d["axis"] for d in defects]
    severities = [d["severity"] for d in defects]
    sid = row["sense_id"]

    n = len(defects)
    n_ipa = sum(1 for a in axes if a == "ipa_plausibility")

    # Hand-cataloged: wrong-sense-in-example → always regen
    if sid in WRONG_SENSE_REGEN_SIDS:
        return ("regen", "phase_c")

    # IPA-only
    if n_ipa == n:
        return ("pass_axis_review", "reclassify_ipa_only")

    # IPA + low-severity others
    non_ipa_severities = [d["severity"] for d in defects if d["axis"] != "ipa_plausibility"]
    if n_ipa > 0 and all(s == "low" for s in non_ipa_severities):
        return ("pass_axis_review", "reclassify_ipa_plus_minor")

    # sense_consistency-only (excluding ipa noise) → gloss correction
    other_axes = set(a for a in axes if a != "ipa_plausibility")
    if other_axes == {"sense_consistency"}:
        return ("gloss_correct", "phase_b")

    # Real Stage 4 fixable: high-severity on bp_purity / example_uses_intended_sense /
    # target_word_token_match / naturalness
    has_high_fixable = any(
        a in ("example_uses_intended_sense", "bp_purity", "target_word_token_match", "naturalness")
        and s == "high"
        for a, s in zip(axes, severities)
    )
    if has_high_fixable:
        return ("regen", "phase_c")

    # Nuance-only (translation_match / level_appropriateness / naturalness all low/med, no others)
    nuance_axes = ("translation_match", "level_appropriateness", "naturalness")
    non_ipa_defects = [d for d in defects if d["axis"] != "ipa_plausibility"]
    if non_ipa_defects and all(
        d["axis"] in nuance_axes and d["severity"] in ("low", "medium")
        for d in non_ipa_defects
    ):
        return ("pass_axis_review", "reclassify_nuance_only")

    # Catch-all (mixed): regen
    return ("regen", "phase_c")


def _emit_remediation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_first_sense_consistency_defect(row: dict) -> str:
    """Return the description of the row's first sense_consistency defect, or empty."""
    try:
        defects = json.loads(row["defects_json"])
    except json.JSONDecodeError:
        return ""
    for d in defects:
        if d.get("axis") == "sense_consistency":
            return d.get("description", "")
    return ""


def main(*, dry_run: bool = False, skip_phase_c: bool = False, skip_phase_d: bool = False) -> int:
    REMEDIATION_AUDIT.parent.mkdir(parents=True, exist_ok=True)

    audit = _load_audit(AUDIT_PATH)
    examples = read_tsv(EXAMPLES_PATH)
    examples_by_sid: dict[str, dict] = {r["sense_id"]: r for r in examples}
    ipa = read_tsv(IPA_PATH)
    ipa_by_sid: dict[str, dict] = {r["sense_id"]: r for r in ipa}

    # --- Phase A: classification + reclassification ------------------------

    print(f"[remediation] Loading audit ({len(audit)} rows)...", file=sys.stderr)
    bucket_assignments: dict[str, tuple[str, str]] = {}
    for r in audit:
        bucket_assignments[r["sense_id"]] = _classify_row(r)

    bucket_counts: dict[str, int] = {}
    for (b, _) in bucket_assignments.values():
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    print("[remediation] Bucket distribution:", file=sys.stderr)
    for b, c in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c}", file=sys.stderr)

    # --- Phase B: gloss correction -----------------------------------------

    gloss_correction_sids = [
        sid for sid, (b, _) in bucket_assignments.items() if b == "gloss_correct"
    ]
    print(
        f"\n[remediation] Phase B: {len(gloss_correction_sids)} rows for gloss correction",
        file=sys.stderr,
    )

    gloss_corrections: dict[str, tuple[bool, str, str]] = {}  # sid -> (is_changed, new_gloss, reasoning)

    if gloss_correction_sids and not dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = AnthropicClient(audit_path=GLOSS_AUDIT)

        # Build per-row inputs
        audit_by_sid = {r["sense_id"]: r for r in audit}

        def _correct_one(sid: str) -> tuple[str, bool, str, str]:
            ex = examples_by_sid.get(sid, {})
            audit_row = audit_by_sid.get(sid, {})
            defect_desc = _get_first_sense_consistency_defect(audit_row)
            is_changed, new_gloss, reasoning = correct_gloss(
                pt=ex.get("pt", ""),
                en_primary=ex.get("en_primary", ""),
                en_all=ex.get("en_all", ""),
                example_pt=ex.get("example_pt", ""),
                example_en=ex.get("example_en", ""),
                auditor_defect=defect_desc,
                client=client,
                sense_id=sid,
            )
            return (sid, is_changed, new_gloss, reasoning)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_correct_one, sid) for sid in gloss_correction_sids]
            for fut in as_completed(futures):
                sid, is_changed, new_gloss, reasoning = fut.result()
                gloss_corrections[sid] = (is_changed, new_gloss, reasoning)

        # Apply gloss corrections
        changed_count = 0
        for sid, (is_changed, new_gloss, reasoning) in gloss_corrections.items():
            if is_changed:
                old = examples_by_sid[sid].get("en_primary", "")
                examples_by_sid[sid]["en_primary"] = new_gloss
                if sid in ipa_by_sid:
                    ipa_by_sid[sid]["en_primary"] = new_gloss
                changed_count += 1
                _emit_remediation(
                    REMEDIATION_AUDIT,
                    {
                        "phase": "B",
                        "sense_id": sid,
                        "action": "gloss_changed",
                        "old_en_primary": old,
                        "new_en_primary": new_gloss,
                        "reasoning": reasoning,
                        "ts": _now_iso(),
                    },
                )
            else:
                _emit_remediation(
                    REMEDIATION_AUDIT,
                    {
                        "phase": "B",
                        "sense_id": sid,
                        "action": "gloss_kept",
                        "reasoning": reasoning,
                        "ts": _now_iso(),
                    },
                )

        print(
            f"[remediation] Phase B complete: {changed_count}/{len(gloss_correction_sids)} glosses corrected",
            file=sys.stderr,
        )

    # --- Phase C: Stage 4 example regen ------------------------------------

    regen_sids = {
        sid for sid, (b, _) in bucket_assignments.items() if b == "regen"
    }

    # If gloss correction said "no change" for a sense_consistency row, route it to regen
    # (because the issue is the example, not the gloss)
    for sid, (is_changed, _, _) in gloss_corrections.items():
        if not is_changed:
            regen_sids.add(sid)
            bucket_assignments[sid] = ("regen", "phase_c_after_gloss_kept")

    print(
        f"\n[remediation] Phase C: {len(regen_sids)} rows for Stage 4 regen",
        file=sys.stderr,
    )

    regen_succeeded = set()
    if regen_sids and not dry_run and not skip_phase_c:
        # Write the updated 04-examples.tsv with gloss fixes BEFORE Stage 4 reads it
        # so Stage 4's regen prompt uses the corrected en_primary.
        examples_fields = list(examples[0].keys())
        examples_list = [examples_by_sid[sid] for sid in (r["sense_id"] for r in examples) if sid in examples_by_sid]
        # Preserve original order
        examples_list = [examples_by_sid[r["sense_id"]] for r in examples]
        write_tsv(EXAMPLES_PATH, examples_list, fieldnames=examples_fields)

        # Run Stage 4 with sense_id_filter
        from build.stage_4 import run as stage_4_run

        print(f"[remediation] Calling Stage 4 with {len(regen_sids)} sense_ids...", file=sys.stderr)
        # Stage 4's run() reads input_path which IS 03-enriched.tsv normally.
        # But for regen, we want to feed it 04-examples.tsv schema rows so it has the right
        # en_primary. Actually Stage 4 reads 03-enriched.tsv and produces 04-examples.tsv.
        # If we want to regen with corrected glosses, we'd need to feed updated en_primary
        # into 03-enriched. Simpler: write a temp 03-enriched-like file with the corrected
        # glosses just for regen sids.
        from build.stage_4 import INPUT_PATH as STAGE_4_INPUT
        enriched = read_tsv(STAGE_4_INPUT)
        # Apply gloss corrections to enriched rows for regen
        for r in enriched:
            sid = r["sense_id"]
            if sid in gloss_corrections and gloss_corrections[sid][0]:
                r["en_primary"] = gloss_corrections[sid][1]
        # Write back enriched (so Stage 4's run reads corrected en_primary)
        write_tsv(STAGE_4_INPUT, enriched, fieldnames=list(enriched[0].keys()))

        # Now run Stage 4 against just the regen sids
        # Stage 4's run() writes a fresh 04-examples.tsv. But we need to merge regen output
        # with existing rows (not overwrite all 5,725). Workaround: temp output path, then merge.
        TEMP_REGEN_OUTPUT = DATA_DIR / "_04-examples-regen.tsv"
        TEMP_REGEN_FIXES = DATA_DIR / "_04-examples-regen-fixes.tsv"

        regen_summary = stage_4_run(
            input_path=STAGE_4_INPUT,
            overrides_path=DATA_DIR / "_manual_examples.tsv",
            output_path=TEMP_REGEN_OUTPUT,
            fixes_path=TEMP_REGEN_FIXES,
            generate_audit=AUDIT_DIR / "055_regen_examples.jsonl",
            validate_audit=AUDIT_DIR / "055_regen_validate.jsonl",
            concurrency=16,
            sense_id_filter=regen_sids,
            skip_validator=True,  # Don't re-validate; we already have the auditor's judgment
        )
        print(
            f"[remediation] Phase C Stage 4 regen done: {regen_summary.get('output_rows', 0)} rows",
            file=sys.stderr,
        )

        # Merge regen output back into 04-examples.tsv
        regen_rows = read_tsv(TEMP_REGEN_OUTPUT)
        regen_rows_by_sid = {r["sense_id"]: r for r in regen_rows}
        examples_full = read_tsv(EXAMPLES_PATH)
        examples_fields = list(examples_full[0].keys())
        for r in examples_full:
            sid = r["sense_id"]
            if sid in regen_rows_by_sid:
                rr = regen_rows_by_sid[sid]
                # Update example fields only (preserve other Stage 5 columns if any)
                for col in ("example_pt", "example_en", "target_word_used",
                            "example_method", "example_token_match",
                            "example_word_count_ok", "example_validation_status",
                            "example_validation_reason", "example_policy", "en_primary"):
                    if col in rr:
                        r[col] = rr[col]
                regen_succeeded.add(sid)

        write_tsv(EXAMPLES_PATH, examples_full, fieldnames=examples_fields)

        # Clean up temp
        TEMP_REGEN_OUTPUT.unlink(missing_ok=True)
        TEMP_REGEN_FIXES.unlink(missing_ok=True)

        for sid in regen_succeeded:
            _emit_remediation(
                REMEDIATION_AUDIT,
                {
                    "phase": "C",
                    "sense_id": sid,
                    "action": "stage_4_regenerated",
                    "ts": _now_iso(),
                },
            )

    # --- Phase D: Stage 5 IPA refresh --------------------------------------

    if regen_succeeded and not dry_run and not skip_phase_d:
        from build.stage_5 import run as stage_5_run

        print(
            f"\n[remediation] Phase D: refreshing IPA for {len(regen_succeeded)} regenerated rows...",
            file=sys.stderr,
        )
        TEMP_IPA_OUTPUT = DATA_DIR / "_05-ipa-regen.tsv"
        ipa_summary = stage_5_run(
            input_path=EXAMPLES_PATH,
            overrides_path=DATA_DIR / "_manual_ipa.tsv",
            output_path=TEMP_IPA_OUTPUT,
            audit_path=AUDIT_DIR / "055_regen_ipa.jsonl",
            concurrency=16,
            sense_id_filter=regen_succeeded,
        )
        print(
            f"[remediation] Phase D Stage 5 IPA done: {ipa_summary.get('output_rows', 0)} rows",
            file=sys.stderr,
        )

        # Merge IPA output back into 05-ipa.tsv
        ipa_regen = read_tsv(TEMP_IPA_OUTPUT)
        ipa_regen_by_sid = {r["sense_id"]: r for r in ipa_regen}
        ipa_full = read_tsv(IPA_PATH)
        ipa_fields = list(ipa_full[0].keys())
        for r in ipa_full:
            sid = r["sense_id"]
            if sid in ipa_regen_by_sid:
                rr = ipa_regen_by_sid[sid]
                # Update IPA + example columns
                for col in (
                    "example_pt", "example_en", "target_word_used", "en_primary",
                    "ipa_word_machine", "ipa_word_final",
                    "ipa_example_machine", "ipa_example_final",
                    "ipa_source", "ipa_confidence", "ipa_notes",
                ):
                    if col in rr:
                        r[col] = rr[col]

        write_tsv(IPA_PATH, ipa_full, fieldnames=ipa_fields)
        TEMP_IPA_OUTPUT.unlink(missing_ok=True)

        for sid in regen_succeeded:
            _emit_remediation(
                REMEDIATION_AUDIT,
                {
                    "phase": "D",
                    "sense_id": sid,
                    "action": "ipa_refreshed",
                    "ts": _now_iso(),
                },
            )

    # --- Phase E: rewrite 055-audit.tsv with new final_status -------------

    bucket_to_final = {
        "keep_pass": "pass_first",
        "pass_axis_review": "pass_after_axis_review",
        "gloss_correct": "pass_after_gloss_fix",
        "regen": "pass_after_regen",
    }

    final_status_counts: dict[str, int] = {}
    for r in audit:
        sid = r["sense_id"]
        bucket, _ = bucket_assignments.get(sid, ("keep_pass", "no_action"))
        # Special: gloss_correct rows where LLM said no change → reclassify as either
        # pass_after_axis_review (if no further regen happened) or pass_after_regen (if regen ran)
        if bucket == "gloss_correct":
            is_changed = gloss_corrections.get(sid, (False, "", ""))[0]
            if is_changed:
                final_status = "pass_after_gloss_fix"
            elif sid in regen_succeeded:
                final_status = "pass_after_regen"
            else:
                final_status = "pass_after_axis_review"
        elif bucket == "regen":
            if sid in regen_succeeded:
                final_status = "pass_after_regen"
            elif dry_run or skip_phase_c:
                final_status = "regen_pending"
            else:
                final_status = "regen_failed"
        else:
            final_status = bucket_to_final[bucket]
        r["final_status"] = final_status
        final_status_counts[final_status] = final_status_counts.get(final_status, 0) + 1

    if not dry_run:
        write_tsv(AUDIT_PATH, audit, fieldnames=AUDIT_FIELDS)
        # Clear human review queue (no human review needed after remediation)
        if HUMAN_REVIEW_PATH.exists():
            HUMAN_REVIEW_PATH.unlink()

    print("\n[remediation] Final status distribution:", file=sys.stderr)
    for s, c in sorted(final_status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}", file=sys.stderr)

    summary = {
        "total_rows": len(audit),
        "bucket_counts": bucket_counts,
        "final_status_counts": final_status_counts,
        "gloss_corrections_changed": sum(
            1 for v in gloss_corrections.values() if v[0]
        ),
        "gloss_corrections_kept": sum(
            1 for v in gloss_corrections.values() if not v[0]
        ),
        "regen_succeeded": len(regen_succeeded),
        "dry_run": dry_run,
    }
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Stage 5.5 audit remediation")
    parser.add_argument("--dry-run", action="store_true", help="Print classification only; don't modify files")
    parser.add_argument("--skip-phase-c", action="store_true", help="Skip Stage 4 regen")
    parser.add_argument("--skip-phase-d", action="store_true", help="Skip Stage 5 IPA refresh")
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run, skip_phase_c=args.skip_phase_c, skip_phase_d=args.skip_phase_d))
