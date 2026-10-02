"""Stage 19 — EP→BP spelling fix reaches every derived 06-final column.

derive_final used to build `pt_display_safe` from the pre-fix `pt_display`
and only then ran the spelling map over SPELLING_FIELDS, so the card target
word kept génio / ingénuo / polémica / cómodo. The map now runs on the 03 + 05
inputs before anything derives from them, and a residue check refuses to
write 06-final if a derived column ever bypasses it.

No network, no spend. main() runs only with every output path redirected
under tmp_path (or /dev/null); data/ is read, never written.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build import derive_final as DF  # noqa: E402
from build.lib import stage19_data as D  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

MAP = {"génio": "gênio", "ingénuo": "ingênuo", "polémica": "polêmica",
       "cómodo": "cômodo", "incómodo": "incômodo"}
AUDIT_JSONL = ("USAGE_HINT_AUDIT", "FAMILY_ROOT_AUDIT", "TOPIC_TAG_AUDIT",
               "RISK_REGISTER_AUDIT")

# The five senses the Stage-19 map entries were added for, as 03/05 carry them.
# (sense_id, source_pt / pt, pt_display, pos, example_pt, target_word_used)
EP_SENSES = [
    ("3359.00.01", "génio", "o génio", "noun",
     "Ela tem um génio difícil e fica brava fácil.", "génio"),
    ("3755.00.01", "ingénuo", "ingénuo", "adj",
     "Ele é muito ingênuo e acredita em tudo.", "ingênuo"),
    ("3887.00.01", "polémica", "a polémica", "noun",
     "A polémica sobre o novo imposto dividiu o país.", "polémica"),
    ("4052.00.01", "cómodo", "cómodo", "adj",
     "Este sofá é muito cómodo para assistir TV.", "cómodo"),
    ("4052.00.02", "cómodo", "o cómodo", "noun",
     "A casa tem cinco cômodos, incluindo a cozinha.", "cômodos"),
]
EXPECTED_SAFE = {
    "3359.00.01": "o gênio",
    "3755.00.01": "(adj) ingênuo",
    "3887.00.01": "a polêmica",
    "4052.00.01": "(adj) cômodo",
    "4052.00.02": "o cômodo",
}


def _enriched(sid, pt, display, pos, source_pt=None):
    src = source_pt or pt
    return {"sense_id": sid, "rank": sid[:4].lstrip("0"), "source_pt": src, "pt": pt,
            "pt_display": display, "pt_type": "word", "gender": "", "pos": pos,
            "sense_index": sid[-1], "en_primary": "x", "en_all": "x", "annotation": "",
            "normalization_action": "", "bp_status": "standard", "family_root": "",
            "tags": f"#{pos}", "source_line": f"{src} = x", "source_line_number": "1"}


def _ipa(sid, pt, display, pos, example, twu):
    return {"sense_id": sid, "pt": pt, "pt_display": display, "pos": pos,
            "example_pt": example, "example_en": "x", "target_word_used": twu,
            "ipa_word_final": "w", "ipa_example_final": "e"}


# ── helpers ──────────────────────────────────────────────────────────────────
def test_apply_spelling_to_inputs_fixes_03_and_05_only_text_fields():
    e = {sid: _enriched(sid, pt, disp, pos) for sid, pt, disp, pos, _, _ in EP_SENSES}
    ip = {sid: _ipa(sid, pt, disp, pos, ex, twu) for sid, pt, disp, pos, ex, twu in EP_SENSES}
    e["0001.00.01"] = _enriched("0001.00.01", "de", "de", "prep")
    fixed = DF._apply_spelling_to_inputs(e, ip, MAP)

    assert set(fixed) == set(EXPECTED_SAFE)                   # "de" untouched
    assert fixed["3359.00.01"] == 6      # 03 pt+display, 05 pt+display+example+twu
    assert fixed["3755.00.01"] == 4      # example / twu were already BP
    assert e["3359.00.01"]["pt_display"] == "o gênio"
    assert ip["3359.00.01"]["pt"] == "gênio"
    assert ip["3359.00.01"]["example_pt"] == "Ela tem um gênio difícil e fica brava fácil."
    assert ip["4052.00.01"]["target_word_used"] == "cômodo"
    assert ip["4052.00.02"]["target_word_used"] == "cômodos"
    # Provenance keeps the source's EP form.
    assert e["4052.00.02"]["source_pt"] == "cómodo"
    assert e["4052.00.02"]["source_line"] == "cómodo = x"


def test_bp_family_root_row_copies_and_leaves_sidecar_raw():
    sc = {"sense_id": "2991.00.01", "family_root": "polémica", "confidence": "high"}
    out = DF._bp_family_root_row(sc, MAP)
    assert out["family_root"] == "polêmica" and out["confidence"] == "high"
    assert sc["family_root"] == "polémica"                    # audit sees the raw value
    assert DF._bp_family_root_row(None, MAP) is None
    empty = {"sense_id": "x", "family_root": ""}
    assert DF._bp_family_root_row(empty, MAP) is empty


def test_ep_spelling_residue_covers_derived_columns_not_provenance():
    rows = [
        {"sense_id": "3359.00.01", "pt": "gênio", "pt_display": "o gênio",
         "pt_display_safe": "o génio", "source_pt": "génio", "source_line": "génio = x",
         "risk_note": 'BP spelling is "cômodo", not "cómodo".'},
        {"sense_id": "4052.00.02", "pt": "cômodo", "pt_display": "o cômodo",
         "pt_display_safe": "o cômodo", "source_pt": "cómodo"},
    ]
    assert DF._ep_spelling_residue(rows, MAP) == [
        ("3359.00.01", "pt_display_safe", "o génio")]
    assert "pt_display_safe" in DF.EP_FREE_FIELDS
    assert not {"source_pt", "source_line", "risk_note"} & set(DF.EP_FREE_FIELDS)


# ── main() on synthetic inputs, every repo path under tmp_path ───────────────
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """derive_final with every module-level repo path redirected under tmp_path."""
    for name, val in list(vars(DF).items()):
        if isinstance(val, Path) and val != DF.REPO_ROOT and val.is_relative_to(DF.REPO_ROOT):
            monkeypatch.setattr(DF, name, tmp_path / val.relative_to(DF.REPO_ROOT))
    for name, val in vars(DF).items():
        if isinstance(val, Path) and val != DF.REPO_ROOT:
            assert val.is_relative_to(tmp_path), name
    monkeypatch.setattr(sys, "argv", ["derive_final.py"])

    enriched = [_enriched(sid, pt, disp, pos) for sid, pt, disp, pos, _, _ in EP_SENSES]
    ipa = [_ipa(sid, pt, disp, pos, ex, twu) for sid, pt, disp, pos, ex, twu in EP_SENSES]
    # polêmico was already BP upstream; its LLM family_root names the EP noun.
    enriched.append(_enriched("2991.00.01", "polêmico", "polêmico", "adj",
                              source_pt="polémico"))
    ipa.append(_ipa("2991.00.01", "polêmico", "polêmico", "adj", "Um tema polêmico.",
                    "polêmico"))
    cols = list(enriched[0])
    write_tsv(DF.ENRICHED_PATH, enriched, fieldnames=cols)
    write_tsv(DF.IPA_PATH, ipa, fieldnames=list(ipa[0]))
    write_tsv(DF.SPELLING_MAP_PATH,
              [{"source_form": k, "bp_form": v, "rule_type": "accent"} for k, v in MAP.items()],
              fieldnames=["source_form", "bp_form", "rule_type"])
    fr_cols = ["sense_id", "family_root", "confidence", "source"]
    write_tsv(DF.FAMILY_ROOTS_PATH, [
        # EP root → must validate as its BP form and ship BP.
        {"sense_id": "2991.00.01", "family_root": "polémica", "confidence": "high",
         "source": "llm"},
        # BP root of an EP pt → only in the pt set once 03/05 are fixed.
        {"sense_id": "4052.00.01", "family_root": "cômodo", "confidence": "high",
         "source": "postprocess_self_root"},
        {"sense_id": "3887.00.01", "family_root": "polêmico", "confidence": "high",
         "source": "llm"},
    ], fieldnames=fr_cols)
    write_tsv(DF.RISK_REGISTER_PATH, [
        {"sense_id": "4052.00.01", "bp_validity": "ep_leaning", "register": "neutral",
         "risk_flags": "ep_misleading", "source": "llm",
         "risk_note": 'In Brazil the spelling is "cômodo", not "cómodo".'},
    ], fieldnames=["sense_id", "bp_validity", "register", "risk_flags", "risk_note",
                   "source"])
    return tmp_path


def _final_rows():
    return {r["sense_id"]: r for r in read_tsv(DF.FINAL_PATH)}


def test_main_ships_bp_spelling_in_every_derived_field(sandbox):
    assert DF.main() == 0
    rows = _final_rows()

    for sid, safe in EXPECTED_SAFE.items():
        r = rows[sid]
        assert r["pt_display_safe"] == safe, sid
        assert not DF._ep_spelling_residue([r], MAP), sid
        # Downstream Stage-19 derivations read 06-final: the word clip's TTS
        # text and the expected-IPA article prefix see one BP spelling.
        assert D.spoken_headword(r["pt_display"]) == r["pt_display"]
        assert r["pt_display"].endswith(r["pt"])

    g = rows["3359.00.01"]
    assert (g["pt"], g["pt_display"], g["target_word_used"]) == ("gênio", "o gênio", "gênio")
    assert g["example_pt"] == "Ela tem um gênio difícil e fica brava fácil."
    assert D._article_prefix_ipa(g["pt_display"], g["pt"]) == D.B.WEAK_FORMS["o"]
    assert rows["3887.00.01"]["example_pt"].startswith("A polêmica ")
    assert rows["4052.00.01"]["example_pt"] == "Este sofá é muito cômodo para assistir TV."
    assert rows["4052.00.02"]["pt_display"] == "o cômodo"

    # Provenance and the risk_note that quotes the EP form stay as written.
    assert rows["4052.00.02"]["source_pt"] == "cómodo"
    assert rows["3359.00.01"]["source_line"] == "génio = x"
    assert '"cómodo"' in rows["4052.00.01"]["risk_note"]

    # family_root validated against the BP pt set.
    assert rows["2991.00.01"]["family_root"] == "polêmica"
    assert rows["4052.00.01"]["family_root"] == "cômodo"     # was dropped pre-fix
    assert rows["3887.00.01"]["family_root"] == "polêmico"

    log = DF.LOG_PATH.read_text(encoding="utf-8")
    assert "ep_spelling_rows_fixed: 5" in log and "ep_spelling_residue: 0" in log
    assert "ep_spelling senses: " + " ".join(sorted(EXPECTED_SAFE)) in log

    # The family_root audit records the raw sidecar root next to the shipped one.
    audit = [json.loads(x) for x in
             DF.FAMILY_ROOT_AUDIT.read_text(encoding="utf-8").splitlines()]
    rec = next(a for a in audit if a["sense_id"] == "2991.00.01")
    assert (rec["sidecar_family_root"], rec["shipped_family_root"]) == ("polémica", "polêmica")


def test_main_refuses_to_write_when_a_derived_column_skips_the_fix(sandbox, monkeypatch,
                                                                   capsys):
    # Simulate a future derived column computed from an unfixed value.
    monkeypatch.setattr(DF, "_pt_display_safe", lambda pt_display, pos: "o génio")
    assert DF.main() == 1
    assert not DF.FINAL_PATH.exists()
    err = capsys.readouterr().err
    assert "EP spelling survived in" in err and "pt_display_safe='o génio'" in err
    assert "ep_spelling_residue: 6" in DF.LOG_PATH.read_text(encoding="utf-8")


# ── main() on the real, read-only inputs ─────────────────────────────────────
def test_main_real_inputs_ship_bp_for_affected_senses(tmp_path, monkeypatch):
    for p in (DF.ENRICHED_PATH, DF.IPA_PATH, DF.SPELLING_MAP_PATH):
        if not p.exists():
            pytest.skip(f"{p.name} not present")

    out = tmp_path / "06-final.tsv"
    monkeypatch.setattr(DF, "FINAL_PATH", out)
    monkeypatch.setattr(DF, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(DF, "LOG_PATH", tmp_path / "audit" / "derive_final.log")
    for name in AUDIT_JSONL:
        monkeypatch.setattr(DF, name, Path(os.devnull))
    captured: dict = {}

    def fake_write(path, rows, *, fieldnames):
        assert Path(path) == out
        captured["rows"] = list(rows)
        return len(captured["rows"])

    monkeypatch.setattr(DF, "write_tsv", fake_write)
    monkeypatch.setattr(sys, "argv", ["derive_final.py"])
    for name in ("FINAL_PATH", "AUDIT_DIR", "LOG_PATH"):    # inputs only from data/
        assert getattr(DF, name).is_relative_to(tmp_path), name

    assert DF.main() == 0
    rows = {r["sense_id"]: r for r in captured["rows"]}
    for sid, safe in EXPECTED_SAFE.items():
        assert rows[sid]["pt_display_safe"] == safe, sid
    assert rows["3359.00.01"]["pt"] == "gênio"
    real_map = DF.load_spelling_map(DF.SPELLING_MAP_PATH)
    assert DF._ep_spelling_residue(captured["rows"], real_map) == []
