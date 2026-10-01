"""Stage 19 / Step 3 — IPA v2 normalizer + validators (build/lib/bp_ipa.py).

Golden São Paulo cases for the convention, the Stage-05 artefact fixes, the
espeak stress oracle and the written-accent vowel rule. The espeak-dependent
tests skip when espeak-ng isn't installed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import bp_ipa as B  # noqa: E402

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng missing")


# ── notation conversion (Stage 05 → São Paulo convention) ────────────────────
@pytest.mark.parametrize("stage5,expected", [
    ("vˈeʁdʒi", "ˈveɾdʒi"),          # espeak-style stress, uvular coda r → tap
    ("vˈeɾdʒi", "ˈveɾdʒi"),
    ("kwˈaɾtu", "ˈkwaɾtu"),          # onset cluster kw moves with the stress mark
    ("muʎˈɛʁ", "muˈʎɛɾ"),
    ("eɾədˈejɾu", "eɾˈdejɾu"),       # eSpeak schwa artefact dropped
    ("kˈaʁu", "ˈkahu"),              # intervocalic strong r → h
    ("ˈõʁɐ", "ˈõhɐ"),                # strong r after nasal vowel → h
    ("ʁˈua", "ˈhuɐ"),                # word-initial r → h; final a → ɐ
    ("ɐ̃ŋtʃˈiɡu", "ɐ̃ˈtʃiɡu"),       # ŋ after nasal vowel dropped
    ("mostrˈaʁ", "mosˈtɾaɾ"),        # cluster r → ɾ, onset tɾ legal, coda r → ɾ
    ("izʁaˈɛw", "izhaˈɛw"),          # r after z → h
    ("mˈũjtu", "ˈmũj̃tu"),           # nasal diphthong glide nasalized
    ("pˈɐ̃w", "ˈpɐ̃w̃"),
    ("mˈeʒmu", "ˈmezmu"),            # coda ʒ before voiced → z
    ("ˌaniˈmaw", "aniˈmaw"),         # secondary stress dropped
    ("sˌidˈadʒi", "siˈdadʒi"),
    ("ʒˈɛnju", "ˈʒɛnju"),
    ("tɐ̃ˈbẽj", "tɐ̃ˈbẽj̃"),
    ("aɡɾadεˈsu", "aɡɾadɛˈsu"),      # Greek epsilon → ɛ
    ("asosi̯ˈaɾ", "asosiˈaɾ"),        # non-syllabic diacritic dropped
])
def test_to_sp(stage5, expected):
    got, issues = B.to_sp(stage5)
    assert got == expected
    assert not B.whitelist_issues(got)


def test_to_sp_flags_missing_and_multiple_stress():
    assert "stress:none" in B.to_sp("kwaɾtʊ")[1]
    assert "stress:multiple" in B.to_sp("ˈestˈa")[1]


def test_stress_positions_glide_insensitive():
    assert B.stress_positions("isˈtɔɾjɐ")[:2] == (2, 2)      # história
    assert B.stress_positions("kɾiaˈsɐ̃w̃")[0] == 1           # criação (i, w̃ glides)
    assert B.stress_positions("ˈpɐ̃w̃")[0] == 1


# ── stress oracle (espeak-ng pt-br) ──────────────────────────────────────────
@needs_espeak
@pytest.mark.parametrize("word,ipa,ok", [
    ("bonito", "ˈbonitu", False), ("bonito", "boˈnitu", True),
    ("está", "ˈestɐ", False), ("está", "esˈta", True),
    ("pessoa", "ˈpesoɐ", False), ("pessoa", "peˈsoɐ", True),
    ("criação", "kɾiˈasɐ̃w̃", False), ("criação", "kɾiaˈsɐ̃w̃", True),
    ("história", "isˈtɔɾjɐ", True), ("ciência", "sjˈẽsjɐ", True),
    ("verde", "ˈveɾdʒi", True), ("quarto", "ˈkwaɾtu", True),
])
def test_stress_oracle(word, ipa, ok):
    assert B.stress_matches_oracle(word, ipa) is ok


@needs_espeak
@pytest.mark.parametrize("word,bad,fixed", [
    ("bonito", "bˈonitu", "boˈnitu"),
    ("quarto", "kwaɾtʊ", "ˈkwaɾtu"),
    ("semana", "sˈemɐ̃nɐ", "seˈmɐ̃nɐ"),
])
def test_oracle_stress_target_fixes(word, bad, fixed):
    target = B.oracle_stress_target(word, bad)
    assert target is not None
    assert B.to_sp(bad, stressed_override=target)[0] == fixed


# ── written accent ⇒ stressed vowel quality ──────────────────────────────────
@pytest.mark.parametrize("word,ipa,need", [
    ("gênio", "ˈʒɛnju", "e"), ("cômodo", "ˈkɔmodu", "o"), ("polêmica", "poˈlɛmikɐ", "e"),
    ("café", "kaˈfɛ", None), ("avó", "aˈvɔ", None), ("avô", "aˈvo", None),
    ("também", "tɐ̃ˈbẽj̃", None),      # nasal vowel: no open/closed contrast
    ("verde", "ˈveɾdʒi", None),        # no written accent
])
def test_accent_issue(word, ipa, need):
    assert B.accent_issue(word, ipa) == need


def test_fix_stressed_vowel():
    assert B.fix_stressed_vowel("ˈʒɛnju", "e") == "ˈʒenju"
    assert B.fix_stressed_vowel("ˈkɔmodu", "o") == "ˈkomodu"


# ── whitelist / weak forms / heterophones ────────────────────────────────────
def test_whitelist():
    assert B.whitelist_issues("ˈveɾdʒi") == []
    assert B.whitelist_issues("ˈveʁdʒi")            # ʁ is not in the SP convention
    assert B.whitelist_issues("ˈxatu")


def test_weak_forms_are_valid_ipa():
    for word, ipa in B.WEAK_FORMS.items():
        assert not B.whitelist_issues(ipa), (word, ipa)
    assert B.WEAK_FORMS["o"] == "u" and B.WEAK_FORMS["de"] == "dʒi"
    assert B.WEAK_FORMS["que"] == "ki" and B.WEAK_FORMS["em"] == "ẽj̃"


def test_heterophones_exclude_unambiguous_words():
    for w in ("gosto", "jogo", "olho", "colher", "sede", "governo"):
        assert w in B.HETEROPHONES
    for w in ("sobre", "pode", "medo", "fora"):
        assert w not in B.HETEROPHONES


# ── MFA soft check ───────────────────────────────────────────────────────────
def test_mfa_skeleton_collapses_known_mfa_quirks():
    # MFA writes l before i as ʎ, nh as a nasal glide, strong/coda r as x
    assert B.mfa_consonants("p o ʎ i tʃ i k u".split()) == B.consonant_skeleton("poˈlitʃiku")
    assert B.mfa_consonants("tʃ ĩ j̃ a".split()) == B.consonant_skeleton("ˈtʃĩɲɐ")
    assert B.mfa_consonants("v e x d e".split()) == B.consonant_skeleton("ˈveɾdʒi")


def test_mfa_stressed_vowel_alignment():
    assert B.mfa_stressed_vowels("f o x s ɐ".split(), "ˈfɔɾsɐ") == {"o"}


# ── generated outputs (skip until 19_1 has run) ──────────────────────────────
IPA_V2 = REPO_ROOT / "data" / "_ipa_v2.tsv"


@pytest.mark.skipif(not IPA_V2.exists(), reason="run build/19_1_ipa_v2.py first")
def test_ipa_v2_rows_validate():
    from build.lib.tsv import read_tsv
    rows = read_tsv(IPA_V2)
    assert len(rows) == 5725
    bad = []
    for r in rows:
        for tok in r["ipa_example"].split() + r["ipa_word"].split():
            if B.whitelist_issues(tok):
                bad.append((r["sense_id"], tok))
    assert len(bad) <= 5, bad[:10]       # only flagged LLM-unresolved leftovers may remain
