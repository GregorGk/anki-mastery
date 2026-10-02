"""Stage 19 — shared data access for the pilot (19_3) and the full run (19_4).

Builds one `ClipJob` per (sense, word|example) with everything a take needs:
display text, the voice that voices it today (manifest row — keeps the
16.9 swaps), expected IPA for J2 judges (data/_ipa_v2.tsv), the spoken form
for digit sentences, the rank (ASR threshold), manual ASR overrides, and the
v3 clip (path + duration) for relative-duration sanity and baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build.lib import bp_ipa as B
from build.lib.tsv import read_tsv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
ANKI_MEDIA = DATA / "anki_media"


@dataclass
class ClipJob:
    sense_id: str
    clip_type: str                 # "word" | "example"
    display_text: str              # what the clip says (TTS input, ASR reference, judge text)
    voice_id: str
    voice_gender: str
    expected_ipa: str
    spoken_reference: str
    rank: int
    manual_pass: bool
    v3_object_key: str
    v3_model: str
    order: int                     # Stage-17 spaced_topic_order (study order)

    @property
    def key(self) -> tuple[str, str]:
        return (self.sense_id, self.clip_type)

    @property
    def v3_path(self) -> Path:
        return ANKI_MEDIA / self.v3_object_key.split("/", 1)[1]

    @property
    def is_top_1000(self) -> bool:
        return 0 < self.rank <= 1000


def spoken_headword(pt_display: str) -> str:
    """What the word clip SAYS. "o/a presidente" is shown on the card but
    spoken as "o presidente, a presidente" (user decision 2026-10-01: v4 reads
    the slash aloud as "barra")."""
    if pt_display.startswith("o/a "):
        noun = pt_display[4:].strip()
        return f"o {noun}, a {noun}"
    return pt_display


def _article_prefix_ipa(pt_display: str, pt: str) -> str:
    """Weak-form IPA of the article in front of the headword ('a cara' → 'a')."""
    if not pt_display.endswith(pt) or pt_display == pt:
        return ""
    art = pt_display[: -len(pt)].strip().lower()
    return " ".join(B.WEAK_FORMS.get(t, "") for t in art.split()).strip()


def load_jobs() -> dict[tuple[str, str], ClipJob]:
    final = {r["sense_id"]: r for r in read_tsv(DATA / "06-final.tsv")}
    ipa = {r["sense_id"]: r for r in read_tsv(DATA / "_ipa_v2.tsv")}
    manual = {(r["sense_id"], r["clip_type"]) for r in read_tsv(DATA / "_manual_audio.tsv")
              if r.get("decision_override") == "pass"}
    order = {r["sense_id"]: int(r["spaced_topic_order"] or 0)
             for r in read_tsv(DATA / "_ordering.tsv")}
    gender = {r["voice_id"]: r["gender"] for r in read_tsv(CONFIG / "voices.tsv")}
    jobs: dict[tuple[str, str], ClipJob] = {}
    for m in read_tsv(DATA / "_audio_manifest.tsv"):
        ct = m["clip_type"]
        if ct not in ("word", "example") or m["status"] != "uploaded":
            continue
        sid = m["sense_id"]
        f = final[sid]
        v2 = ipa.get(sid, {})
        if ct == "word":
            display = spoken_headword(f["pt_display"])
            w = v2.get("ipa_word") or f["ipa_word"]
            if display != f["pt_display"]:                       # o/a → both forms
                expected = f"u {w} a {w}"
            else:
                art = _article_prefix_ipa(f["pt_display"], f["pt"])
                expected = f"{art} {w}".strip() if art else w
            spoken = ""
        else:
            display = f["example_pt"]
            expected = v2.get("ipa_example") or f["ipa_example"]
            spoken = v2.get("example_spoken", "")
        jobs[(sid, ct)] = ClipJob(
            sense_id=sid, clip_type=ct, display_text=display, voice_id=m["voice_id"],
            voice_gender=gender.get(m["voice_id"], m.get("voice_gender", "")),
            expected_ipa=expected, spoken_reference=spoken,
            rank=int(f.get("rank") or 0), manual_pass=(sid, ct) in manual,
            v3_object_key=m["object_key"], v3_model=m["tts_model"],
            order=order.get(sid, 10**6))
    return jobs


def voice_names() -> dict[str, str]:
    return {r["voice_id"]: r["bp_name"] for r in read_tsv(CONFIG / "voices.tsv")}


def swap_pool(gender: str) -> list[str]:
    """Same-gender BP voices eligible as swap targets (active first, escape
    hatch last)."""
    rows = read_tsv(CONFIG / "voices.tsv")
    active = [r["voice_id"] for r in rows if r["gender"] == gender and r["status"] == "active"]
    hatch = [r["voice_id"] for r in rows if r["gender"] == gender
             and r["status"] == "active_escape_hatch"]
    return active + hatch


def risky_word_sids() -> set[str]:
    """Senses whose WORD clip is risky: P0/P1 risk class, confirmed mispronunciations,
    heterophones, and headwords whose IPA the LLM had to adjudicate. 19_4 gives them
    an alternate variant and the extra-vote tie-break; judges scoped `risky_word` in
    config/stage19_models.tsv (Gemini Pro) run only on these."""
    out = {r["sense_id"] for r in read_tsv(DATA / "_audio_risk_classification.tsv")
           if r["priority"] in ("P0", "P1")}
    out |= {r["sense_id"] for r in read_tsv(DATA / "_audio_mispronunciation_confirmed.tsv")}
    out |= {r["sense_id"] for r in read_tsv(DATA / "06-final.tsv")
            if r["pt"].strip().lower() in B.HETEROPHONES}
    out |= {r["sense_id"] for r in read_tsv(DATA / "_ipa_v2.tsv")
            if r.get("ipa_word_status") in ("llm", "llm_context")}
    return out
