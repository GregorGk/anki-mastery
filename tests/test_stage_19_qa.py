"""Stage 19 — per-take QA gate (build/lib/audio_qa.py). No network.

Relaxed-ASR cases come from real v3 transcripts: spacing / article
differences pass, genuine content defects (dropped infinitive -r, wrong
word) fail, manual overrides and spelled-out numbers pass, and leaked tags or
slash words always fail.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import audio_qa as QA  # noqa: E402


def _relaxed(transcript, reference, *, decision="regen", spoken="", manual=False, top=False):
    return QA.relaxed_asr_pass(transcripts=[transcript], reference=reference, clip_type="word",
                               is_top_1000=top, production_decision=decision,
                               spoken_reference=spoken, manual_pass=manual)


@pytest.mark.parametrize("transcript,reference,ok", [
    ("Acorda", "a corda", True),            # spacing only
    ("Antena.", "a antena", True),          # article dropped by ASR
    ("Pensa", "pensar", False),             # dropped infinitive -r is a real defect
    ("A uva", "a honra", False),            # wrong word
    ("Ou fingir", "ofender", False),
    ("Utapit", "o tapete", False),          # EP-like reduction
])
def test_relaxed_asr_cases(transcript, reference, ok):
    assert _relaxed(transcript, reference)[0] is ok


def test_production_pass_short_circuits():
    assert _relaxed("anything", "verde", decision="pass") == (True, "production", 0.0) or \
        _relaxed("anything", "verde", decision="pass")[0] is True


def test_manual_override_and_spoken_reference():
    assert _relaxed("Por quê?", "porque", manual=True)[0] is True
    ok, mode, _ = QA.relaxed_asr_pass(
        transcripts=["Custa dois mil reais."], reference="Custa 2.000 reais.",
        clip_type="example", is_top_1000=False, production_decision="regen",
        spoken_reference="Custa dois mil reais.")
    assert ok and mode in ("spoken", "relaxed")


@pytest.mark.parametrize("transcript", ["sotaque paulistano verde", "barra verde barra",
                                        "clear careful pronunciation verde"])
def test_leaks_always_fail(transcript):
    assert _relaxed(transcript, "verde", decision="pass")[0] is False


def _pcm(seconds: float, *, amp: int = 8000, lead_silence: float = 0.0,
         trail_silence: float = 0.0, sr: int = 44100) -> bytes:
    out = bytearray()
    out += b"\x00\x00" * int(lead_silence * sr)
    for i in range(int(seconds * sr)):
        out += struct.pack("<h", int(amp * math.sin(2 * math.pi * 220 * i / sr)))
    out += b"\x00\x00" * int(trail_silence * sr)
    return bytes(out)


def test_pcm_sanity_ok_word():
    s = QA.pcm_sanity(_pcm(0.6, lead_silence=0.1, trail_silence=0.1), "word", v3_duration_s=0.8)
    assert s.ok, s.issues


@pytest.mark.parametrize("pcm,clip,v3,issue", [
    (b"", "word", None, "pcm:empty_or_odd"),
    (b"\x00\x00" * 44100, "word", None, "pcm:silent"),
    (_pcm(0.1), "word", None, "dur:word"),
    (_pcm(0.6, lead_silence=1.0), "word", None, "edge:lead"),
    (_pcm(2.5), "word", 0.7, "dur:vs_v3"),
    (_pcm(1.0), "example", 3.0, "dur:vs_v3"),
])
def test_pcm_sanity_failures(pcm, clip, v3, issue):
    s = QA.pcm_sanity(pcm, clip, v3_duration_s=v3)
    assert not s.ok and any(i.startswith(issue) for i in s.issues), s.issues


def test_loudness_gate():
    assert QA.loudness_ok(-16.2, -1.6)[0] is True
    assert QA.loudness_ok(float("-inf"), -2.0)[0] is False
    assert QA.loudness_ok(-16.0, -0.4)[0] is False


def test_gate_pass_requires_every_judge():
    assert QA.gate_pass({"a": "bp_ok", "b": "bp_ok"}) is True
    assert QA.gate_pass({"a": "bp_ok", "b": "non_bp"}) is False
    assert QA.gate_pass({"a": "bp_ok", "b": "error"}) is False
    assert QA.gate_pass({}) is False
