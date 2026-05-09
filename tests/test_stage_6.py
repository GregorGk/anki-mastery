"""Stage 6 lib unit tests.

Covers what we can test without hitting external APIs:
- audio_manifest: schema invariants, version bump, init helpers.
- audio_logger: render formatting + atomic dual-write.
- progress: tracker thread-safety, ring buffer, decision counters.
- elevenlabs_client: stable_seed determinism, retry classification.
- asr: normalize_text, text_similarity (no network).
- loudness: smoke test on a synthetic sine (real ffmpeg).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# audio_manifest
# --------------------------------------------------------------------------- #


def test_text_hash_is_stable():
    from build.lib.audio_manifest import text_hash

    a = text_hash("casa")
    b = text_hash("casa")
    c = text_hash("caso")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_object_key_and_url_format():
    from build.lib.audio_manifest import object_key_for, url_for

    assert object_key_for("0042.00.01", "word", 1) == "audio/0042.00.01-word-v1.mp3"
    assert object_key_for("0042.00.01", "example", 3) == "audio/0042.00.01-ex-v3.mp3"
    assert (
        url_for("https://example.com", "0042.00.01", "word", 1)
        == "https://example.com/audio/0042.00.01-word-v1.mp3"
    )


def test_init_row_and_bump_version():
    from build.lib.audio_manifest import (
        STATUS_PENDING,
        bump_version,
        init_row,
    )

    row = init_row(
        sense_id="0001.00.01",
        clip_type="word",
        voice_gender="female",
        voice_id="VOICE_F",
        text_input="casa",
        public_base="https://x.r2.dev",
        version=1,
    )
    assert row["status"] == STATUS_PENDING
    assert row["object_key"] == "audio/0001.00.01-word-v1.mp3"
    assert row["url"].endswith("/audio/0001.00.01-word-v1.mp3")
    assert row["version"] == "1"
    assert row["text_hash"]  # non-empty

    bump_version(row, public_base="https://x.r2.dev")
    assert row["version"] == "2"
    assert row["object_key"].endswith("-word-v2.mp3")
    assert row["status"] == STATUS_PENDING
    assert row["md5"] == ""


def test_manifest_roundtrip(tmp_path):
    from build.lib.audio_manifest import (
        init_manifest_for_senses,
        read_manifest,
        write_manifest,
    )

    senses = [
        {
            "sense_id": "0001.00.01",
            "voice_id": "V_A",
            "voice_gender_assigned": "female",
            "pt": "casa",
            "example_pt": "A casa é grande.",
        },
        {
            "sense_id": "0002.00.01",
            "voice_id": "V_B",
            "voice_gender_assigned": "male",
            "pt": "bom dia",
            "example_pt": "Bom dia, professor!",
        },
    ]
    rows = init_manifest_for_senses(senses, public_base="https://x.r2.dev")
    assert len(rows) == 4  # 2 senses × 2 clip_types

    path = tmp_path / "_audio_manifest.tsv"
    write_manifest(rows, path)
    rows_back = read_manifest(path)
    assert len(rows_back) == 4
    keys = {(r["sense_id"], r["clip_type"]) for r in rows_back}
    assert keys == {
        ("0001.00.01", "word"),
        ("0001.00.01", "example"),
        ("0002.00.01", "word"),
        ("0002.00.01", "example"),
    }


def test_find_pending_excludes_uploaded_and_failed_permanent():
    from build.lib.audio_manifest import (
        STATUS_FAILED_PERMANENT,
        STATUS_FAILED_TRANSIENT,
        STATUS_PENDING,
        STATUS_UPLOADED,
        find_pending,
    )

    rows = [
        {"sense_id": "1", "clip_type": "word", "status": STATUS_UPLOADED},
        {"sense_id": "2", "clip_type": "word", "status": STATUS_PENDING},
        {"sense_id": "3", "clip_type": "word", "status": STATUS_FAILED_TRANSIENT},
        {"sense_id": "4", "clip_type": "word", "status": STATUS_FAILED_PERMANENT},
    ]
    pending = find_pending(rows)
    assert {r["sense_id"] for r in pending} == {"2", "3"}


# --------------------------------------------------------------------------- #
# audio_logger
# --------------------------------------------------------------------------- #


def test_render_outcome_columns_pass():
    from build.lib.audio_logger import PASS, AsrOutcome, AudioLogger

    o = AsrOutcome(
        sense_id="0042.00.01",
        clip_type="word",
        voice_id="MZLCplaCGYxFwJ9LXmx1",
        input_text="casa",
        asr_transcript="casa",
        similarity=1.0,
        phonetic_distance=0.0,
        decision=PASS,
        attempt=1,
        latency_ms=1400,
        cost_usd=0.0006,
    )
    line = AudioLogger.render_outcome(o)
    assert line.startswith("PASS")
    assert "casa" in line
    assert "sim=1.00" in line
    assert "attempt=1" in line
    assert "[1.4s]" in line


def test_audio_logger_dual_write(tmp_path):
    from build.lib.audio_logger import PASS, REGEN, AsrOutcome, AudioLogger

    jsonl = tmp_path / "audio.jsonl"
    log = tmp_path / "transcript.log"
    al = AudioLogger(jsonl, log)
    al.started("0001.00.01", "word", "V_A", "casa")
    o = AsrOutcome(
        sense_id="0001.00.01",
        clip_type="word",
        voice_id="V_A",
        input_text="casa",
        asr_transcript="casa",
        similarity=1.0,
        phonetic_distance=0.0,
        decision=PASS,
        attempt=1,
        latency_ms=1500,
    )
    al.asr_completed(o)
    o2 = AsrOutcome(
        sense_id="0001.00.01",
        clip_type="word",
        voice_id="V_A",
        input_text="casa",
        asr_transcript="caso",
        similarity=0.75,
        phonetic_distance=0.2,
        decision=REGEN,
        attempt=1,
        latency_ms=1600,
    )
    al.asr_completed(o2)

    # JSONL has 3 lines (started + 2 asr_completed)
    jsonl_lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(jsonl_lines) == 3

    # Transcript has 2 lines (one per asr_completed)
    log_lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(log_lines) == 2
    assert "PASS" in log_lines[0]
    assert "REGEN" in log_lines[1]


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #


def test_progress_tracker_basic_lifecycle(tmp_path):
    from build.lib.progress import ProgressTracker

    pt = ProgressTracker(
        tmp_path / "progress.jsonl",
        total=3,
        decision_axes=("pass", "regen", "human", "err"),
        stage_label="Test",
    )
    pt.started("a")
    pt.completed("a", decision="pass", attempt=1, cost_usd=0.001, rendered="A passed")
    pt.started("b")
    pt.completed("b", decision="regen", attempt=1, rendered="B regen")
    pt.started("c")
    pt.errored("c", error_type="X", error_msg="boom", attempt=1, rendered="C err")
    snap = pt.snapshot()
    assert snap["done"] == 2
    assert snap["errored"] == 1
    assert snap["decisions"]["pass"] == 1
    assert snap["decisions"]["regen"] == 1
    assert snap["in_flight"] == 0
    assert len(snap["last_events"]) == 3
    assert snap["cost_usd"] == 0.001


def test_progress_tracker_thread_safe(tmp_path):
    from build.lib.progress import ProgressTracker

    pt = ProgressTracker(tmp_path / "pp.jsonl", total=200)

    def worker(n: int) -> None:
        for i in range(n):
            key = f"{threading.get_ident()}-{i}"
            pt.started(key)
            time.sleep(0.001)
            pt.completed(key, decision="pass", attempt=1)

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = pt.snapshot()
    assert snap["done"] == 200
    assert snap["errored"] == 0


def test_progress_tracker_ring_buffer_size(tmp_path):
    from build.lib.progress import ProgressTracker

    pt = ProgressTracker(
        tmp_path / "p.jsonl", total=20, ring_buffer_size=3
    )
    for i in range(10):
        pt.started(f"k{i}")
        pt.completed(f"k{i}", decision="pass", attempt=1, rendered=f"line {i}")
    snap = pt.snapshot()
    assert len(snap["last_events"]) == 3
    assert snap["last_events"][-1]["rendered"] == "line 9"


# --------------------------------------------------------------------------- #
# elevenlabs_client
# --------------------------------------------------------------------------- #


def test_stable_seed_is_deterministic():
    from build.lib.elevenlabs_client import stable_seed

    a = stable_seed("0042.00.01", "word", 1)
    b = stable_seed("0042.00.01", "word", 1)
    c = stable_seed("0042.00.01", "word", 2)
    d = stable_seed("0042.00.01", "example", 1)
    assert a == b
    assert a != c
    assert a != d
    assert 0 <= a < 4_294_967_295


def test_elevenlabs_retryable_classification(monkeypatch):
    from elevenlabs.core.api_error import ApiError as ElevenLabsApiError

    monkeypatch.setenv("ELEVENLABS_API_KEY", "dummy_test_key")
    from build.lib.elevenlabs_client import ElevenLabsClient

    c = ElevenLabsClient()
    err_429 = ElevenLabsApiError(status_code=429, body={})
    err_500 = ElevenLabsApiError(status_code=500, body={})
    err_401 = ElevenLabsApiError(status_code=401, body={})
    err_400 = ElevenLabsApiError(status_code=400, body={})
    assert c._is_retryable(err_429) is True
    assert c._is_retryable(err_500) is True
    assert c._is_retryable(err_401) is False
    assert c._is_retryable(err_400) is False


# --------------------------------------------------------------------------- #
# asr
# --------------------------------------------------------------------------- #


def test_normalize_text_strips_diacritics_and_punct():
    from build.lib.asr import normalize_text

    assert normalize_text("Olá, Mundo!") == "ola mundo"
    assert normalize_text("A casa é grande.") == "a casa e grande"
    assert normalize_text("Obrigado!") == "obrigado"


def test_text_similarity_levenshtein():
    from build.lib.asr import text_similarity

    assert text_similarity("casa", "casa") == 1.0
    assert text_similarity("casa", "caso") == 0.75
    # Diacritics + punctuation normalized away
    assert text_similarity("Olá!", "ola") == 1.0
    # obrigado vs obrigada — 1 char diff in 8 chars = 7/8 = 0.875
    assert text_similarity("obrigado", "obrigada") == pytest.approx(0.875, abs=0.001)


def test_phonetic_distance_returns_a_number_or_none():
    from build.lib.asr import phonetic_distance

    # eSpeak should be installed; if not, returns None — test handles both.
    d = phonetic_distance("casa", "casa")
    if d is None:
        pytest.skip("eSpeak not installed")
    assert d == 0.0
    d2 = phonetic_distance("casa", "caso")
    assert d2 is not None and d2 > 0.0


# --------------------------------------------------------------------------- #
# loudness — real ffmpeg, fast (~0.5s)
# --------------------------------------------------------------------------- #


def _synthetic_pcm(*, freq_hz: int = 440, dur_s: float = 1.0, peak_dbfs: float = -23.0) -> bytes:
    sr = 44100
    n = int(sr * dur_s)
    amp = int(min(0.99, 10 ** (peak_dbfs / 20.0)) * 32767)
    out = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * freq_hz * i / sr))
        out += struct.pack("<h", v)
    return bytes(out)


def test_loudness_measure_pcm_then_normalize():
    from build.lib.loudness import (
        measure_pcm,
        normalize_pcm_to_mp3,
        verify_target,
    )

    pcm = _synthetic_pcm(peak_dbfs=-23.0)
    m = measure_pcm(pcm)
    assert m.input_i is not None
    mp3, m2 = normalize_pcm_to_mp3(pcm)
    assert mp3.startswith(b"ID3") or mp3[:2] in (b"\xff\xfb", b"\xff\xfa", b"\xff\xf3")
    within, lufs, tp = verify_target(mp3, target_lufs=-16.0, tolerance_lu=2.0)
    assert within, f"loudness {lufs} not within ±2 LU of -16"
