"""Loudness normalization via ffmpeg loudnorm (EBU R128) — gain only.

Strategy (LOCKED — see docs/plan.md § "Loudness normalization"):

  Target: I = -16 LUFS, TP = -1.5 dB, LRA = 11.

  Per-clip pipeline:
    pcm_bytes  -- ffmpeg loudnorm pass 1 (measure) -->  measurements dict
    pcm_bytes  -- ffmpeg loudnorm pass 2 (linear=true, measured_*) --> mp3 bytes

  The pass-2 invocation runs in `linear=true` mode so the gain applied is a
  single multiplicative factor — no dynamic range compression, no transient
  smearing. This is the "gain only, no distortion" guarantee the plan promises.

  Per-voice baseline shortcut (Stage 7):
    measure 30 random clips per voice during pilot,
    cache median per-clip target_offset,
    apply volume={median_dB}dB to every full-corpus clip of that voice,
    spot-check 10 clips per voice post-encode for ±1 LU compliance.

ffmpeg ≥ 4.2 required (linear=true mode introduced there). Verified at
import time so missing/old ffmpeg fails loudly.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass

DEFAULT_TARGET_I = -16.0  # LUFS integrated
DEFAULT_TARGET_TP = -1.5  # dB true peak
DEFAULT_TARGET_LRA = 11.0  # LU loudness range
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_PCM_FORMAT = "s16le"
DEFAULT_PCM_CHANNELS = 1
MP3_BITRATE_KBPS = 192
MP3_SAMPLE_RATE = 44100

VERIFY_TOLERANCE_LU = 1.0  # acceptable deviation from target after encode


# --- ffmpeg discovery ----------------------------------------------------- #


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg binary not found on PATH — install ffmpeg ≥ 4.2 "
            "(brew install ffmpeg on macOS)."
        )
    return path


def _find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError("ffprobe not found on PATH (ships with ffmpeg).")
    return path


FFMPEG = _find_ffmpeg()
FFPROBE = _find_ffprobe()


def _check_ffmpeg_version() -> None:
    """Verify ffmpeg ≥ 4.2 (linear=true mode requirement)."""
    out = subprocess.run(
        [FFMPEG, "-version"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    m = re.search(r"ffmpeg version (\d+)\.(\d+)", out)
    if not m:
        return  # custom build; trust it
    major, minor = int(m.group(1)), int(m.group(2))
    if (major, minor) < (4, 2):
        raise RuntimeError(
            f"ffmpeg {major}.{minor} too old; need ≥ 4.2 for loudnorm linear=true mode"
        )


_check_ffmpeg_version()


# --- Data classes --------------------------------------------------------- #


@dataclass
class LoudnessMeasurement:
    """Output of ffmpeg loudnorm pass 1."""

    input_i: float           # integrated LUFS
    input_tp: float          # true peak dBFS
    input_lra: float         # loudness range LU
    input_thresh: float      # threshold dBFS
    output_i: float
    output_tp: float
    output_lra: float
    output_thresh: float
    normalization_type: str  # 'dynamic' or 'linear'
    target_offset: float     # the per-clip gain that pass 2 will apply (dB)


@dataclass
class VoiceBaseline:
    voice_id: str
    median_gain_db: float
    measured_n: int
    notes: str = ""


# --- Internal helpers ----------------------------------------------------- #


_PCM_INPUT_FLAGS = (
    "-f", DEFAULT_PCM_FORMAT,
    "-ar", str(DEFAULT_SAMPLE_RATE),
    "-ac", str(DEFAULT_PCM_CHANNELS),
    "-i", "pipe:0",
)


def _run_ffmpeg(args: list[str], stdin_bytes: bytes) -> tuple[bytes, str]:
    """Run ffmpeg with given args, piping stdin_bytes; return (stdout, stderr_text)."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", *args],
        input=stdin_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[:1000]}"
        )
    return proc.stdout, proc.stderr.decode("utf-8", "replace")


def _parse_loudnorm_json(stderr_text: str) -> LoudnessMeasurement:
    """Extract the JSON block ffmpeg loudnorm prints to stderr.

    The block is a single JSON object preceded by a header line. We look for
    the last well-formed JSON object in the stderr stream.
    """
    # ffmpeg prints the JSON near the end; find the last `{ ... }` block.
    matches = list(re.finditer(r"\{[^{}]*\}", stderr_text, flags=re.DOTALL))
    if not matches:
        raise RuntimeError("loudnorm pass 1 produced no JSON in stderr")
    last = matches[-1].group(0)
    data = json.loads(last)
    return LoudnessMeasurement(
        input_i=float(data["input_i"]),
        input_tp=float(data["input_tp"]),
        input_lra=float(data["input_lra"]),
        input_thresh=float(data["input_thresh"]),
        output_i=float(data["output_i"]),
        output_tp=float(data["output_tp"]),
        output_lra=float(data["output_lra"]),
        output_thresh=float(data["output_thresh"]),
        normalization_type=data.get("normalization_type", ""),
        target_offset=float(data.get("target_offset", 0.0)),
    )


# --- Public API ----------------------------------------------------------- #


def measure_pcm(
    pcm_bytes: bytes,
    *,
    target_i: float = DEFAULT_TARGET_I,
    target_tp: float = DEFAULT_TARGET_TP,
    target_lra: float = DEFAULT_TARGET_LRA,
) -> LoudnessMeasurement:
    """Run ffmpeg loudnorm pass 1 on raw PCM bytes; return the measurement struct."""
    af = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
        f":print_format=json"
    )
    args = list(_PCM_INPUT_FLAGS) + ["-af", af, "-f", "null", "-"]
    _, stderr = _run_ffmpeg(args, pcm_bytes)
    # loglevel default writes to stderr; ensure info-level
    return _parse_loudnorm_json(stderr)


def encode_mp3_with_loudnorm(
    pcm_bytes: bytes,
    measurement: LoudnessMeasurement,
    *,
    target_i: float = DEFAULT_TARGET_I,
    target_tp: float = DEFAULT_TARGET_TP,
    target_lra: float = DEFAULT_TARGET_LRA,
    bitrate_kbps: int = MP3_BITRATE_KBPS,
) -> bytes:
    """Apply pass 2 (linear loudnorm with measured values) and encode to MP3."""
    af = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:linear=true"
        f":measured_I={measurement.input_i}"
        f":measured_LRA={measurement.input_lra}"
        f":measured_TP={measurement.input_tp}"
        f":measured_thresh={measurement.input_thresh}"
        f":offset={measurement.target_offset}"
    )
    args = list(_PCM_INPUT_FLAGS) + [
        "-af", af,
        "-c:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        "-ar", str(MP3_SAMPLE_RATE),
        "-ac", "1",
        "-f", "mp3",
        "pipe:1",
    ]
    stdout, _ = _run_ffmpeg(args, pcm_bytes)
    return stdout


def encode_mp3_with_volume_gain(
    pcm_bytes: bytes,
    *,
    gain_db: float,
    bitrate_kbps: int = MP3_BITRATE_KBPS,
) -> bytes:
    """Apply a simple volume gain (no DRC) and encode to MP3.

    Used in Stage 7 with the cached per-voice median gain. Faster than the
    two-pass loudnorm and just as distortion-free at small gain values.
    """
    af = f"volume={gain_db:.3f}dB"
    args = list(_PCM_INPUT_FLAGS) + [
        "-af", af,
        "-c:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        "-ar", str(MP3_SAMPLE_RATE),
        "-ac", "1",
        "-f", "mp3",
        "pipe:1",
    ]
    stdout, _ = _run_ffmpeg(args, pcm_bytes)
    return stdout


def measure_mp3_loudness(
    mp3_bytes: bytes,
    *,
    target_i: float = DEFAULT_TARGET_I,
    target_tp: float = DEFAULT_TARGET_TP,
    target_lra: float = DEFAULT_TARGET_LRA,
) -> LoudnessMeasurement:
    """Run loudnorm pass 1 on an encoded MP3 (post-encode verification)."""
    af = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json"
    )
    args = ["-i", "pipe:0", "-af", af, "-f", "null", "-"]
    _, stderr = _run_ffmpeg(args, mp3_bytes)
    return _parse_loudnorm_json(stderr)


def verify_target(
    mp3_bytes: bytes,
    *,
    target_lufs: float = DEFAULT_TARGET_I,
    tolerance_lu: float = VERIFY_TOLERANCE_LU,
) -> tuple[bool, float, float]:
    """Returns (within_tolerance, measured_i, measured_tp)."""
    m = measure_mp3_loudness(mp3_bytes, target_i=target_lufs)
    measured_i = m.input_i  # input_i of the encoded MP3 = its actual loudness
    within = abs(measured_i - target_lufs) <= tolerance_lu
    return within, measured_i, m.input_tp


def normalize_pcm_to_mp3(
    pcm_bytes: bytes,
    *,
    target_i: float = DEFAULT_TARGET_I,
    target_tp: float = DEFAULT_TARGET_TP,
    target_lra: float = DEFAULT_TARGET_LRA,
    bitrate_kbps: int = MP3_BITRATE_KBPS,
) -> tuple[bytes, LoudnessMeasurement]:
    """Two-pass: measure then normalize+encode. Returns (mp3_bytes, pass-1 measurement).

    DEPRECATED — use normalize_pcm_to_mp3_verified() for closed-loop
    correction. Kept for backward compat with older code paths.
    """
    m = measure_pcm(
        pcm_bytes,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
    )
    mp3 = encode_mp3_with_loudnorm(
        pcm_bytes,
        m,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
        bitrate_kbps=bitrate_kbps,
    )
    return mp3, m


@dataclass
class VerifiedNormalizationResult:
    """Output of the closed-loop normalization pipeline."""

    mp3_bytes: bytes
    pcm_measurement: LoudnessMeasurement      # initial PCM measurement
    final_mp3_lufs: float                      # post-encode measured LUFS
    final_mp3_tp: float                        # post-encode measured TP
    applied_gain_db: float                     # cumulative gain we ended up applying
    correction_iterations: int                 # how many correction encodes ran
    within_tolerance: bool                     # |final - target| <= tolerance_lu
    tp_limited: bool                           # True if TP ceiling stopped us short


def normalize_pcm_to_mp3_verified(
    pcm_bytes: bytes,
    *,
    target_i: float = DEFAULT_TARGET_I,
    target_tp: float = DEFAULT_TARGET_TP,
    target_lra: float = DEFAULT_TARGET_LRA,
    tolerance_lu: float = VERIFY_TOLERANCE_LU,
    max_corrections: int = 2,
    bitrate_kbps: int = MP3_BITRATE_KBPS,
) -> VerifiedNormalizationResult:
    """Closed-loop loudness normalization — gain only, distortion-free.

    Strategy:
      1. Measure source PCM via loudnorm pass 1.
      2. Compute desired gain = target_i - input_i, clamped by the TP
         ceiling (target_tp - input_tp).
      3. Apply that gain via the `volume` filter and encode to MP3.
         (Skipping loudnorm pass 2 — we're explicit about gain, no DRC.)
      4. Measure the encoded MP3.
      5. If outside tolerance AND TP ceiling has headroom, compute
         a correction delta and re-encode the SAME PCM with the cumulative
         gain. Up to `max_corrections` extra iterations (default 2 → up to
         3 total encodes per clip).
      6. Stop early if TP ceiling is reached (no more gain available).

    Why volume-only, not loudnorm linear=true?
      loudnorm linear=true predicts a single multiplicative gain that
      *should* hit the target on PCM, but ffmpeg's MP3 encoder shifts
      integrated loudness by 0.5–2 LU which loudnorm can't account for.
      Using `volume={gain}dB` directly gives us closed-loop control: we
      observe the actual encoded result and correct by the exact delta.
    """
    # Pass 1: measure PCM
    m = measure_pcm(
        pcm_bytes,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
    )
    # Compute initial gain, capped by TP constraint
    desired_gain = target_i - m.input_i
    max_safe_gain = target_tp - m.input_tp
    applied_gain = min(desired_gain, max_safe_gain)
    tp_limited_initially = desired_gain > max_safe_gain

    # First encode
    mp3 = encode_mp3_with_volume_gain(
        pcm_bytes,
        gain_db=applied_gain,
        bitrate_kbps=bitrate_kbps,
    )

    iterations = 0
    tp_limited = tp_limited_initially
    for _ in range(max_corrections):
        m_enc = measure_mp3_loudness(
            mp3,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
        )
        delta = target_i - m_enc.input_i
        if abs(delta) <= tolerance_lu:
            break  # converged
        # Cap by remaining TP headroom on the ENCODED audio
        if delta > 0:
            tp_headroom = target_tp - m_enc.input_tp
            if tp_headroom < 0.1:
                tp_limited = True
                break  # no room to grow without exceeding TP
            delta = min(delta, tp_headroom)
        # If delta is tiny (< 0.05 dB) further iterations won't help
        if abs(delta) < 0.05:
            break
        applied_gain += delta
        mp3 = encode_mp3_with_volume_gain(
            pcm_bytes,
            gain_db=applied_gain,
            bitrate_kbps=bitrate_kbps,
        )
        iterations += 1

    # Final measurement for the manifest / diagnostics
    m_final = measure_mp3_loudness(
        mp3,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
    )
    within = abs(m_final.input_i - target_i) <= tolerance_lu
    return VerifiedNormalizationResult(
        mp3_bytes=mp3,
        pcm_measurement=m,
        final_mp3_lufs=round(m_final.input_i, 3),
        final_mp3_tp=round(m_final.input_tp, 3),
        applied_gain_db=round(applied_gain, 3),
        correction_iterations=iterations,
        within_tolerance=within,
        tp_limited=tp_limited,
    )


# --- Per-voice baseline --------------------------------------------------- #


def voice_baseline_from_offsets(
    voice_id: str,
    target_offsets_db: list[float],
    *,
    notes: str = "",
) -> VoiceBaseline:
    """Compute the median per-clip gain across N pilot measurements for a voice."""
    if not target_offsets_db:
        raise ValueError("voice baseline requires at least one measurement")
    return VoiceBaseline(
        voice_id=voice_id,
        median_gain_db=round(statistics.median(target_offsets_db), 3),
        measured_n=len(target_offsets_db),
        notes=notes,
    )
