"""Per-clip audio logger — writes JSONL + transcript.log atomically.

Two artifacts per Stage 6/7 run, lock-shared so a single critical section
records the same event in both files:

- `audit/06_audio.jsonl` — machine-readable lifecycle (started / asr_completed
  / regenerated / errored).
- `audit/06_audio_transcript.log` — human-readable, fixed-width, tail-friendly.

The transcript line is written ONLY on `asr_completed` events (one line per
ASR decision per attempt) so a `tail -f` watcher sees a clean, growing log of
"input -> Whisper transcript, similarity, decision".
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Decision tag widths so the log columns line up.
DECISION_TAG_WIDTH = 6
SENSE_ID_WIDTH = 11   # "0001.00.01"
CLIP_TYPE_WIDTH = 4   # "word" / "ex"
TEXT_COL_WIDTH = 36

# Decision tag set used in the transcript:
#   PASS   — similarity >= threshold; clip accepted
#   REGEN  — similarity < threshold; clip queued for regeneration
#   HUMAN  — failed twice; routed to _audio_human_review.tsv
#   ERR    — TTS or ASR API error
PASS = "PASS"
REGEN = "REGEN"
HUMAN = "HUMAN"
ERR = "ERR"


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def _fmt_voice(voice_id: str, *, width: int = 6) -> str:
    return (voice_id[:width] + "…") if len(voice_id) > width else voice_id.ljust(width + 1)


@dataclass
class AsrOutcome:
    """A single ASR decision for one clip-attempt."""

    sense_id: str
    clip_type: str         # "word" or "ex"
    voice_id: str
    input_text: str
    asr_transcript: str
    similarity: float
    phonetic_distance: float | None
    decision: str          # PASS / REGEN / HUMAN / ERR
    attempt: int
    latency_ms: int
    cost_usd: float = 0.0
    note: str = ""         # extra context (e.g. "twice-failed -> human review")


class AudioLogger:
    """Thread-safe dual-writer: machine JSONL + human transcript.

    Both files are append-only. Single shared lock so one critical section
    appends to both files for one event — no risk of interleaving.
    """

    def __init__(
        self,
        jsonl_path: Path,
        transcript_path: Path,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.transcript_path = transcript_path
        for p in (self.jsonl_path, self.transcript_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- Internal write helpers ------------------------------------------ #

    def _write_both(self, jsonl_payload: dict, transcript_line: str | None) -> None:
        line = json.dumps(jsonl_payload, ensure_ascii=False, default=str)
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(line + "\n")
            if transcript_line is not None:
                with self.transcript_path.open("a", encoding="utf-8") as tf:
                    tf.write(transcript_line + "\n")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Public events ---------------------------------------------------- #

    def started(
        self,
        sense_id: str,
        clip_type: str,
        voice_id: str,
        input_text: str,
        attempt: int = 1,
    ) -> None:
        """Log the start of a clip-generation attempt. JSONL only (no transcript line)."""
        self._write_both(
            {
                "event": "started",
                "sense_id": sense_id,
                "clip_type": clip_type,
                "voice_id": voice_id,
                "input_text": input_text,
                "attempt": attempt,
                "started_at": self._now_iso(),
            },
            transcript_line=None,
        )

    def asr_completed(self, outcome: AsrOutcome) -> str:
        """Log a completed ASR decision; emits both JSONL and transcript line.

        Returns the rendered transcript line (without timestamp prefix) so the
        caller can also feed it to the ProgressTracker's ring buffer.
        """
        rendered = self.render_outcome(outcome)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_line = f"{ts}  {rendered}"
        if outcome.note:
            full_line = f"{full_line}  ({outcome.note})"
        self._write_both(
            {
                "event": "asr_completed",
                "sense_id": outcome.sense_id,
                "clip_type": outcome.clip_type,
                "voice_id": outcome.voice_id,
                "input_text": outcome.input_text,
                "asr_transcript": outcome.asr_transcript,
                "levenshtein_similarity": round(outcome.similarity, 4),
                "phonetic_distance": (
                    round(outcome.phonetic_distance, 4)
                    if outcome.phonetic_distance is not None
                    else None
                ),
                "decision": outcome.decision,
                "attempt": outcome.attempt,
                "latency_ms": outcome.latency_ms,
                "cost_usd": outcome.cost_usd,
                "note": outcome.note,
                "completed_at": self._now_iso(),
            },
            transcript_line=full_line,
        )
        return rendered

    def regenerated(
        self,
        sense_id: str,
        clip_type: str,
        voice_id: str,
        attempt: int,
        reason: str,
    ) -> None:
        """Log that a clip is being regenerated (between attempts)."""
        self._write_both(
            {
                "event": "regenerated",
                "sense_id": sense_id,
                "clip_type": clip_type,
                "voice_id": voice_id,
                "attempt": attempt,
                "reason": reason,
                "ts": self._now_iso(),
            },
            transcript_line=None,
        )

    def errored(
        self,
        sense_id: str,
        clip_type: str,
        voice_id: str,
        attempt: int,
        error_type: str,
        error_msg: str,
        stage: str = "tts",
    ) -> None:
        """Log a TTS or ASR API error.

        `stage` should be 'tts' or 'asr' so we can split the failure surface.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rendered = (
            f"{ERR.ljust(DECISION_TAG_WIDTH)}  "
            f"{sense_id.ljust(SENSE_ID_WIDTH)} "
            f"{clip_type.ljust(CLIP_TYPE_WIDTH)}  "
            f"{stage.upper()} {error_type}: {error_msg[:60]}"
        )
        self._write_both(
            {
                "event": "errored",
                "sense_id": sense_id,
                "clip_type": clip_type,
                "voice_id": voice_id,
                "attempt": attempt,
                "stage": stage,
                "error_type": error_type,
                "error_msg": error_msg[:300],
                "errored_at": self._now_iso(),
            },
            transcript_line=f"{ts}  {rendered}",
        )

    # --- Format ----------------------------------------------------------- #

    @staticmethod
    def render_outcome(outcome: AsrOutcome) -> str:
        """Render one ASR decision to a fixed-width string for the transcript log.

        Format:
          PASS    0042.00.01 word    casa                    -> casa                    sim=1.00  voice=MZL...  attempt=1  [1.4s]
        """
        tag = outcome.decision.ljust(DECISION_TAG_WIDTH)
        sid = outcome.sense_id.ljust(SENSE_ID_WIDTH)
        ctype = outcome.clip_type.ljust(CLIP_TYPE_WIDTH)
        in_text = _truncate(outcome.input_text, TEXT_COL_WIDTH)
        out_text = _truncate(outcome.asr_transcript, TEXT_COL_WIDTH)
        sim = f"sim={outcome.similarity:.2f}"
        voice = f"voice={_fmt_voice(outcome.voice_id)}"
        att = f"attempt={outcome.attempt}"
        lat = f"[{outcome.latency_ms / 1000:.1f}s]"
        return (
            f"{tag}  {sid} {ctype}  "
            f"{in_text}  ->  {out_text}  "
            f"{sim}  {voice}  {att}  {lat}"
        )
