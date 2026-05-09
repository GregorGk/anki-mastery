"""Audio manifest — source of truth for every audio clip.

`data/_audio_manifest.tsv` carries one row per (sense_id, clip_type) pair.
Stage 6/7 read it at startup, skip rows already `uploaded`, and update it
in-place as clips are generated, normalized, and uploaded to R2.

Schema (LOCKED):

    sense_id        join key
    clip_type       'word' | 'example'
    voice_gender    'male' | 'female' (from Stage 4.5)
    tts_provider    'elevenlabs'
    tts_model       'eleven_multilingual_v2'
    voice_id        ElevenLabs voice ID (resolves in config/voices.tsv)
    text_input      verbatim input text sent to ElevenLabs
    text_hash       sha256 of text_input (for cache invalidation)
    object_key      R2 object key (e.g. 'audio/0001.00.01-word-v1.mp3')
    url             public URL (R2_PUBLIC_BASE + '/' + object_key)
    version         integer; bumped on regeneration; baked into filename
    md5             hex md5 of the encoded MP3 bytes
    asr_transcript  Whisper transcription of the encoded MP3
    asr_similarity  Levenshtein normalized similarity vs text_input
    asr_decision    'pass' | 'regen' | 'human'
    generated_at    ISO 8601 UTC timestamp of last successful upload
    status          'pending' | 'uploading' | 'uploaded' | 'failed_transient' | 'failed_permanent'
    notes           free text (errors, etc.)
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "_audio_manifest.tsv"

CLIP_TYPES = ("word", "example")

# Status enum
STATUS_PENDING = "pending"
STATUS_UPLOADING = "uploading"
STATUS_UPLOADED = "uploaded"
STATUS_FAILED_TRANSIENT = "failed_transient"
STATUS_FAILED_PERMANENT = "failed_permanent"

ALL_STATUSES = (
    STATUS_PENDING,
    STATUS_UPLOADING,
    STATUS_UPLOADED,
    STATUS_FAILED_TRANSIENT,
    STATUS_FAILED_PERMANENT,
)

MANIFEST_FIELDS = [
    "sense_id",
    "clip_type",
    "voice_gender",
    "tts_provider",
    "tts_model",
    "voice_id",
    "text_input",
    "text_hash",
    "object_key",
    "url",
    "version",
    "md5",
    "asr_transcript",
    "asr_similarity",
    "asr_decision",
    # Loudness diagnostics (post-encode measured) — added 2026-05 with the
    # closed-loop normalization fix.
    "applied_gain_db",
    "final_lufs",
    "final_tp",
    "loudness_within_tolerance",
    "tp_limited",
    "generated_at",
    "status",
    "notes",
]


@dataclass
class ManifestKey:
    sense_id: str
    clip_type: str

    def __post_init__(self) -> None:
        if self.clip_type not in CLIP_TYPES:
            raise ValueError(f"clip_type must be one of {CLIP_TYPES}, got {self.clip_type!r}")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def object_key_for(sense_id: str, clip_type: str, version: int) -> str:
    """Stable filename pattern. Version baked into the filename — Anki strips
    URL query strings, so `?v=N` would silently drop on mobile."""
    short = "word" if clip_type == "word" else "ex"
    return f"audio/{sense_id}-{short}-v{version}.mp3"


def url_for(public_base: str, sense_id: str, clip_type: str, version: int) -> str:
    base = public_base.rstrip("/")
    return f"{base}/{object_key_for(sense_id, clip_type, version)}"


def _normalize_row(row: dict) -> dict:
    """Ensure every manifest field is present (default empty string)."""
    out = {f: row.get(f, "") for f in MANIFEST_FIELDS}
    return out


def init_row(
    *,
    sense_id: str,
    clip_type: str,
    voice_gender: str,
    voice_id: str,
    text_input: str,
    public_base: str,
    tts_provider: str = "elevenlabs",
    tts_model: str = "eleven_multilingual_v2",
    version: int = 1,
) -> dict:
    """Build a fresh manifest row in `pending` status."""
    return _normalize_row(
        {
            "sense_id": sense_id,
            "clip_type": clip_type,
            "voice_gender": voice_gender,
            "tts_provider": tts_provider,
            "tts_model": tts_model,
            "voice_id": voice_id,
            "text_input": text_input,
            "text_hash": text_hash(text_input),
            "object_key": object_key_for(sense_id, clip_type, version),
            "url": url_for(public_base, sense_id, clip_type, version),
            "version": str(version),
            "md5": "",
            "asr_transcript": "",
            "asr_similarity": "",
            "asr_decision": "",
            "applied_gain_db": "",
            "final_lufs": "",
            "final_tp": "",
            "loudness_within_tolerance": "",
            "tp_limited": "",
            "generated_at": "",
            "status": STATUS_PENDING,
            "notes": "",
        }
    )


def read_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [_normalize_row(r) for r in read_tsv(path)]


def write_manifest(rows: list[dict], path: Path = DEFAULT_MANIFEST_PATH) -> None:
    normalized = [_normalize_row(r) for r in rows]
    write_tsv(path, normalized, fieldnames=MANIFEST_FIELDS)


def index_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """{(sense_id, clip_type): row} — useful for in-place mutation."""
    return {(r["sense_id"], r["clip_type"]): r for r in rows}


def update_row(
    rows_by_key: dict[tuple[str, str], dict],
    sense_id: str,
    clip_type: str,
    **fields,
) -> dict:
    """Mutate the row in place; raises KeyError if absent."""
    row = rows_by_key[(sense_id, clip_type)]
    for k, v in fields.items():
        if k not in MANIFEST_FIELDS:
            raise ValueError(f"unknown manifest field: {k!r}")
        row[k] = "" if v is None else str(v)
    return row


def bump_version(row: dict, public_base: str) -> dict:
    """Increment `version`, recompute object_key + url, reset status to pending."""
    cur = int(row.get("version", "1") or "1")
    new_v = cur + 1
    row["version"] = str(new_v)
    row["object_key"] = object_key_for(row["sense_id"], row["clip_type"], new_v)
    row["url"] = url_for(public_base, row["sense_id"], row["clip_type"], new_v)
    row["md5"] = ""
    row["asr_transcript"] = ""
    row["asr_similarity"] = ""
    row["asr_decision"] = ""
    row["generated_at"] = ""
    row["status"] = STATUS_PENDING
    return row


def init_manifest_for_senses(
    senses_with_voice: list[dict],
    *,
    public_base: str,
    tts_provider: str = "elevenlabs",
    tts_model: str = "eleven_multilingual_v2",
) -> list[dict]:
    """Build initial manifest from senses joined to Stage 4.5 voice assignments.

    Each input dict must have:
        sense_id, voice_gender_assigned, voice_id,
        pt (-> word clip text), example_pt (-> example clip text)
    """
    out: list[dict] = []
    for s in senses_with_voice:
        sid = s["sense_id"]
        voice_id = s["voice_id"]
        voice_gender = s.get("voice_gender_assigned") or s.get("voice_gender") or ""
        out.append(
            init_row(
                sense_id=sid,
                clip_type="word",
                voice_gender=voice_gender,
                voice_id=voice_id,
                text_input=s["pt"],
                public_base=public_base,
                tts_provider=tts_provider,
                tts_model=tts_model,
            )
        )
        out.append(
            init_row(
                sense_id=sid,
                clip_type="example",
                voice_gender=voice_gender,
                voice_id=voice_id,
                text_input=s["example_pt"],
                public_base=public_base,
                tts_provider=tts_provider,
                tts_model=tts_model,
            )
        )
    return out


@dataclass
class ManifestSummary:
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)

    def fmt(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.by_status.items())]
        return f"manifest: total={self.total}  " + "  ".join(parts)


def summarize(rows: list[dict]) -> ManifestSummary:
    s = ManifestSummary(total=len(rows))
    for r in rows:
        st = r.get("status") or STATUS_PENDING
        s.by_status[st] = s.by_status.get(st, 0) + 1
    return s


def find_pending(rows: list[dict]) -> list[dict]:
    """Rows that still need work (anything except `uploaded` or `failed_permanent`)."""
    return [
        r
        for r in rows
        if r.get("status") not in (STATUS_UPLOADED, STATUS_FAILED_PERMANENT)
    ]
