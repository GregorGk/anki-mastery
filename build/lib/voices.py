"""Voice pool helpers for Stage 4.5 + Stage 6/7.

Loads `config/voices.tsv` (committed file with user-pinned ElevenLabs voice
IDs) and exposes the round-robin assignment algorithm. Per the plan:

- 4 female + 7 male voices (count enforced for first run; can grow later)
- Within each gender bucket, voices are assigned via seeded shuffle then
  round-robin so every voice gets ⌈N/k⌉ or ⌊N/k⌋ senses (off-by-1 max)
- All seeds are locked: 44 (M/F balance for neutrals), 45 (female pool),
  46 (male pool)
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv  # noqa: E402

VOICES_PATH = REPO_ROOT / "config" / "voices.tsv"

SEED_NEUTRAL_BALANCE = 44
SEED_FEMALE_POOL = 45
SEED_MALE_POOL = 46


@dataclass(frozen=True)
class Voice:
    voice_id: str
    gender: str  # 'male' | 'female'
    pool_index: int
    status: str = "active"  # 'active' | 'retired_bp_only'
    notes: str = ""


def load_voices(path: Path = VOICES_PATH) -> list[Voice]:
    """Load voice pool from config/voices.tsv. Returns sorted by (gender, pool_index).

    Voices whose `status` is not "active" are excluded from the returned pool —
    they're kept in the TSV for provenance (e.g., to preserve the en_voice_id
    of a retired BP voice that still voices en_ex rows in the manifest), but
    they are NOT eligible for round-robin assignment.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Voice pool not found at {path}. Create config/voices.tsv "
            f"with voice_id, gender, pool_index, status, notes columns."
        )
    rows = read_tsv(path)
    voices = []
    for r in rows:
        status = (r.get("status") or "active").strip() or "active"
        if status != "active":
            # Retired / preserved-en-only rows: keep file row but skip pool.
            continue
        voices.append(
            Voice(
                voice_id=r["voice_id"].strip(),
                gender=r["gender"].strip().lower(),
                pool_index=int(r["pool_index"]),
                status=status,
                notes=(r.get("notes") or "").strip(),
            )
        )
    # Stable sort: gender first (female < male alphabetically) then pool_index
    voices.sort(key=lambda v: (v.gender, v.pool_index))
    return voices


def split_by_gender(voices: list[Voice]) -> tuple[list[str], list[str]]:
    """Return (FEMALE_VOICES, MALE_VOICES) — voice_ids ordered by pool_index."""
    female = sorted(
        [v for v in voices if v.gender == "female"], key=lambda v: v.pool_index
    )
    male = sorted(
        [v for v in voices if v.gender == "male"], key=lambda v: v.pool_index
    )
    return ([v.voice_id for v in female], [v.voice_id for v in male])


def resolve_neutrals(
    sense_ids_with_gender: dict[str, str], seed: int = SEED_NEUTRAL_BALANCE
) -> dict[str, str]:
    """Phase 2a: assign every `neutral` sense_id to either `male` or `female`.

    Input: {sense_id -> 'male' | 'female' | 'neutral'}
    Output: {sense_id -> 'male' | 'female'} (no neutrals remain)

    Algorithm: sort all neutral sense_ids, shuffle with `seed`, alternate
    female/male by index. Even indices get female, odd get male.

    Properties:
      - Deterministic (same seed -> same assignment)
      - Off-by-at-most-1 split (⌈N/2⌉ female / ⌊N/2⌋ male)
    """
    out: dict[str, str] = {}
    neutral: list[str] = []
    for sid, g in sense_ids_with_gender.items():
        if g in ("male", "female"):
            out[sid] = g
        elif g == "neutral":
            neutral.append(sid)
        else:
            raise ValueError(f"Unknown gender for {sid!r}: {g!r}")

    neutral.sort()
    rng = random.Random(seed)
    rng.shuffle(neutral)
    for i, sid in enumerate(neutral):
        out[sid] = "female" if i % 2 == 0 else "male"
    return out


def assign_voice_ids(
    voice_gender: dict[str, str],
    voices: list[Voice],
    *,
    female_seed: int = SEED_FEMALE_POOL,
    male_seed: int = SEED_MALE_POOL,
) -> dict[str, str]:
    """Phase 2b: round-robin assign specific voice_id within each gender pool.

    Input: voice_gender = {sense_id -> 'male' | 'female'} (no neutrals)
           voices = full pool from load_voices()

    Output: {sense_id -> voice_id}

    Algorithm:
      - female_sids = sorted IDs whose gender is 'female'
        shuffle(seed=45), then voice_id = FEMALE_VOICES[i % len(FEMALE_VOICES)]
      - same for male with seed=46

    Properties:
      - Deterministic
      - Per-voice usage is balanced: every voice gets ⌈N/k⌉ or ⌊N/k⌋ senses
        (where k = pool size for that gender) — off-by-at-most-1
    """
    female_voices, male_voices = split_by_gender(voices)
    if not female_voices:
        raise ValueError("Voice pool has no female voices")
    if not male_voices:
        raise ValueError("Voice pool has no male voices")

    female_sids = sorted(sid for sid, g in voice_gender.items() if g == "female")
    male_sids = sorted(sid for sid, g in voice_gender.items() if g == "male")

    out: dict[str, str] = {}

    rng_f = random.Random(female_seed)
    rng_f.shuffle(female_sids)
    for i, sid in enumerate(female_sids):
        out[sid] = female_voices[i % len(female_voices)]

    rng_m = random.Random(male_seed)
    rng_m.shuffle(male_sids)
    for i, sid in enumerate(male_sids):
        out[sid] = male_voices[i % len(male_voices)]

    return out


def usage_counts(voice_id_for: dict[str, str]) -> dict[str, int]:
    """Return {voice_id -> count of senses assigned to it}."""
    counts: dict[str, int] = {}
    for vid in voice_id_for.values():
        counts[vid] = counts.get(vid, 0) + 1
    return counts


def verify_balanced_within_pool(
    voice_id_for: dict[str, str], voices: list[Voice]
) -> tuple[bool, str]:
    """Check that within each gender pool, max usage − min usage ≤ 1.

    Returns (is_balanced, message).
    """
    counts = usage_counts(voice_id_for)
    voice_by_id = {v.voice_id: v for v in voices}

    by_gender: dict[str, list[int]] = {"female": [], "male": []}
    for vid, c in counts.items():
        if vid in voice_by_id:
            by_gender[voice_by_id[vid].gender].append(c)

    msgs = []
    is_balanced = True
    for gender, cs in by_gender.items():
        if not cs:
            continue
        spread = max(cs) - min(cs)
        msgs.append(f"{gender}: min={min(cs)} max={max(cs)} spread={spread}")
        if spread > 1:
            is_balanced = False
    return is_balanced, "; ".join(msgs)
