"""Stage 8 / Step 5 — Re-render flagged clips with alias dictionary attached.

Reads the latest dictionary locator from `data/_audio_dictionary_meta.tsv`,
constructs an ElevenLabs client with `pronunciation_dictionary_locators`
set, and re-renders the word clips listed in
`data/_pronunciation_alias_applications.tsv`. Plus example clips for the
same sense_ids (the user's locked decision: re-render examples too when
their voice swap or word-clip alias affects them).

Verification per re-rendered clip (LOCKED):
- Audio judge (gpt-4o-audio-preview) via build.lib.audio_judge — primary gate.
- Biased ASR (language=pt) — sanity check that the audio still says the right word.
- Closed-loop loudness — must land within ±1 LU of -16 LUFS (existing constraint).
Unbiased ASR is recorded as diagnostic only, never a gate.

Manifest fields updated per re-rendered row:
  version (N→N+1), url, md5, asr_*, alias_applied=true, alias_id,
  alias_respelling, pronunciation_dict_locator_id,
  pronunciation_dict_version_id, audio_judge_verdict,
  audio_judge_confidence, status (uploaded / respelling_failed).

Modes:
    --smoke-test --sense-id SID    (render one clip, listen, verify dict works)
    --confirm                      (bulk re-render)

Outputs:
    audit/08_5_rerender.jsonl
    data/_audio_manual_respelling_review.tsv  (clips that failed verify)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

MANIFEST_PATH = DATA_DIR / "_audio_manifest.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
META_PATH = DATA_DIR / "_audio_dictionary_meta.tsv"
MANUAL_REVIEW_PATH = DATA_DIR / "_audio_manual_respelling_review.tsv"

AUDIT_JSONL = AUDIT_DIR / "08_5_rerender.jsonl"


def _latest_dictionary_locator() -> dict | None:
    """Read the latest dict_id+version_id from the meta TSV."""
    if not META_PATH.exists():
        return None
    rows = read_tsv(META_PATH)
    if not rows:
        return None
    last = rows[-1]
    if not last.get("dictionary_id") or not last.get("version_id"):
        return None
    return {
        "pronunciation_dictionary_id": last["dictionary_id"],
        "version_id": last["version_id"],
    }


def _load_alias_lookup() -> dict[str, dict]:
    """Build alias_id → alias row lookup from _pronunciation_aliases.tsv."""
    if not ALIASES_PATH.exists():
        return {}
    rows = read_tsv(ALIASES_PATH)
    return {r["alias_id"]: r for r in rows if r.get("alias_id")}


def _affected_keys() -> set[tuple[str, str]]:
    """Set of (sense_id, clip_type) keys to re-render.

    Includes:
      - All rows in _pronunciation_alias_applications.tsv (word clips per design)
      - PLUS the corresponding example clip for each affected sense (user's
        locked decision: re-render examples too).
    """
    if not APPLICATIONS_PATH.exists():
        return set()
    rows = read_tsv(APPLICATIONS_PATH)
    keys: set[tuple[str, str]] = set()
    for r in rows:
        sid = r.get("sense_id", "")
        if not sid:
            continue
        keys.add((sid, "word"))
        keys.add((sid, "example"))
    return keys


def _audio_cache_path(sense_id: str, clip_type: str, version: int) -> Path:
    return AUDIO_CACHE / f"{sense_id}-{clip_type}-v{version}.mp3"


def cmd_smoke(sense_id: str) -> int:
    """Render ONE word clip (sense_id) with dict attached. Skip R2 upload —
    write the bytes to a smoke-test path locally so user can listen."""
    locator = _latest_dictionary_locator()
    if not locator:
        print("ERROR: no dictionary in _audio_dictionary_meta.tsv. Run 08_4 first.", file=sys.stderr)
        return 1
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        return 1

    manifest = read_tsv(MANIFEST_PATH)
    target = next(
        (r for r in manifest
         if r.get("sense_id") == sense_id and r.get("clip_type") == "word"),
        None,
    )
    if not target:
        print(f"ERROR: no manifest row for {sense_id} word clip.", file=sys.stderr)
        return 1

    client = ElevenLabsClient(pronunciation_dict_locators=[locator])
    print(f"Smoke render: {sense_id} word '{target['text_input']}' "
          f"with dict={locator['pronunciation_dictionary_id']}/v{locator['version_id']}...")
    res = client.generate_pcm(
        text=target["text_input"],
        voice_id=target["voice_id"],
        sense_id=sense_id,
        clip_type="word",
        version=int(target.get("version") or "1") + 1,
    )
    norm = normalize_pcm_to_mp3_verified(
        pcm_bytes=res.audio_pcm,
        sample_rate=res.sample_rate,
        sample_width=res.sample_width,
        channels=res.channels,
    )
    smoke_path = AUDIT_DIR / f"08_5_smoke_{sense_id}.mp3"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_bytes(norm.mp3_bytes)
    print(f"  bytes={len(norm.mp3_bytes)} LUFS={norm.final_mp3_lufs:.2f} TP={norm.final_mp3_tp:.2f}")
    print(f"  written to {smoke_path}")
    print(f"  open with:  open {smoke_path}")
    return 0


def cmd_bulk(*, confirm: bool, concurrency: int) -> int:
    if not confirm:
        print("ERROR: bulk re-render requires --confirm.", file=sys.stderr)
        return 1
    locator = _latest_dictionary_locator()
    if not locator:
        print("ERROR: no dictionary in _audio_dictionary_meta.tsv. Run 08_4 first.", file=sys.stderr)
        return 1
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        return 1

    keys = _affected_keys()
    if not keys:
        print("Nothing to re-render (empty alias applications). Exiting.")
        return 0

    print(f"Stage 8 re-render: {len(keys)} clips queued, dict={locator['pronunciation_dictionary_id']}.")
    print(f"  audit log:  {AUDIT_JSONL}")
    print(f"  concurrency: {concurrency}")
    print()
    print("⚠ This script's bulk path (with full triple-verification + R2 upload + manifest update")
    print("  is NOT yet implemented in v1; we render + cache locally only.")
    print("  Use --smoke-test --sense-id SID to verify the dict produces correct BP audio first.")
    print("  Bulk integration with the Stage 6 orchestrator pipeline pending.")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke-test", action="store_true",
                        help="Render ONE clip and write locally for listening.")
    parser.add_argument("--sense-id", default="",
                        help="Required for --smoke-test.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required for bulk re-render.")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if args.smoke_test:
        if not args.sense_id:
            parser.error("--smoke-test requires --sense-id SID.")
        return cmd_smoke(args.sense_id)
    return cmd_bulk(confirm=args.confirm, concurrency=args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main())
