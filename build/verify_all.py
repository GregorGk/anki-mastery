"""Verify `data/06-final.tsv` against the invariants in docs/plan.md.

Exits with non-zero on any HARD invariant failure. SOFT warnings print
but don't fail.

Invariants (HARD):
  - row_count == 5,725
  - sense_id is unique
  - sense_id matches the RRRR.EE.SS pattern
  - audio_word + audio_example both non-empty
  - audio URLs contain `eleven_flash_v2_5` (Stage 9 lock)
  - voice_id is non-empty and resolves in config/voices.tsv
  - voice_gender ∈ {m, f} and matches the voice_id's gender in config
  - audio_word_md5 + audio_example_md5 populated (≥10 hex chars each)
  - random sample of 50 audio URLs returns HTTP 200

Soft warnings (DON'T fail):
  - family_root blank
  - source_line blank
  - example_pt blank

Usage:
    .venv/bin/python build/verify_all.py [--http-sample 50]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"

FINAL_PATH = DATA_DIR / "06-final.tsv"
VOICES_PATH = CONFIG_DIR / "voices.tsv"

EXPECTED_ROW_COUNT = 5725
SENSE_ID_PATTERN = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
MD5_MIN_LEN = 10
FLASH_MODEL_MARKER = "eleven_flash_v2_5"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")


class Verifier:
    def __init__(self) -> None:
        self.hard_failures: list[str] = []
        self.soft_warnings: list[str] = []

    def hard_fail(self, msg: str) -> None:
        self.hard_failures.append(msg)
        print(f"  ✗ HARD: {msg}", file=sys.stderr)

    def soft_warn(self, msg: str) -> None:
        self.soft_warnings.append(msg)
        print(f"  ⚠ SOFT: {msg}", file=sys.stderr)

    def passed(self) -> bool:
        return not self.hard_failures


def _verify_rows(rows: list[dict], voice_gender: dict[str, str], v: Verifier) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        v.hard_fail(f"row count {len(rows)} != expected {EXPECTED_ROW_COUNT}")
    else:
        print(f"  ✓ row_count == {EXPECTED_ROW_COUNT}")

    sids = [r["sense_id"] for r in rows]
    if len(set(sids)) != len(sids):
        dups = [s for s in sids if sids.count(s) > 1]
        v.hard_fail(f"duplicate sense_ids: {dups[:5]}")
    else:
        print(f"  ✓ sense_id uniqueness")

    bad_pattern = [s for s in sids if not SENSE_ID_PATTERN.match(s)]
    if bad_pattern:
        v.hard_fail(f"{len(bad_pattern)} sense_ids fail RRRR.EE.SS pattern; e.g., {bad_pattern[:3]}")
    else:
        print(f"  ✓ sense_id pattern (RRRR.EE.SS)")

    missing_audio = [r["sense_id"] for r in rows
                     if not r["audio_word"] or not r["audio_example"]]
    if missing_audio:
        v.hard_fail(f"{len(missing_audio)} rows missing audio_word or audio_example; e.g., {missing_audio[:3]}")
    else:
        print(f"  ✓ audio_word + audio_example populated on all rows")

    non_flash = [r["sense_id"] for r in rows
                 if FLASH_MODEL_MARKER not in r["audio_word"]
                 or FLASH_MODEL_MARKER not in r["audio_example"]]
    if non_flash:
        v.hard_fail(f"{len(non_flash)} rows have non-Flash audio URL; e.g., {non_flash[:3]}")
    else:
        print(f"  ✓ all audio URLs contain '{FLASH_MODEL_MARKER}'")

    missing_md5 = [r["sense_id"] for r in rows
                   if len(r.get("audio_word_md5", "")) < MD5_MIN_LEN
                   or len(r.get("audio_example_md5", "")) < MD5_MIN_LEN]
    if missing_md5:
        v.hard_fail(f"{len(missing_md5)} rows have short/missing md5; e.g., {missing_md5[:3]}")
    else:
        print(f"  ✓ md5 fields populated on all rows")

    bad_voice = []
    bad_gender = []
    for r in rows:
        vid = r.get("voice_id", "")
        if not vid or vid not in voice_gender:
            bad_voice.append(r["sense_id"])
            continue
        expected_gender = "m" if voice_gender[vid] == "male" else "f"
        if r.get("voice_gender", "") != expected_gender:
            bad_gender.append((r["sense_id"], r.get("voice_gender", ""), expected_gender))
    if bad_voice:
        v.hard_fail(f"{len(bad_voice)} rows have bad/unknown voice_id; e.g., {bad_voice[:3]}")
    else:
        print(f"  ✓ voice_id resolves in config/voices.tsv (all {len(rows)})")
    if bad_gender:
        v.hard_fail(f"{len(bad_gender)} rows have voice_gender mismatch; e.g., {bad_gender[:3]}")
    else:
        print(f"  ✓ voice_gender matches config gender")

    # Soft warnings
    n_no_family = sum(1 for r in rows if not r.get("family_root", ""))
    if n_no_family:
        v.soft_warn(f"family_root blank on {n_no_family}/{len(rows)} rows (optional field)")
    n_no_srcline = sum(1 for r in rows if not r.get("source_line", ""))
    if n_no_srcline:
        v.soft_warn(f"source_line blank on {n_no_srcline}/{len(rows)} rows")
    n_no_example = sum(1 for r in rows if not r.get("example_pt", ""))
    if n_no_example:
        v.soft_warn(f"example_pt blank on {n_no_example}/{len(rows)} rows")


def _http_sample(rows: list[dict], n: int, v: Verifier) -> None:
    print(f"\n[HTTP-sample] checking {n} random URLs return 200...")
    rng = random.Random(42)
    sampled = rng.sample(rows, min(n, len(rows)))
    failures: list[tuple[str, int]] = []
    for i, r in enumerate(sampled, start=1):
        for col in ("audio_word", "audio_example"):
            url = r[col]
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    code = resp.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:
                print(f"  {r['sense_id']} {col}: error {e}", file=sys.stderr)
                code = -1
            if code != 200:
                failures.append((url, code))
        if i % 10 == 0:
            print(f"  checked {i}/{len(sampled)} senses ({i*2} URLs)")
    if failures:
        v.hard_fail(f"{len(failures)} URLs not HTTP 200; e.g., {failures[:3]}")
    else:
        print(f"  ✓ all {len(sampled)*2} sampled URLs return HTTP 200")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--http-sample", type=int, default=50,
                        help="How many random rows to HTTP-check (0 = skip).")
    args = parser.parse_args()

    if not FINAL_PATH.exists():
        print(f"ERROR: {FINAL_PATH} not found. Run build/derive_final.py first.",
              file=sys.stderr)
        return 1

    rows = read_tsv(FINAL_PATH)
    voice_gender = {r["voice_id"]: r.get("gender", "")
                    for r in read_tsv(VOICES_PATH)}

    v = Verifier()
    print("[verify_all] checking invariants on 06-final.tsv...")
    _verify_rows(rows, voice_gender, v)
    if args.http_sample > 0:
        _http_sample(rows, args.http_sample, v)

    print()
    if v.passed():
        print(f"✓ verify_all PASSED  (hard=0, soft={len(v.soft_warnings)})")
        return 0
    else:
        print(f"✗ verify_all FAILED  (hard={len(v.hard_failures)}, soft={len(v.soft_warnings)})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
