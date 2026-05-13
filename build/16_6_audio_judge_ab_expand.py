"""Stage 16 / Step 6 — Expand the AudioJudge A/B pool by 100 more senses.

Additively grows `data/_audio_judge_ab.tsv` to 400 rows (200 BP + 200 EP).
New senses are biased toward EP-divergent phonetic patterns + a
hard-coded must-include list (real, hospital, gente, tia, verdade).

Composition of the 100 new senses (target):
   5  must-include (user-named EP-tricky cases)
  20  bp_status ∈ {false_friend, nsfw, uncommon}  (sensitive metadata)
  35  final -l / -al / -el / -il words            (EP keeps consonant, BP vocalizes)
  20  final -de / -te words                       (EP silent, BP palatalized)
  10  final -s codas                              (EP [ʃ], BP [s])
  10  verbs in -ar / -er / -ir                    (varied prosody contrast)

Each new sense gets ONE BP clip (from the existing v3 regen on R2) plus
one freshly synthesized EP clip via Nelson Silvestre. Row IDs continue
from 101_… so the existing 1..100 numbering stays intact.

Usage:
    .venv/bin/python build/16_6_audio_judge_ab_expand.py --dry-run
    .venv/bin/python build/16_6_audio_judge_ab_expand.py --yes
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client, R2Config  # noqa: E402

DATA = REPO_ROOT / "data"
FINAL_TSV = DATA / "06-final.tsv"
AB_TSV = DATA / "_audio_judge_ab.tsv"

V3_MODEL_ID = "eleven_v3"
EP_VOICE_ID = "hOLl3246BMBsdy0qtYLb"
EP_VOICE_NAME = "Nelson Silvestre"
R2_PREFIX = "audio_judge_ab"
SEED = 1606
WORKERS = 3
DEFAULT_N_NEW = 100

# Hard-coded must-include sense_ids — the user-named tricky cases.
MUST_INCLUDE_SENSE_IDS = [
    "0200.00.03",  # real (BRL currency) — final -l + cognate
    "0777.00.01",  # hospital — final -l + cognate
    "0281.00.01",  # gente — -nte palatalization marker
    "3789.00.01",  # tia — ti palatalization
    "0236.00.01",  # verdade — -de palatalization
]

# Composition fractions of the total `--n` after the must-include picks.
# Sum ≈ 0.95; the remaining ~5% comes from must_include or filler.
FRAC_FALSE_FRIEND_NSFW = 0.20
FRAC_FINAL_L           = 0.35
FRAC_FINAL_DE_TE       = 0.20
FRAC_FINAL_S           = 0.10
FRAC_VERB_INFINITIVE   = 0.10


def _voices_map() -> dict[str, str]:
    with (REPO_ROOT / "config" / "voices.tsv").open(encoding="utf-8") as f:
        return {r["voice_id"]: r["bp_name"]
                for r in csv.DictReader(f, dialect="excel-tab")
                if (r.get("status") or "active") == "active"}


def _has_suffix(s: str, suffixes: tuple[str, ...]) -> bool:
    w = s.lower().strip()
    return any(w.endswith(suf) for suf in suffixes)


def _is_function_word(row: dict) -> bool:
    pos = row.get("pos", "")
    return pos in ("art", "prep", "conj", "pron", "num", "interj")


def _pick_senses(rng: random.Random, n_new: int) -> list[dict]:
    existing = set()
    if AB_TSV.exists():
        existing = {r["sense_id"] for r in
                    csv.DictReader(AB_TSV.open(encoding="utf-8"), dialect="excel-tab")}
    print(f"  existing pool sense_ids: {len(existing)}")

    final = list(csv.DictReader(FINAL_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    by_sid = {r["sense_id"]: r for r in final}

    active_voices = set(_voices_map())
    # Use 045-speaker_gender for current live BP voice (post-Lair swap).
    current_voice = {r["sense_id"]: r["voice_id"]
                     for r in csv.DictReader(
                         (DATA / "045-speaker_gender.tsv").open(encoding="utf-8"),
                         dialect="excel-tab")}

    def _eligible(r: dict) -> bool:
        if r["sense_id"] in existing:
            return False
        if not (r.get("pt_display") or r.get("pt")):
            return False
        if not r.get("example_pt"):
            return False
        # Skip ultra-short pt (1 char) — those have rendering edge cases ('ó').
        if len((r.get("pt_display") or r["pt"]).strip()) <= 1:
            return False
        # Voice must be active (post-swap).
        v = current_voice.get(r["sense_id"])
        return v in active_voices

    candidates = [r for r in final if _eligible(r)]
    # Annotate each candidate with its current BP voice.
    for r in candidates:
        r["_live_voice"] = current_voice.get(r["sense_id"], r["voice_id"])

    picks: list[dict] = []
    used_sids: set[str] = set()

    def _add(r: dict, source: str) -> None:
        sid = r["sense_id"]
        if sid in used_sids:
            return
        used_sids.add(sid)
        picks.append({**r, "_source": source})

    # 1. Must-include
    for sid in MUST_INCLUDE_SENSE_IDS:
        r = by_sid.get(sid)
        if r and sid not in existing and sid not in used_sids:
            r["_live_voice"] = current_voice.get(sid, r["voice_id"])
            _add(r, "must_include")

    # Compute bucket targets scaled by n_new (must_include is fixed).
    t_sensitive       = round(n_new * FRAC_FALSE_FRIEND_NSFW)
    t_final_l         = round(n_new * FRAC_FINAL_L)
    t_final_de_te     = round(n_new * FRAC_FINAL_DE_TE)
    t_final_s         = round(n_new * FRAC_FINAL_S)
    t_verb_infinitive = round(n_new * FRAC_VERB_INFINITIVE)

    # 2. Sensitive bp_status (false_friend / nsfw / uncommon)
    sensitive = [r for r in candidates
                 if r.get("bp_status") in ("false_friend", "nsfw", "uncommon")
                 and r["sense_id"] not in used_sids]
    rng.shuffle(sensitive)
    for r in sensitive[:t_sensitive]:
        _add(r, "sensitive")

    # 3. Final -l / -al / -el / -il / -ol / -ul (exclude function words & short)
    final_l_pool = [r for r in candidates
                    if _has_suffix(r["pt"], ("l", "al", "el", "il", "ol", "ul"))
                    and r["sense_id"] not in used_sids
                    and not _is_function_word(r)
                    and len(r["pt"]) >= 3]
    rng.shuffle(final_l_pool)
    for r in final_l_pool[:t_final_l]:
        _add(r, "final_l")

    # 4. Final -de / -te
    de_te_pool = [r for r in candidates
                  if _has_suffix(r["pt"], ("de", "te"))
                  and r["sense_id"] not in used_sids
                  and not _is_function_word(r)
                  and len(r["pt"]) >= 3]
    rng.shuffle(de_te_pool)
    for r in de_te_pool[:t_final_de_te]:
        _add(r, "final_de_te")

    # 5. Final -s coda
    s_pool = [r for r in candidates
              if _has_suffix(r["pt"], ("s",))
              and r["sense_id"] not in used_sids
              and not _is_function_word(r)
              and len(r["pt"]) >= 3]
    rng.shuffle(s_pool)
    for r in s_pool[:t_final_s]:
        _add(r, "final_s")

    # 6. Verb infinitives
    verb_pool = [r for r in candidates
                 if r.get("pos") == "verb"
                 and _has_suffix(r["pt"], ("ar", "er", "ir"))
                 and r["sense_id"] not in used_sids]
    rng.shuffle(verb_pool)
    for r in verb_pool[:t_verb_infinitive]:
        _add(r, "verb_infinitive")

    # 7. Filler — if we're short, pull random from candidates.
    if len(picks) < n_new:
        leftover = [r for r in candidates if r["sense_id"] not in used_sids]
        rng.shuffle(leftover)
        for r in leftover[:n_new - len(picks)]:
            _add(r, "filler")

    # Cap at n_new (the bias quotas overshoot slightly if pools are sparse).
    picks = picks[:n_new]

    # Decide clip_type per sense: 50/50 word/example, randomized.
    rng.shuffle(picks)
    half = n_new // 2
    for i, r in enumerate(picks):
        r["_ct"] = "word" if i < half else "example"

    return picks


def _next_row_id_start() -> int:
    """Find the next row-number after the existing pool (e.g., 101 if pool has 1..100)."""
    if not AB_TSV.exists():
        return 1
    max_n = 0
    for r in csv.DictReader(AB_TSV.open(encoding="utf-8"), dialect="excel-tab"):
        rid = r["row_id"]   # 'NNN_bp' or 'NNN_ep'
        try:
            n = int(rid.split("_", 1)[0])
        except ValueError:
            continue
        max_n = max(max_n, n)
    return max_n + 1


@dataclass
class EPJob:
    new_id: int
    sense_id: str
    clip_type: str
    text: str
    bp_voice_id: str
    bp_voice_name: str
    bp_v3_url: str
    source: str

    @property
    def ep_object_key(self) -> str:
        short = "word" if self.clip_type == "word" else "ex"
        return f"{R2_PREFIX}/{self.sense_id}-{short}-{V3_MODEL_ID}-ep.mp3"


def _render_one(job: EPJob, el: ElevenLabsClient, r2: R2Client) -> tuple[EPJob, str, str]:
    try:
        tts = el.generate_pcm(
            text=job.text, voice_id=EP_VOICE_ID,
            sense_id=job.sense_id, clip_type=job.clip_type, version=1,
        )
        norm = normalize_pcm_to_mp3_verified(tts.audio_pcm)
        up = r2.upload_bytes(
            norm.mp3_bytes, job.ep_object_key,
            extra_metadata={
                "sense_id": job.sense_id, "clip_type": job.clip_type,
                "voice_id": EP_VOICE_ID, "model": V3_MODEL_ID,
                "stage": "16_6_expand",
            },
        )
        return job, up.url, ""
    except Exception as exc:  # noqa: BLE001
        return job, "", f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--n", type=int, default=DEFAULT_N_NEW,
                    help=f"Number of NEW senses to add (default {DEFAULT_N_NEW}).")
    args = ap.parse_args()

    voices = _voices_map()
    rng = random.Random(SEED)
    picks = _pick_senses(rng, args.n)
    if len(picks) < args.n:
        print(f"  WARN: only found {len(picks)} eligible new senses (target {args.n})",
              file=sys.stderr)

    print()
    print(f"  picked {len(picks)} new senses:")
    src_counts: dict[str, int] = {}
    for r in picks:
        src_counts[r["_source"]] = src_counts.get(r["_source"], 0) + 1
    for src, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {src:<22} {n:>3}")

    next_n = _next_row_id_start()
    print(f"  new row_ids will start at: {next_n:03d}_bp / {next_n:03d}_ep")

    jobs: list[EPJob] = []
    rows_to_append: list[dict] = []
    for idx, r in enumerate(picks):
        new_id = next_n + idx
        ct = r["_ct"]
        text = r["pt_display"] if ct == "word" else r["example_pt"]
        v3_short = "word" if ct == "word" else "ex"
        bp_url = (
            f"https://pub-155fa287ae724688aaa870d99135311a.r2.dev/audio/"
            f"{r['sense_id']}-{v3_short}-{V3_MODEL_ID}-v1.mp3"
        )
        live_voice = r["_live_voice"]
        live_voice_name = voices.get(live_voice, live_voice[:10])
        jobs.append(EPJob(
            new_id=new_id,
            sense_id=r["sense_id"], clip_type=ct, text=text,
            bp_voice_id=live_voice, bp_voice_name=live_voice_name,
            bp_v3_url=bp_url, source=r["_source"],
        ))
        # BP row (pre-filled — no API needed):
        rows_to_append.append({
            "row_id": f"{new_id:03d}_bp",
            "sense_id": r["sense_id"], "clip_type": ct, "dialect": "BP",
            "text": text,
            "voice_id": live_voice, "voice_name": live_voice_name,
            "bp_url_existing": bp_url, "url": bp_url,
        })

    total_chars = sum(len(j.text) for j in jobs)
    print(f"  EP renders to issue:  {len(jobs)}")
    print(f"  total chars:          {total_chars:,}")
    print(f"  EP TTS cost (est):    ${total_chars * 22.50e-6:.4f}")
    print(f"  workers:              {WORKERS}")
    print()
    if args.dry_run:
        print("--dry-run: stopping.")
        return 0
    if not args.yes:
        sys.stdout.write("Type GO to render the EP clips: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    r2 = R2Client(R2Config.from_env())
    el = ElevenLabsClient(
        model_id=V3_MODEL_ID,
        language_code="pt",
        pronunciation_dict_locators=None,
    )

    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(_render_one, j, el, r2) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            job, url, err = fut.result()
            if err:
                print(f"  [{i:>3}/{len(jobs)}] FAIL {job.sense_id} {job.clip_type:<7} "
                      f"{err[:80]}", file=sys.stderr)
                continue
            n_ok += 1
            rows_to_append.append({
                "row_id": f"{job.new_id:03d}_ep",
                "sense_id": job.sense_id, "clip_type": job.clip_type, "dialect": "EP",
                "text": job.text,
                "voice_id": EP_VOICE_ID, "voice_name": EP_VOICE_NAME,
                "bp_url_existing": job.bp_v3_url, "url": url,
            })
            print(f"  [{i:>3}/{len(jobs)}] OK   {job.sense_id} {job.clip_type:<7} "
                  f"src={job.source:<18} → {url}", flush=True)

    print(f"\n  rendered: {n_ok}/{len(jobs)} new EP clips")

    # Append to existing AB TSV (preserves header).
    existing_rows = list(csv.DictReader(AB_TSV.open(encoding="utf-8"), dialect="excel-tab"))
    fields = list(existing_rows[0].keys())
    merged = existing_rows + rows_to_append
    merged.sort(key=lambda r: r["row_id"])
    with AB_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, dialect="excel-tab",
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"  AB TSV now: {len(merged)} rows (was {len(existing_rows)}, added {len(rows_to_append)})")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
