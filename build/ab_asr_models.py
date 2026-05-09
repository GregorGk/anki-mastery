"""A/B test 4 ASR models on the cached Stage 6 MP3s.

Reads `data/_audio_manifest.tsv` to find the current (versioned) MP3 for
every (sense_id, clip_type), reads the corresponding file from
`build/audio_cache/`, and transcribes each through:

  1. openai/whisper-1               (current default)
  2. openai/gpt-4o-mini-transcribe  (newer, cheaper)
  3. openai/gpt-4o-transcribe       (newer, full quality)
  4. elevenlabs/scribe_v2

Word clips ≤3 chars (function words) get judged by phonetic distance;
everything else by Levenshtein on normalized text. Same thresholds as
Stage 6 production policy.

Outputs:
  audit/ab_asr.jsonl                      machine-readable per-clip results
  audit/ab_asr_summary.tsv                tabulated PASS/REGEN per (model, axis)
  audit/ab_asr_disagreements.log          where models disagree on the same clip

Usage:
  .venv/bin/python build/ab_asr_models.py
  .venv/bin/python build/ab_asr_models.py --limit 50
  .venv/bin/python build/ab_asr_models.py --models whisper-1 gpt-4o-mini-transcribe
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import (  # noqa: E402
    SHORT_INPUT_MAX_CHARS,
    SHORT_INPUT_PHONETIC_PASS_DISTANCE,
    THRESHOLD_LONGTAIL,
    THRESHOLD_TOP1000,
    is_short_input,
    pad_mp3_with_silence,
    phonetic_distance,
    text_similarity,
    _audio_duration_seconds,
)
from build.lib.asr_alt import (  # noqa: E402
    PRICE_PER_MINUTE,
    make_elevenlabs_client,
    make_openai_client,
    transcribe_elevenlabs,
    transcribe_openai,
)
from build.lib.audio_manifest import read_manifest  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
CACHE_DIR = REPO_ROOT / "build" / "audio_cache"

OUT_JSONL = AUDIT_DIR / "ab_asr.jsonl"
OUT_SUMMARY = AUDIT_DIR / "ab_asr_summary.tsv"
OUT_DISAGREE = AUDIT_DIR / "ab_asr_disagreements.log"

DEFAULT_MODELS = (
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "scribe_v2",
)
TOP_1000_RANK_THRESHOLD = 1000


@dataclass
class ClipResult:
    sense_id: str
    clip_type: str
    voice_id: str
    input_text: str
    is_short: bool
    rank: int
    model: str
    transcript: str
    text_sim: float
    phonetic_dist: float | None
    decision: str         # 'pass' | 'regen'
    threshold_used: float
    cost_usd: float
    attempts: int
    notes: str = ""


def _decide(
    *,
    input_text: str,
    transcript: str,
    is_top_1000: bool,
) -> tuple[str, float, float | None, float, str]:
    """Mirror Stage 6 decision logic. Returns (decision, sim, pdist, threshold, mode)."""
    if is_short_input(input_text):
        pdist = phonetic_distance(input_text, transcript)
        sim = text_similarity(transcript, input_text)
        if pdist is None:
            decision = "pass" if sim >= 0.99 else "regen"
            return decision, sim, None, 0.99, "phonetic-fallback-text"
        decision = "pass" if pdist <= SHORT_INPUT_PHONETIC_PASS_DISTANCE else "regen"
        return decision, sim, pdist, SHORT_INPUT_PHONETIC_PASS_DISTANCE, "phonetic"
    sim = text_similarity(transcript, input_text)
    pdist = phonetic_distance(input_text, transcript)
    threshold = THRESHOLD_TOP1000 if is_top_1000 else THRESHOLD_LONGTAIL
    decision = "pass" if sim >= threshold else "regen"
    return decision, sim, pdist, threshold, "text"


def _resolve_cache_path(manifest_row: dict) -> Path:
    """Map the manifest row to the cached MP3 path.

    object_key is e.g. "audio/0001.00.01-word-v1.mp3" — we cache as
    "{sense_id}-{ctype}-v{N}.mp3" (no "audio/" prefix, full clip_type).
    """
    sid = manifest_row["sense_id"]
    ctype = manifest_row["clip_type"]
    version = manifest_row["version"]
    return CACHE_DIR / f"{sid}-{ctype}-v{version}.mp3"


def _is_top_1000(rank: int) -> bool:
    return 0 < rank <= TOP_1000_RANK_THRESHOLD


def _per_clip(
    *,
    manifest_row: dict,
    rank: int,
    model: str,
    openai_client,
    elevenlabs_client,
) -> list[ClipResult]:
    sid = manifest_row["sense_id"]
    ctype = manifest_row["clip_type"]
    voice_id = manifest_row["voice_id"]
    input_text = manifest_row["text_input"]
    cache = _resolve_cache_path(manifest_row)
    if not cache.exists():
        return [
            ClipResult(
                sense_id=sid,
                clip_type=ctype,
                voice_id=voice_id,
                input_text=input_text,
                is_short=is_short_input(input_text),
                rank=rank,
                model=model,
                transcript="",
                text_sim=0.0,
                phonetic_dist=None,
                decision="regen",
                threshold_used=0.0,
                cost_usd=0.0,
                attempts=0,
                notes=f"cache miss: {cache}",
            )
        ]
    mp3_bytes = cache.read_bytes()
    is_word = ctype == "word"

    # Pad word clips before sending to OpenAI ASR (matches Stage 6 path).
    send_bytes = mp3_bytes
    if is_word:
        try:
            send_bytes = pad_mp3_with_silence(mp3_bytes)
        except RuntimeError:
            pass

    duration = _audio_duration_seconds(send_bytes if is_word else mp3_bytes)

    # Build a biased prompt for OpenAI word clips (matches Stage 6 path).
    prompt = f"Palavra em português brasileiro: {input_text}" if is_word else None

    try:
        if model == "scribe_v2":
            resp = transcribe_elevenlabs(
                client=elevenlabs_client,
                mp3_bytes=send_bytes if is_word else mp3_bytes,
                language_code="por",
                duration_seconds=duration,
            )
        else:
            resp = transcribe_openai(
                client=openai_client,
                model=model,
                mp3_bytes=send_bytes if is_word else mp3_bytes,
                language="pt",
                prompt=prompt,
                filename=f"{sid}-{ctype}.mp3",
                duration_seconds=duration,
            )
        decision, sim, pdist, thresh, mode = _decide(
            input_text=input_text,
            transcript=resp.transcript,
            is_top_1000=_is_top_1000(rank),
        )
        return [
            ClipResult(
                sense_id=sid,
                clip_type=ctype,
                voice_id=voice_id,
                input_text=input_text,
                is_short=is_short_input(input_text),
                rank=rank,
                model=model,
                transcript=resp.transcript,
                text_sim=round(sim, 4),
                phonetic_dist=(round(pdist, 4) if pdist is not None else None),
                decision=decision,
                threshold_used=round(thresh, 4),
                cost_usd=resp.cost_usd,
                attempts=resp.attempts,
                notes=mode,
            )
        ]
    except Exception as exc:  # noqa: BLE001
        return [
            ClipResult(
                sense_id=sid,
                clip_type=ctype,
                voice_id=voice_id,
                input_text=input_text,
                is_short=is_short_input(input_text),
                rank=rank,
                model=model,
                transcript="",
                text_sim=0.0,
                phonetic_dist=None,
                decision="regen",
                threshold_used=0.0,
                cost_usd=0.0,
                attempts=0,
                notes=f"error: {type(exc).__name__}: {str(exc)[:160]}",
            )
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help=f"Models to compare. Default: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit clips per model")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    manifest = read_manifest()
    if not manifest:
        print("[ab_asr] no manifest; run Stage 6 first.", file=sys.stderr)
        return 1
    manifest = [r for r in manifest if r.get("status") == "uploaded"]
    if args.limit:
        manifest = manifest[: args.limit]

    # Need rank for top-1000 threshold logic.
    from build.lib.tsv import read_tsv

    ipa_rows = read_tsv(DATA_DIR / "05-ipa.tsv")
    rank_by_sid = {r["sense_id"]: int(r.get("rank", 0) or 0) for r in ipa_rows}

    print(f"[ab_asr] manifest rows: {len(manifest):,}", file=sys.stderr)
    print(f"[ab_asr] models: {args.models}", file=sys.stderr)
    print(f"[ab_asr] total ASR calls planned: {len(manifest) * len(args.models):,}", file=sys.stderr)

    openai_client = make_openai_client()
    elevenlabs_client = None
    if "scribe_v2" in args.models:
        elevenlabs_client = make_elevenlabs_client()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")  # truncate

    all_results: list[ClipResult] = []
    for model in args.models:
        print(f"\n[ab_asr] === {model} ===", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(
                    _per_clip,
                    manifest_row=row,
                    rank=rank_by_sid.get(row["sense_id"], 0),
                    model=model,
                    openai_client=openai_client,
                    elevenlabs_client=elevenlabs_client,
                ): row
                for row in manifest
            }
            done = 0
            for fut in as_completed(futs):
                results = fut.result()
                for r in results:
                    all_results.append(r)
                    with OUT_JSONL.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                done += 1
                if done % 25 == 0:
                    print(f"  [{model}] {done}/{len(manifest)}", file=sys.stderr)

    # ----- Summary -----
    summary_rows: list[dict] = []
    by_model: dict[str, list[ClipResult]] = defaultdict(list)
    for r in all_results:
        by_model[r.model].append(r)

    print("\n=== A/B summary by model ===", file=sys.stderr)
    print(f"{'model':<28} {'pass':>6} {'regen':>6} {'pass_rate':>10} {'short_pass':>10} {'long_pass':>10} {'cost_usd':>9}", file=sys.stderr)
    for model in args.models:
        rs = by_model.get(model, [])
        if not rs:
            continue
        passes = sum(1 for r in rs if r.decision == "pass")
        regens = sum(1 for r in rs if r.decision == "regen")
        short_clips = [r for r in rs if r.is_short]
        long_clips = [r for r in rs if not r.is_short]
        short_pass = sum(1 for r in short_clips if r.decision == "pass")
        long_pass = sum(1 for r in long_clips if r.decision == "pass")
        cost = sum(r.cost_usd for r in rs)
        pr = passes / len(rs) if rs else 0
        spr = short_pass / len(short_clips) if short_clips else 0
        lpr = long_pass / len(long_clips) if long_clips else 0
        line = f"{model:<28} {passes:>6} {regens:>6} {pr * 100:>9.1f}% {spr * 100:>9.1f}% {lpr * 100:>9.1f}% ${cost:>7.3f}"
        print(line, file=sys.stderr)
        summary_rows.append(
            {
                "model": model,
                "total": len(rs),
                "pass": passes,
                "regen": regens,
                "pass_rate": f"{pr:.4f}",
                "short_clips": len(short_clips),
                "short_pass": short_pass,
                "short_pass_rate": f"{spr:.4f}" if short_clips else "",
                "long_clips": len(long_clips),
                "long_pass": long_pass,
                "long_pass_rate": f"{lpr:.4f}" if long_clips else "",
                "cost_usd": f"{cost:.4f}",
                "price_per_min": f"{PRICE_PER_MINUTE.get(model, 0):.4f}",
            }
        )
    OUT_SUMMARY.write_text(
        "model\ttotal\tpass\tregen\tpass_rate\tshort_clips\tshort_pass\tshort_pass_rate\tlong_clips\tlong_pass\tlong_pass_rate\tcost_usd\tprice_per_min\n"
        + "\n".join("\t".join(str(r[k]) for k in r) for r in summary_rows)
        + "\n",
        encoding="utf-8",
    )

    # ----- Disagreements -----
    by_clip: dict[tuple[str, str], dict[str, ClipResult]] = defaultdict(dict)
    for r in all_results:
        by_clip[(r.sense_id, r.clip_type)][r.model] = r
    disagree_lines: list[str] = []
    for (sid, ctype), per_model in by_clip.items():
        decisions = {m: r.decision for m, r in per_model.items()}
        if len(set(decisions.values())) > 1:
            base = next(iter(per_model.values()))
            disagree_lines.append(
                f"=== {sid} {ctype}  '{base.input_text}'  (rank={base.rank}, voice={base.voice_id[:6]}…)"
            )
            for m, r in per_model.items():
                pdist_str = "  -  " if r.phonetic_dist is None else f"{r.phonetic_dist:>4.2f}"
                disagree_lines.append(
                    f"   {m:<28} {r.decision:<5} sim={r.text_sim:>4.2f} pdist={pdist_str}  ->  {r.transcript}"
                )
            disagree_lines.append("")
    OUT_DISAGREE.parent.mkdir(parents=True, exist_ok=True)
    OUT_DISAGREE.write_text("\n".join(disagree_lines), encoding="utf-8")
    print(f"\n[ab_asr] disagreements: {sum(1 for v in by_clip.values() if len({r.decision for r in v.values()}) > 1)}", file=sys.stderr)

    # Cross-model agreement matrix on PASS verdicts
    print("\n=== Pairwise PASS-agreement matrix ===", file=sys.stderr)
    models = list(args.models)
    header = "         " + "".join(f"{m[:14]:<16}" for m in models)
    print(header, file=sys.stderr)
    for m1 in models:
        row = f"{m1[:8]:<9}"
        for m2 in models:
            both_pass = 0
            both_count = 0
            for per_model in by_clip.values():
                if m1 in per_model and m2 in per_model:
                    both_count += 1
                    if per_model[m1].decision == "pass" and per_model[m2].decision == "pass":
                        both_pass += 1
            agree = both_pass / both_count if both_count else 0
            row += f"{agree * 100:>14.1f}% "
        print(row, file=sys.stderr)

    print(f"\n[ab_asr] wrote: {OUT_JSONL}", file=sys.stderr)
    print(f"[ab_asr] wrote: {OUT_SUMMARY}", file=sys.stderr)
    print(f"[ab_asr] wrote: {OUT_DISAGREE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
