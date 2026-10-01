"""Azure Pronunciation Assessment spike — does Azure agree with Gemini?

Cross-validates Azure Speech pronunciation-assessment scores against the
Stage 16.8 Gemini drift verdicts, on the SAME audio bytes: the original
v1 clips still sitting in build/audio_cache/ from the 16.8 prefetch
(downloaded BEFORE Stage 16.9 re-rendered the 222 drifting clips).

Samples N clips Gemini called `non_bp` + N clips Gemini called `bp_ok`,
scores each with Azure (pt-BR, phoneme granularity), and reports whether
Azure's pronunciation / accuracy / worst-phoneme scores separate the two
Gemini groups — and where the two judges disagree.

NOT a production stage — a proof-of-value spike. ~50 clips ≈ 3 min of
audio, well within Azure's F0 free tier.

Usage:
    .venv/bin/python build/azure_pa_spike.py
    .venv/bin/python build/azure_pa_spike.py --n 15 --workers 4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import azure.cognitiveservices.speech as speechsdk  # noqa: E402

from build.lib.loudness import FFMPEG  # noqa: E402

DATA = REPO_ROOT / "data"
AUDIT = REPO_ROOT / "audit"
CACHE = REPO_ROOT / "build" / "audio_cache"
GEMINI_AUDIT = AUDIT / "16_8_gemini_qa_gate.jsonl"

# clip_type → filename segment (the Stage-17 "-ex-" trap).
CLIP_SEG = {"word": "word", "example": "ex"}


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


def _cache_path(sense_id: str, clip_type: str) -> Path:
    seg = CLIP_SEG.get(clip_type, clip_type)
    return CACHE / f"{sense_id}-{seg}-eleven_v3-v1.mp3"


def _load_gemini_records() -> list[dict]:
    """Stage 16.8 Gemini verdicts that have a local v1 cache file."""
    out: list[dict] = []
    for line in GEMINI_AUDIT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") != "judged":
            continue
        if not _cache_path(r.get("sense_id", ""), r.get("clip_type", "")).exists():
            continue
        out.append(r)
    return out


# Global request pacer — the F0 free tier caps at 20 requests / 60 s, so
# every Azure call (across all worker threads) is spaced >= min_interval.
_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


def _pace(min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _PACE_LOCK:
        wait = min_interval - (time.monotonic() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.monotonic()


def _mp3_to_wav(mp3_path: Path) -> str:
    """ffmpeg → temp 16 kHz mono 16-bit PCM WAV (Azure's preferred format).
    Returns the temp path; caller deletes it."""
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav],
        check=True, capture_output=True,
    )
    return wav


@dataclass
class AzureScore:
    ok: bool
    accuracy: float = 0.0
    fluency: float = 0.0
    completeness: float = 0.0
    pronunciation: float = 0.0
    worst_phoneme: float = 100.0
    worst_phoneme_label: str = ""
    recognized: str = ""
    error: str = ""


def _assess(key: str, region: str, mp3_path: Path, reference_text: str,
            min_interval: float = 3.2, max_attempts: int = 5) -> AzureScore:
    """One Azure pronunciation-assessment call on one clip.

    A global pacer spaces calls >= min_interval apart (the F0 free tier
    caps at 20 requests / 60 s); also retries on `Canceled`.
    """
    try:
        wav = _mp3_to_wav(mp3_path)
    except Exception as exc:  # noqa: BLE001
        return AzureScore(ok=False, error=f"ffmpeg: {type(exc).__name__}: {exc}")
    try:
        last_err = ""
        for attempt in range(1, max_attempts + 1):
            try:
                speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
                speech_config.speech_recognition_language = "pt-BR"
                audio_config = speechsdk.audio.AudioConfig(filename=wav)
                pa_config = speechsdk.PronunciationAssessmentConfig(
                    reference_text=reference_text,
                    grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                    granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
                    enable_miscue=False,
                )
                # IPA-format phonemes — they line up with the Stage-5 IPA.
                try:
                    pa_config.phoneme_alphabet = "IPA"
                except Exception:  # noqa: BLE001 — older SDKs lack the property
                    pass
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config, audio_config=audio_config)
                pa_config.apply_to(recognizer)
                _pace(min_interval)
                result = recognizer.recognize_once()
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30.0, 2.0 ** attempt))
                continue

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                pa = speechsdk.PronunciationAssessmentResult(result)
                worst, worst_label = 100.0, ""
                for w in pa.words or []:
                    for ph in (w.phonemes or []):
                        if ph.accuracy_score < worst:
                            worst = ph.accuracy_score
                            worst_label = f"{ph.phoneme or '?'} in {w.word!r}"
                return AzureScore(
                    ok=True,
                    accuracy=pa.accuracy_score or 0.0,
                    fluency=pa.fluency_score or 0.0,
                    completeness=pa.completeness_score or 0.0,
                    pronunciation=pa.pronunciation_score or 0.0,
                    worst_phoneme=worst,
                    worst_phoneme_label=worst_label,
                    recognized=result.text or "",
                )
            if result.reason == speechsdk.ResultReason.Canceled:
                c = result.cancellation_details
                last_err = f"Canceled/{c.reason}: {(c.error_details or '')[:120]}"
                low = (c.error_details or "").lower()
                if any(k in low for k in ("401", "forbidden", "unauthorized",
                                          "invalid subscription")):
                    return AzureScore(ok=False, error=last_err)  # not retryable
                time.sleep(min(30.0, 2.0 ** attempt))  # throttle/transient → retry
                continue
            # NoMatch — audio didn't recognize as speech; not retryable
            return AzureScore(ok=False, error=f"reason={result.reason}")
        return AzureScore(ok=False, error=last_err or "max retries exhausted")
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


@dataclass
class Row:
    rec: dict
    score: AzureScore = field(default_factory=lambda: AzureScore(ok=False))


def _summary(label: str, rows: list[Row]) -> dict:
    ok = [r for r in rows if r.score.ok]
    if not ok:
        print(f"  {label}: 0 / {len(rows)} scored OK")
        return {}
    pron = [r.score.pronunciation for r in ok]
    acc = [r.score.accuracy for r in ok]
    worst = [r.score.worst_phoneme for r in ok]
    print(f"  {label}  (n={len(ok)}/{len(rows)} scored)")
    print(f"    pronunciation : mean {mean(pron):5.1f}  median {median(pron):5.1f}  "
          f"min {min(pron):5.1f}  max {max(pron):5.1f}")
    print(f"    accuracy      : mean {mean(acc):5.1f}  median {median(acc):5.1f}")
    print(f"    worst phoneme : mean {mean(worst):5.1f}  median {median(worst):5.1f}  "
          f"min {min(worst):5.1f}")
    return {"pron": pron, "acc": acc, "worst": worst, "rows": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=25,
                    help="Clips per Gemini group (non_bp + bp_ok). Default 25.")
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrency for ffmpeg + I/O. Azure calls are "
                         "serialized by the global pacer. Default 3.")
    ap.add_argument("--min-interval", type=float, default=3.2,
                    help="Seconds between Azure calls (F0 free tier caps at "
                         "20 / 60 s). Default 3.2.")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    _load_env()
    key = os.environ.get("SPEECH_KEY")
    region = os.environ.get("SPEECH_REGION")
    if not key or not region:
        print("ERROR: SPEECH_KEY / SPEECH_REGION not set in .env", file=sys.stderr)
        return 1

    records = _load_gemini_records()
    non_bp = [r for r in records if r.get("verdict") == "non_bp"]
    bp_ok = [r for r in records if r.get("verdict") == "bp_ok"]
    print(f"=== Azure PA spike — Azure vs Gemini on the same v1 audio ===")
    print(f"  Gemini 16.8 verdicts with a local cache file: "
          f"{len(records)}  ({len(non_bp)} non_bp / {len(bp_ok)} bp_ok)")

    rng = random.Random(args.seed)
    pick_non = rng.sample(non_bp, min(args.n, len(non_bp)))
    pick_bp = rng.sample(bp_ok, min(args.n, len(bp_ok)))
    sample = [("non_bp", r) for r in pick_non] + [("bp_ok", r) for r in pick_bp]
    print(f"  sampling {len(pick_non)} non_bp + {len(pick_bp)} bp_ok = "
          f"{len(sample)} clips  (seed {args.seed})")
    print(f"  Azure: pt-BR, HundredMark, phoneme granularity. "
          f"~{len(sample) * 3 / 60:.1f} min audio (F0 free tier covers it).")
    print()

    def _work(item: tuple[str, dict]) -> Row:
        _group, rec = item
        path = _cache_path(rec["sense_id"], rec["clip_type"])
        return Row(rec=rec, score=_assess(key, region, path, rec.get("pt", ""),
                                          min_interval=args.min_interval))

    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, it): it for it in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if i % 10 == 0 or i == len(sample):
                print(f"  scored {i}/{len(sample)}", flush=True)

    # ── per-clip table ────────────────────────────────────────────
    print()
    print(f"{'sense_id':<12}{'clip':<8}{'gemini':<9}{'pron':>6}{'acc':>6}"
          f"{'worst':>7}  {'worst phoneme':<22}{'reference'}")
    print("-" * 110)
    rows.sort(key=lambda r: (r.rec.get("verdict", ""),
                             r.score.pronunciation if r.score.ok else 999))
    for r in rows:
        s = r.score
        ref = (r.rec.get("pt", "") or "")[:34]
        if not s.ok:
            print(f"{r.rec['sense_id']:<12}{r.rec['clip_type']:<8}"
                  f"{r.rec.get('verdict',''):<9}{'ERR':>6}{'':>6}{'':>7}  "
                  f"{s.error[:22]:<22}{ref}")
            continue
        print(f"{r.rec['sense_id']:<12}{r.rec['clip_type']:<8}"
              f"{r.rec.get('verdict',''):<9}{s.pronunciation:>6.1f}{s.accuracy:>6.1f}"
              f"{s.worst_phoneme:>7.1f}  {s.worst_phoneme_label[:22]:<22}{ref}")

    # ── group summary ─────────────────────────────────────────────
    non_rows = [r for r in rows if r.rec.get("verdict") == "non_bp"]
    bp_rows = [r for r in rows if r.rec.get("verdict") == "bp_ok"]
    print()
    print("=== group summary ===")
    g_non = _summary("Gemini non_bp", non_rows)
    g_bp = _summary("Gemini bp_ok ", bp_rows)

    # ── separation verdict ────────────────────────────────────────
    if g_non and g_bp:
        print()
        print("=== separation ===")
        bp_p25 = sorted(g_bp["pron"])[max(0, len(g_bp["pron"]) // 4 - 1)]
        non_below = sum(1 for p in g_non["pron"] if p < bp_p25)
        d_pron = median(g_bp["pron"]) - median(g_non["pron"])
        d_worst = median(g_bp["worst"]) - median(g_non["worst"])
        print(f"  median pronunciation:  bp_ok {median(g_bp['pron']):.1f}  "
              f"vs non_bp {median(g_non['pron']):.1f}   Δ = {d_pron:+.1f}")
        print(f"  median worst-phoneme:  bp_ok {median(g_bp['worst']):.1f}  "
              f"vs non_bp {median(g_non['worst']):.1f}   Δ = {d_worst:+.1f}")
        print(f"  non_bp clips below bp_ok's 25th-pctile pronunciation "
              f"({bp_p25:.1f}): {non_below}/{len(g_non['pron'])}")
        # best single-threshold accuracy on pronunciation score
        labelled = ([(r.score.pronunciation, 1) for r in non_rows if r.score.ok]
                    + [(r.score.pronunciation, 0) for r in bp_rows if r.score.ok])
        best_acc, best_thr = 0.0, 0.0
        for thr in sorted({p for p, _ in labelled}):
            # predict non_bp if pronunciation < thr
            correct = sum(1 for p, y in labelled
                          if (p < thr) == bool(y))
            acc = correct / len(labelled)
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        print(f"  best single pronunciation-score threshold: < {best_thr:.1f} "
              f"→ {best_acc*100:.0f}% agreement with Gemini's bp_ok/non_bp split")

    # ── disagreements ─────────────────────────────────────────────
    print()
    print("=== where Azure & Gemini disagree ===")
    if g_bp:
        thr = median(g_bp["pron"])
        gem_flag_azure_clean = [r for r in non_rows if r.score.ok
                                and r.score.pronunciation >= thr]
        azure_flag_gem_clean = [r for r in bp_rows if r.score.ok
                                and r.score.pronunciation < median(g_non["pron"])] \
            if g_non else []
        print(f"  Gemini=non_bp but Azure pron ≥ {thr:.0f} (Azure didn't flag): "
              f"{len(gem_flag_azure_clean)}")
        for r in gem_flag_azure_clean[:6]:
            print(f"    {r.rec['sense_id']} {r.rec['clip_type']:<7} "
                  f"pron={r.score.pronunciation:.0f}  gemini: "
                  f"{(r.rec.get('evidence','') or '')[:70]}")
        print(f"  Gemini=bp_ok but Azure pron low (Azure flagged what Gemini cleared): "
              f"{len(azure_flag_gem_clean)}")
        for r in azure_flag_gem_clean[:6]:
            print(f"    {r.rec['sense_id']} {r.rec['clip_type']:<7} "
                  f"pron={r.score.pronunciation:.0f} worst={r.score.worst_phoneme:.0f} "
                  f"({r.score.worst_phoneme_label})  ref={r.rec.get('pt','')[:30]!r}")

    n_err = sum(1 for r in rows if not r.score.ok)
    if n_err:
        print()
        print(f"  ({n_err} clip(s) failed Azure recognition — see ERR rows above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
