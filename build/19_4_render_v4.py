"""Stage 19 / Step 5 — full eleven_v4 re-render of the Portuguese word + example clips.

Per clip (in Stage-17 study order — `spaced_topic_order` — so the first cards
studied finish first), a ladder of takes through the shared engine
(build/lib/v4_tts.py). Pilot takes with identical parameters are reused for
free from the takes ledger.

  words    rung 1: policy variant ×2 takes (best-of-2) + alternate variant ×1
                   for risky words
           rung 2: 2 more takes (+1 alternate for risky)
           rung 3: up to 2 same-gender voice swaps (pilot fail-rate order;
                   demoted voices last; Dani = last female option)
           cap 8 renders
  examples take 1 → take 2 → 2 voice swaps; cap 5

Acceptance (build/lib/audio_qa.py): TTS ok ∧ PCM sanity ∧ loudness ∧ relaxed
ASR on the plain display text ∧ every pinned judge bp_ok. Among passing takes:
more judge passes → production ASR pass → policy variant → ASR similarity →
LUFS closest to −16.

Accepted winners ONLY are uploaded, as
`audio/{sid}-{word|ex}-eleven_v4-v{N}.mp3` (N monotonic: R2 listing + in-run
registry; HEAD before PUT; never overwrite). The manifest row is rewritten
only on accept; unresolved clips keep their v3 row byte-identical and are
logged to data/_v4_unresolved.tsv. en_ex rows are asserted unchanged at exit.

Safety: `.bak_pre_stage19` backup (refuses to overwrite), atomic manifest
writes every 25 accepts, budget guard (subscription polled every 200
renders; stops cleanly < 30K credits or on QuotaExceeded → exit 3),
`--dry-run` prints the projected spend, `--yes` required.

Usage:
  uv run python build/19_4_render_v4.py --dry-run
  uv run python build/19_4_render_v4.py --yes --limit 20 --no-upload --manifest-out /tmp/m.tsv
  uv run python build/19_4_render_v4.py --yes
"""
from __future__ import annotations

import argparse
import collections
import html
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib import bp_ipa as B  # noqa: E402
from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    object_key_for,
    read_manifest,
    text_hash,
    write_manifest,
)
from build.lib.tsv import append_tsv, read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
MANIFEST = DATA / "_audio_manifest.tsv"
BACKUP = DATA / "_audio_manifest.tsv.bak_pre_stage19"
POLICY = CONFIG / "stage19_policy.tsv"
UNRESOLVED = DATA / "_v4_unresolved.tsv"
PROVENANCE = DATA / "_audio_render_provenance.tsv"
REPORT = REPO_ROOT / "reports" / "19_4_render.html"
PT_DICT_LOCATOR = {"pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
                   "version_id": "th9qzGumY1q3fkvV3Fi3"}
WORD_CAP, EXAMPLE_CAP = 8, 5
MAX_SWAPS = 2
FLUSH_EVERY = 25
POLL_EVERY = 200
MIN_CREDITS_LEFT = 30_000
WORKERS = 48
CREDITS_PER_CHAR = 0.133
EXIT_BUDGET = 3

UNRESOLVED_FIELDS = ["sense_id", "clip_type", "voice_id", "attempts", "reasons", "v3_kept",
                     "logged_at"]
PROVENANCE_FIELDS = ["object_key", "sense_id", "clip_type", "voice_id", "variant", "tts_input",
                     "seed", "take", "stability", "dict_on", "judge_hash", "judges", "asr_mode",
                     "selected_from", "generated_at"]


class RenderError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_4_render_v4): {msg}")


class BudgetStop(RuntimeError):
    pass


def load_policy() -> dict[str, str]:
    p = {r["key"]: r["value"] for r in read_tsv(POLICY)}
    if not p:
        raise RenderError(f"{POLICY} missing — run build/19_3_pilot.py --decide")
    if p.get("go") != "yes":
        raise RenderError(f"pilot decision go={p.get('go')!r} — not rendering")
    return p


def risky_words() -> set[str]:
    out = {r["sense_id"] for r in read_tsv(DATA / "_audio_risk_classification.tsv")
           if r["priority"] in ("P0", "P1")}
    out |= {r["sense_id"] for r in read_tsv(DATA / "_audio_mispronunciation_confirmed.tsv")}
    final = read_tsv(DATA / "06-final.tsv")
    out |= {r["sense_id"] for r in final if r["pt"].strip().lower() in B.HETEROPHONES}
    out |= {r["sense_id"] for r in read_tsv(DATA / "_ipa_v2.tsv")
            if r.get("ipa_word_status") in ("llm", "llm_context")}
    return out


class Run:
    def __init__(self, args, policy: dict[str, str]):
        from build.lib.r2_client import R2Client, R2Config

        self.args = args
        self.policy = policy
        self.jobs = D.load_jobs()
        self.manifest = read_manifest(MANIFEST)
        self.idx = {(r["sense_id"], r["clip_type"]): r for r in self.manifest}
        self.en_ex_before = [dict(r) for r in self.manifest if r["clip_type"] == "en_ex"]
        self.risky = risky_words()
        self.variants = [v for v in policy.get("word_variant_order", "plain").split(",") if v]
        self.stability = float(policy.get("stability", T.DEFAULT_STABILITY))
        self.dict_on = policy.get("dict_on") == "yes"
        order = [v for v in policy.get("voice_order", "").split(",") if v]
        demoted = set(filter(None, policy.get("voices_demoted", "").split(",")))
        self.voice_rank = {v: i + (100 if v in demoted else 0) for i, v in enumerate(order)}
        self.short_policy = policy.get("short_words", "keep_v3")
        self.engine = T.Engine(judge_configs=T.load_judge_configs(),
                               asr_model=T.load_asr_model(), el_concurrency=args.concurrency,
                               dict_locator=PT_DICT_LOCATOR if self.dict_on else None)
        self.r2 = None if args.no_upload else R2Client(R2Config.from_env())
        self.lock = threading.Lock()
        self.renders = 0
        self.accepted = 0
        self.stats = collections.Counter()
        self.versions_used: dict[tuple[str, str], int] = {}
        self.stop = threading.Event()
        self.manifest_out = Path(args.manifest_out) if args.manifest_out else MANIFEST

    # ── scope ───────────────────────────────────────────────────────────────
    def scope(self) -> list[D.ClipJob]:
        out = []
        for k, j in self.jobs.items():
            row = self.idx[k]
            if row["tts_model"] == T.V4:
                continue                                   # already accepted on v4
            if j.clip_type == "word" and len(j.display_text.split()[-1]) <= 2 \
                    and self.short_policy == "keep_v3":
                continue
            out.append(j)
        out.sort(key=lambda j: (j.order, j.sense_id, j.clip_type))
        return out[: self.args.limit] if self.args.limit else out

    # ── ladder ──────────────────────────────────────────────────────────────
    def _spec(self, j: D.ClipJob, *, voice: str, variant: str, take: int,
              v3_dur: float | None) -> T.RenderSpec:
        return T.RenderSpec(
            sense_id=j.sense_id, clip_type=j.clip_type, display_text=j.display_text,
            voice_id=voice, variant=variant, take=take, stability=self.stability,
            dict_on=self.dict_on, expected_ipa=j.expected_ipa,
            spoken_reference=j.spoken_reference, is_top_1000=j.is_top_1000,
            manual_pass=j.manual_pass, v3_duration_s=v3_dur)

    def _swap_voices(self, j: D.ClipJob) -> list[str]:
        pool = [v for v in D.swap_pool(j.voice_gender) if v != j.voice_id]
        pool.sort(key=lambda v: self.voice_rank.get(v, 50))
        return pool[:MAX_SWAPS]

    def _render(self, spec: T.RenderSpec) -> T.TakeResult:
        if self.stop.is_set():
            raise BudgetStop("stopping")
        res = self.engine.render_take(spec)
        with self.lock:
            self.renders += 1
            n = self.renders
        if n % POLL_EVERY == 0:
            self._poll_budget()
        return res

    def ladder(self, j: D.ClipJob) -> tuple[T.TakeResult | None, list[T.TakeResult]]:
        v3_dur = T.mp3_duration(j.v3_path)
        tried: list[T.TakeResult] = []
        if j.clip_type == "word":
            v1 = self.variants[0]
            alt = self.variants[1] if len(self.variants) > 1 else None
            risky = j.sense_id in self.risky
            rungs = [[(j.voice_id, v1, 1), (j.voice_id, v1, 2)]
                     + ([(j.voice_id, alt, 1)] if alt and risky else []),
                     [(j.voice_id, v1, 3), (j.voice_id, v1, 4)]
                     + ([(j.voice_id, alt, 2)] if alt and risky else [])]
            rungs += [[(v, v1, 1)] for v in self._swap_voices(j)]
            cap = WORD_CAP
        else:
            rungs = [[(j.voice_id, "plain", 1)], [(j.voice_id, "plain", 2)]]
            rungs += [[(v, "plain", 1)] for v in self._swap_voices(j)]
            cap = EXAMPLE_CAP
        for rung in rungs:
            for voice, variant, take in rung:
                if len(tried) >= cap:
                    break
                tried.append(self._render(self._spec(j, voice=voice, variant=variant, take=take,
                                                     v3_dur=v3_dur)))
            passing = [t for t in tried if t.qa_pass]
            if passing:
                if j.clip_type == "word" and len(self.jobs[j.key].display_text.split()[-1]) <= 2 \
                        and self.short_policy == "v4_if_both_takes_pass" and len(passing) < 2:
                    continue
                pref = self.variants[0]
                best = max(passing, key=lambda t: (t.score(), t.spec.variant == pref))
                return best, tried
        return None, tried

    # ── accept / upload ─────────────────────────────────────────────────────
    def _next_version(self, j: D.ClipJob) -> int:
        short = "word" if j.clip_type == "word" else "ex"
        prefix = f"audio/{j.sense_id}-{short}-{T.V4}-v"
        n = self.versions_used.get(j.key, 0)
        if self.r2 is not None:
            resp = self.r2._s3.list_objects_v2(Bucket=self.r2.config.bucket, Prefix=prefix)
            for o in resp.get("Contents", []):
                try:
                    n = max(n, int(o["Key"][len(prefix):].split(".")[0]))
                except ValueError:
                    pass
        return n + 1

    def accept(self, j: D.ClipJob, best: T.TakeResult, tried: list[T.TakeResult]) -> None:
        mp3 = Path(best.path).read_bytes()
        version = self._next_version(j)
        key = object_key_for(j.sense_id, j.clip_type, version, T.V4)
        url = ""
        if self.r2 is not None:
            if self.r2.head(key) is not None:
                raise RenderError(f"{key} already exists — refusing to overwrite")
            up = self.r2.upload_bytes(mp3, key, extra_metadata={
                "sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": best.spec.voice_id,
                "model": T.V4, "stage": "19_4"})
            url = up.url
        else:
            import os
            url = f"{os.environ.get('R2_PUBLIC_BASE', '').rstrip('/')}/{key}"
        with self.lock:
            self.versions_used[j.key] = version
            row = self.idx[j.key]
            row.update({
                "voice_gender": "female" if j.voice_gender == "female" else "male",
                "tts_provider": "elevenlabs", "tts_model": T.V4,
                "voice_id": best.spec.voice_id, "text_input": j.display_text,
                "text_hash": text_hash(j.display_text), "object_key": key, "url": url,
                "version": str(version), "md5": best.md5,
                "asr_transcript": best.asr_transcript,
                "asr_similarity": f"{best.asr_similarity:.4f}", "asr_decision": "pass",
                "applied_gain_db": f"{best.gain_db:.2f}", "final_lufs": f"{best.lufs:.2f}",
                "final_tp": f"{best.true_peak:.2f}",
                "loudness_within_tolerance": "True" if best.within_tolerance else "False",
                "tp_limited": "True" if best.tp_limited else "False",
                "generated_at": best.generated_at, "status": "uploaded",
                "notes": f"stage_19_v4:{best.spec.variant}",
            })
            append_tsv(PROVENANCE, [{
                "object_key": key, "sense_id": j.sense_id, "clip_type": j.clip_type,
                "voice_id": best.spec.voice_id, "variant": best.spec.variant,
                "tts_input": best.spec.tts_text, "seed": best.spec.seed, "take": best.spec.take,
                "stability": best.spec.stability, "dict_on": int(best.spec.dict_on),
                "judge_hash": best.judge_hash,
                "judges": ";".join(f"{k}={v.get('verdict')}" for k, v in best.judges.items()),
                "asr_mode": best.asr_mode, "selected_from": len(tried),
                "generated_at": best.generated_at}], fieldnames=PROVENANCE_FIELDS)
            self.accepted += 1
            self.stats[f"accepted_rung_takes_{len(tried)}"] += 1
            self.stats[f"accepted_variant_{best.spec.variant}"] += 1
            if best.spec.voice_id != j.voice_id:
                self.stats["accepted_after_voice_swap"] += 1
            if self.accepted % FLUSH_EVERY == 0:
                self.flush()

    def unresolved(self, j: D.ClipJob, tried: list[T.TakeResult]) -> None:
        reasons = collections.Counter()
        for t in tried:
            if t.error:
                reasons["error"] += 1
            elif t.sanity:
                reasons["sanity"] += 1
            elif t.loudness:
                reasons["loudness"] += 1
            elif not t.asr_pass:
                reasons["asr"] += 1
            elif not t.gate_pass:
                reasons["judge"] += 1
        with self.lock:
            append_tsv(UNRESOLVED, [{
                "sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": j.voice_id,
                "attempts": len(tried), "reasons": dict(reasons), "v3_kept": "yes",
                "logged_at": datetime.now(timezone.utc).isoformat()}],
                fieldnames=UNRESOLVED_FIELDS)
            self.stats["unresolved"] += 1

    def flush(self) -> None:
        write_manifest(self.manifest, self.manifest_out)

    # ── budget ──────────────────────────────────────────────────────────────
    def _poll_budget(self) -> None:
        import os
        try:
            s = httpx.get("https://api.elevenlabs.io/v1/user/subscription", timeout=30,
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}).json()
            left = s["character_limit"] - s["character_count"]
        except Exception:  # noqa: BLE001
            return
        with self.lock:
            self.stats["credits_left_last_poll"] = left
        if left < MIN_CREDITS_LEFT:
            self.stop.set()

    # ── driver ──────────────────────────────────────────────────────────────
    def process(self, j: D.ClipJob) -> None:
        from build.lib.elevenlabs_client import QuotaExceeded
        if self.stop.is_set():
            return
        try:
            best, tried = self.ladder(j)
        except (BudgetStop, QuotaExceeded):
            self.stop.set()
            return
        if best is not None:
            self.accept(j, best, tried)
        else:
            self.unresolved(j, tried)


def backup_manifest() -> None:
    if BACKUP.exists():
        return
    shutil.copy2(MANIFEST, BACKUP)
    print(f"  manifest backup → {BACKUP.relative_to(REPO_ROOT)}")


def write_report(run: Run, scope_n: int, t0: float) -> None:
    esc = html.escape
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{v:,}</td></tr>" for k, v in sorted(run.stats.items()))
    REPORT.write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>v4 render</title><style>"
        ":root{--bg:#0f1115;--border:#2c3140;--text:#e6e9ef}body{background:var(--bg);"
        "color:var(--text);font:14px/1.5 -apple-system,sans-serif;padding:24px 16px;margin:0}"
        ".wrap{max-width:800px;margin:0 auto}table{border-collapse:collapse;width:100%}"
        "td,th{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left}"
        "</style></head><body><div class='wrap'><h1>Stage 19 — eleven_v4 render</h1>"
        f"<p>{scope_n:,} clips in scope · {run.accepted:,} accepted · {run.renders:,} renders · "
        f"{run.engine.credits_used:,} credits (header sum) · {(time.time() - t0) / 60:.0f} min</p>"
        f"<table><tr><th>stat</th><th>count</th></tr>{rows}</table></div></body></html>",
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true", help="render + QA only (smoke test)")
    ap.add_argument("--manifest-out", default="", help="write the manifest elsewhere (smoke test)")
    ap.add_argument("--concurrency", type=int, default=9, help="ElevenLabs parallel requests")
    args = ap.parse_args()
    policy = load_policy()
    run = Run(args, policy)
    scope = run.scope()
    words = sum(1 for j in scope if j.clip_type == "word")
    chars = sum(len(j.display_text) for j in scope)
    print(f"=== Stage 19.4 — eleven_v4 render === {len(scope):,} clips "
          f"({words:,} word, {len(scope) - words:,} example)")
    print(f"  policy: variants={run.variants} stability={run.stability} dict={run.dict_on} "
          f"short_words={run.short_policy}")
    print(f"  projected: first pass ~{chars * 2 * CREDITS_PER_CHAR:,.0f} credits "
          f"(words best-of-2; ×1.3 with retries); judges "
          f"{[c['config'] for c in run.engine.judge_configs]}; ASR {T.load_asr_model()}")
    if args.dry_run:
        return 0
    if not args.yes:
        raise RenderError("pass --yes to render (see the projection above)")
    if run.manifest_out == MANIFEST and not args.no_upload:
        backup_manifest()
    t0 = time.time()
    run._poll_budget()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run.process, j) for j in scope]
        for i, f in enumerate(futs, 1):
            f.result()
            if i % 250 == 0:
                print(f"  {i:,}/{len(scope):,} clips · accepted {run.accepted:,} · renders "
                      f"{run.renders:,} · credits {run.engine.credits_used:,} · "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    run.flush()
    en_ex_after = [r for r in read_manifest(run.manifest_out) if r["clip_type"] == "en_ex"]
    if en_ex_after != run.en_ex_before:
        raise RenderError("en_ex rows changed — restore from the backup")
    write_report(run, len(scope), t0)
    print(f"  done: accepted {run.accepted:,}, unresolved {run.stats['unresolved']:,}, "
          f"renders {run.renders:,}, credits {run.engine.credits_used:,} → "
          f"{REPORT.relative_to(REPO_ROOT)}")
    return EXIT_BUDGET if run.stop.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
