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

An outage is not a bad take: a take with no verdict (the TTS call or loudnorm
failed, or ASR / a judge didn't answer) ends its clip's ladder. The clip stays
*pending* — no further takes or voice swaps, no unresolved row — and the next
run re-checks its takes from the cache.

Accepted winners ONLY are uploaded, as
`audio/{sid}-{word|ex}-eleven_v4-v{N}.mp3` (N monotonic: R2 listing + in-run
registry; HEAD before PUT; never overwrite). An accept uploaded by a run that
stopped before flushing it is found on R2 by its md5 and reused, not uploaded
again as v(N+1). The manifest row is rewritten only on accept; unresolved
clips keep their v3 row byte-identical and are listed in data/_v4_unresolved.tsv
(current state: one row per clip, removed once the clip is accepted). en_ex
rows are asserted unchanged at exit.

Safety: one engine process at a time (build/cache/v4_engine.lock, checked
before anything else); `.bak_pre_stage19` backup before any write to the real
manifest (refuses to overwrite, and refuses a manifest already on v4); atomic
manifest writes every 25 accepts and however the run ends; budget guard
(subscription polled every 200 renders); `--dry-run` prints the projected
spend, `--yes` required. A stop drops the queued clips, lets clips in flight
finish their take, flushes, and is resumable (re-run):
  exit 3  budget: < 30K credits left, or QuotaExceeded
  exit 4  outage: TTSUnavailable / QaUnavailable, or PENDING_STREAK clips in a
          row pending
  exit 5  an unexpected worker exception (traceback printed)
  exit 6  finished, but some clips are pending: re-run to finish them
  exit 130  Ctrl-C or SIGTERM (`systemctl --user stop`)

`--no-upload` smoke runs never write data/ or reports/: the manifest goes to
--manifest-out (default tmp/19_4_smoke/_audio_manifest.tsv; the real manifest
is refused), with `.provenance.tsv`, `.unresolved.tsv` and `.report.html`
sidecars next to it. Their takes still go to the takes ledger, so the full run
reuses them for free.

Usage:
  uv run python build/19_4_render_v4.py --dry-run
  uv run python build/19_4_render_v4.py --yes --limit 20 --no-upload
  uv run python build/19_4_render_v4.py --yes
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib import stage19_data as D  # noqa: E402
from build.lib import v4_tts as T  # noqa: E402
from build.lib.audio_manifest import (  # noqa: E402
    object_key_for,
    read_manifest,
    text_hash,
    write_manifest,
)
from build.lib.tsv import append_tsv, iter_tsv, read_tsv, write_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
MANIFEST = DATA / "_audio_manifest.tsv"
BACKUP = DATA / "_audio_manifest.tsv.bak_pre_stage19"
POLICY = CONFIG / "stage19_policy.tsv"
UNRESOLVED = DATA / "_v4_unresolved.tsv"
PROVENANCE = DATA / "_audio_render_provenance.tsv"
REPORT = REPO_ROOT / "reports" / "19_4_render.html"
SMOKE_MANIFEST = REPO_ROOT / "tmp" / "19_4_smoke" / "_audio_manifest.tsv"   # on disk, not /tmp
PT_DICT_LOCATOR = {"pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
                   "version_id": "th9qzGumY1q3fkvV3Fi3"}
WORD_CAP, EXAMPLE_CAP = 8, 5
MAX_SWAPS = 2
FLUSH_EVERY = 25
POLL_EVERY = 200
MIN_CREDITS_LEFT = 30_000
PENDING_STREAK = 20          # clips in a row with a no-verdict take: a provider is down
WORKERS = 48
CREDITS_PER_CHAR = 0.133
ASR_USD_PER_TAKE = 0.0005   # gpt-4o-transcribe, plain + biased pass on a short clip
LADDER_FACTOR = 1.3         # re-takes and voice swaps on top of the first pass
EXIT_BUDGET = 3
EXIT_OUTAGE = 4
EXIT_ERROR = 5
EXIT_PENDING = 6
EXIT_INTERRUPTED = 130

UNRESOLVED_FIELDS = ["sense_id", "clip_type", "voice_id", "attempts", "reasons", "v3_kept",
                     "logged_at"]
PROVENANCE_FIELDS = ["object_key", "sense_id", "clip_type", "voice_id", "variant", "tts_input",
                     "seed", "take", "stability", "dict_on", "judge_hash", "judges", "asr_mode",
                     "selected_from", "generated_at"]


class RenderError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_4_render_v4): {msg}")


class RunStopped(RuntimeError):
    """The run is stopping (Run.halt): no new take."""


def no_verdict(t: T.TakeResult) -> bool:
    """The take was never judged on its audio: TTS or loudnorm failed (no MP3), or
    ASR / a judge didn't answer (`qa_error`: the MP3 is kept and the cache re-checks
    it). An outage, not a bad take — never a reason to buy the next rung."""
    return bool(t.error) or t.qa_error


def manifest_out_path(args) -> Path:
    """Where this run writes the manifest. A --no-upload smoke run never writes the
    real one: it defaults to SMOKE_MANIFEST and refuses MANIFEST."""
    if not args.manifest_out:
        return SMOKE_MANIFEST if args.no_upload else MANIFEST
    out = Path(args.manifest_out).resolve()
    if args.no_upload and out == MANIFEST.resolve():
        raise RenderError("--no-upload never writes the real manifest — drop --manifest-out "
                          f"(default {_rel(SMOKE_MANIFEST)}) or point it elsewhere")
    return out


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def load_policy() -> dict[str, str]:
    p = {r["key"]: r["value"] for r in read_tsv(POLICY)}
    if not p:
        raise RenderError(f"{POLICY} missing — run build/19_3_pilot.py --decide")
    if p.get("go") != "yes":
        raise RenderError(f"pilot decision go={p.get('go')!r} — not rendering")
    if p.get("gate_ok") != "yes":   # D3: the gate must not reject what the user rated Good
        raise RenderError(f"pilot decision gate_ok={p.get('gate_ok')!r} — not rendering")
    gate = T.judge_hash(T.load_judge_configs())
    if p.get("judge_hash") != gate:   # decided under other judges: re-run 19_3 --rejudge/--decide
        raise RenderError(f"pilot decision was made under judges {p.get('judge_hash')!r}, "
                          f"pinned judges are {gate!r} — re-run 19_3 --rejudge --baseline --decide")
    return p


def risky_words() -> set[str]:
    return D.risky_word_sids()


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
        self.exit_code, self.stop_reason, self.stop_exc = 0, "", None
        self.pending_streak = 0
        self.manifest_out = manifest_out_path(args)
        # Any other manifest is a smoke run: its provenance, unresolved list and report
        # are sidecars next to it, never the data/ and reports/ files.
        self.smoke = self.manifest_out.resolve() != MANIFEST.resolve()
        out, stem = self.manifest_out, self.manifest_out.stem
        self.provenance = out.with_name(f"{stem}.provenance.tsv") if self.smoke else PROVENANCE
        self.unresolved_path = out.with_name(f"{stem}.unresolved.tsv") if self.smoke else UNRESOLVED
        self.report = out.with_name(f"{stem}.report.html") if self.smoke else REPORT
        # Current state, one row per clip (files from older runs repeat a clip: last wins).
        self.unresolved_rows = {(r["sense_id"], r["clip_type"]): r
                                for r in read_tsv(self.unresolved_path)
                                if self.idx.get((r["sense_id"], r["clip_type"]),
                                                {}).get("tts_model") != T.V4}

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
            manual_pass=j.manual_pass, v3_duration_s=v3_dur, risky=j.sense_id in self.risky)

    def _swap_voices(self, j: D.ClipJob) -> list[str]:
        pool = [v for v in D.swap_pool(j.voice_gender) if v != j.voice_id]
        pool.sort(key=lambda v: self.voice_rank.get(v, 50))
        return pool[:MAX_SWAPS]

    def _render(self, spec: T.RenderSpec) -> T.TakeResult:
        if self.stop.is_set():
            raise RunStopped(self.stop_reason)
        res = self.engine.render_take(spec)
        with self.lock:
            self.renders += 1
            n = self.renders
        if n % POLL_EVERY == 0:
            self._poll_budget()
        return res

    def first_rung_specs(self, scope: list[D.ClipJob]) -> list[T.RenderSpec]:
        """The takes ladder() tries first (no v3 duration needed): what --dry-run prices."""
        out = []
        for j in scope:
            if j.clip_type == "word":
                v1, alt = self.variants[0], (self.variants[1] if len(self.variants) > 1 else None)
                out += [self._spec(j, voice=j.voice_id, variant=v1, take=t, v3_dur=None) for t in (1, 2)]
                if alt and j.sense_id in self.risky:
                    out.append(self._spec(j, voice=j.voice_id, variant=alt, take=1, v3_dur=None))
            else:
                out.append(self._spec(j, voice=j.voice_id, variant="plain", take=1, v3_dur=None))
        return out

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
                t = self._render(self._spec(j, voice=voice, variant=variant, take=take,
                                            v3_dur=v3_dur))
                tried.append(t)
                if no_verdict(t):
                    return None, tried               # pending: no next take, no voice swap
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
    def _version(self, j: D.ClipJob, md5: str) -> tuple[int, bool]:
        """(N, already on R2). N is one past every version on R2 and in this run's
        registry — unless R2 already holds these exact bytes for the clip (an accept
        uploaded by a run that stopped before flushing it): then that version, reused
        rather than uploaded again as v(N+1)."""
        short = "word" if j.clip_type == "word" else "ex"
        prefix = f"audio/{j.sense_id}-{short}-{T.V4}-v"
        n = self.versions_used.get(j.key, 0)
        if self.r2 is None:
            return n + 1, False
        etags: dict[int, str] = {}
        resp = self.r2._s3.list_objects_v2(Bucket=self.r2.config.bucket, Prefix=prefix)
        for o in resp.get("Contents", []):
            try:
                etags[int(o["Key"][len(prefix):].split(".")[0])] = o.get("ETag", "").strip('"')
            except ValueError:
                pass
        if not etags:
            return n + 1, False
        # A single-part PUT's ETag is its MD5; the md5 metadata covers any other ETag.
        same = [v for v, etag in etags.items() if etag == md5]
        newest = max(etags)
        if not same:
            head = self.r2.head(object_key_for(j.sense_id, j.clip_type, newest, T.V4)) or {}
            same = [newest] if head.get("Metadata", {}).get("md5") == md5 else []
        return (max(same), True) if same else (max(n, newest) + 1, False)

    def accept(self, j: D.ClipJob, best: T.TakeResult, tried: list[T.TakeResult]) -> None:
        mp3 = Path(best.path).read_bytes()
        if hashlib.md5(mp3).hexdigest() != best.md5:
            raise RuntimeError(f"{best.path} no longer matches the md5 it was checked with "
                               "— not uploading")
        version, on_r2 = self._version(j, best.md5)
        key = object_key_for(j.sense_id, j.clip_type, version, T.V4)
        if self.r2 is None:
            url = f"{os.environ.get('R2_PUBLIC_BASE', '').rstrip('/')}/{key}"
        elif on_r2:
            url = self.r2.public_url(key)
        else:
            if self.r2.head(key) is not None:
                raise RenderError(f"{key} already exists — refusing to overwrite")
            up = self.r2.upload_bytes(mp3, key, extra_metadata={
                "sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": best.spec.voice_id,
                "model": T.V4, "stage": "19_4", "md5": best.md5})
            url = up.url
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
            append_tsv(self.provenance, [{
                "object_key": key, "sense_id": j.sense_id, "clip_type": j.clip_type,
                "voice_id": best.spec.voice_id, "variant": best.spec.variant,
                "tts_input": best.spec.tts_text, "seed": best.spec.seed, "take": best.spec.take,
                "stability": best.spec.stability, "dict_on": int(best.spec.dict_on),
                "judge_hash": best.judge_hash,
                "judges": ";".join(f"{k}={v.get('verdict')}" for k, v in best.judges.items()),
                "asr_mode": best.asr_mode, "selected_from": len(tried),
                "generated_at": best.generated_at}], fieldnames=PROVENANCE_FIELDS)
            self.accepted += 1
            self.pending_streak = 0
            self.stats[f"accepted_rung_takes_{len(tried)}"] += 1
            self.stats[f"accepted_variant_{best.spec.variant}"] += 1
            if best.spec.voice_id != j.voice_id:
                self.stats["accepted_after_voice_swap"] += 1
            if on_r2:
                self.stats["accepted_reused_unflushed_upload"] += 1
            if self.unresolved_rows.pop(j.key, None) is not None:
                self._write_unresolved()
            if self.accepted % FLUSH_EVERY == 0:
                self._flush()

    def unresolved(self, j: D.ClipJob, tried: list[T.TakeResult]) -> None:
        reasons = collections.Counter()
        for t in tried:                      # every take has a verdict (else: pending)
            if t.sanity:
                reasons["sanity"] += 1
            elif t.loudness:
                reasons["loudness"] += 1
            elif not t.asr_pass:
                reasons["asr"] += 1
            elif not t.gate_pass:
                reasons["judge"] += 1
        with self.lock:
            self.unresolved_rows[j.key] = {
                "sense_id": j.sense_id, "clip_type": j.clip_type, "voice_id": j.voice_id,
                "attempts": len(tried), "reasons": dict(reasons), "v3_kept": "yes",
                "logged_at": datetime.now(timezone.utc).isoformat()}
            self._write_unresolved()
            self.stats["unresolved"] += 1
            self.pending_streak = 0

    def pending(self, j: D.ClipJob, tried: list[T.TakeResult]) -> None:
        """A take got no verdict: the clip keeps its v3 row and any earlier unresolved
        row, and the next run re-checks its takes from the cache. A streak of these
        means a provider is down (TTS 5xx and timeouts have no streak of their own)."""
        t = tried[-1]
        why = t.error or "; ".join(f"{k}: {v.get('error', '')}" for k, v in t.judges.items()
                                   if v.get("verdict") == "error")
        with self.lock:
            self.stats["pending"] += 1
            self.pending_streak += 1
            n = self.pending_streak
        if n >= PENDING_STREAK:
            self.halt(EXIT_OUTAGE, f"{n} clips in a row got no verdict — last "
                                   f"{j.sense_id} {j.clip_type}: {why}"[:400])

    def _write_unresolved(self) -> None:          # caller holds self.lock
        write_tsv(self.unresolved_path, self.unresolved_rows.values(),
                  fieldnames=UNRESOLVED_FIELDS)

    def flush(self) -> None:
        with self.lock:
            self._flush()

    def _flush(self) -> None:                     # caller holds self.lock
        if not self.smoke and not BACKUP.exists():
            raise RenderError(f"{BACKUP.name} missing — refusing to write {MANIFEST.name}")
        write_manifest(self.manifest, self.manifest_out)

    # ── budget / stop ───────────────────────────────────────────────────────
    def _poll_budget(self) -> None:
        try:
            s = httpx.get("https://api.elevenlabs.io/v1/user/subscription", timeout=30,
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}).json()
            left = s["character_limit"] - s["character_count"]
        except Exception:  # noqa: BLE001
            return
        with self.lock:
            self.stats["credits_left_last_poll"] = left
        if left < MIN_CREDITS_LEFT:
            self.halt(EXIT_BUDGET, f"{left:,} credits left (< {MIN_CREDITS_LEFT:,})")

    def halt(self, code: int, why: str, exc: BaseException | None = None) -> None:
        """Stop the run: no new clip, no new take. The first reason sets the exit code."""
        with self.lock:
            first = not self.exit_code
            if first:
                self.exit_code, self.stop_reason, self.stop_exc = code, why, exc
        self.stop.set()
        if first:
            print(f"  stopping (exit {code}): {why}", flush=True)

    # ── driver ──────────────────────────────────────────────────────────────
    def process(self, j: D.ClipJob) -> None:
        from build.lib.elevenlabs_client import QuotaExceeded, TTSUnavailable
        if self.stop.is_set():
            return
        try:
            best, tried = self.ladder(j)
            if best is not None:
                self.accept(j, best, tried)
            elif any(no_verdict(t) for t in tried):
                self.pending(j, tried)
            else:
                self.unresolved(j, tried)
        except RunStopped:
            pass                                       # the clip stays pending
        except QuotaExceeded as exc:
            self.halt(EXIT_BUDGET, f"ElevenLabs QuotaExceeded: {exc}")
        except (TTSUnavailable, T.QaUnavailable) as exc:
            self.halt(EXIT_OUTAGE, f"{type(exc).__name__}: {exc}")
        except BaseException as exc:  # noqa: BLE001 — R2, a bug: stop spending, keep what's done
            self.halt(EXIT_ERROR, f"{j.sense_id} {j.clip_type}: {type(exc).__name__}: {exc}", exc)


def drive(run: Run, scope: list[D.ClipJob], t0: float) -> None:
    """Every clip in `scope` through Run.process, in study order. The first stop
    (budget, outage, a worker exception, Ctrl-C / SIGTERM) drops the queued clips;
    clips in flight finish their current take. The manifest is flushed however the
    loop ends — before the wait too, so a SIGKILL during it loses no accept."""
    ex = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futs = [ex.submit(run.process, j) for j in scope]
        for i, _ in enumerate(as_completed(futs), 1):
            if run.stop.is_set():
                break
            if i % 250 == 0:
                print(f"  {i:,}/{len(scope):,} clips · accepted {run.accepted:,} · renders "
                      f"{run.renders:,} · credits {run.engine.credits_used:,} · "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        run.halt(EXIT_INTERRUPTED, "interrupted (Ctrl-C or SIGTERM)")
    finally:
        run.stop.set()
        ex.shutdown(wait=False, cancel_futures=True)     # drop the queued clips now
        run.flush()
        try:
            ex.shutdown(wait=True)                       # clips in flight finish their take
        finally:
            run.flush()


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def backup_manifest() -> None:
    """The pre-Stage-19 manifest, copied once, before the run's first write to it."""
    if BACKUP.exists():
        return
    on_v4 = sum(1 for r in iter_tsv(MANIFEST) if r["tts_model"] == T.V4)
    if on_v4:
        raise RenderError(f"{MANIFEST.name} already has {on_v4} {T.V4} rows and no "
                          f"{BACKUP.name}: it is not the pre-Stage-19 state — restore it first")
    shutil.copy2(MANIFEST, BACKUP)
    print(f"  manifest backup → {_rel(BACKUP)}")


def write_report(run: Run, scope_n: int, t0: float) -> None:
    esc = html.escape
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{v:,}</td></tr>" for k, v in sorted(run.stats.items()))
    stopped = (f"<p>Stopped (exit {run.exit_code}): {esc(run.stop_reason)} — re-run to resume.</p>"
               if run.exit_code else "")
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>v4 render</title><style>"
        ":root{--bg:#0f1115;--border:#2c3140;--text:#e6e9ef}body{background:var(--bg);"
        "color:var(--text);font:14px/1.5 -apple-system,sans-serif;padding:24px 16px;margin:0}"
        ".wrap{max-width:800px;margin:0 auto}table{border-collapse:collapse;width:100%}"
        "td,th{padding:6px 10px;border-bottom:1px solid var(--border);text-align:left}"
        "</style></head><body><div class='wrap'><h1>Stage 19 — eleven_v4 render</h1>"
        f"<p>{scope_n:,} clips in scope · {run.accepted:,} accepted · {run.renders:,} renders · "
        f"{run.engine.credits_used:,} credits (header sum) · {(time.time() - t0) / 60:.0f} min</p>"
        f"{stopped}<table><tr><th>stat</th><th>count</th></tr>{rows}</table></div></body></html>",
        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true",
                    help=f"render + QA only (smoke test; manifest → {_rel(SMOKE_MANIFEST)})")
    ap.add_argument("--manifest-out", default="",
                    help="write the manifest (+ provenance/unresolved/report sidecars) elsewhere")
    ap.add_argument("--concurrency", type=int, default=9, help="ElevenLabs parallel requests")
    args = ap.parse_args()
    manifest_out_path(args)                  # a smoke run onto the real manifest: refuse now
    try:
        T.acquire_engine_lock()              # before any other work: one engine process at a time
    except RuntimeError as exc:
        raise RenderError(str(exc)) from None
    policy = load_policy()
    run = Run(args, policy)
    scope = run.scope()
    words = sum(1 for j in scope if j.clip_type == "word")
    chars = sum(len(j.display_text) for j in scope)
    print(f"=== Stage 19.4 — eleven_v4 render === {len(scope):,} clips "
          f"({words:,} word, {len(scope) - words:,} example)")
    print(f"  policy: variants={run.variants} stability={run.stability} dict={run.dict_on} "
          f"short_words={run.short_policy}")
    rung1 = run.first_rung_specs(scope)
    cached = getattr(run.engine, "cached", {})
    new = [s for s in rung1 if cached.get(s.take_id, {}).get("tts_text") != s.tts_text]
    configs = [c for c, _ in getattr(run.engine, "judges", [])]
    calls = collections.Counter(n for s in new for n in T.judges_to_buy(configs, s, {}))
    credits = sum(len(s.tts_text) for s in new) * CREDITS_PER_CHAR
    usd = T.judge_spend(calls, configs) + len(new) * ASR_USD_PER_TAKE
    print(f"  projected first pass: {len(new):,} new takes ({len(rung1) - len(new):,} reused from "
          f"the pilot) ≈ {credits:,.0f} credits; judge calls {dict(calls)} ≈ "
          f"${T.judge_spend(calls, configs):,.0f}; ASR ({T.load_asr_model()}) ≈ "
          f"${len(new) * ASR_USD_PER_TAKE:,.0f}")
    print(f"  with ladder retries (~×{LADDER_FACTOR}): ≈ {credits * LADDER_FACTOR:,.0f} credits, "
          f"≈ ${usd * LADDER_FACTOR:,.0f} judges + ASR")
    print(f"  manifest → {_rel(run.manifest_out)}"
          + (" (smoke: provenance/unresolved/report sidecars next to it)" if run.smoke else ""))
    if args.dry_run:
        return 0
    if not args.yes:
        raise RenderError("pass --yes to render (see the projection above)")
    if not run.smoke:
        backup_manifest()
    signal.signal(signal.SIGTERM, _interrupt)    # `systemctl stop` → the Ctrl-C path
    t0 = time.time()
    run._poll_budget()
    drive(run, scope, t0)
    en_ex_after = [r for r in read_manifest(run.manifest_out) if r["clip_type"] == "en_ex"]
    if en_ex_after != run.en_ex_before:
        raise RenderError("en_ex rows changed — restore from the backup")
    write_report(run, len(scope), t0)
    print(f"  done: accepted {run.accepted:,}, unresolved {run.stats['unresolved']:,}, "
          f"pending {run.stats['pending']:,}, renders {run.renders:,}, "
          f"credits {run.engine.credits_used:,} → {_rel(run.report)}")
    if run.stop_exc is not None:
        traceback.print_exception(run.stop_exc)
    if run.exit_code:
        print(f"  STOPPED (exit {run.exit_code}): {run.stop_reason} — fix it, then re-run to "
              "resume", flush=True)
        return run.exit_code
    if run.stats["pending"]:
        print(f"  {run.stats['pending']:,} clips pending (a take got no verdict) — re-run to "
              "finish them (their takes are re-checked from the cache)", flush=True)
        return EXIT_PENDING
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
