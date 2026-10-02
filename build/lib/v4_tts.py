"""Stage 19 — eleven_v4 render engine shared by the pilot (19_3) and the full run (19_4).

A *take* is one TTS render of one clip with one (variant, voice, take#,
stability, dictionary) combination. `Engine.render_take` runs:

    TTS (eleven_v4, pcm_44100) → PCM sanity → closed-loop loudnorm → MP3
    cached under build/audio_cache/v4_takes/ → relaxed ASR on the PLAIN display
    text → every pinned judge → TakeResult, appended to data/_v4_takes.tsv

Nothing is uploaded and the manifest is never touched here — 19_4 uploads
accepted winners only. Takes are cached by `take_id`, so re-runs resume for
free; if the judge configuration changed, cached MP3s are re-judged instead
of re-rendered. A take whose ASR or judge call failed keeps its MP3 and is
re-checked from it; a take that failed PCM sanity (no MP3) is re-rendered only
if today's sanity rules would pass it.

Outages stop the run instead of failing takes (a failed take sends 19_4 up its
ladder of paid renders): `QuotaExceeded` / `TTSUnavailable` from the TTS
client, and `QaUnavailable` (`AsrUnavailable`, `JudgeUnavailable`) when a QA
provider refuses on billing or auth (HTTP 401/402/403, quota or credits
depleted) or errors ASR_ERROR_STREAK / JUDGE_ERROR_STREAK times in a row. A
QaUnavailable is raised after the take's row is appended, and from then on the
engine renders no new paid take. Callers stop, flush, and resume later.

One engine process at a time: `Engine()` holds an exclusive flock on
build/cache/v4_engine.lock, so two runs can't buy the same take twice or
overwrite each other's MP3s.

Variants (TTS input; ASR and judges always see the display text):
    plain   the orthographic display text (IPA was rejected by the Step-1 probe)
    tag     a leading delivery tag, e.g. "[sotaque paulistano] a cara"
"""
from __future__ import annotations

import fcntl
import collections
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from array import array
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from build.lib import audio_qa as QA
from build.lib.tsv import append_tsv, read_tsv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "data"
TAKES_DIR = REPO_ROOT / "build" / "audio_cache" / "v4_takes"
LEDGER = DATA / "_v4_takes.tsv"
PINNED_MODELS = REPO_ROOT / "config" / "stage19_models.tsv"
ANKI_MEDIA = DATA / "anki_media"
JUDGE_AUDIT = REPO_ROOT / "audit" / "19_judges.jsonl"
ENGINE_LOCK = REPO_ROOT / "build" / "cache" / "v4_engine.lock"

V4 = "eleven_v4"
DEFAULT_STABILITY = 0.65
SIMILARITY = 0.80
DEFAULT_TAG = "[sotaque paulistano]"
SHORT_CLIP = {"word": "word", "example": "ex"}

# Consecutive errors (across threads) after which a QA provider counts as down.
ASR_ERROR_STREAK = 20
JUDGE_ERROR_STREAK = 20      # per judge: one dead judge fails every gate alone
# Answers that mean "refused until a human acts", not "try again later".
UNAVAILABLE_HTTP = {401, 402, 403}
# Not "quota" / "billing": Gemini's per-minute 429 also says "You exceeded your
# current quota, please check your plan and billing details".
UNAVAILABLE_MARKERS = ("insufficient_quota", "credits are depleted", "billing_not_active",
                       "api_key_invalid", "api key not valid", "invalid_api_key")

LEDGER_FIELDS = [
    "take_id", "sense_id", "clip_type", "variant", "voice_id", "take", "seed", "stability",
    "dict_on", "tts_text", "display_text", "path", "md5", "duration_s", "lufs", "true_peak",
    "within_tolerance", "tp_limited", "gain_db", "sanity", "loudness", "asr_transcript",
    "asr_similarity", "asr_mode", "asr_pass", "judges", "judge_hash", "gate_pass", "qa_pass",
    "credits", "latency_ms", "error", "generated_at",
]


def take_seed(sense_id: str, clip_type: str, variant: str, voice_id: str, take: int) -> int:
    payload = f"{sense_id}|{clip_type}|{V4}|{variant}|{voice_id}|t{take}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % 4_294_967_295


@dataclass
class RenderSpec:
    sense_id: str
    clip_type: str            # "word" | "example"
    display_text: str         # orthographic text: ASR reference + judge "Word:"
    voice_id: str
    variant: str = "plain"
    take: int = 1
    stability: float = DEFAULT_STABILITY
    dict_on: bool = False
    tag: str = DEFAULT_TAG
    expected_ipa: str = ""    # for J2 judges
    spoken_reference: str = ""
    is_top_1000: bool = False
    manual_pass: bool = False
    v3_duration_s: float | None = None
    risky: bool = False       # a risky word: judges scoped 'risky_word' also run

    @property
    def tts_text(self) -> str:
        if self.variant == "tag":
            return f"{self.tag} {self.display_text}"
        return self.display_text

    @property
    def seed(self) -> int:
        return take_seed(self.sense_id, self.clip_type, self.variant, self.voice_id, self.take)

    @property
    def take_id(self) -> str:
        return (f"{self.sense_id}|{self.clip_type}|{self.variant}|{self.voice_id}|t{self.take}"
                f"|s{self.stability:g}|d{int(self.dict_on)}")

    @property
    def path(self) -> Path:
        return TAKES_DIR / (f"{self.sense_id}-{SHORT_CLIP[self.clip_type]}-{self.variant}-"
                            f"{self.voice_id[:6]}-t{self.take}-s{self.stability:g}"
                            f"{'-d' if self.dict_on else ''}.mp3")


@dataclass
class TakeResult:
    spec: RenderSpec
    path: str = ""
    md5: str = ""
    duration_s: float = 0.0
    lufs: float = float("nan")
    true_peak: float = float("nan")
    within_tolerance: bool = False
    tp_limited: bool = False
    gain_db: float = 0.0
    sanity: list = field(default_factory=list)
    loudness: list = field(default_factory=list)
    asr_transcript: str = ""
    asr_similarity: float = 0.0
    asr_mode: str = ""
    asr_pass: bool = False
    judges: dict = field(default_factory=dict)
    judge_hash: str = ""
    gate_pass: bool = False
    credits: int = 0
    latency_ms: int = 0
    error: str = ""
    generated_at: str = ""

    @property
    def qa_pass(self) -> bool:
        return self.pre_judge_pass and self.gate_pass

    @property
    def pre_judge_pass(self) -> bool:
        """Everything except the judges (used while judges are still pending)."""
        return not self.error and not self.sanity and not self.loudness and self.asr_pass

    @property
    def judge_error(self) -> bool:
        """A judge gave no verdict (429, 5xx, 402…); the cache re-asks it."""
        return any(v.get("verdict") == "error" for v in self.judges.values())

    @property
    def qa_error(self) -> bool:
        """ASR or a judge failed to answer: the take is unchecked, not failed. Re-check
        it later (the cache does, from its MP3) rather than render another take."""
        return self.judge_error or self.error.startswith("asr:")

    def score(self) -> tuple:
        """Higher is better: judge passes, raw ASR, similarity, loudness fit."""
        passes = sum(1 for v in self.judges.values() if v.get("verdict") == "bp_ok")
        return (int(self.qa_pass), passes, int(self.asr_mode == "production"),
                round(self.asr_similarity, 3), -abs((self.lufs if self.lufs == self.lufs else -99) + 16))

    def ledger_row(self) -> dict:
        s = self.spec
        return {
            "take_id": s.take_id, "sense_id": s.sense_id, "clip_type": s.clip_type,
            "variant": s.variant, "voice_id": s.voice_id, "take": s.take, "seed": s.seed,
            "stability": s.stability, "dict_on": int(s.dict_on), "tts_text": s.tts_text,
            "display_text": s.display_text, "path": self.path, "md5": self.md5,
            "duration_s": round(self.duration_s, 3), "lufs": round(self.lufs, 2),
            "true_peak": round(self.true_peak, 2), "within_tolerance": int(self.within_tolerance),
            "tp_limited": int(self.tp_limited), "gain_db": round(self.gain_db, 2),
            "sanity": "|".join(self.sanity), "loudness": "|".join(self.loudness),
            "asr_transcript": self.asr_transcript, "asr_similarity": round(self.asr_similarity, 4),
            "asr_mode": self.asr_mode, "asr_pass": int(self.asr_pass),
            "judges": json.dumps(self.judges, ensure_ascii=False), "judge_hash": self.judge_hash,
            "gate_pass": int(self.gate_pass), "qa_pass": int(self.qa_pass),
            "credits": self.credits, "latency_ms": self.latency_ms, "error": self.error,
            "generated_at": self.generated_at,
        }


def load_judge_configs(path: Path = PINNED_MODELS, role: str = "judge",
                       allow_empty: bool = False) -> list[dict]:
    rows = [r for r in read_tsv(path) if r["role"] == role]
    if not rows and not allow_empty:
        raise RuntimeError(f"no '{role}' rows in {path} — run build/19_2_model_selection.py")
    bad = [r["config"] for r in rows if judge_scope(r) not in JUDGE_SCOPES]
    if bad:   # an unknown scope would silently never judge, failing every clip
        raise RuntimeError(f"unknown judge scope for {bad} in {path}; use one of {JUDGE_SCOPES}")
    if rows and not any(judge_scope(r) == "all" for r in rows):
        raise RuntimeError(f"no judge with scope 'all' in {path}: example clips would have none")
    return rows


def load_asr_model(path: Path = PINNED_MODELS) -> str:
    rows = [r for r in read_tsv(path) if r["role"] == "asr"]
    return rows[0]["model"] if rows else "gpt-4o-transcribe"


JUDGE_SCOPES = ("all", "risky_word")


def judge_scope(c: dict) -> str:
    return c.get("scope") or "all"


def judge_applies(c: dict, spec: RenderSpec) -> bool:
    """`all` judges every clip; `risky_word` only the word clips of risky senses."""
    scope = judge_scope(c)
    return scope == "all" or (scope == "risky_word" and spec.risky and spec.clip_type == "word")


# Verdicts bought before verdicts recorded their judge (the Stage-19 pilot): the audit
# log (audit/19_judges.jsonl) shows each of these names came from exactly this judge.
LEGACY_JUDGE_IDS = {"pro-J2": ("gemini-3.1-pro-preview", "J2"),
                    "flash38-J2": ("gemini-3.8-flash", "J2")}


def judge_context(c: dict, spec: RenderSpec) -> str:
    """What the judge was shown besides the audio: the text, and the IPA for J2."""
    ipa = spec.expected_ipa if c["prompt_version"] == "J2" else ""
    return hashlib.sha256(f"{spec.display_text}|{ipa}".encode()).hexdigest()[:10]


def reusable_verdict(c: dict, prior: dict | None, spec: RenderSpec | None = None) -> bool:
    """A stored verdict counts for config `c` only if it is a real verdict from the
    same model and prompt (recorded on the verdict, or a known legacy name), given
    the same text and IPA (recorded as `ctx`; legacy verdicts predate it and were
    checked against audit/19_judges.jsonl)."""
    if not prior or prior.get("verdict") not in ("bp_ok", "non_bp", "unclear"):
        return False
    ident = ((prior["model"], prior.get("prompt_version")) if prior.get("model")
             else LEGACY_JUDGE_IDS.get(c["config"]))
    if ident != (c["model"], c["prompt_version"]):
        return False
    return not (spec is not None and prior.get("ctx") and prior["ctx"] != judge_context(c, spec))


# USD per judge call, measured from audit/19_judges.jsonl (thinking tokens included).
JUDGE_USD = {"gemini-3.8-flash": 0.00125, "gemini-3.1-pro-preview": 0.0085}


def judge_spend(calls: "collections.Counter[str]", configs: list[dict]) -> float:
    model = {c["config"]: c["model"] for c in configs}
    return sum(n * JUDGE_USD.get(model.get(name, ""), 0.01) for name, n in calls.items())


def judges_to_buy(configs: list[dict], spec: RenderSpec, prior: dict) -> list[str]:
    """Names of the judges that apply to `spec` and have no reusable verdict in `prior`."""
    return [c["config"] for c in configs
            if judge_applies(c, spec) and not reusable_verdict(c, prior.get(c["config"]), spec)]


def judge_hash(configs: list[dict]) -> str:
    sig = ";".join(f"{c['backend']}:{c['model']}:{c['prompt_version']}"
                   + (f":{judge_scope(c)}" if judge_scope(c) != "all" else "") for c in configs)
    return hashlib.sha256(sig.encode()).hexdigest()[:10]


def local_take_path(path: str) -> str:
    """A ledger `path` resolved in this checkout. Paths are stored absolute and the
    pilot began on another machine (Mac → VPS); every take sits in TAKES_DIR under
    a unique name, so the basename is enough."""
    return str(TAKES_DIR / Path(path).name) if path else ""


def read_ledger(path: Path | None = None) -> list[dict]:
    """The takes ledger with every `path` resolved by `local_take_path`."""
    rows = read_tsv(path or LEDGER)
    for r in rows:
        r["path"] = local_take_path(r["path"])
    return rows


def sanity_now(row: dict, spec: RenderSpec) -> list[str]:
    """A ledger row's PCM-sanity issues re-checked under today's audio_qa rules and
    this spec (clip type, v3 duration). A failed take saved no audio, so duration
    rules run through QA.pcm_sanity on a constant tone of the stored length (no copy
    of its thresholds here), and edge silences are re-compared from the stored
    values. Anything else (empty or silent audio) still fails."""
    issues = []
    for i in filter(None, row["sanity"].split("|")):
        if i.startswith("dur:"):
            continue                                   # re-derived below
        if i.startswith("edge:"):                      # edge:lead_0.85s / edge:trail_0.91s
            try:
                if float(i.rsplit("_", 1)[1].rstrip("s")) <= QA.MAX_EDGE_SILENCE_S:
                    continue
            except ValueError:
                pass
        issues.append(i)
    n = round(float(row.get("duration_s") or 0) * QA.SAMPLE_RATE)
    tone = (array("h", [8000]) * n).tobytes()          # loud throughout: no edge issues
    issues += [i for i in QA.pcm_sanity(tone, spec.clip_type, spec.v3_duration_s).issues
               if i.startswith("dur:")]
    return issues


class QaUnavailable(RuntimeError):
    """A QA provider refuses every call (HTTP 401/402/403, quota or credits depleted)
    or has errored its streak limit in a row. Raised after the take's row is
    appended. Stop the run like QuotaExceeded and resume once the provider is back:
    a QA outage must never buy another take."""


class AsrUnavailable(QaUnavailable):
    """The ASR provider is down."""


class JudgeUnavailable(QaUnavailable):
    """A pinned judge is down."""


def unavailable_reason(exc: BaseException) -> str:
    """Why `exc` means its provider refuses every call until a human acts (billing,
    key), or '' when it may be transient."""
    resp = getattr(exc, "response", None)
    for status in (getattr(exc, "status_code", None), getattr(exc, "code", None),
                   getattr(resp, "status_code", None)):
        if isinstance(status, int) and status in UNAVAILABLE_HTTP:
            return f"HTTP {status}"
    text = f"{exc} {getattr(exc, 'code', '')}".lower()
    return next((m for m in UNAVAILABLE_MARKERS if m in text), "")


class QaHealth:
    """Error streaks of the QA providers (ASR, each judge), shared by every worker
    thread. A provider is down after a billing/auth refusal or `limit` errors in a
    row (any success resets its streak). Once one is down the engine renders no new
    paid take in this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streak: dict[str, int] = {}
        self._down: tuple[type[QaUnavailable], str] | None = None

    def ok(self, key: str) -> None:
        with self._lock:
            self._streak[key] = 0

    def error(self, key: str, exc: Exception, limit: int,
              cls: type[QaUnavailable]) -> QaUnavailable | None:
        """Count one error of `key`; the exception to raise if it is now down."""
        why = unavailable_reason(exc)
        with self._lock:
            n = self._streak[key] = self._streak.get(key, 0) + 1
            if not why and n < limit:
                return None
            msg = f"{key}: {why or f'{n} errors in a row'} — {type(exc).__name__}: {exc}"[:400]
            self._down = (cls, msg)
        return cls(msg)

    def check(self) -> None:
        """Raise if a QA provider is down (before paying for a render)."""
        with self._lock:
            down = self._down
        if down:
            raise down[0](f"{down[1]} (no new renders)")


_engine_lock_fd: int | None = None
_engine_lock_guard = threading.Lock()


def acquire_engine_lock() -> None:
    """Hold ENGINE_LOCK until this process exits, or raise if another process holds it.

    Two engines at once (19_3 --render and 19_4, or a 19_4 relaunched under another
    unit name) would both pay for the same uncached takes, overwrite each other's
    MP3s, and leave ledger md5s that match the other run's bytes. The kernel drops
    the flock however the process ends; a second Engine in the same process reuses it."""
    global _engine_lock_fd
    with _engine_lock_guard:
        if _engine_lock_fd is not None:
            return
        ENGINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(ENGINE_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 300).decode("utf-8", "replace").strip()
            os.close(fd)
            raise RuntimeError(
                f"another Stage 19 render engine holds {ENGINE_LOCK} ({holder or 'holder unknown'})"
                " — check `systemctl --user list-units 'anki-*'`; never run two at once") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}: {' '.join(sys.argv)}\n".encode())
        _engine_lock_fd = fd


def mp3_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) if out else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


class Engine:
    """Thread-safe take renderer. One instance per run, one engine process at a time."""

    def __init__(self, *, judge_configs: list[dict], asr_model: str, el_concurrency: int = 9,
                 gemini_rpm: int = 900, dict_locator: dict | None = None) -> None:
        acquire_engine_lock()
        from build.lib.asr import AsrClient
        from build.lib.audio_judge import AudioJudgeClient
        from build.lib.elevenlabs_client import ElevenLabsClient
        from build.lib.gemini_audio_judge import GeminiAudioJudgeClient
        from build.lib.rate_limit import SlidingWindowRateLimiter

        self.el = ElevenLabsClient(model_id=V4, language_code="pt")
        self.el_dict = (ElevenLabsClient(model_id=V4, language_code="pt",
                                         pronunciation_dict_locators=[dict_locator])
                        if dict_locator else None)
        self.asr = AsrClient(model=asr_model)
        self.judge_configs = judge_configs
        self.judge_hash = judge_hash(judge_configs)
        self.judges = []
        for c in judge_configs:
            cls = GeminiAudioJudgeClient if c["backend"] == "gemini" else AudioJudgeClient
            self.judges.append((c, cls(model=c["model"], prompt_version=c["prompt_version"],
                                       audit_path=JUDGE_AUDIT)))
        self.el_sem = threading.BoundedSemaphore(el_concurrency)
        self.asr_sem = threading.BoundedSemaphore(16)
        self.openai_judge_sem = threading.BoundedSemaphore(8)
        self.gemini_limiter = SlidingWindowRateLimiter(max_rpm=gemini_rpm)
        self.ledger_lock = threading.Lock()
        self.qa_health = QaHealth()
        self.credits_used = 0
        self.cached = self._load_ledger()
        TAKES_DIR.mkdir(parents=True, exist_ok=True)

    # ── ledger ──────────────────────────────────────────────────────────────
    def _load_ledger(self) -> dict[str, dict]:
        """Latest row per take (re-judged takes append). Also pools every real verdict
        ever bought for each (take, audio md5), so a gate change can reuse verdicts
        that a later row of a narrower gate no longer carries."""
        out: dict[str, dict] = {}
        self.seen_verdicts: dict[tuple[str, str], dict] = {}
        for r in read_ledger():
            out[r["take_id"]] = r          # last write wins
            self._pool_verdicts(r)
        return out

    def _pool_verdicts(self, row: dict) -> None:
        pool = getattr(self, "seen_verdicts", None)
        if pool is None or not row.get("md5"):
            return
        judges = row["judges"] if isinstance(row["judges"], dict) else json.loads(row["judges"] or "{}")
        seen = pool.setdefault((row["take_id"], row["md5"]), {})
        seen.update({k: v for k, v in judges.items() if v.get("verdict") != "error"})

    def _append(self, res: TakeResult) -> None:
        row = {k: "" if v is None else str(v) for k, v in res.ledger_row().items()}  # as stored
        with self.ledger_lock:
            append_tsv(LEDGER, [row], fieldnames=LEDGER_FIELDS)
            self.cached[res.spec.take_id] = row
            self._pool_verdicts(row)

    def cached_result(self, spec: RenderSpec) -> TakeResult | None:
        row = self.cached.get(spec.take_id)
        if not row or row.get("tts_text") != spec.tts_text:   # new, or the text to speak changed
            return None
        if row.get("sanity") and not row.get("path"):
            return self._cached_sanity_fail(spec, row)
        # A TTS or loudnorm error saved no MP3 (loudnorm fails before the write), so
        # those takes are re-rendered. An ASR error kept its MP3: only ASR is re-run.
        error = row.get("error") or ""
        if ((error and not error.startswith("asr:")) or not row.get("path")
                or not Path(row["path"]).is_file()):
            return None
        mp3 = Path(row["path"]).read_bytes()
        # The MP3 must be the one the row describes (its ASR/judge verdicts and the md5
        # 19_4 uploads with); a file overwritten outside the engine is re-rendered.
        if row.get("md5") and hashlib.md5(mp3).hexdigest() != row["md5"]:
            return None
        res = _result_from_row(spec, row)
        if res.error:                                   # asr: — transcribe the saved MP3
            res.error = ""
            return self._qa(res, mp3)
        changed = False
        # Re-check a stored ASR failure under the current relaxed rules (they
        # only get more accurate; the transcript itself is reused, not re-bought).
        if not res.asr_pass and res.asr_transcript and not res.sanity and not res.loudness:
            ok, mode, sim = QA.relaxed_asr_pass(
                transcripts=[res.asr_transcript], reference=spec.display_text,
                clip_type=spec.clip_type, is_top_1000=spec.is_top_1000,
                production_decision="regen", spoken_reference=spec.spoken_reference,
                manual_pass=spec.manual_pass)
            if ok:
                res.asr_pass, res.asr_mode = True, f"recheck:{mode}"
                changed = True
        # A judge that errored (429, 5xx, 402…) gave no verdict: ask again rather than
        # keep a transient failure as a permanent gate fail.
        applies = {c["config"] for c, _ in self.judges if judge_applies(c, spec)}
        if (self.judges and (changed or res.judge_error or row.get("judge_hash") != self.judge_hash
                             or set(res.judges) != applies)
                and not res.sanity and not res.loudness and res.asr_pass):
            return self._qa(res, mp3, asr=False)
        if changed:
            self._append(res)
        return res

    def _cached_sanity_fail(self, spec: RenderSpec, row: dict) -> TakeResult | None:
        """A take that failed PCM sanity saved no MP3. Re-render it only if today's rules
        and spec would pass its stored duration; otherwise the failure stands, instead
        of being paid for again on every run (the ladder's next take is the retry)."""
        issues = sanity_now(row, spec)
        if not issues:
            return None
        res = _result_from_row(spec, row)
        if issues != res.sanity:
            res.sanity = issues
            self._append(res)
        return res

    # ── one take ────────────────────────────────────────────────────────────
    def render_take(self, spec: RenderSpec) -> TakeResult:
        hit = self.cached_result(spec)
        if hit is not None:
            return hit
        from elevenlabs import VoiceSettings

        from build.lib.elevenlabs_client import QuotaExceeded, TTSUnavailable
        from build.lib.loudness import normalize_pcm_to_mp3_verified

        self.qa_health.check()            # a QA outage: no take could be checked, buy none
        res = TakeResult(spec=spec, judge_hash=self.judge_hash,
                         generated_at=datetime.now(timezone.utc).isoformat())
        client = self.el_dict if spec.dict_on and self.el_dict else self.el
        try:
            with self.el_sem:
                tts = client.generate_pcm_meta(
                    text=spec.tts_text, voice_id=spec.voice_id, sense_id=spec.sense_id,
                    clip_type=spec.clip_type, version=0, seed=spec.seed,
                    voice_settings=VoiceSettings(stability=spec.stability,
                                                 similarity_boost=SIMILARITY))
            res.credits = tts.character_cost or 0
            res.latency_ms = tts.latency_ms
            with self.ledger_lock:
                self.credits_used += res.credits
        except (QuotaExceeded, TTSUnavailable):     # account-wide: stop the run, no row
            raise
        except Exception as exc:  # noqa: BLE001
            res.error = f"tts:{type(exc).__name__}: {exc}"[:300]
            self._append(res)
            return res
        san = QA.pcm_sanity(tts.audio_pcm, spec.clip_type, spec.v3_duration_s)
        res.duration_s = san.duration_s
        res.sanity = san.issues
        if not san.ok:
            self._append(res)
            return res
        try:
            norm = normalize_pcm_to_mp3_verified(tts.audio_pcm)
        except Exception as exc:  # noqa: BLE001
            res.error = f"loudnorm:{type(exc).__name__}: {exc}"[:300]
            self._append(res)
            return res
        res.lufs, res.true_peak = norm.final_mp3_lufs, norm.final_mp3_tp
        res.within_tolerance, res.tp_limited = norm.within_tolerance, norm.tp_limited
        res.gain_db = norm.applied_gain_db
        ok, res.loudness = QA.loudness_ok(norm.final_mp3_lufs, norm.final_mp3_tp)
        spec.path.write_bytes(norm.mp3_bytes)
        res.path = str(spec.path)
        res.md5 = hashlib.md5(norm.mp3_bytes).hexdigest()
        if not ok:
            self._append(res)
            return res
        return self._qa(res, norm.mp3_bytes)

    def _qa(self, res: TakeResult, mp3: bytes, *, asr: bool = True) -> TakeResult:
        """ASR (unless `asr` is False: the stored transcript stands), then every judge,
        then append the row. A QaUnavailable is re-raised after the append, so the
        saved MP3 is re-checked, not re-bought, when the run resumes."""
        try:
            if asr:
                self._asr(res, mp3)
            if res.asr_pass and self.judges:
                self._judge(res, mp3)
        except QaUnavailable:
            self._append(res)
            raise
        self._append(res)
        return res

    def _asr(self, res: TakeResult, mp3: bytes) -> None:
        from build.lib.asr import asr_roundtrip

        spec = res.spec
        try:
            with self.asr_sem:
                a = asr_roundtrip(asr=self.asr, mp3_bytes=mp3,
                                  input_text=spec.display_text, clip_type=spec.clip_type,
                                  sense_id=spec.sense_id, is_top_1000=spec.is_top_1000)
            transcripts = [a.transcript, a.biased_transcript]
            res.asr_transcript = a.transcript
            res.asr_pass, res.asr_mode, res.asr_similarity = QA.relaxed_asr_pass(
                transcripts=transcripts, reference=spec.display_text, clip_type=spec.clip_type,
                is_top_1000=spec.is_top_1000, production_decision=a.decision,
                spoken_reference=spec.spoken_reference, manual_pass=spec.manual_pass)
        except Exception as exc:  # noqa: BLE001
            # The row keeps its path and md5: cached_result re-runs the ASR on the MP3.
            res.error = f"asr:{type(exc).__name__}: {exc}"[:300]
            down = self.qa_health.error("asr", exc, ASR_ERROR_STREAK, AsrUnavailable)
            if down:
                raise down from exc
            return
        self.qa_health.ok("asr")

    def prior_verdicts(self, res: TakeResult) -> dict:
        """Every real verdict bought for this take's audio (any ledger row with the same
        take_id and md5), overlaid with those already on `res`."""
        pool = getattr(self, "seen_verdicts", None) or {}
        seen = pool.get((res.spec.take_id, res.md5), {}) if res.md5 else {}
        return {**seen, **(res.judges or {})}

    def missing_judges(self, res: TakeResult) -> list[str]:
        return judges_to_buy([c for c, _ in self.judges], res.spec, self.prior_verdicts(res))

    def _judge(self, res: TakeResult, mp3: bytes) -> None:
        """The verdict of every judge that applies to this clip (see judge_applies),
        recorded on `res`. A verdict already on `res` from the same model and prompt
        (reusable_verdict) is kept, so a gate change only buys the missing judges. Then raises JudgeUnavailable if a judge refused on billing/auth or has
        errored JUDGE_ERROR_STREAK times in a row."""
        spec = res.spec
        verdicts = {}
        down = []
        prior = self.prior_verdicts(res)    # verdicts already bought for this same audio
        for c, client in self.judges:
            name = c["config"]
            if not judge_applies(c, spec):
                continue
            if reusable_verdict(c, prior.get(name), spec):
                verdicts[name] = prior[name]   # same audio, model, prompt, text and IPA
                continue
            ipa = spec.expected_ipa if c["prompt_version"] == "J2" else ""
            try:
                if c["backend"] == "gemini":
                    self.gemini_limiter.wait_before_call()
                    r = client.judge(audio_bytes=mp3, pt=spec.display_text, ipa_word_final=ipa,
                                     voice_id=spec.voice_id, sense_id=spec.sense_id,
                                     clip_type=spec.clip_type)
                else:
                    with self.openai_judge_sem:
                        r = client.judge(audio_bytes=mp3, pt=spec.display_text,
                                         ipa_word_final=ipa, voice_id=spec.voice_id,
                                         sense_id=spec.sense_id, clip_type=spec.clip_type)
                verdicts[name] = {"verdict": r.pronunciation_verdict, "drift": r.drift,
                                  "severity": r.severity, "evidence": r.evidence[:160],
                                  "model": c["model"], "prompt_version": c["prompt_version"],
                                  "ctx": judge_context(c, spec)}
            except Exception as exc:  # noqa: BLE001
                verdicts[name] = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}"[:160]}
                outage = self.qa_health.error(f"judge:{name}", exc, JUDGE_ERROR_STREAK,
                                              JudgeUnavailable)
                if outage:
                    down.append(str(outage))
            else:
                self.qa_health.ok(f"judge:{name}")
        res.judges = verdicts
        res.judge_hash = self.judge_hash
        res.gate_pass = QA.gate_pass({k: v["verdict"] for k, v in verdicts.items()})
        if down:
            raise JudgeUnavailable("; ".join(down))


def _result_from_row(spec: RenderSpec, r: dict) -> TakeResult:
    def f(x, d=float("nan")):
        try:
            return float(x)
        except (TypeError, ValueError):
            return d
    return TakeResult(
        spec=spec, path=r["path"], md5=r["md5"], duration_s=f(r["duration_s"], 0.0),
        lufs=f(r["lufs"]), true_peak=f(r["true_peak"]),
        within_tolerance=r["within_tolerance"] == "1", tp_limited=r["tp_limited"] == "1",
        gain_db=f(r["gain_db"], 0.0), sanity=[x for x in r["sanity"].split("|") if x],
        loudness=[x for x in r["loudness"].split("|") if x], asr_transcript=r["asr_transcript"],
        asr_similarity=f(r["asr_similarity"], 0.0), asr_mode=r["asr_mode"],
        asr_pass=r["asr_pass"] == "1", judges=json.loads(r["judges"] or "{}"),
        judge_hash=r["judge_hash"], gate_pass=r["gate_pass"] == "1",
        credits=int(float(r["credits"] or 0)), latency_ms=int(float(r["latency_ms"] or 0)),
        error=r["error"], generated_at=r["generated_at"],
    )


def spec_dict(spec: RenderSpec) -> dict:
    return asdict(spec)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sleep_s(s: float) -> None:
    time.sleep(s)
