"""Stage 19 — eleven_v4 render engine shared by the pilot (19_3) and the full run (19_4).

A *take* is one TTS render of one clip with one (variant, voice, take#,
stability, dictionary) combination. `Engine.render_take` runs:

    TTS (eleven_v4, pcm_44100) → PCM sanity → closed-loop loudnorm → MP3
    cached under build/audio_cache/v4_takes/ → relaxed ASR on the PLAIN display
    text → every pinned judge → TakeResult, appended to data/_v4_takes.tsv

Nothing is uploaded and the manifest is never touched here — 19_4 uploads
accepted winners only. Takes are cached by `take_id`, so re-runs resume for
free; if the judge configuration changed, cached MP3s are re-judged instead
of re-rendered.

Variants (TTS input; ASR and judges always see the display text):
    plain   the orthographic display text (IPA was rejected by the Step-1 probe)
    tag     a leading delivery tag, e.g. "[sotaque paulistano] a cara"
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
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

V4 = "eleven_v4"
DEFAULT_STABILITY = 0.65
SIMILARITY = 0.80
DEFAULT_TAG = "[sotaque paulistano]"
SHORT_CLIP = {"word": "word", "example": "ex"}

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
    return rows


def load_asr_model(path: Path = PINNED_MODELS) -> str:
    rows = [r for r in read_tsv(path) if r["role"] == "asr"]
    return rows[0]["model"] if rows else "gpt-4o-transcribe"


def judge_hash(configs: list[dict]) -> str:
    sig = ";".join(f"{c['backend']}:{c['model']}:{c['prompt_version']}" for c in configs)
    return hashlib.sha256(sig.encode()).hexdigest()[:10]


def mp3_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) if out else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


class Engine:
    """Thread-safe take renderer. One instance per run."""

    def __init__(self, *, judge_configs: list[dict], asr_model: str, el_concurrency: int = 9,
                 gemini_rpm: int = 900, dict_locator: dict | None = None) -> None:
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
        self.credits_used = 0
        self.cached = self._load_ledger()
        TAKES_DIR.mkdir(parents=True, exist_ok=True)

    # ── ledger ──────────────────────────────────────────────────────────────
    def _load_ledger(self) -> dict[str, dict]:
        out = {}
        for r in read_tsv(LEDGER):
            out[r["take_id"]] = r          # last write wins (re-judged takes append)
        return out

    def _append(self, res: TakeResult) -> None:
        with self.ledger_lock:
            append_tsv(LEDGER, [res.ledger_row()], fieldnames=LEDGER_FIELDS)
            self.cached[res.spec.take_id] = res.ledger_row()

    def cached_result(self, spec: RenderSpec) -> TakeResult | None:
        row = self.cached.get(spec.take_id)
        # Rows without a saved MP3 (TTS error, failed PCM sanity) are re-rendered:
        # the sanity rules may have changed since.
        if not row or row.get("error") or not row.get("path") or not Path(row["path"]).is_file():
            return None
        if row.get("tts_text") != spec.tts_text:      # the text to speak changed → re-render
            return None
        res = _result_from_row(spec, row)
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
        if (self.judges and (changed or row.get("judge_hash") != self.judge_hash)
                and not res.sanity and not res.loudness and res.asr_pass):
            self._judge(res, Path(row["path"]).read_bytes())
            changed = True
        if changed:
            self._append(res)
        return res

    # ── one take ────────────────────────────────────────────────────────────
    def render_take(self, spec: RenderSpec) -> TakeResult:
        hit = self.cached_result(spec)
        if hit is not None:
            return hit
        from elevenlabs import VoiceSettings

        from build.lib.asr import asr_roundtrip
        from build.lib.elevenlabs_client import QuotaExceeded
        from build.lib.loudness import normalize_pcm_to_mp3_verified

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
        except QuotaExceeded:
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
        try:
            with self.asr_sem:
                a = asr_roundtrip(asr=self.asr, mp3_bytes=norm.mp3_bytes,
                                  input_text=spec.display_text, clip_type=spec.clip_type,
                                  sense_id=spec.sense_id, is_top_1000=spec.is_top_1000)
            transcripts = [a.transcript, a.biased_transcript]
            res.asr_transcript = a.transcript
            res.asr_pass, res.asr_mode, res.asr_similarity = QA.relaxed_asr_pass(
                transcripts=transcripts, reference=spec.display_text, clip_type=spec.clip_type,
                is_top_1000=spec.is_top_1000, production_decision=a.decision,
                spoken_reference=spec.spoken_reference, manual_pass=spec.manual_pass)
        except Exception as exc:  # noqa: BLE001
            res.error = f"asr:{type(exc).__name__}: {exc}"[:300]
            self._append(res)
            return res
        if res.asr_pass and self.judges:
            self._judge(res, norm.mp3_bytes)
        self._append(res)
        return res

    def _judge(self, res: TakeResult, mp3: bytes) -> None:
        spec = res.spec
        verdicts = {}
        for c, client in self.judges:
            name = c["config"]
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
                                  "severity": r.severity, "evidence": r.evidence[:160]}
            except Exception as exc:  # noqa: BLE001
                verdicts[name] = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}"[:160]}
        res.judges = verdicts
        res.judge_hash = self.judge_hash
        res.gate_pass = QA.gate_pass({k: v["verdict"] for k, v in verdicts.items()})


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
