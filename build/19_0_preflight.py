"""Stage 19 / Step 1 — preflight smoke test: every key + constraint, before any paid step.

Free mode (default) — read-only / free calls only:
  * env var names present
  * ElevenLabs: subscription tier/credits, `/v1/models` (eleven_v4, pt),
    every production BP voice (`GET /v1/voices/{id}`)
  * Anthropic / OpenAI / Gemini model lists contain every Stage-19 candidate
  * Cloudflare R2: list + total size, put/get/delete of a tiny `_smoke/` object
  * local: ffmpeg, espeak-ng, google-genai, disk free, v3 BP clips cached
  * Azure Speech token (reported only — not used by Stage 19)

Paid mode (`--paid`) — adds micro-tests (≤ ~3K ElevenLabs credits, < $0.20):
  * one eleven_v4 / pcm_44100 render per production voice (access, credits per
    character from the `character-cost` header + subscription delta,
    concurrency headers, latency)
  * seed determinism; acceptance of the v6 alias dictionary, the full locked
    voice settings, and apply_text_normalization=off with IPA text
  * IPA adherence: stress minimal pairs (sábia/sabia/sabiá …) as IPA vs plain,
    leakage ("/ˈɡatu/" must come back as "gato"), nasal diphthongs, strong r /
    coda tap, article placement + pause check, inline IPA inside a sentence
  * direction-tag probes ([sotaque paulistano], [clear, careful pronunciation])
  * one call per judge / ASR candidate; Claude 5.x structured tool calls

Outputs:
  reports/19_0_preflight.html          (with playable probe audio)
  audit/19_0_preflight.jsonl           (one record per check)
  config/stage19_preflight.json        (machine-readable gate for Step 4)
  build/audio_cache/v4_takes/preflight/*.mp3

Usage:
  uv run python build/19_0_preflight.py
  uv run python build/19_0_preflight.py --paid
Exit code 1 if any check FAILs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env", override=True)

from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
CONFIG = REPO_ROOT / "config"
REPORT = REPO_ROOT / "reports" / "19_0_preflight.html"
AUDIT = REPO_ROOT / "audit" / "19_0_preflight.jsonl"
GATE_JSON = CONFIG / "stage19_preflight.json"
PROBE_DIR = REPO_ROOT / "build" / "audio_cache" / "v4_takes" / "preflight"
MANIFEST = DATA / "_audio_manifest.tsv"
VOICES = CONFIG / "voices.tsv"
ANKI_MEDIA = DATA / "anki_media"

V4 = "eleven_v4"
REQUIRED_ENV = ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
                "R2_PUBLIC_BASE")
CANDIDATES = {
    "anthropic": ("claude-opus-5-5", "claude-fable-5-1"),
    "openai": ("gpt-4o-transcribe", "gpt-transcribe", "gpt-audio"),
    "gemini": ("gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.5-flash",
               "gemini-3.5-transcribe"),
}
# v6 alias dictionary (hospital → ospitau), pinned by Stages 11 / 16.9.
PT_DICT_LOCATOR = {"pronunciation_dictionary_id": "Ht0OpvDQAJQkJMHRv7Rs",
                   "version_id": "th9qzGumY1q3fkvV3Fi3"}
MIN_FREE_DISK_GB = 10
ESCAPE_HATCH_STATUS = "active_escape_hatch"

# Stress minimal pairs: (written form ASR should return, São Paulo IPA).
STRESS_PROBES = [
    ("sábia", "ˈsabjɐ"), ("sabia", "saˈbiɐ"), ("sabiá", "sabiˈa"),
    ("fábrica", "ˈfabɾikɐ"), ("fabrica", "faˈbɾikɐ"),
    ("secretária", "sekɾeˈtaɾjɐ"), ("secretaria", "sekɾetaˈɾiɐ"),
    ("público", "ˈpubliku"), ("publico", "puˈbliku"), ("publicou", "publiˈkow"),
]
SEGMENT_PROBES = [  # (kind, tts text, expected transcript)
    ("leak", '"/ˈɡatu/"', "gato"),
    ("nasal", '"/ˈpɐ̃w̃/"', "pão"),
    ("nasal", '"/ˈmɐ̃j̃/"', "mãe"),
    ("strong_r", '"/ˈhatu/"', "rato"),
    ("coda_tap", '"/ˈveɾdʒi/"', "verde"),
    ("article", 'o "/doˈmĩɡu/"', "o domingo"),
    ("article_inside", '"/u doˈmĩɡu/"', "o domingo"),
    ("sentence", 'A grama do parque é muito "/ˈveɾdʒi/".', "a grama do parque é muito verde"),
]
LEAK_TOKENS = ("barra", "aspas", "slash", "quote", "ipa", "ʃ", "ɾ", "ɐ", "ˈ")
TAG_VARIANTS = ("[sotaque paulistano]", "[clear, careful pronunciation]")
TAG_LEAK_TOKENS = ("sotaque", "paulistano", "clear", "careful", "pronunciation", "pronúncia")
TAG_WORDS = ("verde", "cara", "quarto", "sede", "proveniente", "salarial", "policial",
             "roda", "poço", "série")
JUDGE_SENSE = "0534.00.01"   # "verde" — judge/ASR candidate smoke on its current v3 clip


class PreflightError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (19_0_preflight): {msg}")


@dataclass
class Check:
    group: str
    name: str
    status: str          # PASS / WARN / FAIL / INFO
    detail: str
    data: dict = field(default_factory=dict)
    audio: list = field(default_factory=list)   # [(label, path)]


CHECKS: list[Check] = []
CHECK_ROWS: list[dict] = []   # per-probe rows (IPA / dictionary probes)


def add(group: str, name: str, status: str, detail: str, data: dict | None = None,
        audio: list | None = None) -> Check:
    c = Check(group, name, status, detail, data or {}, audio or [])
    CHECKS.append(c)
    mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗", "INFO": "·"}[status]
    print(f"  {mark} [{group}] {name}: {detail}", flush=True)
    return c


def _env(k: str) -> str:
    import os
    return os.environ.get(k, "")


def _xi() -> dict:
    return {"xi-api-key": _env("ELEVENLABS_API_KEY")}


def _norm_accented(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── free checks ──────────────────────────────────────────────────────────────
def production_voices() -> list[dict]:
    """BP voices that voice word/example clips today + the escape hatch."""
    names = {r["voice_id"]: r for r in read_tsv(VOICES)}
    used: dict[str, int] = {}
    for r in read_tsv(MANIFEST):
        if r["clip_type"] in ("word", "example") and r["status"] == "uploaded":
            used[r["voice_id"]] = used.get(r["voice_id"], 0) + 1
    for vid, row in names.items():
        if row.get("status") == ESCAPE_HATCH_STATUS:
            used.setdefault(vid, 0)
    return [{"voice_id": v, "name": names.get(v, {}).get("bp_name", "?"),
             "gender": names.get(v, {}).get("gender", "?"), "clips": n}
            for v, n in sorted(used.items(), key=lambda kv: -kv[1])]


def check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not _env(k)]
    if missing:
        add("env", "required variables", "FAIL", f"missing: {', '.join(missing)}")
    else:
        add("env", "required variables", "PASS", f"{len(REQUIRED_ENV)} present")


def check_elevenlabs(c: httpx.Client) -> dict:
    r = c.get("https://api.elevenlabs.io/v1/user/subscription", headers=_xi())
    if r.status_code != 200:
        add("elevenlabs", "subscription", "FAIL", f"HTTP {r.status_code}")
        return {}
    s = r.json()
    left = s["character_limit"] - s["character_count"]
    reset = datetime.fromtimestamp(s["next_character_count_reset_unix"], timezone.utc)
    st = "PASS" if s.get("tier") == "pro" else "FAIL"
    add("elevenlabs", "subscription", st,
        f"tier={s.get('tier')} credits left={left:,} of {s['character_limit']:,}, "
        f"resets {reset:%Y-%m-%d}, overage cap={s.get('max_credit_limit_extension')}",
        {"tier": s.get("tier"), "credits_left": left, "reset": reset.isoformat()})
    r = c.get("https://api.elevenlabs.io/v1/models", headers=_xi())
    m = next((x for x in r.json() if x.get("model_id") == V4), None) if r.status_code == 200 else None
    if not m:
        add("elevenlabs", "eleven_v4 model", "FAIL", "not listed")
    else:
        langs = [lang.get("language_id") for lang in m.get("languages") or []]
        ok = m.get("can_do_text_to_speech") and "pt" in langs
        add("elevenlabs", "eleven_v4 model", "PASS" if ok else "FAIL",
            f"tts={m.get('can_do_text_to_speech')} pt={'pt' in langs} "
            f"rates={m.get('model_rates')} max_len={m.get('maximum_text_length_per_request')}",
            {"model_rates": m.get("model_rates")})
    return s


def check_voices(c: httpx.Client) -> None:
    for v in production_voices():
        r = c.get(f"https://api.elevenlabs.io/v1/voices/{v['voice_id']}", headers=_xi())
        if r.status_code != 200:
            add("voices", v["name"], "FAIL", f"HTTP {r.status_code}")
            continue
        j = r.json()
        ft = (j.get("fine_tuning") or {}).get("state") or {}
        hq = j.get("high_quality_base_model_ids") or []
        sharing = j.get("sharing") or {}
        detail = (f"{v['gender']}, {v['clips']} clips, category={j.get('category')}, "
                  f"v4 fine-tune={ft.get(V4, '—')}, v4 HQ={'yes' if V4 in hq else 'no'}, "
                  f"free_users_allowed={sharing.get('free_users_allowed')}")
        add("voices", v["name"], "PASS", detail, {"voice_id": v["voice_id"]})


def _model_list_check(c: httpx.Client, provider: str) -> None:
    if provider == "anthropic":
        r = c.get("https://api.anthropic.com/v1/models", params={"limit": 200},
                  headers={"x-api-key": _env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
        ids = {m["id"] for m in r.json().get("data", [])} if r.status_code == 200 else set()
    elif provider == "openai":
        r = c.get("https://api.openai.com/v1/models",
                  headers={"Authorization": f"Bearer {_env('OPENAI_API_KEY')}"})
        ids = {m["id"] for m in r.json().get("data", [])} if r.status_code == 200 else set()
    else:
        r = c.get("https://generativelanguage.googleapis.com/v1beta/models",
                  params={"pageSize": 300}, headers={"x-goog-api-key": _env("GEMINI_API_KEY")})
        ids = ({m["name"].split("/")[-1] for m in r.json().get("models", [])}
               if r.status_code == 200 else set())
    if r.status_code != 200:
        add(provider, "key + model list", "FAIL", f"HTTP {r.status_code}")
        return
    missing = [m for m in CANDIDATES[provider] if m not in ids]
    add(provider, "key + model list", "WARN" if missing else "PASS",
        f"{len(ids)} models; candidates {'missing: ' + ', '.join(missing) if missing else 'all present'}")


def check_r2() -> None:
    from build.lib.r2_client import R2Client, R2Config
    try:
        r2 = R2Client(R2Config.from_env())
        n = size = 0
        for page in r2._s3.get_paginator("list_objects_v2").paginate(Bucket=r2.config.bucket):
            for o in page.get("Contents", []):
                n += 1
                size += o["Size"]
        add("r2", "list + size", "PASS" if size < 9e9 else "WARN",
            f"{n:,} objects, {size / 1e9:.2f} GB of the 10 GB free tier", {"objects": n, "bytes": size})
        key = f"_smoke/preflight-{int(time.time())}.txt"
        body = b"stage19 preflight"
        r2._s3.put_object(Bucket=r2.config.bucket, Key=key, Body=body)
        got = r2._s3.get_object(Bucket=r2.config.bucket, Key=key)["Body"].read()
        r2._s3.delete_object(Bucket=r2.config.bucket, Key=key)
        add("r2", "put/get/delete", "PASS" if got == body else "FAIL", f"{key} round-trip")
    except Exception as exc:  # noqa: BLE001
        add("r2", "access", "FAIL", f"{type(exc).__name__}: {exc}")


def check_local() -> None:
    ff = shutil.which("ffmpeg")
    if ff:
        ver = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        add("local", "ffmpeg", "PASS", ver[:60])
    else:
        add("local", "ffmpeg", "FAIL", "not on PATH")
    es = shutil.which("espeak-ng")
    add("local", "espeak-ng", "PASS" if es else "FAIL", es or "not on PATH")
    try:
        import google.genai  # noqa: F401
        from importlib.metadata import version
        add("local", "google-genai", "PASS", version("google-genai"))
    except Exception as exc:  # noqa: BLE001
        add("local", "google-genai", "FAIL", str(exc))
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    add("local", "disk free", "PASS" if free_gb >= MIN_FREE_DISK_GB else "FAIL", f"{free_gb:.0f} GB")
    bp = [r for r in read_tsv(MANIFEST) if r["clip_type"] in ("word", "example")]
    have = sum(1 for r in bp if (ANKI_MEDIA / r["object_key"].split("/", 1)[1]).exists())
    add("local", "v3 BP clips cached", "PASS" if have == len(bp) else "WARN",
        f"{have:,}/{len(bp):,} in data/anki_media/")


def check_azure(c: httpx.Client) -> None:
    region = _env("SPEECH_REGION")
    if not _env("SPEECH_KEY") or not region:
        add("azure", "speech key", "INFO", "not configured (not used by Stage 19)")
        return
    r = c.post(f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
               headers={"Ocp-Apim-Subscription-Key": _env("SPEECH_KEY"), "Content-Length": "0"})
    add("azure", "speech key", "INFO",
        f"HTTP {r.status_code} ({'valid' if r.status_code == 200 else 'invalid'}; not used by Stage 19)")


# ── paid checks ──────────────────────────────────────────────────────────────
class Paid:
    def __init__(self, voices: list[dict]):
        from build.lib.asr import AsrClient
        from build.lib.elevenlabs_client import ElevenLabsClient
        self.voices = voices
        self.el = ElevenLabsClient(model_id=V4, language_code="pt")
        self.el_off = ElevenLabsClient(model_id=V4, language_code="pt",
                                       apply_text_normalization="off")
        self.asr = AsrClient()
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        self.credits_header = 0
        self.chars_sent = 0
        self.concurrency: dict = {}
        # One female + one male voice that carry production clips (IPA / tag probes).
        self.fem = next(v for v in voices if v["gender"] == "female" and v["clips"])
        self.mal = next(v for v in voices if v["gender"] == "male" and v["clips"])

    def render(self, text: str, voice_id: str, label: str, *, client=None,
               seed: int | None = None) -> tuple[bytes, Path, dict]:
        from build.lib.loudness import normalize_pcm_to_mp3_verified
        cl = client or self.el
        res = cl.generate_pcm_meta(text=text, voice_id=voice_id, sense_id="preflight",
                                   clip_type=label, version=1, seed=seed)
        self.chars_sent += len(text)
        self.credits_header += res.character_cost or 0
        for k in ("current-concurrent-requests", "maximum-concurrent-requests"):
            if k in res.headers:
                self.concurrency[k] = res.headers[k]
        norm = normalize_pcm_to_mp3_verified(res.audio_pcm)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{label}-{voice_id[:6]}")[:80]
        path = PROBE_DIR / f"{safe}.mp3"
        path.write_bytes(norm.mp3_bytes)
        return res.audio_pcm, path, {"latency_ms": res.latency_ms, "cost": res.character_cost,
                                     "request_id": res.request_id}

    def transcribe(self, mp3: bytes) -> str:
        from build.lib.asr import pad_mp3_with_silence
        text, _ = self.asr.transcribe(pad_mp3_with_silence(mp3))
        return text


def _internal_silences(mp3_path: Path, min_s: float = 0.15) -> int:
    """Count silences ≥ min_s strictly inside the clip (not leading/trailing)."""
    from build.lib.loudness import FFMPEG
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", str(mp3_path), "-af",
                        f"silencedetect=noise=-35dB:d={min_s}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    dur_m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", p.stderr)
    dur = (int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))) if dur_m else 0
    return sum(1 for s in starts if 0.25 < s < dur - 0.35)


PAID_SECTIONS = ("voices", "params", "ipa", "ipa_dict", "tags", "candidates", "claude")

# IPA delivery mechanisms probed on the stress minimal pairs.
IPA_VARIANTS = {
    "quoted": lambda ipa, w: f'"/{ipa}/"',            # ElevenLabs v4 docs syntax
    "bare": lambda ipa, w: f"/{ipa}/",
    "ctx_quoted": lambda ipa, w: f'Eu disse "/{ipa}/".',
    "plain": lambda ipa, w: w,                          # orthography baseline
    "plain_ctx": lambda ipa, w: f"Eu disse {w}.",
}
DICT_PROBE_EXTRA = [("pão", "ˈpɐ̃w̃"), ("mãe", "ˈmɐ̃j̃"), ("rato", "ˈhatu"), ("verde", "ˈveɾdʒi")]


def _hit(transcript: str, expect: str) -> bool:
    got, exp = _norm_accented(transcript), _norm_accented(expect)
    return got == exp or got.endswith(" " + exp) or got.endswith(exp)


def _current_clip_name(sense_id: str, clip_type: str) -> str:
    for r in read_tsv(MANIFEST):
        if r["sense_id"] == sense_id and r["clip_type"] == clip_type:
            return r["object_key"].split("/", 1)[1]
    raise PreflightError(f"no manifest row for {sense_id} {clip_type}")


def sec_voices(pv: "Paid", gate: dict) -> None:
    for v in pv.voices:
        try:
            _, path, meta = pv.render("Bom dia.", v["voice_id"], f"voice-{v['name']}")
            add("paid:voices", v["name"], "PASS",
                f"pcm_44100 ok, {meta['latency_ms']} ms, character-cost={meta['cost']}",
                meta, [(f"{v['name']}: Bom dia.", path)])
        except Exception as exc:  # noqa: BLE001
            add("paid:voices", v["name"], "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}")


def sec_params(pv: "Paid", gate: dict) -> None:
    from build.lib.elevenlabs_client import LOCKED_VOICE_SETTINGS, ElevenLabsClient
    fem = pv.fem
    try:
        a, _, _ = pv.render("Bom dia.", fem["voice_id"], "seed-a", seed=424242)
        b, _, _ = pv.render("Bom dia.", fem["voice_id"], "seed-b", seed=424242)
        same = hashlib.md5(a).hexdigest() == hashlib.md5(b).hexdigest()
        gate["seed_deterministic"] = same
        add("paid:params", "seed determinism", "INFO",
            "identical PCM for the same seed" if same else "same seed → different audio "
            "(best-effort per docs; takes stay distinct either way)")
    except Exception as exc:  # noqa: BLE001
        add("paid:params", "seed determinism", "WARN", str(exc)[:160])
    try:
        dict_client = ElevenLabsClient(model_id=V4, language_code="pt",
                                       pronunciation_dict_locators=[PT_DICT_LOCATOR])
        _, path, _ = pv.render("o hospital", fem["voice_id"], "dict-hospital", client=dict_client)
        gate["dict_locator_ok"] = True
        add("paid:params", "alias dictionary v6 on v4", "PASS", "accepted",
            audio=[("o hospital (dict)", path)])
    except Exception as exc:  # noqa: BLE001
        gate["dict_locator_ok"] = False
        add("paid:params", "alias dictionary v6 on v4", "WARN", str(exc)[:160])
    try:
        chunks = pv.el._client.text_to_speech.convert(
            voice_id=fem["voice_id"], text="Bom dia.", model_id=V4, output_format="pcm_44100",
            language_code="pt", voice_settings=LOCKED_VOICE_SETTINGS, seed=7)
        pcm = b"".join(chunks)
        pv.chars_sent += len("Bom dia.")
        gate["full_voice_settings_accepted"] = bool(pcm)
        add("paid:params", "full locked voice settings", "INFO",
            "accepted (style/speaker_boost are ignored by v4)")
    except Exception as exc:  # noqa: BLE001
        gate["full_voice_settings_accepted"] = False
        add("paid:params", "full locked voice settings", "INFO", f"rejected: {str(exc)[:120]}")


def _run_jobs(pv: "Paid", jobs: list[tuple], client=None) -> list[dict]:
    """jobs: (kind, variant, voice, tts_text, expect). Renders + transcribes."""
    def run(job):
        kind, variant, v, text, expect = job
        label = f"{kind}-{variant}-{_norm_accented(expect).replace(' ', '_')}-{v['name']}"
        try:
            _, path, _ = pv.render(text, v["voice_id"], label, client=client)
            tr = pv.transcribe(path.read_bytes())
            return {"kind": kind, "variant": variant, "voice": v["name"], "text": text,
                    "expect": expect, "transcript": tr, "hit": _hit(tr, expect),
                    "leak": any(t in tr.lower() for t in LEAK_TOKENS), "path": str(path)}
        except Exception as exc:  # noqa: BLE001
            return {"kind": kind, "variant": variant, "voice": v["name"], "text": text,
                    "expect": expect, "transcript": f"ERROR {type(exc).__name__}: {exc}"[:200],
                    "hit": False, "leak": False, "path": ""}
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(run, jobs))


def sec_ipa(pv: "Paid", gate: dict) -> None:
    jobs = []
    for v in (pv.fem, pv.mal):
        for word, ipa in STRESS_PROBES:
            for variant, fmt in IPA_VARIANTS.items():
                jobs.append(("stress", variant, v, fmt(ipa, word), word))
        for kind, text, expect in SEGMENT_PROBES:
            jobs.append((kind, "quoted", v, text, expect))
    rows = _run_jobs(pv, jobs)
    off_rows = _run_jobs(pv, [("norm_off", "quoted", v, '"/ˈveɾdʒi/"', "verde")
                              for v in (pv.fem, pv.mal)], client=pv.el_off)
    rows += off_rows
    stats = {}
    for variant in IPA_VARIANTS:
        rs = [r for r in rows if r["kind"] == "stress" and r["variant"] == variant]
        stats[variant] = f"{sum(r['hit'] for r in rs)}/{len(rs)}"
    leak = any(r["leak"] for r in rows if r["variant"] not in ("plain", "plain_ctx"))
    nasals_ok = all(r["hit"] for r in rows if r["kind"] == "nasal")
    seg_fail = [f"{r['kind']}:{r['expect']}→{r['transcript']!r}" for r in rows
                if r["kind"] not in ("stress",) and not r["hit"]]
    pauses = {f"{r['kind']}-{r['voice']}": _internal_silences(Path(r["path"]))
              for r in rows if r["kind"] in ("article", "article_inside", "sentence") and r["path"]}
    gate.setdefault("ipa_probe", {}).update({
        "stress_honored_by_variant": stats, "leak_free": not leak, "nasals_ok": nasals_ok,
        "segment_failures": seg_fail, "internal_pauses": pauses,
    })
    for r in rows:
        r_short = {k: r[k] for k in ("kind", "variant", "voice", "text", "expect", "transcript", "hit")}
        CHECK_ROWS.append(r_short)
    audio = [(f"{r['voice']} {r['kind']}/{r['variant']}: {r['text']} → {r['transcript']}", r["path"])
             for r in rows if r["path"] and (r["kind"] != "stress" or r["variant"] in ("quoted", "plain"))]
    add("paid:ipa", "stress minimal pairs by mechanism", "INFO",
        "honored: " + ", ".join(f"{k}={v}" for k, v in stats.items()),
        {"rows": [r for r in CHECK_ROWS if r["kind"] == "stress"]})
    add("paid:ipa", "leakage / nasals / r / article / sentence",
        "PASS" if (not leak and nasals_ok and not seg_fail) else "WARN",
        f"leak={'none' if not leak else 'YES'}; nasals={'ok' if nasals_ok else 'FAIL'}; "
        f"failures={seg_fail or 'none'}; internal pauses={pauses}", audio=audio[:60])


def sec_ipa_dict(pv: "Paid", gate: dict) -> None:
    """Phoneme rules in a pronunciation dictionary — the other IPA mechanism."""
    from build.lib.elevenlabs_client import ElevenLabsClient
    rules = [{"string_to_replace": w, "type": "phoneme", "phoneme": ipa, "alphabet": "ipa",
              "case_sensitive": False, "word_boundaries": True}
             for w, ipa in STRESS_PROBES + DICT_PROBE_EXTRA]
    try:
        resp = pv.el._client.pronunciation_dictionaries.create_from_rules(
            rules=rules, name="stage19-preflight-ipa-phoneme-probe",
            description="Stage 19 preflight: does eleven_v4 honor IPA phoneme rules for pt-BR?",
            workspace_access="admin")
        loc = {"pronunciation_dictionary_id": resp.id, "version_id": resp.version_id}
        gate.setdefault("ipa_probe", {})["phoneme_dictionary"] = loc
    except Exception as exc:  # noqa: BLE001
        add("paid:ipa", "phoneme-rule dictionary", "WARN", f"create failed: {str(exc)[:160]}")
        return
    client = ElevenLabsClient(model_id=V4, language_code="pt", pronunciation_dict_locators=[loc])
    jobs = [("stress", "dict_phoneme", v, w, w) for v in (pv.fem, pv.mal) for w, _ in STRESS_PROBES]
    jobs += [("segment", "dict_phoneme", v, w, w) for v in (pv.fem, pv.mal) for w, _ in DICT_PROBE_EXTRA]
    rows = _run_jobs(pv, jobs, client=client)
    for r in rows:
        CHECK_ROWS.append({k: r[k] for k in ("kind", "variant", "voice", "text", "expect",
                                             "transcript", "hit")})
    st = [r for r in rows if r["kind"] == "stress"]
    sg = [r for r in rows if r["kind"] == "segment"]
    honored = sum(r["hit"] for r in st)
    gate["ipa_probe"].setdefault("stress_honored_by_variant", {})["dict_phoneme"] = f"{honored}/{len(st)}"
    add("paid:ipa", "phoneme-rule dictionary", "INFO",
        f"stress honored {honored}/{len(st)}; segments {sum(r['hit'] for r in sg)}/{len(sg)}: "
        + "; ".join(f"{r['expect']}→{r['transcript']!r}" for r in sg),
        audio=[(f"{r['voice']} dict: {r['text']} → {r['transcript']}", r["path"])
               for r in rows if r["path"]][:24])


def ipa_gate(gate: dict) -> None:
    probe = gate.get("ipa_probe", {})
    stats = probe.get("stress_honored_by_variant", {})
    if not stats:
        return

    def rate(s: str) -> float:
        a, b = s.split("/")
        return int(a) / max(int(b), 1)

    plain = max(rate(stats.get("plain", "0/1")), rate(stats.get("plain_ctx", "0/1")))
    ipa_rates = {k: rate(v) for k, v in stats.items() if k not in ("plain", "plain_ctx")}
    best = max(ipa_rates, key=ipa_rates.get) if ipa_rates else None
    ok = (best is not None and ipa_rates[best] >= 0.8 and ipa_rates[best] >= plain
          and probe.get("leak_free", False) and probe.get("nasals_ok", False))
    gate["ipa_enabled"] = ok
    gate["ipa_mechanism"] = best if ok else None
    add("paid:ipa", "GATE: IPA in pilot", "PASS" if ok else "WARN",
        (f"enabled via {best} ({stats[best]})" if ok else
         f"disabled — best IPA mechanism {best} {stats.get(best)} vs plain baseline "
         f"{plain:.0%}; pilot renders plain text (\"don't force it\")"))


def sec_tags(pv: "Paid", gate: dict) -> None:
    jobs = [("tag", tag, pv.fem, f"{tag} {w}", w) for tag in TAG_VARIANTS for w in TAG_WORDS]
    rows = _run_jobs(pv, jobs)
    res = {}
    for tag in TAG_VARIANTS:
        rs = [r for r in rows if r["variant"] == tag]
        leaks = [r["transcript"] for r in rs if any(t in r["transcript"].lower() for t in TAG_LEAK_TOKENS)]
        hits = sum(r["hit"] for r in rs)
        res[tag] = not leaks
        add("paid:tags", tag, "PASS" if not leaks else "WARN",
            f"leaks={len(leaks)}, word recognized {hits}/{len(rs)}",
            {"transcripts": [r["transcript"] for r in rs]},
            [(f"{r['text']} → {r['transcript']}", r["path"]) for r in rs[:4] if r["path"]])
    gate["direction_tags_leak_free"] = res


def sec_candidates(pv: "Paid", gate: dict) -> None:
    clip = (ANKI_MEDIA / _current_clip_name(JUDGE_SENSE, "word")).read_bytes()
    from build.lib.audio_judge import AudioJudgeClient
    from build.lib.gemini_audio_judge import GeminiAudioJudgeClient
    for model, version in (("gemini-3.1-pro-preview", "J1"), ("gemini-3.1-pro-preview", "J2"),
                           ("gemini-3.8-flash", "J2"), ("gemini-3.5-flash", "J2")):
        try:
            j = GeminiAudioJudgeClient(model=model, prompt_version=version)
            r = j.judge(audio_bytes=clip, pt="verde", ipa_word_final="ˈveɾdʒi",
                        voice_id="smoke", sense_id=JUDGE_SENSE, clip_type="word")
            add("paid:judges", f"{model} {version}", "PASS",
                f"{r.pronunciation_verdict}/{r.drift} ({r.latency_ms} ms, ~${r.cost_usd:.4f}): "
                f"{r.evidence[:90]}")
        except Exception as exc:  # noqa: BLE001
            add("paid:judges", f"{model} {version}", "WARN", f"{type(exc).__name__}: {str(exc)[:140]}")
    try:
        j = AudioJudgeClient(model="gpt-audio", prompt_version="J2")
        r = j.judge(audio_bytes=clip, pt="verde", ipa_word_final="ˈveɾdʒi",
                    voice_id="smoke", sense_id=JUDGE_SENSE, clip_type="word")
        add("paid:judges", "gpt-audio J2", "PASS",
            f"{r.pronunciation_verdict}/{r.drift} ({r.latency_ms} ms, ~${r.cost_usd:.4f})")
    except Exception as exc:  # noqa: BLE001
        add("paid:judges", "gpt-audio J2", "WARN", f"{type(exc).__name__}: {str(exc)[:140]}")

    from build.lib import asr_alt
    from build.lib.asr import pad_mp3_with_silence
    padded = pad_mp3_with_silence(clip)
    oa = asr_alt.make_openai_client()
    for model in ("gpt-4o-transcribe", "gpt-transcribe"):
        try:
            r = asr_alt.transcribe_openai(client=oa, model=model, mp3_bytes=padded)
            add("paid:asr", model, "PASS", f"transcript={r.transcript!r}")
        except Exception as exc:  # noqa: BLE001
            add("paid:asr", model, "WARN", f"{type(exc).__name__}: {str(exc)[:140]}")
    try:
        r = asr_alt.transcribe_elevenlabs(client=asr_alt.make_elevenlabs_client(), mp3_bytes=padded)
        add("paid:asr", "scribe_v2", "PASS", f"transcript={r.transcript!r}")
    except Exception as exc:  # noqa: BLE001
        add("paid:asr", "scribe_v2", "WARN", f"{type(exc).__name__}: {str(exc)[:140]}")
    try:
        from google import genai
        r = asr_alt.transcribe_gemini(client=genai.Client(api_key=_env("GEMINI_API_KEY")),
                                      mp3_bytes=padded)
        add("paid:asr", "gemini-3.5-transcribe", "PASS", f"transcript={r.transcript!r}")
    except Exception as exc:  # noqa: BLE001
        add("paid:asr", "gemini-3.5-transcribe", "WARN", f"{type(exc).__name__}: {str(exc)[:140]}")


def sec_claude(pv: "Paid", gate: dict) -> None:
    from build.lib.llm import AnthropicClient
    schema = {"type": "object", "additionalProperties": False, "required": ["ipa", "confidence"],
              "properties": {"ipa": {"type": "string"},
                             "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}
    for model in CANDIDATES["anthropic"]:
        try:
            cl = AnthropicClient(model=model, enable_caching=False)
            d = cl.call_tool(system="You are a Brazilian Portuguese (São Paulo) phonetician.",
                             user_message="Broad IPA for 'bonito'; primary stress mark before the "
                                          "stressed syllable; no slashes.",
                             tool_name="give_ipa", tool_input_schema=schema, strict=True)
            add("paid:claude", model, "PASS", f"bonito → {d}")
        except Exception as exc:  # noqa: BLE001
            add("paid:claude", model, "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}")


def paid_checks(gate: dict, sections: tuple[str, ...]) -> None:
    pv = Paid(production_voices())
    with httpx.Client(timeout=30) as c:
        before = c.get("https://api.elevenlabs.io/v1/user/subscription",
                       headers=_xi()).json().get("character_count", 0)
    funcs = {"voices": sec_voices, "params": sec_params, "ipa": sec_ipa,
             "ipa_dict": sec_ipa_dict, "tags": sec_tags, "candidates": sec_candidates,
             "claude": sec_claude}
    try:
        for name in sections:
            try:
                funcs[name](pv, gate)
            except Exception as exc:  # noqa: BLE001
                add(f"paid:{name}", "section", "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")
        if "ipa" in sections or "ipa_dict" in sections:
            ipa_gate(gate)
    finally:
        time.sleep(2)  # let the usage counter settle
        with httpx.Client(timeout=30) as c:
            after = c.get("https://api.elevenlabs.io/v1/user/subscription",
                          headers=_xi()).json().get("character_count", 0)
        used = after - before
        per_char = used / pv.chars_sent if pv.chars_sent else 0.0
        gate.setdefault("budget", {}).update({
            "credits_used": used, "chars_sent": pv.chars_sent,
            "credits_per_char": round(per_char, 3),
            "character_cost_header_sum": pv.credits_header, "concurrency": pv.concurrency})
        add("paid:budget", "credits per character", "INFO",
            f"{used:,} credits for {pv.chars_sent:,} chars sent → {per_char:.3f}/char "
            f"(header sum {pv.credits_header:,}); concurrency headers {pv.concurrency or 'none'}")


# ── report ───────────────────────────────────────────────────────────────────
def write_outputs(gate: dict, paid: bool) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with AUDIT.open("a", encoding="utf-8") as f:
        for c in CHECKS:
            f.write(json.dumps({"run_at": ts, "paid": paid, "group": c.group, "name": c.name,
                                "status": c.status, "detail": c.detail, "data": c.data},
                               ensure_ascii=False, default=str) + "\n")
        for r in CHECK_ROWS:
            f.write(json.dumps({"run_at": ts, "group": "probe_row", **r},
                               ensure_ascii=False, default=str) + "\n")
    if paid:
        gate["generated_at"] = ts
        GATE_JSON.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    css = """
    :root{--bg:#0f1115;--panel:#181b22;--border:#2c3140;--text:#e6e9ef;--muted:#9098a6;
      --ok:#4ade80;--warn:#fbbf24;--bad:#ff6b6b;--info:#5aa9ff;}
    *{box-sizing:border-box}
    body{background:var(--bg);color:var(--text);margin:0;padding:24px 16px;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .wrap{max-width:1100px;margin:0 auto}
    h1{font-size:21px;margin:0 0 4px} h2{font-size:16px;margin:22px 0 8px}
    .sub{color:var(--muted);margin:0 0 16px}
    table{border-collapse:collapse;width:100%;font-size:13px}
    td,th{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
    th{color:var(--muted);font-weight:500}
    .PASS{color:var(--ok)} .WARN{color:var(--warn)} .FAIL{color:var(--bad)} .INFO{color:var(--info)}
    audio{height:28px;max-width:100%} .clip{display:flex;gap:8px;align-items:center;margin:3px 0}
    .clip span{color:var(--muted);font-size:12px;min-width:260px}
    pre{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:10px;
      overflow-x:auto;font-size:12px}
    """
    groups: dict[str, list[Check]] = {}
    for c in CHECKS:
        groups.setdefault(c.group, []).append(c)
    parts = []
    for g, cs in groups.items():
        rows = []
        for c in cs:
            clips = "".join(
                f"<div class='clip'><span>{html.escape(lbl)}</span><audio controls preload='none' "
                f"src='data:audio/mpeg;base64,{base64.b64encode(Path(p).read_bytes()).decode()}'>"
                f"</audio></div>" for lbl, p in c.audio)
            rows.append(f"<tr><td>{html.escape(c.name)}</td><td class='{c.status}'>{c.status}</td>"
                        f"<td>{html.escape(c.detail)}{clips}</td></tr>")
        parts.append(f"<h2>{html.escape(g)}</h2><table><tr><th>check</th><th>status</th>"
                     f"<th>detail</th></tr>{''.join(rows)}</table>")
    n_fail = sum(c.status == "FAIL" for c in CHECKS)
    gate_html = (f"<h2>gate (config/stage19_preflight.json)</h2><pre>"
                 f"{html.escape(json.dumps(gate, ensure_ascii=False, indent=2))}</pre>") if paid else ""
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Stage 19 preflight</title><style>{css}</style></head><body><div class='wrap'>"
           f"<h1>Stage 19 — preflight</h1><p class='sub'>{ts[:19]} UTC · mode "
           f"{'free + paid' if paid else 'free'} · {len(CHECKS)} checks · "
           f"<span class='{'FAIL' if n_fail else 'PASS'}'>{n_fail} FAIL</span></p>"
           f"{gate_html}{''.join(parts)}</div></body></html>")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paid", action="store_true",
                    help="also run the paid micro-tests (≤ ~3K ElevenLabs credits, < $0.20)")
    ap.add_argument("--sections", default=",".join(PAID_SECTIONS),
                    help=f"paid sections to run (default all): {','.join(PAID_SECTIONS)}")
    args = ap.parse_args()
    sections = tuple(s.strip() for s in args.sections.split(",") if s.strip())
    unknown = [s for s in sections if s not in PAID_SECTIONS]
    if unknown:
        raise PreflightError(f"unknown --sections: {unknown}")
    print(f"=== Stage 19.0 — preflight ({'free + paid' if args.paid else 'free'}) ===")
    # Partial paid runs merge into the existing gate file.
    gate: dict = (json.loads(GATE_JSON.read_text()) if args.paid and GATE_JSON.exists() else {})
    check_env()
    with httpx.Client(timeout=30) as c:
        check_elevenlabs(c)
        check_voices(c)
        for p in ("anthropic", "openai", "gemini"):
            _model_list_check(c, p)
        check_azure(c)
    check_r2()
    check_local()
    try:
        if args.paid:
            if any(c.status == "FAIL" for c in CHECKS):
                print("  free checks have FAILs — skipping paid checks")
            else:
                paid_checks(gate, sections)
    finally:
        write_outputs(gate, args.paid)
    n_fail = sum(c.status == "FAIL" for c in CHECKS)
    print(f"\n{len(CHECKS)} checks, {n_fail} FAIL → {REPORT.relative_to(REPO_ROOT)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
