"""Stage 11 / Step 0 — Pilot 100 senses on Eleven v3.

Render both the word clip and the example clip for 100 senses (distributed
across all 8 BP voices) on `eleven_v3`, run ASR roundtrip on each, write to
a side TSV that does NOT touch the real `_audio_manifest.tsv`, and build
a detailed comparison HTML at `audit/11_0_pilot_v3.html`.

The HTML is the GO/HOLD gate for Stage 11.1 (full re-render).

Pipeline per clip:
    ElevenLabs PCM (v3, language=pt, pron-dict locator from latest meta)
      -> ffmpeg loudnorm (-16 LUFS, closed-loop)
        -> Whisper ASR roundtrip via build/lib/asr.py
          -> R2 upload (prefix audio_pilot_v3/, NOT shared with prod)
            -> append side-manifest row

Usage:
    .venv/bin/python build/11_0_pilot_v3.py
    .venv/bin/python build/11_0_pilot_v3.py --n 100 --concurrency 10 --yes
"""
from __future__ import annotations

import argparse
import csv
import html as _html
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.asr import AsrClient, asr_roundtrip  # noqa: E402
from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
AUDIT_DIR = REPO_ROOT / "audit"
AUDIO_CACHE = REPO_ROOT / "build" / "audio_cache"

FINAL_TSV = DATA_DIR / "06-final.tsv"
VOICES_TSV = CONFIG_DIR / "voices.tsv"
MAIN_MANIFEST = DATA_DIR / "_audio_manifest.tsv"
PILOT_MANIFEST = DATA_DIR / "_audio_manifest_v3_pilot.tsv"
DICT_META = DATA_DIR / "_audio_dictionary_meta.tsv"
HTML_OUT = AUDIT_DIR / "11_0_pilot_v3.html"
LOG_PATH = AUDIT_DIR / "11_0_pilot_v3.log"

V3_MODEL_ID = "eleven_v3"
FLASH_MODEL_ID = "eleven_flash_v2_5"
LANGUAGE = "pt"
R2_PILOT_PREFIX = "audio_pilot_v3"
DEFAULT_CONCURRENCY = 10
DEFAULT_N = 100


PILOT_MANIFEST_FIELDS = [
    "sense_id", "clip_type", "voice_gender", "voice_id", "text_input",
    "object_key", "url", "version", "md5",
    "asr_transcript", "asr_similarity", "asr_phonetic_distance",
    "asr_decision", "asr_attempts", "asr_cost_usd", "asr_notes",
    "applied_gain_db", "final_lufs", "final_tp",
    "loudness_within_tolerance", "tp_limited",
    "generated_at", "status",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_dict_locator() -> dict | None:
    """Return the most recent {pronunciation_dictionary_id, version_id}."""
    if not DICT_META.exists():
        return None
    rows = read_tsv(DICT_META)
    if not rows:
        return None
    last = rows[-1]
    if not last.get("dictionary_id") or not last.get("version_id"):
        return None
    return {
        "pronunciation_dictionary_id": last["dictionary_id"],
        "version_id": last["version_id"],
    }


def _verify_voices_on_v3(voice_ids: list[str], api_key: str) -> tuple[int, int]:
    """HTTP GET /v1/voices/{id} for each; report HQ-tuned-for-v3 status.

    `high_quality_base_model_ids` is a positive "voice was tuned for these
    models" list — voices NOT listed for v3 are still callable, just not
    HQ-guaranteed. Verified live with `m151rjrbWX..` (not HQ-listed) which
    returned HTTP 200 + audio bytes. So this is informational, not a gate.

    Returns (reachable, unreachable). Only HTTP errors count as fail.
    """
    reachable = fail = 0
    print(f"{'voice_id':24}  {'name':30}  {'gender':6}  HQ-tuned-for-v3?")
    print("-" * 90)
    with httpx.Client(timeout=30.0, headers={"xi-api-key": api_key}) as c:
        for vid in voice_ids:
            try:
                r = c.get(f"https://api.elevenlabs.io/v1/voices/{vid}")
                if r.status_code != 200:
                    print(f"{vid:24}  {'?':30}  {'?':6}  FAIL ({r.status_code})")
                    fail += 1
                    continue
                j = r.json()
                hq = j.get("high_quality_base_model_ids") or []
                if not hq:
                    tag = "yes (no HQ list — compatible with all)"
                elif V3_MODEL_ID in hq:
                    tag = "yes (HQ-tuned)"
                else:
                    tag = "render OK but not HQ-tuned"
                gender = (j.get("labels") or {}).get("gender", "?")
                name = (j.get("name") or "?")[:30]
                print(f"{vid:24}  {name:30}  {gender:6}  {tag}")
                reachable += 1
            except Exception as exc:  # noqa: BLE001
                print(f"{vid:24}  ERR {type(exc).__name__}: {exc}")
                fail += 1
    return reachable, fail


def _load_voice_pairing() -> dict[str, dict]:
    """{bp_voice_id: {en_voice_id, gender, pool_index, en_name}}"""
    rows = read_tsv(VOICES_TSV)
    out: dict[str, dict] = {}
    for r in rows:
        bp = r["voice_id"]
        notes = r.get("notes", "")
        en_name = ""
        if "en pair:" in notes:
            en_name = notes.split("en pair:", 1)[1].strip().split(";")[0].strip()
        out[bp] = {
            "en_voice_id": r.get("en_voice_id", ""),
            "gender": r.get("gender", ""),
            "pool_index": r.get("pool_index", ""),
            "en_name": en_name,
        }
    return out


def _select_pilot_senses(n: int, pairing: dict[str, dict]) -> list[dict]:
    """Pick `n` senses distributed across all 8 BP voices, lowest rank first."""
    final_rows = read_tsv(FINAL_TSV)
    senses: list[dict] = []
    for r in final_rows:
        if not r.get("example_pt", "").strip():
            continue
        senses.append({
            "sense_id": r["sense_id"],
            "rank": int(r.get("rank") or "0"),
            "pt": r.get("pt", ""),
            "pt_display": r.get("pt_display", "") or r.get("pt", ""),
            "example_pt": r.get("example_pt", ""),
            "en_primary": r.get("en_primary", ""),
            "example_en": r.get("example_en", ""),
            "bp_voice_id": r.get("voice_id", ""),
            "voice_gender": r.get("voice_gender", ""),
            "audio_word": r.get("audio_word", ""),       # current Flash URL
            "audio_example": r.get("audio_example", ""),  # current Flash URL
        })

    by_voice: dict[str, list[dict]] = {}
    for s in senses:
        by_voice.setdefault(s["bp_voice_id"], []).append(s)
    for v in by_voice.values():
        v.sort(key=lambda x: x["rank"])

    voices = list(by_voice.keys())
    picked: list[dict] = []
    idx = {v: 0 for v in voices}
    while len(picked) < n and any(idx[v] < len(by_voice[v]) for v in voices):
        for v in voices:
            if len(picked) >= n:
                break
            i = idx[v]
            if i < len(by_voice[v]):
                picked.append(by_voice[v][i])
                idx[v] = i + 1
    picked.sort(key=lambda x: x["sense_id"])
    return picked


def _en_url_for_sense(sid: str, manifest_rows: list[dict]) -> str:
    for r in manifest_rows:
        if r["sense_id"] == sid and r["clip_type"] == "en_ex":
            return r["url"]
    return ""


@dataclass
class ClipOut:
    sense_id: str
    clip_type: str  # 'word' | 'example'
    voice_id: str
    voice_gender: str
    text_input: str
    object_key: str
    url: str
    md5: str
    asr_transcript: str
    asr_similarity: float
    asr_phonetic_distance: float | None
    asr_decision: str
    asr_attempts: int
    asr_cost_usd: float
    asr_notes: str
    applied_gain_db: float
    final_lufs: float
    final_tp: float
    loudness_within_tolerance: bool
    tp_limited: bool
    status: str  # 'uploaded' | 'failed'
    error: str = ""


def _process_one_clip(
    *,
    sense: dict,
    clip_type: str,  # 'word' | 'example'
    el: ElevenLabsClient,
    asr: AsrClient,
    r2: R2Client,
) -> ClipOut:
    sid = sense["sense_id"]
    text = sense["pt_display"] if clip_type == "word" else sense["example_pt"]
    bp_voice = sense["bp_voice_id"]
    gender = sense["voice_gender"]
    short = "word" if clip_type == "word" else "ex"
    object_key = f"{R2_PILOT_PREFIX}/{sid}-{short}-{V3_MODEL_ID}-v1.mp3"
    out_base = dict(
        sense_id=sid, clip_type=clip_type, voice_id=bp_voice,
        voice_gender=gender, text_input=text, object_key=object_key,
        url="", md5="", asr_transcript="", asr_similarity=0.0,
        asr_phonetic_distance=None, asr_decision="", asr_attempts=0,
        asr_cost_usd=0.0, asr_notes="",
        applied_gain_db=0.0, final_lufs=0.0, final_tp=0.0,
        loudness_within_tolerance=False, tp_limited=False,
        status="failed", error="",
    )

    try:
        tts = el.generate_pcm(
            text=text, voice_id=bp_voice,
            sense_id=sid, clip_type=clip_type, version=1,
        )
    except Exception as exc:  # noqa: BLE001
        out_base["error"] = f"tts: {type(exc).__name__}: {exc}"
        return ClipOut(**out_base)

    try:
        vres = normalize_pcm_to_mp3_verified(tts.audio_pcm)
    except Exception as exc:  # noqa: BLE001
        out_base["error"] = f"loudnorm: {type(exc).__name__}: {exc}"
        return ClipOut(**out_base)

    mp3 = vres.mp3_bytes
    out_base.update(
        applied_gain_db=vres.applied_gain_db,
        final_lufs=vres.final_mp3_lufs,
        final_tp=vres.final_mp3_tp,
        loudness_within_tolerance=vres.within_tolerance,
        tp_limited=vres.tp_limited,
    )

    # ASR roundtrip — same length-aware policy as Stage 6
    try:
        ar = asr_roundtrip(
            asr=asr, mp3_bytes=mp3, input_text=text,
            clip_type=clip_type, sense_id=sid,
            is_top_1000=(sense["rank"] > 0 and sense["rank"] <= 1000),
        )
        out_base.update(
            asr_transcript=ar.transcript,
            asr_similarity=ar.text_similarity,
            asr_phonetic_distance=ar.phonetic_distance,
            asr_decision=ar.decision,
            asr_attempts=ar.attempts,
            asr_cost_usd=ar.cost_usd,
            asr_notes=ar.notes,
        )
    except Exception as exc:  # noqa: BLE001
        out_base["error"] = f"asr: {type(exc).__name__}: {exc}"
        # Continue to R2 upload anyway so the listener can hear it.

    try:
        upload = r2.upload_bytes(
            mp3, object_key,
            extra_metadata={"sense_id": sid, "clip_type": clip_type,
                            "voice_id": bp_voice, "model": V3_MODEL_ID,
                            "stage": "11_0_pilot"},
        )
        out_base["url"] = upload.url
        out_base["md5"] = upload.content_md5
        out_base["status"] = "uploaded"
    except Exception as exc:  # noqa: BLE001
        out_base["error"] = f"r2: {type(exc).__name__}: {exc}"

    return ClipOut(**out_base)


def _write_side_manifest(clips: list[ClipOut]) -> None:
    PILOT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PILOT_MANIFEST_FIELDS,
                           dialect="excel-tab", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        now = _now_iso()
        for c in clips:
            w.writerow({
                "sense_id": c.sense_id, "clip_type": c.clip_type,
                "voice_gender": c.voice_gender, "voice_id": c.voice_id,
                "text_input": c.text_input, "object_key": c.object_key,
                "url": c.url, "version": "1", "md5": c.md5,
                "asr_transcript": c.asr_transcript,
                "asr_similarity": f"{c.asr_similarity:.4f}",
                "asr_phonetic_distance": "" if c.asr_phonetic_distance is None
                                          else f"{c.asr_phonetic_distance:.4f}",
                "asr_decision": c.asr_decision,
                "asr_attempts": str(c.asr_attempts),
                "asr_cost_usd": f"{c.asr_cost_usd:.5f}",
                "asr_notes": c.asr_notes,
                "applied_gain_db": f"{c.applied_gain_db:.3f}",
                "final_lufs": f"{c.final_lufs:.3f}",
                "final_tp": f"{c.final_tp:.3f}",
                "loudness_within_tolerance": "true" if c.loudness_within_tolerance else "false",
                "tp_limited": "true" if c.tp_limited else "false",
                "generated_at": now, "status": c.status,
            })


def _render_html(senses: list[dict], clips: list[ClipOut], pairing: dict[str, dict]) -> None:
    """Build audit/11_0_pilot_v3.html with the rich-row layout."""
    # Index clips by (sense_id, clip_type)
    clip_by_key: dict[tuple[str, str], ClipOut] = {(c.sense_id, c.clip_type): c for c in clips}

    # Get Flash URLs from the main manifest (read once)
    main = read_tsv(MAIN_MANIFEST)
    flash_url: dict[tuple[str, str], str] = {}
    en_url: dict[str, str] = {}
    for r in main:
        if r["clip_type"] in ("word", "example"):
            flash_url[(r["sense_id"], r["clip_type"])] = r["url"]
        elif r["clip_type"] == "en_ex":
            en_url[r["sense_id"]] = r["url"]

    # Voice display names — mine from API once or hardcode
    name_by_voice: dict[str, str] = {}
    for bp, p in pairing.items():
        # Use BP gender+pool as label; EN name is in pairing
        name_by_voice[bp] = f"{p['gender']} pool {p['pool_index']}"

    # Aggregate metrics
    total_clips = len(clips)
    asr_pass = sum(1 for c in clips if c.asr_decision == "pass")
    asr_regen = sum(1 for c in clips if c.asr_decision == "regen")
    tol_pass = sum(1 for c in clips if c.loudness_within_tolerance)
    tp_lim = sum(1 for c in clips if c.tp_limited)
    upload_ok = sum(1 for c in clips if c.status == "uploaded")
    upload_fail = sum(1 for c in clips if c.status == "failed")

    def badge(text: str, kind: str) -> str:
        return f'<span class="badge {kind}">{_html.escape(text)}</span>'

    def asr_cell(c: ClipOut | None) -> str:
        if c is None:
            return '<span class="muted">—</span>'
        if c.asr_decision == "pass":
            tag = badge("pass", "ok")
        elif c.asr_decision == "regen":
            tag = badge("regen", "no")
        else:
            tag = badge(c.asr_decision or "?", "warn")
        pd = (f", pdist {c.asr_phonetic_distance:.2f}"
              if c.asr_phonetic_distance is not None else "")
        notes = f'<br><span class="muted small">{_html.escape(c.asr_notes)}</span>' if c.asr_notes else ""
        return (f'<div class="asr">'
                f'<div class="ttx">{_html.escape(c.asr_transcript or "—")}</div>'
                f'<div class="metrics">sim {c.asr_similarity:.2f}{pd} {tag}</div>'
                f'{notes}'
                f'</div>')

    def loud_cell(c: ClipOut | None) -> str:
        if c is None:
            return '<span class="muted">—</span>'
        tol = badge("ok", "ok") if c.loudness_within_tolerance else (
            badge("tp", "warn") if c.tp_limited else badge("drift", "no")
        )
        return (f'<div class="num">{c.final_lufs:.2f} LUFS</div>'
                f'<div class="muted small">TP {c.final_tp:.2f} · '
                f'gain {c.applied_gain_db:+.2f}</div>'
                f'{tol}')

    def audio_cell(url: str) -> str:
        if not url:
            return '<span class="muted">—</span>'
        return f'<audio controls preload="none" src="{_html.escape(url)}"></audio>'

    rows_html: list[str] = []
    for s in sorted(senses, key=lambda x: x["sense_id"]):
        sid = s["sense_id"]
        v3_word = clip_by_key.get((sid, "word"))
        v3_ex = clip_by_key.get((sid, "example"))
        bp_voice_lbl = name_by_voice.get(s["bp_voice_id"], s["bp_voice_id"][:10])
        en_voice = pairing.get(s["bp_voice_id"], {})
        en_voice_lbl = (en_voice.get("en_name") or
                        en_voice.get("en_voice_id", "")[:10] or "—")
        en_voice_id_short = en_voice.get("en_voice_id", "")[:10] + ".." if en_voice.get("en_voice_id") else ""

        rows_html.append(f"""
        <tr>
          <td><code>{_html.escape(sid)}</code><br>
              <span class="muted small">rank {s["rank"]}</span></td>
          <td><strong>{_html.escape(s["pt_display"])}</strong></td>
          <td>{_html.escape(s["example_pt"])}</td>
          <td><span class="muted small">{_html.escape(s["en_primary"])}</span><br>
              <em>{_html.escape(s["example_en"])}</em></td>
          <td>BP: <code>{_html.escape(s["bp_voice_id"][:10])}..</code><br>
              <span class="muted small">{_html.escape(bp_voice_lbl)}</span><br>
              EN: <strong>{_html.escape(en_voice_lbl)}</strong><br>
              <code class="small">{_html.escape(en_voice_id_short)}</code></td>
          <td>{audio_cell(flash_url.get((sid, "word"), ""))}</td>
          <td>{audio_cell(v3_word.url if v3_word else "")}</td>
          <td>{audio_cell(flash_url.get((sid, "example"), ""))}</td>
          <td>{audio_cell(v3_ex.url if v3_ex else "")}</td>
          <td>{audio_cell(en_url.get(sid, ""))}</td>
          <td>{asr_cell(v3_word)}</td>
          <td>{asr_cell(v3_ex)}</td>
          <td>{loud_cell(v3_word)}</td>
          <td>{loud_cell(v3_ex)}</td>
          <td class="verdict">
            <label><input type="radio" name="v_{_html.escape(sid)}" value="GO"> GO</label><br>
            <label><input type="radio" name="v_{_html.escape(sid)}" value="REGEN_VOICE_SWAP"> swap</label><br>
            <label><input type="radio" name="v_{_html.escape(sid)}" value="TAG_ALIAS"> alias</label><br>
            <label><input type="radio" name="v_{_html.escape(sid)}" value="HOLD"> HOLD</label>
          </td>
        </tr>""")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 11.0 — v3 pilot ({len(senses)} senses)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 2100px; margin: 0 auto; padding: 0 1em; line-height: 1.4; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
.toolbar {{ position: sticky; top: 0; background: #ffd; padding: 0.7em 1em;
  border: 2px solid #ba0; border-radius: 4px; margin: 0.5em 0;
  display: flex; align-items: center; gap: 1.5em; flex-wrap: wrap; z-index: 100;
  font-size: 0.92em; }}
.metric {{ display: inline-block; }}
.metric strong {{ font-size: 1.1em; color: #06598f; }}
button {{ font-size: 1em; padding: 0.4em 0.8em; cursor: pointer; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
th, td {{ border: 1px solid #aaa; padding: 0.35em 0.5em; text-align: left;
  vertical-align: top; }}
th {{ background: #e6ecf2; position: sticky; top: 4em; z-index: 5;
  font-size: 0.88em; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px;
  font-size: 0.92em; }}
audio {{ width: 180px; height: 28px; }}
.muted {{ color: #888; }}
.small {{ font-size: 0.85em; }}
em {{ color: #06598f; font-style: italic; }}
.num {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.95em; }}
.asr .ttx {{ font-style: italic; color: #444; max-width: 200px; }}
.asr .metrics {{ font-family: ui-monospace, monospace; font-size: 0.82em;
  margin-top: 0.2em; }}
.badge {{ display: inline-block; padding: 0.1em 0.4em; border-radius: 3px;
  font-size: 0.78em; font-weight: bold; }}
.badge.ok {{ background: #d4edda; color: #155724; }}
.badge.warn {{ background: #fff3cd; color: #664d03; }}
.badge.no {{ background: #fde2e2; color: #842029; }}
.verdict label {{ display: block; white-space: nowrap; }}
.verdict {{ min-width: 110px; }}
#export-output {{ width: 100%; height: 180px; font-family: ui-monospace, monospace;
  font-size: 0.85em; margin-top: 0.5em; }}
</style>
</head>
<body>

<h1>Stage 11.0 — Eleven v3 pilot ({len(senses)} senses · {total_clips} clips)</h1>

<div class="toolbar">
  <span class="metric">Clips: <strong>{total_clips}</strong></span>
  <span class="metric">ASR pass: <strong>{asr_pass}</strong> / regen: <strong>{asr_regen}</strong> ({100*asr_pass//max(total_clips,1)}%)</span>
  <span class="metric">Loudness within ±1 LU: <strong>{tol_pass}</strong> ({100*tol_pass//max(total_clips,1)}%)</span>
  <span class="metric">TP-limited: <strong>{tp_lim}</strong></span>
  <span class="metric">R2 upload: <strong>{upload_ok}</strong> ok / <strong>{upload_fail}</strong> fail</span>
  <button onclick="exportVerdicts()">Export verdicts</button>
</div>

<p class="muted small" style="margin-top: 0.5em;">Listen to <strong>Flash word</strong> vs <strong>v3 word</strong> (and same for example) for each row. The EN example column is the back-of-card audio from Stage 10 (Flash v2.5) for cross-side feel. Mark each row GO / swap / alias / HOLD. Export to a CSV at the bottom.</p>

<textarea id="export-output" readonly placeholder="Click 'Export verdicts' — a CSV appears here. Copy into a TSV file or paste back into chat."></textarea>

<table>
  <thead>
    <tr>
      <th>sense_id</th>
      <th>BP word</th>
      <th>BP example</th>
      <th>EN gloss<br>EN example</th>
      <th>voices</th>
      <th>Flash word</th>
      <th>v3 word</th>
      <th>Flash example</th>
      <th>v3 example</th>
      <th>EN example<br><span class="muted small">(Stage 10)</span></th>
      <th>v3 ASR<br>(word)</th>
      <th>v3 ASR<br>(example)</th>
      <th>v3 LUFS<br>(word)</th>
      <th>v3 LUFS<br>(example)</th>
      <th>verdict</th>
    </tr>
  </thead>
  <tbody>{"".join(rows_html)}
  </tbody>
</table>

<script>
// Cross-row audio pause: when one starts, pause the others.
document.querySelectorAll('audio').forEach(a => {{
  a.addEventListener('play', () => {{
    document.querySelectorAll('audio').forEach(o => {{ if (o !== a) o.pause(); }});
  }});
}});

function exportVerdicts() {{
  const lines = ['sense_id\\tverdict'];
  const rows = document.querySelectorAll('tbody tr');
  rows.forEach(r => {{
    const sid = r.querySelector('td code').textContent;
    const sel = r.querySelector('input[type=radio]:checked');
    lines.push(sid + '\\t' + (sel ? sel.value : ''));
  }});
  document.getElementById('export-output').value = lines.join('\\n');
}}
</script>
</body>
</html>
""", encoding="utf-8")
    print(f"Wrote {HTML_OUT}")


class _TeeStdout:
    def __init__(self, *streams) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for s in self.streams:
                try:
                    s.write(data); s.flush()
                except Exception:  # noqa: BLE001
                    pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try: s.flush()
            except Exception: pass  # noqa: BLE001


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="Number of senses to render (default 100; renders 2 clips each)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="Pro `standard` tier limit is 10")
    ap.add_argument("--yes", action="store_true", help="Skip GO prompt")
    ap.add_argument("--dry-run", action="store_true", help="Pre-flight only")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set", file=sys.stderr)
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (needed for ASR)", file=sys.stderr)
        return 1

    pairing = _load_voice_pairing()
    bp_voices = list(pairing.keys())

    # 1. Pre-flight: verify all 8 voices reachable on v3
    print(f"\n=== Pre-flight: verify {len(bp_voices)} BP voices on v3 ===\n")
    reachable, fail = _verify_voices_on_v3(bp_voices, api_key)
    print(f"\n{reachable}/{len(bp_voices)} reachable, {fail} unreachable")
    if fail:
        print("ERROR: some voices unreachable (HTTP error). Abort.", file=sys.stderr)
        return 1
    print("Note: voices not HQ-tuned for v3 still render; pilot will reveal quality.")

    # 2. Pick senses
    senses = _select_pilot_senses(args.n, pairing)
    n_chars = sum(len(s["pt_display"]) + len(s["example_pt"]) for s in senses)
    n_clips = len(senses) * 2

    # 3. Dict locator
    locator = _latest_dict_locator()
    locator_repr = locator["pronunciation_dictionary_id"][:8] + ".." if locator else "(none)"

    print(f"\n=== Pre-flight: pilot budget ===")
    print(f"  Senses:        {len(senses)}")
    print(f"  Clips:         {n_clips}  (word + example)")
    print(f"  Chars:         {n_chars:,}")
    print(f"  Credits (est): {n_chars} (v3 = 1.0 credits/char)")
    print(f"  Concurrency:   {args.concurrency}")
    print(f"  Pron dict:     {locator_repr}")
    print(f"  R2 prefix:     {R2_PILOT_PREFIX}/")
    print(f"  Side TSV:      {PILOT_MANIFEST.name}")
    print(f"  HTML:          {HTML_OUT.name}")
    print()

    if args.dry_run:
        print("--dry-run: stopping.")
        return 0

    if not args.yes:
        sys.stdout.write("Type GO to proceed: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "GO":
            print("Cancelled.")
            return 0

    # 4. Tee stdout to log
    log = LOG_PATH.open("a", encoding="utf-8")
    log.write(f"\n=== stage 11.0 pilot started {_now_iso()} | n={len(senses)} ===\n")
    log.flush()
    original_stdout = sys.stdout
    sys.stdout = _TeeStdout(original_stdout, log)  # type: ignore[assignment]

    # 5. Build clients
    r2 = R2Client()
    el = ElevenLabsClient(
        model_id=V3_MODEL_ID,
        language_code=LANGUAGE,
        pronunciation_dict_locators=[locator] if locator else None,
    )
    asr = AsrClient()

    # 6. Render all 2N clips
    t_start = time.time()
    work = [(s, ct) for s in senses for ct in ("word", "example")]
    clips: list[ClipOut] = []

    def _worker(item):
        sense, ct = item
        return _process_one_clip(sense=sense, clip_type=ct, el=el, asr=asr, r2=r2)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(_worker, w): w for w in work}
        for fut in as_completed(futs):
            try:
                c = fut.result()
                clips.append(c)
                tag = "OK " if c.status == "uploaded" else "FAIL"
                line = (f"[{len(clips):>3}/{len(work)}] {tag} {c.sense_id} {c.clip_type:<7} "
                        f"voice={c.voice_id[:8]}.. lufs={c.final_lufs:.2f} "
                        f"asr={c.asr_decision} sim={c.asr_similarity:.2f}")
                if c.error:
                    line += f" err={c.error[:60]}"
                print(line, flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[err] worker raised {type(exc).__name__}: {exc}", flush=True)

    elapsed = time.time() - t_start

    # 7. Write side manifest
    _write_side_manifest(clips)
    print(f"\nWrote {PILOT_MANIFEST}  ({len(clips)} rows)")

    # 8. Build HTML
    _render_html(senses, clips, pairing)

    # 9. Summary
    uploaded = sum(1 for c in clips if c.status == "uploaded")
    failed = len(clips) - uploaded
    asr_pass = sum(1 for c in clips if c.asr_decision == "pass")
    asr_regen = sum(1 for c in clips if c.asr_decision == "regen")
    tol = sum(1 for c in clips if c.loudness_within_tolerance)
    tp_lim = sum(1 for c in clips if c.tp_limited)
    total_asr_cost = sum(c.asr_cost_usd for c in clips)
    print()
    print("=== Stage 11.0 pilot summary ===")
    print(f"  Total:        {len(clips)} clips")
    print(f"  Uploaded:     {uploaded}")
    print(f"  Failed:       {failed}")
    print(f"  ASR pass:     {asr_pass}/{len(clips)} ({100*asr_pass/max(len(clips),1):.0f}%)")
    print(f"  ASR regen:    {asr_regen}")
    print(f"  Loudness ok:  {tol}/{len(clips)} ({100*tol/max(len(clips),1):.0f}%)")
    print(f"  TP-limited:   {tp_lim}")
    print(f"  ASR cost:     ${total_asr_cost:.4f}")
    print(f"  Wall:         {elapsed:.1f}s "
          f"({elapsed/max(len(clips),1):.2f}s/clip)")
    print(f"\nOpen {HTML_OUT} in a browser to listen + decide GO/HOLD for 11.1.")

    sys.stdout = original_stdout  # type: ignore[assignment]
    log.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
