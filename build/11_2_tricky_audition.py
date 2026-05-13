"""Stage 11 / Step 2 — Tricky-word audition: flash vs v3 × all 6 male BP voices.

Five EP-vs-BP differentiator entries are rendered on EVERY male BP voice
in config/voices.tsv with both models. Output is an HTML for listening.

Tricky entries:
  0033.00.01 dizer    'Ela sempre diz a verdade.'
  0200.00.03 real     'O dólar está valendo cinco reais hoje.'
  0777.00.01 hospital 'Meu pai está no hospital desde ontem.'
  0281.00.01 gente    'A gente precisa de água para viver.'
  3789.00.01 tia      'Minha tia mora em São Paulo com dois gatos.'

Pipeline per (sense, voice, model, clip_type):
  ElevenLabs PCM → loudnorm to -16 LUFS → MP3 → R2 (audio_pilot_tricky/)

No ASR, no audit JSONL — this is a listening tool, not production.

Cost: ~120 clips × ~$0.003 ≈ ~$0.40 (v3 dominates).
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import escape
from pathlib import Path

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.r2_client import R2Client  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FINAL_TSV = REPO / "data" / "06-final.tsv"
VOICES_TSV = REPO / "config" / "voices.tsv"
OUT_HTML = REPO / "audit" / "11_2_tricky_audition.html"

FLASH_MODEL = "eleven_flash_v2_5"
V3_MODEL = "eleven_v3"
R2_PREFIX = "audio_pilot_tricky"
MAX_WORKERS = 8

TRICKY_SENSE_IDS = [
    "0033.00.01",  # dizer  — Lair v3 EP drift case
    "0200.00.03",  # real   — multiple senses, final [l] vs [w]
    "0777.00.01",  # hospital — common medical word, stressed [al]
    "0281.00.01",  # gente  — BP [ʒẽtʃi] vs EP palatalization marker
    "3789.00.01",  # tia    — ti- palatalization marker
]


@dataclass
class Clip:
    sense_id: str
    pt: str
    en: str
    text: str
    clip_type: str   # 'word' | 'example'
    voice_id: str
    voice_label: str  # e.g., 'M#1 Eduardo'
    model_id: str    # flash or v3
    url: str = ""
    lufs: float = 0.0
    error: str = ""


def load_senses() -> dict[str, dict]:
    with FINAL_TSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, dialect="excel-tab"))
    by_sid = {r["sense_id"]: r for r in rows}
    return {sid: by_sid[sid] for sid in TRICKY_SENSE_IDS}


def load_male_voices() -> list[dict]:
    with VOICES_TSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, dialect="excel-tab"))
    males = [r for r in rows if r["gender"] == "male"]
    males.sort(key=lambda r: int(r["pool_index"]))
    return males


def build_jobs(senses: dict[str, dict], voices: list[dict]) -> list[Clip]:
    jobs: list[Clip] = []
    for sid, sense in senses.items():
        for clip_type in ("word", "example"):
            text = sense["pt_display"] if clip_type == "word" else sense["example_pt"]
            for v in voices:
                label = f"M#{v['pool_index']} {v['bp_name']}"
                for model in (FLASH_MODEL, V3_MODEL):
                    jobs.append(Clip(
                        sense_id=sid, pt=sense["pt"], en=sense["en_primary"],
                        text=text, clip_type=clip_type,
                        voice_id=v["voice_id"], voice_label=label,
                        model_id=model,
                    ))
    return jobs


def run_clip(job: Clip, r2: R2Client) -> Clip:
    short = "word" if job.clip_type == "word" else "ex"
    key = f"{R2_PREFIX}/{job.sense_id}-{short}-{job.model_id}-{job.voice_id}.mp3"
    try:
        el = ElevenLabsClient(model_id=job.model_id)
        tts = el.generate_pcm(
            text=job.text, voice_id=job.voice_id,
            sense_id=job.sense_id, clip_type=job.clip_type, version=1,
        )
        norm = normalize_pcm_to_mp3_verified(tts.audio_pcm)
        up = r2.upload_bytes(
            norm.mp3_bytes, key,
            extra_metadata={
                "sense_id": job.sense_id, "clip_type": job.clip_type,
                "voice_id": job.voice_id, "model": job.model_id,
                "stage": "11_2_tricky_audition",
            },
        )
        job.url = up.url
        job.lufs = norm.final_mp3_lufs
    except Exception as exc:  # noqa: BLE001
        job.error = f"{type(exc).__name__}: {exc}"
    return job


def render_html(senses: dict[str, dict], voices: list[dict],
                clips_by_key: dict[tuple, Clip]) -> str:
    """Render a grouped audition page.

    For each sense: a word-row section + example-row section.
    Rows = voices (M#1..M#6); columns = flash | v3.
    Each cell holds an <audio controls> tag.
    """
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>11.2 tricky-word audition — flash vs v3 × 6 male BP voices</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;",
        "     max-width:1100px;margin:24px auto;padding:0 20px;color:#222;}",
        "h1{font-size:1.4em;margin-bottom:.2em;}",
        "h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.25em;}",
        "h3{margin-top:1.3em;font-size:1.05em;color:#555;}",
        ".meta{color:#777;font-size:.92em;margin-bottom:1em;}",
        "table{border-collapse:collapse;margin:.6em 0 1.2em;width:100%;}",
        "th,td{border:1px solid #e0e0e0;padding:8px 10px;vertical-align:middle;}",
        "th{background:#f7f7f7;text-align:left;font-weight:600;}",
        "td.voice{white-space:nowrap;font-family:ui-monospace,monospace;color:#444;}",
        "audio{width:260px;}",
        ".err{color:#b00;font-size:.85em;font-family:ui-monospace,monospace;}",
        ".lufs{color:#999;font-size:.78em;margin-left:6px;}",
        ".sentence{font-style:italic;color:#444;}",
        "</style></head><body>",
        "<h1>Stage 11.2 — Tricky-word audition: flash vs v3</h1>",
        f"<p class='meta'>5 tricky entries × 6 male BP voices × 2 models × 2 clip types "
        f"= {len(clips_by_key)} clips. All loudnorm -16 LUFS, "
        f"R2 prefix <code>{R2_PREFIX}/</code>.</p>",
    ]

    for sid, sense in senses.items():
        parts.append(f"<h2>{sid} &middot; <em>{escape(sense['pt'])}</em> "
                     f"<span class='meta'>({escape(sense['en_primary'])})</span></h2>")
        for clip_type in ("word", "example"):
            text = sense["pt_display"] if clip_type == "word" else sense["example_pt"]
            title = "word" if clip_type == "word" else "example sentence"
            parts.append(f"<h3>{title}: <span class='sentence'>{escape(text)}</span></h3>")
            parts.append("<table><thead><tr><th>voice</th>"
                         "<th>flash (eleven_flash_v2_5)</th>"
                         "<th>v3 (eleven_v3)</th></tr></thead><tbody>")
            for v in voices:
                label = f"M#{v['pool_index']} {v['bp_name']}"
                parts.append(f"<tr><td class='voice'>{escape(label)}</td>")
                for model in (FLASH_MODEL, V3_MODEL):
                    c = clips_by_key.get(
                        (sid, clip_type, v["voice_id"], model))
                    if c is None or c.error:
                        msg = escape(c.error) if c else "missing"
                        parts.append(f"<td><span class='err'>{msg}</span></td>")
                    else:
                        parts.append(
                            f"<td><audio controls preload='none' src='{c.url}'></audio>"
                            f"<span class='lufs'>{c.lufs:+.2f} LUFS</span></td>")
                parts.append("</tr>")
            parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    senses = load_senses()
    voices = load_male_voices()
    print(f"  senses:   {len(senses)} ({', '.join(senses)})")
    print(f"  voices:   {len(voices)} male BP "
          f"({', '.join(v['bp_name'] for v in voices)})")

    jobs = build_jobs(senses, voices)
    print(f"  clips:    {len(jobs)} = "
          f"{len(senses)} senses × {len(voices)} voices × 2 models × 2 clip_types")

    if "--dry-run" in sys.argv:
        return 0

    r2 = R2Client()
    print(f"  rendering with {MAX_WORKERS} workers...")
    results: list[Clip] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(run_clip, j, r2) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            c = fut.result()
            results.append(c)
            tag = "OK" if not c.error else "FAIL"
            voice_short = c.voice_id[:8]
            print(f"  [{i:>3}/{len(jobs)}] {tag:<4} {c.sense_id} {c.clip_type:<7} "
                  f"{c.model_id:<18} v={voice_short}.. {c.error}")

    ok = sum(1 for c in results if not c.error)
    print(f"  rendered: {ok}/{len(results)} OK")

    clips_by_key = {
        (c.sense_id, c.clip_type, c.voice_id, c.model_id): c
        for c in results
    }
    html = render_html(senses, voices, clips_by_key)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"  wrote:    {OUT_HTML}")
    print(f"  open with: open {OUT_HTML}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
