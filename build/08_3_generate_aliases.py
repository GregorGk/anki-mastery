"""Stage 8 / Step 3 — Alias generation.

Three sub-modes:

1. `--sentinel-smoke` — Render every candidate respelling from
   `data/_pronunciation_alias_seeds.tsv` via ElevenLabs (one voice,
   throwaway audio, no R2 upload). Build
   `audit/08_3_sentinel_smoke.html` with audio elements + reviewer guide
   + per-family radio voting. User listens, picks the winning candidate
   per family, exports TSV → `data/_pronunciation_alias_template_winners.tsv`.

2. `--apply-templates` — Read user-picked winners. Convert to mechanical
   family rules and write `data/_pronunciation_alias_locked_templates.tsv`.

3. `--fill-individuals --confirm` — Read
   `data/_audio_mispronunciation_confirmed.tsv` (output of 08_2). For
   each unique `pt`:
     a) If pt's risk_patterns matches a locked family template, apply
        the template mechanically.
     b) Else, call Sonnet 4.5 per-pt for a custom respelling.
   Write `data/_pronunciation_aliases.tsv` (deduped by pt) and
   `data/_pronunciation_alias_applications.tsv` (per sense_id).
   Detect conflicts (same pt → different respellings) and route to
   `data/_pronunciation_alias_conflicts.tsv` (blocks 08_4 upload).
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402
from build.lib.tsv import read_tsv, write_tsv  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = REPO_ROOT / "audit"
DOCS_DIR = REPO_ROOT / "docs"
SENTINEL_CACHE = REPO_ROOT / "build" / "audio_cache" / "sentinels"

SEEDS_PATH = DATA_DIR / "_pronunciation_alias_seeds.tsv"
WINNERS_PATH = DATA_DIR / "_pronunciation_alias_template_winners.tsv"
LOCKED_TEMPLATES_PATH = DATA_DIR / "_pronunciation_alias_locked_templates.tsv"
CONFIRMED_PATH = DATA_DIR / "_audio_mispronunciation_confirmed.tsv"
ALIASES_PATH = DATA_DIR / "_pronunciation_aliases.tsv"
APPLICATIONS_PATH = DATA_DIR / "_pronunciation_alias_applications.tsv"
CONFLICTS_PATH = DATA_DIR / "_pronunciation_alias_conflicts.tsv"
RISK_PATH = DATA_DIR / "_audio_risk_classification.tsv"
SMOKE_HTML_PATH = AUDIT_DIR / "08_3_sentinel_smoke.html"
REVIEWER_GUIDE_PATH = DOCS_DIR / "reviewer_guide.md"

# Sentinel smoke renders use a stable known-good BP-only voice
SENTINEL_VOICE_ID = "4r3G9XKliGgVZLKMgjik"  # Lair (pool 6)


# ---------------------------------------------------------------------------
# Sentinel smoke: render candidates → HTML
# ---------------------------------------------------------------------------
def _render_sentinel(client: ElevenLabsClient, text: str, voice_id: str, sentinel_id: str, candidate_n: str) -> bytes:
    """Render one sentinel candidate. Returns MP3 bytes (no R2 upload)."""
    cache_key = f"sentinel-{sentinel_id}-{candidate_n}-{voice_id[:8]}.mp3"
    cache_path = SENTINEL_CACHE / cache_key
    if cache_path.exists():
        return cache_path.read_bytes()

    res = client.generate_pcm(
        text=text,
        voice_id=voice_id,
        sense_id=f"sentinel-{sentinel_id}-{candidate_n}",
        clip_type="word",
        version=1,
    )
    norm = normalize_pcm_to_mp3_verified(res.audio_pcm)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(norm.mp3_bytes)
    return norm.mp3_bytes


def cmd_sentinel_smoke() -> int:
    """Render every candidate from seeds + build smoke HTML."""
    if not SEEDS_PATH.exists():
        print(f"ERROR: seeds file not found at {SEEDS_PATH}", file=sys.stderr)
        return 1
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set in environment.", file=sys.stderr)
        return 1

    rows = read_tsv(SEEDS_PATH)
    seeds = [r for r in rows if r.get("pt_respelling")]

    client = ElevenLabsClient()
    SENTINEL_CACHE.mkdir(parents=True, exist_ok=True)

    # Render all candidates
    print(f"Rendering {len(seeds)} sentinel candidates with voice={SENTINEL_VOICE_ID}...")
    rendered: list[dict] = []
    for i, r in enumerate(seeds):
        family = r.get("family_name", "")
        rep_pt = r.get("representative_pt", "")
        respelling = r.get("pt_respelling", "")
        pron_guide = r.get("pronunciation_guide", "")
        candidate_n = r.get("candidate_n", "0")
        sentinel_id = r.get("family_id", "0")
        try:
            mp3 = _render_sentinel(
                client, respelling, SENTINEL_VOICE_ID, sentinel_id, candidate_n
            )
            audio_b64 = base64.b64encode(mp3).decode("ascii")
            rendered.append({
                "family_id": sentinel_id,
                "family_name": family,
                "representative_pt": rep_pt,
                "candidate_n": candidate_n,
                "pt_respelling": respelling,
                "pronunciation_guide": pron_guide,
                "rationale": r.get("rationale", ""),
                "confidence": r.get("confidence", ""),
                "is_recommended": r.get("is_recommended", "false"),
                "is_anti_example": r.get("is_anti_example", "false"),
                "audio_b64": audio_b64,
            })
            print(f"  [{i+1}/{len(seeds)}] {family} {rep_pt} → {respelling}  ({len(mp3)} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{len(seeds)}] FAILED {family} {respelling}: {exc}", file=sys.stderr)

    # Group by family for the HTML
    by_family: dict[str, list[dict]] = {}
    for r in rendered:
        by_family.setdefault(r["family_id"], []).append(r)

    return _build_smoke_html(by_family)


def _build_smoke_html(by_family: dict[str, list[dict]]) -> int:
    """Write the sentinel smoke HTML and exit."""
    # Inline the md→html converter from 08_2 to avoid cross-module mess
    sys.path.insert(0, str(REPO_ROOT / "build"))
    # Import md_to_html lazily by re-reading 08_2's source (duck-import)
    spec_path = REPO_ROOT / "build" / "08_2_calibration.py"
    if not spec_path.exists():
        print("ERROR: build/08_2_calibration.py missing — needed for md_to_html", file=sys.stderr)
        return 1
    import importlib.util
    spec = importlib.util.spec_from_file_location("calibration", spec_path)
    if spec is None or spec.loader is None:
        print("ERROR: failed to load 08_2_calibration.py", file=sys.stderr)
        return 1
    cal_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal_mod)
    md_to_html = cal_mod.md_to_html  # type: ignore[attr-defined]

    if REVIEWER_GUIDE_PATH.exists():
        guide_html = md_to_html(REVIEWER_GUIDE_PATH.read_text(encoding="utf-8"))
    else:
        guide_html = "<p><em>(reviewer guide not found)</em></p>"

    family_blocks: list[str] = []
    for family_id in sorted(by_family.keys(), key=int):
        candidates = sorted(by_family[family_id], key=lambda c: c["candidate_n"])
        family_name = candidates[0]["family_name"]
        rep_pt = candidates[0]["representative_pt"]
        radio_name = f"family_{family_id}_winner"

        cards: list[str] = []
        for c in candidates:
            anti = c["is_anti_example"].lower() == "true"
            recommended = c["is_recommended"].lower() == "true"
            badge = ""
            if recommended:
                badge = '<span class="badge rec">recommended</span>'
            elif anti:
                badge = '<span class="badge anti">anti-example (do NOT pick)</span>'
            cards.append(f"""
<div class="candidate {'recommended' if recommended else ''} {'anti' if anti else ''}">
  <header>
    <input type="radio" name="{radio_name}" value="{html.escape(c['pt_respelling'])}"
      data-respelling="{html.escape(c['pt_respelling'])}"
      data-confidence="{html.escape(c['confidence'])}"
      data-anti="{anti}">
    <strong>{html.escape(c['pt_respelling'])}</strong> {badge}
  </header>
  <div class="guide-text"><em>Should sound like:</em> {html.escape(c['pronunciation_guide'])}</div>
  <div class="rationale"><em>Why:</em> {html.escape(c['rationale'])}</div>
  <audio controls preload="none" src="data:audio/mp3;base64,{c['audio_b64']}"></audio>
</div>
""")

        family_blocks.append(f"""
<section class="family">
  <h3>Family {family_id}: <code>{html.escape(family_name)}</code> — representative <em>{html.escape(rep_pt)}</em></h3>
  <p>Pick the candidate whose audio matches the pronunciation guide. Skip the anti-examples (they're calibration sanity checks).</p>
  {''.join(cards)}
  <div class="reject"><label><input type="radio" name="{radio_name}" value="REJECT_FAMILY"> None of these match — reject this family template</label></div>
</section>
""")

    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — Sentinel smoke test</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 920px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1, h2, h3, h4 {{ color: #111; margin-top: 1.6em; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
.guide {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin-bottom: 2em; font-size: 0.95em; }}
.md-table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }}
.md-table th, .md-table td {{ border: 1px solid #aaa; padding: 0.3em 0.7em; text-align: left; vertical-align: top; }}
.md-table th {{ background: #e6ecf2; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
.family {{ border-left: 4px solid #468; padding: 0.6em 1em; margin: 1.2em 0; background: #f8f9fa; }}
.candidate {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.6em 1em; margin: 0.6em 0; background: white; }}
.candidate.recommended {{ border-color: #2a6; background: #f3fbf5; }}
.candidate.anti {{ border-color: #d44; background: #fff5f5; opacity: 0.85; }}
.candidate header {{ font-size: 1.1em; }}
.badge {{ display: inline-block; font-size: 0.75em; padding: 0.1em 0.5em; border-radius: 3px;
  margin-left: 0.5em; vertical-align: middle; }}
.badge.rec {{ background: #2a6; color: white; }}
.badge.anti {{ background: #d44; color: white; }}
.guide-text, .rationale {{ font-size: 0.9em; color: #444; margin: 0.3em 0; }}
.reject {{ margin-top: 0.8em; font-style: italic; color: #844; }}
audio {{ width: 100%; margin: 0.4em 0; }}
.export {{ position: sticky; top: 0; background: #ffd; padding: 1em; border: 2px solid #ba0;
  border-radius: 4px; margin: 1em 0; z-index: 10; }}
button {{ font-size: 1em; padding: 0.5em 1em; cursor: pointer; }}
#export-output {{ width: 100%; height: 200px; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>

<h1>Stage 8 — Sentinel smoke test</h1>

<p>For each risk family below, listen to the candidate respellings and pick
the one whose audio matches the <em>pronunciation guide</em>. Skip
candidates marked anti-example (those are calibration controls — they
should sound clearly wrong; pick a non-anti candidate).</p>

<div class="export">
  <button onclick="exportWinners()">Export winners (copy TSV to clipboard)</button>
  <span id="export-status" style="margin-left: 1em; color: #060;"></span>
  <details style="margin-top: 0.6em;">
    <summary>Show raw TSV</summary>
    <textarea id="export-output" readonly></textarea>
  </details>
</div>

<div class="guide">
{guide_html}
</div>

<h2>Family templates</h2>

{family_blocks}

<script>
function exportWinners() {{
  const rows = ["family_id\\tfamily_name\\trepresentative_pt\\twinning_respelling"];
  document.querySelectorAll('section.family').forEach(s => {{
    const familyId = s.querySelector('h3 code').textContent;
    const familyName = familyId; // already rendered as code
    const rep = s.querySelector('h3 em').textContent;
    const sel = s.querySelector('input[type=radio]:checked');
    const winner = sel ? sel.value : "UNANSWERED";
    rows.push([familyId, familyName, rep, winner].join('\\t'));
  }});
  const tsv = rows.join('\\n');
  document.getElementById('export-output').value = tsv;
  navigator.clipboard.writeText(tsv).then(() => {{
    document.getElementById('export-status').textContent = '✓ Copied to clipboard.';
  }}, () => {{
    document.getElementById('export-status').textContent = 'Copy failed — copy manually from the textarea below.';
  }});
}}
</script>

</body>
</html>
"""

    out = template.format(
        guide_html=guide_html,
        family_blocks="\n".join(family_blocks),
    )
    SMOKE_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_HTML_PATH.write_text(out, encoding="utf-8")
    print(f"\nWrote {SMOKE_HTML_PATH}")
    print(f"  open with:  open {SMOKE_HTML_PATH}")
    return 0


# ---------------------------------------------------------------------------
# Apply user-picked winners → locked family templates
# ---------------------------------------------------------------------------
def cmd_apply_templates() -> int:
    if not WINNERS_PATH.exists():
        print(
            f"ERROR: winners file not found at {WINNERS_PATH}. "
            "Paste the exported TSV from sentinel-smoke HTML there first.",
            file=sys.stderr,
        )
        return 1
    rows = read_tsv(WINNERS_PATH)
    seeds = read_tsv(SEEDS_PATH)
    seeds_by_id = {(s["family_id"], s["pt_respelling"]): s for s in seeds}

    locked: list[dict] = []
    for r in rows:
        fid = r.get("family_id", "").strip()
        winner = r.get("winning_respelling", "").strip()
        family_name = r.get("family_name", "").strip()
        rep = r.get("representative_pt", "").strip()
        if winner in ("", "UNANSWERED", "REJECT_FAMILY"):
            continue
        # Look up the seed for rationale + guide
        s = seeds_by_id.get((fid, winner), {})
        locked.append({
            "family_id": fid,
            "family_name": family_name,
            "representative_pt": rep,
            "winning_respelling": winner,
            "pronunciation_guide": s.get("pronunciation_guide", ""),
            "rationale": s.get("rationale", ""),
            "confidence": s.get("confidence", ""),
            "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        })

    fns = [
        "family_id", "family_name", "representative_pt", "winning_respelling",
        "pronunciation_guide", "rationale", "confidence", "locked_at",
    ]
    write_tsv(LOCKED_TEMPLATES_PATH, locked, fieldnames=fns)
    print(f"Wrote {len(locked)} locked family templates to {LOCKED_TEMPLATES_PATH}")
    if not locked:
        print("(no winners — sentinel smoke test produced no actionable templates)")
    return 0


# ---------------------------------------------------------------------------
# Bulk fill: per-pt aliases (template OR Sonnet 4.5 per-word)
# ---------------------------------------------------------------------------
def _apply_template(pt: str, family_name: str, locked_templates: list[dict]) -> str | None:
    """Apply a family template mechanically, if one matches.

    For final_l_*: replace final 'al'/'el'/'il'/'ol' with the template suffix
    pattern derived from the winner. The winner is e.g., 'animau' for the -al
    family with rep 'animal'; we extract the suffix substitution 'al'→'au'
    and apply mechanically.
    """
    for t in locked_templates:
        if t["family_name"] != family_name:
            continue
        rep = t["representative_pt"]
        winner = t["winning_respelling"]
        # Find the divergence point (longest common prefix)
        i = 0
        while i < min(len(rep), len(winner)) and rep[i] == winner[i]:
            i += 1
        old_suffix = rep[i:]
        new_suffix = winner[i:]
        # Apply mechanically
        if pt.endswith(old_suffix):
            return pt[: -len(old_suffix)] + new_suffix
    return None


def cmd_fill_individuals(*, confirmed: bool) -> int:
    if not confirmed:
        print("ERROR: --fill-individuals requires --confirm to actually run.", file=sys.stderr)
        return 1
    if not CONFIRMED_PATH.exists():
        print(
            f"ERROR: confirmed file not found at {CONFIRMED_PATH}. "
            "Run 08_2_calibration.py --apply-calibration first.",
            file=sys.stderr,
        )
        return 1
    confirmed_rows = read_tsv(CONFIRMED_PATH)
    risk_rows = read_tsv(RISK_PATH)
    risk_by_sid = {r["sense_id"]: r for r in risk_rows}
    locked = read_tsv(LOCKED_TEMPLATES_PATH) if LOCKED_TEMPLATES_PATH.exists() else []

    # Build the by-pt alias map (deduped)
    aliases: dict[str, dict] = {}
    applications: list[dict] = []
    conflicts: list[dict] = []

    next_alias_id = 1

    for row in confirmed_rows:
        sid = row.get("sense_id", "")
        pt = row.get("pt", "").strip()
        if not pt:
            continue
        risk = risk_by_sid.get(sid, {})
        patterns = risk.get("risk_patterns", "").split(",")

        # Try each pattern's template
        respelling = None
        applied_family = None
        for p in patterns:
            p = p.strip()
            if not p:
                continue
            cand = _apply_template(pt, p, locked)
            if cand and cand != pt:
                respelling = cand
                applied_family = p
                break

        # If no template matched, this would be a Sonnet per-word call.
        # For now, leave for manual fill (Sonnet integration pending).
        if respelling is None:
            # Skip silently — these will need a per-word LLM pass
            continue

        # Conflict detection
        if pt in aliases:
            existing = aliases[pt]
            if existing["pt_respelling"] != respelling:
                conflicts.append({
                    "pt": pt,
                    "existing_respelling": existing["pt_respelling"],
                    "new_respelling": respelling,
                    "existing_family": existing["applied_family"],
                    "new_family": applied_family,
                    "sense_id_new": sid,
                })
            else:
                # Same respelling — record application but no conflict
                applications.append({
                    "sense_id": sid, "clip_type": "word",
                    "alias_id": existing["alias_id"], "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                })
            continue

        alias_id = next_alias_id
        next_alias_id += 1
        aliases[pt] = {
            "alias_id": alias_id,
            "pt": pt,
            "pt_respelling": respelling,
            "rationale": f"Family template: {applied_family}",
            "applied_family": applied_family,
            "confidence": "high",
            "source": "family_template_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        }
        applications.append({
            "sense_id": sid, "clip_type": "word", "alias_id": alias_id,
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        })

    # Write outputs
    aliases_list = sorted(aliases.values(), key=lambda a: a["alias_id"])
    fns_aliases = ["alias_id", "pt", "pt_respelling", "rationale", "applied_family", "confidence", "source", "created_at"]
    fns_apps = ["sense_id", "clip_type", "alias_id", "applied_at"]
    fns_conflicts = ["pt", "existing_respelling", "new_respelling", "existing_family", "new_family", "sense_id_new"]

    write_tsv(ALIASES_PATH, aliases_list, fieldnames=fns_aliases)
    write_tsv(APPLICATIONS_PATH, applications, fieldnames=fns_apps)
    write_tsv(CONFLICTS_PATH, conflicts, fieldnames=fns_conflicts)

    print(f"Wrote {len(aliases_list)} aliases (deduped by pt) → {ALIASES_PATH}")
    print(f"Wrote {len(applications)} applications → {APPLICATIONS_PATH}")
    print(f"Conflicts: {len(conflicts)}  → {CONFLICTS_PATH}")
    if conflicts:
        print("⚠ Conflicts present; 08_4 upload will be blocked until resolved.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sentinel-smoke", action="store_true",
                        help="Render sentinel candidates + build smoke HTML.")
    parser.add_argument("--apply-templates", action="store_true",
                        help="Read user-picked winners + write locked templates.")
    parser.add_argument("--fill-individuals", action="store_true",
                        help="Bulk-generate per-pt aliases.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required for --fill-individuals.")
    args = parser.parse_args()

    if args.sentinel_smoke:
        return cmd_sentinel_smoke()
    if args.apply_templates:
        return cmd_apply_templates()
    if args.fill_individuals:
        return cmd_fill_individuals(confirmed=args.confirm)

    parser.error("Specify one of --sentinel-smoke, --apply-templates, --fill-individuals.")


if __name__ == "__main__":
    raise SystemExit(main())
