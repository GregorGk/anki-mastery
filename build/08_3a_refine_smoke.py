"""Stage 8 / Step 3a — v3 alias respelling refinement smoke.

User listened to v2 alias re-renders (08_6 HTML) and flagged 8
phonetic issues where the family-template respellings don't capture
BP phonotactics (palatalization of `di`/`ti`, /ʒ/ vs /d͡ʒ/, initial
/ʁ/, etc.). This renders 8 alternate v3 respellings via the Lair
voice (BP-only, no R2 upload, no dict needed — direct text-to-audio
of the alternate string) and builds an HTML for browser listening.

User listens, reports per-word GOOD or NOT_QUITE, then we apply v3 to
the production aliases TSV and re-run 08_4 → 08_5.
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.elevenlabs_client import ElevenLabsClient  # noqa: E402
from build.lib.loudness import normalize_pcm_to_mp3_verified  # noqa: E402

AUDIT_DIR = REPO_ROOT / "audit"
LAIR_VOICE = "4r3G9XKliGgVZLKMgjik"  # matches 08_3.SENTINEL_VOICE_ID

# (pt, last_attempt, new_proposal, target_ipa, why)
# Round 11 (v3.10) — habitual still missing final stress (the 'h' drop worked).
# Try hyphenated patterns matching Zhe-ne-ráu and In-dús-tri-áu winners.
REFINEMENTS: list[tuple[str, str, str, str, str]] = [
    ("habitual", "abituáu", "Abi-tu-áu",
     "[a.bi.tu.ˈaw]",
     "alt-A: hyphenated 3-syll (Zhe-ne-ráu pattern) + final acute"),
    ("habitual", "abituáu", "A-bi-tu-áu",
     "[a.bi.tu.ˈaw]",
     "alt-B: 4-syll explicit hyphens (every syllable separated)"),
    ("habitual", "abituáu", "abitúáu",
     "[a.bi.tu.ˈaw]",
     "alt-C: no hyphens, double acute on ú + á (mark stressed nucleus + glide)"),
]


def _render_one(client: ElevenLabsClient, pt: str, v3: str) -> bytes:
    """Render the v3 respelling via Lair voice. Returns MP3 bytes."""
    out = AUDIT_DIR / f"08_3a_smoke_{pt}_{v3}.mp3"
    if out.exists():
        return out.read_bytes()
    res = client.generate_pcm(
        text=v3,
        voice_id=LAIR_VOICE,
        sense_id=f"sentinel-{pt}-v3",
        clip_type="word",
        version=1,
    )
    norm = normalize_pcm_to_mp3_verified(res.audio_pcm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(norm.mp3_bytes)
    return norm.mp3_bytes


def _build_html(rendered: list[tuple[str, str, str, str, str, bytes]]) -> Path:
    """Build the smoke comparison HTML.

    Each row encodes audio inline (base64 data URL) so the file is fully
    portable (matches 08_3 sentinel HTML pattern).
    """
    rows: list[str] = []
    for pt, v2, v3, ipa, why, mp3 in rendered:
        b64 = base64.b64encode(mp3).decode("ascii")
        rows.append(f"""
<tr>
  <td><strong>{html.escape(pt)}</strong></td>
  <td><code>{html.escape(v2)}</code></td>
  <td><code class="v3">{html.escape(v3)}</code></td>
  <td>{html.escape(ipa)}</td>
  <td>{html.escape(why)}</td>
  <td><audio controls preload="none" src="data:audio/mp3;base64,{b64}"></audio></td>
</tr>
""")

    out = AUDIT_DIR / "08_3a_refine_smoke.html"
    out.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 8 — v3 alias refinement smoke</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px; margin: 2em auto; padding: 0 1em; line-height: 1.5; color: #222; }}
h1 {{ border-bottom: 2px solid #888; padding-bottom: 0.3em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
th, td {{ border: 1px solid #aaa; padding: 0.5em 0.7em; text-align: left; vertical-align: top; }}
th {{ background: #e6ecf2; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.95em; }}
code.v3 {{ background: #d4edda; color: #155724; font-weight: bold; }}
audio {{ width: 280px; }}
.intro {{ background: #f5f8fb; border: 1px solid #cdd; border-radius: 6px;
  padding: 1em 1.5em; margin: 1em 0 2em; font-size: 0.95em; }}
</style>
</head>
<body>

<h1>Stage 8 — v3 alias refinement smoke</h1>

<div class="intro">
  <p><strong>What this is:</strong> 8 alternate v3 respellings for words you flagged
  in the v2 listening pass. Rendered via Lair voice (BP-only) without dict — straight
  text-to-audio of the green column on the right.</p>
  <p><strong>How to listen:</strong> Click play on each row. Compare against your memory
  of the v2 AFTER you listened to in 08_6. The target IPA column shows what BP should
  produce.</p>
  <p><strong>Reply per word:</strong> GOOD (apply this v3 in production) or NOT_QUITE
  (suggest a different respelling). Rows that need iteration get a second smoke pass
  before we touch the production dict.</p>
</div>

<table>
<tr>
  <th>pt</th>
  <th>last attempt</th>
  <th>new proposal</th>
  <th>target IPA</th>
  <th>why</th>
  <th>audio</th>
</tr>
{''.join(rows)}
</table>

</body>
</html>
""", encoding="utf-8")
    return out


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    import os
    if not os.environ.get("ELEVENLABS_API_KEY"):
        print("ERROR: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        return 1

    client = ElevenLabsClient()
    rendered: list[tuple[str, str, str, str, str, bytes]] = []

    print(f"Rendering {len(REFINEMENTS)} v3 respellings via Lair voice...")
    for pt, v2, v3, ipa, why in REFINEMENTS:
        try:
            mp3 = _render_one(client, pt, v3)
            rendered.append((pt, v2, v3, ipa, why, mp3))
            print(f"  {pt:<14} {v2:<16} → {v3:<18} ({len(mp3)} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {pt} → {v3}: {exc}", file=sys.stderr)

    if not rendered:
        print("Nothing rendered — exiting.", file=sys.stderr)
        return 1

    html_path = _build_html(rendered)
    print()
    print(f"Wrote {html_path}")
    print(f"  open with:  open {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
