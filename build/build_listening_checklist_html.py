"""Convert audit/listening_checklist.txt to HTML with clickable URLs.

Very simple: monospace <pre> block, URLs wrapped in <a> tags that open
in a new tab. No styling beyond what makes URLs visibly clickable.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TXT = REPO_ROOT / "audit" / "listening_checklist.txt"
HTML_OUT = REPO_ROOT / "audit" / "listening_checklist.html"

URL_RE = re.compile(r"(https://[^\s]+)")


def main():
    if not TXT.exists():
        print(f"missing {TXT}", file=sys.stderr)
        return 1
    body = TXT.read_text(encoding="utf-8")
    body_escaped = html.escape(body, quote=False)

    def linkify(m):
        url = m.group(1)
        return f'<a href="{url}" target="_blank">{url}</a>'

    body_linked = URL_RE.sub(linkify, body_escaped)

    out = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Listening checklist — Stage 6 pilot</title>\n"
        "<style>body{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;line-height:1.4;margin:24px}"
        "a{color:#0645AD}a:visited{color:#663399}pre{white-space:pre-wrap;word-wrap:break-word}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<pre>{body_linked}</pre>\n"
        "</body>\n"
        "</html>\n"
    )
    HTML_OUT.write_text(out, encoding="utf-8")
    print(f"wrote {HTML_OUT}", file=sys.stderr)
    print(f"open: file://{HTML_OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
