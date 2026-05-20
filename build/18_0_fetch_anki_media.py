"""Stage 18 / Step 0 — Fetch Anki media (word + example MP3s) to a local dir.

Reads (read-only):  data/07-anki-media-manifest.tsv  (url + md5 per clip)
                    data/07-anki-listening-general.tsv (only for --pilot)
Writes:             data/anki_media/<media_filename>.mp3   (gitignored)
                    reports/18_media_fetch.html             (fetch summary)

Selects manifest rows where `clip_type in {word, example, en_ex}` AND
`status == "uploaded"` — NO model filter (the legacy/holdout filename case
is already handled by `media_filename`, which is the stored object_key
minus the `audio/` prefix). All three clip types are bundled: `example`
+ `en_ex` render as `[sound:…]`; `word` is bundled for the click-only
`<audio src>` element.

Downloads in parallel with a real User-Agent (R2 403s the default urllib
UA), verifies each file's md5 against the manifest, skips files already on
disk whose md5 matches, and re-downloads on mismatch.

Run AFTER build/17_1_export_anki.py.

Usage:
    .venv/bin/python build/18_0_fetch_anki_media.py            # full word+example
    .venv/bin/python build/18_0_fetch_anki_media.py --pilot    # pilot senses only
    .venv/bin/python build/18_0_fetch_anki_media.py --limit 300
"""
from __future__ import annotations

import argparse
import hashlib
import html
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib.anki_pilot import read_anki_export, select_pilot_sense_ids  # noqa: E402
from build.lib.tsv import read_tsv  # noqa: E402

DATA = REPO_ROOT / "data"
MANIFEST = DATA / "07-anki-media-manifest.tsv"
ANKI_TSV = DATA / "07-anki-listening-general.tsv"
MEDIA_DIR = DATA / "anki_media"
REPORT = REPO_ROOT / "reports" / "18_media_fetch.html"

DEFAULT_CLIP_TYPES = ("word", "example", "en_ex")
DEFAULT_WORKERS = 32  # well under the macOS DNS-storm threshold (see 16_8)

_HTTP: httpx.Client | None = None


class FetchError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (18_0_fetch_anki_media): {msg}")


def _http() -> httpx.Client:
    global _HTTP
    if _HTTP is None:
        _HTTP = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=httpx.Timeout(60.0, connect=15.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64,
                                keepalive_expiry=300.0),
            transport=httpx.HTTPTransport(retries=5),
            http2=False,
        )
    return _HTTP


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _fetch_verify(url: str, dest: Path, expected_md5: str,
                  max_attempts: int = 5) -> tuple[bool, str]:
    """Download url → dest, verifying md5. Atomic via tmp + rename."""
    client = _http()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            if expected_md5:
                got = _md5_bytes(content)
                if got != expected_md5:
                    last_exc = ValueError(
                        f"md5 mismatch (got {got[:8]}…, want {expected_md5[:8]}…)")
                    if attempt < max_attempts:
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                        continue
                    return False, str(last_exc)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(content)
            tmp.rename(dest)
            return True, ""
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError, httpx.PoolTimeout,
                httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(0.5 * (2 ** (attempt - 1)))
                continue
            break
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    return False, f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"


def _select_rows(clip_types: tuple[str, ...], pilot: bool, limit: int) -> list[dict]:
    rows = read_tsv(MANIFEST)
    if not rows:
        raise FetchError(f"{MANIFEST} is empty — run build/17_1_export_anki.py first.")
    sel = [r for r in rows
           if r.get("clip_type") in clip_types and r.get("status") == "uploaded"]
    if pilot:
        if not ANKI_TSV.exists():
            raise FetchError(f"{ANKI_TSV} missing — run build/17_1_export_anki.py first.")
        pilot_ids = set(select_pilot_sense_ids(read_anki_export(ANKI_TSV)))
        sel = [r for r in sel if r.get("sense_id") in pilot_ids]
    if limit:
        sel = [r for r in sel if _order(r) <= limit]
    return sel


def _order(r: dict) -> int:
    try:
        return int(r.get("spaced_topic_order") or 0)
    except (TypeError, ValueError):
        return 0


def _write_report(stats: dict, failures: list[str], clip_types, pilot, limit) -> None:
    css = """
    :root{--bg:#0f1115;--panel:#181b22;--panel-2:#20242d;--border:#2c3140;
      --text:#e6e9ef;--muted:#9098a6;--accent:#5aa9ff;--ok:#4ade80;--bad:#ff6b6b;}
    *{box-sizing:border-box;}
    body{background:var(--bg);color:var(--text);margin:0;padding:24px;
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
    .wrap{max-width:900px;margin:0 auto;}
    h1{font-size:21px;margin:0 0 4px;}
    .sub{color:var(--muted);margin:0 0 18px;}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 20px;}
    td,th{padding:7px 12px;text-align:left;border-bottom:1px solid var(--border);}
    th{color:var(--muted);font-weight:500;}
    td.num{text-align:right;font-family:ui-monospace,Menlo,monospace;}
    .ok{color:var(--ok);} .bad{color:var(--bad);}
    code{background:var(--panel-2);padding:1px 5px;border-radius:3px;font-size:12px;}
    .fail{background:var(--panel);border:1px solid var(--border);border-radius:8px;
      padding:10px 14px;font-family:ui-monospace,Menlo,monospace;font-size:12px;
      white-space:pre-wrap;}
    """
    scope = "pilot" if pilot else (f"limit={limit}" if limit else "full")
    rows_html = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{v:,}</td></tr>"
        for k, v in stats.items()
    )
    fail_html = ""
    if failures:
        fail_html = ("<h2 style='font-size:16px;'>failures</h2><div class='fail'>"
                     + html.escape("\n".join(failures)) + "</div>")
    doc = (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Stage 18 — media fetch</title><style>{css}</style></head><body>"
        f"<div class='wrap'><h1>Stage 18 — Anki media fetch</h1>"
        f"<p class='sub'>scope <code>{html.escape(scope)}</code> · clip types "
        f"<code>{html.escape(', '.join(clip_types))}</code> · dest "
        f"<code>data/anki_media/</code></p>"
        f"<table><tr><th>metric</th><th class='num'>count</th></tr>{rows_html}</table>"
        f"{fail_html}</div></body></html>"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0,
                    help="only senses with spaced_topic_order <= N")
    ap.add_argument("--pilot", action="store_true",
                    help="only the deterministic pilot sense set")
    ap.add_argument("--clip-types", default=",".join(DEFAULT_CLIP_TYPES),
                    help="comma-separated (default: word,example)")
    args = ap.parse_args()

    clip_types = tuple(t.strip() for t in args.clip_types.split(",") if t.strip())
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    rows = _select_rows(clip_types, args.pilot, args.limit)

    # Plan: skip files already present with matching md5; (re)download the rest.
    plan: list[tuple[dict, Path]] = []
    skipped = 0
    redownload = 0
    for r in rows:
        fname = r["media_filename"]
        dest = MEDIA_DIR / fname
        want = (r.get("md5") or "").strip()
        if dest.exists() and dest.stat().st_size > 0:
            if not want or _md5_file(dest) == want:
                skipped += 1
                continue
            redownload += 1  # md5 mismatch → re-fetch
        plan.append((r, dest))

    scope = "pilot" if args.pilot else (f"limit={args.limit}" if args.limit else "full")
    print(f"=== Stage 18.0 — fetch Anki media ({scope}) → {MEDIA_DIR} ===")
    print(f"  clip types:        {', '.join(clip_types)}")
    print(f"  selected clips:    {len(rows):,}")
    print(f"  already cached:    {skipped:,}")
    print(f"  md5 re-downloads:  {redownload:,}")
    print(f"  to download:       {len(plan):,}")
    print(f"  workers:           {args.workers}")

    t0 = time.time()
    n_ok = n_err = 0
    failures: list[str] = []
    if plan:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_fetch_verify, r["url"], dest, (r.get("md5") or "").strip()):
                    (r, dest) for r, dest in plan}
            for i, fut in enumerate(as_completed(futs), start=1):
                row, _dest = futs[fut]
                ok, err = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_err += 1
                    if len(failures) < 25:
                        failures.append(f"{row['sense_id']:<12} {row['clip_type']:<8} "
                                        f"{row['media_filename']}  {err[:90]}")
                if i % 200 == 0 or i == 1 or i == len(plan):
                    el = time.time() - t0
                    rate = n_ok / max(el, 0.01)
                    eta = (len(plan) - i) / max(rate, 0.01)
                    print(f"  [{i:>6}/{len(plan)}] ok={n_ok} err={n_err} "
                          f"rate={rate:.1f}/s eta={eta/60:.1f}m", flush=True)

    stats = {
        "selected_clips": len(rows),
        "already_cached": skipped,
        "md5_redownloads": redownload,
        "downloaded_ok": n_ok,
        "failed": n_err,
    }
    _write_report(stats, failures, clip_types, args.pilot, args.limit)

    print()
    print(f"=== fetch summary ===")
    for k, v in stats.items():
        print(f"  {k:<18} {v:,}")
    print(f"  wall:              {time.time()-t0:.0f}s")
    print(f"  report:            {REPORT.relative_to(REPO_ROOT)}")
    if failures:
        print("  first failures:")
        for s in failures[:10]:
            print(f"    {s}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
