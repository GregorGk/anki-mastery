"""Stage 17 / Step 2 — Render the ordering review report.

Reads (read-only):  data/06-final.sqlite  (anki_ordering, senses, topic_order)
Writes:             reports/17_ordering.html

A static HTML review of the Stage 17 deterministic ordering: topic
sequence, per-custom-topic bucket coverage, polysemy spacing, the
strict↔spaced diff, Level B cross-topic collision warnings, manual
overrides, and a head/tail preview of the final deck order.

Pipeline order is strict: run build/16_0_build_sqlite.py then
build/17_0_build_ordering.py first.

Usage:
    .venv/bin/python build/17_2_ordering_html.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from build.lib import order_rules  # noqa: E402

DATA = REPO_ROOT / "data"
DB_PATH = DATA / "06-final.sqlite"
OUT_HTML = REPO_ROOT / "reports" / "17_ordering.html"

LEVEL_B_THRESHOLD = 30  # cross-topic same-pt collision warning window


class ReportError(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"ERROR (17_2_ordering_html): {msg}")


HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 17 — ordering review</title>
<style>
  :root {
    --bg:#0f1115; --panel:#181b22; --panel-2:#20242d; --border:#2c3140;
    --text:#e6e9ef; --muted:#9098a6; --accent:#5aa9ff;
    --warn:#ffa94d; --bad:#ff6b6b; --ok:#4ade80;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         margin:0; padding:24px; }
  .container { max-width:1280px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:16px; color:var(--muted); text-transform:uppercase;
       letter-spacing:.5px; margin:28px 0 10px; }
  .sub { color:var(--muted); margin:0 0 20px; }
  .panel { background:var(--panel); border:1px solid var(--border);
           border-radius:8px; padding:14px 16px; margin-bottom:14px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th,td { padding:5px 9px; text-align:left; border-bottom:1px solid var(--border);
          vertical-align:top; }
  th { color:var(--muted); font-weight:600; position:sticky; top:0;
       background:var(--panel); }
  td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
  .badge { padding:1px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-custom { background:rgba(90,169,255,.18); color:var(--accent); }
  .b999 { color:var(--bad); font-weight:600; }
  .b999-zero { color:var(--ok); }
  .warn { color:var(--warn); }
  details { margin-bottom:14px; }
  summary { cursor:pointer; font-weight:600; color:var(--accent); padding:6px 0; }
  .stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  .stat { background:var(--panel-2); border-radius:6px; padding:10px 12px; }
  .stat .v { font-size:22px; font-weight:700; }
  .stat .k { color:var(--muted); font-size:12px; }
  .empty { color:var(--muted); font-style:italic; }
</style>
</head>
<body>
<div class="container">
"""


def _esc(v: object) -> str:
    return escape(str(v if v is not None else ""))


def _build(conn: sqlite3.Connection) -> str:
    p: list[str] = [HEAD]
    cur = conn.cursor()

    # ── Section 1 — summary ────────────────────────────────────────
    n = cur.execute("SELECT COUNT(*) FROM anki_ordering").fetchone()[0]
    n_topics = cur.execute("SELECT COUNT(*) FROM topic_order").fetchone()[0]
    n_poly_lemmas = cur.execute(
        "SELECT COUNT(DISTINCT pt) FROM senses WHERE pt IN "
        "(SELECT pt FROM senses GROUP BY pt HAVING COUNT(*) > 1)").fetchone()[0]
    max_poly = cur.execute(
        "SELECT MAX(polysemy_count_global) FROM anki_ordering").fetchone()[0]
    n_diff = cur.execute(
        "SELECT COUNT(*) FROM anki_ordering "
        "WHERE strict_topic_order != spaced_topic_order").fetchone()[0]

    p.append('<h1>Stage 17 — ordering review</h1>')
    p.append(f'<p class="sub">Deterministic topic → bucket → polysemy-spaced '
             f'ordering of <b>{n:,}</b> senses across <b>{n_topics}</b> topics.</p>')
    p.append('<div class="stat-grid">')
    for k, v in (("senses ordered", f"{n:,}"), ("topics", n_topics),
                 ("custom-ordered topics", len(order_rules.CUSTOM_TOPICS)),
                 ("polysemous lemmas", f"{n_poly_lemmas:,}"),
                 ("max senses / lemma", max_poly),
                 ("moved by spacing", f"{n_diff:,}")):
        p.append(f'<div class="stat"><div class="v">{_esc(v)}</div>'
                 f'<div class="k">{_esc(k)}</div></div>')
    p.append('</div>')

    # ── Section 2 — 50-topic order table ───────────────────────────
    p.append('<h2>Topic order (Stage 14 taxonomy)</h2>')
    p.append('<div class="panel"><table>')
    p.append('<tr><th class="num">#</th><th>topic</th><th class="num">senses</th>'
             '<th class="num">spaced order span</th><th>custom?</th></tr>')
    rows = cur.execute(
        "SELECT topic_primary, topic_index, COUNT(*) AS n, "
        "MIN(spaced_topic_order) AS lo, MAX(spaced_topic_order) AS hi "
        "FROM anki_ordering GROUP BY topic_primary, topic_index "
        "ORDER BY topic_index").fetchall()
    for tp, ti, cnt, lo, hi in rows:
        is_custom = tp in order_rules.CUSTOM_TOPICS
        badge = ('<span class="badge badge-custom">custom</span>'
                 if is_custom else '')
        p.append(f'<tr><td class="num">{ti}</td>'
                 f'<td class="mono">{_esc(tp)}</td>'
                 f'<td class="num">{cnt}</td>'
                 f'<td class="num">{lo}–{hi}</td><td>{badge}</td></tr>')
    p.append('</table></div>')

    # ── Section 3 — per-custom-topic bucket coverage ───────────────
    p.append('<h2>Custom-topic bucket coverage</h2>')
    custom_in_order = [t for t in order_rules.ALLOWED_TOPIC_TAGS
                       if t in order_rules.CUSTOM_TOPICS]
    for topic in custom_in_order:
        labels = dict(order_rules.all_bucket_labels(topic))
        brows = cur.execute(
            "SELECT o.topic_bucket, COUNT(*) AS n, "
            "  GROUP_CONCAT(s.pt, ', ') AS lemmas "
            "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
            "WHERE o.topic_primary = ? GROUP BY o.topic_bucket "
            "ORDER BY o.topic_bucket", (topic,)).fetchall()
        total = sum(r[1] for r in brows)
        b999 = next((r[1] for r in brows if r[0] == 999), 0)
        curated = total - b999
        pct = 100 * curated / total if total else 0.0
        p.append(f'<div class="panel"><b class="mono">{_esc(topic)}</b> — '
                 f'{curated}/{total} curated ({pct:.0f}%), '
                 f'<span class="{"b999-zero" if b999 == 0 else "b999"}">'
                 f'{b999} in bucket-999</span>')
        p.append('<table style="margin-top:8px">')
        p.append('<tr><th class="num">bucket</th><th>label</th>'
                 '<th class="num">n</th><th>sample lemmas</th></tr>')
        for bid, cnt, lemmas in brows:
            sample = ", ".join((lemmas or "").split(", ")[:8])
            label = labels.get(bid, "?")
            cls = ' class="b999"' if bid == 999 else ''
            p.append(f'<tr><td class="num"{cls}>{bid}</td><td{cls}>{_esc(label)}</td>'
                     f'<td class="num">{cnt}</td><td>{_esc(sample)}</td></tr>')
        p.append('</table></div>')

    # ── Section 4 — polysemy spacing view ──────────────────────────
    p.append('<h2>Polysemy spacing — same-topic same-lemma groups</h2>')
    p.append('<p class="sub">Did Level A actually separate the multiple senses? '
             '<code>gap</code> = rows between this sense and the previous sense '
             'of the same lemma in <code>spaced_topic_order</code>.</p>')
    poly = cur.execute(
        "SELECT o.topic_primary, s.pt, o.sense_id, s.en_primary, "
        "  o.polysemy_round_in_topic, o.spaced_topic_order "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "WHERE o.polysemy_count_in_topic > 1 "
        "ORDER BY o.topic_primary, s.pt, o.polysemy_round_in_topic").fetchall()
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for tp, pt, sid, en, rnd, spaced in poly:
        groups[(tp, pt)].append((sid, en, rnd, spaced))
    p.append('<details open><summary>{} polysemous groups within topics'
             '</summary>'.format(len(groups)))
    p.append('<div class="panel"><table>')
    p.append('<tr><th>topic</th><th>lemma</th><th>sense_id</th><th>en</th>'
             '<th class="num">round</th><th class="num">spaced</th>'
             '<th class="num">gap</th></tr>')
    for (tp, pt), members in sorted(groups.items(),
                                    key=lambda kv: (-len(kv[1]), kv[0])):
        prev_spaced = None
        for sid, en, rnd, spaced in members:
            gap = "" if prev_spaced is None else str(spaced - prev_spaced)
            prev_spaced = spaced
            p.append(f'<tr><td class="mono">{_esc(tp)}</td><td><b>{_esc(pt)}</b></td>'
                     f'<td class="mono">{_esc(sid)}</td><td>{_esc(en)}</td>'
                     f'<td class="num">{rnd}</td><td class="num">{spaced}</td>'
                     f'<td class="num">{gap}</td></tr>')
    p.append('</table></div></details>')

    # ── Section 5 — strict↔spaced diff ─────────────────────────────
    p.append('<h2>Strict → spaced displacement (top 100)</h2>')
    diff = cur.execute(
        "SELECT o.sense_id, s.pt, o.topic_primary, o.strict_topic_order, "
        "  o.spaced_topic_order, "
        "  (o.spaced_topic_order - o.strict_topic_order) AS delta "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "WHERE o.strict_topic_order != o.spaced_topic_order "
        "ORDER BY ABS(o.spaced_topic_order - o.strict_topic_order) DESC "
        "LIMIT 100").fetchall()
    p.append('<details><summary>{} of {} moved rows (largest displacement)'
             '</summary>'.format(min(100, n_diff), n_diff))
    p.append('<div class="panel"><table>')
    p.append('<tr><th>sense_id</th><th>lemma</th><th>topic</th>'
             '<th class="num">strict</th><th class="num">spaced</th>'
             '<th class="num">Δ</th></tr>')
    for sid, pt, tp, st, sp, delta in diff:
        p.append(f'<tr><td class="mono">{_esc(sid)}</td><td>{_esc(pt)}</td>'
                 f'<td class="mono">{_esc(tp)}</td><td class="num">{st}</td>'
                 f'<td class="num">{sp}</td><td class="num">{delta:+d}</td></tr>')
    p.append('</table></div></details>')

    # ── Section 6 — Level B warnings ───────────────────────────────
    p.append('<h2>Level B — cross-topic same-lemma collisions '
             f'(&lt; {LEVEL_B_THRESHOLD} rows apart)</h2>')
    p.append('<p class="sub">Report-only (per locked decision). The same lemma '
             'appearing in two topics that land near each other in the deck.</p>')
    allrows = cur.execute(
        "SELECT s.pt, o.sense_id, o.topic_primary, o.spaced_topic_order "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "WHERE s.pt IN (SELECT pt FROM senses GROUP BY pt HAVING COUNT(*) > 1) "
        "ORDER BY s.pt, o.spaced_topic_order").fetchall()
    by_pt: dict[str, list] = defaultdict(list)
    for pt, sid, tp, sp in allrows:
        by_pt[pt].append((sid, tp, sp))
    collisions = []
    for pt, members in by_pt.items():
        for i in range(1, len(members)):
            (s0, t0, sp0), (s1, t1, sp1) = members[i - 1], members[i]
            if t0 != t1 and (sp1 - sp0) < LEVEL_B_THRESHOLD:
                collisions.append((pt, s0, t0, sp0, s1, t1, sp1, sp1 - sp0))
    collisions.sort(key=lambda r: r[7])
    if not collisions:
        p.append('<div class="panel empty">No cross-topic collisions within '
                 f'{LEVEL_B_THRESHOLD} rows.</div>')
    else:
        p.append(f'<div class="panel"><b class="warn">{len(collisions)} '
                 f'collision(s)</b><table style="margin-top:8px">')
        p.append('<tr><th>lemma</th><th>sense A</th><th>topic A</th>'
                 '<th class="num">spaced A</th><th>sense B</th><th>topic B</th>'
                 '<th class="num">spaced B</th><th class="num">gap</th></tr>')
        for pt, s0, t0, sp0, s1, t1, sp1, gap in collisions:
            p.append(f'<tr><td><b>{_esc(pt)}</b></td>'
                     f'<td class="mono">{_esc(s0)}</td><td class="mono">{_esc(t0)}</td>'
                     f'<td class="num">{sp0}</td>'
                     f'<td class="mono">{_esc(s1)}</td><td class="mono">{_esc(t1)}</td>'
                     f'<td class="num">{sp1}</td>'
                     f'<td class="num warn">{gap}</td></tr>')
        p.append('</table></div>')

    # ── Section 7 — manual overrides ───────────────────────────────
    p.append('<h2>Manual overrides applied</h2>')
    manual = cur.execute(
        "SELECT o.sense_id, s.pt, o.topic_primary, o.topic_bucket, "
        "  o.topic_subrank, o.spacing_action, o.order_reason "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "WHERE o.order_reason LIKE 'manual%' OR o.spacing_action LIKE 'forced_after%' "
        "ORDER BY o.spaced_topic_order").fetchall()
    if not manual:
        p.append('<div class="panel empty">No manual overrides — '
                 'data/_manual_ordering.tsv is header-only.</div>')
    else:
        p.append('<div class="panel"><table>')
        p.append('<tr><th>sense_id</th><th>lemma</th><th>topic</th>'
                 '<th class="num">bucket</th><th class="num">subrank</th>'
                 '<th>spacing_action</th><th>order_reason</th></tr>')
        for sid, pt, tp, tb, ts, sa, orr in manual:
            p.append(f'<tr><td class="mono">{_esc(sid)}</td><td>{_esc(pt)}</td>'
                     f'<td class="mono">{_esc(tp)}</td><td class="num">{tb}</td>'
                     f'<td class="num">{ts}</td><td>{_esc(sa)}</td>'
                     f'<td>{_esc(orr)}</td></tr>')
        p.append('</table></div>')

    # ── Section 8 — ordered deck preview ───────────────────────────
    p.append('<h2>Ordered deck preview</h2>')

    def _preview_table(title: str, sql: str) -> None:
        p.append(f'<details><summary>{escape(title)}</summary>')
        p.append('<div class="panel"><table>')
        p.append('<tr><th class="num">#</th><th>sense_id</th><th>display</th>'
                 '<th>topic</th><th class="num">bucket</th></tr>')
        for sp, sid, disp, tp, tb in cur.execute(sql):
            p.append(f'<tr><td class="num">{sp}</td><td class="mono">{_esc(sid)}</td>'
                     f'<td>{_esc(disp)}</td><td class="mono">{_esc(tp)}</td>'
                     f'<td class="num">{tb}</td></tr>')
        p.append('</table></div></details>')

    _preview_table(
        "First 200 cards",
        "SELECT o.spaced_topic_order, o.sense_id, s.pt_display_safe, "
        "  o.topic_primary, o.topic_bucket "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "ORDER BY o.spaced_topic_order LIMIT 200")
    _preview_table(
        "Last 50 cards",
        "SELECT o.spaced_topic_order, o.sense_id, s.pt_display_safe, "
        "  o.topic_primary, o.topic_bucket "
        "FROM anki_ordering o JOIN senses s ON s.sense_id = o.sense_id "
        "ORDER BY o.spaced_topic_order DESC LIMIT 50")

    p.append('</div></body></html>')
    return "".join(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    if not args.db.exists():
        raise ReportError(f"{args.db} does not exist.")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "anki_ordering" not in have:
            raise ReportError("anki_ordering missing. Run "
                              "build/17_0_build_ordering.py first.")
        html = _build(conn)
    finally:
        conn.close()

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML.relative_to(REPO_ROOT)}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
