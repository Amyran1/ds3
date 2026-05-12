"""HTML rendering helper for user-EDA findings.

Each finding produces one self-contained HTML page (no external fetches; PNG
charts embedded as base64). Pages follow the template defined in
.cursor/plans (or wherever the plan currently lives) under "HTML page template".

Usage:

    from projects.civic_shout_action_rate_increase.eda.user._html import (
        render_finding,
    )

    html = render_finding(
        finding_id="F00",
        title="Outcome cardinality check",
        tldr="Goal says petitions/email but target is boolean.",
        wave=1,
        severity="BLOCKING",
        classification="confounder",
        question_md="...",
        method_py_source="...",
        chart_png_bytes=b"...",
        numbers_table=[("pairs with >=2 actions", "0.4%")],
        impact_md="...",
        provenance={"data": "v2 / v1", "sample": "1%", "ts": "...", "git": "..."},
    )

The same helper renders both mockup pages (synthetic data, MOCKUP banner) and
the eventual real EDA pages (real data, no banner).
"""

from __future__ import annotations

import base64
import html as _html
import textwrap

_BASE_CSS = """
:root {
  --bg: #fafaf8;
  --fg: #1a1a1a;
  --muted: #666;
  --accent: #2956a3;
  --warn: #c5500f;
  --block: #b00020;
  --rule: #e6e6e2;
  --code-bg: #f3f3ef;
  --table-stripe: #f7f7f3;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 920px;
  margin: 2em auto;
  padding: 0 1.25em;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.5;
}
header {
  border-bottom: 2px solid var(--rule);
  padding-bottom: 0.75em;
  margin-bottom: 1.25em;
}
header h1 {
  margin: 0 0 0.25em 0;
  font-size: 1.55em;
}
.tldr {
  font-size: 1em;
  color: var(--muted);
  margin: 0.25em 0 0.6em 0;
  font-style: italic;
}
.badges {
  display: flex;
  gap: 0.5em;
  flex-wrap: wrap;
  margin-top: 0.5em;
}
.badge {
  display: inline-block;
  font-size: 0.72em;
  padding: 0.15em 0.6em;
  border-radius: 4px;
  background: #e8e8e2;
  color: #333;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.badge.wave1 { background: #4a3aff; color: white; }
.badge.wave2 { background: #2956a3; color: white; }
.badge.wave3 { background: #207e3c; color: white; }
.badge.severity-BLOCKING { background: var(--block); color: white; }
.badge.severity-IMPORTANT { background: var(--warn); color: white; }
.badge.severity-SUGGESTION { background: #555; color: white; }
.badge.mockup { background: #e0c100; color: black; }
section { margin: 1.5em 0; }
section h2 {
  font-size: 1.05em;
  margin: 0 0 0.5em 0;
  color: var(--accent);
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.2em;
}
.method {
  background: var(--code-bg);
  padding: 0.75em 1em;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.82em;
  overflow-x: auto;
  white-space: pre;
}
.chart-wrap {
  text-align: center;
  margin: 1em 0;
}
.chart-wrap img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--rule);
  background: white;
  padding: 0.25em;
  border-radius: 3px;
}
table.numbers {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92em;
  margin-top: 0.75em;
}
table.numbers td {
  padding: 0.35em 0.6em;
  border-bottom: 1px solid var(--rule);
}
table.numbers tr:nth-child(even) { background: var(--table-stripe); }
table.numbers td.metric { color: var(--muted); }
table.numbers td.value {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  text-align: right;
  font-weight: 600;
}
.impact {
  background: #f3f5f9;
  padding: 0.75em 1em;
  border-left: 4px solid var(--accent);
  border-radius: 3px;
}
.impact p, .impact ul { margin: 0.4em 0; }
.impact li { margin: 0.15em 0; }
.plain-english {
  background: #f5faf2;
  padding: 0.75em 1em;
  border-left: 4px solid #207e3c;
  border-radius: 3px;
  font-size: 0.97em;
}
.plain-english p, .plain-english ul { margin: 0.4em 0; }
.plain-english li { margin: 0.15em 0; }
.plain-english .lead {
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #207e3c;
  font-weight: 700;
  margin-bottom: 0.3em;
}
footer.provenance {
  font-size: 0.75em;
  color: var(--muted);
  border-top: 1px solid var(--rule);
  padding-top: 0.6em;
  margin-top: 2em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.mockup-banner {
  background: #fff7d6;
  border: 1px solid #e0c100;
  padding: 0.5em 0.75em;
  border-radius: 3px;
  font-size: 0.85em;
  margin-bottom: 1em;
}
.mockup-banner strong { color: #7a6700; }
a, a:visited { color: var(--accent); }
"""


def _esc(s: str) -> str:
    return _html.escape(s, quote=True)


def _render_badges(
    wave: int,
    severity: str,
    classification: str,
    is_mockup: bool,
) -> str:
    badges = [
        f'<span class="badge wave{wave}">Wave {wave}</span>',
        f'<span class="badge severity-{_esc(severity)}">{_esc(severity)}</span>',
        f'<span class="badge">{_esc(classification)}</span>',
    ]
    if is_mockup:
        badges.append('<span class="badge mockup">MOCKUP</span>')
    return '<div class="badges">' + "".join(badges) + "</div>"


def _render_numbers_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    body = "".join(
        f'<tr><td class="metric">{_esc(m)}</td><td class="value">{_esc(v)}</td></tr>'
        for m, v in rows
    )
    return f'<table class="numbers"><tbody>{body}</tbody></table>'


def _render_impact(impact_md: str) -> str:
    """Lightweight markdown — supports paragraphs, single-line bullet lists.

    Avoids a full markdown parser dependency. The plan caps Impact at ≤200 words
    and discourages nesting, so this is sufficient.
    """
    lines = [line.rstrip() for line in impact_md.strip().splitlines()]
    out: list[str] = []
    bullets: list[str] = []

    def flush_bullets() -> None:
        if bullets:
            out.append("<ul>" + "".join(f"<li>{_esc(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{_esc(' '.join(paragraph))}</p>")
            paragraph.clear()

    for line in lines:
        if not line:
            flush_bullets()
            flush_paragraph()
        elif line.lstrip().startswith(("- ", "* ")):
            flush_paragraph()
            bullets.append(line.lstrip()[2:])
        else:
            flush_bullets()
            paragraph.append(line)
    flush_bullets()
    flush_paragraph()
    return f'<div class="impact">{"".join(out)}</div>'


def render_finding(
    *,
    finding_id: str,
    title: str,
    tldr: str,
    wave: int,
    severity: str,
    classification: str,
    question_md: str,
    plain_english_md: str,
    method_py_source: str,
    chart_png_bytes: bytes,
    numbers_table: list[tuple[str, str]],
    impact_md: str,
    provenance: dict[str, str],
    is_mockup: bool = False,
) -> str:
    """Render one finding page to self-contained HTML."""
    png_b64 = base64.b64encode(chart_png_bytes).decode("ascii")
    badges = _render_badges(wave, severity, classification, is_mockup)
    numbers_html = _render_numbers_table(numbers_table)
    impact_html = _render_impact(impact_md)
    # Re-use the markdown-ish renderer for plain-english, then swap the wrapper class
    plain_english_inner = _render_impact(plain_english_md).replace(
        'class="impact"', 'class="plain-english"', 1
    )
    plain_english_html = plain_english_inner.replace(
        'class="plain-english">',
        'class="plain-english"><div class="lead">In plain English</div>',
        1,
    )
    method_dedented = textwrap.dedent(method_py_source).strip()

    mockup_banner = ""
    if is_mockup:
        mockup_banner = (
            '<div class="mockup-banner"><strong>Mockup preview.</strong> '
            "Chart uses synthetic shape-accurate data; the polars sketch and "
            '"Impact" section are the real plan content. The actual EDA run '
            "will replace the chart and numbers, keeping the layout.</div>"
        )

    provenance_parts = [f"<span>{_esc(k)}: {_esc(v)}</span>" for k, v in provenance.items()]
    provenance_html = " &middot; ".join(provenance_parts)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(finding_id)} — {_esc(title)}</title>
<style>{_BASE_CSS}</style>
</head>
<body>
<header>
  <h1>{_esc(finding_id)}. {_esc(title)}</h1>
  <p class="tldr">{_esc(tldr)}</p>
  {badges}
</header>
{mockup_banner}
<section>
  <h2>Question</h2>
  <p>{_esc(question_md)}</p>
</section>
<section>
  {plain_english_html}
</section>
<section>
  <h2>Method</h2>
  <pre class="method">{_esc(method_dedented)}</pre>
</section>
<section>
  <h2>Result</h2>
  <div class="chart-wrap">
    <img src="data:image/png;base64,{png_b64}" alt="{_esc(finding_id)} chart">
  </div>
  {numbers_html}
</section>
<section>
  <h2>Impact on harness / features</h2>
  {impact_html}
</section>
<footer class="provenance">
  {provenance_html}
</footer>
</body>
</html>
"""


def render_index(findings: list[dict]) -> str:
    """Render index.html linking all findings, sortable by wave and severity."""
    rows = []
    for f in findings:
        wave_badge = f'<span class="badge wave{f["wave"]}">Wave {f["wave"]}</span>'
        sev_badge = f'<span class="badge severity-{f["severity"]}">{f["severity"]}</span>'
        class_badge = f'<span class="badge">{_esc(f["classification"])}</span>'
        mockup_badge = '<span class="badge mockup">MOCKUP</span>' if f.get("is_mockup") else ""
        rows.append(
            f"""<tr>
  <td class="id"><a href="{_esc(f["id"])}_{_esc(f["slug"])}.html">{_esc(f["id"])}</a></td>
  <td class="title">{_esc(f["title"])}<div class="tldr-row">{_esc(f["tldr"])}</div></td>
  <td class="badges-cell">{wave_badge} {sev_badge} {class_badge} {mockup_badge}</td>
</tr>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>User-EDA findings — civic_shout_action_rate_increase</title>
<style>
{_BASE_CSS}
table.index {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 1em;
}}
table.index td {{
  padding: 0.55em 0.6em;
  border-bottom: 1px solid var(--rule);
  vertical-align: top;
}}
table.index tr:nth-child(even) {{ background: var(--table-stripe); }}
table.index td.id {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 700;
  width: 4em;
}}
table.index td.title {{ font-weight: 600; }}
.tldr-row {{
  font-weight: 400;
  font-size: 0.88em;
  color: var(--muted);
  font-style: italic;
  margin-top: 0.15em;
}}
table.index td.badges-cell {{ text-align: right; width: 18em; }}
</style>
</head>
<body>
<header>
  <h1>User-EDA findings index</h1>
  <p class="tldr">
    11 findings split across 3 waves. Wave 1 gates the harness target +
    split convention + feature whitelist; Wave 2 fixes the evaluation regime
    and audits content confounds; Wave 3 tests user-feature hypotheses.
  </p>
</header>
<table class="index"><tbody>
{"".join(rows)}
</tbody></table>
<footer class="provenance">
  Plan: /Users/aaronmyran/.claude/plans/great-that-s-perfect-next-steady-mochi.md &middot;
  Data: civic_shout_user_emails v2, civic_shout_emails v1 &middot;
  Project: projects/civic_shout_action_rate_increase
</footer>
</body>
</html>
"""
