"""Leaderboard renderer for DS v2 projects.

Reads the three typed ledgers (results.jsonl, eda_results.jsonl,
discovery_results.jsonl) and emits a self-contained HTML leaderboard.
Meta-refresh=15s for browser auto-update.

CLI: python -m libs.leaderboard <project> [--watch] [--port 8765]
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from libs.responses import DiscoveryResultRow, EDAResultRow

ROOT = Path(__file__).resolve().parents[1]


class _FlexRow(BaseModel):
    """Lenient harness row — only fields the leaderboard actually renders.

    Accepts real ledger rows that predate the typed RunResultRow schema.
    All extra project-specific fields (ci_low, ci_high, elapsed_seconds, etc.)
    land in model_extra.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    run_id: str
    project: str = ""
    comparison_group: str = ""
    scope: str = "comparison"
    status: str = "completed"
    verdict: str | None = None
    recorded_at_utc: str = ""
    git_sha: str = ""
    outcome_version: str = "v1"
    primary_metric_name: str = ""
    primary_metric_value: float | None = None
    baseline_metric_value: float | None = None
    lift_vs_baseline: float | None = None
    metric_direction: str = "higher_is_better"
    n_rows: int = 0
    sample_frac: float = 0.0
    is_full_run: bool = False
    n_features: int = 0
    total_seconds: float = 0.0
    cpu_utilization_pct: float | None = None
    fingerprint: str = ""
    notes: str = ""


def _tier_label(frac: float | None) -> str:
    if frac is None:
        return "unknown"
    if frac <= 0.001:
        return "smoke 0.1%"
    if frac <= 0.01:
        return "1%"
    if frac <= 0.05:
        return "5%"
    return "full"


def fmt_dur(seconds: int | float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "_managed_by" not in obj:
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


class _FlexEDARow(EDAResultRow):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class _FlexDiscRow(DiscoveryResultRow):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


def _load_harness_rows(project: str) -> list[_FlexRow]:
    p = ROOT / "projects" / project / "runs" / "results.jsonl"
    rows: list[_FlexRow] = []
    for raw in _read_jsonl(p):
        try:
            row = _FlexRow.model_validate(raw)
            rows.append(row)
        except ValidationError:
            continue
    return rows


def _load_eda_rows(project: str) -> list[_FlexEDARow]:
    p = ROOT / "projects" / project / "runs" / "eda_results.jsonl"
    rows: list[_FlexEDARow] = []
    for raw in _read_jsonl(p):
        try:
            row = _FlexEDARow.model_validate(raw)
            rows.append(row)
        except ValidationError:
            continue
    return rows


def _load_disc_rows(project: str) -> list[_FlexDiscRow]:
    p = ROOT / "projects" / project / "runs" / "discovery_results.jsonl"
    rows: list[_FlexDiscRow] = []
    for raw in _read_jsonl(p):
        try:
            row = _FlexDiscRow.model_validate(raw)
            rows.append(row)
        except ValidationError:
            continue
    return rows


def _fmt_ci(row: _FlexRow) -> str:
    extra = row.model_extra or {}
    lo = extra.get("ci_low")
    hi = extra.get("ci_high")
    if lo is not None and hi is not None:
        return f"[{lo:.4f}, {hi:.4f}]"
    return "n/a"


def _fmt_lift(row: _FlexRow) -> str:
    if row.lift_vs_baseline is not None:
        return f"{row.lift_vs_baseline:+.4f}"
    return "—"


def _fmt_elapsed(row: _FlexRow) -> str:
    extra = row.model_extra or {}
    secs = extra.get("elapsed_seconds") or row.total_seconds
    return fmt_dur(secs)


def _fmt_metric(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def _build_champion_table(
    harness: list[_FlexRow],
    eda_by_run: dict[str, _FlexEDARow],
    disc_by_run: dict[str, _FlexDiscRow],
    has_eda: bool,
    has_disc: bool,
) -> str:
    if not harness:
        return "<div class='muted'>No harness runs found.</div>"

    header_cols = [
        "run_id",
        "group",
        "scope",
        "primary",
        "CI",
        "lift",
        "elapsed",
        "recorded_at",
    ]
    if has_eda:
        header_cols += ["eda_warn", "eda_err"]
    if has_disc:
        header_cols += ["disc_findings"]

    th = "".join(f"<th>{c}</th>" for c in header_cols)
    rows_html: list[str] = []

    for i, row in enumerate(harness):
        trophy = "🏆 " if i == 0 else ""
        eid = row.run_id
        eda_row = eda_by_run.get(eid)
        disc_row = disc_by_run.get(eid)

        cells = [
            f"<td class='mono'>{trophy}{html.escape(row.run_id)}</td>",
            f"<td class='mono muted'>{html.escape(row.comparison_group)}</td>",
            f"<td><span class='pill scope-{html.escape(row.scope)}'>{html.escape(row.scope)}</span></td>",
            f"<td class='mono num'>{_fmt_metric(row.primary_metric_value)}</td>",
            f"<td class='mono muted'>{html.escape(_fmt_ci(row))}</td>",
            f"<td class='mono'>{html.escape(_fmt_lift(row))}</td>",
            f"<td class='mono'>{html.escape(_fmt_elapsed(row))}</td>",
            f"<td class='mono muted'>{html.escape(row.recorded_at_utc[:19])}</td>",
        ]
        if has_eda:
            if eda_row:
                cells.append(f"<td class='mono'>{eda_row.eda_n_findings_warning}</td>")
                cells.append(f"<td class='mono'>{eda_row.eda_n_findings_error}</td>")
            else:
                cells.append("<td class='muted'>n/a</td>")
                cells.append("<td class='muted'>n/a</td>")
        if has_disc:
            if disc_row:
                n = disc_row.discovery_n_findings_warning + disc_row.discovery_n_findings_error
                cells.append(f"<td class='mono'>{n}</td>")
            else:
                cells.append("<td class='muted'>n/a</td>")

        row_class = "champion-row" if i == 0 else ""
        rows_html.append(f"<tr class='{row_class}'>{''.join(cells)}</tr>")

    return f"""
<div class='section'>
  <h2>Champion Table</h2>
  <table class='data-table'>
    <thead><tr>{th}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>"""


def _build_metric_svg(
    harness: list[_FlexRow],
    champion_primary: float | None,
) -> str:
    values = [r.primary_metric_value for r in harness if r.primary_metric_value is not None]
    if not values:
        return "<div class='muted'>No metric values to plot.</div>"

    target = (champion_primary * 1.10) if champion_primary is not None else None
    target_label = f"{target:.4f}" if target is not None else None

    W, H = 700, 120
    ml, mr, mt, mb = 50, 80, 18, 28
    pw = W - ml - mr
    ph = H - mt - mb

    candidates = list(values)
    if target is not None:
        candidates.append(target)
    y_min = max(0.0, min(candidates) - 0.03)
    y_max = min(1.0, max(candidates) + 0.03)
    y_range = max(y_max - y_min, 1e-9)

    def sy(v: float) -> float:
        return mt + ph - (v - y_min) / y_range * ph

    def sx(i: int, n: int) -> float:
        if n <= 1:
            return ml + pw / 2
        return ml + (i / (n - 1)) * pw

    n = len(harness)
    parts: list[str] = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">',
        "<title>Primary metric values — all runs</title>",
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#ddd" stroke-width="1"/>',
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#ddd" stroke-width="1"/>',
    ]

    y_tick = math.floor(y_min * 20) / 20
    while y_tick <= y_max + 1e-9:
        if y_tick >= y_min - 1e-9:
            yp = sy(y_tick)
            parts.append(
                f'<line x1="{ml - 4}" y1="{yp:.1f}" x2="{ml + pw}" y2="{yp:.1f}" '
                f'stroke="#ddd" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{ml - 6}" y="{yp + 3:.1f}" text-anchor="end" font-size="9" '
                f'fill="#888" font-family="ui-monospace,monospace">{y_tick:.2f}</text>'
            )
        y_tick = round(y_tick + 0.05, 10)

    if target is not None and y_min <= target <= y_max:
        gyp = sy(target)
        parts.append(
            f'<line x1="{ml}" y1="{gyp:.1f}" x2="{ml + pw}" y2="{gyp:.1f}" '
            f'stroke="var(--ok)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.7"/>'
        )
        parts.append(
            f'<text x="{ml + pw + 3}" y="{gyp + 4:.1f}" font-size="9" fill="var(--ok)" '
            f'font-family="ui-monospace,monospace" opacity="0.9">'
            f"X × 1.10 target (= {target_label})</text>"
        )

    for i, row in enumerate(harness):
        v = row.primary_metric_value
        if v is None:
            continue
        xp = sx(i, n)
        yp = sy(v)
        above = target is not None and v >= target
        color = "var(--ok)" if above else "var(--accent)"
        outline = ' stroke="var(--ok)" stroke-width="2"' if above else ""
        tooltip = html.escape(f"{row.run_id} · {v:.4f}")
        parts.append(
            f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="5" fill="{color}" opacity="0.8"{outline}>'
            f"<title>{tooltip}</title></circle>"
        )
        lbl = html.escape(row.run_id)
        parts.append(
            f'<text x="{xp:.1f}" y="{yp - 8:.1f}" text-anchor="middle" font-size="9" '
            f'fill="var(--muted)" font-family="ui-monospace,monospace">{lbl}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_metric_grid(harness: list[_FlexRow], champion_primary: float | None) -> str:
    if not harness:
        return "<div class='muted'>No runs.</div>"

    target = (champion_primary * 1.10) if champion_primary else None
    target_str = f"{target:.4f}" if target else "n/a"

    rows_html: list[str] = []
    for row in harness:
        extra = row.model_extra or {}
        raw_pair = extra.get("roc_auc_raw_pair")
        raw_str = f"{raw_pair:.4f}" if raw_pair is not None else "—"
        residualized = row.primary_metric_value
        raw_f = raw_pair if raw_pair is not None else 0.0
        gap = (
            (raw_f - residualized) if (raw_pair is not None and residualized is not None) else None
        )
        gap_str = f"{gap:.4f}" if gap is not None else "—"
        rows_html.append(
            f"<tr>"
            f"<td class='mono'>{html.escape(row.run_id)}</td>"
            f"<td class='mono num'>{_fmt_metric(residualized)}</td>"
            f"<td class='mono muted'>{raw_str}</td>"
            f"<td class='mono muted'>{gap_str}</td>"
            f"</tr>"
        )

    svg = _build_metric_svg(harness, champion_primary)

    return f"""
<div class='section'>
  <h2>Primary vs Secondary Metric Grid</h2>
  <p class='meta'>Target: X × 1.10 = {html.escape(target_str)}</p>
  <table class='data-table'>
    <thead>
      <tr>
        <th>run_id</th>
        <th>primary_metric_value</th>
        <th>roc_auc_raw_pair</th>
        <th>residualization_gap (raw − residualized)</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
  <div class='svg-strip'>{svg}</div>
</div>"""


def _build_sample_tier(harness: list[_FlexRow]) -> str:
    if not harness:
        return "<div class='muted'>No runs.</div>"

    tier_order = ["smoke 0.1%", "1%", "5%", "full", "unknown"]
    groups: dict[str, list[float]] = {}
    for row in harness:
        extra = row.model_extra or {}
        sf = extra.get("sample_frac") or row.sample_frac
        label = _tier_label(sf)
        v = row.primary_metric_value
        if v is not None:
            groups.setdefault(label, []).append(v)

    rows_html: list[str] = []
    for tier in tier_order:
        vals = groups.get(tier)
        if not vals:
            continue
        mn = min(vals)
        mx = max(vals)
        med = statistics.median(vals)
        rows_html.append(
            f"<tr>"
            f"<td class='mono'>{html.escape(tier)}</td>"
            f"<td class='mono num'>{len(vals)}</td>"
            f"<td class='mono'>{mn:.4f}</td>"
            f"<td class='mono'>{med:.4f}</td>"
            f"<td class='mono'>{mx:.4f}</td>"
            f"</tr>"
        )

    if not rows_html:
        return "<div class='section'><h2>Sample-Tier Breakdown</h2><div class='muted'>No data.</div></div>"

    return f"""
<div class='section'>
  <h2>Sample-Tier Breakdown</h2>
  <table class='data-table'>
    <thead>
      <tr><th>tier</th><th>count</th><th>min primary</th><th>median primary</th><th>max primary</th></tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</div>"""


def _build_drift_report(
    harness: list[_FlexRow],
    eda_rows: list[_FlexEDARow],
    disc_rows: list[_FlexDiscRow],
    has_eda: bool,
    has_disc: bool,
    project: str,
) -> str:
    if not has_eda or not has_disc:
        return f"""
<div class='section drift-deferred'>
  <h2>Three-Way Drift Report</h2>
  <p class='warn-text'>
    EDA + discovery ledgers not yet present; three-way drift check deferred.
    (Remove this section when <code>projects/{html.escape(project)}/eda.py</code>
    + <code>discovery.py</code> + the paired ledgers exist.)
  </p>
</div>"""

    eda_by_key: dict[tuple[str, str, str, str], _FlexEDARow] = {}
    for r in eda_rows:
        key = (r.comparison_group, r.scope, r.recorded_at_utc, r.git_sha)
        eda_by_key[key] = r

    disc_by_key: dict[tuple[str, str, str, str], _FlexDiscRow] = {}
    for r in disc_rows:
        key = (r.comparison_group, r.scope, r.recorded_at_utc, r.git_sha)
        disc_by_key[key] = r

    flags: list[str] = []
    clean = 0
    for row in harness:
        key = (row.comparison_group, row.scope, row.recorded_at_utc, row.git_sha)
        missing = []
        if key not in eda_by_key:
            missing.append("EDA")
        if key not in disc_by_key:
            missing.append("discovery")
        if missing:
            flags.append(
                f"<li class='bad-text'>{html.escape(row.run_id)}: missing {', '.join(missing)} row "
                f"for key ({html.escape(row.comparison_group)}, {html.escape(row.scope)}, "
                f"{html.escape(row.recorded_at_utc[:19])}, {html.escape(row.git_sha[:8])})</li>"
            )
        else:
            clean += 1

    if flags:
        body = f"<ul class='drift-flags'>{''.join(flags)}</ul>"
        verdict = f"<span class='warn-text'>R4 violations found — {len(flags)} harness row(s) missing paired EDA/discovery rows.</span>"
    else:
        body = ""
        verdict = f"<span class='ok-text'>clean — {clean} of {len(harness)} in-scope harness rows have paired EDA + discovery rows</span>"

    return f"""
<div class='section'>
  <h2>Three-Way Drift Report (R4)</h2>
  <p>{verdict}</p>
  {body}
</div>"""


def _render_body(project: str) -> str:
    harness_rows_raw = _load_harness_rows(project)
    eda_rows = _load_eda_rows(project)
    disc_rows = _load_disc_rows(project)

    has_eda = bool(eda_rows)
    has_disc = bool(disc_rows)

    harness_sorted = sorted(
        harness_rows_raw,
        key=lambda r: r.primary_metric_value if r.primary_metric_value is not None else -999,
        reverse=True,
    )

    champion_primary = harness_sorted[0].primary_metric_value if harness_sorted else None

    eda_by_run = {r.run_id: r for r in eda_rows}
    disc_by_run = {r.run_id: r for r in disc_rows}

    champion_section = _build_champion_table(
        harness_sorted, eda_by_run, disc_by_run, has_eda, has_disc
    )
    metric_section = _build_metric_grid(harness_sorted, champion_primary)
    tier_section = _build_sample_tier(harness_sorted)
    drift_section = _build_drift_report(
        harness_sorted, eda_rows, disc_rows, has_eda, has_disc, project
    )

    now = dt.datetime.now().strftime("%H:%M:%S")
    n_runs = len(harness_sorted)
    champion_str = f"{champion_primary:.4f}" if champion_primary is not None else "—"

    return f"""
  <h1>leaderboard · {html.escape(project)}</h1>
  <div class='meta'>{n_runs} run(s) · champion {html.escape(champion_str)} · rendered {now}</div>

  {champion_section}
  {metric_section}
  {tier_section}
  {drift_section}

  <div class='footer'>
    source: projects/{html.escape(project)}/runs/{{results,eda_results,discovery_results}}.jsonl
    · regenerate with <code>python -m libs.leaderboard {html.escape(project)}</code>
  </div>
"""


_STYLE = """
    :root {
      --ink: #222; --paper: #fafafa; --muted: #888;
      --line: #ddd; --softline: #eee;
      --accent: #3b82f6; --accent-soft: #dbeafe;
      --ok: #16a34a; --ok-soft: #dcfce7;
      --warn: #d97706; --warn-soft: #fef3c7;
      --bad: #dc2626; --bad-soft: #fee2e2;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink); background: var(--paper);
      max-width: 1100px; margin: 24px auto; padding: 0 16px;
      font-size: 14px; line-height: 1.5;
    }
    h1 { margin: 0 0 4px 0; font-size: 22px; }
    h2 { font-size: 13px; color: var(--muted); margin: 0 0 8px 0;
         text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .meta { color: var(--muted); font-size: 12px; font-family: ui-monospace, "SF Mono", Menlo, monospace; margin-bottom: 16px; }
    code { font-family: ui-monospace, monospace; font-size: 12px; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
    .num { text-align: right; }
    .ok-text { color: var(--ok); font-weight: 600; }
    .warn-text { color: var(--warn); }
    .bad-text { color: var(--bad); }

    .pill {
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-family: ui-monospace, monospace;
    }
    .pill.scope-smoke { background: var(--softline); color: var(--muted); }
    .pill.scope-comparison { background: var(--accent-soft); color: var(--accent); }
    .pill.scope-champion_candidate { background: var(--ok-soft); color: var(--ok); }
    .pill.scope-reproduction { background: var(--warn-soft); color: var(--warn); }

    .section {
      border: 1px solid var(--line); border-radius: 6px;
      padding: 16px; margin: 16px 0;
    }
    .drift-deferred { border-left: 4px solid var(--warn); }

    .data-table {
      width: 100%; border-collapse: collapse; font-size: 12.5px;
    }
    .data-table th {
      text-align: left; padding: 6px 8px; border-bottom: 2px solid var(--line);
      color: var(--muted); font-size: 11px; text-transform: uppercase;
      letter-spacing: 0.04em; font-weight: 600;
    }
    .data-table td {
      padding: 6px 8px; border-bottom: 1px solid var(--softline);
      vertical-align: middle;
    }
    .data-table tbody tr:hover { background: var(--softline); }
    .champion-row td { font-weight: 700; }

    .svg-strip { margin-top: 12px; }

    .drift-flags { margin: 8px 0 0 0; padding-left: 20px; }
    .drift-flags li { margin: 4px 0; font-family: ui-monospace, monospace; font-size: 12px; }

    .footer {
      margin-top: 24px; color: var(--muted); font-size: 11px;
      font-family: ui-monospace, monospace;
    }
"""


def _render_shell(content_html: str, project: str, include_meta_refresh: bool = True) -> str:
    meta_refresh = '<meta http-equiv="refresh" content="15">' if include_meta_refresh else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {meta_refresh}
  <title>Leaderboard — {html.escape(project)}</title>
  <style>
{_STYLE}
  </style>
</head>
<body>
  <div id="content">{content_html}</div>
</body>
</html>
"""


def render(project: str, out_path: str | Path | None = None) -> Path:
    if out_path is None:
        out_dir = ROOT / "tmp" / "visualize" / project
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "leaderboard.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        body = _render_body(project)
        text = _render_shell(body, project, include_meta_refresh=True)
    except Exception as e:
        import html as _html

        text = (
            "<!doctype html><meta http-equiv='refresh' content='5'>"
            f"<pre style='color:#dc2626;font-family:monospace;padding:24px;'>"
            f"render error at {dt.datetime.now()}: {_html.escape(str(e))}</pre>"
        )

    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, out_path)
    print(f"[{dt.datetime.now():%H:%M:%S}] rendered {out_path} ({len(text):,} bytes)", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render the DS v2 project leaderboard.")
    parser.add_argument("project", help="Project name (subdirectory of projects/).")
    parser.add_argument(
        "--watch", action="store_true", help="Re-render every 15s until interrupted."
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="(reserved for future HTTP serve mode)"
    )
    args = parser.parse_args(argv)

    out_path = render(args.project)

    if not args.watch:
        return 0

    print(f"[watch] re-rendering every 15s — Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(15)
            render(args.project, out_path=out_path)
    except KeyboardInterrupt:
        print("\n[watch] stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
