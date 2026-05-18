"""Pure HTML/SVG renderer for the autoloop live dashboard.

Reads JSONL state files (brainstorm.jsonl, iterations.jsonl, budget.json,
config.yaml) from `projects/{project}/autoloop/` and assembles a
self-contained HTML page. No HTTP server; no side effects beyond writing HTML.

Output modes:
  Static: `python -m libs.autoloop.dashboard.render --project NAME` writes
          run-dashboard-{NAME}.html once and exits. `--watch` loops with a
          short sleep between renders.
  Dynamic: `serve_dashboard.py` calls `_render_body(...)` for polling updates
           and `_render_shell(...)` once per new browser tab; does not re-invoke
           this script as a subprocess.

Contract:
  Every `_build_*` function accepts parsed state dicts/lists and returns an
  HTML string. `_render_body(...)` assembles them into the page body fragment.
  `_render_shell(...)` wraps the body in a full <html> page with inline CSS/JS.
  `render_once(...)` is the top-level entry point for static rendering.

Key entry points:
  render_once(project, out_path)  -- static full-page render + write
  _render_shell(project, body)    -- full page wrapper (called by serve_dashboard)
  _render_body(project)           -- body fragment (called by /fragment endpoint)

Side effect: on every full render, each queued BrainstormItem gets its own
detail page written to `tmp/visualize/autoloop/brainstorm/{id}.html`.

Project resolution order: --project CLI flag > AUTOLOOP_PROJECT env var > default.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml

Stage = Literal["idle", "planning", "building", "running", "committing"]

PROJECT = os.environ.get("AUTOLOOP_PROJECT", "california_housing_demo")
ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "Makefile").exists():
    raise RuntimeError(f"repo root not found at {ROOT}; expected Makefile sentinel")
_OUT_DIR = ROOT / "tmp" / "visualize" / "autoloop"
AUTOLOOP_DIR = ROOT / "projects" / PROJECT / "autoloop"
RUNS_DIR = ROOT / "projects" / PROJECT / "runs"
OUT = _OUT_DIR / f"run-dashboard-{PROJECT}.html"
OUT_CANONICAL = _OUT_DIR / "run-dashboard.html"

CFG = AUTOLOOP_DIR / "config.yaml"
BUDGET = AUTOLOOP_DIR / "budget.json"
ITERS = AUTOLOOP_DIR / "iterations.jsonl"
BRAINSTORM = AUTOLOOP_DIR / "brainstorm.jsonl"
RESULTS = RUNS_DIR / "results.jsonl"


def _infer_stage_and_action(running: dict, iters: list[dict]) -> tuple[Stage, str | None]:
    try:
        if not running:
            if iters:
                last_iter = iters[-1]
                ts_end = last_iter.get("ts_end")
                if ts_end:
                    try:
                        ended = dt.datetime.fromisoformat(ts_end)
                        age_s = (dt.datetime.now() - ended).total_seconds()
                        if age_s <= 30:
                            return ("committing", "Writing iter record")
                    except Exception:
                        pass
            return ("idle", None)

        phase = running.get("phase", "")
        iter_n = running.get("iter")
        if iter_n is None:
            return ("idle", None)

        log_name = f"iter_{iter_n:03d}_{phase}.jsonl"
        log_path = AUTOLOOP_DIR / "logs" / log_name
        if not log_path.exists():
            stage: Stage = "planning" if phase == "planner" else "building"
            return (stage, None)

        lines = log_path.read_text().splitlines()
        tail = lines[-50:]

        last_tool: dict | None = None
        for raw in tail:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message", {})
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    last_tool = block

        if last_tool is None:
            stage = "planning" if phase == "planner" else "building"
            return (stage, None)

        tool_name = last_tool.get("name", "")
        tool_input = last_tool.get("input", {})

        if tool_name == "Read":
            fp = tool_input.get("file_path", "")
            desc = f"Read {Path(fp).name}" if fp else "Read"
        elif tool_name in ("Edit", "Write"):
            fp = tool_input.get("file_path", "")
            desc = f"Edit {Path(fp).name}" if fp else tool_name
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            short = cmd[:50]
            desc = f"Bash: {short}..." if len(cmd) > 50 else f"Bash: {cmd}"
        elif tool_name == "Skill":
            skill_name = tool_input.get("skill_name", "")
            desc = f"Skill: {skill_name}" if skill_name else "Skill"
        elif tool_name == "Agent":
            description = tool_input.get("description", "")
            short = description[:40]
            desc = f"Agent: {short}..." if len(description) > 40 else f"Agent: {description}"
        else:
            desc = tool_name

        if phase == "planner":
            inferred: Stage = "planning"
        elif tool_name in ("Edit", "Write"):
            inferred = "building"
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if "python -m projects" in cmd or "python projects/" in cmd:
                inferred = "running"
            else:
                inferred = "building"
        else:
            inferred = "building"

        return (inferred, desc)
    except Exception:
        return ("idle", None)


def _compute_iter_api_spend(iters: list[dict], cumulative_dollars_used: float) -> float:
    if iters:
        prev = iters[-1].get("budget_after", {}).get("dollars_used", 0.0)
        return max(0.0, cumulative_dollars_used - float(prev))
    return max(0.0, cumulative_dollars_used)


def _extract_family_from_wrote_files(wrote_files: list[str] | None) -> str | None:
    """Extract feature family name from any path like `features/{family}/...`."""
    if not wrote_files:
        return None
    for fp in wrote_files:
        m = re.search(r"/features/([^/]+)/", fp)
        if m:
            return m.group(1)
    return None


def _parse_run_num(run_id: str) -> int | None:
    if not run_id:
        return None
    m = re.search(r"_(\d+)$", run_id)
    if m:
        return int(m.group(1))
    m = re.search(r"run_(\d+)", run_id)
    if m:
        return int(m.group(1))
    return None


def _count_tools_used(running: dict) -> dict[str, int]:
    try:
        if not running:
            return {}
        phase = running.get("phase", "")
        iter_n = running.get("iter")
        if iter_n is None:
            return {}
        log_name = f"iter_{iter_n:03d}_{phase}.jsonl"
        log_path = AUTOLOOP_DIR / "logs" / log_name
        if not log_path.exists():
            return {}
        counts: dict[str, int] = {}
        for raw in log_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message", {})
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "unknown")
                    counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}


def _read_per_session_wall_cap() -> int:
    try:
        cfg = yaml.safe_load(CFG.read_text())
        return int(cfg["budget"]["per_session_wall_seconds_max"])
    except Exception:
        return 900


def _read_goal_metric() -> float | None:
    try:
        cfg = yaml.safe_load(CFG.read_text())
        v = cfg.get("goal_metric")
        return float(v) if v is not None else None
    except Exception:
        return None


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(obj, dict)
            and "_managed_by" in obj
            and isinstance(obj.get("schema_version"), str)
            and obj["schema_version"].startswith("ledger/")
        ):
            continue
        out.append(obj)
    return out


def read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def detect_running_phase() -> dict:
    """Return {'iter': N, 'phase': 'planner'|'executor', 'pid': X, 'started': iso}
    or {} if no autoloop session is currently running.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,lstart,command"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return {}
    for line in out.splitlines():
        if "claude -p" not in line:
            continue
        if "AUTOLOOP PLANNER" not in line and "AUTOLOOP EXECUTOR" not in line:
            continue
        phase = "planner" if "AUTOLOOP PLANNER" in line else "executor"
        iter_n = None
        if "iteration " in line:
            try:
                tail = line.split("iteration ", 1)[1]
                num = tail.split("/", 1)[0].strip()
                iter_n = int(num)
            except Exception:
                pass
        toks = line.split()
        pid = toks[0]
        try:
            started_str = " ".join(toks[1:6])
            started = dt.datetime.strptime(started_str, "%a %b %d %H:%M:%S %Y")
        except Exception:
            started = None
        return {
            "iter": iter_n,
            "phase": phase,
            "pid": pid,
            "started": started.isoformat() if started else None,
            "elapsed_s": int((dt.datetime.now() - started).total_seconds()) if started else None,
        }
    return {}


def fmt_dur(seconds: float | None) -> str:
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


def fmt_money(d: float | None) -> str:
    if d is None:
        return "—"
    return f"${d:,.2f}"


def _read_primary_metric_name(results: list[dict]) -> str:
    for r in results:
        name = r.get("primary_metric_name")
        if name:
            return str(name)
    return "metric"


_METRIC_NO_SKILL_FLOOR: dict[str, float] = {
    "roc_auc": 0.5,
    "pr_auc": 0.5,
    "auc": 0.5,
    "r2": 0.0,
    "accuracy": 0.5,
    "f1": 0.0,
}


def _read_baseline_metric(results: list[dict]) -> float | None:
    primary_metric = _read_primary_metric_name(results)
    if not primary_metric:
        return None
    if primary_metric in _METRIC_NO_SKILL_FLOOR:
        return _METRIC_NO_SKILL_FLOOR[primary_metric]
    lower = primary_metric.lower()
    for key, floor in _METRIC_NO_SKILL_FLOOR.items():
        if key in lower:
            return floor
    return None


def _read_champion_run_id() -> str | None:
    champion_path = RUNS_DIR / "CHAMPION.md"
    if not champion_path.exists():
        return None
    text = champion_path.read_text()
    m = re.search(r"(run_\d\w*)", text, re.ASCII)
    return m.group(1) if m else None


def _build_mockup_a_chart(
    iters: list[dict],
    baseline_metric: float | None,
    goal_metric: float | None,
    metric_name: str,
    champion_run_id: str | None,
) -> str:
    H = 200
    mt, mb = 20, 20
    ph = H - mt - mb

    iter_count = len(iters)
    iters_total = max(iter_count, 1)
    plot_w = max(740, 60 + iters_total * 44)

    metrics_with_iter: list[tuple[int, float]] = []
    for it in iters:
        exec_d = it.get("executor") or {}
        m = exec_d.get("harness_metric")
        if m is not None:
            metrics_with_iter.append((it.get("iteration_id", 0), float(m)))

    y_candidates: list[float] = [v for _, v in metrics_with_iter]
    if baseline_metric is not None:
        y_candidates.append(baseline_metric)
    if goal_metric is not None:
        y_candidates.append(goal_metric)

    if not y_candidates:
        y_min_raw, y_max_raw = 0.0, 1.0
    else:
        span = max(y_candidates) - min(y_candidates)
        pad = max(span * 0.10, 0.005)
        y_min_raw = min(y_candidates) - pad
        y_max_raw = max(y_candidates) + pad
    y_min = max(0.0, y_min_raw)
    y_max = y_max_raw
    y_range = y_max - y_min if y_max > y_min else 1.0

    n_ticks = 5
    tick_values = [y_min + i * y_range / (n_ticks - 1) for i in range(n_ticks)]

    def sy(v: float) -> float:
        return mt + ph - (v - y_min) / y_range * ph

    def x_for_iter(iter_id: int) -> float:
        return (iter_id - 1) * 44 + 22

    axis_parts: list[str] = []
    axis_parts.append(
        f'<svg viewBox="0 0 60 {H}" width="60" height="{H}" xmlns="http://www.w3.org/2000/svg">'
    )
    axis_parts.append(
        f'<line x1="59" y1="{mt}" x2="59" y2="{mt + ph}" stroke="#cbd5e1" stroke-width="1"/>'
    )
    for tv in tick_values:
        yp = sy(tv)
        axis_parts.append(
            f'<text x="55" y="{yp + 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#64748b" font-family="ui-monospace,monospace">'
            f"{tv:.3f}</text>"
        )
    escaped_name = html.escape(metric_name[:12])
    axis_parts.append(
        f'<text x="10" y="{mt + ph // 2}" text-anchor="middle" '
        f'transform="rotate(-90 10 {mt + ph // 2})" '
        f'font-size="9" fill="#64748b">{escaped_name}</text>'
    )
    axis_parts.append("</svg>")

    plot_parts: list[str] = []
    plot_parts.append(
        f'<svg viewBox="0 0 {plot_w} {H}" width="{plot_w}" height="{H}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    )

    plot_parts.append(
        f'<line x1="0" y1="{mt + ph}" x2="{plot_w}" y2="{mt + ph}" stroke="#cbd5e1" stroke-width="1"/>'
    )
    for tv in tick_values:
        yp = sy(tv)
        plot_parts.append(
            f'<line x1="0" y1="{yp:.1f}" x2="{plot_w}" y2="{yp:.1f}" '
            f'stroke="#e2e8f0" stroke-dasharray="2 3" stroke-width="0.8"/>'
        )

    if baseline_metric is not None and y_min <= baseline_metric <= y_max:
        byp = sy(baseline_metric)
        plot_parts.append(
            f'<line x1="0" y1="{byp:.1f}" x2="{plot_w}" y2="{byp:.1f}" '
            f'stroke="#475569" stroke-width="1.5" stroke-dasharray="6 3"/>'
        )
        plot_parts.append(
            f'<text x="{plot_w - 2}" y="{byp - 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#475569" font-family="ui-monospace,monospace">'
            f"baseline {baseline_metric:.3f}</text>"
        )

    if goal_metric is not None and y_min <= goal_metric <= y_max:
        gyp = sy(goal_metric)
        plot_parts.append(
            f'<line x1="0" y1="{gyp:.1f}" x2="{plot_w}" y2="{gyp:.1f}" '
            f'stroke="#16a34a" stroke-width="1.5" stroke-dasharray="6 3"/>'
        )
        plot_parts.append(
            f'<text x="{plot_w - 2}" y="{gyp - 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#16a34a" font-family="ui-monospace,monospace">'
            f"goal {goal_metric:.3f}</text>"
        )

    best_so_far_points: list[str] = []
    running_max: float | None = None
    for it in sorted(iters, key=lambda i: i.get("iteration_id", 0)):
        iter_id = it.get("iteration_id", 0)
        exec_d = it.get("executor") or {}
        m = exec_d.get("harness_metric")
        if m is not None:
            fv = float(m)
            xp = x_for_iter(iter_id)
            if running_max is not None and fv <= running_max:
                best_so_far_points.append(f"{xp:.1f},{sy(running_max):.1f}")
            else:
                if best_so_far_points:
                    best_so_far_points.append(
                        f"{xp:.1f},{sy(running_max if running_max is not None else fv):.1f}"
                    )
                running_max = fv
                best_so_far_points.append(f"{xp:.1f},{sy(running_max):.1f}")

    if len(best_so_far_points) > 1:
        pts = " ".join(best_so_far_points)
        plot_parts.append(
            f'<polyline fill="none" stroke="#6366f1" stroke-width="2.5" '
            f'stroke-linejoin="round" points="{pts}"/>'
        )

    for it in sorted(iters, key=lambda i: i.get("iteration_id", 0)):
        iter_id = it.get("iteration_id", 0)
        exec_d = it.get("executor")
        xp = x_for_iter(iter_id)

        x_tick_label_y = mt + ph + 14
        plot_parts.append(
            f'<text x="{xp:.1f}" y="{x_tick_label_y}" text-anchor="middle" '
            f'font-size="9" fill="#64748b" font-family="ui-monospace,monospace">{iter_id}</text>'
        )

        if exec_d is None:
            cross_y = mt + ph - 8
            plot_parts.append(
                f'<line x1="{xp - 6:.1f}" y1="{cross_y - 6:.1f}" '
                f'x2="{xp + 6:.1f}" y2="{cross_y + 6:.1f}" stroke="#dc2626" stroke-width="2"/>'
            )
            plot_parts.append(
                f'<line x1="{xp + 6:.1f}" y1="{cross_y - 6:.1f}" '
                f'x2="{xp - 6:.1f}" y2="{cross_y + 6:.1f}" stroke="#dc2626" stroke-width="2"/>'
            )
            continue

        m = exec_d.get("harness_metric")
        if m is None:
            continue

        fv = float(m)
        yp = sy(fv)
        exec_run_id = exec_d.get("run_id") or ""
        is_champion = champion_run_id is not None and exec_run_id == champion_run_id
        fill = "#f59e0b" if is_champion else "#6366f1"
        r = 7 if is_champion else 6
        plot_parts.append(
            f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="{r}" '
            f'fill="{fill}" stroke="white" stroke-width="1.5"/>'
        )
        if is_champion:
            plot_parts.append(
                f'<text x="{xp:.1f}" y="{yp - r - 3:.1f}" text-anchor="middle" '
                f'font-size="9" fill="#f59e0b" font-family="ui-monospace,monospace">★</text>'
            )

    plot_parts.append("</svg>")

    axis_svg = "\n".join(axis_parts)
    plot_svg = "\n".join(plot_parts)
    return (
        f'<div class="scroll-demo">'
        f'<div class="y-frozen">{axis_svg}</div>'
        f'<div class="x-scroll">{plot_svg}</div>'
        f"</div>"
    )


def _build_iter_table(
    iters: list[dict],
    baseline_metric: float | None,
    champion_run_id: str | None,
    metric_name: str,
    running: dict,
) -> str:
    current_iter_n = running.get("iter")

    rows: list[str] = []
    for it in sorted(iters, key=lambda i: i.get("iteration_id", 0)):
        iter_id = it.get("iteration_id", 0)
        planner = it.get("planner") or {}
        exec_d = it.get("executor")
        exec_dict = exec_d or {}

        run_id = exec_dict.get("run_id") or "—"
        wrote_files = exec_dict.get("wrote_files") if exec_d else None
        family = _extract_family_from_wrote_files(wrote_files) or "—"

        scope_raw = "—"
        if run_id != "—":
            if "_smoke" in run_id:
                scope_raw = "smoke"
            else:
                scope_raw = "comparison"

        metric = exec_dict.get("harness_metric") if exec_d else None
        metric_str = f"{float(metric):.3f}" if metric is not None else "—"

        if metric is not None and baseline_metric is not None:
            delta = float(metric) - baseline_metric
            delta_str = f"{delta:+.3f}"
            delta_cls = "up" if delta > 0 else "down"
        else:
            delta_str = "—"
            delta_cls = "muted"

        planner_dollars = float(planner.get("dollars") or 0.0)
        exec_dollars = float(exec_dict.get("dollars") or 0.0) if exec_d else 0.0
        total_dollars = planner_dollars + exec_dollars
        dollars_str = f"${total_dollars:.2f}"

        planner_wall = float(planner.get("wall_seconds") or 0.0)
        exec_wall = float(exec_dict.get("wall_seconds") or 0.0) if exec_d else 0.0
        total_wall = planner_wall + exec_wall
        w_m, w_s = divmod(int(total_wall), 60)
        wall_str = f"{w_m}m{w_s:02d}s"

        exec_run_id = exec_dict.get("run_id") or ""
        is_champion = champion_run_id is not None and exec_run_id == champion_run_id

        if is_champion:
            status_html = '<span class="status-ok">✓ champion</span>'
            row_cls = "champion-row"
        elif exec_d is not None:
            rrs = exec_dict.get("run_record_status", "unknown")
            rrs_esc = html.escape(rrs)
            if rrs == "completed":
                status_html = '<span class="status-ok">✓ shipped</span>'
                row_cls = ""
            elif rrs.startswith("failed_"):
                status_html = f'<span class="status-bad">× {rrs_esc}</span>'
                row_cls = "crashed-row"
            elif rrs in ("pending", "running", "in_progress"):
                status_html = f'<span class="status-running">⌛ {rrs_esc}</span>'
                row_cls = "running-row"
            else:
                status_html = f'<span class="status-running">⚠ {rrs_esc}</span>'
                row_cls = ""
        else:
            planner_kill = planner.get("kill_reason")
            planner_exit = planner.get("exit_code", 0)
            if planner_kill != "normal" or (planner_exit != 0):
                status_html = '<span class="status-bad">× planner crashed</span>'
                row_cls = "crashed-row"
            else:
                status_html = '<span class="status-bad">× planner crashed</span>'
                row_cls = "crashed-row"

        rows.append(
            f'<tr class="{row_cls}">'
            f"<td>{iter_id:02d}</td>"
            f"<td>{html.escape(run_id)}</td>"
            f"<td>{html.escape(family)}</td>"
            f"<td>{html.escape(scope_raw)}</td>"
            f'<td class="num">{metric_str}</td>'
            f'<td class="num {delta_cls}">{delta_str}</td>'
            f'<td class="num">{dollars_str}</td>'
            f'<td class="num">{wall_str}</td>'
            f"<td>{status_html}</td>"
            f"</tr>"
        )

    if running and current_iter_n is not None:
        is_highest = not any(it.get("iteration_id", 0) >= current_iter_n for it in iters)
        if is_highest:
            planner_elapsed = running.get("elapsed_s") or 0
            w_m, w_s = divmod(int(planner_elapsed), 60)
            wall_str = f"{w_m}m{w_s:02d}s"
            rows.append(
                f'<tr class="running-row">'
                f"<td>{current_iter_n:02d}</td>"
                f"<td>—</td>"
                f"<td>(in progress)</td>"
                f"<td>—</td>"
                f'<td class="num">—</td>'
                f'<td class="num">—</td>'
                f'<td class="num">…</td>'
                f'<td class="num">{wall_str}</td>'
                f'<td><span class="status-running">⌛ in progress</span></td>'
                f"</tr>"
            )

    metric_col_header = html.escape(metric_name[:20])
    thead = (
        f"<thead><tr>"
        f"<th>iter</th>"
        f"<th>run_id</th>"
        f"<th>family</th>"
        f"<th>scope</th>"
        f'<th class="num">{metric_col_header}</th>'
        f'<th class="num">Δ baseline</th>'
        f'<th class="num">$</th>'
        f'<th class="num">wall</th>'
        f"<th>status</th>"
        f"</tr></thead>"
    )
    return (
        f'<div class="data-table-wrap" style="max-height:320px;overflow-y:auto;margin-top:12px;">'
        f'<table class="data-table">'
        f"{thead}"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


_STAGE_COLOR: dict[str, str] = {
    "data_load": "var(--st-load)",
    "load": "var(--st-load)",
    "data_conversion": "var(--st-load)",
    "eda": "var(--st-eda)",
    "train": "var(--st-train)",
    "fit": "var(--st-train)",
    "predict": "var(--st-predict)",
    "eval": "var(--st-eval)",
    "metrics": "var(--st-eval)",
    "score": "var(--st-eval)",
    "aggregate": "var(--st-discover)",
    "discover_gaps": "var(--st-discover)",
    "discovery": "var(--st-discover)",
    "write": "var(--st-write)",
    "persist": "var(--st-write)",
    "column_selection": "var(--st-write)",
    "split_generation": "var(--st-write)",
}

_STAGE_LEGEND_HTML = (
    '<div class="stage-legend">'
    '<span><span class="swatch sw-load"></span>load/convert</span>'
    '<span><span class="swatch sw-eda"></span>eda</span>'
    '<span><span class="swatch sw-train"></span>fit/split</span>'
    '<span><span class="swatch sw-predict"></span>predict</span>'
    '<span><span class="swatch sw-eval"></span>score/eval</span>'
    '<span><span class="swatch sw-discover"></span>discover</span>'
    '<span><span class="swatch sw-write"></span>write</span>'
    "</div>"
)


def _read_timing_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _fmt_wall(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def _build_stage_breakdown(harness_timing: list[dict]) -> str:
    if not harness_timing:
        return (
            '<div class="alert">No harness timing data yet for this project. '
            "Timing rows are appended by <code>HarnessResponse.performance</code> "
            "when runs complete.</div>"
        )

    rows = sorted(harness_timing, key=lambda r: r.get("recorded_at_utc", ""))[-10:]
    if not rows:
        return (
            '<div class="alert">No harness timing data yet for this project. '
            "Timing rows are appended by <code>HarnessResponse.performance</code> "
            "when runs complete.</div>"
        )

    max_total = max((float(r.get("total_seconds", 0)) for r in rows), default=1.0)
    if max_total <= 0:
        max_total = 1.0

    bar_max_px = 520.0
    x_start = 180
    bar_h = 26
    row_gap = 38
    grid_step = 100

    n_rows = len(rows)
    svg_h = 40 + n_rows * row_gap + 40
    grid_end_y = svg_h - 30

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 820 {svg_h}" width="100%" height="{svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:820px;">'
    )

    grid_n = int(max_total / grid_step) + 1
    scale = bar_max_px / max_total
    for gi in range(1, grid_n + 1):
        gx = x_start + gi * grid_step * scale
        if gx > x_start + bar_max_px + 2:
            break
        parts.append(
            f'<line x1="{gx:.1f}" y1="30" x2="{gx:.1f}" y2="{grid_end_y}" '
            f'stroke="#e2e8f0" stroke-dasharray="2 3" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{grid_end_y + 14}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8" font-family="ui-monospace,monospace">'
            f"{gi * grid_step}s</text>"
        )

    callout_lines: list[str] = []

    for i, row in enumerate(rows):
        y_bar = 40 + i * row_gap
        y_label = y_bar + bar_h // 2 + 4
        run_id = str(row.get("run_id", f"row_{i}"))
        total_s = float(row.get("total_seconds", 0))
        stage_seconds: dict[str, float] = row.get("stage_seconds", {})

        parts.append(
            f'<text x="{x_start - 5}" y="{y_label}" text-anchor="end" '
            f'font-size="10" fill="#334155" font-family="ui-monospace,monospace" font-weight="600">'
            f"{html.escape(run_id[:20])}</text>"
        )

        x_cursor = float(x_start)
        stage_items = sorted(stage_seconds.items(), key=lambda kv: kv[1], reverse=True)
        for stage, secs in stage_items:
            w = secs * scale
            if w < 0.5:
                continue
            color = _STAGE_COLOR.get(stage, "#9ca3af")
            parts.append(
                f'<rect x="{x_cursor:.1f}" y="{y_bar}" width="{w:.1f}" height="{bar_h}" '
                f'fill="{color}"/>'
            )
            x_cursor += w
            if total_s > 0 and secs / total_s > 0.30 and secs > 30:
                pct = secs / total_s * 100
                callout_lines.append(
                    f"<li><code>{html.escape(run_id)}</code>: <code>{html.escape(stage)}</code> "
                    f"= {secs:.0f}s = {pct:.0f}% of wall</li>"
                )

        wall_x = x_start + total_s * scale + 6
        parts.append(
            f'<text x="{wall_x:.1f}" y="{y_label}" '
            f'font-size="10" fill="#64748b" font-family="ui-monospace,monospace">'
            f"{_fmt_wall(total_s)}</text>"
        )

    parts.append("</svg>")
    svg_html = "\n".join(parts)

    alert_html = ""
    if callout_lines:
        alert_html = (
            '<div class="alert"><strong>Auto-callouts:</strong>'
            f"<ul>{''.join(callout_lines)}</ul></div>"
        )

    return _STAGE_LEGEND_HTML + svg_html + alert_html


def _sample_tier_color(sample_frac: float | None) -> tuple[str, str]:
    """Return (color_hex, tier_label) for a sample_frac value."""
    if sample_frac is None or sample_frac >= 1.0:
        return ("#16a34a", "full")
    if sample_frac < 0.01:
        return ("#94a3b8", "smoke")
    return ("#6366f1", "sample")


_SCOPE_LEGEND_HTML = (
    '<div class="scope-legend">'
    '<span><span class="swatch" style="background:#94a3b8"></span>smoke (&lt;&nbsp;1%)</span>'
    '<span><span class="swatch" style="background:#6366f1"></span>sample (1–99%)</span>'
    '<span><span class="swatch" style="background:#16a34a"></span>full (&ge;&nbsp;100%)</span>'
    "</div>"
)


def _build_scaling_curve(harness_timing: list[dict]) -> str:
    samples: list[tuple[float, float, float | None, str]] = []
    for row in harness_timing:
        total_s = float(row.get("total_seconds", 0))
        n_test = float(row.get("n_test_rows", 0) or 0)
        n_train = float(row.get("n_train_rows", 0) or 0)
        rows = n_test + n_train
        if rows <= 0 or total_s <= 0:
            continue
        raw_sf = row.get("sample_frac")
        sf: float | None = float(raw_sf) if raw_sf is not None else None
        run_id = str(row.get("run_id", ""))
        sf_display = sf if sf is not None else 0.0
        label = (
            f"{sf_display * 100:.0f}%/{_fmt_wall(total_s)}"
            if sf_display > 0
            else f"{run_id}/{_fmt_wall(total_s)}"
        )
        samples.append((rows, total_s, sf, label))

    unique_rows = {s[0] for s in samples}
    if len(unique_rows) < 3:
        n = len(unique_rows)
        return (
            '<div class="alert">'
            f"Need &ge;3 sample tiers (e.g. 1%/10%/100%) to draw a scaling curve. "
            f"Currently have <code>{n}</code>. Run <code>make autoloop-once</code> at "
            "different sample fractions to populate.</div>"
        )

    samples.sort(key=lambda s: s[0])

    rows_min_data = min(s[0] for s in samples)
    rows_max_data = max(s[0] for s in samples)
    secs_min_data = min(s[1] for s in samples)
    secs_max_data = max(s[1] for s in samples)

    x_floor_exp = math.floor(math.log10(rows_min_data))
    x_ceil_exp = math.ceil(math.log10(rows_max_data))
    if x_ceil_exp <= x_floor_exp:
        x_ceil_exp = x_floor_exp + 1
    x_min_log = float(x_floor_exp)
    x_max_log = float(x_ceil_exp)

    y_floor_exp = math.floor(math.log10(max(secs_min_data, 1)))
    y_ceil_exp = math.ceil(math.log10(max(secs_max_data, 1)))
    if y_ceil_exp <= y_floor_exp:
        y_ceil_exp = y_floor_exp + 1
    y_min_log = float(y_floor_exp)
    y_max_log = float(y_ceil_exp)

    px_l, px_r = 70.0, 770.0
    py_t, py_b = 30.0, 310.0
    pw = px_r - px_l
    ph = py_b - py_t

    def sx(log_rows: float) -> float:
        return px_l + (log_rows - x_min_log) / (x_max_log - x_min_log) * pw

    def sy_log(log_secs: float) -> float:
        return py_t + (y_max_log - log_secs) / (y_max_log - y_min_log) * ph

    svg_parts: list[str] = []
    svg_parts.append(
        '<svg viewBox="0 0 800 380" width="100%" height="380" '
        'xmlns="http://www.w3.org/2000/svg" style="max-width:800px;">'
    )

    svg_parts.append(f'<line x1="{px_l}" y1="{py_t}" x2="{px_l}" y2="{py_b}" stroke="#cbd5e1"/>')
    svg_parts.append(f'<line x1="{px_l}" y1="{py_b}" x2="{px_r}" y2="{py_b}" stroke="#cbd5e1"/>')

    def _fmt_rows_tick(v: float) -> str:
        if v >= 1e9:
            return f"{v / 1e9:.0f}B"
        if v >= 1e6:
            return f"{v / 1e6:.0f}M"
        if v >= 1e3:
            return f"{v / 1e3:.0f}k"
        return str(int(v))

    x_ticks = [(10**e, _fmt_rows_tick(10**e)) for e in range(int(x_min_log), int(x_max_log) + 1)]
    for val, lbl in x_ticks:
        xp = sx(math.log10(val))
        svg_parts.append(
            f'<text x="{xp:.1f}" y="{py_b + 15}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8" font-family="ui-monospace,monospace">{lbl}</text>'
        )
        if math.log10(val) > x_min_log:
            svg_parts.append(
                f'<line x1="{xp:.1f}" y1="{py_t}" x2="{xp:.1f}" y2="{py_b}" '
                f'stroke="#e2e8f0" stroke-dasharray="2 3"/>'
            )

    def _fmt_secs_tick(v: float) -> str:
        if v >= 3600:
            return f"{v / 3600:.0f}h"
        if v >= 60:
            return f"{v / 60:.0f}m"
        return f"{v:.0f}s"

    y_ticks = [(10**e, _fmt_secs_tick(10**e)) for e in range(int(y_min_log), int(y_max_log) + 1)]
    for val, lbl in y_ticks:
        yp = sy_log(math.log10(val))
        svg_parts.append(
            f'<text x="{px_l - 6}" y="{yp + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#94a3b8" font-family="ui-monospace,monospace">{lbl}</text>'
        )
        svg_parts.append(
            f'<line x1="{px_l}" y1="{yp:.1f}" x2="{px_r}" y2="{yp:.1f}" '
            f'stroke="#e2e8f0" stroke-dasharray="2 3"/>'
        )

    svg_parts.append(
        '<text font-size="11" fill="#64748b" font-family="ui-monospace,monospace" '
        f'x="{(px_l + px_r) / 2:.0f}" y="{py_b + 32}" text-anchor="middle">'
        "rows processed (log scale)</text>"
    )
    svg_parts.append(
        f'<text font-size="11" fill="#64748b" font-family="ui-monospace,monospace" '
        f'x="16" y="{(py_t + py_b) / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {(py_t + py_b) / 2:.0f})">wall_seconds (log)</text>'
    )

    ref_x0_log = math.log10(max(samples[0][0], 1))
    ref_y0_log = math.log10(max(samples[0][1], 1))
    ref_x1_log = x_max_log
    ref_x0_s = sx(ref_x0_log)
    ref_x1_s = sx(ref_x1_log)
    ref_anchor_y = sy_log(ref_y0_log)

    on_slope = 1.0
    on_logn_slope = 1.1
    on2_slope = 2.0
    dx_log = ref_x1_log - ref_x0_log

    on_y0 = ref_anchor_y
    on_y1 = sy_log(ref_y0_log + on_slope * dx_log)
    svg_parts.append(
        f'<line x1="{ref_x0_s:.1f}" y1="{on_y0:.1f}" x2="{ref_x1_s:.1f}" y2="{on_y1:.1f}" '
        f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.6"/>'
    )
    svg_parts.append(
        f'<text font-size="10" fill="#64748b" font-family="ui-monospace,monospace" '
        f'x="{ref_x1_s - 4:.1f}" y="{max(on_y1 - 4, py_t + 12):.1f}" text-anchor="end" opacity="0.9">O(n)</text>'
    )

    onlogn_y1 = sy_log(ref_y0_log + on_logn_slope * dx_log)
    svg_parts.append(
        f'<line x1="{ref_x0_s:.1f}" y1="{on_y0:.1f}" x2="{ref_x1_s:.1f}" y2="{onlogn_y1:.1f}" '
        f'stroke="#f97316" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.6"/>'
    )
    svg_parts.append(
        f'<text font-size="10" fill="#f97316" font-family="ui-monospace,monospace" '
        f'x="{ref_x1_s - 4:.1f}" y="{max(onlogn_y1 - 4, py_t + 12):.1f}" text-anchor="end" opacity="0.9">O(n log n)</text>'
    )

    dy_log_to_top = y_max_log - ref_y0_log
    on2_dx_max = dy_log_to_top / on2_slope if on2_slope > 0 else dx_log
    on2_x1_log = min(ref_x0_log + on2_dx_max, ref_x1_log)
    on2_y1 = sy_log(ref_y0_log + on2_slope * (on2_x1_log - ref_x0_log))
    svg_parts.append(
        f'<line x1="{ref_x0_s:.1f}" y1="{on_y0:.1f}" x2="{sx(on2_x1_log):.1f}" y2="{on2_y1:.1f}" '
        f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6 4" opacity="0.5"/>'
    )
    svg_parts.append(
        f'<text font-size="10" fill="#dc2626" font-family="ui-monospace,monospace" '
        f'x="{sx(on2_x1_log) - 4:.1f}" y="{max(on2_y1 - 4, py_t + 12):.1f}" text-anchor="end" opacity="0.9">O(n²)</text>'
    )

    dot_pts: list[str] = []
    for rows, total_s, _sf, label in samples:
        log_r = math.log10(max(rows, 1))
        log_s = math.log10(max(total_s, 1))
        if log_r < x_min_log or log_r > x_max_log + 0.1:
            continue
        if log_s < y_min_log or log_s > y_max_log + 0.3:
            continue
        xp = sx(log_r)
        yp = sy_log(log_s)
        dot_pts.append(f"{xp:.1f},{yp:.1f}")

    if len(dot_pts) > 1:
        svg_parts.append(
            f'<polyline fill="none" stroke="#6366f1" stroke-width="2" opacity="0.4" '
            f'points="{" ".join(dot_pts)}"/>'
        )

    for j, (rows, total_s, sf, label) in enumerate(samples):
        log_r = math.log10(max(rows, 1))
        log_s = math.log10(max(total_s, 1))
        if log_r < x_min_log or log_r > x_max_log + 0.1:
            continue
        if log_s < y_min_log or log_s > y_max_log + 0.3:
            continue
        xp = sx(log_r)
        yp = sy_log(log_s)
        dot_color, _tier = _sample_tier_color(sf)
        svg_parts.append(
            f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="6" fill="{dot_color}" stroke="white" stroke-width="2"/>'
        )
        svg_parts.append(
            f'<text font-size="10" fill="#64748b" font-family="ui-monospace,monospace" '
            f'x="{xp + 9:.1f}" y="{yp - 4:.1f}">{html.escape(label)}</text>'
        )

    svg_parts.append("</svg>")
    svg_html = "\n".join(svg_parts)

    def _last_segment_slope(pts: list[tuple[float, float, float | None, str]]) -> float | None:
        if len(pts) < 2:
            return None
        (x1, y1, _, _), (x2, y2, _, _) = pts[-2], pts[-1]
        if x1 <= 0 or x2 <= 0 or y1 <= 0 or y2 <= 0 or x1 == x2:
            return None
        return math.log(y2 / y1) / math.log(x2 / x1)

    callout_html = ""
    slope = _last_segment_slope(samples)
    if slope is not None:
        slope_str = f"{slope:.2f}"
        if slope > 1.8:
            callout_html = (
                f'<div class="alert"><strong>Warning:</strong> Latest run scales super-quadratically '
                f"(slope ≈ {slope_str}) — likely a missing index, bad join, or accidental full-shuffle.</div>"
            )
        elif slope > 1.2:
            callout_html = (
                f'<div class="alert warn"><strong>Note:</strong> Latest run scales worse than linear '
                f"(slope ≈ {slope_str}) — runtime grows faster than rows.</div>"
            )
        else:
            callout_html = (
                f'<div class="reco">Healthy scaling: latest run grows roughly linearly with rows '
                f"(slope ≈ {slope_str}).</div>"
            )

    return _SCOPE_LEGEND_HTML + svg_html + callout_html


def _build_family_throughput(project_root: Path) -> str:
    features_dir = project_root / "features"
    if not features_dir.is_dir():
        return (
            '<div class="alert">No <code>features/</code> directory found for this project.</div>'
        )

    def _row_count(r: dict) -> float:
        for key in ("rows", "rows_processed", "rows_sample", "n_rows", "sample_n_rows"):
            v = r.get(key)
            if v is not None:
                return float(v)
        return 0.0

    family_data: list[tuple[str, float, float, str]] = []
    for timing_path in sorted(features_dir.glob("*/timing_performance.jsonl")):
        family = timing_path.parent.name
        rows_data = _read_timing_jsonl(timing_path)
        if not rows_data:
            continue

        best = max(rows_data, key=_row_count)

        rps = float(best.get("rows_per_second", 0) or 0)
        wall = float(
            best.get("build_seconds", best.get("wall_seconds", best.get("total_seconds", 0))) or 0
        )

        if rps <= 0 and wall > 0:
            rows_count = _row_count(best)
            if rows_count > 0:
                rps = rows_count / wall

        tier = str(best.get("tier", ""))

        if rps > 0:
            family_data.append((family, rps, wall, tier))

    if not family_data:
        return (
            '<div class="alert">No feature family timing data yet. Timing rows are written by '
            "<code>features/{family}/timing_performance.jsonl</code> during feature cache builds.</div>"
        )

    family_data.sort(key=lambda t: t[1])

    max_rps = max(t[1] for t in family_data)
    if max_rps <= 0:
        max_rps = 1.0

    bar_max_px = 530.0
    x_start = 240.0
    bar_h = 28
    row_gap = 40
    n = len(family_data)
    svg_h = 30 + n * row_gap + 30

    def _color(wall: float) -> str:
        if wall > 60:
            return "var(--bad)"
        if wall > 30:
            return "var(--warn)"
        if wall > 15:
            return "#fbbf24"
        if wall > 5:
            return "#84cc16"
        return "#15803d"

    def _fmt_rps(rps: float) -> str:
        if rps >= 1_000_000:
            return f"{rps / 1_000_000:.1f}M"
        if rps >= 1_000:
            return f"{rps / 1_000:.1f}k"
        return f"{rps:.0f}"

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 820 {svg_h}" width="100%" height="{svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:820px;">'
    )

    callout_parts: list[str] = []

    for i, (family, rps, wall, tier) in enumerate(family_data):
        y_bar = 30 + i * row_gap
        y_label = y_bar + bar_h // 2 + 4
        bar_w = rps / max_rps * bar_max_px
        color = _color(wall)

        parts.append(
            f'<text x="{x_start - 6:.1f}" y="{y_label}" text-anchor="end" '
            f'font-size="10" fill="#334155" font-family="ui-monospace,monospace" font-weight="600">'
            f"{html.escape(family[:28])}</text>"
        )
        parts.append(
            f'<rect x="{x_start:.1f}" y="{y_bar}" width="{bar_w:.1f}" height="{bar_h}" '
            f'fill="{color}"/>'
        )
        tier_suffix = f" · tier={tier}" if tier and tier != "full" else ""
        ann = f"{_fmt_rps(rps)} rows/s · {_fmt_wall(wall)} build{tier_suffix}"
        ann_x = x_start + bar_w + 6
        parts.append(
            f'<text x="{ann_x:.1f}" y="{y_label}" '
            f'font-size="10" fill="#64748b" font-family="ui-monospace,monospace">'
            f"{html.escape(ann)}</text>"
        )

    parts.append("</svg>")
    svg_html = "\n".join(parts)

    if len(family_data) >= 2:
        total_wall = sum(t[2] for t in family_data)
        slowest_name = family_data[0][0]
        slowest_rps = family_data[0][1]
        median_rps = family_data[len(family_data) // 2][1]
        if median_rps > 0 and slowest_rps < median_rps / 2:
            ratio = median_rps / slowest_rps
            callout_parts.append(
                f"<li><code>{html.escape(slowest_name)}</code> is {ratio:.1f}&times; slower than the median family</li>"
            )

        if len(family_data) >= 3 and total_wall > 0:
            top3_wall = sum(t[2] for t in family_data[:3])
            pct = top3_wall / total_wall * 100
            if pct > 50:
                names = ", ".join(f"<code>{html.escape(t[0])}</code>" for t in family_data[:3])
                callout_parts.append(
                    f"<li>Top 3 ({names}) consume {pct:.0f}% of total feature build wall — optimization candidates</li>"
                )

    alert_html = ""
    if callout_parts:
        alert_html = (
            '<div class="alert"><strong>Auto-callouts:</strong>'
            f"<ul>{''.join(callout_parts)}</ul></div>"
        )

    return svg_html + alert_html


def _build_performance_section(project_root: Path) -> str:
    harness_timing = _read_timing_jsonl(project_root / "timing_performance.jsonl")

    panel_a = _build_stage_breakdown(harness_timing)
    panel_b = _build_scaling_curve(harness_timing)
    panel_c = _build_family_throughput(project_root)

    return f"""
<div class="perf-section">
  <h2>Performance</h2>
  <p class="blurb">Where the actual computation time goes &mdash; three views on the timing data we already collect.</p>

  <div class="perf-panel">
    <h3>A &mdash; Per-run stage breakdown</h3>
    {panel_a}
  </div>

  <div class="perf-panel">
    <h3>B &mdash; Scaling curve (log-log)</h3>
    {panel_b}
  </div>

  <div class="perf-panel">
    <h3>C &mdash; Feature family throughput</h3>
    {panel_c}
  </div>
</div>
"""


def latest_brainstorm_by_id(brainstorm_rows: list[dict]) -> dict[str, dict]:
    """Return the latest row per brainstorm id, skipping non-brainstorm schemas."""
    result: dict[str, dict] = {}
    for row in brainstorm_rows:
        if row.get("schema") == "brainstorm/v1":
            rid = row.get("id")
            if rid:
                result[rid] = row
    return result


def _render_body() -> str:
    budget = read_json(BUDGET)
    iters = read_jsonl(ITERS)
    brainstorm = read_jsonl(BRAINSTORM)
    results = read_jsonl(RESULTS)

    iters_used = budget.get("iterations_used", len(iters))
    iters_max = budget.get("iterations_max", 10)
    dollars_used = budget.get("dollars_used", 0.0)
    dollars_max = budget.get("dollars_max", 100.0)
    wall_used = budget.get("wall_seconds_used", 0.0)
    wall_max = budget.get("wall_seconds_max", 7200.0)
    updated = budget.get("updated_at", "—")

    latest_by_id = latest_brainstorm_by_id(brainstorm)
    consumed_ids: set[str] = set()
    for it in iters:
        rid = it.get("brainstorm_id")
        if rid:
            consumed_ids.add(rid)

    queued = [
        b
        for b in latest_by_id.values()
        if b.get("id") not in consumed_ids and b.get("status", "queued") == "queued"
    ]
    queued.sort(key=lambda b: float(b.get("tier_score") or 0), reverse=True)

    res_by_run = {r.get("run_id"): r for r in results if isinstance(r, dict)}

    running = detect_running_phase()
    stage, last_action = _infer_stage_and_action(running, iters)
    iter_api_spend = _compute_iter_api_spend(iters, dollars_used)
    tools = _count_tools_used(running)

    current_iter_n = running.get("iter")
    if current_iter_n is None and running:
        current_iter_n = len(iters) + 1

    rows_html: list[str] = []
    icon_map = {
        "shipped": ("ok", "✓"),
        "planner_crashed": ("bad", "×"),
        "executor_crashed": ("bad", "×"),
        "no_executor": ("warn", "≈"),
        "unknown": ("muted", "◯"),
    }
    for it in iters:
        n = it.get("iteration_id")
        planner = it.get("planner") or {}
        executor = it.get("executor")
        executor_d = executor or {}

        run_id = executor_d.get("run_id") if executor else None
        family = (
            _extract_family_from_wrote_files(executor_d.get("wrote_files") if executor else None)
            or "—"
        )

        planner_dollars = float(planner.get("dollars") or 0.0)
        executor_dollars = float(executor_d.get("dollars") or 0.0) if executor else 0.0
        dollars = planner_dollars + executor_dollars

        planner_wall = float(planner.get("wall_seconds") or 0.0)
        executor_wall = float(executor_d.get("wall_seconds") or 0.0) if executor else 0.0
        wall = planner_wall + executor_wall

        if executor is None:
            planner_kill = planner.get("kill_reason")
            outcome = "no_executor" if planner_kill == "normal" else "planner_crashed"
        elif executor_d.get("exit_code") != 0 or executor_d.get("kill_reason") != "normal":
            outcome = "executor_crashed"
        elif executor_d.get("run_id"):
            outcome = "shipped"
        else:
            outcome = "unknown"

        metric = None
        if run_id and run_id in res_by_run:
            r = res_by_run[run_id]
            metric = r.get("primary_metric_value")
        if metric is None and executor_d.get("harness_metric") is not None:
            metric = executor_d.get("harness_metric")

        ok_class, icon_char = icon_map.get(outcome, ("muted", "◯"))
        metric_str = f"{metric:.4f}" if metric is not None else "—"
        rows_html.append(
            f"""
            <div class="task done">
              <span class="status {ok_class}">{icon_char}</span>
              <span class="iter-n">iter {n:02d}</span>
              <span class="family">{html.escape(family)}</span>
              <span class="run-id">{html.escape(run_id or "—")}</span>
              <span class="metric">{metric_str}</span>
              <span class="cost">{fmt_money(dollars)}</span>
              <span class="wall">{fmt_dur(wall)}</span>
              <span class="bid">—</span>
            </div>
            """
        )
    if running:
        phase_pill = f"<span class='pill phase-{running['phase']}'>{running['phase']}</span>"
        elapsed = fmt_dur(running.get("elapsed_s"))
        next_idea = queued[0] if queued else None
        next_title = html.escape(next_idea.get("title", "?")) if next_idea else "—"
        rows_html.append(
            f"""
            <div class="task running">
              <span class="status accent">■</span>
              <span class="iter-n">iter {current_iter_n:02d}</span>
              <span class="family">{phase_pill} <span class="muted">running {elapsed}</span></span>
              <span class="run-id muted">—</span>
              <span class="metric muted">…</span>
              <span class="cost muted">…</span>
              <span class="wall muted">{elapsed}</span>
              <span class="bid muted">next: {next_title}</span>
            </div>
            """
        )
    completed_count = len(iters) + (1 if running else 0)
    remaining = max(0, iters_max - completed_count)
    queue_for_display = queued[:remaining] if running else queued[: max(0, iters_max - len(iters))]
    start_at = 1 if running else 0
    for i, idea in enumerate(queue_for_display[start_at:], start=completed_count + 1):
        if i > iters_max:
            break
        title = html.escape(idea.get("title", "?"))
        tier = idea.get("tier", "?")
        score = idea.get("tier_score")
        score_s = f"{float(score):.2f}" if score is not None else "—"
        rows_html.append(
            f"""
            <div class="task queued">
              <span class="status muted">◯</span>
              <span class="iter-n">iter {i:02d}</span>
              <span class="family muted">{title}</span>
              <span class="run-id muted">—</span>
              <span class="metric muted">—</span>
              <span class="cost muted">—</span>
              <span class="wall muted">—</span>
              <span class="bid muted">{tier} · score {score_s}</span>
            </div>
            """
        )

    iters_pct = (iters_used / iters_max * 100) if iters_max else 0
    dollars_pct = (dollars_used / dollars_max * 100) if dollars_max else 0
    wall_pct = (wall_used / wall_max * 100) if wall_max else 0

    queue_rows_html_parts: list[str] = []
    for rank, b in enumerate(queued[:8], start=1):
        b_id = b.get("id", "")
        b_id_esc = html.escape(b_id)
        b_id_js = b_id.replace("'", "\\'")
        b_tier = html.escape(b.get("tier", "?"))
        b_title = html.escape(b.get("title", "?"))
        b_href = f"brainstorm/{b_id_esc}.html"
        queue_rows_html_parts.append(
            f"<a href='{b_href}' target='_blank' rel='noopener' class='qrow-link' data-bs-id='{b_id_esc}'>"
            f"<div class='qrow'><span class='qrank'>{rank}.</span> <b>{b_tier}</b> · {b_title}"
            f"<span class='bs-actions'>"
            f"<button class='bs-act' onclick=\"bsOp(event,'{b_id_js}','boost')\" title='boost (move up)'>&#x2191;</button>"
            f"<button class='bs-act' onclick=\"bsOp(event,'{b_id_js}','demote')\" title='demote (move down)'>&#x2193;</button>"
            f"<button class='bs-act bs-act-danger' onclick=\"bsOp(event,'{b_id_js}','reject')\" title='reject (remove from queue)'>&#x2715;</button>"
            f"</span>"
            f"<span class='qrow-arrow'>→</span></div>"
            f"</a>"
        )
    queue_rows_html = "".join(queue_rows_html_parts) or "<div class='muted'>queue is empty</div>"

    goal_metric = _read_goal_metric()
    metric_name = _read_primary_metric_name(results)
    baseline_metric = _read_baseline_metric(results)
    champion_run_id = _read_champion_run_id()
    scatter_chart = _build_mockup_a_chart(
        iters, baseline_metric, goal_metric, metric_name, champion_run_id
    )
    iter_table = _build_iter_table(iters, baseline_metric, champion_run_id, metric_name, running)
    project_root = ROOT / "projects" / PROJECT
    perf_section = _build_performance_section(project_root)

    _stage_icon = {
        "idle": "◯",
        "planning": "✎",
        "building": "⚒",
        "running": "▶",
        "committing": "⤓",
    }
    _stage_label = {
        "idle": "idle / between iters",
        "planning": "planning (brainstorming + ranking)",
        "building": "building (engineering features)",
        "running": "running (executing harness)",
        "committing": "committing changes",
    }

    cap_s = _read_per_session_wall_cap()

    if running:
        elapsed_s = running.get("elapsed_s") or 0
        iter_n_val = running.get("iter") or current_iter_n or "?"
        iter_label = f"iter {iter_n_val}/{iters_max}"
        elapsed_str = fmt_dur(elapsed_s)
        cap_min = cap_s // 60
        wall_str = f'<span data-seconds="{elapsed_s}">{html.escape(elapsed_str)}</span>'
        cap_suffix = f" / {cap_min}m"
        api_spend_str = html.escape(fmt_money(iter_api_spend))
    else:
        last_iter = iters[-1] if iters else None
        if last_iter:
            _last_exec = last_iter.get("executor") or {}
            _last_plan = last_iter.get("planner") or {}
            last_family = html.escape(
                _extract_family_from_wrote_files(_last_exec.get("wrote_files")) or "?"
            )
            last_n = last_iter.get("iteration_id", "?")
            iter_label = f"last: iter {last_n} · {last_family}"
            last_wall = float(_last_plan.get("wall_seconds") or 0.0) + float(
                _last_exec.get("wall_seconds") or 0.0
            )
            ts_end = last_iter.get("ts_end")
            ago_str = ""
            if ts_end:
                try:
                    ended = dt.datetime.fromisoformat(ts_end)
                    age_s = int((dt.datetime.now() - ended).total_seconds())
                    ago_str = f" · ended {fmt_dur(age_s)} ago"
                except Exception:
                    pass
            wall_str = f"last iter: {html.escape(fmt_dur(last_wall))}{html.escape(ago_str)}"
            cap_suffix = ""
        else:
            iter_label = "no iters yet"
            wall_str = "—"
            cap_suffix = ""
        api_spend_str = html.escape(fmt_money(dollars_used))

    icon = html.escape(_stage_icon.get(stage, "◯"))
    label = html.escape(_stage_label.get(stage, stage))

    card_rows: list[str] = []
    card_rows.append(
        f'<div class="card-row"><span class="key">wall</span>'
        f'<span class="val">{wall_str}{html.escape(cap_suffix)}</span></div>'
    )
    card_rows.append(
        f'<div class="card-row"><span class="key">api spend</span>'
        f'<span class="val">{api_spend_str}</span></div>'
    )

    if stage in ("building", "running", "planning") and queued:
        top_idea = queued[0]
        top_title = html.escape((top_idea.get("title") or "")[:60])
        card_rows.append(
            f'<div class="card-row"><span class="key">working on</span>'
            f'<span class="val">{top_title}</span></div>'
        )

    if last_action:
        card_rows.append(
            f'<div class="card-row"><span class="key">last action</span>'
            f'<span class="val mono">{html.escape(last_action)}</span></div>'
        )

    if tools:
        sorted_tools = sorted(tools.items(), key=lambda kv: kv[1], reverse=True)[:4]
        tools_str = " · ".join(f"{k}: {v}" for k, v in sorted_tools)
        card_rows.append(
            f'<div class="card-row"><span class="key">tools</span>'
            f'<span class="val mono">{html.escape(tools_str)}</span></div>'
        )

    if stage == "idle" and queued:
        top = queued[0]
        top_title = html.escape((top.get("title") or "")[:60])
        top_tier = html.escape(top.get("tier", "?"))
        top_score = top.get("tier_score")
        score_str = f"{float(top_score):.2f}" if top_score is not None else "—"
        card_rows.append(
            f'<div class="card-row"><span class="key">next idea</span>'
            f'<span class="val">{top_title} '
            f'<span class="muted">({top_tier} · score {html.escape(score_str)})</span></span></div>'
        )

    progress_bar = ""
    if running and cap_s:
        elapsed_s_for_bar = running.get("elapsed_s") or 0
        pct = min((elapsed_s_for_bar / cap_s) * 100, 100)
        color = "ok" if pct < 60 else ("warn" if pct < 85 else "bad")
        progress_bar = (
            f'<div class="track"><div class="fill {color}" style="width:{pct:.1f}%"></div></div>'
        )

    live_dot = '<span class="live-dot"></span> ' if running else ""
    banner_html = f"""
<div class="session-card stage-{stage}">
  <div class="card-header">
    <span class="stage-icon">{icon}</span>
    <span class="stage-label">{live_dot}{label}</span>
    <span class="meta">{html.escape(iter_label)}</span>
  </div>
  <div class="card-body">
    {"".join(card_rows)}
  </div>
  {progress_bar}
</div>"""

    now = dt.datetime.now().strftime("%H:%M:%S")
    is_running = bool(running)
    status_line = (
        f"<span class='pill phase-{running['phase']}'>running iter {current_iter_n} · {running['phase']}</span>"
        if is_running
        else "<span class='pill done'>idle / between iterations</span>"
    )

    return f"""
  <h1>autoloop · {PROJECT}</h1>
  <div class="meta">budget snapshot updated {html.escape(str(updated))} · rendered {now} · {status_line}</div>

  {banner_html}

  <div class="bars">
    <div class="bar-card">
      <h3>iterations</h3>
      <div class="v">{iters_used} <span class="max">/ {iters_max}</span></div>
      <div class="bar-track"><div class="bar-fill accent" style="width:{iters_pct:.1f}%"></div></div>
    </div>
    <div class="bar-card">
      <h3>API spend</h3>
      <div class="v">${dollars_used:,.2f} <span class="max">/ ${dollars_max:,.0f}</span></div>
      <div class="bar-track"><div class="bar-fill {("warn" if dollars_pct > 75 else "ok")}" style="width:{dollars_pct:.1f}%"></div></div>
    </div>
    <div class="bar-card">
      <h3>wall time</h3>
      <div class="v">{fmt_dur(wall_used)} <span class="max">/ {fmt_dur(wall_max)}</span></div>
      <div class="bar-track"><div class="bar-fill {("warn" if wall_pct > 75 else "ok")}" style="width:{wall_pct:.1f}%"></div></div>
    </div>
  </div>

  <div class="scatter-section">
    <div class="scatter-header">
      <h2>autoloop progress — {html.escape(metric_name)}</h2>
      <div class="legend">
        <span class="legend-item"><span class="legend-dot comparison"></span> iter</span>
        <span class="legend-item"><span class="legend-dot champion"></span> champion</span>
        <span class="legend-item" style="color:#dc2626;">&#xd7; planner crashed</span>
        <span class="legend-item"><span class="legend-line baseline"></span> baseline</span>
        <span class="legend-item"><span class="legend-line goal"></span> goal</span>
        <span class="legend-item" style="color:#6366f1;font-weight:700;">&#x2501; best-so-far</span>
      </div>
    </div>
    {scatter_chart}
    {iter_table}
  </div>

  <div class="tasks">
    <div class="task" style="border-bottom:2px solid var(--ink); color:var(--muted); font-size:11px; text-transform:uppercase;">
      <span class="status"></span>
      <span class="iter-n">iter</span>
      <span class="family">feature family</span>
      <span class="run-id">run_id</span>
      <span class="metric">{html.escape(metric_name[:8])}</span>
      <span class="cost">cli est</span>
      <span class="wall">wall</span>
      <span class="bid">brainstorm</span>
    </div>
    {"".join(rows_html)}
  </div>

  <div class="panel">
    <h2>brainstorm queue ({len(queued)} pending of {len(latest_by_id)} total)</h2>
    <div class="queue">
      {queue_rows_html}
    </div>
  </div>

  {perf_section}

  <div class="footer">
    state files: {AUTOLOOP_DIR.relative_to(ROOT)} · regenerate with <code>python -m libs.autoloop.dashboard.render</code>
  </div>
"""


_STYLE = """
    :root {
      --ink:#222; --paper:#fff; --muted:#888;
      --line:#ddd; --softline:#eee;
      --accent:#3b82f6; --accent-soft:#dbeafe;
      --ok:#16a34a; --ok-soft:#dcfce7;
      --warn:#d97706; --warn-soft:#fef3c7;
      --bad:#dc2626; --bad-soft:#fee2e2;
      --comparison:#6366f1; --champion:#f59e0b; --baseline:#475569; --goal:#16a34a;
      --st-load:#3b82f6; --st-eda:#14b8a6; --st-train:#16a34a;
      --st-predict:#8b5cf6; --st-eval:#f97316; --st-discover:#ec4899; --st-write:#94a3b8;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink); background: var(--paper);
      max-width: 1100px; margin: 24px auto; padding: 0 16px;
      font-size: 14px; line-height: 1.5;
    }
    h1 { margin: 0 0 4px 0; font-size: 22px; }
    .meta { color: var(--muted); font-size: 12px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-family: ui-monospace, monospace; }
    .pill.phase-planner { background: var(--accent-soft); color: var(--accent); }
    .pill.phase-executor { background: var(--warn-soft); color: var(--warn); }
    .pill.done { background: var(--softline); color: var(--muted); }

    .bars { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0 8px 0; }
    .bar-card { border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
    .bar-card h3 { margin:0 0 6px 0; font-size:12px; color:var(--muted); font-weight:500; text-transform:uppercase; letter-spacing:0.05em; }
    .bar-card .v { font-family: ui-monospace, monospace; font-size:18px; }
    .bar-card .v .max { color: var(--muted); font-size:13px; }
    .bar-track { height:6px; background:var(--softline); border-radius:3px; margin-top:6px; overflow:hidden; }
    .bar-fill { height:100%; border-radius:3px; }
    .bar-fill.accent { background: var(--accent); }
    .bar-fill.ok { background: var(--ok); }
    .bar-fill.warn { background: var(--warn); }

    .tasks { margin-top:18px; border-top:1px solid var(--line); }
    .task {
      display:grid;
      grid-template-columns: 28px 60px 1fr 90px 70px 60px 60px 1fr;
      align-items:center; gap:8px;
      padding:8px 4px; border-bottom:1px solid var(--softline);
      font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size:12.5px;
    }
    .task.running { background: var(--accent-soft); }
    .task .status { text-align:center; font-weight:700; }
    .task .status.ok { color: var(--ok); }
    .task .status.warn { color: var(--warn); }
    .task .status.bad { color: var(--bad); }
    .task .status.accent { color: var(--accent); }
    .task .status.muted { color: var(--muted); }
    .task .iter-n { color: var(--muted); }
    .task .muted { color: var(--muted); }
    s { color: var(--muted); }

    .panel { margin-top:24px; }
    .panel h2 { font-size:13px; color:var(--muted); margin:0 0 8px 0; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }
    .queue { font-family: ui-monospace, monospace; font-size:11.5px; }
    .queue .qrow { padding:6px 8px; border-bottom:1px solid var(--softline); }
    .queue .qrow-link:nth-child(even) .qrow { background:#f8fafc; }
    .qrow .qscore { display:inline-block; min-width:50px; color: var(--muted); }
    .footer { margin-top:24px; color:var(--muted); font-size:11px; font-family: ui-monospace, monospace; }

    .scatter-section { margin: 18px 0 12px 0; }
    .scatter-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; flex-wrap: wrap; gap: 6px; }
    .scatter-header h2 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin: 0; }
    .legend { display: flex; gap: 14px; font-size: 11px; color: var(--muted); flex-wrap: wrap; }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; font-family: ui-monospace, monospace; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot-auto { background: var(--accent); opacity: 0.75; }
    .dot-user { background: var(--ok); opacity: 0.85; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .legend-dot.comparison { background: var(--comparison); }
    .legend-dot.champion { background: var(--champion); }
    .legend-line { width: 16px; height: 2px; display: inline-block; }
    .legend-line.baseline { background: var(--baseline); }
    .legend-line.goal { background: var(--goal); }
    .legend-goal .line { width: 14px; height: 0; border-top: 1.5px dashed var(--ok); }

    .scroll-demo { display: flex; margin-top: 8px; border: 1px solid var(--softline); border-radius: 6px; overflow: hidden; }
    .scroll-demo .y-frozen { flex: 0 0 60px; background: #fafafa; border-right: 1px solid var(--softline); }
    .scroll-demo .x-scroll { flex: 1; overflow-x: auto; }
    .scroll-demo .x-scroll svg { display: block; }

    .data-table-wrap { margin-top: 12px; max-height: 320px; overflow-y: auto; border: 1px solid var(--softline); border-radius: 6px; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: ui-monospace, SFMono-Regular, monospace; }
    .data-table thead { position: sticky; top: 0; background: #f8fafc; z-index: 2; }
    .data-table th { text-align: left; padding: 8px 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; font-size: 10px; border-bottom: 1px solid var(--line); }
    .data-table th.num, .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .data-table td { padding: 6px 10px; border-bottom: 1px solid var(--softline); color: var(--ink); }
    .data-table tr:last-child td { border-bottom: none; }
    .data-table tr.champion-row { background: #fef3c7; }
    .data-table tr.champion-row td:first-child::before { content: "★ "; color: #f59e0b; }
    .data-table tr.running-row { background: var(--accent-soft); font-style: italic; }
    .data-table tr.crashed-row td { color: var(--bad); }
    .data-table td.up { color: var(--ok); }
    .data-table td.down { color: var(--bad); }
    .data-table td.muted { color: var(--muted); }
    .data-table .status-ok { color: var(--ok); font-weight: 600; }
    .data-table .status-bad { color: var(--bad); font-weight: 600; }
    .data-table .status-running { color: var(--accent); font-weight: 600; }

    .session-card {
      background: var(--paper); border: 1px solid var(--line);
      border-left: 4px solid var(--muted);
      border-radius: 0 6px 6px 0;
      padding: 14px 18px; margin: 14px 0 10px 0;
      transition: border-color 0.3s;
    }
    .session-card.stage-idle      { border-left-color: var(--muted); }
    .session-card.stage-planning  { border-left-color: var(--accent); background: var(--accent-soft); }
    .session-card.stage-building  { border-left-color: var(--warn); background: var(--warn-soft); }
    .session-card.stage-running   { border-left-color: var(--ok); background: var(--ok-soft); }
    .session-card.stage-committing { border-left-color: var(--accent); }

    .session-card .card-header {
      display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;
    }
    .session-card .stage-icon { font-size: 22px; line-height: 1; }
    .session-card .stage-label { font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600; }
    .session-card .meta { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); margin-left: auto; }

    .session-card .card-body { display: flex; flex-direction: column; gap: 4px; }
    .session-card .card-row { display: flex; gap: 12px; font-size: 13px; }
    .session-card .card-row .key { color: var(--muted); min-width: 90px; font-family: ui-monospace, monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
    .session-card .card-row .val { font-family: ui-monospace, monospace; }
    .session-card .card-row .val.mono { font-size: 12px; color: var(--ink); }
    .session-card .muted { color: var(--muted); }

    .session-card .track { height: 6px; background: rgba(255,255,255,0.6); border-radius: 3px; overflow: hidden; margin-top: 10px; }
    .session-card .fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    .session-card .fill.ok { background: var(--ok); }
    .session-card .fill.warn { background: var(--warn); }
    .session-card .fill.bad { background: var(--bad); }

    .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--ok); animation: pulse 1.4s infinite; vertical-align: middle; margin-right: 4px; }

    @keyframes pulse {
      0%,100% { opacity: 1; }
      50% { opacity: 0.45; }
    }

    .perf-section { margin-top: 32px; border-top: 2px solid var(--softline); padding-top: 12px; }
    .perf-section > h2 { font-size: 16px; margin: 0 0 4px 0; }
    .perf-section > .blurb { color: var(--muted); font-size: 12px; margin: 0 0 20px 0; font-family: ui-monospace, monospace; }
    .perf-panel { margin-bottom: 28px; border: 1px solid var(--softline); border-radius: 8px; padding: 16px 20px; }
    .perf-panel h3 { font-size: 13px; margin: 0 0 8px 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }

    .stage-legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin:8px 0 12px; }
    .stage-legend span { display:inline-flex; align-items:center; gap:6px; }
    .scope-legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin:8px 0 12px; }
    .scope-legend span { display:inline-flex; align-items:center; gap:6px; }
    .swatch { width:10px; height:10px; border-radius:2px; }
    .sw-load { background:var(--st-load); }
    .sw-eda { background:var(--st-eda); }
    .sw-train { background:var(--st-train); }
    .sw-predict { background:var(--st-predict); }
    .sw-eval { background:var(--st-eval); }
    .sw-discover { background:var(--st-discover); }
    .sw-write { background:var(--st-write); }

    .alert { background:var(--bad-soft); border-left:4px solid var(--bad); padding:14px 18px; border-radius:4px; margin:16px 0; font-size:13px; }
    .alert strong { color:var(--bad); }
    .alert ul { margin:6px 0 0; padding-left:18px; font-family:ui-monospace,monospace; font-size:12px; }
    .alert li { margin:3px 0; }
    .reco { background:var(--ok-soft); border-left:4px solid var(--ok); padding:14px 18px; border-radius:4px; margin:16px 0; font-size:13px; }

    .qrow-link { display:block; color:inherit; text-decoration:none; }
    .qrow-link:hover .qrow { background-color:var(--accent-soft); }
    .qrow-arrow { color:var(--accent); margin-left:6px; font-size:12px; }

    .bs-actions { display:inline-flex; gap:4px; margin-left:auto; }
    .bs-act { border:none; padding:2px 6px; font-size:11px; background:transparent; color:var(--muted); cursor:pointer; border-radius:3px; line-height:1.4; }
    .bs-act:hover { background:var(--accent-soft); color:var(--accent); }
    .bs-act-danger:hover { background:var(--bad-soft); color:var(--bad); }

    .qrank { display:inline-block; min-width:24px; color:var(--muted); font-family:ui-monospace,monospace; }

    @keyframes bsFlash {
      0%   { background-color: #fef3c7; }
      100% { background-color: transparent; }
    }
    .qrow.just-moved { animation: bsFlash 1.2s ease-out; }
"""


_DETAIL_STYLE = """
    :root {
      --ink:#222; --paper:#fff; --muted:#94a3b8;
      --line:#ddd; --softline:#f1f5f9;
      --accent:#6366f1; --accent-soft:#eef2ff;
      --ok:#16a34a; --ok-soft:#dcfce7;
      --warn:#d97706; --warn-soft:#fef3c7;
      --bad:#dc2626; --bad-soft:#fee2e2;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink); background: var(--paper);
      max-width: 720px; margin: 32px auto; padding: 0 20px;
      font-size: 15px; line-height: 1.6;
    }
    a { color: var(--accent); }
    .back-link { font-size: 12px; color: var(--muted); text-decoration: none; display: inline-block; margin-bottom: 20px; }
    .back-link:hover { color: var(--accent); }
    h1 { font-size: 22px; margin: 0 0 8px 0; }
    .title-meta { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
    .tier-badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; }
    .tier-T0 { background: #eef2ff; color: #4338ca; }
    .tier-T1 { background: #f0fdfa; color: #0d9488; }
    .tier-T2 { background: var(--softline); color: var(--muted); }
    .status-badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .status-queued  { background: var(--softline); color: var(--muted); }
    .status-picked  { background: var(--warn-soft); color: var(--warn); }
    .status-shipped { background: var(--ok-soft); color: var(--ok); }
    .status-blocked { background: var(--bad-soft); color: var(--bad); }
    .box { background: var(--softline); border-radius: 6px; padding: 14px 18px; margin: 16px 0; font-size: 14px; line-height: 1.65; }
    .box.accent { border-left: 4px solid var(--accent); background: var(--accent-soft); }
    section h2 { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin: 24px 0 6px 0; }
    .detail-table { border-collapse: collapse; font-size: 13px; font-family: ui-monospace, monospace; width: 100%; }
    .detail-table td { padding: 5px 10px; vertical-align: top; }
    .detail-table td:first-child { color: var(--muted); white-space: nowrap; padding-right: 18px; }
    .detail-table tr + tr td { border-top: 1px solid var(--softline); }
    .muted { color: var(--muted); }
    .num { font-variant-numeric: tabular-nums; }
    .iter-table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: ui-monospace, monospace; margin-top: 6px; }
    .iter-table th { text-align: left; padding: 6px 10px; font-size: 11px; color: var(--muted); text-transform: uppercase; border-bottom: 1px solid var(--line); }
    .iter-table td { padding: 5px 10px; border-bottom: 1px solid var(--softline); }
    details { margin-top: 24px; }
    details summary { cursor: pointer; font-size: 12px; color: var(--muted); user-select: none; }
    details summary:hover { color: var(--ink); }
    .hash-table { font-family: ui-monospace, monospace; font-size: 12px; margin-top: 10px; border-collapse: collapse; }
    .hash-table td { padding: 4px 10px; vertical-align: top; }
    .hash-table td:first-child { color: var(--muted); white-space: nowrap; padding-right: 14px; }
    .hash-table td:last-child { word-break: break-all; }
"""


def _build_brainstorm_detail(item: dict, project: str, project_root: Path) -> str:
    dash_project = _OUT_DIR / f"run-dashboard-{project}.html"
    back_href = (
        f"../run-dashboard-{project}.html" if dash_project.exists() else "../run-dashboard.html"
    )

    item_id = html.escape(item.get("id", ""))
    title = html.escape(item.get("title", "(untitled)"))
    ts = html.escape(item.get("ts", ""))
    tier_raw = item.get("tier", "")
    tier = html.escape(tier_raw)
    tier_cls = tier_raw if tier_raw in ("T0", "T1", "T2") else "T2"
    status_raw = item.get("status", "unknown")
    status = html.escape(status_raw)
    status_cls = (
        status_raw if status_raw in ("queued", "picked", "shipped", "blocked") else "queued"
    )

    hypothesis = html.escape(item.get("hypothesis", ""))
    tier_rationale = html.escape(item.get("tier_rationale", ""))

    source = item.get("source") or {}
    source_kind = html.escape(str(source.get("kind", "—")))
    source_ref = html.escape(str(source.get("ref", "—")))
    finding_id_val = source.get("finding_id")
    finding_id = html.escape(str(finding_id_val)) if finding_id_val else ""
    family_hint = html.escape(str(item.get("feature_family_name_hint") or "—"))
    row_grain = html.escape(str(item.get("row_grain") or "—"))

    cost_raw = item.get("estimated_cost_dollars")
    cost_str = f"${float(cost_raw):.2f}" if cost_raw is not None else "—"
    impl_raw = item.get("estimated_implement_minutes")
    impl_str = f"{impl_raw}m" if impl_raw is not None else "—"
    score_raw = item.get("tier_score")
    score_str = f"{float(score_raw):.2f}" if score_raw is not None else "—"

    finding_row = f"<tr><td>finding id</td><td>{finding_id}</td></tr>" if finding_id else ""

    source_table = f"""
<table class="detail-table">
  <tr><td>source kind</td><td>{source_kind}</td></tr>
  <tr><td>source ref</td><td>{source_ref}</td></tr>
  {finding_row}
  <tr><td>feature family hint</td><td>{family_hint}</td></tr>
  <tr><td>row grain</td><td>{row_grain}</td></tr>
</table>"""

    estimates_table = f"""
<table class="detail-table">
  <tr><td>cost</td><td class="num">{cost_str}</td></tr>
  <tr><td>implement time</td><td class="num">{impl_str}</td></tr>
  <tr><td>tier score</td><td class="num">{score_str}</td></tr>
</table>"""

    iter_history = item.get("iteration_history") or []
    if iter_history:
        iter_rows_parts: list[str] = []
        for entry in iter_history:
            if isinstance(entry, dict):
                iter_rows_parts.append(
                    f"<tr>"
                    f"<td>{html.escape(str(entry.get('iteration_id', '—')))}</td>"
                    f"<td>{html.escape(str(entry.get('run_id', '—')))}</td>"
                    f"<td>{html.escape(str(entry.get('status', '—')))}</td>"
                    f"<td>{html.escape(str(entry.get('note', '')))}</td>"
                    f"</tr>"
                )
            else:
                iter_rows_parts.append(
                    f"<tr><td>{html.escape(str(entry))}</td><td>—</td><td>—</td><td></td></tr>"
                )
        iter_section = f"""
<table class="iter-table">
  <thead><tr><th>iteration_id</th><th>run_id</th><th>status</th><th>note</th></tr></thead>
  <tbody>{"".join(iter_rows_parts)}</tbody>
</table>"""
    else:
        iter_section = '<p class="muted">No iterations yet.</p>'

    content_hash = html.escape(item.get("content_hash") or "")
    embedding_hash_val = item.get("embedding_hash")
    embedding_uri_val = item.get("embedding_uri")
    hash_rows = f"<tr><td>content_hash</td><td>{content_hash}</td></tr>"
    if embedding_hash_val:
        hash_rows += (
            f"<tr><td>embedding_hash</td><td>{html.escape(str(embedding_hash_val))}</td></tr>"
        )
    if embedding_uri_val:
        hash_rows += (
            f"<tr><td>embedding_uri</td><td>{html.escape(str(embedding_uri_val))}</td></tr>"
        )

    hashes_block = f"""
<details>
  <summary>content/embedding hashes</summary>
  <table class="hash-table">
    {hash_rows}
  </table>
</details>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} — brainstorm item</title>
  <style>
{_DETAIL_STYLE}
  </style>
</head>
<body>
  <a class="back-link" href="{back_href}">← back to dashboard</a>
  <h1>{title}</h1>
  <div class="title-meta">
    <span>{item_id}</span>
    <span>{ts}</span>
    <span class="tier-badge tier-{html.escape(tier_cls)}">{tier}</span>
    <span class="status-badge status-{html.escape(status_cls)}">{status}</span>
  </div>

  <section>
    <h2>Hypothesis</h2>
    <div class="box">{hypothesis}</div>
  </section>

  <section>
    <h2>Tier rationale</h2>
    <div class="box accent">{tier_rationale}</div>
  </section>

  <section>
    <h2>Source</h2>
    {source_table}
  </section>

  <section>
    <h2>Estimates</h2>
    {estimates_table}
  </section>

  <section>
    <h2>Iteration history</h2>
    {iter_section}
  </section>

  {hashes_block}
</body>
</html>
"""


def _write_brainstorm_details(
    latest_by_id: dict[str, dict],
    project: str,
    project_root: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item_id, item in latest_by_id.items():
        page_html = _build_brainstorm_detail(item, project, project_root)
        safe_id = item_id.replace("/", "_").replace("\\", "_")
        dest = out_dir / f"{safe_id}.html"
        dest.write_text(page_html, encoding="utf-8")


_BS_OP_JS = """
<script>
async function bsOp(event, id, op) {
  event.preventDefault();
  event.stopPropagation();
  if (op === 'reject' && !confirm('Reject "' + id + '"? This appends a rejected-status row; you can restore via the file.')) return;
  const btn = event.currentTarget;
  btn.disabled = true;
  btn.style.opacity = '0.5';
  try {
    const r = await fetch('/brainstorm/' + encodeURIComponent(id) + '/' + op, {method: 'POST'});
    if (!r.ok) {
      console.error('brainstorm op failed:', r.status, await r.text());
      return;
    }
    if (typeof refreshFragment === 'function') await refreshFragment();
    if (op !== 'reject') {
      const anchor = document.querySelector('[data-bs-id="' + CSS.escape(id) + '"]');
      const moved = anchor ? anchor.querySelector('.qrow') : null;
      if (moved) {
        moved.classList.remove('just-moved');
        void moved.offsetWidth;
        moved.classList.add('just-moved');
      }
    }
  } catch (e) {
    console.error('brainstorm op error:', e);
  } finally {
    btn.disabled = false;
    btn.style.opacity = '';
  }
}
</script>
"""


def _render_shell(content_html: str = "", include_meta_refresh: bool = True) -> str:
    meta_refresh = '<meta http-equiv="refresh" content="15">' if include_meta_refresh else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  {meta_refresh}
  <title>Autoloop — {PROJECT} — run dashboard</title>
  <style>
{_STYLE}
  </style>
{_BS_OP_JS}
</head>
<body>
  <div id="content">{content_html}</div>
</body>
</html>
"""


def render() -> str:
    return _render_shell(_render_body(), include_meta_refresh=True)


def render_once() -> None:
    try:
        text = render()
    except Exception as e:
        text = (
            "<!doctype html><meta http-equiv='refresh' content='5'>"
            f"<pre style='color:#dc2626;font-family:monospace;padding:24px;'>"
            f"render error at {dt.datetime.now()}: {html.escape(str(e))}</pre>"
        )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, OUT)
    canonical_tmp = OUT_CANONICAL.with_suffix(".tmp")
    canonical_tmp.write_text(text)
    os.replace(canonical_tmp, OUT_CANONICAL)

    try:
        brainstorm_rows = read_jsonl(BRAINSTORM)
        latest_by_id = latest_brainstorm_by_id(brainstorm_rows)
        detail_dir = _OUT_DIR / "brainstorm"
        project_root = ROOT / "projects" / PROJECT
        _write_brainstorm_details(latest_by_id, PROJECT, project_root, detail_dir)
    except Exception:
        pass

    print(
        f"[{dt.datetime.now():%H:%M:%S}] rendered {OUT.name} + {OUT_CANONICAL.name} ({len(text):,} bytes)",
        flush=True,
    )


def main() -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Render the autoloop run dashboard.")
    parser.add_argument(
        "--project",
        default=None,
        help="Project name (overrides AUTOLOOP_PROJECT env var).",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Re-render on a timer until interrupted."
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Watch interval in seconds (default 10)."
    )
    args = parser.parse_args()

    if args.project:
        global PROJECT, AUTOLOOP_DIR, RUNS_DIR, OUT, OUT_CANONICAL, CFG, BUDGET, ITERS, BRAINSTORM, RESULTS  # noqa: PLW0603
        PROJECT = args.project
        AUTOLOOP_DIR = ROOT / "projects" / PROJECT / "autoloop"
        RUNS_DIR = ROOT / "projects" / PROJECT / "runs"
        OUT = _OUT_DIR / f"run-dashboard-{PROJECT}.html"
        OUT_CANONICAL = _OUT_DIR / "run-dashboard.html"
        CFG = AUTOLOOP_DIR / "config.yaml"
        BUDGET = AUTOLOOP_DIR / "budget.json"
        ITERS = AUTOLOOP_DIR / "iterations.jsonl"
        BRAINSTORM = AUTOLOOP_DIR / "brainstorm.jsonl"
        RESULTS = RUNS_DIR / "results.jsonl"

    render_once()
    if not args.watch:
        return 0

    print(f"[watch] re-rendering every {args.interval}s — Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(args.interval)
            render_once()
    except KeyboardInterrupt:
        print("\n[watch] stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
