"""HTTP server for the autoloop live dashboard.

Owns three endpoints:
  GET /              -- returns the full page shell (HTML + CSS + JS). Called
                        once per new browser tab; subsequent updates are polled.
  GET /fragment      -- returns the rendered body fragment; the client-side JS
                        calls this every ~5 s and swaps `#body-wrap` innerHTML.
                        Click handlers also call `refreshFragment()` for
                        immediate post-action updates.
  POST /brainstorm/{id}/{op}
                     -- user action endpoint (op ∈ {up, down, reject}).
                        Appends a new row to brainstorm.jsonl (event-sourced,
                        never edits existing rows). Renumbers tier_score to
                        evenly-spaced fractions so displayed rank is always
                        authoritative. Rewrites the brainstorm/{id}.html detail
                        page after each op so the next visit is fresh.

Rendering is fully delegated to render.py — this module imports
`_render_body` and `_render_shell` and does not duplicate any HTML generation.

Invariants:
  - brainstorm.jsonl is append-only; no row is ever mutated or deleted.
  - tier_score renumbering after up/down ensures no two items share a score.
  - reject op soft-deletes: the item disappears from the queue UI but remains
    readable in the JSONL log with status="rejected".
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socketserver
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

# Set project env var before importing render so module-level constants resolve correctly.
_parser_pre = argparse.ArgumentParser(add_help=False)
_parser_pre.add_argument("--project", default=None)
_pre_args, _ = _parser_pre.parse_known_args()
if _pre_args.project:
    os.environ["AUTOLOOP_PROJECT"] = _pre_args.project

from libs.autoloop.dashboard.render import (
    _OUT_DIR,
    AUTOLOOP_DIR,
    BRAINSTORM,
    PROJECT,
    ROOT,
    _build_brainstorm_detail,
    _render_body,
    _render_shell,
    latest_brainstorm_by_id,
    read_jsonl,
)

_POLL_JS = """
<script>
const POLL_INTERVAL_MS = 5000;

async function refreshFragment() {
  const content = document.getElementById('content');
  if (!content) return;
  try {
    const r = await fetch('/fragment', { cache: 'no-store' });
    if (!r.ok) return;
    const html = await r.text();
    content.innerHTML = html;
    tickElapsedFromDOM();
  } catch (e) { /* network blip — try again next tick */ }
}

let elapsedBaseSec = null;
let elapsedBaseTs = null;
function tickElapsedFromDOM() {
  const content = document.getElementById('content');
  if (!content) return;
  const el = content.querySelector('.session-banner .elapsed');
  if (!el) { elapsedBaseSec = null; return; }
  const raw = el.dataset.seconds;
  if (!raw) return;
  elapsedBaseSec = parseInt(raw, 10);
  elapsedBaseTs = Date.now();
}
function fmtDur(s) {
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60), rem = s % 60;
  if (m < 60) return m + 'm' + String(rem).padStart(2, '0') + 's';
  const h = Math.floor(m / 60), mr = m % 60;
  return h + 'h' + String(mr).padStart(2, '0') + 'm';
}
setInterval(() => {
  const content = document.getElementById('content');
  if (!content || elapsedBaseSec === null) return;
  const el = content.querySelector('.session-banner .elapsed');
  if (!el) return;
  const extra = Math.floor((Date.now() - elapsedBaseTs) / 1000);
  const cap = el.querySelector('.cap');
  el.firstChild.textContent = fmtDur(elapsedBaseSec + extra);
  if (cap) el.appendChild(cap);
}, 1000);

refreshFragment();
setInterval(refreshFragment, POLL_INTERVAL_MS);
</script>
"""


def _shell_with_js() -> str:
    base = _render_shell("", include_meta_refresh=False)
    return base.replace("</body>", _POLL_JS + "\n</body>")


def _append_brainstorm_row(row: dict, brainstorm_path: Path) -> None:
    with open(brainstorm_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_detail_page(row: dict, bs_id: str) -> None:
    detail_dir = _OUT_DIR / "brainstorm"
    detail_dir.mkdir(parents=True, exist_ok=True)
    project_root = ROOT / "projects" / PROJECT
    page_html = _build_brainstorm_detail(row, PROJECT, project_root)
    safe_id = bs_id.replace("/", "_").replace("\\", "_")
    dest = detail_dir / f"{safe_id}.html"
    dest.write_text(page_html, encoding="utf-8")


def _renumber_and_append(
    by_id: dict, bs_id: str, op: str, brainstorm_path: Path
) -> tuple[int, dict | str]:
    queued = [v for v in by_id.values() if v.get("status", "queued") == "queued"]
    if not queued:
        return 400, "no queued items"

    queued.sort(key=lambda b: float(b.get("tier_score") or 0), reverse=True)

    try:
        idx = next(i for i, it in enumerate(queued) if it.get("id") == bs_id)
    except StopIteration:
        return 400, "unknown id (not in queued list)"

    if op == "boost" and idx > 0:
        queued[idx], queued[idx - 1] = queued[idx - 1], queued[idx]
    elif op == "demote" and idx < len(queued) - 1:
        queued[idx], queued[idx + 1] = queued[idx + 1], queued[idx]

    n = len(queued)
    step = 1.0 / (n + 1)
    new_clicked_score: float | None = None

    for new_idx, item in enumerate(queued):
        new_score = round(1.0 - (new_idx + 1) * step, 4)
        old_score = float(item.get("tier_score") or 0)
        if abs(new_score - old_score) > 1e-6:
            new_row = dict(item)
            new_row["tier_score"] = new_score
            new_row["ts"] = datetime.now(timezone.utc).isoformat()
            existing_rationale = item.get("tier_rationale", "")
            item_id = item.get("id", "")
            if item_id == bs_id:
                new_row["tier_rationale"] = (existing_rationale + f" · {op}ed by user").strip(" ·")
                new_clicked_score = new_score
            else:
                new_row["tier_rationale"] = (existing_rationale + " · renormalized").strip(" ·")
            try:
                _append_brainstorm_row(new_row, brainstorm_path)
            except Exception as exc:
                return 500, f"error appending to brainstorm file: {exc}"
            try:
                _write_detail_page(new_row, item_id)
            except Exception:
                pass

    if new_clicked_score is None:
        clicked = next(it for it in queued if it.get("id") == bs_id)
        cur = float(clicked.get("tier_score") or 0)
        delta = 0.0001 if op == "boost" else -0.0001
        new_row = dict(clicked)
        new_row["tier_score"] = round(cur + delta, 6)
        new_row["ts"] = datetime.now(timezone.utc).isoformat()
        new_row["tier_rationale"] = (
            clicked.get("tier_rationale", "") + f" · {op}ed by user (at boundary)"
        ).strip(" ·")
        try:
            _append_brainstorm_row(new_row, brainstorm_path)
        except Exception as exc:
            return 500, f"error appending to brainstorm file: {exc}"
        try:
            _write_detail_page(new_row, bs_id)
        except Exception:
            pass
        new_clicked_score = new_row["tier_score"]

    return 200, {
        "id": bs_id,
        "op": op,
        "new_tier_score": new_clicked_score,
        "renumbered": True,
    }


def _handle_brainstorm_post(bs_id: str, op: str) -> tuple[int, bytes]:
    brainstorm_path = BRAINSTORM
    try:
        rows = read_jsonl(brainstorm_path)
    except Exception as exc:
        return 500, f"error reading brainstorm file: {exc}".encode()

    by_id = latest_brainstorm_by_id(rows)
    prior = by_id.get(bs_id)
    if prior is None:
        return 400, b"unknown brainstorm id"

    if op in ("boost", "demote"):
        code, payload = _renumber_and_append(by_id, bs_id, op, brainstorm_path)
        if code != 200:
            return code, str(payload).encode()
        return 200, json.dumps(payload).encode()

    if op == "reject":
        new_row = dict(prior)
        new_row["ts"] = datetime.now(timezone.utc).isoformat()
        new_row["status"] = "rejected"
        new_row["tier_rationale"] = str(prior.get("tier_rationale") or "") + " · rejected by user"
        try:
            _append_brainstorm_row(new_row, brainstorm_path)
        except Exception as exc:
            return 500, f"error appending to brainstorm file: {exc}".encode()
        try:
            _write_detail_page(new_row, bs_id)
        except Exception:
            pass
        result = json.dumps(
            {
                "id": bs_id,
                "op": op,
                "new_tier_score": new_row.get("tier_score", prior.get("tier_score")),
                "new_status": "rejected",
            }
        ).encode()
        return 200, result

    return 400, f"unknown op: {op}".encode()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = _shell_with_js().encode()
            self._respond(200, "text/html; charset=utf-8", body)
        elif path == "/fragment":
            try:
                body = _render_body().encode()
            except Exception as exc:
                body = f"<pre style='color:#dc2626'>{exc}</pre>".encode()
            self._respond(200, "text/html; charset=utf-8", body)
        else:
            self._respond(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        parts = path.strip("/").split("/")
        if (
            len(parts) == 3
            and parts[0] == "brainstorm"
            and parts[2] in ("boost", "demote", "reject")
        ):
            bs_id = unquote(parts[1])
            op = parts[2]
            try:
                code, body = _handle_brainstorm_post(bs_id, op)
            except Exception as exc:
                code, body = 500, f"internal error: {exc}".encode()
            content_type = "application/json" if code == 200 else "text/plain"
            self._respond(code, content_type, body)
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-polling autoloop dashboard server.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--project",
        default=None,
        help="Project name (overrides AUTOLOOP_PROJECT env var and default).",
    )
    args = parser.parse_args()

    host = "localhost"
    port: int = args.port
    project: str = PROJECT

    socketserver.TCPServer.allow_reuse_address = True
    server = HTTPServer((host, port), _Handler)

    print(f"Serving at http://{host}:{port}/ for project '{project}' — Ctrl-C to stop", flush=True)
    print(f"State files: {AUTOLOOP_DIR}", flush=True)

    def _shutdown(sig: int, frame: object) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
