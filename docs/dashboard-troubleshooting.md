# Dashboard Troubleshooting

Common issues running the autoloop dashboard, with one-line fixes. Most problems boil down to: stale server, stale browser cache, or wrong python interpreter.

---

### Issue: Buttons appear but clicks do nothing

The running server is serving stale code that predates the POST handlers. Restart it:

```bash
make autoloop-dashboard-restart PROJECT=<name>
```

Then hard-refresh the browser tab (Cmd-Shift-R on Mac).

---

### Issue: POST endpoints return 501 (Unsupported method)

Same cause as above — the running server is too old. Restart per the command above.

---

### Issue: Click → row position doesn't change but score does

This was an algorithm bug that is now fixed. You're on stale server code. Restart the server and hard-refresh the browser.

---

### Issue: `!` slash command in chat doesn't run

`!` must be the **first character on the line** with no leading whitespace. Move the cursor to the start of the line before typing `!`.

---

### Issue: CSS changes not visible after an update

The polling loop replaces only the body content fragment, not the page shell (which carries the CSS). Hard-refresh (Cmd-Shift-R on Mac) to force the browser to re-fetch the shell and bypass its HTTP cache.

---

### Issue: `ModuleNotFoundError: No module named 'yaml'` when starting the server

The server is running under the system Python instead of the project venv. Either activate the venv first:

```bash
source .venv/bin/activate
make autoloop-dashboard PROJECT=<name>
```

Or call the venv Python directly:

```bash
./.venv/bin/python -m libs.autoloop.dashboard.serve --project <name>
```

---

### Issue: Dashboard shows the wrong project's data

The renderer reads `AUTOLOOP_PROJECT` env var and the `--project` CLI flag. Confirm both point to the same project name, then restart the server.

---

### Issue: Brainstorm queue shows "0 of 0" but the file has rows

This is a schema mismatch — the writer used a different schema string than the reader expects (e.g. `brainstorm/v1` vs `brainstorm_item/v1`). It is fixed in the current code. If you see it again, grep both the writer and reader for the schema string and confirm they match exactly.

---

### Issue: Detail pages are stale after clicking

The server rewrites each `brainstorm/{id}.html` detail page on every action. If the old page is still showing, the browser cached it. Reload the detail page directly (the row's link in the queue table).

---

### Issue: Server died and port 8765 is reported in use

An orphaned Python process is still holding the port. Find and kill it:

```bash
lsof -i :8765
kill -9 <PID>
```

Or just run the restart script, which handles all of this automatically:

```bash
./scripts/restart-dashboard.sh <project>
```
