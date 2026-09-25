"""Stdlib HTTP server: a JSON API plus a single-page chat UI.

    GET  /            chat page
    GET  /api/health  index stats
    POST /api/ask     {"question": str, "k": int?, "mode": "hybrid"|"bm25"|"dense"?, "llm": bool?}
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .generate import answer, claude_available
from .index import MODES, Index

MAX_BODY = 64 * 1024
MAX_QUESTION = 2000

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-engine</title>
<style>
  :root { --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e3de; --accent:#2f5d8a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --card:#1f1f1d; --line:#33332f; --accent:#8db4dc; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 16px 120px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
  .msg { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
  .q { font-weight: 600; }
  .a { white-space: pre-wrap; }
  .meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
  .warn { color: #b0602a; font-size: 13px; margin-top: 6px; }
  details { margin-top: 10px; font-size: 14px; }
  summary { cursor: pointer; color: var(--accent); }
  .src { border-top: 1px solid var(--line); padding: 8px 0; }
  .src b { color: var(--accent); }
  .src p { margin: 4px 0 0; color: var(--muted); white-space: pre-wrap; }
  form { position: fixed; left: 0; right: 0; bottom: 0; background: var(--bg); border-top: 1px solid var(--line); }
  .row { max-width: 760px; margin: 0 auto; padding: 12px 16px; display: flex; gap: 8px; }
  input { flex: 1; min-width: 0; font: inherit; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--fg); }
  button { font: inherit; padding: 10px 16px; border-radius: 8px; border: 0; background: var(--accent); color: var(--bg); cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<main>
  <h1>rag-engine</h1>
  <div class="sub" id="status">Loading index...</div>
  <div id="log" aria-live="polite"></div>
</main>
<form id="f">
  <div class="row">
    <label for="q" style="position:absolute;left:-9999px">Question</label>
    <input id="q" autocomplete="off" placeholder="Ask a question about the indexed documents" required maxlength="2000">
    <button id="b">Ask</button>
  </div>
</form>
<script>
const log = document.getElementById("log"), q = document.getElementById("q"), b = document.getElementById("b");
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
fetch("/api/health").then(r => r.json()).then(h => {
  document.getElementById("status").textContent =
    `${h.chunks} chunks from ${h.sources} documents - answers by ${h.llm ? "Claude" : "extractive fallback (set ANTHROPIC_API_KEY for Claude)"}`;
});
document.getElementById("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const question = q.value.trim(); if (!question) return;
  const card = el("div", "msg"); card.append(el("div", "q", question));
  const body = el("div", "a", "Thinking..."); card.append(body); log.prepend(card);
  q.value = ""; b.disabled = true;
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    body.textContent = data.answer;
    card.append(el("div", "meta", `mode: ${data.mode} - ${data.sources.length} sources`));
    if (data.warning) card.append(el("div", "warn", data.warning));
    const det = el("details"); det.append(el("summary", null, "Sources"));
    for (const s of data.sources) {
      const d = el("div", "src"); d.append(el("b", null, `[${s.n}] `), document.createTextNode(`${s.source} - ${s.heading || s.title}`));
      d.append(el("p", null, s.text)); det.append(d);
    }
    card.append(det);
  } catch (e) { body.textContent = "Error: " + e.message; }
  finally { b.disabled = false; q.focus(); }
});
</script>
</body>
</html>
"""


def make_handler(index: Index, default_k: int = 5):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rag-engine/0.1"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self._json(200, {"chunks": len(index.chunks), "sources": len(index.sources()),
                                 "llm": claude_available()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "invalid Content-Length"})
            if length <= 0 or length > MAX_BODY:
                return self._json(413 if length > MAX_BODY else 400, {"error": "body must be 1..65536 bytes"})
            try:
                req = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._json(400, {"error": "body must be JSON"})
            if not isinstance(req, dict):
                return self._json(400, {"error": "body must be a JSON object"})
            question = req.get("question")
            k = req.get("k", default_k)
            mode = req.get("mode", "hybrid")
            use_llm = req.get("llm", True)
            if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
                return self._json(400, {"error": f"'question' must be a non-empty string up to {MAX_QUESTION} chars"})
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:
                return self._json(400, {"error": "'k' must be an integer between 1 and 20"})
            if mode not in MODES:
                return self._json(400, {"error": f"'mode' must be one of {list(MODES)}"})
            hits = index.search(question, k=k, mode=mode)
            self._json(200, answer(question, hits, use_llm=bool(use_llm)).to_dict())

        def log_message(self, fmt, *args):  # quieter, single-line access log
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def serve(index: Index, host: str = "127.0.0.1", port: int = 8000, k: int = 5) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(index, k))
    print(f"serving {len(index.chunks)} chunks on http://{host}:{httpd.server_port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
