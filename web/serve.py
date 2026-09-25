#!/usr/bin/env python3
"""Serve web/dist on http://localhost:8000 with the headers the web build
needs (cross-origin isolation, for the wasm memory shared with its Web
Workers) and without caching, so a rebuild shows on reload."""
import functools, http.server, os, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
handler = functools.partial(Handler, directory=root)
print(f"serving {root} on http://localhost:{port}")
http.server.ThreadingHTTPServer(("localhost", port), handler).serve_forever()
