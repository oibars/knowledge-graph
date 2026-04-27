"""
One-time Reddit OAuth code-flow helper for accounts without a Reddit password
(e.g. Google-SSO-only accounts). Produces a refresh_token you set as
REDDIT_REFRESH_TOKEN — PRAW then auths without username/password.

Usage:
    python scripts/reddit_oauth_setup.py

Prerequisites (already done if your Reddit app was created earlier):
    - https://www.reddit.com/prefs/apps app of type "script"
    - redirect_uri set to http://localhost:8080
    - REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in env

Output:
    Prints `set -gx REDDIT_REFRESH_TOKEN <token>` for you to paste into env.fish.
"""

from __future__ import annotations

import base64
import http.server
import os
import secrets
import socketserver
import sys
import urllib.parse
import urllib.request
import webbrowser
from typing import Optional

REDIRECT_HOST = "localhost"
REDIRECT_PORT = 8080
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}"
SCOPES = "identity history save read"
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "knowledge-graph/1.0 oauth-setup")


class _Handler(http.server.BaseHTTPRequestHandler):
    captured: dict = {}

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _Handler.captured.update({k: v[0] for k, v in params.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            "<html><body><h2>Reddit auth received.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args, **kwargs):
        pass


def _exchange_code(client_id: str, client_secret: str, code: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    data = urllib.parse.urlencode(
        {"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI}
    ).encode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=data,
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json
        payload = json.loads(resp.read())
    if "refresh_token" not in payload:
        raise SystemExit(f"No refresh_token in response: {payload}")
    return payload["refresh_token"]


def main() -> None:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET first.")

    state = secrets.token_urlsafe(16)
    auth_url = "https://www.reddit.com/api/v1/authorize?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "duration": "permanent",
            "scope": SCOPES,
        }
    )

    print(f"Open this URL in a browser logged into your Reddit account:\n  {auth_url}\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    with socketserver.TCPServer((REDIRECT_HOST, REDIRECT_PORT), _Handler) as httpd:
        print(f"Waiting for Reddit redirect on {REDIRECT_URI} ...")
        while "code" not in _Handler.captured and "error" not in _Handler.captured:
            httpd.handle_request()

    if "error" in _Handler.captured:
        sys.exit(f"Reddit returned error: {_Handler.captured['error']}")
    if _Handler.captured.get("state") != state:
        sys.exit("State mismatch — possible CSRF, aborting.")

    code = _Handler.captured["code"]
    print("Exchanging code for refresh_token ...")
    refresh_token = _exchange_code(client_id, client_secret, code)

    print("\n=== SUCCESS ===")
    print("Add this line to ~/.config/fish/env.fish:\n")
    print(f'set -gx REDDIT_REFRESH_TOKEN "{refresh_token}"')
    print("\nThen open a new fish shell (or source the file) and run:")
    print("  kg-ingest-bookmarks --source reddit\n")


if __name__ == "__main__":
    main()
