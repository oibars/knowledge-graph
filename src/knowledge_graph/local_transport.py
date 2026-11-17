"""HTTP transport for local inference: loopback only, no proxies or redirects."""
from contextlib import contextmanager
import fcntl
import os
import time
from pathlib import Path
import ipaddress
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def loopback_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except (ValueError, TypeError):
        local = False
    if (not local or parsed.scheme != "http" or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Local inference requires a loopback HTTP endpoint")
    return base_url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Local model redirects are forbidden")


def local_json(path: str, payload: dict, base_url: str, timeout: float) -> dict:
    endpoint = loopback_url(base_url)
    request = urllib.request.Request(endpoint + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    # Environment HTTP_PROXY/HTTPS_PROXY must never receive private model input.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def local_stream(path, payload, base_url, timeout):
    """Yield parsed NDJSON objects from a loopback stream, same rules as local_json."""
    endpoint = loopback_url(base_url)
    request = urllib.request.Request(endpoint + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        for raw in response:
            line = raw.strip()
            if line:
                yield json.loads(line)


# Share the laptop's generation queue with archive extraction and local answers.
@contextmanager
def local_generation_slot(timeout: float):
    directory = Path(os.environ.get("JARVIS_LOCK_DIR", "/Users/oscar/Apps/persona-os/data/locks"))
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with (directory / "local-generation.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Local generation queue timed out")
                time.sleep(.2)
        try:
            yield max(.1, deadline - time.monotonic())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
