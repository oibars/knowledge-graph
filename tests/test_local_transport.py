"""Inference must not route private inputs through proxies or redirects."""
import io
import urllib.error
import urllib.request

import pytest

from knowledge_graph.local_transport import local_json, loopback_url, _NoRedirect


@pytest.mark.parametrize("url", ["https://example.com", "http://localhost.example.com", "http://127.0.0.1@outside.invalid",
                                  "http://localhost/path", "http://localhost?route=cloud", "http://user:pw@localhost"])
def test_remote_or_ambiguous_urls_rejected_before_transport(url, monkeypatch):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("Network transport created"))
    with pytest.raises(ValueError):
        local_json("/api/embed", {"input": "PRIVATE"}, url, 1)


def test_loopback_request_disables_environment_proxy(monkeypatch):
    captured = {}
    class Opener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            return io.BytesIO(b'{"embedding": [1, 0]}')
    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()
    monkeypatch.setattr(urllib.request, "build_opener", build)
    monkeypatch.setenv("HTTP_PROXY", "http://outside.invalid")
    assert local_json("/api/embed", {"input": "PRIVATE"}, "http://127.0.0.1:11434", 1)["embedding"] == [1, 0]
    assert next(h for h in captured["handlers"] if isinstance(h, urllib.request.ProxyHandler)).proxies == {}
    assert any(isinstance(h, _NoRedirect) for h in captured["handlers"])


def test_local_redirect_cannot_replay_private_request_to_cloud():
    request = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=b'PRIVATE')
    with pytest.raises(urllib.error.URLError, match="redirects are forbidden"):
        _NoRedirect().redirect_request(request, None, 307, "redirect", {}, "https://outside.invalid")


def test_loopback_forms_supported():
    assert loopback_url("http://localhost:11434/") == "http://localhost:11434"
    assert loopback_url("http://[::1]:11434") == "http://[::1]:11434"


def test_generation_uses_shared_lock_name_and_releases_it(tmp_path, monkeypatch):
    from knowledge_graph.local_transport import local_generation_slot
    import fcntl
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(tmp_path))
    with local_generation_slot(1) as remaining:
        assert 0 < remaining <= 1
        with (tmp_path / "local-generation.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (tmp_path / "local-generation.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other, fcntl.LOCK_UN)
