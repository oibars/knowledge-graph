"""No private prompt may leave loopback; only complete, schema-valid results count."""
from contextlib import contextmanager

import pytest

from knowledge_graph import local_llm as llm


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434?forward=remote", "http://localhost/#secret", "http://", "https://localhost"])
def test_endpoint_rejects_noncanonical_local_urls(monkeypatch, host):
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(ValueError):
        llm._endpoint()


@pytest.fixture
def transport(monkeypatch):
    @contextmanager
    def slot(timeout):
        yield timeout
    monkeypatch.setattr(llm, "local_generation_slot", slot, raising=False)
    responses = []
    monkeypatch.setattr(llm, "_call", lambda *args: responses.pop(0))
    return responses


@pytest.mark.parametrize("response", [{"response": "{}"}, {"done": False, "response": "{}"},
                                     {"done": True, "done_reason": "length", "response": "{}"}])
def test_incomplete_response_rejected(transport, response):
    transport.append(response)
    with pytest.raises(ValueError):
        llm.local_generate("synthetic")


def test_schema_validation_rejects_missing_and_unexpected_fields(transport):
    schema = {"type": "object", "required": ["notes"], "properties": {"notes": {"type": "array", "items": {"type": "string"}}}, "additionalProperties": False}
    for data in ['{}', '{"notes":[4]}', '{"notes":[],"private":"extra"}']:
        transport.append({"done": True, "response": data})
        with pytest.raises(ValueError):
            llm.local_generate("synthetic", schema)
    transport.append({"done": True, "response": '{"notes":["valid"]}'})
    assert llm.local_generate("synthetic", schema) == {"notes": ["valid"]}


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_rejected_before_network(transport, timeout):
    with pytest.raises(ValueError):
        llm.local_generate("synthetic", timeout=timeout)


def test_chat_content_must_be_nonempty_text(transport):
    for content in [None, [], ""]:
        transport.append({"done": True, "message": {"content": content}})
        with pytest.raises(ValueError):
            llm.local_chat([{"role": "user", "content": "synthetic"}])


@pytest.mark.parametrize("reasoning, expected", [(None, False), ("0", False), ("1", True), ("true", False)])
def test_local_generation_thinking_requires_explicit_opt_in(transport, monkeypatch, reasoning, expected):
    if reasoning is None:
        monkeypatch.delenv("JARVIS_LOCAL_THINK", raising=False)
    else:
        monkeypatch.setenv("JARVIS_LOCAL_THINK", reasoning)
    requests = []
    def call(path, payload, timeout):
        requests.append(payload)
        return {"done": True, "response": "{}", "message": {"content": "Answer"}}
    monkeypatch.setattr(llm, "_call", call)
    llm.local_generate("synthetic")
    llm.local_chat([{"role": "user", "content": "synthetic"}])
    assert [request["think"] for request in requests] == [expected, expected]


def test_structured_extraction_and_recall_use_their_configured_models(transport, monkeypatch):
    monkeypatch.delenv('JARVIS_LOCAL_MODEL', raising=False)
    monkeypatch.setenv('JARVIS_EXTRACTION_MODEL', 'extractor-test:latest')
    monkeypatch.setenv('JARVIS_RECALL_MODEL', 'recall-test:latest')
    requests = []
    def call(path, payload, timeout):
        requests.append(payload)
        return {'done': True, 'response': '{}', 'message': {'content': 'Answer'}}
    monkeypatch.setattr(llm, '_call', call)
    llm.local_generate('synthetic')
    llm.local_chat([{'role': 'user', 'content': 'synthetic'}])
    assert [request['model'] for request in requests] == ['extractor-test:latest', 'recall-test:latest']
    monkeypatch.setenv('JARVIS_LOCAL_MODEL', 'global-test:latest')
    llm.local_generate('synthetic')
    llm.local_chat([{'role': 'user', 'content': 'synthetic'}])
    assert [request['model'] for request in requests[-2:]] == ['global-test:latest', 'global-test:latest']
