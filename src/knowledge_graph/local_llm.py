"""Loopback-only, serialized generation with complete, schema-checked responses."""
import json
import math
import os

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from knowledge_graph.local_transport import local_generation_slot, local_json, local_stream, loopback_url

DEFAULT_MODEL = "jarvis-recall:qwen3.5-9b"
DEFAULT_EXTRACTION_MODEL = "jarvis-extract:gemma4-12b"
# Documented active-context and output limits. The answer reserve is subtracted
# before a prompt is sent so instructions and top sources cannot be truncated
# silently by the runtime.
NUM_CTX = 16384
NUM_PREDICT = 4096
# Deliberately conservative: English prose runs ~4 chars/token, JSON and
# timestamped transcripts less, so 3.5 leaves headroom instead of overflowing.
CHARS_PER_TOKEN = 3.5


def resolve_local_model(model=None, *, structured=False):
    role = "JARVIS_EXTRACTION_MODEL" if structured else "JARVIS_RECALL_MODEL"
    default = DEFAULT_EXTRACTION_MODEL if structured else DEFAULT_MODEL
    return model or os.getenv("JARVIS_LOCAL_MODEL") or os.getenv(role) or default


def _endpoint():
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in host:
        host = "http://" + host
    return loopback_url(host)


def _call(path, payload, timeout):
    return local_json(path, payload, _endpoint(), timeout)


def prompt_budget_chars():
    """Characters that fit beside the reserved answer tokens."""
    return int((NUM_CTX - NUM_PREDICT) * CHARS_PER_TOKEN)


def _trim(text, keep):
    """Keep both ends of a long block: the head is instruction, the tail is latest."""
    if len(text) <= keep or keep <= 0:
        return text
    head = int(keep * 0.4)
    tail = max(0, keep - head - 24)
    return text[:head] + "\n[… trimmed to fit the local context …]\n" + text[-tail:]


def fit_messages(messages, budget=None):
    """Bound a chat prompt without dropping the system rules or the live question.

    Oldest history is dropped first; a still-oversized block is trimmed from its
    middle with a visible marker. Nothing here leaves the machine.
    """
    budget = prompt_budget_chars() if budget is None else budget
    fitted = [dict(message) for message in messages]
    while len(fitted) > 2 and sum(len(m.get("content") or "") for m in fitted) > budget:
        fitted.pop(1)
    total = sum(len(m.get("content") or "") for m in fitted)
    if total <= budget:
        return fitted
    # Trim the largest trimmable block, never the last message outright.
    order = sorted(range(len(fitted) - 1), key=lambda i: len(fitted[i].get("content") or ""), reverse=True)
    for index in order:
        overflow = total - budget
        content = fitted[index].get("content") or ""
        if overflow <= 0 or not content:
            break
        keep = max(len(content) - overflow, min(len(content), 512))
        fitted[index]["content"] = _trim(content, keep)
        total = sum(len(m.get("content") or "") for m in fitted)
    return fitted


def fit_prompt(prompt, budget=None):
    return _trim(prompt, prompt_budget_chars() if budget is None else budget)


def _keep_alive():
    """Role-appropriate residency; the documented default unloads after two idle minutes."""
    return os.getenv("JARVIS_OLLAMA_KEEP_ALIVE") or "2m"


def _generate(payload, timeout, model, options=None):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Generation timeout must be positive and finite")
    _endpoint()
    with local_generation_slot(timeout) as remaining:
        request = dict(payload, model=resolve_local_model(model, structured="messages" not in payload),
                       stream=False, keep_alive=_keep_alive(), think=os.getenv("JARVIS_LOCAL_THINK", "0") == "1",
                       options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0,
                                **(options or {})})
        result = _call("/api/chat" if "messages" in payload else "/api/generate", request, remaining)
    if not isinstance(result, dict) or result.get("done") is not True:
        raise ValueError("Incomplete local response")
    return result


def local_chat(messages, timeout=300, model=None, options=None):
    payload = {"messages": fit_messages(messages)}
    result = _generate(payload, timeout, model, options)
    truncated = result.get("done_reason") in {"length", "max_tokens"}
    if result.get("done") is not True:
        raise ValueError("Incomplete local response")
    message = result.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Expected nonempty local chat content")
    # A cut-off answer is still evidence-bearing prose; say so instead of
    # discarding it. Structured generation stays strict.
    return content + ("\n\n[Answer truncated at the local output limit; ask a narrower question for the rest.]"
                      if truncated else "")


def local_chat_stream(messages, timeout=300, model=None, options=None, state=None):
    """Yield answer deltas from a loopback chat stream.

    Same loopback/proxy/redirect rules as local_chat, serialized on the shared
    generation lock. A truncated stream still yields its text; when a caller
    passes a ``state`` dict it receives ``truncated`` rather than losing the answer.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Generation timeout must be positive and finite")
    endpoint = _endpoint()
    request = {"messages": fit_messages(messages), "model": resolve_local_model(model, structured=False),
               "stream": True, "keep_alive": _keep_alive(),
               "think": os.getenv("JARVIS_LOCAL_THINK", "0") == "1",
               "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0,
                           **(options or {})}}
    done, truncated = False, False
    with local_generation_slot(timeout) as remaining:
        for event in local_stream("/api/chat", request, endpoint, remaining):
            message = event.get("message") or {}
            piece = message.get("content") if isinstance(message, dict) else None
            if isinstance(piece, str) and piece:
                yield piece
            if event.get("done"):
                done = True
                truncated = event.get("done_reason") in {"length", "max_tokens"}
    if not done:
        raise ValueError("Incomplete local response")
    if state is not None:
        state["truncated"] = truncated


def local_generate(prompt, schema=None, timeout=300, model=None, options=None):
    if schema is not None:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError("Invalid generation schema") from error
    result = _generate({"prompt": fit_prompt(prompt), "format": schema if schema is not None else "json"},
                       timeout, model, options)
    if result.get("done_reason") in {"length", "max_tokens"}:
        raise ValueError("Incomplete local response")
    response = result.get("response")
    if not isinstance(response, str):
        raise ValueError("Expected structured local response")
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("Expected structured object")
    if schema is not None:
        try:
            Draft202012Validator(schema).validate(parsed)
        except ValidationError as error:
            raise ValueError("Local response does not match its schema") from error
    return parsed
