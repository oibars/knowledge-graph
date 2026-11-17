"""Behavior contracts for durable capture, export permission and source evidence."""
import importlib
import importlib.util
import json
from datetime import datetime, timezone

import pytest
from knowledge_graph.models import Entity


def feature(name):
    assert importlib.util.find_spec(f"knowledge_graph.{name}") is not None, f"{name} capability missing"
    return importlib.import_module(f"knowledge_graph.{name}")


def record(text="[00:01] Oscar: I will send the proposal tomorrow."):
    return {"id": "C1", "title": "Project planning", "startTime": "2026-10-12T12:00:00Z",
            "endTime": "2026-10-12T13:00:00Z", "transcript": text,
            "segments": [{"start": 1, "end": 5, "speaker": "Oscar", "text": "I will send the proposal tomorrow."}]}


def test_professional_tag_does_not_grant_export():
    p = feature("privacy")
    e = Entity(id="e", label="Event", name="Project", description="private work",
               tags=["professional"], properties={"domain": "professional"})
    assert not p.export_allowed(e)
    p.approve_export(e, purposes=("cloud",))
    assert p.export_allowed(e)
    assert not p.export_allowed(e, purpose="publish")
    e.description = "different private work"
    assert not p.export_allowed(e)


def test_personal_and_raw_captures_cannot_be_export_approved():
    p = feature("privacy")
    for props in [{"domain": "personal"}, {"domain": "mixed"}, {"domain": "professional", "contains_raw": True}]:
        e = Entity(id="e", label="Event", name="Project", description="private", properties=props)
        with pytest.raises(ValueError): p.approve_export(e)
        assert not p.export_allowed(e)


def test_changed_evidence_invalidates_permission():
    p = feature("privacy")
    e = Entity(id="e", label="Decision", name="Choice", description="Use option A",
               properties={"domain": "professional", "evidence": [{"quote": "Use option A"}]})
    p.approve_export(e)
    assert p.export_allowed(e)
    e.properties["evidence"][0]["quote"] = "PRIVATE"
    assert not p.export_allowed(e)


def test_archive_retains_revisions_and_retries_failed_classification(tmp_path):
    c = feature("capture")
    a = c.CaptureArchive(tmp_path)
    key = a.archive("fieldy", record())
    original = a.get(*key)
    assert original["status"] == "pending"
    a.fail(*key, "local_unavailable")
    assert a.get(*key)["status"] == "failed"
    a.archive("fieldy", record("[00:02] Oscar: We chose option B."))
    revised = a.get(*key)
    assert revised["status"] == "pending"
    assert revised["content_hash"] != original["content_hash"]
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 2
    assert "I will send" in json.loads(__import__('pathlib').Path(original["raw_path"]).read_text())["transcript"]


def test_extraction_rejects_invented_quote(tmp_path):
    c = feature("capture")
    a = c.CaptureArchive(tmp_path); key = a.archive("fieldy", record())
    extraction = {"domain": "professional", "confidence": 0.9, "topics": ["project"],
                  "professional_slice": None, "actions": [{"kind": "Task", "title": "Send proposal", "text": "Send proposal",
                      "owner": "Oscar", "due_at": None, "status": "open", "domain": "professional",
                      "evidence": [{"quote": "I promised a million dollars", "start": 1, "end": 5, "speaker": "Oscar"}]}]}
    with pytest.raises(ValueError): a.validate(*key, extraction)
    assert a.get(*key)["status"] == "pending"


def test_source_linked_projection_is_idempotent_and_timezone_aware(tmp_path):
    c = feature("capture")
    from knowledge_graph.services.graph_store import KnowledgeGraphStore
    a = c.CaptureArchive(tmp_path / "archive"); key = a.archive("fieldy", record())
    extraction = {"domain": "professional", "confidence": 0.9, "topics": ["project"], "professional_slice": None,
                  "actions": [{"kind": "Task", "title": "Send proposal", "text": "Send the proposal tomorrow",
                      "owner": "Oscar", "due_at": "2026-10-13", "status": "open", "domain": "professional",
                      "evidence": [{"quote": "I will send the proposal tomorrow.", "start": 1, "end": 5, "speaker": "Oscar"}]}]}
    a.validate(*key, extraction)
    pro = KnowledgeGraphStore(str(tmp_path / "pro"), enable_snapshots=False)
    per = KnowledgeGraphStore(str(tmp_path / "per"), enable_snapshots=False)
    a.project(*key, stores={"professional": pro, "personal": per})
    a.project(*key, stores={"professional": pro, "personal": per})
    tasks = pro.find_by_label("Task")
    assert len(tasks) == 1
    assert tasks[0].properties["owner"] == "Oscar"
    assert tasks[0].properties["evidence"][0]["source_url"] == "fieldy://C1"
    assert tasks[0].created_at == datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    assert pro.get_entity("conv-fieldy-C1").description.endswith("proposal tomorrow.")
    assert len(per.get_all_entities()) == 0


def test_mixed_projection_keeps_raw_local_and_requires_slice_evidence(tmp_path):
    c = feature("capture")
    from knowledge_graph.services.graph_store import KnowledgeGraphStore
    a = c.CaptureArchive(tmp_path / "archive"); key = a.archive("fieldy", record())
    ext = {"domain": "mixed", "confidence": 0.9, "topics": [], "professional_slice": None, "actions": []}
    a.validate(*key, ext)
    pro = KnowledgeGraphStore(str(tmp_path / "pro"), enable_snapshots=False)
    per = KnowledgeGraphStore(str(tmp_path / "per"), enable_snapshots=False)
    a.project(*key, stores={"professional": pro, "personal": per})
    assert not pro.get_all_entities()
    assert per.get_entity("conv-fieldy-C1").properties["contains_raw"] is True


def test_local_client_refuses_remote_model_endpoint(monkeypatch):
    llm = feature("local_llm")
    monkeypatch.setenv("OLLAMA_HOST", "https://example.invalid")
    with pytest.raises(ValueError): llm.local_generate("synthetic", {"type": "object"})
