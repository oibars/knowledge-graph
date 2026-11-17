"""Review regressions for canonical captures and evidence-bound extraction."""
import json
import math
from datetime import datetime, timezone

import pytest

from knowledge_graph.capture import CaptureArchive
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.models import Entity


def record(text="[00:01] Oscar: I will send the proposal tomorrow."):
    return {"id": "C1", "title": "Project planning", "startTime": "2026-10-12T12:00:00Z",
            "endTime": "2026-10-12T13:00:00Z", "transcript": text,
            "segments": [{"start": 1, "end": 5, "speaker": "Oscar", "text": "I will send the proposal tomorrow."}]}


def extraction(domain="professional", actions=None):
    return {"domain": domain, "confidence": .9, "topics": ["project"], "professional_slice": None, "actions": actions or []}


def task(**overrides):
    data = {"kind": "Task", "title": "Send proposal", "text": "Send the proposal", "owner": "Oscar",
            "due_at": "2026-10-13", "status": "open", "domain": "professional",
            "evidence": [{"quote": "I will send the proposal tomorrow.", "start": 1, "end": 5, "speaker": "Oscar"}]}
    data.update(overrides)
    return data


@pytest.fixture
def archive(tmp_path):
    a = CaptureArchive(tmp_path / "archive")
    a.archive("fieldy", record())
    return a


@pytest.fixture
def stores(tmp_path):
    return {scope: KnowledgeGraphStore(str(tmp_path / scope), enable_snapshots=False) for scope in ("professional", "personal")}


def test_corrected_extraction_reprojects_same_source_and_removes_old_route(archive, stores):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    archive.validate("fieldy", "C1", extraction("personal"))
    archive.project("fieldy", "C1", stores)
    assert stores["professional"].get_all_entities() == []
    assert stores["personal"].get_entity("conv-fieldy-C1") is not None


def test_rebuild_into_new_store_does_not_trust_previous_projection_checkpoint(archive, stores, tmp_path):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    rebuilt = {"professional": KnowledgeGraphStore(str(tmp_path / "rebuilt"), enable_snapshots=False)}
    assert archive.project("fieldy", "C1", rebuilt) > 0
    assert rebuilt["professional"].get_entity("conv-fieldy-C1")


def test_deleted_managed_projection_is_repaired(archive, stores):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    stores["professional"].delete_entity("conv-fieldy-C1:chunk:0")
    assert archive.project("fieldy", "C1", stores) > 0
    assert stores["professional"].get_entity("conv-fieldy-C1:chunk:0")


def test_end_span_cannot_extend_beyond_quoted_segment(archive):
    action = task()
    action["evidence"][0]["end"] = 99999
    with pytest.raises(ValueError, match="span|Span"):
        archive.validate("fieldy", "C1", extraction(actions=[action]))


@pytest.mark.parametrize("date", ["2040-01-01", "2026-10-13T18:39:00"])
def test_due_date_must_be_supported_by_evidence(archive, date):
    with pytest.raises(ValueError, match="due|Due"):
        archive.validate("fieldy", "C1", extraction(actions=[task(due_at=date)]))


def test_task_cannot_claim_done_based_only_on_future_promise(archive):
    with pytest.raises(ValueError, match="status|state|Status"):
        archive.validate("fieldy", "C1", extraction(actions=[task(status="done")]))


def test_fabricated_speaker_cannot_justify_owner_without_segments(archive):
    raw = record("I will send the proposal tomorrow.")
    raw["segments"] = []
    archive.archive("fieldy", raw)
    with pytest.raises(ValueError, match="Speaker|speaker|Owner|owner"):
        archive.validate("fieldy", "C1", extraction(actions=[task()]))


def test_mixed_professional_action_evidence_must_be_inside_work_slice(archive):
    raw = record("Oscar: We agreed on a launch.\nOscar: I will book my family vacation tomorrow.")
    raw["segments"] = [{"start": 1, "end": 5, "speaker": "Oscar", "text": "We agreed on a launch."},
                       {"start": 6, "end": 10, "speaker": "Oscar", "text": "I will book my family vacation tomorrow."}]
    archive.archive("fieldy", raw)
    ex = extraction("mixed", [task(title="Book vacation", text="Book vacation", evidence=[{
        "quote": "I will book my family vacation tomorrow.", "start": 6, "end": 10, "speaker": "Oscar"}])])
    ex["professional_slice"] = {"title": "Launch plan", "text": "We agreed on a launch.", "evidence": [{
        "quote": "We agreed on a launch.", "start": 1, "end": 5, "speaker": "Oscar"}]}
    with pytest.raises(ValueError, match="slice|Slice"):
        archive.validate("fieldy", "C1", ex)


def test_raw_revision_integrity_checked_before_validation(archive):
    row = archive.get("fieldy", "C1")
    from pathlib import Path
    path = Path(row["raw_path"])
    data = json.loads(path.read_text()); data["transcript"] = "Altered outside archive"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity|hash|revision"):
        archive.validate("fieldy", "C1", extraction())


def test_stale_local_extraction_cannot_bind_to_new_source_revision(archive):
    old_hash = archive.get("fieldy", "C1")["content_hash"]
    raw = record(); raw["title"] = "Revised title"
    archive.archive("fieldy", raw)
    with pytest.raises(ValueError, match="changed|revision|stale"):
        archive.validate("fieldy", "C1", extraction(actions=[task()]), expected_content_hash=old_hash)


@pytest.mark.parametrize("mutation", [{"transcript": ["not text"]}, {"segments": {}}, {"segments": ["not a segment"]}])
def test_malformed_capture_does_not_become_plausible_text(archive, mutation):
    with pytest.raises(ValueError):
        archive.archive("fieldy", {**record(), **mutation})


def test_changed_action_status_reprojects_without_creating_duplicate_action(archive, stores):
    first = extraction(actions=[task()])
    archive.validate("fieldy", "C1", first)
    archive.project("fieldy", "C1", stores)
    before = stores["professional"].find_by_label("Task")[0].id
    second = extraction(actions=[task(text="Send the proposal tomorrow")])
    archive.validate("fieldy", "C1", second)
    archive.project("fieldy", "C1", stores)
    tasks = stores["professional"].find_by_label("Task")
    assert len(tasks) == 1 and tasks[0].id == before
    assert tasks[0].description == "Send the proposal tomorrow"


def test_missing_relation_is_repaired_despite_entity_checkpoint(archive, stores):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    pro = stores["professional"]
    relation_id = "conv-fieldy-C1:chunk:0:part_of:conv-fieldy-C1"
    pro.delete_relation(relation_id)
    assert archive.project("fieldy", "C1", stores) > 0
    assert pro.get_relation(relation_id)


def test_modified_managed_entity_is_restored_from_canonical_source(archive, stores):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    entity = stores["professional"].get_entity("conv-fieldy-C1:chunk:0")
    entity.description = "incorrect external projection"
    stores["professional"].update_entity(entity)
    assert archive.project("fieldy", "C1", stores) > 0
    assert stores["professional"].get_entity(entity.id).description == record()["transcript"]


def test_validated_legacy_quarantine_is_removed_only_after_correct_projection(archive, stores):
    legacy = Entity(id="conv-fieldy-C1", label="Event", name="Legacy raw", description="Old raw",
                    source_app="fieldy", properties={"source_id": "C1", "migration_state": "awaiting_validation", "legacy_origin": "professional"})
    stores["personal"].add_entity(legacy)
    manual = Entity(id="manual", label="Event", name="Manual", source_app="fieldy",
                    properties={"source_id": "C1", "migration_state": "awaiting_validation", "legacy_origin": "professional"})
    stores["personal"].add_entity(manual)
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    assert stores["professional"].get_entity(legacy.id)
    assert not stores["personal"].get_entity(legacy.id)
    assert stores["personal"].get_entity(manual.id)


def test_relative_due_uses_source_occurrence_in_configured_timezone(archive, monkeypatch):
    monkeypatch.setenv("JARVIS_TIMEZONE", "America/New_York")
    raw = record(); raw["startTime"] = "2026-10-13T01:00:00Z"
    archive.archive("fieldy", raw)
    archive.validate("fieldy", "C1", extraction(actions=[task(due_at="2026-10-13")]))
    with pytest.raises(ValueError, match="Due"):
        archive.validate("fieldy", "C1", extraction(actions=[task(due_at="2026-10-14")]))


def test_chunk_retains_source_speaker_and_time_evidence(archive, stores):
    archive.validate("fieldy", "C1", extraction())
    archive.project("fieldy", "C1", stores)
    chunk = stores["professional"].get_entity("conv-fieldy-C1:chunk:0")
    assert chunk.properties["evidence"][0] == {"quote": "I will send the proposal tomorrow.", "speaker": "Oscar", "start": 1, "end": 5,
        "source_id": "C1", "source_app": "fieldy", "source_url": "fieldy://C1"}
