"""Regressions for the audit fixes: retry bounds, projection short-circuit,
snapshot slimming, relevance gating, prompt budget and export-mode stats."""
import gzip
import json
import sqlite3

import pytest

from knowledge_graph import local_llm as llm
from knowledge_graph.capture import MAX_ATTEMPTS, CaptureArchive
from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore


def record(text="[00:01] Oscar: I will send the proposal tomorrow."):
    return {"id": "c1", "title": "Synthetic capture", "startTime": "2026-10-13T09:00:00Z",
            "endTime": "2026-10-13T10:00:00Z", "transcript": text,
            "segments": [{"text": text.split(": ", 1)[-1], "speaker": "Oscar", "start": 1, "end": 5}]}


def test_repeated_failures_dead_letter_instead_of_retrying_forever(tmp_path):
    archive = CaptureArchive(tmp_path)
    key = archive.archive("fieldy", record())
    for _ in range(MAX_ATTEMPTS - 1):
        archive.fail(*key, "invalid_extraction_domain")
        assert archive.get(*key)["status"] == "failed"
    archive.fail(*key, "invalid_extraction_domain")
    row = archive.get(*key)
    assert row["status"] == "dead"
    assert row["attempts"] == MAX_ATTEMPTS
    # Dead captures leave the pending queue entirely.
    assert archive.pending(limit=10) == []
    archive.db.close()


def test_projection_short_circuits_when_checkpoints_match(tmp_path, monkeypatch):
    archive = CaptureArchive(tmp_path)
    key = archive.archive("fieldy", record())
    extraction = {"domain": "professional", "confidence": 0.9, "topics": ["proposal"],
                  "professional_slice": None,
                  "actions": [{"kind": "Task", "title": "Send proposal", "text": "Send proposal",
                               "owner": "Oscar", "due_at": None, "status": "open", "domain": "professional",
                               "evidence": [{"quote": "I will send the proposal tomorrow.",
                                             "start": 1, "end": 5, "speaker": "Oscar"}]}]}
    archive.validate(*key, extraction)
    stores = {scope: KnowledgeGraphStore(data_dir=str(tmp_path / scope)) for scope in ("professional", "personal")}
    assert archive.project(*key, stores=stores) > 0

    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: opened.append(a[0]) or real_connect(*a, **k))
    before = opened[:]
    assert archive.project(*key, stores=stores) == 0
    assert opened == before, "a fully projected capture must not open private connections"
    archive.db.close()


def test_projection_chunks_tile_the_transcript_without_splitting_words(tmp_path):
    text = ("word " * 700).strip()
    offsets = list(CaptureArchive._projection_chunks(text))
    assert offsets[0][0] == 0 and offsets[-1][1] == len(text)
    for (start, end), (next_start, _) in zip(offsets, offsets[1:]):
        assert end == next_start
        assert not text[:end].endswith("wor"), "chunk edges snap to whitespace, not mid-word"
    archive = CaptureArchive(tmp_path)
    key = archive.archive("fieldy", record(text))
    archive.validate(*key, {"domain": "personal", "confidence": 0.8, "topics": [],
                            "professional_slice": None, "actions": []})
    store = KnowledgeGraphStore(data_dir=str(tmp_path / "personal"))
    archive.project(*key, stores={"personal": store})
    chunks = [e for e in store.get_all_entities() if e.label == "Chunk"]
    assert "".join(chunk.description for chunk in chunks) == text
    ids = {chunk.id for chunk in chunks}
    assert ids == {f"conv-fieldy-c1:chunk:{i}" for i in range(len(chunks))}
    archive.db.close()


def test_snapshot_omits_vectors_compresses_and_restores(tmp_path):
    store = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=True)
    store.add_entity(Entity(id="s1", label="Concept", name="Snapshot Concept",
                            embedding=[0.1] * 768, properties={"domain": "professional"}))
    store.add_entity(Entity(id="s2", label="Task", name="Snapshot Task"))
    store.add_relation(Relation(id="r1", source_id="s1", target_id="s2", relation_type="prerequisite_for"))

    path = store.create_snapshot()
    assert path.endswith(".json.gz")
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    assert payload["embeddings"] == "omitted"
    assert all(not entity["embedding"] for entity in payload["entities"].values())
    assert all(entity["properties"].get("embedding_status") == "pending"
               for entity in payload["entities"].values())

    restored = KnowledgeGraphStore(data_dir=str(tmp_path / "fresh"), enable_snapshots=False)
    assert restored.load_snapshot(path) is True
    assert restored.get_entity("s1").name == "Snapshot Concept"
    assert restored.get_entity("s1").embedding is None
    assert restored.get_relation("r1").relation_type == "prerequisite_for"


def test_importance_cannot_lift_an_incidental_match_over_a_relevance_floor(tmp_path):
    store = KnowledgeGraphStore(data_dir=str(tmp_path))
    # The audit's exact shape: one accidental description token, maximum
    # importance, brand-new record — it used to clear a 0.15 floor on priors alone.
    store.add_entity(Entity(id="noise", label="Concept", name="archive notes",
                            description="mentions pricing once", importance_score=1.0))
    store.add_entity(Entity(id="real", label="Concept", name="quarterly pricing strategy review",
                            importance_score=0.2))
    scored = dict((entity.id, score) for score, entity in
                  store.search_entities_scored("quarterly pricing strategy review", semantic=False))
    assert scored["real"] > scored["noise"]
    assert scored["real"] >= 0.15, "a real match still clears the caller's floor"
    assert scored["noise"] < 0.15, "priors may reorder matches, never manufacture relevance"


def test_prompt_budget_keeps_system_and_question_and_trims_history():
    budget = 400
    messages = [
        {"role": "system", "content": "SYSTEM RULES " + "x" * 300},
        {"role": "user", "content": "old question " + "y" * 300},
        {"role": "assistant", "content": "old answer " + "z" * 300},
        {"role": "user", "content": "live question"},
    ]
    fitted = llm.fit_messages(messages, budget=budget)
    total = sum(len(m["content"]) for m in fitted)
    assert total <= budget
    assert fitted[0]["role"] == "system"
    assert fitted[-1]["content"] == "live question"
    assert "SYSTEM RULES" in fitted[0]["content"]


def test_chat_keeps_a_truncated_answer_but_structured_output_stays_strict(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def slot(timeout):
        yield timeout

    monkeypatch.setattr(llm, "local_generation_slot", slot)
    monkeypatch.setattr(llm, "_call", lambda *a: {"done": True, "done_reason": "length",
                                                  "response": "{}", "message": {"content": "Partial answer"}})
    answer = llm.local_chat([{"role": "user", "content": "synthetic"}])
    assert answer.startswith("Partial answer")
    assert "truncated" in answer
    with pytest.raises(ValueError):
        llm.local_generate("synthetic")


def test_export_only_stats_count_approved_records_only(tmp_path, monkeypatch):
    import importlib
    import sys

    monkeypatch.setenv("KG_DATA_DIR", str(tmp_path))
    sys.modules.pop("knowledge_graph.mcp_server", None)
    server = importlib.import_module("knowledge_graph.mcp_server")
    from knowledge_graph.privacy import approve_export

    approved = Entity(id="ok", label="Concept", name="Shareable fact", properties={"domain": "professional"})
    approve_export(approved)
    server.store.add_entity(approved)
    server.store.add_entity(Entity(id="private", label="Decision", name="Private fact",
                                   properties={"domain": "professional"}))
    monkeypatch.setenv("KG_EXPORT_ONLY", "1")
    stats = server.kg_get_stats()
    assert stats["entity_count"] == 1
    assert stats["label_distribution"] == {"Concept": 1}
    assert stats["read_mode"] == "approved_exports"
    sys.modules.pop("knowledge_graph.mcp_server", None)


def test_auto_snapshot_cooldown_prevents_burst_eviction(tmp_path):
    from datetime import timedelta

    store = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=True)
    for i in range(110):
        store.add_entity(Entity(id=f"b{i}", label="Concept", name=f"Burst {i}"))
    assert list(store.snapshot_dir.glob("kg_snapshot_*")) == []
    store._last_snapshot_at -= timedelta(seconds=301)
    store.add_entity(Entity(id="b110", label="Concept", name="Burst trigger"))
    assert len(list(store.snapshot_dir.glob("kg_snapshot_*"))) == 1

