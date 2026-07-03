"""Shared ingest pipeline tests."""

import json

from knowledge_graph.pipeline import SCOPES, conversation_entity, ingest_staging_jsonl
from knowledge_graph.services.graph_store import KnowledgeGraphStore


RECORD = {
    "id": "abc123",
    "title": "Roadmap sync",
    "date": "2026-06-30T10:00:00",
    "topics": ["roadmap", "pricing"],
    "speakers": ["Oscar", "Sam"],
    "text": "Discussed Q3 pricing changes.",
}


def test_conversation_entity_shape_professional():
    e = conversation_entity("fieldy", RECORD, "professional")
    assert e.id == "conv-fieldy-abc123"
    assert e.label == "Event"
    assert e.properties["domain"] == "professional"
    assert "professional" in e.tags and "fieldy" in e.tags
    assert e.importance_score == SCOPES["professional"]["importance"]
    assert e.created_at.year == 2026


def test_conversation_entity_shape_personal():
    e = conversation_entity("plaud", RECORD, "personal")
    assert e.properties["domain"] == "personal"
    assert "personal" in e.tags
    assert e.importance_score == SCOPES["personal"]["importance"]


def test_ingest_staging_jsonl(tmp_path):
    staging = tmp_path / "staging.jsonl"
    other = {**RECORD, "id": "def456", "title": "1:1 with Sam", "text": "Career growth chat."}
    staging.write_text(json.dumps(RECORD) + "\n" + json.dumps(other) + "\n")
    store = KnowledgeGraphStore(data_dir=str(tmp_path / "db"))
    n = ingest_staging_jsonl("fieldy", staging, "professional", store=store)
    assert n == 2
    assert store.get_entity("conv-fieldy-abc123") is not None
    # idempotent re-run: same ids, no duplicates
    ingest_staging_jsonl("fieldy", staging, "professional", store=store)
    assert store.get_stats()["entity_count"] == 2


def test_trend_entity_shape():
    from knowledge_graph.pipeline import trend_entity

    item = {
        "title": "Anthropic ships Fable 5",
        "summary": "New frontier model released.",
        "url": "https://example.com/fable-5",
        "category": "ai",
        "topic": "models",
        "keywords": ["anthropic", "fable"],
        "sourceDomain": "example.com",
        "sourceTier": "tier_1",
        "popularity": 0.5,
        "publishedAt": "2026-07-01T12:00:00.000Z",
    }
    e = trend_entity(item)
    assert e.id.startswith("trend-swiftrecap-")
    assert e.label == "Document"
    assert e.properties["domain"] == "professional"
    assert "trend" in e.tags and "ai" in e.tags
    assert e.importance_score == 0.6  # 0.4 + 0.4*0.5
    assert "models" in e.topics and "anthropic" in e.topics
    assert e.created_at.year == 2026

    # stable id: same url → same entity id
    assert trend_entity(item).id == e.id


def test_trend_entity_survives_sparse_item():
    from knowledge_graph.pipeline import trend_entity

    e = trend_entity({"url": "https://example.com/x", "publishedAt": None})
    assert e.name == "Untitled"
    assert e.importance_score == 0.4


def test_ingest_content_identical_records_dedupe(tmp_path):
    # different staging ids but identical title+text → one entity (content dedup)
    staging = tmp_path / "staging.jsonl"
    staging.write_text(json.dumps(RECORD) + "\n" + json.dumps({**RECORD, "id": "zzz"}) + "\n")
    store = KnowledgeGraphStore(data_dir=str(tmp_path / "db"))
    ingest_staging_jsonl("fieldy", staging, "professional", store=store)
    assert store.get_stats()["entity_count"] == 1
