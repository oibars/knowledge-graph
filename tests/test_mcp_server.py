"""Tests for the MCP server tool layer — the product's headline interface.

Imports the server module fresh against a tmp data dir (the store is built at
module import time from KG_DATA_DIR).
"""

import importlib
import sys

import pytest

from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_DATA_DIR", str(tmp_path))
    sys.modules.pop("knowledge_graph.mcp_server", None)
    mod = importlib.import_module("knowledge_graph.mcp_server")
    yield mod
    sys.modules.pop("knowledge_graph.mcp_server", None)


def test_add_search_get_roundtrip(server):
    eid = server.kg_add_entity(label="Concept", name="Vector search", description="ANN retrieval")
    assert eid.startswith("concept-")

    results = server.kg_search("vector")
    assert any(r["id"] == eid for r in results)
    hit = next(r for r in results if r["id"] == eid)
    assert set(hit) >= {"id", "name", "label", "description", "tags", "importance_score"}

    full = server.kg_get_entity(eid)
    assert full["name"] == "Vector search"
    assert "created_at" in full


def test_add_entity_is_idempotent_by_content(server):
    a = server.kg_add_entity(label="Concept", name="Dedup", description="same content")
    b = server.kg_add_entity(label="Concept", name="Dedup", description="same content")
    assert a == b, "re-storing identical content must not mint a duplicate"


def test_add_relation_rejects_unknown_type(server):
    a = server.kg_add_entity(label="Concept", name="A")
    b = server.kg_add_entity(label="Concept", name="B")
    with pytest.raises(ValueError):
        server.kg_add_relation(source_id=a, target_id=b, relation_type="banana")


def test_bidirectional_inverse_survives_restart(server, tmp_path):
    a = server.kg_add_entity(label="Concept", name="Whole")
    b = server.kg_add_entity(label="Concept", name="Piece")
    server.kg_add_relation(source_id=b, target_id=a, relation_type="part_of", bidirectional=True)

    # A brand-new store instance sees the persisted inverse (was memory-only before)
    fresh = KnowledgeGraphStore(data_dir=str(tmp_path))
    types = {r.relation_type for r in fresh._relations.values()}
    assert "part_of" in types
    assert "contains" in types, "inverse edge must be persisted, not memory-only"


def test_refresh_sees_external_writes(server, tmp_path):
    from knowledge_graph.models import Entity

    # Another process (second store) writes the same DB
    other = KnowledgeGraphStore(data_dir=str(tmp_path))
    other.add_entity(Entity(id="ext-1", label="Concept", name="External write"))

    results = server.kg_search("External write")
    assert any(r["id"] == "ext-1" for r in results), "server must reload on external change"


def test_stats_shape(server):
    server.kg_add_entity(label="Concept", name="S")
    stats = server.kg_get_stats()
    assert stats["entity_count"] >= 1
    assert "label_distribution" in stats


def test_main_entry_point_exists(server):
    assert callable(server.main), "kg-mcp console script requires main()"
