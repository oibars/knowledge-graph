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


def test_default_recall_excludes_reference_and_returns_provenance(server):
    from knowledge_graph.models import Entity
    server.store.add_entity(Entity(id="news", label="Document", name="Pricing news", source_app="swiftrecap"))
    server.store.add_entity(Entity(id="decision", label="Decision", name="Pricing decision", source_app="fieldy",
                                   source_url="fieldy://recording", properties={"domain": "professional", "export_policy": "local_only",
                                   "evidence": [{"source_id": "recording", "quote": "We agreed on pricing"}]}))
    hits = server.kg_search("pricing")
    assert [h["id"] for h in hits] == ["decision"]
    assert hits[0]["source_url"] == "fieldy://recording"
    assert hits[0]["domain"] == "professional"
    assert hits[0]["export_allowed"] is False
    assert hits[0]["properties"]["evidence"]
    assert {h["id"] for h in server.kg_search("pricing", corpus="all")} == {"decision", "news"}


def test_action_recall_requires_evidence_and_filters_owner_status(server):
    from knowledge_graph.models import Entity
    for eid, props in [("grounded", {"owner": "Oscar", "status": "open", "evidence": [{"source_id": "meeting", "quote": "I will send it"}]}),
                       ("unsupported", {"owner": "Oscar", "status": "open"}),
                       ("other", {"owner": "Someone", "status": "open", "evidence": [{"source_id": "meeting", "quote": "I will send it"}]})]:
        server.store.add_entity(Entity(id=eid, label="Task", name="Send proposal", properties=props))
    hits = server.kg_recall_actions("proposal", owner="Oscar", status="open")
    assert [h["id"] for h in hits] == ["grounded"]
    assert hits[0]["properties"]["owner"] == "Oscar"
    assert server.kg_recall_actions(owner="Oscar", status="done") == []


def test_export_only_connector_exposes_only_approved_facts(server, monkeypatch):
    from knowledge_graph.models import Entity
    from knowledge_graph.privacy import approve_export
    for eid in ("safe", "private"):
        e = Entity(id=eid, label="Concept", name=f"Pricing {eid}", description="Selected fact" if eid == "safe" else "PRIVATE_SENTINEL",
                   properties={"domain": "professional"})
        if eid == "safe": approve_export(e)
        server.store.add_entity(e)
    monkeypatch.setenv("KG_EXPORT_ONLY", "1")
    assert [h["id"] for h in server.kg_search("pricing")] == ["safe"]
    assert server.kg_get_entity("safe")["export_allowed"] is True
    assert server.kg_get_entity("private")["policy_blocked"] is True
    assert all(h["id"] == "safe" for h in server.kg_recent())
    assert server.kg_policy()["read_mode"] == "approved_exports"
    with pytest.raises(PermissionError, match="read-only"):
        server.kg_add_entity("Concept", "Forbidden mutation")
    with pytest.raises(PermissionError, match="read-only"):
        server.kg_update_entity("safe", properties={"export_policy": "approved"})
    with pytest.raises(PermissionError, match="read-only"):
        server.kg_delete_entity("private")


def test_export_only_traversal_cannot_cross_unapproved_nodes(server, monkeypatch):
    from knowledge_graph.models import Entity, Relation
    from knowledge_graph.privacy import approve_export
    for eid in ("a", "private", "b"):
        e = Entity(id=eid, label="Concept", name=f"Node {eid}", properties={"domain": "professional"})
        if eid != "private": approve_export(e)
        server.store.add_entity(e)
    for start, end in (("a", "private"), ("private", "b")):
        server.store.add_relation(Relation(id=f"{start}-{end}", source_id=start, target_id=end,
                                          relation_type="references", connection_reason="PRIVATE_REASON"))
    monkeypatch.setenv("KG_EXPORT_ONLY", "1")
    assert server.kg_get_neighbors("a", depth=3) == {"1": [], "2": [], "3": []}
    assert server.kg_find_path("a", "b") is None
    assert server.kg_get_relations("a") == []
    server.store.add_relation(Relation(id="PRIVATE_RELATION_ID", source_id="a", target_id="b", relation_type="references",
                                      connection_reason="PRIVATE_REASON"))
    assert [h["id"] for h in server.kg_find_path("a", "b")] == ["a", "b"]
    assert "PRIVATE_REASON" not in str(server.kg_get_relations("a"))
    assert "PRIVATE_RELATION_ID" not in str(server.kg_get_relations("a"))


def test_export_only_omits_unbound_derived_and_approval_metadata(server, monkeypatch):
    from knowledge_graph.privacy import approve_export
    from knowledge_graph.models import Entity
    store = server.store
    monkeypatch.setenv("KG_EXPORT_ONLY", "1")
    entity = Entity(id="approved", label="Decision", name="Work fact", description="Use option A", properties={"domain": "professional"})
    approve_export(entity)
    entity.properties.update(embedding_model="PRIVATE", approved_by="PRIVATE", embedding_status="PRIVATE")
    store.add_entity(entity)
    hit = server.kg_get_entity(entity.id)
    assert hit["export_allowed"]
    assert "updated_at" not in hit and "access_count" not in hit and "importance_score" not in hit
    assert not {"embedding_model", "approved_by", "embedding_status", "approved_export_hash"} & hit["properties"].keys()
