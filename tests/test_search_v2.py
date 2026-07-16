"""Hybrid search + MCP surface v2 tests."""

import importlib
import sys

import pytest

from knowledge_graph.models import Entity
from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture()
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path))


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("KG_DATA_DIR", str(tmp_path))
    sys.modules.pop("knowledge_graph.mcp_server", None)
    mod = importlib.import_module("knowledge_graph.mcp_server")
    yield mod
    sys.modules.pop("knowledge_graph.mcp_server", None)


def test_search_matches_regardless_of_word_order(store):
    store.add_entity(Entity(id="e1", label="Concept", name="Routing for agents"))
    store.add_entity(Entity(id="e2", label="Concept", name="Baking sourdough"))
    hits = store.search_entities("agent routing", semantic=False)
    assert [e.id for e in hits] == ["e1"], "tokenized search must not require contiguous substring"


def test_search_filters(store):
    store.add_entity(Entity(id="a", label="Document", name="AI weekly digest",
                            tags=["signal-high"], source_app="chrome"))
    store.add_entity(Entity(id="b", label="Document", name="AI monthly digest",
                            tags=["signal-low"], source_app="fieldy"))

    assert [e.id for e in store.search_entities("digest", tags=["signal-high"], semantic=False)] == ["a"]
    assert [e.id for e in store.search_entities("digest", source_app="fieldy", semantic=False)] == ["b"]
    assert store.search_entities("digest", label="Concept", semantic=False) == []


def test_search_blends_importance(store):
    store.add_entity(Entity(id="lo", label="Concept", name="pricing strategy", importance_score=0.1))
    store.add_entity(Entity(id="hi", label="Concept", name="pricing strategy notes", importance_score=0.95))
    hits = store.search_entities("pricing strategy", semantic=False)
    assert hits, "both should match"
    # exact-name phrase bonus can outweigh importance; both must at least rank
    assert {e.id for e in hits} == {"lo", "hi"}


def test_search_semantic_blend_with_fake_embedder(store, monkeypatch):
    store.add_entity(Entity(id="lex", label="Concept", name="postgres tuning"))
    store.add_entity(Entity(id="sem", label="Concept", name="database performance",
                            embedding=[1.0, 0.0, 0.0]))
    monkeypatch.setattr(store, "_embed_query", lambda q: [1.0, 0.0, 0.0])
    hits = store.search_entities("postgres tuning", semantic=True)
    ids = [e.id for e in hits]
    assert "lex" in ids
    assert "sem" in ids, "high-cosine entity must surface even with zero lexical overlap"


def test_search_entities_scored_scores_are_absolute(store):
    store.add_entity(Entity(id="hit", label="Concept", name="postgres tuning"))
    store.add_entity(Entity(id="weak", label="Concept", name="tuning notes"))
    scored = store.search_entities_scored("postgres tuning", semantic=False)
    assert [e.id for _, e in scored] == [entity.id for entity in store.search_entities("postgres tuning", semantic=False)]
    by_id = {e.id: s for s, e in scored}
    assert by_id["hit"] > by_id["weak"], "scores must reflect absolute relevance, not position"
    assert all(0 <= s <= 1.2 for s in by_id.values())

    # cross-store comparability: a store whose ONLY match is weak must not
    # score it like a perfect match (the old observed-max normalization did)
    lonely = KnowledgeGraphStore(data_dir=str(store.data_dir) + "-lonely")
    lonely.add_entity(Entity(id="w2", label="Concept", name="tuning notes"))
    lonely_score = lonely.search_entities_scored("postgres tuning", semantic=False)[0][0]
    assert abs(lonely_score - by_id["weak"]) < 1e-9, "same entity, same query → same score in any store"
    assert lonely_score < by_id["hit"]


def test_centrality_cache_invalidated_on_write(store):
    store.add_entity(Entity(id="x", label="Concept", name="X"))
    store.get_centrality("x", metric="pagerank")
    assert "pagerank" in store._centrality_cache
    store.add_entity(Entity(id="y", label="Concept", name="Y"))
    assert store._centrality_cache == {}, "writes must invalidate centrality cache"


def test_mcp_bulk_add_dedupes(server):
    ids = server.kg_add_entities([
        {"label": "Concept", "name": "Bulk one", "description": "same"},
        {"label": "Concept", "name": "Bulk two"},
        {"label": "Concept", "name": "Bulk one", "description": "same"},
    ])
    assert len(ids) == 3
    assert ids[0] == ids[2], "content-identical bulk entries dedupe"
    assert ids[0] != ids[1]


def test_mcp_relations_and_typed_neighbors(server):
    a = server.kg_add_entity(label="Concept", name="Service")
    b = server.kg_add_entity(label="Concept", name="Database")
    server.kg_add_relation(source_id=a, target_id=b, relation_type="depends_on", reason="storage")

    rels = server.kg_get_relations(a, direction="out")
    assert len(rels) == 1
    assert rels[0]["relation_type"] == "depends_on"
    assert rels[0]["reason"] == "storage"

    typed = server.kg_get_neighbors(a, depth=1, relation_type="depends_on")
    assert any(n["id"] == b for level in typed.values() for n in level)
    empty = server.kg_get_neighbors(a, depth=1, relation_type="part_of")
    assert all(not level for level in empty.values())


def test_mcp_find_by_tag(server):
    server.kg_add_entity(label="Document", name="Tagged doc", tags=["signal-high"])
    hits = server.kg_find_by_tag("signal-high")
    assert any(h["name"] == "Tagged doc" for h in hits)
    assert server.kg_find_by_tag("signal-high", label="Concept") == []


def test_mcp_find_similar_without_embeddings_is_empty(server):
    eid = server.kg_add_entity(label="Concept", name="No vectors here")
    assert server.kg_find_similar(eid) == []


def test_recent_entities_window_and_order(store):
    from datetime import datetime, timedelta

    now = datetime.now()
    store.add_entity(Entity(id="old", label="Document", name="ancient doc",
                            created_at=now - timedelta(days=30)))
    store.add_entity(Entity(id="mid", label="Document", name="last week doc",
                            created_at=now - timedelta(days=5)))
    store.add_entity(Entity(id="new", label="Event", name="yesterday event",
                            created_at=now - timedelta(days=1), source_app="fieldy"))

    week = store.recent_entities(days=7)
    assert [e.id for e in week] == ["new", "mid"], "windowed, newest first"
    assert [e.id for e in store.recent_entities(days=7, label="Event")] == ["new"]
    assert [e.id for e in store.recent_entities(days=7, source_app="fieldy")] == ["new"]
    assert [e.id for e in store.recent_entities(days=7, limit=1)] == ["new"]
    assert [e.id for e in store.recent_entities(days=60)][-1] == "old"


def test_kg_recent_tool(server):
    from datetime import datetime, timedelta

    server.store.add_entity(Entity(id="r1", label="Decision", name="ship phase 5",
                                   created_at=datetime.now() - timedelta(days=2)))
    rows = server.kg_recent(days=7)
    assert rows and rows[0]["id"] == "r1"
    assert rows[0]["created_at"] is not None


def test_recent_entities_mixed_tz(store):
    from datetime import datetime, timedelta, timezone

    store.add_entity(Entity(id="aware", label="Event", name="utc stamped",
                            created_at=datetime.now(timezone.utc) - timedelta(days=1)))
    store.add_entity(Entity(id="naive", label="Event", name="local stamped",
                            created_at=datetime.now() - timedelta(days=2)))
    assert [e.id for e in store.recent_entities(days=7)] == ["aware", "naive"]
