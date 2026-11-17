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


@pytest.mark.parametrize("semantic", [False, True])
def test_hybrid_search_handles_mixed_dates_and_filters(store, monkeypatch, semantic):
    from datetime import datetime, timezone

    instant = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    store.add_entity(Entity(id="aware", label="Event", name="routing aware",
                            created_at=instant, updated_at=instant, embedding=[1., 0.]))
    store.add_entity(Entity(id="naive", label="Event", name="routing naive",
                            created_at=instant.astimezone().replace(tzinfo=None),
                            updated_at=instant.astimezone().replace(tzinfo=None)))
    monkeypatch.setattr(store, "_embed_query", lambda _: [1., 0.])
    hits = store.search_entities_scored(
        "routing", semantic=semantic,
        created_after="2026-07-01T07:59:59-04:00", created_before="2026-07-01T12:00:00Z",
    )
    assert {entity.id for _, entity in hits} == {"aware", "naive"}
    assert store.search_entities("routing", semantic=semantic,
                                 created_after="2026-07-01T12:00:01Z") == []


def test_hybrid_recency_ranks_mixed_datetimes_without_filters(store):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    store.add_entity(Entity(id="new", label="Event", name="routing new",
                            updated_at=now))
    store.add_entity(Entity(id="old", label="Event", name="routing old",
                            updated_at=(now - timedelta(days=365)).astimezone().replace(tzinfo=None)))
    assert [entity.id for entity in store.search_entities("routing", semantic=False)] == ["new", "old"]


def test_hybrid_naive_filters_keep_legacy_local_time_meaning(store):
    from datetime import datetime, timezone

    instant = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    local = instant.astimezone().replace(tzinfo=None).isoformat()
    store.add_entity(Entity(id="e1", label="Event", name="routing notes", created_at=instant))
    assert [entity.id for entity in store.search_entities(
        "routing", semantic=False, created_after=local, created_before=local,
    )] == ["e1"]


def test_utc_storage_roundtrip_and_legacy_rows(store):
    from datetime import datetime, timezone
    from knowledge_graph.models import Relation

    instant = datetime(2026, 7, 1, 12, tzinfo=timezone.utc)
    naive = instant.astimezone().replace(tzinfo=None)
    store.add_entity(Entity(id="e1", label="Event", name="routing notes",
                            created_at=naive, updated_at=instant, last_accessed=naive))
    store.add_entity(Entity(id="e2", label="Event", name="routing other"))
    store.add_relation(Relation(id="r1", source_id="e1", target_id="e2",
                                relation_type="related_to", created_at=naive))
    with store._connect() as connection:
        created, updated, accessed = connection.execute(
            "SELECT created_at, updated_at, last_accessed FROM entities WHERE id='e1'",
        ).fetchone()
        assert all(datetime.fromisoformat(value).tzinfo == timezone.utc
                   for value in (created, updated, accessed))
        # Simulate pre-migration database rows, without touching any real graph.
        connection.execute("UPDATE entities SET created_at=?, updated_at=? WHERE id='e1'",
                           (naive.isoformat(), "2026-07-01T08:00:00-04:00"))
    reloaded = KnowledgeGraphStore(data_dir=str(store.data_dir), enable_snapshots=False)
    entity = reloaded._entities["e1"]
    assert entity.created_at == entity.updated_at == entity.last_accessed == instant
    assert entity.created_at.tzinfo == entity.updated_at.tzinfo == entity.last_accessed.tzinfo == timezone.utc
    assert reloaded.get_relation("r1").created_at == instant
    assert reloaded.get_relation("r1").created_at.tzinfo == timezone.utc
    assert reloaded.search_entities("routing", semantic=False, created_after=instant.isoformat())


def test_lexical_cache_tracks_direct_field_mutations_and_upsert(store):
    entity = Entity(id="e1", label="Document", name="oldword", description="oldtext",
                    tags=["oldtag"], topics=["oldtopic"])
    store.add_entity(entity)
    for query in ("oldword", "oldtext", "oldtag", "oldtopic"):
        assert store.search_entities(query, semantic=False)
    # Public records are detached; persist edited fields before search sees them.
    entity.name = "newword"
    entity.description = "newtext"
    entity.tags[:] = ["newtag"]
    entity.topics[:] = ["newtopic"]
    assert store.search_entities("newword", semantic=False) == []
    store.update_entity(entity)
    for old, new in (("oldword", "newword"), ("oldtext", "newtext"),
                     ("oldtag", "newtag"), ("oldtopic", "newtopic")):
        assert store.search_entities(old, semantic=False) == []
        assert [hit.id for hit in store.search_entities(new, semantic=False)] == ["e1"]
    store.add_entity(Entity(id="e1", label="Document", name="replacement"))
    assert store.search_entities("newword", semantic=False) == []
    assert store.search_entities("replacement", semantic=False)
    store.delete_entity("e1")
    assert "e1" not in store._lexical_cache
    assert store.search_entities("replacement", semantic=False) == []


def test_lexical_cache_is_store_local_and_refreshes_external_writes(store):
    store.add_entity(Entity(id="e1", label="Document", name="original"))
    assert store.search_entities("original", semantic=False)
    other = KnowledgeGraphStore(data_dir=str(store.data_dir), enable_snapshots=False)
    assert other._lexical_cache == {}
    other.add_entity(Entity(id="e1", label="Document", name="changed"))
    store.refresh_if_stale()
    assert store.search_entities("original", semantic=False) == []
    assert [hit.id for hit in store.search_entities("changed", semantic=False)] == ["e1"]


@pytest.mark.parametrize("zone, utc_hour", [("UTC", 12), ("America/New_York", 16)])
def test_legacy_naive_timestamp_uses_host_local_zone(monkeypatch, zone, utc_hour):
    import time
    from datetime import datetime, timezone
    from knowledge_graph.models.timestamps import as_utc

    if not hasattr(time, "tzset"):
        pytest.skip("Host timezone override needs tzset")
    try:
        with monkeypatch.context() as context:
            context.setenv("TZ", zone)
            time.tzset()
            assert as_utc(datetime(2026, 7, 1, 12)) == datetime(2026, 7, 1, utc_hour, tzinfo=timezone.utc)
    finally:
        time.tzset()
def test_action_metadata_and_evidence_are_lexically_searchable(store):
    entity = Entity(id="task-source", label="Task", name="Send proposal", description="Prepare final proposal",
                    properties={"owner": "Marisol", "due_at": "2026-11-23", "status": "cancelled",
                                "evidence": [{"quote": "We need the heliotrope attachment", "speaker": "Aurelio"}]})
    store.add_entity(entity)
    for query in ("Marisol", "2026-11-23", "cancelled", "heliotrope", "Aurelio"):
        assert [hit.id for hit in store.search_entities(query, semantic=False)] == [entity.id]


def test_action_metadata_refreshes_lexical_cache_without_changing_embedding_input(store):
    entity = Entity(id="decision-source", label="Decision", name="Choose option A", description="Option A selected",
                    embedding=[1., 0.], properties={"owner": "Marisol", "status": "decided", "due_at": "2026-11-23"})
    store.add_entity(entity)
    assert store.search_entities("Marisol", semantic=False)
    original_hash = store.get_entity(entity.id).content_hash
    entity.properties.update(owner="Aurelio", status="superseded", due_at="2026-12-14")
    store.update_entity(entity)
    for query in ("Aurelio", "superseded", "2026-12-14"):
        assert store.search_entities(query, semantic=False)
    assert not store.search_entities("Marisol", semantic=False)
    result = store.get_entity(entity.id)
    assert result.content_hash == original_hash and result.embedding == [1., 0.]


def test_action_lexical_metadata_excludes_arbitrary_properties_and_other_labels(store):
    store.add_entity(Entity(id="task", label="Task", name="Task", properties={"arbitrary": "sensitivehiddenword"}))
    store.add_entity(Entity(id="concept", label="Concept", name="Concept", properties={"owner": "secretownerword", "evidence": [{"quote": "hiddenquoteword"}]}))
    for query in ("sensitivehiddenword", "secretownerword", "hiddenquoteword"):
        assert not store.search_entities(query, semantic=False)
