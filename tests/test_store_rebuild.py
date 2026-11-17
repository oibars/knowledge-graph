"""Canonical-store regressions: identity, embedding freshness and typed recall."""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)


def entity(eid="a", **kwargs):
    return Entity(id=eid, label=kwargs.pop("label", "Concept"), name=kwargs.pop("name", "Original"),
                  description=kwargs.pop("description", "old content"), **kwargs)


def test_old_hash_cannot_discard_another_record(store):
    original = entity()
    old_hash = original.content_hash
    store.add_entity(original)
    edited = store.get_entity("a")
    edited.description = "new content"
    store.update_entity(edited)
    assert store.get_entity("a").content_hash != old_hash
    assert store.add_entity(entity("b")) == "b"
    assert len(store.get_all_entities()) == 2


def test_independent_stores_dedup_atomically(store, tmp_path):
    other = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    assert store.add_entity(entity()) == "a"
    assert other.add_entity(entity("b")) == "a"
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT count(*) FROM entities").fetchone()[0] == 1


@pytest.mark.parametrize("label", ["Event", "Task", "Decision", "Chunk"])
def test_distinct_source_occurrences_survive_identical_text(store, label):
    assert store.add_entity(entity("a", label=label)) == "a"
    assert store.add_entity(entity("b", label=label)) == "b"


def test_label_and_document_source_boundaries(store):
    store.add_entity(entity("a"))
    assert store.add_entity(entity("b", label="Document")) == "b"
    assert store.add_entity(entity("doc-1", label="Document", source_url="fieldy://one")) == "doc-1"
    assert store.add_entity(entity("doc-2", label="Document", source_url="fieldy://two")) == "doc-2"


def test_vector_invalidated_on_changed_content_but_retained_for_unchanged(store):
    store.add_entity(entity(embedding=[1., 0.]))
    store.add_entity(entity())
    assert store.get_entity("a").embedding == [1., 0.]
    edit = store.get_entity("a")
    edit.description = "changed through returned entity"
    store.update_entity(edit)
    assert store.get_entity("a").embedding is None
    assert store.get_entity("a").properties["embedding_status"] == "pending"


def test_same_id_reads_birth_and_vector_from_sqlite(store, tmp_path):
    other = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    born = datetime(2020, 1, 1, tzinfo=timezone.utc)
    other.add_entity(entity(created_at=born, embedding=[.2, .3]))
    store.add_entity(entity())
    got = store.get_entity("a")
    assert got.created_at == born
    assert got.embedding == [.2, .3]


def test_failed_write_does_not_change_cached_entity_or_relation(store):
    store.add_entity(entity())
    store.add_entity(entity("b", name="Other"))
    with sqlite3.connect(store.db_path) as db:
        db.execute("CREATE TRIGGER reject_entity BEFORE UPDATE ON entities BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        db.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON entities WHEN NEW.id='a' BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        db.execute("CREATE TRIGGER reject_relation BEFORE INSERT ON relations BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    edit = store.get_entity("a")
    edit.description = "uncommitted"
    with pytest.raises(sqlite3.IntegrityError):
        store.update_entity(edit)
    assert store.get_entity("a").description == "old content"
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relation(Relation(id="bad", source_id="a", target_id="b", relation_type="references"))
    assert store.get_relation("bad") is None
    assert store.get_neighbors("a")[1] == []


def test_multiple_typed_edges_survive_deletion_reload_and_path(store, tmp_path):
    store.add_entity(entity())
    store.add_entity(entity("b", name="Other"))
    store.add_relation(Relation(id="ref", source_id="a", target_id="b", relation_type="references", bidirectional=True))
    store.add_relation(Relation(id="dep", source_id="a", target_id="b", relation_type="depends_on"))
    assert len(store.get_entity_relations("a", "out")) == 2
    assert {e.id for e in store.get_neighbors("a", "references")[1]} == {"b"}
    assert store.delete_relation("ref")
    assert {e.id for e in store.get_neighbors("a", "depends_on")[1]} == {"b"}
    assert store.get_relation("ref_inverse") is None
    fresh = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    assert {e.id for e in fresh.get_neighbors("a", "depends_on")[1]} == {"b"}
    assert fresh.find_path("a", "b")
    assert fresh.get_centrality("a", "degree") == 1


def test_relation_replacement_moves_edges_and_removes_old_inverse(store):
    for eid in ("a", "b", "c"):
        store.add_entity(entity(eid, name=eid))
    store.add_relation(Relation(id="r", source_id="a", target_id="b", relation_type="references", bidirectional=True))
    store.add_relation(Relation(id="r", source_id="a", target_id="c", relation_type="depends_on"))
    assert {e.id for e in store.get_neighbors("a")[1]} == {"c"}
    assert store.get_relation("r_inverse") is None


def test_reference_corpus_is_explicit_and_inferred(store):
    store.add_entity(entity("memory", name="Pricing decision"))
    store.add_entity(entity("news", name="Pricing trends", source_app="swiftrecap"))
    store.add_entity(entity("trend", name="Pricing report", source_app="trend-monitor"))
    store.add_entity(entity("declared", name="Pricing research", properties={"corpus": "reference"}))
    assert {e.id for e in store.search_entities("pricing", semantic=False, corpus="memory")} == {"memory"}
    assert {e.id for e in store.search_entities("pricing", semantic=False, corpus="reference")} == {"news", "trend", "declared"}
    assert len(store.search_entities("pricing", semantic=False, corpus="all")) == 4
    with pytest.raises(ValueError):
        store.search_entities("pricing", corpus="unknown")


def test_invalid_snapshot_does_not_replace_sql_or_memory(store, tmp_path):
    store.add_entity(entity())
    snapshot = tmp_path / "invalid.json"
    snapshot.write_text(json.dumps({"entities": {"b": entity("b").to_dict()}, "relations": {
        "r": Relation(id="r", source_id="b", target_id="missing", relation_type="references").to_dict()}}))
    assert store.load_snapshot(str(snapshot)) is False
    assert store.get_entity("a") is not None
    assert store.get_entity("b") is None
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT id FROM entities").fetchall() == [("a",)]


def test_contending_writers_share_content_identity(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    stores = [KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False) for _ in range(2)]
    barrier = Barrier(2)
    def write(index):
        barrier.wait(timeout=5)
        return stores[index].add_entity(entity(f"writer-{index}"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(write, range(2)))
    assert ids[0] == ids[1]
    assert len(stores[0].get_all_entities()) == len(stores[1].get_all_entities()) == 1


def test_inverse_failure_rolls_back_original_relation(store):
    store.add_entity(entity())
    store.add_entity(entity("b", name="Other"))
    with sqlite3.connect(store.db_path) as db:
        db.execute("CREATE TRIGGER reject_inverse BEFORE INSERT ON relations WHEN NEW.id='r_inverse' BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relation(Relation(id="r", source_id="a", target_id="b", relation_type="references", bidirectional=True))
    assert store.get_all_relations() == []
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT count(*) FROM relations").fetchone()[0] == 0


def test_failed_delete_keeps_canonical_and_cached_record(store):
    store.add_entity(entity())
    with sqlite3.connect(store.db_path) as db:
        db.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON entities BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.delete_entity("a")
    assert store.get_entity("a") is not None


def test_external_content_survives_our_access_flush_and_write(store, tmp_path):
    store.add_entity(entity())
    store.get_entity("a")
    other = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    other.add_entity(entity(description="external revised content"))
    store._flush_access()
    store.add_entity(entity("b", name="Our new concept"))
    assert store.get_entity("a").description == "external revised content"
    assert len(store.get_all_entities()) == 2


def test_known_embedding_models_are_not_compared(store, monkeypatch):
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    store.add_entity(entity("a", name="Irrelevant one", embedding=[1., 0.], properties={"embedding_model": "other-model"}))
    store.add_entity(entity("b", name="Irrelevant two", embedding=[1., 0.], properties={"embedding_model": "embeddinggemma:latest"}))
    monkeypatch.setattr(store, "_embed_query", lambda _: [1., 0.])
    assert [e.id for e in store.search_entities("unrelated phrase")] == ["b"]
    assert store.find_similar_entities("a") == []


def test_embedding_model_latest_aliases_match_for_query_and_neighbors(store, monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    store.add_entity(entity("a", name="Irrelevant one", embedding=[1., 0.], properties={"embedding_model": "embeddinggemma:latest"}))
    store.add_entity(entity("b", name="Irrelevant two", embedding=[1., 0.], properties={"embedding_model": "embeddinggemma"}))
    monkeypatch.setattr(store, "_embed_query", lambda _: [1., 0.])
    assert {e.id for e in store.search_entities("unrelated phrase")} == {"a", "b"}
    assert [e.id for e, _ in store.find_similar_entities("a")] == ["b"]


def test_query_embedding_defaults_to_installed_laptop_model(store, monkeypatch):
    from knowledge_graph.services import graph_store
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    requests = []
    def local(path, payload, url, timeout):
        requests.append(payload)
        return {"embeddings": [[1., 0.]]}
    monkeypatch.setattr(graph_store, "local_json", local)
    assert store._embed_query("synthetic") == [1., 0.]
    assert requests[0]["model"] == "embeddinggemma:latest"
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "other-model:custom")
    store._embed_query("synthetic")
    assert requests[-1]["model"] == "other-model:custom"


def test_canonical_model_keeps_registry_ports_and_explicit_tags():
    from knowledge_graph.services.graph_store import canonical_embedding_model
    assert canonical_embedding_model("localhost:5000/team/model") == "localhost:5000/team/model:latest"
    assert canonical_embedding_model("team/model:v2") == "team/model:v2"


def test_self_relation_reported_once(store):
    store.add_entity(entity())
    store.add_relation(Relation(id="self", source_id="a", target_id="a", relation_type="references"))
    assert len(store.get_entity_relations("a")) == 1


def test_read_does_not_change_content_update_time(store):
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    store.add_entity(entity(updated_at=old))
    assert store.get_entity("a").updated_at == old


def test_read_counts_not_doubled_when_entity_is_updated_before_flush(store):
    store.add_entity(entity())
    edited = store.get_entity("a")
    edited.description = "revision"
    store.update_entity(edited)
    store._flush_access()
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT access_count FROM entities WHERE id='a'").fetchone()[0] == 1


def test_pending_read_counts_survive_same_id_ingestion(store):
    store.add_entity(entity())
    store.get_entity("a")
    store.add_entity(entity())
    assert store.get_entity("a").access_count == 2
    store._flush_access()
    with sqlite3.connect(store.db_path) as db:
        assert db.execute("SELECT access_count FROM entities WHERE id='a'").fetchone()[0] == 2


def test_entity_upsert_keeps_relations_under_foreign_key_enforcement(store):
    store.add_entity(entity())
    store.add_entity(entity("b", name="Other"))
    store.add_relation(Relation(id="r", source_id="a", target_id="b", relation_type="references"))
    store.add_entity(entity(description="replacement content"))
    assert store.get_relation("r") is not None
    assert {e.id for e in store.get_neighbors("a", "references")[1]} == {"b"}


def test_generic_dedup_preserves_domain_and_source_boundaries(store):
    store.add_entity(entity("a", source_app="local", properties={"domain": "professional"}))
    assert store.add_entity(entity("personal", source_app="local", properties={"domain": "personal"})) == "personal"
    assert store.add_entity(entity("reference", source_app="local", properties={"domain": "professional", "corpus": "reference"})) == "reference"
    assert store.add_entity(entity("other-app", source_app="other", properties={"domain": "professional"})) == "other-app"


def test_zero_metadata_survives_reload(store, tmp_path):
    store.add_entity(entity(importance_score=0., confidence_score=0.))
    store.add_relation(Relation(id="r", source_id="a", target_id="a", relation_type="references", strength=0., confidence=0.))
    fresh = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    assert fresh.get_entity("a").importance_score == fresh.get_entity("a").confidence_score == 0.
    assert fresh.get_relation("r").strength == fresh.get_relation("r").confidence == 0.


def test_search_embedding_refuses_non_loopback_endpoint(store, monkeypatch):
    import urllib.request
    calls = []
    monkeypatch.setenv("OLLAMA_URL", "https://example.invalid")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: calls.append(args))
    assert store._embed_query("private query") is None
    assert calls == []
