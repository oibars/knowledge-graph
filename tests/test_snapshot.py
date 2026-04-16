"""Tests for snapshot create/load round-trip."""

import pytest

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=True)


def test_snapshot_create_and_load(tmp_path):
    store1 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=True)
    store1.add_entity(Entity(id="s1", label="Concept", name="Snapshot Concept",
                             topics=["persistence"], tags=["test"]))
    store1.add_entity(Entity(id="s2", label="Task", name="Snapshot Task"))
    store1.add_relation(Relation(id="r1", source_id="s1", target_id="s2",
                                  relation_type="prerequisite_for"))

    snapshot_path = store1.create_snapshot()
    assert snapshot_path != ""

    store2 = KnowledgeGraphStore(data_dir=str(tmp_path) + "/fresh", enable_snapshots=False)
    result = store2.load_snapshot(snapshot_path)
    assert result is True

    e = store2.get_entity("s1")
    assert e is not None
    assert e.name == "Snapshot Concept"
    assert e.topics == ["persistence"]
    assert e.tags == ["test"]

    r = store2.get_relation("r1")
    assert r is not None
    assert r.relation_type == "prerequisite_for"


def test_snapshot_stats(store):
    store.add_entity(Entity(id="a", label="Concept", name="A"))
    store.add_entity(Entity(id="b", label="Agent", name="B"))
    store.add_relation(Relation(id="r", source_id="a", target_id="b", relation_type="routed_to"))

    snap = store.create_snapshot()
    assert snap != ""

    stats = store.get_stats()
    assert stats["entity_count"] == 2
    assert stats["relation_count"] == 1
    assert "Concept" in stats["label_distribution"]
    assert "Agent" in stats["label_distribution"]


def test_no_snapshot_when_disabled(tmp_path):
    store = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
    store.add_entity(Entity(id="x", label="Concept", name="X"))
    path = store.create_snapshot()
    assert path == ""
