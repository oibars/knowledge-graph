"""Tests for Entity CRUD operations in KnowledgeGraphStore."""

import pytest
import tempfile
from pathlib import Path

from knowledge_graph.models import Entity, ENTITY_LABELS
from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture
def store(tmp_path):
    """Fresh store backed by a temp directory."""
    return KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)


def make_entity(**kwargs) -> Entity:
    defaults = dict(id="e1", label="Concept", name="Test Concept", description="A test concept")
    defaults.update(kwargs)
    return Entity(**defaults)


class TestAddAndGet:
    def test_add_and_retrieve(self, store):
        e = make_entity()
        store.add_entity(e)
        result = store.get_entity("e1")
        assert result is not None
        assert result.id == "e1"
        assert result.name == "Test Concept"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_entity("does-not-exist") is None

    def test_add_updates_access_count(self, store):
        e = make_entity()
        store.add_entity(e)
        result = store.get_entity("e1")
        assert result.access_count == 1

    def test_upsert_overwrites(self, store):
        e = make_entity()
        store.add_entity(e)
        e.name = "Updated Name"
        store.add_entity(e)
        assert store.get_entity("e1").name == "Updated Name"
        assert len(store.get_all_entities()) == 1

    def test_all_entity_labels_accepted(self, store):
        for i, label in enumerate(ENTITY_LABELS.keys()):
            e = Entity(id=f"e-{i}", label=label, name=f"Entity {i}")
            store.add_entity(e)
        assert len(store.get_all_entities()) == len(ENTITY_LABELS)


class TestUpdate:
    def test_update_entity(self, store):
        e = make_entity()
        store.add_entity(e)
        e.description = "Updated description"
        store.update_entity(e)
        assert store.get_entity("e1").description == "Updated description"


class TestDelete:
    def test_delete_entity(self, store):
        e = make_entity()
        store.add_entity(e)
        assert store.delete_entity("e1") is True
        assert store.get_entity("e1") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_entity("ghost") is False


class TestSearch:
    def test_search_by_name(self, store):
        store.add_entity(make_entity(id="a", name="Authentication Service"))
        store.add_entity(make_entity(id="b", name="Payment Gateway"))
        results = store.search_entities("authentication")
        assert any(e.id == "a" for e in results)
        assert all(e.id != "b" for e in results)

    def test_search_by_tag(self, store):
        e = make_entity(tags=["security", "auth"])
        store.add_entity(e)
        results = store.search_entities("security")
        assert any(r.id == "e1" for r in results)

    def test_search_label_filter(self, store):
        store.add_entity(make_entity(id="c", label="Concept", name="Caching"))
        store.add_entity(make_entity(id="t", label="Task", name="Caching Task"))
        results = store.search_entities("caching", label="Concept")
        assert all(e.label == "Concept" for e in results)

    def test_find_by_label(self, store):
        store.add_entity(make_entity(id="a1", label="Agent", name="Code Reviewer"))
        store.add_entity(make_entity(id="a2", label="Agent", name="Security Engineer"))
        store.add_entity(make_entity(id="c1", label="Concept", name="JWT"))
        agents = store.find_by_label("Agent")
        assert len(agents) == 2
        assert all(a.label == "Agent" for a in agents)

    def test_find_by_tag(self, store):
        store.add_entity(make_entity(id="x", tags=["python", "backend"]))
        store.add_entity(make_entity(id="y", tags=["frontend"]))
        results = store.find_by_tag("python")
        assert len(results) == 1
        assert results[0].id == "x"


class TestPersistence:
    def test_survives_reload(self, tmp_path):
        store1 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        store1.add_entity(make_entity(id="persist-me", name="Persistent Entity"))

        store2 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        result = store2.get_entity("persist-me")
        assert result is not None
        assert result.name == "Persistent Entity"

    def test_topics_and_tags_serialized_correctly(self, tmp_path):
        store1 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        e = make_entity(topics=["distributed systems"], tags=["backend", "database"])
        store1.add_entity(e)

        store2 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        result = store2.get_entity("e1")
        assert result.topics == ["distributed systems"]
        assert result.tags == ["backend", "database"]

    def test_is_auto_generated_persisted(self, tmp_path):
        store1 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        e = make_entity(is_auto_generated=True)
        store1.add_entity(e)

        store2 = KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)
        assert store2.get_entity("e1").is_auto_generated is True
