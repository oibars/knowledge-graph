"""Tests for Relation CRUD and bidirectional inverse generation."""

import pytest

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore


@pytest.fixture
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)


@pytest.fixture
def two_entities(store):
    a = Entity(id="a", label="Concept", name="Entity A")
    b = Entity(id="b", label="Concept", name="Entity B")
    store.add_entity(a)
    store.add_entity(b)
    return a, b


def make_relation(source="a", target="b", rtype="depends_on", **kwargs) -> Relation:
    return Relation(
        id=f"rel-{source}-{target}-{rtype}",
        source_id=source,
        target_id=target,
        relation_type=rtype,
        **kwargs,
    )


class TestAddAndGet:
    def test_add_and_retrieve(self, store, two_entities):
        r = make_relation()
        store.add_relation(r)
        result = store.get_relation(r.id)
        assert result is not None
        assert result.source_id == "a"
        assert result.target_id == "b"

    def test_missing_source_raises(self, store):
        store.add_entity(Entity(id="b", label="Concept", name="B"))
        r = make_relation(source="ghost", target="b")
        with pytest.raises(ValueError, match="Source entity not found"):
            store.add_relation(r)

    def test_missing_target_raises(self, store):
        store.add_entity(Entity(id="a", label="Concept", name="A"))
        r = make_relation(source="a", target="ghost")
        with pytest.raises(ValueError, match="Target entity not found"):
            store.add_relation(r)


class TestBidirectional:
    def test_bidirectional_creates_inverse(self, store, two_entities):
        r = make_relation(rtype="contains", bidirectional=True)
        store.add_relation(r)
        # Original
        assert store.get_relation(r.id) is not None
        # Inverse (contains → part_of)
        inverse_id = r.get_inverse_id()
        inverse = store.get_relation(inverse_id)
        assert inverse is not None
        assert inverse.source_id == "b"
        assert inverse.target_id == "a"
        assert inverse.relation_type == "part_of"

    def test_unidirectional_no_inverse(self, store, two_entities):
        r = make_relation(rtype="depends_on", bidirectional=False)
        store.add_relation(r)
        assert store.get_relation(r.get_inverse_id()) is None


class TestDelete:
    def test_delete_relation(self, store, two_entities):
        r = make_relation()
        store.add_relation(r)
        assert store.delete_relation(r.id) is True
        assert store.get_relation(r.id) is None

    def test_deleting_entity_removes_relations(self, store, two_entities):
        r = make_relation()
        store.add_relation(r)
        store.delete_entity("a")
        assert store.get_relation(r.id) is None


class TestEntityRelations:
    def test_get_outgoing_relations(self, store, two_entities):
        r = make_relation()
        store.add_relation(r)
        rels = store.get_entity_relations("a", direction="out")
        assert len(rels) == 1
        assert rels[0].source_id == "a"

    def test_get_incoming_relations(self, store, two_entities):
        r = make_relation()
        store.add_relation(r)
        rels = store.get_entity_relations("b", direction="in")
        assert len(rels) == 1
        assert rels[0].target_id == "b"

    def test_get_both_directions(self, store, two_entities):
        store.add_relation(make_relation(source="a", target="b", rtype="depends_on"))
        store.add_relation(make_relation(source="b", target="a", rtype="references"))
        rels_a = store.get_entity_relations("a", direction="both")
        assert len(rels_a) == 2


class TestGraphTraversal:
    def test_neighbors(self, store):
        for eid in ["x", "y", "z"]:
            store.add_entity(Entity(id=eid, label="Concept", name=eid.upper()))
        store.add_relation(make_relation("x", "y", "uses"))
        store.add_relation(make_relation("y", "z", "uses"))

        neighbors = store.get_neighbors("x", depth=1)
        assert "y" in [e.id for e in neighbors[1]]

        neighbors2 = store.get_neighbors("x", depth=2)
        assert "z" in [e.id for e in neighbors2[2]]

    def test_find_path(self, store):
        for eid in ["p", "q", "r"]:
            store.add_entity(Entity(id=eid, label="Concept", name=eid.upper()))
        store.add_relation(make_relation("p", "q", "depends_on"))
        store.add_relation(make_relation("q", "r", "depends_on"))

        path = store.find_path("p", "r")
        assert path is not None
        assert [e.id for e in path] == ["p", "q", "r"]

    def test_find_path_none_when_disconnected(self, store):
        store.add_entity(Entity(id="isolated1", label="Concept", name="I1"))
        store.add_entity(Entity(id="isolated2", label="Concept", name="I2"))
        assert store.find_path("isolated1", "isolated2") is None


class TestRelationInverseMap:
    def test_known_inverse_types(self):
        cases = [
            ("contains", "part_of"),
            ("depends_on", "required_by"),
            ("implements", "implemented_by"),
            ("uses", "used_by"),
            ("produces", "produced_by"),
        ]
        for rtype, expected_inverse in cases:
            r = Relation(id="r", source_id="a", target_id="b", relation_type=rtype)
            assert r.get_inverse_type() == expected_inverse

    def test_unknown_type_returns_none(self):
        r = Relation(id="r", source_id="a", target_id="b", relation_type="routed_to")
        assert r.get_inverse_type() is None
