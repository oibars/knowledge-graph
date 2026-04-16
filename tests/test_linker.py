"""Tests for SemanticLinker with mocked LLM and embeddings."""

import pytest
from unittest.mock import MagicMock, patch

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.services.linker import SemanticLinker


@pytest.fixture
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path), enable_snapshots=False)


@pytest.fixture
def linker(store):
    return SemanticLinker(store, use_ollama=False)


@pytest.fixture
def linker_with_llm(store):
    mock_client = MagicMock()
    # Simulate Claude API response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="authentication, JWT, session management")]
    mock_client.messages.create.return_value = mock_response
    # Disable Ollama so the mock LLM fallback is used
    return SemanticLinker(store, llm_client=mock_client, use_ollama=False)


class TestExtractConcepts:
    def test_no_llm_returns_empty(self, linker):
        result = linker._extract_concepts_from_text("Some important text about caching")
        assert result == []

    def test_with_mock_llm_returns_concepts(self, linker_with_llm):
        result = linker_with_llm._extract_concepts_from_text("Text about auth and sessions")
        assert "authentication" in result
        assert "JWT" in result
        assert "session management" in result

    def test_llm_api_error_returns_empty(self, store):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        linker = SemanticLinker(store, llm_client=mock_client, use_ollama=False)
        result = linker._extract_concepts_from_text("Some text")
        assert result == []


class TestLinkFileToConcepts:
    def test_no_embedding_no_similar(self, store, linker):
        file_entity = Entity(id="f1", label="File", name="auth.py",
                              description="JWT authentication module")
        store.add_entity(file_entity)
        relations = linker.link_file_to_concepts("f1", extract_concepts=False)
        assert relations == []

    def test_extract_concepts_creates_entities_and_relations(self, store, linker_with_llm):
        file_entity = Entity(id="f1", label="File", name="auth.py",
                              description="JWT authentication module")
        store.add_entity(file_entity)
        relations = linker_with_llm.link_file_to_concepts("f1", extract_concepts=True)

        # Should have created concept entities + relations
        assert len(relations) > 0
        # Concept entities should exist
        concepts = store.find_by_label("Concept")
        assert len(concepts) > 0

    def test_nonexistent_entity_returns_empty(self, linker):
        relations = linker.link_file_to_concepts("does-not-exist")
        assert relations == []


class TestSourceFilePath:
    """Regression test for the source_file_path.parent type error (Bug #2)."""

    def test_task_linking_with_string_paths(self, store, linker):
        t1 = Entity(id="t1", label="Task", name="Task One",
                    source_file_path="/home/user/project/tasks/task1.md",
                    tags=["backend"])
        t2 = Entity(id="t2", label="Task", name="Task Two",
                    source_file_path="/home/user/project/tasks/task2.md",
                    tags=["backend"])
        store.add_entity(t1)
        store.add_entity(t2)

        # Should not raise AttributeError: 'str' object has no attribute 'parent'
        relations = linker.link_related_tasks("t1")
        # Both tasks share same directory and tag — should find similarity
        # (Won't meet threshold without embeddings, but should not crash)
        assert isinstance(relations, list)


class TestSuggestConnections:
    def test_suggests_by_name_overlap(self, store, linker):
        # The overlap threshold is 3 common words — names must share enough terms
        store.add_entity(Entity(id="a", label="Concept",
                                name="distributed database connection pooling strategy"))
        store.add_entity(Entity(id="b", label="Concept",
                                name="distributed database connection management layer"))
        store.add_entity(Entity(id="c", label="Concept",
                                name="unrelated frontend rendering pipeline"))

        suggestions = linker.suggest_connections("a", limit=5)
        suggestion_ids = [e.id for e, _, _ in suggestions]
        # "b" shares "distributed", "database", "connection" (3+) with "a"
        assert "b" in suggestion_ids
        assert "c" not in suggestion_ids

    def test_no_self_suggestion(self, store, linker):
        store.add_entity(Entity(id="self", label="Concept", name="singleton pattern"))
        suggestions = linker.suggest_connections("self", limit=5)
        assert all(e.id != "self" for e, _, _ in suggestions)

    def test_nonexistent_entity_returns_empty(self, linker):
        assert linker.suggest_connections("ghost") == []
