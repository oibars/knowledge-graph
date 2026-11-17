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


def test_related_tasks_compare_latest_embedding_aliases(store, linker):
    linker.similarity_threshold = .4
    for entity_id, model in (("t1", "embeddinggemma"), ("t2", "embeddinggemma:latest")):
        store.add_entity(Entity(id=entity_id, label="Task", name=entity_id, embedding=[1., 0.],
                                properties={"embedding_model": model}))
    assert linker.link_related_tasks("t1")


def test_linker_routine_generation_disables_thinking(tmp_path, monkeypatch):
    from knowledge_graph.services import linker
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(tmp_path))
    requests = []
    def local(path, payload, url, timeout):
        requests.append(payload)
        return {"response": "concept"}
    monkeypatch.setattr(linker, "local_json", local)
    monkeypatch.setenv("JARVIS_LOCAL_THINK", "1")
    assert linker._ollama_generate("synthetic") == "concept"
    assert requests[0]["think"] is False


class TestExtractConcepts:
    def test_no_llm_returns_empty(self, linker):
        result = linker._extract_concepts_from_text("Some important text about caching")
        assert result == []

    def test_with_mock_llm_returns_concepts(self, linker_with_llm):
        from knowledge_graph.privacy import approve_export
        source = Entity(id="approved", label="File", name="Approved", description="Text about auth and sessions",
                        properties={"domain": "professional"})
        approve_export(source)
        result = linker_with_llm._extract_concepts_from_text(source.description, source_entity=source)
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
        from knowledge_graph.privacy import approve_export
        file_entity = Entity(id="f1", label="File", name="auth.py",
                              description="JWT authentication module", properties={"domain": "professional"})
        approve_export(file_entity)
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


def test_unapproved_personal_and_unknown_cannot_reach_cloud(store):
    client = MagicMock()
    linker = SemanticLinker(store, llm_client=client, use_ollama=False)
    for domain in ("personal", "unknown", "professional", "mixed"):
        source = Entity(id=domain, label="File", name="sentinel", description="PRIVATE_SENTINEL",
                        properties={"domain": domain})
        store.add_entity(source)
        assert linker.link_file_to_concepts(domain) == []
    assert linker._extract_concepts_from_text("PRIVATE_SENTINEL") == []
    client.messages.create.assert_not_called()


def test_cloud_requires_matching_approved_source_text(store):
    from knowledge_graph.privacy import approve_export
    client = MagicMock()
    source = Entity(id="source", label="File", name="Selected fact", description="Approved source text",
                    properties={"domain": "professional"})
    approve_export(source)
    linker = SemanticLinker(store, llm_client=client, use_ollama=False)
    assert linker._extract_concepts_from_text("PRIVATE_SENTINEL", source_entity=source) == []
    client.messages.create.assert_not_called()


def test_ollama_helpers_refuse_remote_endpoint(monkeypatch):
    from knowledge_graph.services.linker import _ollama_embed, _ollama_generate
    import urllib.request
    calls = []
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: calls.append(args))
    assert _ollama_embed("PRIVATE", base_url="https://example.invalid") is None
    assert _ollama_generate("PRIVATE", base_url="https://example.invalid") is None
    assert calls == []


def test_local_concept_extraction_works_without_cloud_client(store, monkeypatch):
    from knowledge_graph.services import linker as module
    store.add_entity(Entity(id="f", label="File", name="module", description="Caching design",
                           properties={"domain": "professional"}))
    monkeypatch.setattr(module, "_ollama_generate", lambda *args: "Caching")
    monkeypatch.setattr(module, "_ollama_embed", lambda *args: [1., 0.])
    linker = SemanticLinker(store, use_ollama=True)
    assert len(linker.link_file_to_concepts("f")) == 1
    concept = store.find_by_label("Concept")[0]
    assert concept.properties["domain"] == "professional"
    assert concept.properties["export_policy"] == "local_only"
    assert concept.properties["parent_id"] == "f"
