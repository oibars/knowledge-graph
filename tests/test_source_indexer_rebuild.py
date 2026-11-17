"""Source versions are distinct from embedded-content identity."""
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.source_indexer import index_source_file


def test_full_file_source_version_keeps_indexing_idempotent(tmp_path):
    path = tmp_path / "source.md"
    path.write_text("---\ntitle: Chosen architecture\nsource: https://example.com/paper\n---\n\n## Thesis\nUse SQLite as canonical state.\n\n## Key claims\n- A validated archive can rebuild the graph.\n")
    store = KnowledgeGraphStore(str(tmp_path / "db"), enable_snapshots=False)
    assert index_source_file(path, store)[0] == 2
    assert index_source_file(path, store) == (0, 0)
    doc = store.get_entity("doc-source")
    assert doc.properties["source_version"]
    assert doc.content_hash == doc.compute_content_hash()


def test_claims_from_two_sources_keep_both_provenances(tmp_path):
    store = KnowledgeGraphStore(str(tmp_path / "db"), enable_snapshots=False)
    for filename in ("first", "second"):
        path = tmp_path / f"{filename}.md"
        path.write_text("---\ntitle: Source\n---\n\n## Key claims\n- Same claim made in two different sources.\n")
        index_source_file(path, store)
    claims = store.find_by_label("Concept")
    assert len(claims) == 2
    assert {e.properties["parent_id"] for e in claims} == {"doc-first", "doc-second"}
    assert all(e.properties["export_policy"] == "local_only" for e in claims)
