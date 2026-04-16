"""Tests for the agent indexer."""

import pytest
from pathlib import Path

from knowledge_graph.agent_indexer import parse_frontmatter, index_agents
from knowledge_graph.services.graph_store import KnowledgeGraphStore


SAMPLE_AGENT_MD = """\
---
name: Test Engineer
description: Specialist in writing and reviewing automated tests. Enforces coverage requirements.
color: green
emoji: 🧪
---

# Test Engineer Agent

You are Test Engineer...
"""

SAMPLE_NO_NAME_MD = """\
---
color: blue
---

# Some file without a name field.
"""


@pytest.fixture
def agents_dir(tmp_path):
    d = tmp_path / "agents"
    d.mkdir()
    (d / "testing-test-engineer.md").write_text(SAMPLE_AGENT_MD)
    (d / "no-name.md").write_text(SAMPLE_NO_NAME_MD)
    return d


@pytest.fixture
def store(tmp_path):
    return KnowledgeGraphStore(data_dir=str(tmp_path / "kg"), enable_snapshots=False)


class TestParseFrontmatter:
    def test_parses_name_and_description(self):
        fm = parse_frontmatter(SAMPLE_AGENT_MD)
        assert fm["name"] == "Test Engineer"
        assert "Specialist in writing" in fm["description"]

    def test_no_frontmatter_returns_empty(self):
        assert parse_frontmatter("No frontmatter here") == {}

    def test_missing_field_not_in_result(self):
        fm = parse_frontmatter(SAMPLE_NO_NAME_MD)
        assert "name" not in fm


class TestIndexAgents:
    def test_indexes_valid_agents(self, agents_dir, store, tmp_path):
        added, skipped = index_agents(agents_dir=agents_dir, store=store)
        assert added == 1   # only the one with a name
        assert skipped == 1  # the no-name file

    def test_indexed_entity_has_correct_fields(self, agents_dir, store):
        index_agents(agents_dir=agents_dir, store=store)
        agents = store.find_by_label("Agent")
        assert len(agents) == 1
        agent = agents[0]
        assert agent.name == "Test Engineer"
        assert "testing" in agent.tags
        assert "claude-code" in agent.tags
        assert agent.is_auto_generated is True

    def test_skips_already_indexed_unchanged(self, agents_dir, store):
        added1, _ = index_agents(agents_dir=agents_dir, store=store)
        added2, skipped2 = index_agents(agents_dir=agents_dir, store=store)
        assert added1 == 1
        assert added2 == 0   # no changes → all skipped
        assert skipped2 == 2  # both files skipped (one no-name, one unchanged)

    def test_dry_run_does_not_write(self, agents_dir, tmp_path):
        store = KnowledgeGraphStore(data_dir=str(tmp_path / "kg2"), enable_snapshots=False)
        added, _ = index_agents(agents_dir=agents_dir, store=store, dry_run=True)
        assert added == 1
        assert len(store.get_all_entities()) == 0  # nothing written

    def test_missing_agents_dir_returns_zeros(self, store, tmp_path):
        added, skipped = index_agents(agents_dir=tmp_path / "nonexistent", store=store)
        assert added == 0
        assert skipped == 0
