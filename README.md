# Knowledge Graph

Hybrid knowledge graph library for AI agent memory. NetworkX for in-memory graph operations, SQLite for persistence. Exposes an MCP server so Claude Code agents can store and retrieve context across sessions.

## Features

- Entity/Relation CRUD with typed models and controlled vocabularies
- Graph traversal (shortest path, neighbors, subgraph extraction)
- Semantic linking via embeddings (cosine similarity)
- LLM-powered concept extraction via Claude API (`claude-haiku-4-5`)
- SQLite persistence with snapshot support
- FastAPI routes for HTTP access
- **MCP server** — queryable by Claude Code agents via `kg_search`, `kg_add_entity`, `kg_add_relation`, etc.
- **Agent indexer** — indexes `~/.claude/agents/*.md` as `Entity(label="Agent")` for semantic routing

## Install

```bash
pip install -e .
# or with uv
uv pip install -e .
```

## MCP Server (Claude Code integration)

Start the MCP server:

```bash
python -m knowledge_graph.mcp_server
# or after install:
kg-mcp
```

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "python",
      "args": ["-m", "knowledge_graph.mcp_server"],
      "cwd": "/path/to/knowledge-graph",
      "env": {
        "KG_DATA_DIR": "/home/youruser/.knowledge-graph/data"
      }
    }
  }
}
```

### Available MCP tools

| Tool | Description |
|------|-------------|
| `kg_search` | Search entities by text query, with optional label filter |
| `kg_get_entity` | Retrieve a single entity by ID |
| `kg_get_neighbors` | Get neighboring entities at depth 1-3 |
| `kg_find_path` | Find shortest path between two entities |
| `kg_get_stats` | Graph statistics |
| `kg_add_entity` | Create a new entity |
| `kg_add_relation` | Link two entities with a typed relation |
| `kg_update_entity` | Update fields on an existing entity |
| `kg_delete_entity` | Delete an entity and its relations |

### Entity labels

`Concept` `Task` `Decision` `Bug` `Agent` `Session` `AgentOutput` `Experiment`
`Document` `Code` `File` `Folder` `Person` `Skill` `Tool` `Event` `Tag`

### Relation types

`contains` `depends_on` `implements` `references` `similar_to` `contradicts`
`prerequisite_for` `learned_from` `authored_by` `located_in` `part_of` `uses`
`produces` `influenced_by` `routed_to` `spawned_by` `resolves` `supersedes` `tracked_in`

## Agent Indexer

Index all installed Claude Code agents as `Agent` entities for semantic routing:

```bash
python -m knowledge_graph.agent_indexer
# or after install:
kg-index-agents

# Custom agents directory
kg-index-agents --agents-dir /path/to/agents

# Preview without writing
kg-index-agents --dry-run
```

After indexing, agents are searchable via MCP:

```
kg_search("debug a production memory leak", label="Agent")
→ Returns: engineering-incident-response-commander, engineering-sre, testing-evidence-collector...
```

## Python API

```python
from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.services.linker import SemanticLinker
import anthropic

# Default data dir: ~/.knowledge-graph/data
store = KnowledgeGraphStore()

# Add entities
entity = Entity(id="e1", label="Concept", name="Authentication", tags=["security"])
store.add_entity(entity)

# Add relations
rel = Relation(id="r1", source_id="e1", target_id="e2", relation_type="depends_on")
store.add_relation(rel)

# Search
results = store.search_entities("authentication security")

# Semantic linking with Claude API
linker = SemanticLinker(store, llm_client=anthropic.Anthropic())
linker.link_file_to_concepts("e1", extract_concepts=True)

# Graph algorithms
path = store.find_path("e1", "e5")
neighbors = store.get_neighbors("e1", depth=2)
centrality = store.get_centrality("e1", metric="pagerank")
```

## FastAPI Routes

Mount the HTTP router in your FastAPI app:

```python
from knowledge_graph.routes.knowledge import router
app.include_router(router, prefix="/knowledge")
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `KG_DATA_DIR` | `~/.knowledge-graph/data` | SQLite and snapshot storage directory |

## Run tests

```bash
pytest
pytest -v tests/test_entity_crud.py   # specific module
```

## Dependencies

- **NetworkX** — graph algorithms (PageRank, shortest path, community detection)
- **SQLite** — persistence (no server required)
- **numpy** — embedding similarity calculations
- **Anthropic SDK** — LLM concept extraction via `claude-haiku-4-5`
- **MCP** — Model Context Protocol server interface
- **structlog** — structured logging
- **FastAPI** — HTTP API layer
