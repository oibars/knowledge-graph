# Knowledge Graph

Hybrid knowledge graph library for AI agent memory. NetworkX for in-memory graph operations, SQLite for persistence.

## Features
- Entity/relation CRUD with typed models
- Graph traversal (shortest path, neighbors, subgraph extraction)
- Semantic linking via embeddings
- SQLite persistence with snapshot support
- FastAPI routes for HTTP access

## Install
```bash
pip install -e .
```

## Usage
```python
from knowledge_graph.models import Entity, Relation, RELATION_TYPES
from knowledge_graph.services.graph_store import KnowledgeGraphStore

store = KnowledgeGraphStore(data_dir="./data")
entity = Entity(id="e1", label="Concept", name="Authentication")
store.add_entity(entity)
```

## API Routes
Mount the FastAPI router:
```python
from knowledge_graph.routes.knowledge import router
app.include_router(router, prefix="/knowledge")
```

## Dependencies
- NetworkX (graph algorithms)
- SQLite (persistence)
- numpy (similarity calculations)
- structlog (logging)
