"""
Knowledge Graph MCP Server

Exposes KnowledgeGraphStore as an MCP server so Claude Code agents can
store findings, retrieve context, and link concepts across sessions.

Usage:
    python -m knowledge_graph.mcp_server

Add to ~/.claude/settings.json:
    {
      "mcpServers": {
        "knowledge-graph": {
          "command": "python",
          "args": ["-m", "knowledge_graph.mcp_server"],
          "cwd": "/path/to/knowledge-graph"
        }
      }
    }
"""

import os
import uuid
from typing import Optional

from mcp.server.fastmcp import FastMCP

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore

# Allow data_dir override via environment variable
data_dir = os.environ.get("KG_DATA_DIR", None)
store = KnowledgeGraphStore(**({"data_dir": data_dir} if data_dir else {}))

mcp = FastMCP(
    "knowledge-graph",
    instructions=(
        "Persistent knowledge graph for storing entities, concepts, decisions, "
        "and agent outputs across sessions. Use kg_add_entity to store findings, "
        "kg_search to retrieve relevant context, kg_add_relation to link concepts."
    ),
)


# ============================================================================
# Search & Retrieval
# ============================================================================


@mcp.tool()
def kg_search(query: str, label: Optional[str] = None, limit: int = 10) -> list[dict]:
    """Search knowledge graph entities by text query.

    Args:
        query: Natural language search query
        label: Optional entity type filter (e.g. "Concept", "Agent", "Decision", "Bug")
        limit: Maximum results to return (default 10)

    Returns:
        List of matching entities with id, name, label, description, tags
    """
    results = store.search_entities(query, label=label, limit=limit)
    return [
        {
            "id": e.id,
            "name": e.name,
            "label": e.label,
            "description": e.description,
            "tags": e.tags,
            "importance_score": e.importance_score,
        }
        for e in results
    ]


@mcp.tool()
def kg_get_entity(entity_id: str) -> dict | None:
    """Get a single entity by ID with full details.

    Args:
        entity_id: The entity ID to retrieve
    """
    entity = store.get_entity(entity_id)
    if not entity:
        return None
    return {
        "id": entity.id,
        "name": entity.name,
        "label": entity.label,
        "description": entity.description,
        "properties": entity.properties,
        "tags": entity.tags,
        "topics": entity.topics,
        "importance_score": entity.importance_score,
        "access_count": entity.access_count,
        "created_at": entity.created_at.isoformat(),
    }


@mcp.tool()
def kg_get_neighbors(entity_id: str, depth: int = 1) -> dict:
    """Get neighboring entities at a given traversal depth.

    Args:
        entity_id: Starting entity ID
        depth: Traversal depth 1-3 (default 1)

    Returns:
        Dict mapping depth level to list of neighbor entities
    """
    depth = max(1, min(depth, 3))
    neighbors = store.get_neighbors(entity_id, depth=depth)
    return {
        str(d): [
            {"id": e.id, "name": e.name, "label": e.label, "description": e.description}
            for e in entities
        ]
        for d, entities in neighbors.items()
    }


@mcp.tool()
def kg_find_path(source_id: str, target_id: str) -> list[dict] | None:
    """Find shortest conceptual path between two entities.

    Args:
        source_id: Starting entity ID
        target_id: Target entity ID

    Returns:
        Ordered list of entities on the path, or null if no path exists
    """
    path = store.find_path(source_id, target_id)
    if not path:
        return None
    return [{"id": e.id, "name": e.name, "label": e.label} for e in path]


@mcp.tool()
def kg_get_stats() -> dict:
    """Get knowledge graph statistics: entity count, relation count, label distribution."""
    return store.get_stats()


# ============================================================================
# Write Operations
# ============================================================================


@mcp.tool()
def kg_add_entity(
    label: str,
    name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    properties: Optional[dict] = None,
    source_app: str = "claude-code",
    importance_score: float = 0.5,
) -> str:
    """Add a new entity to the knowledge graph.

    Valid labels: Concept, Task, Decision, Bug, Agent, Session, AgentOutput,
                  Experiment, Document, Code, File, Folder, Person, Skill, Tool, Event, Tag

    Args:
        label: Entity type from the valid labels list
        name: Short descriptive name
        description: Longer description of the entity
        tags: List of categorical tags
        properties: Arbitrary key-value metadata
        source_app: Source application (default "claude-code")
        importance_score: 0.0-1.0 importance weight (default 0.5)

    Returns:
        The created entity ID
    """
    entity = Entity(
        id=f"{label.lower()}-{uuid.uuid4().hex[:10]}",
        label=label,
        name=name,
        description=description,
        tags=tags or [],
        properties=properties or {},
        source_app=source_app,
        importance_score=max(0.0, min(1.0, importance_score)),
        is_auto_generated=True,
    )
    return store.add_entity(entity)


@mcp.tool()
def kg_add_relation(
    source_id: str,
    target_id: str,
    relation_type: str,
    reason: Optional[str] = None,
    strength: float = 0.7,
    bidirectional: bool = False,
) -> str:
    """Link two entities with a typed relation.

    Valid relation types: contains, depends_on, implements, references, similar_to,
    contradicts, prerequisite_for, learned_from, authored_by, located_in, part_of,
    uses, produces, influenced_by, routed_to, spawned_by, resolves, supersedes, tracked_in

    Args:
        source_id: Source entity ID
        target_id: Target entity ID
        relation_type: Type of relation from the valid types list
        reason: Human-readable explanation of why this relation exists
        strength: 0.0-1.0 relation strength (default 0.7)
        bidirectional: Whether to also create the inverse relation

    Returns:
        The created relation ID
    """
    relation = Relation(
        id=f"rel-{uuid.uuid4().hex[:10]}",
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        strength=max(0.0, min(1.0, strength)),
        connection_reason=reason,
        bidirectional=bidirectional,
        is_auto_generated=True,
    )
    return store.add_relation(relation)


@mcp.tool()
def kg_update_entity(
    entity_id: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    importance_score: Optional[float] = None,
    properties: Optional[dict] = None,
) -> bool:
    """Update fields on an existing entity.

    Args:
        entity_id: ID of entity to update
        description: New description (replaces existing)
        tags: New tags list (replaces existing)
        importance_score: New importance score 0.0-1.0
        properties: Dict of properties to merge into existing properties

    Returns:
        True if updated, False if entity not found
    """
    entity = store.get_entity(entity_id)
    if not entity:
        return False

    if description is not None:
        entity.description = description
    if tags is not None:
        entity.tags = tags
    if importance_score is not None:
        entity.importance_score = max(0.0, min(1.0, importance_score))
    if properties:
        entity.properties.update(properties)

    store.update_entity(entity)
    return True


@mcp.tool()
def kg_delete_entity(entity_id: str) -> bool:
    """Delete an entity and all its relations.

    Args:
        entity_id: ID of entity to delete

    Returns:
        True if deleted, False if not found
    """
    return store.delete_entity(entity_id)


if __name__ == "__main__":
    mcp.run()
