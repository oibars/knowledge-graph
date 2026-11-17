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
import sys
import uuid
from collections import deque
from typing import Optional

import structlog
from mcp.server.fastmcp import FastMCP

# stdio transport frames JSON-RPC on stdout — ALL logging must go to stderr,
# and this must be configured before the store logs its init lines.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore, entity_corpus
from knowledge_graph.privacy import export_allowed, export_properties

# Allow data_dir override via environment variable
data_dir = os.environ.get("KG_DATA_DIR", None)
store = KnowledgeGraphStore(**({"data_dir": data_dir} if data_dir else {}))

mcp = FastMCP(
    "knowledge-graph",
    instructions=(
        "Persistent knowledge graph for storing entities, concepts, decisions, "
        "and agent outputs across sessions. Use kg_add_entity to store findings, "
        "kg_search to retrieve relevant context, kg_add_relation to link concepts. "
        "With KG_EXPORT_ONLY=1, reads include only approved cloud exports and writes "
        "are disabled. Use kg_policy for the active boundary; raw recordings stay local."
    ),
)


def _export_only() -> bool:
    return os.environ.get("KG_EXPORT_ONLY", "0") == "1"


def _read_allowed(entity: Entity) -> bool:
    return not _export_only() or export_allowed(entity)


def _require_local_write():
    if _export_only():
        raise PermissionError("This KG_EXPORT_ONLY connector is read-only. Edit or approve selected facts through the local memory workflow.")


def _policy_blocked() -> dict:
    return {"policy_blocked": True, "message": "This connector exposes approved cloud exports only. Use local recall for unapproved source material."}


_APPROVED_GRAPH_CACHE = {}


def _approved_graph():
    """Build traversal exclusively over approved endpoints, with no private bridge.

    Cached per store revision: neighbor/path tools called in sequence must not
    rebuild and deepcopy the whole approved subgraph every time.
    """
    store.refresh_if_stale()
    key = (id(store), store._known_version, len(store._entities), len(store._relations))
    cached = _APPROVED_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    entities = {e.id: e for e in store.get_all_entities() if export_allowed(e)}
    outgoing = {eid: set() for eid in entities}
    undirected = {eid: set() for eid in entities}
    for r in store.get_all_relations():
        if r.source_id in entities and r.target_id in entities:
            outgoing[r.source_id].add((r.target_id, r.relation_type))
            undirected[r.source_id].add(r.target_id)
            undirected[r.target_id].add(r.source_id)
    _APPROVED_GRAPH_CACHE.clear()
    _APPROVED_GRAPH_CACHE[key] = (entities, outgoing, undirected)
    return _APPROVED_GRAPH_CACHE[key]


def _entity_hit(entity: Entity) -> dict:
    """Local recall always carries its source and independent export decision."""
    hit = {
        "id": entity.id, "name": entity.name, "label": entity.label,
        "description": entity.description, "tags": entity.tags,
        "importance_score": entity.importance_score, "source_app": entity.source_app,
        "source_url": entity.source_url, "source_file_path": entity.source_file_path,
        "created_at": entity.created_at.isoformat(), "updated_at": entity.updated_at.isoformat(),
        "properties": entity.properties, "corpus": entity_corpus(entity),
        "domain": entity.properties.get("domain", "unknown"),
        "export_policy": entity.properties.get("export_policy", "local_only"),
        "export_allowed": export_allowed(entity),
        "publish_allowed": export_allowed(entity, purpose="publish"),
    }
    if _export_only():
        hit["properties"] = export_properties(entity)
        for field in ("updated_at", "importance_score"):
            hit.pop(field, None)
    return hit


# ============================================================================
# Search & Retrieval
# ============================================================================


@mcp.tool()
def kg_search(
    query: str,
    label: Optional[str] = None,
    limit: int = 10,
    tags: Optional[list[str]] = None,
    source_app: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    corpus: str = "memory",
) -> list[dict]:
    """Hybrid search (lexical + semantic + importance + recency) over the graph.

    Args:
        query: Natural language search query
        label: Optional entity type filter (e.g. "Concept", "Agent", "Decision", "Bug")
        limit: Maximum results to return (default 10)
        tags: Only entities carrying ALL of these tags
        source_app: Only entities from this source (e.g. "fieldy", "plaud", "chrome")
        created_after: ISO date — only entities created on/after (e.g. "2026-06-01")
        created_before: ISO date — only entities created on/before
        corpus: memory (default), reference (news/trends), or all

    Returns:
        List of matching entities with id, name, label, description, tags
    """
    store.refresh_if_stale()
    results = store.search_entities(
        query,
        label=label,
        limit=store.get_stats()["entity_count"] if _export_only() else limit,
        tags=tags,
        source_app=source_app,
        created_after=created_after,
        created_before=created_before,
        corpus=corpus,
    )
    return [_entity_hit(e) for e in results if _read_allowed(e)][:max(0, limit)]


@mcp.tool()
def kg_recent(
    days: float = 7,
    label: Optional[str] = None,
    source_app: Optional[str] = None,
    limit: int = 50,
    corpus: str = "memory",
) -> list[dict]:
    """What changed: entities created in the last N days, newest first. No query
    needed — use for "what's new / what changed since last week" style recall.

    Args:
        days: Look-back window in days (default 7)
        label: Optional entity type filter (e.g. "Event", "Document", "Decision")
        source_app: Only entities from this source (e.g. "fieldy", "plaud")
        limit: Maximum results (default 50)

    Returns:
        List of entities (id, name, label, description, tags, source_app,
        created_at), newest first
    """
    store.refresh_if_stale()
    return [_entity_hit(e) for e in store.recent_entities(
        days=days, label=label, source_app=source_app,
        limit=store.get_stats()["entity_count"] if _export_only() else limit, corpus=corpus)
        if _read_allowed(e)][:max(0, limit)]


@mcp.tool()
def kg_get_entity(entity_id: str) -> dict | None:
    """Get a single entity by ID with full details.

    Args:
        entity_id: The entity ID to retrieve
    """
    store.refresh_if_stale()
    entity = store.get_entity(entity_id)
    if _export_only() and (entity is None or not _read_allowed(entity)):
        return _policy_blocked()
    if not entity:
        return None
    hit = {**_entity_hit(entity), "topics": entity.topics}
    if not _export_only():
        hit["access_count"] = entity.access_count
    return hit


@mcp.tool()
def kg_get_neighbors(entity_id: str, depth: int = 1, relation_type: Optional[str] = None) -> dict:
    """Get neighboring entities at a given traversal depth.

    Args:
        entity_id: Starting entity ID
        depth: Traversal depth 1-3 (default 1)
        relation_type: Only traverse edges of this type (e.g. "depends_on", "part_of")

    Returns:
        Dict mapping depth level to list of neighbor entities
    """
    store.refresh_if_stale()
    depth = max(1, min(depth, 3))
    if _export_only():
        entities, outgoing, _ = _approved_graph()
        neighbors = {level: [] for level in range(1, depth + 1)}
        if entity_id not in entities:
            return {str(level): [] for level in neighbors}
        visited, frontier = {entity_id}, {entity_id}
        for level in range(1, depth + 1):
            next_level = {target for start in frontier for target, kind in outgoing[start]
                          if target not in visited and (relation_type is None or kind == relation_type)}
            neighbors[level] = [entities[eid] for eid in sorted(next_level)]
            visited.update(next_level)
            frontier = next_level
    else:
        neighbors = store.get_neighbors(entity_id, relation_type=relation_type, depth=depth)
    return {
        str(d): [
            _entity_hit(e)
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
    store.refresh_if_stale()
    if _export_only():
        entities, _, undirected = _approved_graph()
        if source_id not in entities or target_id not in entities:
            return None
        queue, visited, path = deque([[source_id]]), {source_id}, None
        while queue:
            ids = queue.popleft()
            if ids[-1] == target_id:
                path = [entities[eid] for eid in ids]
                break
            if len(ids) > 5:
                continue
            for neighbor in sorted(undirected[ids[-1]]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(ids + [neighbor])
    else:
        path = store.find_path(source_id, target_id)
    if not path:
        return None
    return [_entity_hit(e) for e in path]


@mcp.tool()
def kg_recall_actions(
    query: str = "",
    label: Optional[str] = None,
    owner: Optional[str] = None,
    status: Optional[str] = None,
    source_app: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Recall source-linked commitments and decisions from memory.

    Only Task/Decision records with quoted evidence are returned. Owner/status
    match exactly, ignoring case. Evidence and timestamps accompany each hit;
    absent evidence is never filled in. Dates filter occurrence creation time.
    """
    if label is not None and label not in {"Task", "Decision"}:
        raise ValueError("label must be Task or Decision")
    store.refresh_if_stale()
    if query.strip():
        candidates = store.search_entities(query, source_app=source_app,
                                           created_after=created_after, created_before=created_before,
                                           corpus="memory", limit=max(store.get_stats()["entity_count"], limit))
    else:
        # Reuse date parsing/filter semantics without relevance gating.
        from datetime import datetime
        from knowledge_graph.models.timestamps import as_utc
        after = as_utc(datetime.fromisoformat(created_after)) if created_after else None
        before = as_utc(datetime.fromisoformat(created_before)) if created_before else None
        candidates = [e for e in store.get_all_entities()
                      if entity_corpus(e) == "memory" and (source_app is None or e.source_app == source_app)
                      and (after is None or as_utc(e.created_at) >= after)
                      and (before is None or as_utc(e.created_at) <= before)]
        candidates.sort(key=lambda e: as_utc(e.updated_at), reverse=True)
    hits = []
    for entity in candidates:
        if not _read_allowed(entity):
            continue
        if entity.label not in ({label} if label else {"Task", "Decision"}):
            continue
        props = entity.properties
        if owner is not None and str(props.get("owner", "")).casefold() != owner.casefold():
            continue
        if status is not None and str(props.get("status", "")).casefold() != status.casefold():
            continue
        evidence = props.get("evidence")
        if not isinstance(evidence, list) or not any(
            isinstance(item, dict) and item.get("quote") and (item.get("source_id") or item.get("source_url"))
            for item in evidence
        ):
            continue
        hits.append(_entity_hit(entity))
    return hits[:max(0, min(limit, 100))]


@mcp.tool()
def kg_policy() -> dict:
    """Explain this connector's read, write and export boundary."""
    return {"read_mode": "approved_exports" if _export_only() else "local_recall",
            "writes_allowed": not _export_only(), "default_corpus": "memory",
            "message": "Raw and unapproved material stays in local recall. Cloud connectors require content-bound cloud approval; publication requires separate approval."}


@mcp.tool()
def kg_get_stats() -> dict:
    """Get knowledge graph statistics: entity count, relation count, label distribution.

    In cloud export mode the counts cover approved exports only, so an aggregate
    response cannot reveal how much private material exists.
    """
    store.refresh_if_stale()
    stats = store.get_stats()
    if not _export_only():
        return stats
    approved = {e.id: e for e in store.get_all_entities() if export_allowed(e)}
    distribution: dict[str, int] = {}
    for entity in approved.values():
        distribution[entity.label] = distribution.get(entity.label, 0) + 1
    relations = sum(1 for r in store.get_all_relations()
                    if r.source_id in approved and r.target_id in approved)
    return {"entity_count": len(approved), "relation_count": relations,
            "label_distribution": distribution, "read_mode": "approved_exports",
            "scope": "Aggregate counts cover approved cloud exports only."}


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
    entity_id: Optional[str] = None,
    source_url: Optional[str] = None,
    source_file_path: Optional[str] = None,
) -> str:
    """Add a new entity to the knowledge graph.

    Valid labels: Concept, Task, Decision, Bug, Agent, Session, AgentOutput,
                  Experiment, Document, Chunk, Code, File, Folder, Person, Skill, Tool, Event, Tag

    Args:
        label: Entity type from the valid labels list
        name: Short descriptive name
        description: Longer description of the entity
        tags: List of categorical tags
        properties: Arbitrary key-value metadata
        source_app: Source application (default "claude-code")
        importance_score: 0.0-1.0 importance weight (default 0.5)
        entity_id: Stable source-derived ID for repeat ingestion
        source_url: Original source link
        source_file_path: Local archived source path

    Returns:
        The created entity ID
    """
    _require_local_write()
    store.refresh_if_stale()
    entity = Entity(
        id=entity_id or f"{label.lower()}-{uuid.uuid4().hex[:10]}",
        label=label,
        name=name,
        description=description,
        tags=tags or [],
        properties=properties or {},
        source_app=source_app,
        source_url=source_url,
        source_file_path=source_file_path,
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
    _require_local_write()
    store.refresh_if_stale()
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
    _require_local_write()
    store.refresh_if_stale()
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
    _require_local_write()
    store.refresh_if_stale()
    return store.delete_entity(entity_id)


@mcp.tool()
def kg_find_by_tag(tag: str, label: Optional[str] = None, limit: int = 20) -> list[dict]:
    """List entities carrying a specific tag.

    Args:
        tag: Tag to match exactly (e.g. "signal-high", "bookmark")
        label: Optional entity type filter
        limit: Maximum results (default 20)
    """
    store.refresh_if_stale()
    results = store.find_by_tag(tag)
    if label:
        results = [e for e in results if e.label == label]
    results = [e for e in results if _read_allowed(e)]
    results.sort(key=lambda e: e.importance_score, reverse=True)
    return [
        _entity_hit(e)
        for e in results[:limit]
    ]


@mcp.tool()
def kg_get_relations(entity_id: str, direction: str = "both") -> list[dict]:
    """List an entity's typed relations.

    Args:
        entity_id: Entity ID
        direction: "out" (this → others), "in" (others → this), or "both"
    """
    store.refresh_if_stale()
    if _export_only():
        entities, _, _ = _approved_graph()
        if entity_id not in entities:
            return []
        relations = [r for r in store.get_entity_relations(entity_id, direction=direction)
                     if r.source_id in entities and r.target_id in entities]
    else:
        relations = store.get_entity_relations(entity_id, direction=direction)
    return [
        {
            "id": "export-rel-" + uuid.uuid5(uuid.NAMESPACE_URL, r.id).hex if _export_only() else r.id,
            "source_id": r.source_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type,
            "strength": r.strength,
            "reason": None if _export_only() else r.connection_reason,
        }
        for r in relations
    ]


@mcp.tool()
def kg_find_similar(entity_id: str, limit: int = 5) -> list[dict]:
    """Find semantically similar entities via stored embeddings.

    Args:
        entity_id: Entity to compare against (must have an embedding)
        limit: Maximum results (default 5)
    """
    store.refresh_if_stale()
    if _export_only():
        entity = store.get_entity(entity_id)
        if entity is None or not _read_allowed(entity):
            return []
    similar = store.find_similar_entities(entity_id, limit=store.get_stats()["entity_count"] if _export_only() else limit)
    return [
        {**_entity_hit(e), "similarity": round(score, 3)}
        for e, score in similar if _read_allowed(e)
    ][:max(0, limit)]


@mcp.tool()
def kg_add_entities(entities: list[dict]) -> list[str]:
    """Bulk-add entities in one call. Each dict: {label, name, description?, tags?,
    properties?, source_app?, importance_score?, id?, source_url?, source_file_path?}.
    Unanchored semantic entities dedupe by label and content; occurrences preserve IDs.

    Returns:
        List of entity ids in input order.
    """
    _require_local_write()
    store.refresh_if_stale()
    ids = []
    for spec in entities:
        entity = Entity(
            id=spec.get("id") or f"{str(spec['label']).lower()}-{uuid.uuid4().hex[:10]}",
            label=spec["label"],
            name=spec["name"],
            description=spec.get("description"),
            tags=spec.get("tags") or [],
            properties=spec.get("properties") or {},
            source_app=spec.get("source_app", "claude-code"),
            source_url=spec.get("source_url"),
            source_file_path=spec.get("source_file_path"),
            importance_score=max(0.0, min(1.0, float(spec.get("importance_score", 0.5)))),
            is_auto_generated=True,
        )
        ids.append(store.add_entity(entity))
    return ids


def main():
    """Console entry point (`kg-mcp` in pyproject)."""
    mcp.run()


if __name__ == "__main__":
    main()
