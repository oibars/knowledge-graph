"""
Knowledge Graph API Routes
FastAPI endpoints for knowledge graph operations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from knowledge_graph.models import Entity, Relation, RELATION_TYPES, ENTITY_LABELS
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.services.linker import SemanticLinker

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# ============================================================================
# Request/Response Models
# ============================================================================

class EntityCreate(BaseModel):
    label: str
    name: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source_file_path: Optional[str] = None
    source_app: Optional[str] = None


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    importance_score: Optional[float] = None


class RelationCreate(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    strength: float = 0.5
    bidirectional: bool = False
    connection_reason: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntityResponse(BaseModel):
    id: str
    label: str
    name: str
    description: Optional[str]
    properties: Dict[str, Any]
    tags: List[str]
    topics: List[str]
    importance_score: float
    confidence_score: float
    created_at: str
    updated_at: str


class RelationResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    bidirectional: bool
    connection_reason: Optional[str]
    created_at: str


class SearchRequest(BaseModel):
    query: str
    label: Optional[str] = None
    limit: int = 10


class PathRequest(BaseModel):
    from_id: str
    to_id: str


class AutoLinkRequest(BaseModel):
    entity_ids: List[str]
    create_links: bool = True


# ============================================================================
# Dependencies
# ============================================================================

def get_kg_store() -> KnowledgeGraphStore:
    """Dependency to get knowledge graph store."""
    return KnowledgeGraphStore()


# ============================================================================
# Entity Endpoints
# ============================================================================

@router.post("/entities", response_model=EntityResponse)
async def create_entity(
    data: EntityCreate,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Create a new entity in the knowledge graph."""
    entity_id = f"{data.label.lower()}_{data.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    entity = Entity(
        id=entity_id,
        label=data.label,
        name=data.name,
        description=data.description,
        properties=data.properties,
        tags=data.tags,
        topics=data.topics,
        source_url=data.source_url,
        source_file_path=data.source_file_path,
        source_app=data.source_app
    )
    
    kg.add_entity(entity)
    
    return EntityResponse(**entity.to_dict())


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: str,
    include_neighbors: bool = False,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Get an entity by ID with optional neighbor information."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return EntityResponse(**entity.to_dict())


@router.get("/entities")
async def list_entities(
    label: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = 0,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """List entities with optional filtering."""
    if label:
        entities = kg.find_by_label(label)
    elif tag:
        entities = kg.find_by_tag(tag)
    else:
        entities = kg.get_all_entities()
    
    total = len(entities)
    entities = entities[offset:offset + limit]
    
    return {
        "entities": [EntityResponse(**e.to_dict()) for e in entities],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.patch("/entities/{entity_id}")
async def update_entity(
    entity_id: str,
    data: EntityUpdate,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Update an existing entity."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    # Update fields
    if data.name is not None:
        entity.name = data.name
    if data.description is not None:
        entity.description = data.description
    if data.properties is not None:
        entity.properties.update(data.properties)
    if data.tags is not None:
        entity.tags = data.tags
    if data.topics is not None:
        entity.topics = data.topics
    if data.importance_score is not None:
        entity.importance_score = data.importance_score
    
    entity.updated_at = datetime.now()
    kg.update_entity(entity)
    
    return EntityResponse(**entity.to_dict())


@router.delete("/entities/{entity_id}")
async def delete_entity(
    entity_id: str,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Delete an entity and all its relations."""
    success = kg.delete_entity(entity_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return {"message": "Entity deleted", "id": entity_id}


@router.post("/entities/search")
async def search_entities(
    data: SearchRequest,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Search entities by query."""
    results = kg.search_entities(
        query=data.query,
        label=data.label,
        limit=data.limit
    )
    
    return {
        "results": [EntityResponse(**e.to_dict()) for e in results],
        "query": data.query,
        "count": len(results)
    }


# ============================================================================
# Relation Endpoints
# ============================================================================

@router.post("/relations", response_model=RelationResponse)
async def create_relation(
    data: RelationCreate,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Create a new relation between entities."""
    relation_id = f"rel_{data.source_id}_{data.target_id}_{data.relation_type}"
    
    relation = Relation(
        id=relation_id,
        source_id=data.source_id,
        target_id=data.target_id,
        relation_type=data.relation_type,
        strength=data.strength,
        bidirectional=data.bidirectional,
        connection_reason=data.connection_reason,
        properties=data.properties
    )
    
    try:
        kg.add_relation(relation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return RelationResponse(**relation.to_dict())


@router.get("/relations/{relation_id}", response_model=RelationResponse)
async def get_relation(
    relation_id: str,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Get a relation by ID."""
    relation = kg.get_relation(relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    return RelationResponse(**relation.to_dict())


@router.get("/entities/{entity_id}/relations")
async def get_entity_relations(
    entity_id: str,
    direction: str = "both",
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Get all relations for an entity."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    relations = kg.get_entity_relations(entity_id, direction)
    
    return {
        "entity_id": entity_id,
        "relations": [RelationResponse(**r.to_dict()) for r in relations],
        "count": len(relations)
    }


@router.delete("/relations/{relation_id}")
async def delete_relation(
    relation_id: str,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Delete a relation."""
    success = kg.delete_relation(relation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Relation not found")
    
    return {"message": "Relation deleted", "id": relation_id}


# ============================================================================
# Graph Traversal Endpoints
# ============================================================================

@router.post("/path")
async def find_path(
    data: PathRequest,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Find shortest path between two entities."""
    path = kg.find_path(data.from_id, data.to_id)
    
    if not path:
        return {
            "from_id": data.from_id,
            "to_id": data.to_id,
            "path": [],
            "found": False
        }
    
    return {
        "from_id": data.from_id,
        "to_id": data.to_id,
        "path": [EntityResponse(**e.to_dict()) for e in path],
        "found": True,
        "length": len(path) - 1
    }


@router.get("/entities/{entity_id}/neighbors")
async def get_neighbors(
    entity_id: str,
    depth: int = Query(default=1, ge=1, le=3),
    relation_type: Optional[str] = None,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Get neighbors at specified depth."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    neighbors = kg.get_neighbors(entity_id, depth=depth, relation_type=relation_type)
    
    return {
        "entity_id": entity_id,
        "depth": depth,
        "neighbors": {
            str(d): [EntityResponse(**e.to_dict()) for e in entities]
            for d, entities in neighbors.items()
        }
    }


@router.get("/entities/{entity_id}/similar")
async def find_similar(
    entity_id: str,
    limit: int = Query(default=5, le=20),
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Find similar entities based on embeddings."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    similar = kg.find_similar_entities(entity_id, limit)
    
    return {
        "entity_id": entity_id,
        "similar": [
            {
                "entity": EntityResponse(**e.to_dict()),
                "similarity": score
            }
            for e, score in similar
        ]
    }


# ============================================================================
# Semantic Linking Endpoints
# ============================================================================

@router.post("/auto-link")
async def auto_link_entities(
    data: AutoLinkRequest,
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Automatically discover and create links for entities."""
    linker = SemanticLinker(kg)
    
    results = linker.batch_link_entities(
        data.entity_ids,
        create_links=data.create_links
    )
    
    return {
        "results": results,
        "entities_processed": len(data.entity_ids)
    }


@router.post("/entities/{entity_id}/suggest-connections")
async def suggest_connections(
    entity_id: str,
    limit: int = Query(default=5, le=10),
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Suggest potential connections for an entity."""
    entity = kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    linker = SemanticLinker(kg)
    suggestions = linker.suggest_connections(entity_id, limit)
    
    return {
        "entity_id": entity_id,
        "suggestions": [
            {
                "entity_id": e.id,
                "entity_name": e.name,
                "score": score,
                "reason": reason
            }
            for e, score, reason in suggestions
        ]
    }


# ============================================================================
# Statistics and Export
# ============================================================================

@router.get("/stats")
async def get_stats(
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Get knowledge graph statistics."""
    stats = kg.get_stats()
    
    return {
        "statistics": stats,
        "entity_labels": list(ENTITY_LABELS.keys()),
        "relation_types": list(RELATION_TYPES.keys())
    }


@router.get("/export/graphml")
async def export_graphml(
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Export knowledge graph to GraphML format."""
    path = kg.export_graphml()
    
    return {
        "export_path": path,
        "format": "graphml",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/snapshot/create")
async def create_snapshot(
    kg: KnowledgeGraphStore = Depends(get_kg_store)
):
    """Create a snapshot of the knowledge graph."""
    path = kg.create_snapshot()
    
    return {
        "snapshot_path": path,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Types and Schema
# ============================================================================

@router.get("/types/entities")
async def get_entity_types():
    """Get available entity types."""
    return {
        "types": [
            {"name": name, "description": desc}
            for name, desc in ENTITY_LABELS.items()
        ]
    }


@router.get("/types/relations")
async def get_relation_types():
    """Get available relation types."""
    return {
        "types": [
            {"name": name, "description": desc}
            for name, desc in RELATION_TYPES.items()
        ]
    }
