"""
Knowledge Graph Models for SwissPy
Entity and Relation dataclasses for graph-based knowledge storage.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import json


# Controlled vocabulary for relation types
RELATION_TYPES = {
    "contains": "Parent-child containment",
    "depends_on": "Dependency relationship",
    "implements": "Code/file implements concept",
    "references": "Cites or links to",
    "similar_to": "Semantic similarity",
    "contradicts": "Opposing viewpoints",
    "prerequisite_for": "Must come before",
    "learned_from": "Knowledge acquisition source",
    "authored_by": "Creator relationship",
    "located_in": "Physical/logical location",
    "part_of": "Component relationship",
    "uses": "Utilization relationship",
    "produces": "Output relationship",
    "influenced_by": "Inspiration or derivation",
}


# Entity type labels
ENTITY_LABELS = {
    "File": "File system entity",
    "Folder": "Directory container",
    "Concept": "Abstract concept or idea",
    "Task": "Actionable task or workflow",
    "Person": "Human entity",
    "Skill": "Executable capability",
    "Tool": "MCP tool or external tool",
    "Document": "Text document or note",
    "Code": "Code snippet or module",
    "Event": "Temporal occurrence",
    "Tag": "Categorical label",
}


@dataclass
class Entity:
    """
    A node in the knowledge graph representing any entity.
    
    Entities can be files, concepts, tasks, people, skills, etc.
    Each entity has a unique ID, label (type), and flexible properties.
    """
    
    # Core identification
    id: str
    label: str  # Entity type from ENTITY_LABELS
    name: str
    
    # Content and properties
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic
    embedding: Optional[List[float]] = None
    topics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Source tracking
    source_url: Optional[str] = None
    source_file_path: Optional[str] = None
    source_app: Optional[str] = None
    source_user: Optional[str] = None
    
    # Metadata
    importance_score: float = 0.5  # 0.0 to 1.0
    confidence_score: float = 1.0  # Confidence in entity existence
    
    # Access tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Content hash for deduplication
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate content hash if not provided."""
        if not self.content_hash and (self.name or self.description):
            content = f"{self.name}:{self.description or ''}"
            self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "id": self.id,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "embedding": self.embedding,
            "topics": self.tags,
            "tags": self.tags,
            "source_url": self.source_url,
            "source_file_path": self.source_file_path,
            "source_app": self.source_app,
            "source_user": self.source_user,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "content_hash": self.content_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary."""
        # Parse datetime strings
        for field_name in ["created_at", "updated_at", "last_accessed"]:
            if field_name in data and data[field_name]:
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def add_topic(self, topic: str):
        """Add a topic if not already present."""
        if topic not in self.topics:
            self.topics.append(topic)
            self.updated_at = datetime.now()
    
    def set_property(self, key: str, value: Any):
        """Set a property value."""
        self.properties[key] = value
        self.updated_at = datetime.now()


@dataclass
class Relation:
    """
    A typed connection between two entities in the knowledge graph.
    
    Relations form the edges of the graph, connecting entities with
    semantic meaning and optional strength scores.
    """
    
    # Core identification
    id: str
    source_id: str  # ID of source entity
    target_id: str  # ID of target entity
    
    # Relation type and properties
    relation_type: str  # From RELATION_TYPES
    strength: float = 0.5  # 0.0 to 1.0 semantic strength
    
    # Additional metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False  # If true, reverse relation is implied
    
    # Explanation
    connection_reason: Optional[str] = None  # Why this relation exists
    
    # Tracking
    is_auto_generated: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # Confidence in relation validity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "properties": self.properties,
            "bidirectional": self.bidirectional,
            "connection_reason": self.connection_reason,
            "is_auto_generated": self.is_auto_generated,
            "created_at": self.created_at.isoformat(),
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create relation from dictionary."""
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_inverse_id(self) -> str:
        """Generate ID for inverse relation if bidirectional."""
        return f"{self.id}_inverse"
    
    def get_inverse_type(self) -> Optional[str]:
        """Get the inverse relation type (if bidirectional)."""
        inverse_map = {
            "contains": "part_of",
            "depends_on": "required_by",
            "implements": "implemented_by",
            "references": "referenced_by",
            "similar_to": "similar_to",  # Symmetric
            "prerequisite_for": "requires",
            "learned_from": "teaches",
            "authored_by": "authored",
            "located_in": "contains",
            "part_of": "contains",
            "uses": "used_by",
            "produces": "produced_by",
            "influenced_by": "influenced",
        }
        return inverse_map.get(self.relation_type)


@dataclass
class KnowledgeGraph:
    """
    Container for a collection of entities and relations.
    
    Represents a subgraph or the entire knowledge graph.
    """
    
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        return entity.id
    
    def add_relation(self, relation: Relation) -> str:
        """Add a relation to the graph."""
        self.relations[relation.id] = relation
        return relation.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self.relations.get(relation_id)
    
    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """Get all relations connected to an entity."""
        return [
            r for r in self.relations.values()
            if r.source_id == entity_id or r.target_id == entity_id
        ]
    
    def get_outgoing_relations(self, entity_id: str) -> List[Relation]:
        """Get relations where entity is the source."""
        return [r for r in self.relations.values() if r.source_id == entity_id]
    
    def get_incoming_relations(self, entity_id: str) -> List[Relation]:
        """Get relations where entity is the target."""
        return [r for r in self.relations.values() if r.target_id == entity_id]
    
    def get_neighbors(self, entity_id: str) -> List[Entity]:
        """Get all neighboring entities."""
        neighbor_ids = set()
        for relation in self.get_entity_relations(entity_id):
            if relation.source_id == entity_id:
                neighbor_ids.add(relation.target_id)
            else:
                neighbor_ids.add(relation.source_id)
        
        return [self.entities[nid] for nid in neighbor_ids if nid in self.entities]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to dictionary."""
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create graph from dictionary."""
        graph = cls()
        
        for entity_data in data.get("entities", {}).values():
            graph.add_entity(Entity.from_dict(entity_data))
        
        for relation_data in data.get("relations", {}).values():
            graph.add_relation(Relation.from_dict(relation_data))
        
        return graph
