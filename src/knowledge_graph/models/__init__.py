"""
Knowledge Graph Models
Entity and Relation models for the knowledge graph system.
"""

from .entity import Entity, Relation, KnowledgeGraph, RELATION_TYPES, ENTITY_LABELS

__all__ = [
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "RELATION_TYPES",
    "ENTITY_LABELS",
]
