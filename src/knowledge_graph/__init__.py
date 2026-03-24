"""
Knowledge Graph App for SwissPy
Provides graph-based knowledge storage and semantic relationship tracking.
"""

from .models import Entity, Relation, KnowledgeGraph
from .services.graph_store import KnowledgeGraphStore
from .services.linker import SemanticLinker

__all__ = [
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "KnowledgeGraphStore",
    "SemanticLinker",
]
