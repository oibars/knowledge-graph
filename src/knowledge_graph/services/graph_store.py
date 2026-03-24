"""
Knowledge Graph Store Service
Hybrid storage using NetworkX for in-memory graph operations and SQLite for persistence.
"""

import json
import pickle
import sqlite3
import networkx as nx
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import asdict
import structlog

from knowledge_graph.models import Entity, Relation, KnowledgeGraph, RELATION_TYPES

logger = structlog.get_logger()


class KnowledgeGraphStore:
    """
    Hybrid knowledge graph storage.
    
    - NetworkX: In-memory graph operations, algorithms, traversal
    - SQLite: Persistent storage of entities and relations
    - Optional: LanceDB for embedding-based similarity search
    
    Provides CRUD operations, graph algorithms, and persistence.
    """
    
    def __init__(
        self,
        data_dir: str = "/home/oscr/Apps/swisspy/data",
        db_name: str = "knowledge_graph.db",
        enable_snapshots: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / db_name
        self.enable_snapshots = enable_snapshots
        self.snapshot_dir = self.data_dir / "kg_snapshots"
        
        if self.enable_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory graph
        self._graph: nx.DiGraph = nx.DiGraph()
        
        # Entity and relation storage
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_from_database()
        
        logger.info(
            "KnowledgeGraphStore initialized",
            db_path=str(self.db_path),
            entities=len(self._entities),
            relations=len(self._relations)
        )
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                embedding BLOB,
                topics TEXT,
                tags TEXT,
                source_url TEXT,
                source_file_path TEXT,
                source_app TEXT,
                source_user TEXT,
                importance_score REAL DEFAULT 0.5,
                confidence_score REAL DEFAULT 1.0,
                created_at TEXT,
                updated_at TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                content_hash TEXT
            )
        """)
        
        # Relations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                properties TEXT,
                bidirectional INTEGER DEFAULT 0,
                connection_reason TEXT,
                is_auto_generated INTEGER DEFAULT 1,
                created_at TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)")
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load entities and relations from SQLite into memory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load entities
        cursor.execute("SELECT * FROM entities")
        rows = cursor.fetchall()
        
        for row in rows:
            entity = self._row_to_entity(row)
            self._entities[entity.id] = entity
            self._graph.add_node(
                entity.id,
                label=entity.label,
                name=entity.name,
                data=entity
            )
        
        # Load relations
        cursor.execute("SELECT * FROM relations")
        rows = cursor.fetchall()
        
        for row in rows:
            relation = self._row_to_relation(row)
            self._relations[relation.id] = relation
            self._graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_type=relation.relation_type,
                strength=relation.strength,
                data=relation
            )
        
        conn.close()
        
        logger.info(
            "Loaded knowledge graph from database",
            entities=len(self._entities),
            relations=len(self._relations)
        )
    
    def _row_to_entity(self, row) -> Entity:
        """Convert database row to Entity."""
        return Entity(
            id=row[0],
            label=row[1],
            name=row[2],
            description=row[3],
            properties=json.loads(row[4]) if row[4] else {},
            embedding=pickle.loads(row[5]) if row[5] else None,
            topics=json.loads(row[6]) if row[6] else [],
            tags=json.loads(row[7]) if row[7] else [],
            source_url=row[8],
            source_file_path=row[9],
            source_app=row[10],
            source_user=row[11],
            importance_score=row[12] or 0.5,
            confidence_score=row[13] or 1.0,
            created_at=datetime.fromisoformat(row[14]) if row[14] else datetime.now(),
            updated_at=datetime.fromisoformat(row[15]) if row[15] else datetime.now(),
            last_accessed=datetime.fromisoformat(row[16]) if row[16] else None,
            access_count=row[17] or 0,
            content_hash=row[18]
        )
    
    def _row_to_relation(self, row) -> Relation:
        """Convert database row to Relation."""
        return Relation(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=row[3],
            strength=row[4] or 0.5,
            properties=json.loads(row[5]) if row[5] else {},
            bidirectional=bool(row[6]),
            connection_reason=row[7],
            is_auto_generated=bool(row[8]),
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
            confidence=row[10] or 1.0
        )
    
    def _entity_to_row(self, entity: Entity) -> tuple:
        """Convert Entity to database row."""
        return (
            entity.id,
            entity.label,
            entity.name,
            entity.description,
            json.dumps(entity.properties),
            pickle.dumps(entity.embedding) if entity.embedding else None,
            json.dumps(entity.topics),
            json.dumps(entity.tags),
            entity.source_url,
            entity.source_file_path,
            entity.source_app,
            entity.source_user,
            entity.importance_score,
            entity.confidence_score,
            entity.created_at.isoformat(),
            entity.updated_at.isoformat(),
            entity.last_accessed.isoformat() if entity.last_accessed else None,
            entity.access_count,
            entity.content_hash
        )
    
    def _relation_to_row(self, relation: Relation) -> tuple:
        """Convert Relation to database row."""
        return (
            relation.id,
            relation.source_id,
            relation.target_id,
            relation.relation_type,
            relation.strength,
            json.dumps(relation.properties),
            int(relation.bidirectional),
            relation.connection_reason,
            int(relation.is_auto_generated),
            relation.created_at.isoformat(),
            relation.confidence
        )
    
    # ========================================================================
    # CRUD Operations
    # ========================================================================
    
    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity."""
        # Update in-memory structures
        self._entities[entity.id] = entity
        self._graph.add_node(
            entity.id,
            label=entity.label,
            name=entity.name,
            data=entity
        )
        
        # Persist to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO entities VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, self._entity_to_row(entity))
        
        conn.commit()
        conn.close()
        
        logger.debug("Entity added", entity_id=entity.id, label=entity.label)
        return entity.id
    
    def add_relation(self, relation: Relation) -> str:
        """Add or update a relation."""
        # Validate entities exist
        if relation.source_id not in self._entities:
            raise ValueError(f"Source entity not found: {relation.source_id}")
        if relation.target_id not in self._entities:
            raise ValueError(f"Target entity not found: {relation.target_id}")
        
        # Update in-memory structures
        self._relations[relation.id] = relation
        self._graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            strength=relation.strength,
            data=relation
        )
        
        # Persist to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO relations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, self._relation_to_row(relation))
        
        conn.commit()
        conn.close()
        
        # Handle bidirectional
        if relation.bidirectional:
            inverse_type = relation.get_inverse_type()
            if inverse_type:
                inverse_id = relation.get_inverse_id()
                inverse = Relation(
                    id=inverse_id,
                    source_id=relation.target_id,
                    target_id=relation.source_id,
                    relation_type=inverse_type,
                    strength=relation.strength,
                    properties=relation.properties,
                    bidirectional=False,  # Prevent infinite recursion
                    connection_reason=f"Inverse of {relation.id}",
                    is_auto_generated=True,
                    confidence=relation.confidence
                )
                # Store inverse without calling add_relation (avoids recursion)
                self._relations[inverse.id] = inverse
                self._graph.add_edge(
                    inverse.source_id,
                    inverse.target_id,
                    relation_type=inverse.relation_type,
                    strength=inverse.strength,
                    data=inverse
                )
        
        logger.debug(
            "Relation added",
            relation_id=relation.id,
            source=relation.source_id,
            target=relation.target_id,
            type=relation.relation_type
        )
        return relation.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.touch()
            self._update_entity_access(entity)
        return entity
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self._relations.get(relation_id)
    
    def update_entity(self, entity: Entity) -> str:
        """Update an existing entity."""
        entity.updated_at = datetime.now()
        return self.add_entity(entity)
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relations."""
        if entity_id not in self._entities:
            return False
        
        # Remove related relations
        relations_to_remove = [
            rid for rid, rel in self._relations.items()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
        
        for rid in relations_to_remove:
            del self._relations[rid]
        
        # Remove from graph
        self._graph.remove_node(entity_id)
        del self._entities[entity_id]
        
        # Update database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        cursor.execute(
            "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.debug("Entity deleted", entity_id=entity_id, relations_removed=len(relations_to_remove))
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation."""
        if relation_id not in self._relations:
            return False
        
        relation = self._relations[relation_id]
        
        # Remove from graph
        if self._graph.has_edge(relation.source_id, relation.target_id):
            self._graph.remove_edge(relation.source_id, relation.target_id)
        
        del self._relations[relation_id]
        
        # Update database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
        conn.commit()
        conn.close()
        
        logger.debug("Relation deleted", relation_id=relation_id)
        return True
    
    def _update_entity_access(self, entity: Entity):
        """Update entity access metadata in database (async)."""
        # Batch updates or background task
        pass
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    def search_entities(
        self,
        query: str,
        label: Optional[str] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search entities by name or description."""
        query_lower = query.lower()
        matches = []
        
        for entity in self._entities.values():
            score = 0
            
            # Name match (highest weight)
            if query_lower in entity.name.lower():
                score += 10
                if entity.name.lower() == query_lower:
                    score += 5
            
            # Description match
            if entity.description and query_lower in entity.description.lower():
                score += 3
            
            # Tag match
            for tag in entity.tags:
                if query_lower in tag.lower():
                    score += 2
            
            # Topic match
            for topic in entity.topics:
                if query_lower in topic.lower():
                    score += 1
            
            if score > 0:
                # Filter by label if specified
                if label and entity.label != label:
                    continue
                matches.append((score, entity))
        
        # Sort by score
        matches.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in matches[:limit]]
    
    def find_by_label(self, label: str) -> List[Entity]:
        """Find all entities of a specific label type."""
        return [e for e in self._entities.values() if e.label == label]
    
    def find_by_tag(self, tag: str) -> List[Entity]:
        """Find all entities with a specific tag."""
        return [e for e in self._entities.values() if tag in e.tags]
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        depth: int = 1
    ) -> Dict[int, List[Entity]]:
        """
        Get neighbors at specified depth.
        
        Returns dict mapping depth to list of entities.
        """
        if entity_id not in self._entities:
            return {}
        
        results: Dict[int, List[Entity]] = {i: [] for i in range(1, depth + 1)}
        visited: Set[str] = {entity_id}
        current_level: Set[str] = {entity_id}
        
        for current_depth in range(1, depth + 1):
            next_level: Set[str] = set()
            
            for node_id in current_level:
                # Get neighbors
                if relation_type:
                    edges = [
                        (u, v, d) for u, v, d in self._graph.edges(node_id, data=True)
                        if d.get("relation_type") == relation_type
                    ]
                else:
                    edges = list(self._graph.edges(node_id, data=True))
                
                for u, v, data in edges:
                    neighbor_id = v if u == node_id else u
                    
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.add(neighbor_id)
                        entity = self._entities.get(neighbor_id)
                        if entity:
                            results[current_depth].append(entity)
            
            current_level = next_level
            if not current_level:
                break
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[List[Entity]]:
        """Find shortest path between two entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        try:
            path_ids = nx.shortest_path(
                self._graph.to_undirected(),
                source_id,
                target_id
            )
            
            if len(path_ids) > max_length + 1:
                return None
            
            return [self._entities[nid] for nid in path_ids if nid in self._entities]
        except nx.NetworkXNoPath:
            return None
    
    def get_entity_relations(
        self,
        entity_id: str,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[Relation]:
        """Get all relations for an entity."""
        if entity_id not in self._entities:
            return []
        
        relations = []
        
        if direction in ("out", "both"):
            for _, target_id, data in self._graph.out_edges(entity_id, data=True):
                relation = data.get("data")
                if relation:
                    relations.append(relation)
        
        if direction in ("in", "both"):
            for source_id, _, data in self._graph.in_edges(entity_id, data=True):
                relation = data.get("data")
                if relation:
                    relations.append(relation)
        
        return relations
    
    # ========================================================================
    # Graph Algorithms
    # ========================================================================
    
    def get_centrality(self, entity_id: str, metric: str = "pagerank") -> float:
        """
        Get centrality score for an entity.
        
        Metrics: pagerank, degree, betweenness, closeness
        """
        if entity_id not in self._entities:
            return 0.0
        
        if metric == "pagerank":
            try:
                pr = nx.pagerank(self._graph)
                return pr.get(entity_id, 0.0)
            except:
                return 0.0
        
        elif metric == "degree":
            return self._graph.degree(entity_id)
        
        elif metric == "betweenness":
            try:
                bc = nx.betweenness_centrality(self._graph)
                return bc.get(entity_id, 0.0)
            except:
                return 0.0
        
        elif metric == "closeness":
            try:
                cc = nx.closeness_centrality(self._graph)
                return cc.get(entity_id, 0.0)
            except:
                return 0.0
        
        return 0.0
    
    def find_communities(self) -> List[Set[str]]:
        """Find communities in the knowledge graph."""
        try:
            communities = nx.community.greedy_modularity_communities(
                self._graph.to_undirected()
            )
            return [set(c) for c in communities]
        except:
            return []
    
    def find_similar_entities(
        self,
        entity_id: str,
        limit: int = 5
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to the given entity based on embedding."""
        entity = self._entities.get(entity_id)
        if not entity or not entity.embedding:
            return []
        
        similarities = []
        entity_embedding = np.array(entity.embedding)
        
        for other_id, other in self._entities.items():
            if other_id == entity_id or not other.embedding:
                continue
            
            other_embedding = np.array(other.embedding)
            
            # Cosine similarity
            similarity = np.dot(entity_embedding, other_embedding) / (
                np.linalg.norm(entity_embedding) * np.linalg.norm(other_embedding)
            )
            
            if similarity > 0.5:  # Threshold
                similarities.append((other, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    # ========================================================================
    # Statistics and Export
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        label_counts = {}
        for entity in self._entities.values():
            label_counts[entity.label] = label_counts.get(entity.label, 0) + 1
        
        relation_type_counts = {}
        for relation in self._relations.values():
            rt = relation.relation_type
            relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1
        
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "label_distribution": label_counts,
            "relation_type_distribution": relation_type_counts,
            "density": nx.density(self._graph),
            "is_connected": nx.is_weakly_connected(self._graph),
            "connected_components": nx.number_weakly_connected_components(self._graph),
        }
    
    def export_graphml(self, path: Optional[str] = None) -> str:
        """Export graph to GraphML format for visualization."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.data_dir / f"knowledge_graph_{timestamp}.graphml")
        
        # Create a copy for export with simplified attributes
        export_graph = nx.DiGraph()
        
        for node_id, data in self._graph.nodes(data=True):
            entity = data.get("data")
            if entity:
                export_graph.add_node(
                    node_id,
                    label=entity.label,
                    name=entity.name,
                    description=entity.description or ""
                )
        
        for u, v, data in self._graph.edges(data=True):
            relation = data.get("data")
            if relation:
                export_graph.add_edge(
                    u, v,
                    relation_type=relation.relation_type,
                    strength=relation.strength
                )
        
        nx.write_graphml(export_graph, path)
        logger.info("Graph exported to GraphML", path=path)
        return path
    
    def create_snapshot(self) -> str:
        """Create a timestamped snapshot of the knowledge graph."""
        if not self.enable_snapshots:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshot_dir / f"kg_snapshot_{timestamp}.pickle"
        
        snapshot_data = {
            "entities": {k: v.to_dict() for k, v in self._entities.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()},
            "timestamp": timestamp,
            "stats": self.get_stats()
        }
        
        with open(snapshot_path, "wb") as f:
            pickle.dump(snapshot_data, f)
        
        logger.info("Snapshot created", path=str(snapshot_path))
        return str(snapshot_path)
    
    def load_snapshot(self, snapshot_path: str) -> bool:
        """Load a snapshot into the knowledge graph."""
        try:
            with open(snapshot_path, "rb") as f:
                snapshot_data = pickle.load(f)
            
            # Clear current data
            self._entities.clear()
            self._relations.clear()
            self._graph.clear()
            
            # Load entities
            for entity_data in snapshot_data["entities"].values():
                entity = Entity.from_dict(entity_data)
                self._entities[entity.id] = entity
                self._graph.add_node(entity.id, data=entity)
            
            # Load relations
            for relation_data in snapshot_data["relations"].values():
                relation = Relation.from_dict(relation_data)
                self._relations[relation.id] = relation
                self._graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    data=relation
                )
            
            # Persist to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM entities")
            cursor.execute("DELETE FROM relations")
            
            for entity in self._entities.values():
                cursor.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._entity_to_row(entity)
                )
            
            for relation in self._relations.values():
                cursor.execute(
                    "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._relation_to_row(relation)
                )
            
            conn.commit()
            conn.close()
            
            logger.info("Snapshot loaded", path=snapshot_path)
            return True
            
        except Exception as e:
            logger.error("Failed to load snapshot", path=snapshot_path, error=str(e))
            return False
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return list(self._entities.values())
    
    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        return list(self._relations.values())
