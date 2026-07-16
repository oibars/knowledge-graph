"""
Knowledge Graph Store Service
Hybrid storage using NetworkX for in-memory graph operations and SQLite for persistence.
"""

import json
import os
import pickle
import re
import sqlite3
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import asdict
import structlog

from knowledge_graph.models import Entity, Relation, KnowledgeGraph, RELATION_TYPES

logger = structlog.get_logger()

# Auto-snapshot cadence and retention (second line of defense behind external backups)
SNAPSHOT_EVERY_WRITES = 50
SNAPSHOT_EVERY_HOURS = 24
SNAPSHOT_KEEP = 10

# Function words excluded from lexical search tokens. Without this, natural-
# language queries ("what can i do this week to…") match on the/this/to/in and
# style-similar-but-irrelevant documents outrank genuinely relevant ones.
_SEARCH_STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from had has have he her his how
    i if in is it its me my of on or our she so that the their them then there these they
    this to und up was we were what when where which who why will with would you your""".split()
)


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
        data_dir: str = str(Path.home() / ".knowledge-graph" / "data"),
        db_name: str = "knowledge_graph.db",
        enable_snapshots: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / db_name
        self.enable_snapshots = enable_snapshots
        # KG_SNAPSHOT_DIR relocates snapshots off the data dir (e.g. into a backup
        # folder); sub-keyed by the data dir's parent so multiple graphs don't collide.
        snapshot_override = os.environ.get("KG_SNAPSHOT_DIR")
        self.snapshot_dir = (
            Path(snapshot_override).expanduser() / self.data_dir.parent.name
            if snapshot_override
            else self.data_dir / "kg_snapshots"
        )
        
        if self.enable_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory graph
        self._graph: nx.DiGraph = nx.DiGraph()
        
        # Entity and relation storage
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}

        # Auto-snapshot accounting
        self._writes_since_snapshot = 0
        self._last_snapshot_at = datetime.now()

        # Batched read-access persistence (reads must not write synchronously)
        self._dirty_access: Set[str] = set()
        # Upsert-by-content index
        self._hash_to_id: Dict[str, str] = {}
        # Whole-graph centrality maps are O(V·E) — cache until the graph changes
        self._centrality_cache: Dict[str, Dict[str, float]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_from_database()
        self._rebuild_hash_index()
        self._known_mtime = self._current_mtime()

        logger.info(
            "KnowledgeGraphStore initialized",
            db_path=str(self.db_path),
            entities=len(self._entities),
            relations=len(self._relations)
        )
    
    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL + busy timeout.

        WAL lets the long-running MCP server and cron ingesters read/write the same
        DB without 'database is locked' failures; busy_timeout waits out short locks.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _rebuild_hash_index(self):
        """Rebuild the content_hash -> entity id upsert index."""
        self._hash_to_id = {
            e.content_hash: eid for eid, e in self._entities.items() if e.content_hash
        }

    def _current_mtime(self) -> int:
        """Newest mtime (ns) across the DB and its WAL — changes on any external write."""
        m = 0
        for p in (self.db_path, Path(str(self.db_path) + "-wal")):
            try:
                m = max(m, p.stat().st_mtime_ns)
            except FileNotFoundError:
                pass
        return m

    def refresh_if_stale(self):
        """Reload from SQLite if another process changed the DB since our load.

        The long-running MCP server otherwise never sees rows written by cron
        ingesters. Costs two stat() calls unless a change is detected.
        """
        if self._current_mtime() <= self._known_mtime:
            return
        self._flush_access()
        self._entities.clear()
        self._relations.clear()
        self._graph.clear()
        self._centrality_cache.clear()
        self._load_from_database()
        self._rebuild_hash_index()
        self._known_mtime = self._current_mtime()
        logger.info(
            "Graph reloaded after external DB change",
            entities=len(self._entities),
            relations=len(self._relations),
        )

    def _flush_access(self):
        """Persist batched read-access metadata (last_accessed, access_count)."""
        if not self._dirty_access:
            return
        try:
            rows = []
            for eid in self._dirty_access:
                e = self._entities.get(eid)
                if e:
                    rows.append((
                        e.last_accessed.isoformat() if e.last_accessed else None,
                        e.access_count,
                        eid,
                    ))
            conn = self._connect()
            conn.executemany(
                "UPDATE entities SET last_accessed=?, access_count=? WHERE id=?", rows
            )
            conn.commit()
            conn.close()
            self._dirty_access.clear()
            self._known_mtime = self._current_mtime()
        except Exception as e:
            logger.warning("Failed to flush access metadata", error=str(e))

    def _maybe_snapshot(self):
        """Per-write hook: record our own write's mtime, then snapshot if due."""
        # Our own commit just changed the DB — don't let refresh_if_stale treat it
        # as an external change.
        self._known_mtime = self._current_mtime()
        self._centrality_cache.clear()
        if not self.enable_snapshots:
            return
        self._writes_since_snapshot += 1
        due_by_count = self._writes_since_snapshot >= SNAPSHOT_EVERY_WRITES
        elapsed = (datetime.now() - self._last_snapshot_at).total_seconds()
        due_by_time = elapsed >= SNAPSHOT_EVERY_HOURS * 3600
        if due_by_count or due_by_time:
            try:
                self.create_snapshot()
            except Exception as e:
                logger.warning("Auto-snapshot failed", error=str(e))

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = self._connect()
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
                content_hash TEXT,
                is_auto_generated INTEGER DEFAULT 0
            )
        """)

        # Migration: add is_auto_generated if upgrading from older schema
        try:
            cursor.execute("ALTER TABLE entities ADD COLUMN is_auto_generated INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_content_hash ON entities(content_hash)")
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """Load entities and relations from SQLite into memory."""
        conn = self._connect()
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
            content_hash=row[18],
            is_auto_generated=bool(row[19]) if len(row) > 19 else False,
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
            entity.content_hash,
            int(entity.is_auto_generated),
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
        """Add or update an entity. Returns the existing id on a content-hash match."""
        # Upsert-by-content: identical name+description already stored → return it
        # instead of minting a duplicate node under a fresh id.
        if entity.content_hash:
            existing_id = self._hash_to_id.get(entity.content_hash)
            if existing_id and existing_id != entity.id and existing_id in self._entities:
                logger.debug("Entity dedup hit", incoming=entity.id, existing=existing_id)
                return existing_id

        # Update in-memory structures
        self._entities[entity.id] = entity
        self._graph.add_node(
            entity.id,
            label=entity.label,
            name=entity.name,
            data=entity
        )
        
        # Persist to database
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO entities VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, self._entity_to_row(entity))
        
        conn.commit()
        conn.close()
        
        if entity.content_hash:
            self._hash_to_id[entity.content_hash] = entity.id

        logger.debug("Entity added", entity_id=entity.id, label=entity.label)
        self._maybe_snapshot()
        return entity.id
    
    def add_relation(self, relation: Relation) -> str:
        """Add or update a relation."""
        if relation.relation_type not in RELATION_TYPES:
            raise ValueError(
                f"Unknown relation_type '{relation.relation_type}' — "
                f"must be one of RELATION_TYPES"
            )
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
        conn = self._connect()
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
                # Persist the inverse too — memory-only twins vanished on restart
                inv_conn = self._connect()
                inv_conn.execute(
                    "INSERT OR REPLACE INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._relation_to_row(inverse)
                )
                inv_conn.commit()
                inv_conn.close()
        
        logger.debug(
            "Relation added",
            relation_id=relation.id,
            source=relation.source_id,
            target=relation.target_id,
            type=relation.relation_type
        )
        self._maybe_snapshot()
        return relation.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID. Access metadata is batched, not written per read."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.touch()
            self._dirty_access.add(entity_id)
            if len(self._dirty_access) >= 25:
                self._flush_access()
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
        self._hash_to_id = {h: i for h, i in self._hash_to_id.items() if i != entity_id}
        self._dirty_access.discard(entity_id)
        
        # Update database
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        cursor.execute(
            "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.debug("Entity deleted", entity_id=entity_id, relations_removed=len(relations_to_remove))
        self._maybe_snapshot()
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

        # Remove the persisted inverse twin, if one was created
        inverse_id = relation.get_inverse_id() if relation.bidirectional else None
        if inverse_id and inverse_id in self._relations:
            inv = self._relations.pop(inverse_id)
            if self._graph.has_edge(inv.source_id, inv.target_id):
                self._graph.remove_edge(inv.source_id, inv.target_id)

        # Update database
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
        if inverse_id:
            cursor.execute("DELETE FROM relations WHERE id = ?", (inverse_id,))
        conn.commit()
        conn.close()
        
        logger.debug("Relation deleted", relation_id=relation_id)
        self._maybe_snapshot()
        return True
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {
            t
            for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) >= 2 and t not in _SEARCH_STOPWORDS
        }

    def _embed_query(self, text: str) -> Optional[List[float]]:
        """Embed a query via local Ollama; None when unavailable (lexical fallback).

        OLLAMA_EMBED_MODEL must match the model that embedded the stored entities —
        vectors from different models are not comparable.
        """
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        try:
            import urllib.request

            body = json.dumps({"model": model, "input": text[:2000]}).encode()
            req = urllib.request.Request(
                f"{url}/api/embed", data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.load(r)
            v = d.get("embeddings") or d.get("embedding")
            return v[0] if v and isinstance(v[0], list) else v
        except Exception:
            return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def search_entities_scored(
        self,
        query: str,
        label: Optional[str] = None,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        source_app: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        semantic: bool = True,
    ) -> List[Tuple[float, Entity]]:
        """Hybrid search returning (score, entity) pairs, best first.

        Scores are comparable across calls with the same query mode — callers
        merging results from multiple stores should rank on them rather than on
        list position. See search_entities for the entities-only convenience.

        Filters: label, tags (entity must carry all), source_app,
        created_after/created_before (ISO dates, compared on created_at).
        """
        q_tokens = self._tokenize(query)
        q_lower = query.lower().strip()
        after = datetime.fromisoformat(created_after) if created_after else None
        before = datetime.fromisoformat(created_before) if created_before else None

        # Filter pass
        pool = []
        for e in self._entities.values():
            if label and e.label != label:
                continue
            if tags and not all(t in e.tags for t in tags):
                continue
            if source_app and e.source_app != source_app:
                continue
            if after and e.created_at < after:
                continue
            if before and e.created_at > before:
                continue
            pool.append(e)
        if not pool:
            return []

        # Lexical: token overlap weighted by field + whole-phrase bonus
        lex: Dict[str, float] = {}
        for e in pool:
            score = 0.0
            name_l = e.name.lower()
            desc_l = (e.description or "").lower()
            name_toks = self._tokenize(e.name)
            desc_toks = self._tokenize(e.description or "")
            tag_toks = self._tokenize(" ".join(e.tags))
            topic_toks = self._tokenize(" ".join(e.topics))
            for t in q_tokens:
                if t in name_toks:
                    score += 3.0
                if t in desc_toks:
                    score += 1.5
                if t in tag_toks:
                    score += 2.0
                if t in topic_toks:
                    score += 1.0
            if q_lower and q_lower in name_l:
                score += 6.0
                if name_l == q_lower:
                    score += 3.0
            elif q_lower and q_lower in desc_l:
                score += 2.0
            if score > 0:
                lex[e.id] = score
        # Normalize against a query-dependent ceiling (perfect name match + phrase
        # bonus) rather than the observed max: scores stay comparable across stores
        # and calls — a store whose best match is weak must not score it 1.0.
        # Within-store ordering is unchanged (monotonic).
        lex_ceiling = 3.0 * len(q_tokens) + 9.0 if q_tokens else 1.0
        lex_max = max(max(lex.values()) if lex else 0.0, lex_ceiling)

        # Semantic: one query embedding, cosine against stored vectors (dim-guarded)
        sem: Dict[str, float] = {}
        if semantic:
            qvec = self._embed_query(query)
            if qvec:
                for e in pool:
                    if e.embedding and len(e.embedding) == len(qvec):
                        s = self._cosine(qvec, e.embedding)
                        if s > 0:
                            sem[e.id] = s

        # Blend: lexical + semantic + importance + recency (90-day half-life)
        now = datetime.now()
        results = []
        for e in pool:
            l = lex.get(e.id, 0.0) / lex_max
            s = sem.get(e.id, 0.0)
            if l <= 0 and s < 0.55:
                continue
            age_days = max(0.0, (now - e.updated_at).total_seconds() / 86400)
            recency = 0.5 ** (age_days / 90.0)
            if sem:
                final = 0.45 * l + 0.30 * s + 0.15 * e.importance_score + 0.10 * recency
            else:
                final = 0.65 * l + 0.20 * e.importance_score + 0.15 * recency
            results.append((final, e))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:limit]

    def search_entities(
        self,
        query: str,
        label: Optional[str] = None,
        limit: int = 10,
        tags: Optional[List[str]] = None,
        source_app: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        semantic: bool = True,
    ) -> List[Entity]:
        """Hybrid search: tokenized lexical + semantic (when Ollama and stored
        embeddings are available) blended with importance and recency.

        Filters: label, tags (entity must carry all), source_app,
        created_after/created_before (ISO dates, compared on created_at).
        """
        return [
            e
            for _, e in self.search_entities_scored(
                query,
                label=label,
                limit=limit,
                tags=tags,
                source_app=source_app,
                created_after=created_after,
                created_before=created_before,
                semantic=semantic,
            )
        ]
    
    def recent_entities(
        self,
        days: float = 7,
        label: Optional[str] = None,
        source_app: Optional[str] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """Entities created in the last `days`, newest first — "what changed"
        recall without needing a search query."""
        cutoff = datetime.now() - timedelta(days=days)
        out = [
            e
            for e in self._entities.values()
            if e.created_at and e.created_at >= cutoff
            and (label is None or e.label == label)
            and (source_app is None or e.source_app == source_app)
        ]
        out.sort(key=lambda e: e.created_at, reverse=True)
        return out[:limit]

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

        if metric == "degree":
            return self._graph.degree(entity_id)

        computers = {
            "pagerank": nx.pagerank,
            "betweenness": nx.betweenness_centrality,
            "closeness": nx.closeness_centrality,
        }
        fn = computers.get(metric)
        if fn is None:
            return 0.0

        # Whole-graph maps are expensive — compute once, reuse until a write clears the cache
        if metric not in self._centrality_cache:
            try:
                self._centrality_cache[metric] = fn(self._graph)
            except Exception:
                return 0.0
        return self._centrality_cache[metric].get(entity_id, 0.0)
    
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
            if len(other.embedding) != len(entity.embedding):
                continue  # different embedding model — not comparable

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
        
        self._flush_access()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshot_dir / f"kg_snapshot_{timestamp}.json"

        snapshot_data = {
            "entities": {k: v.to_dict() for k, v in self._entities.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()},
            "timestamp": timestamp,
            "stats": self.get_stats()
        }

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w") as f:
            json.dump(snapshot_data, f)

        self._writes_since_snapshot = 0
        self._last_snapshot_at = datetime.now()

        # Rotate: keep only the newest SNAPSHOT_KEEP snapshots (json + legacy pickle)
        snaps = sorted(self.snapshot_dir.glob("kg_snapshot_*"))
        for old in snaps[:-SNAPSHOT_KEEP]:
            old.unlink(missing_ok=True)

        logger.info("Snapshot created", path=str(snapshot_path))
        return str(snapshot_path)
    
    def load_snapshot(self, snapshot_path: str) -> bool:
        """Load a snapshot into the knowledge graph."""
        try:
            if str(snapshot_path).endswith(".json"):
                with open(snapshot_path) as f:
                    snapshot_data = json.load(f)
            else:  # legacy pickle snapshots
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
            conn = self._connect()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM entities")
            cursor.execute("DELETE FROM relations")
            
            for entity in self._entities.values():
                cursor.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._entity_to_row(entity)
                )
            
            for relation in self._relations.values():
                cursor.execute(
                    "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self._relation_to_row(relation)
                )
            
            conn.commit()
            conn.close()

            self._rebuild_hash_index()
            self._known_mtime = self._current_mtime()

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
