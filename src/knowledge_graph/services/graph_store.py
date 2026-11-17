"""
Knowledge Graph Store Service
Hybrid storage using NetworkX for in-memory graph operations and SQLite for persistence.
"""

import gzip
import json
from functools import wraps
from copy import deepcopy
from contextlib import contextmanager
from threading import RLock
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
import sys

from knowledge_graph.models import Entity, Relation, KnowledgeGraph, RELATION_TYPES
from knowledge_graph.models.timestamps import as_utc, utc_now
from knowledge_graph.local_transport import local_json

# Library diagnostics must not contaminate JSON-RPC stdio output. Applications
# may configure logging explicitly; otherwise default store diagnostics to stderr.
if not structlog.is_configured():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
logger = structlog.get_logger()


def canonical_embedding_model(model: str) -> str:
    """Ollama's untagged final model name aliases :latest (registry ports do not)."""
    model = model.strip()
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def entity_corpus(entity: Entity) -> str:
    """Classify reference feeds independently of the graph's privacy domain."""
    declared = entity.properties.get("corpus")
    if declared in {"memory", "reference"}:
        return declared
    source = (entity.source_app or "").lower()
    tags = {str(tag).lower() for tag in entity.tags}
    if "swiftrecap" in source or "trend" in source or {"swiftrecap", "trend", "news"} & tags:
        return "reference"
    return "memory"


def _locked(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return run


def _validate_corpus(corpus: str):
    if corpus not in {"memory", "reference", "all"}:
        raise ValueError("corpus must be memory, reference, or all")


# Auto-snapshot cadence and retention (second line of defense behind external backups)
SNAPSHOT_EVERY_WRITES = 50
SNAPSHOT_EVERY_HOURS = 24
SNAPSHOT_MIN_INTERVAL_SECONDS = 300
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
    
    - SQLite: Canonical atomic entity/relation state and identity decisions
    - NetworkX: Rebuildable multigraph cache for algorithms and traversal
    - Stored embeddings: Local similarity retrieval with freshness tracking
    
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
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        
        # Entity and relation storage
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}

        # Auto-snapshot accounting
        self._writes_since_snapshot = 0
        self._last_snapshot_at = datetime.now()

        # Batched read-access persistence (reads must not write synchronously)
        self._dirty_access: Set[str] = set()
        self._access_deltas: Dict[str, int] = {}
        self._lock = RLock()
        # Whole-graph centrality maps are O(V·E) — cache until the graph changes
        self._centrality_cache: Dict[str, Dict[str, float]] = {}
        # Search tokens belong to this store only; never share personal text
        # across graphs. Entries are bounded by the number of live entities.
        self._lexical_cache: Dict[str, Tuple[tuple, tuple]] = {}
        # Normalized float32 vectors, rebuilt only when the store changes, so a
        # query is one BLAS product instead of a Python cosine loop per entity.
        self._vector_cache: Dict[tuple, tuple] = {}
        self._vector_epoch = 0
        
        # Initialize database
        self._init_database()
        self._db = self._connect()
        self._known_version = self._db.execute("PRAGMA data_version").fetchone()[0]
        
        # Load existing data
        self._load_from_database()
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
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

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
        """Refresh derived caches using SQLite's external-commit counter.

        A persistent connection's data_version changes only for other writers;
        unlike DB/WAL mtimes, our own writes cannot hide an external commit.
        """
        with self._lock:
            if self._db.execute("PRAGMA data_version").fetchone()[0] == self._known_version:
                return
            self._flush_access()
            self._db.execute("BEGIN")
            try:
                version = self._db.execute("PRAGMA data_version").fetchone()[0]
                self._load_from_database(self._db)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            self._known_version = version
            self._known_mtime = self._current_mtime()

    @contextmanager
    def _transaction(self):
        """Serialize mutation and identity decisions against canonical SQLite."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                version = self._db.execute("PRAGMA data_version").fetchone()[0]
                if version != self._known_version:
                    self._load_from_database(self._db)
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            self._known_version = version

    def _flush_access(self):
        """Persist read deltas without overwriting another process's counters."""
        if not self._dirty_access:
            return
        try:
            with self._lock:
                rows = [(as_utc(self._entities[eid].last_accessed).isoformat(),
                         self._access_deltas.get(eid, 0), eid)
                        for eid in self._dirty_access if eid in self._entities]
                with self._db:
                    self._db.executemany(
                        "UPDATE entities SET last_accessed=CASE WHEN last_accessed IS NULL OR last_accessed < ? "
                        "THEN ? ELSE last_accessed END, access_count=access_count+? WHERE id=?",
                        [(when, when, delta, eid) for when, delta, eid in rows],
                    )
                self._dirty_access.clear()
                self._access_deltas.clear()
        except Exception as e:
            logger.warning("Failed to flush access metadata", error=str(e))

    def _maybe_snapshot(self):
        """Per-write hook: record our own write's mtime, then snapshot if due."""
        # Retain the mtime for diagnostics; data_version governs cache refresh.
        self._known_mtime = self._current_mtime()
        self._centrality_cache.clear()
        if not self.enable_snapshots:
            return
        self._writes_since_snapshot += 1
        elapsed = (datetime.now() - self._last_snapshot_at).total_seconds()
        due_by_count = (
            self._writes_since_snapshot >= SNAPSHOT_EVERY_WRITES
            and elapsed >= SNAPSHOT_MIN_INTERVAL_SECONDS
        )
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
    
    def _load_from_database(self, conn=None):
        """Build a coherent derived graph, then swap it after all rows validate."""
        own_connection = conn is None
        conn = conn or self._db
        if own_connection:
            conn.execute("BEGIN")
        try:
            entities = {row[0]: self._row_to_entity(row)
                        for row in conn.execute("SELECT * FROM entities")}
            relations = {row[0]: self._row_to_relation(row)
                         for row in conn.execute("SELECT * FROM relations")}
            graph = nx.MultiDiGraph()
            for e in entities.values():
                graph.add_node(e.id, label=e.label, name=e.name, data=e)
            for r in relations.values():
                if r.source_id in entities and r.target_id in entities:
                    graph.add_edge(r.source_id, r.target_id, key=r.id,
                                   relation_type=r.relation_type, strength=r.strength, data=r)
            # Read access may be waiting for persistence while a writer refreshes.
            for eid, delta in self._access_deltas.items():
                if eid in entities and eid in self._entities:
                    entities[eid].access_count += delta
                    entities[eid].last_accessed = self._entities[eid].last_accessed
            if own_connection:
                conn.commit()
        except Exception:
            if own_connection:
                conn.rollback()
            raise
        self._entities, self._relations, self._graph = entities, relations, graph
        self._lexical_cache.clear()
        self._centrality_cache.clear()
        self._vector_cache.clear()
        self._vector_epoch += 1
        logger.info("Loaded knowledge graph from database", entities=len(entities), relations=len(relations))

    @staticmethod
    def _row_to_entity(row) -> Entity:
        """Convert database row to Entity (no instance state required)."""
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
            importance_score=row[12] if row[12] is not None else 0.5,
            confidence_score=row[13] if row[13] is not None else 1.0,
            created_at=as_utc(datetime.fromisoformat(row[14])) if row[14] else utc_now(),
            updated_at=as_utc(datetime.fromisoformat(row[15])) if row[15] else utc_now(),
            last_accessed=as_utc(datetime.fromisoformat(row[16])) if row[16] else None,
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
            strength=row[4] if row[4] is not None else 0.5,
            properties=json.loads(row[5]) if row[5] else {},
            bidirectional=bool(row[6]),
            connection_reason=row[7],
            is_auto_generated=bool(row[8]),
            created_at=as_utc(datetime.fromisoformat(row[9])) if row[9] else utc_now(),
            confidence=row[10] if row[10] is not None else 1.0
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
            as_utc(entity.created_at).isoformat(),
            as_utc(entity.updated_at).isoformat(),
            as_utc(entity.last_accessed).isoformat() if entity.last_accessed else None,
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
            as_utc(relation.created_at).isoformat(),
            relation.confidence
        )
    
    # ========================================================================
    # CRUD Operations
    # ========================================================================
    
    @staticmethod
    def _content_dedup_allowed(entity: Entity) -> bool:
        # Occurrences and source-backed records carry identity, even when their
        # wording repeats. Only unanchored semantic records dedup by content.
        if entity.label not in {"Concept", "Document", "Code", "Skill", "Tool", "Tag"}:
            return False
        return not (entity.source_url or entity.source_file_path or any(
            entity.properties.get(key) for key in
            ("source_id", "conversation_id", "recording_id", "parent_id", "source_parent_id", "evidence")
        ))

    def _cache_entity(self, entity: Entity):
        cached = deepcopy(entity)
        if entity.id in self._access_deltas:
            cached.access_count += self._access_deltas[entity.id]
            old = self._entities.get(entity.id)
            if old and old.last_accessed:
                cached.last_accessed = max(filter(None, [cached.last_accessed, old.last_accessed]), key=as_utc)
        self._entities[entity.id] = cached
        self._lexical_cache.pop(entity.id, None)
        self._vector_cache.clear()
        self._vector_epoch += 1
        self._graph.add_node(entity.id, label=entity.label, name=entity.name,
                             data=self._entities[entity.id])

    def _cache_relation(self, relation: Relation):
        self._uncache_relation(relation.id)
        self._relations[relation.id] = deepcopy(relation)
        self._graph.add_edge(relation.source_id, relation.target_id, key=relation.id,
                             relation_type=relation.relation_type, strength=relation.strength,
                             data=self._relations[relation.id])

    def _uncache_relation(self, relation_id: str):
        old = self._relations.pop(relation_id, None)
        if old and self._graph.has_edge(old.source_id, old.target_id, key=relation_id):
            self._graph.remove_edge(old.source_id, old.target_id, key=relation_id)

    @_locked
    def add_entity(self, entity: Entity) -> str:
        """Atomic upsert; SQLite decides identity, birth time and vector freshness."""
        incoming = deepcopy(entity)
        incoming.created_at = as_utc(incoming.created_at)
        incoming.updated_at = as_utc(incoming.updated_at)
        incoming.last_accessed = as_utc(incoming.last_accessed) if incoming.last_accessed else None
        incoming.content_hash = incoming.compute_content_hash()
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id=?", (incoming.id,)).fetchone()
            existing = self._row_to_entity(row) if row else None
            # Same-ID identity takes priority over semantic dedup.
            if existing is None and self._content_dedup_allowed(incoming):
                for candidate_row in conn.execute(
                    "SELECT * FROM entities WHERE content_hash=? AND label=? ORDER BY created_at,id",
                    (incoming.content_hash, incoming.label),
                ):
                    candidate = self._row_to_entity(candidate_row)
                    if (self._content_dedup_allowed(candidate)
                        and candidate.source_app == incoming.source_app
                        and candidate.properties.get("domain", "unknown") == incoming.properties.get("domain", "unknown")
                        and entity_corpus(candidate) == entity_corpus(incoming)) :
                        incoming = candidate
                        break
            if existing is not None:
                incoming.created_at = existing.created_at
                same_content = incoming.content_hash == existing.compute_content_hash()
                if incoming.embedding is None and same_content:
                    incoming.embedding = existing.embedding
                    for key in ("embedding_model", "embedding_content_hash", "embedding_status"):
                        if key in existing.properties:
                            incoming.properties[key] = existing.properties[key]
                elif (not same_content and incoming.embedding == existing.embedding
                      and incoming.properties.get("embedding_content_hash") != incoming.content_hash):
                    # A caller may have mutated the returned Entity and left its
                    # old vector attached. It cannot be considered a fresh vector.
                    incoming.embedding = None
                # Access deltas remain a separate batch; writing a returned
                # Entity's count here would persist and later add the same read.
                incoming.access_count = existing.access_count
                incoming.last_accessed = existing.last_accessed
            if incoming.embedding is not None:
                bound_hash = incoming.properties.get("embedding_content_hash")
                if bound_hash and bound_hash != incoming.content_hash:
                    incoming.embedding = None
                else:
                    incoming.properties["embedding_content_hash"] = incoming.content_hash
                    incoming.properties["embedding_status"] = "ready"
            if incoming.embedding is None:
                incoming.properties.pop("embedding_content_hash", None)
                incoming.properties.pop("embedding_model", None)
                incoming.properties["embedding_status"] = "pending"
            conn.execute("INSERT OR REPLACE INTO entities VALUES (" + ",".join(["?"] * 20) + ")",
                         self._entity_to_row(incoming))
        self._cache_entity(incoming)
        self._maybe_snapshot()
        return incoming.id

    @_locked
    def add_relation(self, relation: Relation) -> str:
        """Commit a relation and its optional inverse as one mutation."""
        if relation.relation_type not in RELATION_TYPES:
            raise ValueError(f"Unknown relation_type '{relation.relation_type}' — must be one of RELATION_TYPES")
        incoming = deepcopy(relation)
        incoming.created_at = as_utc(incoming.created_at)
        inverse = None
        if incoming.bidirectional and incoming.get_inverse_type():
            inverse = Relation(id=incoming.get_inverse_id(), source_id=incoming.target_id,
                               target_id=incoming.source_id, relation_type=incoming.get_inverse_type(),
                               strength=incoming.strength, properties=deepcopy(incoming.properties),
                               connection_reason=f"Inverse of {incoming.id}", confidence=incoming.confidence)
        with self._transaction() as conn:
            for endpoint, kind in ((incoming.source_id, "Source"), (incoming.target_id, "Target")):
                if not conn.execute("SELECT 1 FROM entities WHERE id=?", (endpoint,)).fetchone():
                    raise ValueError(f"{kind} entity not found: {endpoint}")
            # Replacing a bidirectional edge with another shape removes its twin.
            conn.execute("DELETE FROM relations WHERE id=?", (incoming.get_inverse_id(),))
            for record in [incoming] + ([inverse] if inverse else []):
                conn.execute("INSERT OR REPLACE INTO relations VALUES (" + ",".join(["?"] * 11) + ")",
                             self._relation_to_row(record))
        self._uncache_relation(incoming.get_inverse_id())
        self._cache_relation(incoming)
        if inverse:
            self._cache_relation(inverse)
        self._maybe_snapshot()
        return incoming.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Return a detached entity; edits become visible only after commit."""
        self.refresh_if_stale()
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity:
                entity.touch()
                self._dirty_access.add(entity_id)
                self._access_deltas[entity_id] = self._access_deltas.get(entity_id, 0) + 1
                if len(self._dirty_access) >= 25:
                    self._flush_access()
                return deepcopy(entity)
        return None

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        self.refresh_if_stale()
        return deepcopy(self._relations.get(relation_id))

    @_locked
    def update_entity(self, entity: Entity) -> str:
        incoming = deepcopy(entity)
        incoming.updated_at = utc_now()
        return self.add_entity(incoming)

    @_locked
    def clear_embedding(self, entity_id: str) -> bool:
        """Drop a derived vector without touching the source content.

        `add_entity` deliberately restores a stored vector when the content hash
        is unchanged, so removing an unproven or stale vector needs its own
        committed path.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
            if not row:
                return False
            entity = self._row_to_entity(row)
            entity.embedding = None
            for key in ("embedding_model", "embedding_content_hash"):
                entity.properties.pop(key, None)
            entity.properties["embedding_status"] = "pending"
            conn.execute("INSERT OR REPLACE INTO entities VALUES (" + ",".join(["?"] * 20) + ")",
                         self._entity_to_row(entity))
        self._cache_entity(entity)
        self._maybe_snapshot()
        return True

    @_locked
    def delete_entity(self, entity_id: str) -> bool:
        with self._transaction() as conn:
            if not conn.execute("SELECT 1 FROM entities WHERE id=?", (entity_id,)).fetchone():
                return False
            conn.execute("DELETE FROM relations WHERE source_id=? OR target_id=?", (entity_id, entity_id))
            conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
        for rid, rel in list(self._relations.items()):
            if rel.source_id == entity_id or rel.target_id == entity_id:
                self._uncache_relation(rid)
        if self._graph.has_node(entity_id):
            self._graph.remove_node(entity_id)
        self._entities.pop(entity_id, None)
        self._lexical_cache.pop(entity_id, None)
        self._vector_cache.clear()
        self._vector_epoch += 1
        self._dirty_access.discard(entity_id)
        self._access_deltas.pop(entity_id, None)
        self._maybe_snapshot()
        return True

    @_locked
    def delete_relation(self, relation_id: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM relations WHERE id=?", (relation_id,)).fetchone()
            if row is None:
                return False
            relation = self._row_to_relation(row)
            ids = [relation_id]
            if relation.bidirectional:
                ids.append(relation.get_inverse_id())
            # Deleting a generated inverse removes the owning pair too.
            if relation_id.endswith("_inverse"):
                owner_id = relation_id.removesuffix("_inverse")
                owner = conn.execute("SELECT bidirectional FROM relations WHERE id=?", (owner_id,)).fetchone()
                if owner and owner[0]:
                    ids.append(owner_id)
            conn.executemany("DELETE FROM relations WHERE id=?", [(rid,) for rid in ids])
        for rid in ids:
            self._uncache_relation(rid)
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

    def _lexical_fields(self, entity: Entity) -> tuple:
        # Entities are mutable and callers can modify tags/topics directly.
        # Check content, rather than relying only on CRUD invalidation.
        action_text = ""
        if entity.label in {"Task", "Decision"}:
            # Known recall fields only: arbitrary properties may contain unrelated
            # payloads. Bound both evidence count and the final lexical supplement.
            parts = [value[:256] for key in ("owner", "due_at", "status")
                     if isinstance(value := entity.properties.get(key), str)]
            evidence = entity.properties.get("evidence")
            if isinstance(evidence, list):
                for item in evidence[:32]:
                    if isinstance(item, dict):
                        parts.extend(value[:2048] for key in ("quote", "speaker")
                                     if isinstance(value := item.get(key), str))
            action_text = "\n".join(parts)[:8192]
        description = "\n".join(value for value in (entity.description or "", action_text) if value)
        signature = (entity.name, description, tuple(entity.tags), tuple(entity.topics))
        cached = self._lexical_cache.get(entity.id)
        if cached is None or cached[0] != signature:
            fields = (
                entity.name.lower(), description.lower(),
                self._tokenize(entity.name), self._tokenize(description),
                self._tokenize(" ".join(entity.tags)), self._tokenize(" ".join(entity.topics)),
            )
            cached = (signature, fields)
            self._lexical_cache[entity.id] = cached
        return cached[1]

    def _embed_query(self, text: str) -> Optional[List[float]]:
        """Embed a query via local Ollama; None when unavailable (lexical fallback).

        OLLAMA_EMBED_MODEL must match the model that embedded the stored entities —
        vectors from different models are not comparable.
        """
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
        try:
            d = local_json("/api/embed",
                           {"model": model, "input": text[:2000],
                            "keep_alive": os.environ.get("JARVIS_OLLAMA_KEEP_ALIVE", "2m")},
                           url, timeout=8)
            v = d.get("embeddings") or d.get("embedding")
            return v[0] if v and isinstance(v[0], list) else v
        except Exception:
            return None

    def embed_query(self, text: str) -> Optional[List[float]]:
        """Public wrapper so a caller can reuse one query vector across stores."""
        return self._embed_query(text)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _vector_index(self):
        """Normalized float32 vectors grouped by dimension, cached per model.

        Building numpy arrays once per store change turns every query into one
        BLAS product; per-query Python cosine loops over 15k entities were the
        dominant cost of local recall.
        """
        model_key = canonical_embedding_model(os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma:latest"))
        key = (model_key, self._vector_epoch)
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached
        groups: Dict[int, tuple] = {}
        rows: Dict[int, list] = {}
        positions: Dict[str, tuple] = {}
        for e in self._entities.values():
            declared = e.properties.get("embedding_model")
            if declared and canonical_embedding_model(declared) != model_key:
                continue
            vector = e.embedding
            if not vector:
                continue
            dim = len(vector)
            bucket = rows.get(dim)
            if bucket is None:
                bucket = rows[dim] = []
            positions[e.id] = (dim, len(bucket))
            bucket.append(vector)
        for dim, bucket in rows.items():
            matrix = np.asarray(bucket, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            groups[dim] = matrix / norms
        self._vector_cache[key] = (positions, groups)
        return self._vector_cache[key]

    def _semantic_scores(self, pool, qvec) -> Dict[str, float]:
        """Cosine similarity against cached normalized vectors, dim-guarded."""
        if not qvec:
            return {}
        positions, groups = self._vector_index()
        query = np.asarray(qvec, dtype=np.float32)
        dim = len(qvec)
        query_norm = float(np.linalg.norm(query))
        if dim not in groups or query_norm == 0:
            return {}
        query = query / query_norm
        matrix = groups[dim]
        picked = [(e.id, positions[e.id][1]) for e in pool
                  if e.id in positions and positions[e.id][0] == dim]
        if not picked:
            return {}
        indices = np.fromiter((row for _, row in picked), dtype=np.intp, count=len(picked))
        scores = matrix[indices] @ query
        return {eid: float(score) for (eid, _), score in zip(picked, scores) if score > 0}

    @_locked
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
        corpus: str = "all",
        query_vector: Optional[List[float]] = None,
    ) -> List[Tuple[float, Entity]]:
        """Hybrid search returning (score, entity) pairs, best first.

        Pass `query_vector` to reuse one embedding across stores; a "both scope"
        recall otherwise embeds the same question twice.

        Scores are comparable across calls with the same query mode — callers
        merging results from multiple stores should rank on them rather than on
        list position. See search_entities for the entities-only convenience.

        Filters: label, tags (entity must carry all), source_app,
        created_after/created_before (ISO dates, compared on created_at in UTC).
        Naive timestamps retain their legacy host-local timezone meaning.
        """
        _validate_corpus(corpus)
        self.refresh_if_stale()
        q_tokens = self._tokenize(query)
        q_lower = query.lower().strip()
        after = as_utc(datetime.fromisoformat(created_after)) if created_after else None
        before = as_utc(datetime.fromisoformat(created_before)) if created_before else None

        # Filter pass
        pool = []
        for e in self._entities.values():
            if corpus != "all" and entity_corpus(e) != corpus:
                continue
            if label and e.label != label:
                continue
            if tags and not all(t in e.tags for t in tags):
                continue
            if source_app and e.source_app != source_app:
                continue
            if after and as_utc(e.created_at) < after:
                continue
            if before and as_utc(e.created_at) > before:
                continue
            pool.append(e)
        if not pool:
            return []

        # Lexical: token overlap weighted by field + whole-phrase bonus
        lex: Dict[str, float] = {}
        for e in pool:
            score = 0.0
            name_l, desc_l, name_toks, desc_toks, tag_toks, topic_toks = self._lexical_fields(e)
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

        # Semantic: one query embedding, cached normalized matrix product (dim-guarded)
        sem: Dict[str, float] = {}
        if semantic:
            sem = self._semantic_scores(pool, query_vector if query_vector is not None else self._embed_query(query))

        # Blend: lexical + semantic + importance + recency (90-day half-life).
        # Importance and recency are gated by relevance: they may reorder real
        # matches but must never lift a single accidental token match over a
        # caller's relevance floor.
        now = utc_now()
        results = []
        for e in pool:
            l = lex.get(e.id, 0.0) / lex_max
            s = sem.get(e.id, 0.0)
            if l <= 0 and s < 0.55:
                continue
            relevance = min(1.0, max(l, s) / 0.5)
            age_days = max(0.0, (now - as_utc(e.updated_at)).total_seconds() / 86400)
            recency = 0.5 ** (age_days / 90.0)
            if sem:
                final = 0.45 * l + 0.30 * s + relevance * (0.15 * e.importance_score + 0.10 * recency)
            else:
                final = 0.65 * l + relevance * (0.20 * e.importance_score + 0.15 * recency)
            results.append((final, e))

        results.sort(key=lambda x: x[0], reverse=True)
        return deepcopy(results[:limit])

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
        corpus: str = "all",
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
                corpus=corpus,
            )
        ]
    
    @_locked
    def recent_entities(
        self,
        days: float = 7,
        label: Optional[str] = None,
        source_app: Optional[str] = None,
        limit: int = 50,
        corpus: str = "all",
    ) -> List[Entity]:
        """Entities created in the last `days`, newest first — "what changed"
        recall without needing a search query."""
        _validate_corpus(corpus)
        self.refresh_if_stale()
        cutoff = utc_now() - timedelta(days=days)

        out = [
            e
            for e in self._entities.values()
            if e.created_at and as_utc(e.created_at) >= cutoff
            and (label is None or e.label == label)
            and (source_app is None or e.source_app == source_app)
            and (corpus == "all" or entity_corpus(e) == corpus)
        ]
        out.sort(key=lambda e: as_utc(e.created_at), reverse=True)
        return deepcopy(out[:limit])

    @_locked
    def find_by_label(self, label: str) -> List[Entity]:
        """Find all entities of a specific label type."""
        self.refresh_if_stale()
        return deepcopy([e for e in self._entities.values() if e.label == label])
    
    @_locked
    def find_by_tag(self, tag: str) -> List[Entity]:
        """Find all entities with a specific tag."""
        self.refresh_if_stale()
        return deepcopy([e for e in self._entities.values() if tag in e.tags])
    
    @_locked
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
        self.refresh_if_stale()
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
        
        return deepcopy(results)
    
    @_locked
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[List[Entity]]:
        """Find shortest path between two entities."""
        self.refresh_if_stale()
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
            
            return deepcopy([self._entities[nid] for nid in path_ids if nid in self._entities])
        except nx.NetworkXNoPath:
            return None
    
    @_locked
    def get_entity_relations(
        self,
        entity_id: str,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[Relation]:
        """Get all relations for an entity."""
        self.refresh_if_stale()
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
        
        return deepcopy(list({r.id: r for r in relations}.values()))
    
    # ========================================================================
    # Graph Algorithms
    # ========================================================================
    
    @_locked
    def get_centrality(self, entity_id: str, metric: str = "pagerank") -> float:
        """
        Get centrality score for an entity.
        
        Metrics: pagerank, degree, betweenness, closeness
        """
        self.refresh_if_stale()
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
    
    @_locked
    def find_communities(self) -> List[Set[str]]:
        """Find communities in the knowledge graph."""
        self.refresh_if_stale()
        try:
            communities = nx.community.greedy_modularity_communities(
                self._graph.to_undirected()
            )
            return [set(c) for c in communities]
        except:
            return []
    
    @_locked
    def find_similar_entities(
        self,
        entity_id: str,
        limit: int = 5
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to the given entity based on embedding."""
        self.refresh_if_stale()
        entity = self._entities.get(entity_id)
        if not entity or not entity.embedding:
            return []
        
        similarities = []
        
        for other_id, other in self._entities.items():
            if other_id == entity_id or not other.embedding:
                continue
            if len(other.embedding) != len(entity.embedding):
                continue  # different embedding model — not comparable

            model = entity.properties.get("embedding_model")
            other_model = other.properties.get("embedding_model")
            if model and other_model and canonical_embedding_model(model) != canonical_embedding_model(other_model):
                continue
            
            # Cosine similarity
            similarity = self._cosine(entity.embedding, other.embedding)
            
            if similarity > 0.5:  # Threshold
                similarities.append((other, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return deepcopy(similarities[:limit])
    
    # ========================================================================
    # Statistics and Export
    # ========================================================================
    
    @_locked
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        self.refresh_if_stale()
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
            "is_connected": nx.is_weakly_connected(self._graph) if self._graph else False,
            "connected_components": nx.number_weakly_connected_components(self._graph),
        }
    
    @_locked
    def export_graphml(self, path: Optional[str] = None) -> str:
        """Export graph to GraphML format for visualization."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.data_dir / f"knowledge_graph_{timestamp}.graphml")
        
        self.refresh_if_stale()
        # Create a copy for export with simplified attributes
        export_graph = nx.MultiDiGraph()
        
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
                    u, v, key=relation.id,
                    relation_type=relation.relation_type,
                    strength=relation.strength
                )
        
        nx.write_graphml(export_graph, path)
        logger.info("Graph exported to GraphML", path=path)
        return path
    
    @_locked
    def create_snapshot(self) -> str:
        """Create a timestamped snapshot of the knowledge graph.

        Vectors are deliberately omitted: they are derived, rebuildable from the
        local embedding model, and were ~95% of every snapshot's bytes. Files are
        gzipped, and retention keeps recent runs plus one per day so a bulk
        ingest cannot overwrite yesterday's candidate restore point.
        """
        if not self.enable_snapshots:
            return ""

        self._flush_access()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.snapshot_dir / f"kg_snapshot_{timestamp}.json.gz"

        def slim(entity: Entity) -> dict:
            data = entity.to_dict()
            data["embedding"] = None
            # Mark vectors as rebuildable so a restore re-embeds instead of
            # assuming an untagged name/description pair already has a vector.
            props = dict(data.get("properties") or {})
            props["embedding_status"] = "pending"
            props.pop("embedding_content_hash", None)
            data["properties"] = props
            return data

        snapshot_data = {
            "entities": {k: slim(v) for k, v in self._entities.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()},
            "timestamp": timestamp,
            "stats": self.get_stats(),
            "embeddings": "omitted",
        }

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as f:
            json.dump(snapshot_data, f)
        os.replace(temporary, snapshot_path)

        self._writes_since_snapshot = 0
        self._last_snapshot_at = datetime.now()

        self._rotate_snapshots()
        logger.info("Snapshot created", path=str(snapshot_path))
        return str(snapshot_path)

    def _rotate_snapshots(self):
        """Keep the newest SNAPSHOT_KEEP, plus one per day for the last week."""
        snaps = sorted(self.snapshot_dir.glob("kg_snapshot_*"))
        keep = set(snaps[-SNAPSHOT_KEEP:])
        daily: Dict[str, Path] = {}
        for path in snaps:
            stamp = path.name.split("_")[2][:8] if len(path.name.split("_")) > 2 else ""
            if len(stamp) == 8 and stamp.isdigit():
                daily[stamp] = path  # newest file of that day, since sorted
        for day in sorted(daily)[-7:]:
            keep.add(daily[day])
        for old in snaps:
            if old not in keep:
                old.unlink(missing_ok=True)
    
    @_locked
    def load_snapshot(self, snapshot_path: str) -> bool:
        """Validate and atomically restore a snapshot before replacing caches."""
        try:
            if str(snapshot_path).endswith(".json.gz"):
                with gzip.open(snapshot_path, "rt") as f:
                    snapshot_data = json.load(f)
            elif str(snapshot_path).endswith(".json"):
                with open(snapshot_path) as f:
                    snapshot_data = json.load(f)
            else:  # legacy local pickle snapshots
                with open(snapshot_path, "rb") as f:
                    snapshot_data = pickle.load(f)
            entities = {e.id: e for e in (Entity.from_dict(d) for d in snapshot_data["entities"].values())}
            relations = [Relation.from_dict(d) for d in snapshot_data["relations"].values()]
            for r in relations:
                if r.source_id not in entities or r.target_id not in entities or r.relation_type not in RELATION_TYPES:
                    raise ValueError(f"Invalid snapshot relation: {r.id}")
            for e in entities.values():
                if e.content_hash != e.compute_content_hash():
                    e.content_hash = e.compute_content_hash()
                    e.embedding = None
                    e.properties["embedding_status"] = "pending"
                    e.properties.pop("embedding_content_hash", None)
            with self._transaction() as conn:
                conn.execute("DELETE FROM relations")
                conn.execute("DELETE FROM entities")
                conn.executemany("INSERT INTO entities VALUES (" + ",".join(["?"] * 20) + ")",
                                 [self._entity_to_row(e) for e in entities.values()])
                conn.executemany("INSERT INTO relations VALUES (" + ",".join(["?"] * 11) + ")",
                                 [self._relation_to_row(r) for r in relations])
            self._dirty_access.clear()
            self._access_deltas.clear()
            self._load_from_database()
            self._known_mtime = self._current_mtime()
            logger.info("Snapshot loaded", path=snapshot_path)
            return True
        except Exception as e:
            logger.error("Failed to load snapshot", path=snapshot_path, error=str(e))
            return False

    @_locked
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        self.refresh_if_stale()
        return deepcopy(list(self._entities.values()))
    
    @_locked
    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        self.refresh_if_stale()
        return deepcopy(list(self._relations.values()))
