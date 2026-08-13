"""
Shared always-on ingest pipeline.

persona-os (professional scope) and life-os (personal scope) previously carried
copy-paste forks of the same staging-JSONL → Entity construction. This module is
the single implementation; the repo scripts shrink to thin CLI wrappers.

Staging line shape (produced by the fieldy/plaud/IG pollers):
    {"id": ..., "title": ..., "date": ISO, "topics": [...], "speakers": [...], "text": ...}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from knowledge_graph.models import Entity
from knowledge_graph.services.graph_store import KnowledgeGraphStore

# Scope config: the ONLY thing that differs between the professional and
# personal pipelines. data_dir None = the engine's default (~/.knowledge-graph/data).
SCOPES = {
    "professional": {"data_dir": None, "importance": 0.7},
    "personal": {
        "data_dir": str(Path.home() / ".knowledge-graph-life" / "data"),
        "importance": 0.5,
    },
}


def store_for(scope: str) -> KnowledgeGraphStore:
    """Open the graph store for a scope ("professional" | "personal")."""
    cfg = SCOPES[scope]
    if cfg["data_dir"]:
        return KnowledgeGraphStore(data_dir=cfg["data_dir"])
    return KnowledgeGraphStore()


def conversation_entity(
    source: str,
    record: dict,
    scope: str,
    importance: Optional[float] = None,
) -> Entity:
    """Build the canonical conversation/event Entity from a staging record."""
    try:
        created = datetime.fromisoformat((record.get("date") or "2024-01-01")[:19])
    except ValueError:
        created = datetime.now()
    topics = record.get("topics") or []
    domain = "professional" if scope == "professional" else "personal"
    return Entity(
        id=f"conv-{source}-{record['id']}",
        label="Event",
        name=record.get("title") or "Untitled",
        description=(record.get("text") or "")[:2000],
        source_app=source,
        source_url=f"{source}://{record['id']}",
        topics=topics,
        tags=[source, "conversation", domain] + topics,
        importance_score=importance if importance is not None else SCOPES[scope]["importance"],
        created_at=created,
        properties={
            "domain": domain,
            "speakers": record.get("speakers") or [],
            "start": record.get("date"),
            "source": source,
        },
    )


def trend_entity(item: dict, source: str = "swiftrecap", scope: str = "professional") -> Entity:
    """Build a Document entity from a trend-feed item (e.g. SwiftRecap /api/signal.json).

    Item shape: {title, summary?, url, category?, topic?, keywords?, sourceDomain?,
    sourceTier?, popularity?, publishedAt}. Importance blends a trend baseline with
    the feed's popularity signal (0.4 → 0.8).
    """
    import hashlib

    url = item["url"]
    published = None
    try:
        published = datetime.fromisoformat(str(item.get("publishedAt", ""))[:19])
    except ValueError:
        published = None
    # created_at is ingest time. A feed's publishedAt is often midnight-on-date
    # and can sit in the future; writing it here made every 24h window re-count
    # the item until that calendar day expired.
    created = datetime.now()
    category = item.get("category")
    keywords = [k for k in (item.get("keywords") or []) if k][:5]
    topics = [t for t in [category, item.get("topic")] if t] + keywords
    popularity = min(1.0, max(0.0, float(item.get("popularity") or 0)))
    domain = "professional" if scope == "professional" else "personal"
    return Entity(
        id=f"trend-{source}-{hashlib.sha1(url.encode()).hexdigest()[:12]}",
        label="Document",
        name=(item.get("title") or "Untitled")[:300],
        description=(item.get("summary") or "")[:2000] or None,
        source_app=source,
        source_url=url,
        topics=topics,
        tags=[source, "trend", domain] + ([category] if category else []),
        importance_score=round(0.4 + 0.4 * popularity, 3),
        created_at=created,
        properties={
            "domain": domain,
            "sourceDomain": item.get("sourceDomain"),
            "sourceTier": item.get("sourceTier"),
            "popularity": popularity,
            "publishedAt": published.isoformat() if published else item.get("publishedAt"),
        },
    )


def ingest_staging_jsonl(
    source: str,
    staging_path: str | Path,
    scope: str,
    store: Optional[KnowledgeGraphStore] = None,
) -> int:
    """Ingest a staging JSONL file into the scope's graph. Returns rows ingested."""
    store = store or store_for(scope)
    n = 0
    for line in Path(staging_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        store.add_entity(conversation_entity(source, record, scope))
        n += 1
    return n
