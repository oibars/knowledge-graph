"""
Shared bookmark ingester plumbing.

Each source-specific ingester produces a list of BookmarkRecord. This module
scores them with the interest profile and upserts Entity(label="Document")
into the knowledge graph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import structlog

from knowledge_graph.models import Entity
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.ingesters.interest_profile import (
    InterestProfile,
    build_profile,
    extract_host,
)

logger = structlog.get_logger()


# Scoring thresholds for the `signal-*` tag applied to each bookmark.
SIGNAL_HIGH = 0.7
SIGNAL_MEDIUM = 0.4

# Raw score → normalized importance mapping. Calibrated on "tokens matched in a
# single-sentence title plus domain bonus" — tuned after running once.
IMPORTANCE_PIVOT = 8.0  # raw score at which importance ≈ 0.8


@dataclass
class BookmarkRecord:
    """Normalized bookmark shape produced by every source ingester."""

    url: str
    title: str
    platform: str  # "chrome" | "firefox" | "reddit" | "youtube" | "instagram"
    folder_path: list[str] = field(default_factory=list)  # chrome/firefox folder breadcrumb
    added_at: Optional[datetime] = None
    extra_tags: list[str] = field(default_factory=list)  # platform-specific tags
    extra_properties: dict = field(default_factory=dict)
    collection: Optional[str] = None  # instagram folder / reddit subreddit / youtube playlist

    def stable_id(self) -> str:
        # Dedup by URL (canonical) — different platforms saving the same URL
        # get merged, which is what we want.
        canonical = self.url.strip().rstrip("/").lower()
        h = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return f"doc-bm-{h}"


def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", s)[:60]


def _raw_score(record: BookmarkRecord, profile: InterestProfile) -> float:
    score = 0.0
    score += profile.score_tokens(record.title)
    score += profile.score_tokens(" ".join(record.folder_path))
    if record.collection:
        score += profile.score_tokens(record.collection)
    score += profile.score_domain(record.url)
    # URL path segments often carry topic signal (e.g. /agile-ai-coming-for-you)
    path = re.sub(r"^https?://[^/]+", "", record.url)
    score += profile.score_tokens(path) * 0.5
    return score


def _raw_to_importance(raw: float) -> float:
    """Map raw token+domain score → [0.0, 1.0]. Asymptotic; more hits still climb."""
    if raw <= 0:
        return max(0.1, 0.3 + raw * 0.1)  # clamp floor; penalties pull toward 0.1
    # score / (score + pivot) — smooth, never exceeds 1.0.
    return round(raw / (raw + IMPORTANCE_PIVOT), 3)


def signal_tag(importance: float) -> str:
    if importance >= SIGNAL_HIGH:
        return "signal-high"
    if importance >= SIGNAL_MEDIUM:
        return "signal-medium"
    return "signal-low"


def record_to_entity(record: BookmarkRecord, profile: InterestProfile) -> Entity:
    raw = _raw_score(record, profile)
    importance = _raw_to_importance(raw)

    tags = ["bookmark", record.platform, signal_tag(importance)]
    tags.extend(_slugify(f) for f in record.folder_path if f)
    if record.collection:
        tags.append(_slugify(record.collection))
    tags.extend(record.extra_tags)
    # Dedup preserving order
    seen: set[str] = set()
    tags = [t for t in tags if t and not (t in seen or seen.add(t))]

    properties = {
        "source_type": "bookmark",
        "status": "raw",
        "platform": record.platform,
        "folder_path": record.folder_path,
        "collection": record.collection,
        "raw_score": round(raw, 2),
        "added_at": record.added_at.isoformat() if record.added_at else None,
        **record.extra_properties,
    }

    name = record.title.strip() or extract_host(record.url) or record.url
    return Entity(
        id=record.stable_id(),
        label="Document",
        name=name[:200],
        description=None,
        tags=tags,
        source_url=record.url,
        source_app=record.platform,
        importance_score=importance,
        is_auto_generated=True,
        properties=properties,
    )


def ingest_records(
    records: Iterable[BookmarkRecord],
    store: KnowledgeGraphStore,
    profile: Optional[InterestProfile] = None,
    dry_run: bool = False,
) -> dict:
    if profile is None:
        profile = build_profile(store=store)

    added = 0
    skipped = 0
    updated = 0
    signal_counts = {"signal-high": 0, "signal-medium": 0, "signal-low": 0}

    for record in records:
        entity = record_to_entity(record, profile)
        sig = next((t for t in entity.tags if t.startswith("signal-")), "signal-low")
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

        existing = store.get_entity(entity.id)
        if existing and existing.content_hash == entity.content_hash:
            skipped += 1
            continue
        if dry_run:
            continue
        if existing:
            # Preserve created_at; refresh the rest.
            entity.created_at = existing.created_at
            store.update_entity(entity)
            updated += 1
        else:
            store.add_entity(entity)
            added += 1

    logger.info(
        "Ingest complete",
        added=added,
        updated=updated,
        skipped=skipped,
        **signal_counts,
    )
    return {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        **signal_counts,
    }
