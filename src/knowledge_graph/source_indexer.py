"""
Source Indexer

Scans /sources/*.md capture files and indexes them into the knowledge graph
as Entity(label="Document"), with one Entity(label="Concept") per key claim.

Relations produced:
    Concept --part_of--> Document
    Document --authored_by--> Person (if author present)
    Document --references--> Document (via [[wikilinks]] in Connections)
    Document --tagged_with--> Tag (via `contains` on Tag entity)

Usage:
    python -m knowledge_graph.source_indexer
    python -m knowledge_graph.source_indexer --sources-dir /custom/path
    python -m knowledge_graph.source_indexer --dry-run
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import structlog

from knowledge_graph.models import Entity, Relation
from knowledge_graph.services.graph_store import KnowledgeGraphStore

logger = structlog.get_logger()

DEFAULT_SOURCES_DIR = Path.cwd() / "sources"


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", s)[:80]


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML-lite frontmatter. Supports string and [list] values."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    raw, body = m.group(1), m.group(2)
    fm: dict = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")]
            fm[key] = [i for i in items if i]
        else:
            fm[key] = value.strip('"').strip("'")
    return fm, body


def extract_sections(body: str) -> dict[str, str]:
    """Return {section_title: section_body} by splitting on `## ` headings."""
    sections: dict[str, str] = {}
    positions = [(m.group(1).strip(), m.end()) for m in SECTION_RE.finditer(body)]
    for i, (title, start) in enumerate(positions):
        end = positions[i + 1][1] - len(f"## {positions[i+1][0]}") - 1 if i + 1 < len(positions) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def extract_bullets(section_body: str) -> list[str]:
    """Return non-comment bullet lines from a markdown section."""
    bullets = []
    for line in section_body.splitlines():
        s = line.strip()
        if s.startswith("<!--") or not s.startswith("-"):
            continue
        text = s.lstrip("- ").strip()
        if text:
            bullets.append(text)
    return bullets


def index_source_file(
    md_file: Path,
    store: KnowledgeGraphStore,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Index one source markdown file. Returns (entities_added, relations_added)."""
    content = md_file.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)

    title = fm.get("title") or md_file.stem
    slug = md_file.stem
    doc_id = f"doc-{slug}"
    content_hash = _short_hash(content)

    existing = store.get_entity(doc_id)
    if existing and existing.content_hash == content_hash:
        logger.debug("Source unchanged, skipping", file=md_file.name)
        return 0, 0

    sections = extract_sections(body)
    thesis = sections.get("Thesis", "").strip()
    claims = extract_bullets(sections.get("Key claims", ""))
    connections = WIKILINK_RE.findall(sections.get("Connections", ""))

    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    try:
        importance = float(fm.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5

    doc = Entity(
        id=doc_id,
        label="Document",
        name=title,
        description=thesis or fm.get("title", ""),
        tags=tags + ["source"],
        source_url=fm.get("source") or None,
        source_file_path=str(md_file),
        source_app="knowledge-graph",
        importance_score=importance,
        content_hash=content_hash,
        is_auto_generated=False,
        properties={
            "author": fm.get("author"),
            "date_published": fm.get("date_published"),
            "date_captured": fm.get("date_captured"),
            "source_type": fm.get("source_type"),
            "status": fm.get("status", "raw"),
        },
    )

    entities_added = 0
    relations_added = 0

    if dry_run:
        print(f"  [DRY] Document: {doc_id} — {title}")
    else:
        store.add_entity(doc)
        entities_added += 1

    for claim in claims:
        concept_id = f"concept-{_short_hash(claim)}"
        if store.get_entity(concept_id):
            continue
        concept = Entity(
            id=concept_id,
            label="Concept",
            name=claim[:80],
            description=claim,
            tags=tags,
            source_file_path=str(md_file),
            source_app="knowledge-graph",
            importance_score=importance,
            is_auto_generated=False,
        )
        if dry_run:
            print(f"    [DRY] Concept: {concept_id} — {claim[:60]}")
        else:
            store.add_entity(concept)
            entities_added += 1
            rel = Relation(
                id=f"rel-{concept_id}-part_of-{doc_id}",
                source_id=concept_id,
                target_id=doc_id,
                relation_type="part_of",
                strength=0.9,
                is_auto_generated=True,
                connection_reason="Extracted from `## Key claims`",
            )
            store.add_relation(rel)
            relations_added += 1

    author = fm.get("author")
    if author:
        person_id = f"person-{_slugify(author)}"
        if not store.get_entity(person_id):
            person = Entity(
                id=person_id,
                label="Person",
                name=author,
                is_auto_generated=True,
            )
            if not dry_run:
                store.add_entity(person)
                entities_added += 1
        if not dry_run:
            store.add_relation(Relation(
                id=f"rel-{doc_id}-authored_by-{person_id}",
                source_id=doc_id,
                target_id=person_id,
                relation_type="authored_by",
                strength=1.0,
                is_auto_generated=True,
            ))
            relations_added += 1

    for target in connections:
        target_slug = _slugify(target)
        if not target_slug:
            continue
        target_id = f"concept-{target_slug}"
        # Create a stub Concept if it doesn't exist — later captures will enrich it.
        if not store.get_entity(target_id):
            if dry_run:
                print(f"    [DRY] Concept stub: {target_id}")
            else:
                store.add_entity(Entity(
                    id=target_id,
                    label="Concept",
                    name=target.strip(),
                    is_auto_generated=True,
                    tags=["stub"],
                ))
                entities_added += 1
        if not dry_run:
            store.add_relation(Relation(
                id=f"rel-{doc_id}-references-{target_id}",
                source_id=doc_id,
                target_id=target_id,
                relation_type="references",
                strength=0.6,
                is_auto_generated=True,
                connection_reason="[[wikilink]] in Connections section",
            ))
            relations_added += 1

    logger.info(
        "Indexed source",
        file=md_file.name,
        doc=doc_id,
        concepts=len(claims),
        connections=len(connections),
    )
    return entities_added, relations_added


def index_sources(
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    store: KnowledgeGraphStore | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    if not sources_dir.exists():
        logger.warning("Sources directory not found", path=str(sources_dir))
        return 0, 0

    if store is None:
        store = KnowledgeGraphStore()

    files = sorted(p for p in sources_dir.glob("*.md") if not p.stem.startswith("_"))
    if not files:
        logger.warning("No source .md files found", path=str(sources_dir))
        return 0, 0

    total_entities = 0
    total_relations = 0
    for md_file in files:
        e, r = index_source_file(md_file, store, dry_run=dry_run)
        total_entities += e
        total_relations += r

    return total_entities, total_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Index /sources/*.md into the knowledge graph")
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Scanning: {args.sources_dir}")
    entities, relations = index_sources(sources_dir=args.sources_dir, dry_run=args.dry_run)
    print(f"Done — entities: {entities}, relations: {relations}")


if __name__ == "__main__":
    main()
