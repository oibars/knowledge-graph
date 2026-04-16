"""
Agent Indexer

Scans ~/.claude/agents/*.md and indexes each agent as an Entity(label="Agent")
in the knowledge graph. Enables semantic agent routing via kg_search.

Usage:
    python -m knowledge_graph.agent_indexer
    python -m knowledge_graph.agent_indexer --agents-dir /custom/path
    python -m knowledge_graph.agent_indexer --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import structlog

from knowledge_graph.models import Entity
from knowledge_graph.services.graph_store import KnowledgeGraphStore

logger = structlog.get_logger()

DEFAULT_AGENTS_DIR = Path.home() / ".claude" / "agents"


def parse_frontmatter(content: str) -> dict[str, str]:
    """Extract YAML frontmatter fields from a markdown file.

    Parses the --- delimited block at the top of agent .md files.
    Returns a flat dict of string values (no nested YAML support needed).
    """
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    result = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def index_agents(
    agents_dir: Path = DEFAULT_AGENTS_DIR,
    store: KnowledgeGraphStore | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Index all agent .md files from agents_dir into the knowledge graph.

    Args:
        agents_dir: Directory containing agent .md files
        store: KnowledgeGraphStore instance (created with defaults if None)
        dry_run: If True, parse and report without writing to the graph

    Returns:
        (added, skipped) counts
    """
    if not agents_dir.exists():
        logger.warning("Agents directory not found", path=str(agents_dir))
        return 0, 0

    if store is None:
        store = KnowledgeGraphStore()

    agent_files = sorted(agents_dir.glob("*.md"))
    if not agent_files:
        logger.warning("No .md files found in agents directory", path=str(agents_dir))
        return 0, 0

    added = 0
    skipped = 0

    for md_file in agent_files:
        content = md_file.read_text(encoding="utf-8")
        fm = parse_frontmatter(content)

        name = fm.get("name", "")
        description = fm.get("description", "")

        if not name:
            logger.debug("Skipping agent file (no name in frontmatter)", file=md_file.name)
            skipped += 1
            continue

        entity_id = f"agent-{md_file.stem}"

        if dry_run:
            print(f"  [DRY RUN] {entity_id}: {name[:60]}")
            added += 1
            continue

        # Check if already indexed with same content
        existing = store.get_entity(entity_id)
        combined = f"{name}:{description}"
        import hashlib
        content_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

        if existing and existing.content_hash == content_hash:
            logger.debug("Agent already indexed, skipping", agent=entity_id)
            skipped += 1
            continue

        # Build tags from the filename prefix (e.g. "engineering", "testing")
        parts = md_file.stem.split("-")
        category_tag = parts[0] if parts else "uncategorized"

        entity = Entity(
            id=entity_id,
            label="Agent",
            name=name,
            description=description or f"Claude Code agent: {name}",
            tags=[category_tag, "claude-code", "agent"],
            source_file_path=str(md_file),
            source_app="claude-code",
            importance_score=0.6,
            is_auto_generated=True,
            content_hash=content_hash,
        )

        store.add_entity(entity)
        logger.info("Indexed agent", id=entity_id, name=name, category=category_tag)
        added += 1

    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Claude Code agents into the knowledge graph")
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=DEFAULT_AGENTS_DIR,
        help=f"Directory of agent .md files (default: {DEFAULT_AGENTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse agents and report without writing to the graph",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.agents_dir}")
    added, skipped = index_agents(agents_dir=args.agents_dir, dry_run=args.dry_run)
    print(f"Done — added: {added}, skipped: {skipped}")


if __name__ == "__main__":
    main()
