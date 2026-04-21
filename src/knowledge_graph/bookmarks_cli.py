"""
kg-ingest-bookmarks entry point.

Usage:
    kg-ingest-bookmarks                          # run every locally-available source
    kg-ingest-bookmarks --source chrome
    kg-ingest-bookmarks --source firefox
    kg-ingest-bookmarks --source reddit          # requires env vars
    kg-ingest-bookmarks --source youtube         # requires OAuth client json
    kg-ingest-bookmarks --source instagram --path ~/Downloads/ig_export.zip
    kg-ingest-bookmarks --dry-run                # score + report, no writes
    kg-ingest-bookmarks --show-profile           # print top interest terms, exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.ingesters.common import ingest_records
from knowledge_graph.ingesters.interest_profile import build_profile, top_terms

logger = structlog.get_logger()

LOCAL_SOURCES = {"chrome", "firefox"}
AUTH_SOURCES = {"reddit", "youtube"}
FILE_SOURCES = {"instagram"}
ALL_SOURCES = LOCAL_SOURCES | AUTH_SOURCES | FILE_SOURCES


def _run_source(source: str, args, store, profile) -> dict | None:
    if source == "chrome":
        from knowledge_graph.ingesters.chrome import read_chrome_bookmarks, DEFAULT_CHROME_PATH
        path = Path(args.path) if args.path else DEFAULT_CHROME_PATH
        if not path.exists():
            logger.warning("Chrome bookmarks not found", path=str(path))
            return None
        records = list(read_chrome_bookmarks(path))
        print(f"[chrome] {len(records)} bookmarks")
        return ingest_records(records, store, profile=profile, dry_run=args.dry_run)

    if source == "firefox":
        from knowledge_graph.ingesters.firefox import read_firefox_bookmarks, find_places_db
        path = Path(args.path) if args.path else find_places_db()
        if not path or not path.exists():
            logger.warning("Firefox places.sqlite not found")
            return None
        records = list(read_firefox_bookmarks(path))
        print(f"[firefox] {len(records)} bookmarks")
        return ingest_records(records, store, profile=profile, dry_run=args.dry_run)

    if source == "reddit":
        from knowledge_graph.ingesters.reddit import read_reddit_saved, missing_env
        missing = missing_env()
        if missing:
            logger.warning("Skipping reddit — missing env vars", missing=missing)
            return None
        records = list(read_reddit_saved(limit=args.limit))
        print(f"[reddit] {len(records)} saved posts")
        return ingest_records(records, store, profile=profile, dry_run=args.dry_run)

    if source == "youtube":
        from knowledge_graph.ingesters.youtube import read_youtube_saves, DEFAULT_CLIENT_PATH
        if not DEFAULT_CLIENT_PATH.exists():
            logger.warning("Skipping youtube — OAuth client json missing", path=str(DEFAULT_CLIENT_PATH))
            return None
        records = list(read_youtube_saves())
        print(f"[youtube] {len(records)} items (liked + watch-later)")
        return ingest_records(records, store, profile=profile, dry_run=args.dry_run)

    if source == "instagram":
        from knowledge_graph.ingesters.instagram import read_instagram_saves
        if not args.path:
            logger.warning("Skipping instagram — --path to export zip/dir required")
            return None
        records = list(read_instagram_saves(Path(args.path)))
        print(f"[instagram] {len(records)} saved items")
        return ingest_records(records, store, profile=profile, dry_run=args.dry_run)

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest bookmarks/saves into the knowledge graph")
    parser.add_argument(
        "--source", choices=sorted(ALL_SOURCES) + ["all"], default="all",
        help="Which source to ingest (default: all locally-available)",
    )
    parser.add_argument("--path", help="Override file/dir path (ig export, etc.)")
    parser.add_argument("--limit", type=int, default=1000, help="Max items per paginated source")
    parser.add_argument("--dry-run", action="store_true", help="Score + report only, no DB writes")
    parser.add_argument("--show-profile", action="store_true", help="Print interest profile top terms and exit")
    args = parser.parse_args()

    store = KnowledgeGraphStore()

    if args.show_profile:
        profile = build_profile(store=store)
        print(f"Interest profile: {profile.size()} terms")
        for term, w in top_terms(profile, n=50):
            print(f"  {w:7.1f}  {term}")
        return

    profile = build_profile(store=store)
    print(f"Interest profile: {profile.size()} terms")

    sources = [args.source] if args.source != "all" else sorted(ALL_SOURCES)
    totals = {"added": 0, "updated": 0, "skipped": 0,
              "signal-high": 0, "signal-medium": 0, "signal-low": 0}

    for source in sources:
        result = _run_source(source, args, store, profile)
        if not result:
            continue
        for k, v in result.items():
            totals[k] = totals.get(k, 0) + v

    print()
    print("=== Summary ===")
    for k in ("added", "updated", "skipped", "signal-high", "signal-medium", "signal-low"):
        print(f"  {k:14s} {totals.get(k, 0)}")


if __name__ == "__main__":
    main()
