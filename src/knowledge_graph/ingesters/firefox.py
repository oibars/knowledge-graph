"""
Firefox bookmarks ingester.

Reads places.sqlite (copies it first — Firefox holds a lock on the live DB).
Joins moz_bookmarks against moz_places and walks parent rows to build the
folder breadcrumb.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from knowledge_graph.ingesters.common import BookmarkRecord


DEFAULT_FIREFOX_DIR = Path.home() / ".config" / "mozilla" / "firefox"


def find_places_db(profile_dir: Path = DEFAULT_FIREFOX_DIR) -> Path | None:
    """Locate the default Firefox profile's places.sqlite."""
    if not profile_dir.exists():
        return None
    # Prefer *.default-release over *.default
    candidates = sorted(profile_dir.glob("*.default-release/places.sqlite"))
    if not candidates:
        candidates = sorted(profile_dir.glob("*.default*/places.sqlite"))
    return candidates[0] if candidates else None


def _firefox_micros_to_dt(micros: int | None) -> datetime | None:
    if not micros:
        return None
    try:
        return datetime.fromtimestamp(int(micros) / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def read_firefox_bookmarks(db_path: Path | None = None) -> Iterator[BookmarkRecord]:
    db_path = db_path or find_places_db()
    if not db_path or not db_path.exists():
        return

    # Firefox holds an exclusive lock while running; snapshot first.
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        snapshot = Path(tmp.name)
    try:
        shutil.copy2(db_path, snapshot)
        conn = sqlite3.connect(snapshot)
        conn.row_factory = sqlite3.Row

        folder_titles = {
            row["id"]: row["title"] or ""
            for row in conn.execute(
                "SELECT id, title FROM moz_bookmarks WHERE type=2"
            )
        }
        parent_of = {
            row["id"]: row["parent"]
            for row in conn.execute("SELECT id, parent FROM moz_bookmarks")
        }

        def breadcrumb(start_parent: int) -> list[str]:
            path: list[str] = []
            seen: set[int] = set()
            node = start_parent
            while node and node not in seen:
                seen.add(node)
                title = folder_titles.get(node, "")
                if title:
                    path.append(title)
                node = parent_of.get(node) or 0
                if node == 0:
                    break
            path.reverse()
            return path

        cursor = conn.execute(
            """
            SELECT b.title AS title, p.url AS url, b.dateAdded AS date_added, b.parent AS parent
            FROM moz_bookmarks b
            JOIN moz_places p ON b.fk = p.id
            WHERE b.type = 1 AND p.url IS NOT NULL AND p.url NOT LIKE 'place:%'
            """
        )
        for row in cursor:
            url = row["url"] or ""
            if not url or url.startswith("javascript:"):
                continue
            yield BookmarkRecord(
                url=url,
                title=row["title"] or "",
                platform="firefox",
                folder_path=breadcrumb(row["parent"]),
                added_at=_firefox_micros_to_dt(row["date_added"]),
            )
        conn.close()
    finally:
        try:
            snapshot.unlink()
        except OSError:
            pass
