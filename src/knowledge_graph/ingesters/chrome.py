"""
Chrome bookmarks ingester.

Reads ~/.config/google-chrome/Default/Bookmarks (JSON) and produces one
BookmarkRecord per URL bookmark. Folder names walk up to a breadcrumb tag list.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from knowledge_graph.ingesters.common import BookmarkRecord


DEFAULT_CHROME_PATH = Path.home() / ".config" / "google-chrome" / "Default" / "Bookmarks"


def _chrome_microseconds_to_dt(raw: str | int | None) -> datetime | None:
    """Chrome stores timestamps as microseconds since 1601-01-01 UTC."""
    if not raw:
        return None
    try:
        micros = int(raw)
    except (TypeError, ValueError):
        return None
    if micros <= 0:
        return None
    # 11644473600 seconds between 1601-01-01 and 1970-01-01
    unix_seconds = micros / 1_000_000 - 11644473600
    try:
        return datetime.fromtimestamp(unix_seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _walk(node: dict, folder_path: list[str]) -> Iterator[BookmarkRecord]:
    node_type = node.get("type")
    if node_type == "url":
        url = node.get("url", "")
        if not url or url.startswith("javascript:"):
            return
        yield BookmarkRecord(
            url=url,
            title=node.get("name", ""),
            platform="chrome",
            folder_path=list(folder_path),
            added_at=_chrome_microseconds_to_dt(node.get("date_added")),
        )
    elif node_type == "folder":
        next_path = folder_path + [node.get("name", "")] if node.get("name") else list(folder_path)
        for child in node.get("children", []):
            yield from _walk(child, next_path)


def read_chrome_bookmarks(path: Path = DEFAULT_CHROME_PATH) -> Iterator[BookmarkRecord]:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for root_name, root in data.get("roots", {}).items():
        if not isinstance(root, dict):
            continue
        yield from _walk(root, [root_name])
