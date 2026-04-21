"""
Instagram saved-posts + saved-reels ingester.

Reads from a Meta Account Data Export zip (or an unzipped directory).
Request the export at:
   Instagram → Settings → Account Center → Your info & permissions → Download your information
Select format=JSON, date-range=all, data=Saved posts + Saved reels.
Meta emails a download link (can take up to 48h).

Expected file paths inside the export (paths have shifted over the years;
the ingester tolerates both common layouts):

  your_instagram_activity/saved/saved_posts.json
  your_instagram_activity/saved/saved_collections.json
  your_instagram_activity/saved/saved_reels.json

Folder/collection context: Instagram lets you save into named collections.
When present, collection name becomes BookmarkRecord.collection and is
scored + tagged accordingly.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from knowledge_graph.ingesters.common import BookmarkRecord


CANDIDATE_SAVED_PATHS = [
    "your_instagram_activity/saved/saved_posts.json",
    "your_instagram_activity/saved/saved_reels.json",
    "your_instagram_activity/saved/saved_collections.json",
    # Older export layouts
    "saved/saved_posts.json",
    "saved/saved_reels.json",
    "saved/saved_collections.json",
    "content/saved_posts.json",
]


def _ts_to_dt(raw) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _extract_url(item: dict) -> str | None:
    """Meta exports shift URL location between schema versions."""
    if "string_map_data" in item:
        smd = item["string_map_data"] or {}
        for key in ("Saved on", "Link", "URL", "Href"):
            v = (smd.get(key) or {}).get("href")
            if v:
                return v
    if "media_list_data" in item:
        for media in item["media_list_data"] or []:
            v = media.get("uri") or media.get("url")
            if v:
                return v
    for k in ("uri", "url", "href"):
        if isinstance(item.get(k), str):
            return item[k]
    return None


def _extract_title(item: dict) -> str:
    if "title" in item and isinstance(item["title"], str):
        return item["title"]
    if "string_map_data" in item:
        smd = item["string_map_data"] or {}
        for key in ("Title", "Caption", "Author"):
            v = (smd.get(key) or {}).get("value")
            if v:
                return v
    return ""


def _extract_timestamp(item: dict) -> datetime | None:
    for key in ("timestamp", "creation_timestamp", "saved_timestamp"):
        if key in item:
            dt = _ts_to_dt(item[key])
            if dt:
                return dt
    if "string_map_data" in item:
        smd = item["string_map_data"] or {}
        for key in ("Saved on", "Time"):
            ts = (smd.get(key) or {}).get("timestamp")
            if ts:
                return _ts_to_dt(ts)
    return None


def _iter_export_json(export_root: Path, zip_path: Path | None) -> Iterator[tuple[str, dict]]:
    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            for candidate in CANDIDATE_SAVED_PATHS:
                try:
                    with zf.open(candidate) as f:
                        yield candidate, json.load(f)
                except KeyError:
                    continue
    else:
        for candidate in CANDIDATE_SAVED_PATHS:
            p = export_root / candidate
            if p.exists():
                yield candidate, json.loads(p.read_text(encoding="utf-8"))


def _iter_payload(payload: dict, source_file: str) -> Iterator[tuple[str, dict]]:
    """Yield (section_name, item) from a Meta saved_*.json."""
    if isinstance(payload, list):
        for item in payload:
            yield source_file, item
        return
    for key, val in payload.items():
        if isinstance(val, list):
            for item in val:
                yield key, item


def read_instagram_saves(
    export_path: Path,
) -> Iterator[BookmarkRecord]:
    """
    Args:
        export_path: Either a Meta export zip or an unzipped directory.
    """
    if not export_path.exists():
        raise RuntimeError(f"Instagram export not found: {export_path}")

    if export_path.is_dir():
        export_root, zip_path = export_path, None
    elif export_path.suffix.lower() == ".zip":
        export_root, zip_path = export_path.parent, export_path
    else:
        raise RuntimeError(f"Expected a zip or directory, got: {export_path}")

    for source_file, payload in _iter_export_json(export_root, zip_path):
        collection_hint = "reels" if "reels" in source_file else "posts"
        for section, item in _iter_payload(payload, source_file):
            url = _extract_url(item)
            if not url:
                continue
            title = _extract_title(item)
            dt = _extract_timestamp(item)

            # Collection detection: Meta uses a different key per version.
            # If this item came from saved_collections.json, the section name is the collection.
            collection = None
            if "collections" in source_file or section.lower().startswith("saved_saved"):
                collection = section
            elif isinstance(item, dict) and isinstance(item.get("title"), str) and "collection" in item.get("title", "").lower():
                collection = item["title"]

            folder_path = ["instagram", collection_hint]
            if collection:
                folder_path.append(collection)

            yield BookmarkRecord(
                url=url,
                title=title or f"Instagram {collection_hint[:-1]}",
                platform="instagram",
                folder_path=folder_path,
                added_at=dt,
                collection=collection,
                extra_tags=[f"ig-{collection_hint}"],
                extra_properties={
                    "export_file": source_file,
                    "section": section,
                },
            )
