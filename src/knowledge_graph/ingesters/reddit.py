"""
Reddit saved-posts ingester.

Uses PRAW (script-type OAuth). One-time setup:

  1. Create a Reddit app at https://www.reddit.com/prefs/apps (type: "script")
  2. Export these env vars (e.g. in ~/.config/fish/env.fish):
       set -Ux REDDIT_CLIENT_ID     <client-id>
       set -Ux REDDIT_CLIENT_SECRET <client-secret>
       set -Ux REDDIT_USERNAME      <your reddit username>
       set -Ux REDDIT_PASSWORD      <your reddit password>
       set -Ux REDDIT_USER_AGENT    "knowledge-graph/1.0 by <username>"
  3. pip install praw  (or add `reddit` to pyproject optional deps)
  4. Run: kg-ingest-bookmarks --source reddit

Saved posts become BookmarkRecord(platform="reddit", collection=subreddit).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterator

from knowledge_graph.ingesters.common import BookmarkRecord


REQUIRED_BASE = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
PASSWORD_GRANT = ["REDDIT_USERNAME", "REDDIT_PASSWORD"]


def missing_env() -> list[str]:
    base_missing = [k for k in REQUIRED_BASE if not os.environ.get(k)]
    if base_missing:
        return base_missing
    if os.environ.get("REDDIT_REFRESH_TOKEN"):
        return []
    return [k for k in PASSWORD_GRANT if not os.environ.get(k)]


def read_reddit_saved(limit: int = 1000) -> Iterator[BookmarkRecord]:
    """Yield BookmarkRecord per saved Reddit submission (comments skipped)."""
    try:
        import praw  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "praw is not installed. Run: pip install praw  (or install the [reddit] extra)"
        ) from e

    missing = missing_env()
    if missing:
        raise RuntimeError(
            f"Missing Reddit env vars: {', '.join(missing)}. See reddit.py docstring."
        )

    common = dict(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "knowledge-graph/1.0"),
    )
    if refresh := os.environ.get("REDDIT_REFRESH_TOKEN"):
        reddit = praw.Reddit(refresh_token=refresh, **common)
    else:
        reddit = praw.Reddit(
            username=os.environ["REDDIT_USERNAME"],
            password=os.environ["REDDIT_PASSWORD"],
            **common,
        )

    redditor = reddit.user.me()
    for item in redditor.saved(limit=limit):
        is_submission = hasattr(item, "title") and hasattr(item, "url")
        if not is_submission:
            continue  # skip saved comments for now — their "url" is the thread link
        subreddit = getattr(item.subreddit, "display_name", None) or ""
        yield BookmarkRecord(
            url=item.url,
            title=getattr(item, "title", "") or "",
            platform="reddit",
            folder_path=["saved"],
            added_at=datetime.fromtimestamp(getattr(item, "created_utc", 0), tz=timezone.utc)
            if getattr(item, "created_utc", None)
            else None,
            collection=subreddit,
            extra_tags=[f"r-{subreddit.lower()}"] if subreddit else [],
            extra_properties={
                "reddit_id": getattr(item, "id", None),
                "permalink": f"https://reddit.com{getattr(item, 'permalink', '')}",
                "score": getattr(item, "score", None),
                "num_comments": getattr(item, "num_comments", None),
            },
        )
