"""
YouTube likes + watch-later ingester.

Uses google-api-python-client. One-time setup:

  1. Create a Google Cloud project; enable "YouTube Data API v3".
  2. Create an OAuth 2.0 Desktop-app credential; download as JSON.
  3. Save to ~/.config/google/youtube_oauth_client.json
       (or set YOUTUBE_OAUTH_CLIENT_PATH)
  4. pip install google-api-python-client google-auth-oauthlib
  5. First run launches a browser for consent; token cached next to the client json.

Ingests:
  - "LL" playlist (liked videos)  — always present, owned by the user
  - "WL" playlist (watch later)   — private; read via authenticated mine=true
  - Optional: user-created "Saved" playlists by title prefix (env YOUTUBE_PLAYLIST_PREFIX)

Note: YouTube API v3 does NOT expose a separate "Saves" list anymore — the
concept is folded into playlists. Watch-later is the closest thing to a save.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from knowledge_graph.ingesters.common import BookmarkRecord


DEFAULT_CLIENT_PATH = Path.home() / ".config" / "google" / "youtube_oauth_client.json"
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "google" / "youtube_token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]


def _authenticate():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
        from google.oauth2.credentials import Credentials  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Missing YouTube deps. Run: pip install google-api-python-client google-auth-oauthlib"
        ) from e

    client_path = Path(os.environ.get("YOUTUBE_OAUTH_CLIENT_PATH") or DEFAULT_CLIENT_PATH)
    token_path = Path(os.environ.get("YOUTUBE_OAUTH_TOKEN_PATH") or DEFAULT_TOKEN_PATH)
    if not client_path.exists():
        raise RuntimeError(
            f"YouTube OAuth client JSON not found at {client_path}. See youtube.py docstring."
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return creds


def _youtube_service():
    try:
        from googleapiclient.discovery import build  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Missing googleapiclient. Run: pip install google-api-python-client"
        ) from e
    return build("youtube", "v3", credentials=_authenticate(), cache_discovery=False)


def _iter_playlist_items(service, playlist_id: str, collection: str) -> Iterator[BookmarkRecord]:
    page_token = None
    while True:
        request = service.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        )
        resp = request.execute()
        for item in resp.get("items", []):
            snippet = item.get("snippet", {})
            vid = item.get("contentDetails", {}).get("videoId") or snippet.get("resourceId", {}).get("videoId")
            if not vid:
                continue
            url = f"https://www.youtube.com/watch?v={vid}"
            added = snippet.get("publishedAt")
            added_dt = None
            if added:
                try:
                    added_dt = datetime.fromisoformat(added.replace("Z", "+00:00"))
                except ValueError:
                    added_dt = None
            channel = snippet.get("videoOwnerChannelTitle") or snippet.get("channelTitle") or ""
            yield BookmarkRecord(
                url=url,
                title=snippet.get("title", "") or "",
                platform="youtube",
                folder_path=[collection],
                added_at=added_dt,
                collection=collection,
                extra_tags=[f"yt-channel-{channel.lower().replace(' ', '-')[:40]}"] if channel else [],
                extra_properties={
                    "video_id": vid,
                    "channel": channel,
                },
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            return


def read_youtube_saves() -> Iterator[BookmarkRecord]:
    service = _youtube_service()
    # Liked videos: "LL" per-channel ID. We need the authenticated user's actual LL playlist id.
    channels = service.channels().list(part="contentDetails", mine=True).execute()
    items = channels.get("items", [])
    if not items:
        return
    related = items[0].get("contentDetails", {}).get("relatedPlaylists", {})

    liked_playlist = related.get("likes")
    if liked_playlist:
        yield from _iter_playlist_items(service, liked_playlist, collection="liked")

    watch_later = related.get("watchLater")
    if watch_later:
        try:
            yield from _iter_playlist_items(service, watch_later, collection="watch-later")
        except Exception:
            # Watch-later has been read-only for third-party apps since ~2016.
            # Skip silently if YouTube refuses.
            pass

    # Optional: user-created playlists whose title starts with a prefix
    prefix = os.environ.get("YOUTUBE_PLAYLIST_PREFIX")
    if prefix:
        page_token = None
        while True:
            pl = service.playlists().list(
                part="snippet", mine=True, maxResults=50, pageToken=page_token
            ).execute()
            for p in pl.get("items", []):
                title = p.get("snippet", {}).get("title", "")
                if title.startswith(prefix):
                    yield from _iter_playlist_items(service, p["id"], collection=title)
            page_token = pl.get("nextPageToken")
            if not page_token:
                break
