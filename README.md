# knowledge-graph

[![tests](https://img.shields.io/github/actions/workflow/status/oibars/knowledge-graph/test.yml?branch=main&label=tests)](https://github.com/oibars/knowledge-graph/actions/workflows/test.yml)
[![license](https://img.shields.io/github/license/oibars/knowledge-graph)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![mcp](https://img.shields.io/badge/MCP-native-orange)](https://modelcontextprotocol.io)

Local-first, MCP-native knowledge graph that turns your bookmarks, saves, and reads into a graph your AI agent queries as a tool. Not a notebook you maintain.

**Obsidian assumes you write notes. This doesn't.** Your firehose of saved content (Chrome bookmarks, Reddit saves, YouTube likes, Instagram saves, manually captured articles) becomes the corpus. Claude Code, Cursor, or any MCP client gets your saved-content context preloaded as a tool.

What it gives you. Local-first execution that runs on your machine with no cloud, no login, no telemetry. MCP-native tooling that exposes `kg_search`, `kg_add_entity`, `kg_get_neighbors` and 10 more tools so AI agents reach for it. No notes required because the ingest pipeline does the work and you keep saving things normally. **Hybrid retrieval**: tokenized lexical search blended with local semantic similarity (Ollama embeddings, when available), importance, and recency — degrading gracefully to lexical-only when Ollama is down. Deterministic relevance scoring at ingest time from a lexical interest profile built from your own files and graph entities, with no LLM call needed. Built for concurrent use: a long-running MCP server and cron ingesters can share the same database safely (WAL journaling), and the server picks up external writes without a restart.

## Quickstart

```bash
git clone https://github.com/oibars/knowledge-graph.git
cd knowledge-graph
python -m venv .venv && source .venv/bin/activate
pip install -e .
kg-ingest-bookmarks --source chrome   # ingest your Chrome bookmarks
```

Then add this to `~/.claude/settings.json` so Claude Code reaches the graph as a tool.

```json
{ "mcpServers": { "knowledge-graph": { "command": "kg-mcp" } } }
```

Restart Claude Code, and `kg_search`, `kg_get_neighbors`, and friends are available in any session.

## What it ingests

| Source | Auth | Status |
|---|---|---|
| Chrome bookmarks | none (local file) | working |
| Firefox bookmarks | none (local file) | working |
| Reddit saves | OAuth refresh-token (SSO-compatible helper included) | working |
| YouTube likes + watch-later | Google OAuth (one-time consent) | working |
| Instagram saved posts/reels | Meta data export ZIP | working |
| Article / PDF / podcast / paper | manual capture via skill or template | working |

Every item lands as `Entity(label="Document")` with an `importance_score` (0.0 to 1.0) and a `signal-high | signal-medium | signal-low` tag. Re-runs are idempotent (content_hash dedup).

## Install

```bash
git clone https://github.com/oibars/knowledge-graph.git
cd knowledge-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[reddit,youtube]"   # extras optional per source
```

Requires Python ≥ 3.11.

## MCP server (Claude Code, Cursor, Claude Desktop)

```bash
kg-mcp
```

Add to `~/.claude/settings.json` (or your client's MCP config).

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "command": "kg-mcp"
    }
  }
}
```

### Available MCP tools

| Tool | Purpose |
|---|---|
| `kg_search` | Hybrid search (lexical + semantic + importance + recency); filters: `label`, `tags`, `source_app`, `created_after`/`created_before` |
| `kg_get_entity` | Retrieve an entity by id |
| `kg_get_neighbors` | Neighbors at depth 1 to 3, optionally filtered by `relation_type` |
| `kg_find_path` | Shortest path between two entities |
| `kg_get_relations` | An entity's typed relations (`in` / `out` / `both`) |
| `kg_find_by_tag` | List entities carrying a tag, ranked by importance |
| `kg_find_similar` | Semantically similar entities via stored embeddings |
| `kg_get_stats` | Graph statistics |
| `kg_add_entity` | Create entity (content-identical adds dedupe to the existing id) |
| `kg_add_entities` | Bulk add in one call |
| `kg_add_relation` | Link two entities (relation type validated) |
| `kg_update_entity` | Update fields |
| `kg_delete_entity` | Delete entity and its relations |

### Reliability

- **Safe under concurrency** — every connection uses WAL + busy timeouts; the MCP server detects external writes (cron ingesters) and reloads automatically, so it never serves stale answers.
- **Upsert by content** — entities carry a `content_hash`; re-adding identical name+description returns the existing entity instead of minting a duplicate.
- **Validated relations** — `relation_type` is checked against the vocabulary; bidirectional relations persist their inverse edge (it survives restarts).
- **Auto-snapshots** — JSON snapshots every 50 writes / 24h with rotation (`KG_SNAPSHOT_DIR` relocates them). Snapshots are a second line of defense: still back up the data dir itself.

## Ingest your saves

```bash
# Run all locally-available sources
kg-ingest-bookmarks

# One source at a time
kg-ingest-bookmarks --source chrome
kg-ingest-bookmarks --source firefox
kg-ingest-bookmarks --source reddit       # needs Reddit env vars (see below)
kg-ingest-bookmarks --source youtube      # needs Google OAuth client (see below)
kg-ingest-bookmarks --source instagram --path ~/Downloads/meta_export.zip

# Inspect the interest profile that drives scoring
kg-ingest-bookmarks --show-profile

# Score-only, no DB writes
kg-ingest-bookmarks --source chrome --dry-run
```

### Reddit setup (OAuth refresh-token, works with Google-SSO accounts)

1. Create a "script"-type app at https://www.reddit.com/prefs/apps with redirect URI `http://localhost:8080`.
2. Set in your shell env.
   ```
   REDDIT_CLIENT_ID
   REDDIT_CLIENT_SECRET
   REDDIT_USER_AGENT="knowledge-graph/1.0 by <username>"
   ```
3. One-time OAuth dance to get a permanent refresh-token (no Reddit password needed).
   ```bash
   python scripts/reddit_oauth_setup.py
   ```
   Click Allow in the browser. The script prints the line to add.
   ```
   set -gx REDDIT_REFRESH_TOKEN "<token>"
   ```
4. Then `kg-ingest-bookmarks --source reddit`.

If you have a Reddit-native password, set `REDDIT_USERNAME` and `REDDIT_PASSWORD` instead and skip step 3.

### YouTube setup

1. In Google Cloud Console, enable "YouTube Data API v3".
2. Create an OAuth 2.0 Desktop client, download the JSON, save it to `~/.config/google/youtube_oauth_client.json`.
3. Add yourself as a Test user on the OAuth consent screen.
4. Run `kg-ingest-bookmarks --source youtube`. The first run opens a browser for consent. The token caches at `~/.config/google/youtube_token.json` and subsequent runs are silent.

### Instagram setup

Request a Meta data export (Account Center, Your info & permissions, Download your information, JSON, "Saved posts + Saved reels"). Meta emails the link within 48h. Then run.

```bash
kg-ingest-bookmarks --source instagram --path ~/Downloads/meta_export.zip
```

## Build an always-on ingester (`knowledge_graph.pipeline`)

For continuous sources (meeting transcripts, trend feeds), `knowledge_graph.pipeline` is the shared implementation — write a thin fetcher, not another Entity constructor:

```python
from knowledge_graph.pipeline import store_for, ingest_staging_jsonl, trend_entity

# Conversation-shaped staging JSONL: {"id", "title", "date", "topics": [], "speakers": [], "text"}
n = ingest_staging_jsonl("myrecorder", "staging.jsonl", scope="professional")

# Trend-feed items: {"title", "summary", "url", "category", "keywords": [], "popularity", "publishedAt"}
store = store_for("professional")
store.add_entity(trend_entity(item, source="mynewsfeed"))
```

Both paths get stable ids, domain stamping, and content-hash dedup for free — re-runs are idempotent.

## Capture a single source (article, PDF, podcast)

For deliberate captures with a written thesis instead of bulk firehose ingest, use the `capture-source` skill (`~/.claude/skills/capture-source/SKILL.md`) or copy `sources/_TEMPLATE.md` and run.

```bash
kg-index-sources
```

Each capture produces an `Entity(Document)` plus one `Entity(Concept)` per claim plus an author `Entity(Person)` plus `[[wikilink]]` references.

## Index installed Claude Code agents

```bash
kg-index-agents
kg-index-agents --agents-dir /path/to/agents
kg-index-agents --dry-run
```

After indexing, agents are searchable as a routing layer.

```
kg_search("debug a production memory leak", label="Agent")
→ engineering-incident-response-commander, engineering-sre, testing-evidence-collector
```

Agent indexing and the optional `SemanticLinker` embed text with a local Ollama model (`nomic-embed-text` by default, see Configuration). No cloud key needed. If Ollama isn't running, indexing still works — it just skips the embeddings and falls back to lexical search.

## Python API

```python
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.models import Entity, Relation

store = KnowledgeGraphStore()  # ~/.knowledge-graph/data by default

store.add_entity(Entity(id="e1", label="Concept", name="Authentication", tags=["security"]))
store.add_relation(Relation(id="r1", source_id="e1", target_id="e2", relation_type="depends_on"))

results = store.search_entities("authentication security")
path = store.find_path("e1", "e5")
neighbors = store.get_neighbors("e1", depth=2)
```

## Entity labels

`Concept` `Document` `Person` `Tag` `File` `Folder` `Agent` `Skill` `Tool` `Decision` `Task` `Bug` `Session` `AgentOutput` `Experiment` `Code` `Event`

## Relation types

`contains` `depends_on` `implements` `references` `similar_to` `contradicts` `prerequisite_for` `learned_from` `authored_by` `located_in` `part_of` `uses` `produces` `influenced_by` `routed_to` `spawned_by` `resolves` `supersedes` `tracked_in` `related_to` — plus the auto-created inverse forms (`required_by`, `implemented_by`, `referenced_by`, `requires`, `teaches`, `authored`, `used_by`, `produced_by`, `influenced`). Unknown types are rejected at write time.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `KG_DATA_DIR` | `~/.knowledge-graph/data` | SQLite + snapshot storage |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | (none) | Reddit script app credentials |
| `REDDIT_REFRESH_TOKEN` | (none) | preferred over username/password |
| `REDDIT_USERNAME` / `REDDIT_PASSWORD` | (none) | password-grant fallback |
| `YOUTUBE_OAUTH_CLIENT_PATH` | `~/.config/google/youtube_oauth_client.json` | YouTube OAuth client |
| `YOUTUBE_OAUTH_TOKEN_PATH` | `~/.config/google/youtube_token.json` | cached refresh token |
| `YOUTUBE_PLAYLIST_PREFIX` | (none) | also ingest user playlists with this title prefix |
| `OLLAMA_URL` | `http://localhost:11434` | local Ollama endpoint for semantic search, agent indexing, and linking |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model — **must match the model that embedded your stored entities**; mixed vector spaces are skipped via a dimension guard |
| `OLLAMA_CHAT_MODEL` | `gemma4:e4b` | chat model for concept extraction in `SemanticLinker` |
| `KG_SNAPSHOT_DIR` | `<data_dir>/kg_snapshots` | relocate auto-snapshots (e.g. into a backup folder) |
| `KG_MEMORY_DIR` | newest `~/.claude/projects/*/memory` | memory-files source for the interest profile |

## What this is not

This is not a hosted SaaS. It runs locally and you control the data. It is not an LLM in itself. Your existing AI agent (Claude, Cursor, and others) does the synthesis. It is not a substitute for Obsidian if you actually like writing notes. Different paradigm. There is no OCR for image-only PDFs. Semantic *search* activates automatically when Ollama is running and entities carry embeddings — and falls back to lexical when it isn't. Semantic *linking* (`SemanticLinker`) is optional and runs separately, recommended only after 30+ sources exist.

## Tests

```bash
pytest
pytest -v tests/test_entity_crud.py
```

## Other things I'm building

If this is useful, you might like the other projects I work on.

[**Product Leader Academy**](https://www.productleaderacademy.com). Community and LMS for product leaders. Free-access model with a Premium AI Coach add-on. Next.js, Neon Postgres, Vercel.

[**SwiftRecap**](https://swiftrecap.vercel.app). Autonomous AI tech newsletter for engineers, founders, and PMs. Aggregates RSS, Reddit, GitHub, and HN into weekly digests. Stripe subscriptions plus CPM ad slots.

[**code2figma**](https://code2figma.vercel.app). Convert React, Vue, and Angular components to Figma designs via CLI plus a Figma plugin. Auto-detects the framework and creates structured frames.

[**Imperfit**](https://imperfit.vercel.app). Chat-first AI nutrition app for people who refuse to log meals. Flutter and FastAPI, edge AI (Liquid LFM 2.5) plus Gemini 3 Flash, 3-second voice or photo logging.

## License

MIT
