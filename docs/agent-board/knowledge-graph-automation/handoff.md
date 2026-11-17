# Handoff
Changed:
- `knowledge-graph`: capture retry bound (`MAX_ATTEMPTS` dead-letter), per-part cache invalidation, projection short-circuit + busy timeout + closed connections, utterance-safe projection chunks, snapshot slimming (gzip, no vectors, tiered retention), relevance-gated importance/recency, cached float32 vector matrix, `query_vector` reuse, `clear_embedding()`, export-only stats + cached approved graph.
- `persona-os`: kind/status JSON-Schema coupling, `calendar_dates` hints, merge fixes (small talk no longer downgrades; mixed slice rebuilt from evidence), retry ordering, staged status files (`reconcile_status.json`), failure tracebacks (`failures.log`), flock sync lock, shell exit-code propagation, Drive dataless fallback, incremental `kg_kb_fulltext`, review briefs with `presence_penalty 0`, new `kg_tag_embedding_provenance.py`, trial window restart.
- `secondbrain`: shared prepare/finish path, `/chat/stream` SSE with Stop, cloud fallback rebuilt from full local evidence + local history, grouped `[S1, S2]` citations, truncation kept with a marker, inline feedback with maintenance minutes, thread summarize control, parent context in recall, retired `gpu-guard.sh`.
Verified:
- `knowledge-graph` 227 passed; `persona-os` 175 passed; `secondbrain` 32 passed.
- Live reconcile: `{"validated":3,"projected":88,"failed":0,"budget_exhausted":true}`, `data/memory/reconcile_status.json` written, validated captures 73 → 76.
- Live `/chat/stream`: 264 delta frames from `jarvis-recall:qwen3.5-9b`, `done` with `citation_status=present`, answer persisted, one trial event recorded.
- `sync.sh` returns `already running, skip` with the flock held; stale `data/fieldy/.sync.lock` mkdir lock removed.
Next agent should: commit the three repos separately, watch `data/memory/reconcile_status.json` and the 2,304-capture pending queue over the next runs; the 66 legacy `ValueError` rows will dead-letter on their next failure under the new bound.
