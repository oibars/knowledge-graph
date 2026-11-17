# Recall validation — 2026-10-05

Hybrid recall now normalizes dates to aware UTC when reading or writing SQLite,
serializing snapshots, applying date filters, and ranking recency. New model
timestamps use UTC. Legacy naive timestamps retain their host-local meaning;
they are not reinterpreted as UTC. Use offsets or `Z` in filters to make their
meaning independent of the host timezone. Production databases were untouched.

The first synthetic profile attributed 85% of recall CPU time to tokenization.
Sorting took about 0.3%. The implementation caches entity lexical fields inside
each store, checks mutable content on every search, and invalidates entries on
upsert, deletion, and database reload. It does not cache query embeddings or
replace sorting: provider latency and an indexing bottleneck were not established.

## Reproduce

Run from the repository in an environment with its dependencies installed:

```sh
python scripts/benchmark_recall.py --entities 3000 --repeats 7 --uncached
python scripts/benchmark_recall.py --entities 3000 --repeats 7
python scripts/benchmark_recall.py --entities 3000 --repeats 7 --uncached --profile
```

The benchmark creates and deletes a temporary SQLite graph. It fixes the random
seed and ranking clock and uses generated documents and deterministic fake
64-dimensional embeddings. It never reads user graphs or calls an embedding
provider. `--uncached` runs the prior lexical field calculation for comparison.
`--profile` adds profiler overhead, so use unprofiled runs to compare latency.

## Observed results

3,000 entities, 200-word synthetic descriptions, seven warm queries per mode:

| Mode | Uncached cold | Cached cold | Uncached warm p50 / p95 | Cached warm p50 / p95 |
|---|---:|---:|---:|---:|
| Lexical | 66.066 ms | 69.364 ms | 65.216 / 66.046 ms | 2.989 / 3.499 ms |
| Hybrid | 74.595 ms | 80.824 ms | 74.499 / 75.501 ms | 13.108 / 14.315 ms |

Warm p50 improved about 22× for lexical and 5.7× for hybrid search on this corpus.
Cold queries still tokenize the corpus and pay a small cache population overhead.
Top-result order and rounded-score SHA-256 digests matched between both modes.
The same eight fake embedding calls were made in each run.

These measurements establish a repeated-tokenization improvement, not production
latency or retrieval quality. The generated vocabulary is small; memory use and
latency on real corpora will differ. The token cache uses additional memory and
is bounded by live entities in its owning store.

## Verification and attribution

Five mixed-date and storage regressions reproduced failures before the fix.
The full suite passed after the change (96 tests), including host-timezone
regressions for UTC and America/New_York. Final full-suite verification blocked
socket connections. Cache tests cover direct field/list mutations, replacement,
deletion, store isolation, and an external database write followed by refresh.

Existing uncommitted agent pruning and same-id upsert preservation changes, plus
their tests, were retained unchanged. No graph collection, ingestion, service
restart, cloud call, commit, or push was performed.
