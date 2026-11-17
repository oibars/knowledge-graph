"""Repeatable synthetic recall benchmark. Never opens a user's graph or Ollama."""

import argparse
import cProfile
import hashlib
import json
import pstats
import random
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone

import structlog

from knowledge_graph.models import Entity
from knowledge_graph.services.graph_store import KnowledgeGraphStore
from knowledge_graph.services import graph_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--uncached", action="store_true",
                        help="Reference: retokenize each entity on every query")
    args = parser.parse_args()
    if args.entities < 1 or args.repeats < 1:
        parser.error("entities and repeats must be positive")
    structlog.configure(logger_factory=structlog.ReturnLoggerFactory())
    rng = random.Random(20261005)
    topics = ["routing", "database", "security", "memory", "design", "testing"]
    # Fix data dates and the ranking clock so scores/order reproduce on any host.
    born = datetime(2026, 1, 1, tzinfo=timezone.utc)
    graph_store.utc_now = lambda: datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory(prefix="kg-recall-benchmark-") as temporary:
        store = KnowledgeGraphStore(data_dir=temporary, enable_snapshots=False)
        entities = []
        for index in range(args.entities):
            topic = topics[index % len(topics)]
            entities.append(Entity(
                id=f"synthetic-{index}", label="Document",
                name=f"{topic} notes {index}",
                description=" ".join(rng.choice(topics) for _ in range(200)),
                tags=[topic], topics=[topics[(index + 1) % len(topics)]],
                embedding=[rng.uniform(-1, 1) for _ in range(64)],
                importance_score=(index % 11) / 10,
                created_at=born + timedelta(hours=index),
                updated_at=born + timedelta(hours=index),
            ))
        # Bulk seed only this temporary DB; population is outside measured time.
        with store._connect() as connection:
            connection.executemany(
                "INSERT INTO entities VALUES (" + ",".join(["?"] * 20) + ")",
                [store._entity_to_row(entity) for entity in entities],
            )
        store._load_from_database()
        store._rebuild_hash_index()
        if args.uncached:
            store._lexical_fields = lambda entity: (
                entity.name.lower(), (entity.description or "").lower(),
                store._tokenize(entity.name), store._tokenize(entity.description or ""),
                store._tokenize(" ".join(entity.tags)), store._tokenize(" ".join(entity.topics)),
            )
        embed_calls = 0

        def fake_embed(query):
            nonlocal embed_calls
            embed_calls += 1
            return [1.0] + [0.0] * 63

        store._embed_query = fake_embed
        profiler = cProfile.Profile()
        report = {"entities": args.entities, "repeats": args.repeats, "uncached": args.uncached,
                  "profiled": args.profile, "seed": 20261005,
                  "embedding": "deterministic fake; no network"}
        for semantic in (False, True):
            store._lexical_cache.clear()
            elapsed = []
            # Cold first query and warm repeats are reported separately.
            first_ms = None
            hit_ids = None
            for repeat in range(args.repeats + 1):
                start = time.perf_counter()
                if args.profile and repeat:
                    profiler.enable()
                hits = store.search_entities_scored("routing database", semantic=semantic)
                profiler.disable()
                duration = (time.perf_counter() - start) * 1000
                ids = [entity.id for _, entity in hits]
                if hit_ids is not None and ids != hit_ids:
                    raise RuntimeError("Repeated search changed result ordering")
                hit_ids = ids
                if repeat:
                    elapsed.append(duration)
                else:
                    first_ms = duration
            ordered = sorted(elapsed)
            report["hybrid" if semantic else "lexical"] = {
                "cold_ms": round(first_ms, 3),
                "warm_p50_ms": round(statistics.median(elapsed), 3),
                "warm_p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * .95))], 3),
                "top_ids_sha256": hashlib.sha256(json.dumps(hit_ids).encode()).hexdigest(),
                "scores_sha256": hashlib.sha256(json.dumps([
                    round(score, 12) for score, _ in hits
                ]).encode()).hexdigest(),
            }
        report["fake_embedding_calls"] = embed_calls
        print(json.dumps(report, indent=2))
        if args.profile:
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
