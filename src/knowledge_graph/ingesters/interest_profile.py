"""
Interest Profile

Builds a keyword → weight map from the user's Claude memory files and the
existing knowledge graph. Used by ingesters to score bookmarks for relevance.

Sources of signal:
  - the newest ~/.claude/projects/*/memory/*.md dir, or KG_MEMORY_DIR (project/user/feedback/reference memories)
  - Existing Document, Concept, Tag entities in the graph
  - Known-interest domains (github.com, vercel.com, …) with static weights

Design:
  - Lexical matching only. No embeddings, no LLM calls — bookmark ingest
    runs on thousands of items; must be fast and deterministic.
  - Stopword filter removes common English noise.
  - Multi-token phrases count more than single tokens when a phrase hits.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from knowledge_graph.services.graph_store import KnowledgeGraphStore


def _default_memory_dir() -> Path:
    """Resolve the Claude Code memory dir: KG_MEMORY_DIR env wins, else the most
    recently modified ~/.claude/projects/*/memory directory on this machine."""
    env = os.environ.get("KG_MEMORY_DIR")
    if env:
        return Path(env).expanduser()
    projects = Path.home() / ".claude" / "projects"
    if projects.exists():
        candidates = [p for p in projects.glob("*/memory") if p.is_dir()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return projects / "_none" / "memory"


MEMORY_DIR = _default_memory_dir()

# High-signal domains — matching adds bonus weight regardless of title content.
DOMAIN_WEIGHTS = {
    "github.com": 2.0,
    "gitlab.com": 1.5,
    "vercel.com": 1.5,
    "railway.app": 1.5,
    "neon.tech": 1.5,
    "anthropic.com": 2.0,
    "claude.ai": 2.0,
    "docs.anthropic.com": 2.0,
    "stripe.com": 1.5,
    "nextjs.org": 1.5,
    "react.dev": 1.5,
    "fly.io": 1.5,
    "supabase.com": 1.2,
    "arxiv.org": 1.5,
    "news.ycombinator.com": 1.0,
    "medium.com": 0.8,
    "substack.com": 0.8,
    "producthunt.com": 0.7,
    "reddit.com": 0.6,
    "youtube.com": 0.6,
    "youtu.be": 0.6,
    "twitter.com": 0.5,
    "x.com": 0.5,
}

# Low-signal domains (social noise, marketing pages) — penalize.
DOMAIN_PENALTIES = {
    "facebook.com": -0.5,
    "tiktok.com": -0.3,
    "pinterest.com": -0.5,
    "instagram.com": -0.3,  # raw instagram.com links; Instagram exports handled separately
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "at", "for",
    "with", "by", "as", "from", "into", "about", "how", "what", "why",
    "when", "who", "which", "this", "that", "these", "those", "it", "its",
    "i", "my", "me", "you", "your", "we", "our", "they", "their", "he",
    "she", "his", "her", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "not", "no",
    "yes", "all", "any", "some", "more", "most", "less", "least", "new",
    "old", "get", "got", "make", "made", "one", "two", "three",
    "www", "http", "https", "com", "org", "io", "net", "app", "html",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


@dataclass
class InterestProfile:
    """Keyword → weight map derived from user signals."""

    weights: dict[str, float] = field(default_factory=dict)
    domain_weights: dict[str, float] = field(default_factory=lambda: dict(DOMAIN_WEIGHTS))
    domain_penalties: dict[str, float] = field(default_factory=lambda: dict(DOMAIN_PENALTIES))

    def score_tokens(self, text: str) -> float:
        if not text:
            return 0.0
        tokens = tokenize(text)
        total = 0.0
        for tok in tokens:
            total += self.weights.get(tok, 0.0)
        return total

    def score_domain(self, url: str) -> float:
        host = extract_host(url)
        if not host:
            return 0.0
        for domain, w in self.domain_weights.items():
            if host == domain or host.endswith("." + domain):
                return w
        for domain, w in self.domain_penalties.items():
            if host == domain or host.endswith("." + domain):
                return w
        return 0.0

    def size(self) -> int:
        return len(self.weights)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def extract_host(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url.strip(), re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).lower().removeprefix("www.")


def _weights_from_text(text: str, base_weight: float) -> Counter:
    counter: Counter = Counter()
    for tok in tokenize(text):
        counter[tok] += base_weight
    return counter


def build_profile(
    memory_dir: Path = MEMORY_DIR,
    store: KnowledgeGraphStore | None = None,
) -> InterestProfile:
    """Construct an interest profile from memory files + existing graph entities."""
    weights: Counter = Counter()

    if memory_dir.exists():
        for md_file in memory_dir.glob("*.md"):
            try:
                raw = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            # Frontmatter name/description count heavier than body.
            fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", raw, re.DOTALL)
            if fm_match:
                fm, body = fm_match.group(1), fm_match.group(2)
                weights.update(_weights_from_text(fm, 3.0))
                weights.update(_weights_from_text(body, 1.0))
            else:
                weights.update(_weights_from_text(raw, 1.0))

    if store is not None:
        for entity in store.find_by_label("Document"):
            weights.update(_weights_from_text(entity.name, 2.0))
            if entity.description:
                weights.update(_weights_from_text(entity.description, 1.5))
            for tag in entity.tags:
                weights.update(_weights_from_text(tag, 1.5))
        for entity in store.find_by_label("Concept"):
            weights.update(_weights_from_text(entity.name, 2.0))
        for entity in store.find_by_label("Person"):
            weights.update(_weights_from_text(entity.name, 1.0))

    # Drop low-frequency tokens that only fire once — too noisy to be signal.
    pruned = {k: float(v) for k, v in weights.items() if v >= 1.0 and len(k) >= 3}
    return InterestProfile(weights=pruned)


def top_terms(profile: InterestProfile, n: int = 30) -> list[tuple[str, float]]:
    """Preview the top N terms. Useful for debugging profile quality."""
    return sorted(profile.weights.items(), key=lambda kv: -kv[1])[:n]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preview user interest profile")
    parser.add_argument("--no-graph", action="store_true", help="Skip reading graph entities")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    store = None if args.no_graph else KnowledgeGraphStore()
    profile = build_profile(store=store)
    print(f"Terms: {profile.size()}")
    print(f"Top {args.top}:")
    for term, w in top_terms(profile, n=args.top):
        print(f"  {w:7.1f}  {term}")
