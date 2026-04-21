"""Bookmark + save ingesters. Each module ingests one external source
(Chrome, Firefox, Reddit, YouTube, Instagram) into the knowledge graph
as Entity(label="Document", source_type="bookmark").

Relevance scoring uses the user's interest profile built from
~/.claude memory + existing graph entities (see interest_profile.py).
"""
