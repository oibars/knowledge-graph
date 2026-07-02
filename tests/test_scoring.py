"""Tests for the deterministic relevance scoring the README advertises."""

from pathlib import Path

from knowledge_graph.ingesters.common import _raw_to_importance, signal_tag
from knowledge_graph.ingesters import interest_profile as ip


def test_raw_to_importance_bounds_and_monotonic():
    assert 0.0 < _raw_to_importance(-5.0) <= 0.3
    assert _raw_to_importance(0.0) == 0.3
    prev = 0.0
    for raw in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        score = _raw_to_importance(raw)
        assert prev < score < 1.0
        prev = score


def test_signal_tag_tiers():
    assert signal_tag(0.95) == "signal-high"
    assert signal_tag(0.0) == "signal-low"
    tags = {signal_tag(x / 100) for x in range(0, 100, 5)}
    assert tags == {"signal-low", "signal-medium", "signal-high"}


def test_default_memory_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("KG_MEMORY_DIR", str(tmp_path))
    assert ip._default_memory_dir() == tmp_path


def test_default_memory_dir_never_hardcodes_foreign_user():
    # regression: used to hardcode ~/.claude/projects/-home-oscr/memory
    assert "-home-oscr" not in str(ip._default_memory_dir())


def test_build_profile_reads_memory_files(tmp_path):
    (tmp_path / "note.md").write_text(
        "---\nname: test-note\ndescription: kubernetes observability deep dive\n---\n"
        "Prometheus dashboards and kubernetes autoscaling patterns.\n",
        encoding="utf-8",
    )
    profile = ip.build_profile(memory_dir=tmp_path, store=None)
    assert profile.score_tokens("kubernetes observability") > 0


def test_build_profile_survives_missing_dir(tmp_path):
    profile = ip.build_profile(memory_dir=tmp_path / "nope", store=None)
    assert profile is not None
