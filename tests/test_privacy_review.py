import pytest
from knowledge_graph.models import Entity
from knowledge_graph.privacy import approve_export, export_allowed


def fact(**kwargs):
    return Entity(id="e", label="Decision", name="Work fact", description="Use option A", properties={"domain": "professional"}, **kwargs)


@pytest.mark.parametrize("tags", [["Private"], ["PERSONAL"], ["Sensitive"], ["mixed"]])
def test_private_tags_cannot_bypass_approval_with_capitalization(tags):
    entity = fact(tags=tags)
    with pytest.raises(ValueError):
        approve_export(entity)
    assert not export_allowed(entity)


def test_capitalized_raw_source_cannot_be_approved():
    entity = Entity(id="e", label="Event", name="Conversation", description="Raw", source_app="Fieldy", properties={"domain": "professional"})
    with pytest.raises(ValueError):
        approve_export(entity)


@pytest.mark.parametrize("permissions", ["cloud", None, 3, {"cloud": True}])
def test_malformed_permission_records_fail_closed(permissions):
    entity = fact()
    approve_export(entity)
    entity.properties["export_permissions"] = permissions
    assert not export_allowed(entity)


def test_approval_requires_sequence_of_purposes():
    with pytest.raises(ValueError):
        approve_export(fact(), purposes="cloud")


@pytest.mark.parametrize("change", [{"tags": "personal"}, {"properties": []}, {"source_app": 42}, {"label": None}])
def test_malformed_entity_privacy_metadata_fails_closed(change):
    entity = fact()
    approve_export(entity)
    for field, value in change.items():
        setattr(entity, field, value)
    assert not export_allowed(entity)
