"""Content-bound, purpose-specific export permission. Domain is not permission."""
import hashlib
import json
from datetime import datetime, timezone
from knowledge_graph.models.timestamps import as_utc

_APPROVAL = {'export_policy', 'export_permissions', 'approved_export_hash', 'approved_by', 'approved_at', 'embedding_status', 'embedding_model', 'embedding_content_hash', 'embedding_hash'}

def export_fingerprint(entity):
    p = {k: v for k, v in entity.properties.items() if k not in _APPROVAL}
    payload = dict(id=entity.id, name=entity.name, description=entity.description, label=entity.label,
                   tags=entity.tags, topics=entity.topics, source_app=entity.source_app,
                   source_url=entity.source_url, source_file_path=entity.source_file_path,
                   source_user=entity.source_user, created_at=as_utc(entity.created_at).isoformat(), properties=p)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()

def _eligible(entity):
    p = entity.properties
    if (not isinstance(p, dict) or not isinstance(entity.tags, list)
            or any(not isinstance(tag, str) for tag in entity.tags)
            or not isinstance(entity.label, str)
            or (entity.source_app is not None and not isinstance(entity.source_app, str))):
        return False
    source = (entity.source_app or '').strip().lower()
    tags = {str(tag).strip().lower() for tag in entity.tags}
    raw_capture = source in {'fieldy', 'plaud'} and entity.label.lower() in {'event', 'document', 'chunk'} and p.get('record_kind') != 'fact'
    return (not raw_capture and p.get('domain') == 'professional' and not p.get('contains_raw')
            and not p.get('raw_source') and not {'personal', 'mixed', 'private', 'sensitive'} & tags)

def _valid_purposes(purposes):
    return (isinstance(purposes, (list, tuple, set, frozenset)) and bool(purposes)
            and all(isinstance(value, str) and value in {'cloud', 'publish'} for value in purposes))

def export_properties(entity):
    """Only properties bound by export_fingerprint belong in cloud responses."""
    return {key: value for key, value in entity.properties.items() if key not in _APPROVAL}

def approve_export(entity, purposes=('cloud',), approved_by='Oscar'):
    if not _eligible(entity):
        raise ValueError('Only selected professional facts without raw/private content can be approved')
    if not _valid_purposes(purposes):
        raise ValueError('Unknown export purpose')
    entity.properties.update(export_policy='approved', export_permissions=sorted(set(purposes)),
                             approved_by=approved_by, approved_at=datetime.now(timezone.utc).isoformat())
    entity.properties['approved_export_hash'] = export_fingerprint(entity)
    return entity

def export_allowed(entity, purpose='cloud'):
    if purpose not in {'cloud', 'publish'} or not _eligible(entity):
        return False
    p = entity.properties
    permissions = p.get('export_permissions')
    return (p.get('export_policy') == 'approved' and _valid_purposes(permissions) and purpose in permissions
            and p.get('approved_export_hash') == export_fingerprint(entity))
