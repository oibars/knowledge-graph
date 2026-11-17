"""Revisioned local captures, evidence validation, and rebuildable projections."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from threading import RLock
import time
from urllib.parse import quote as quote_id
from zoneinfo import ZoneInfo

from .models import Entity, Relation
from .models.timestamps import as_utc
from .privacy import export_allowed, export_fingerprint

_PROJECTION_VERSION = 3
_SCOPES = {"professional", "personal"}
_ACTION_STATUSES = {"Task": {"open", "done", "cancelled"}, "Decision": {"decided", "superseded"}}
_APPROVAL_FIELDS = ("export_policy", "export_permissions", "approved_export_hash", "approved_by", "approved_at")
# Deterministic failures never succeed on retry. Bound them so one poisoned
# capture cannot starve the pending queue forever.
MAX_ATTEMPTS = 8
_BUSY_TIMEOUT_MS = 10000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(record):
    if not isinstance(record, dict):
        raise ValueError("Capture must be an object")
    normalized = deepcopy(record)
    source_id = next((normalized.get(key) for key in ("id", "conversationId", "file_id")
                      if normalized.get(key) is not None), None)
    if not isinstance(source_id, (str, int)) or isinstance(source_id, bool) or not str(source_id).strip():
        raise ValueError("Capture requires a source id")
    normalized["id"] = str(source_id)
    transcript = next((normalized.get(key) for key in ("transcript", "content", "text")
                       if normalized.get(key) is not None and normalized.get(key) != ""), "")
    if not isinstance(transcript, str):
        raise ValueError("Capture transcript must be text")
    normalized["transcript"] = transcript
    normalized["startTime"] = normalized.get("startTime") or normalized.get("date") or normalized.get("created_at")
    segments = normalized.get("segments", [])
    if segments is None:
        segments = []
    if not isinstance(segments, list) or any(not isinstance(segment, dict) for segment in segments):
        raise ValueError("Capture segments must be objects in a list")
    if any(segment.get("text") is not None and not isinstance(segment["text"], str) for segment in segments):
        raise ValueError("Segment text must be text")
    normalized["segments"] = segments
    return normalized


def _inline_segments(text):
    """Recognize source-provided speaker lines without inventing attribution."""
    segments = []
    pattern = re.compile(r"^(?:\[(\d+):(\d{2})(?::(\d{2}))?\]\s*)?([^:\n]{1,100}):\s*(.*)$", re.MULTILINE)
    for match in pattern.finditer(text):
        first, second, third, speaker, content = match.groups()
        start = None
        if first is not None:
            start = int(first) * (3600 if third is not None else 60) + int(second) * (60 if third is not None else 1) + int(third or 0)
        segments.append({"speaker": speaker.strip(), "text": content, "start": start, "end": None})
    return segments


def _validate_evidence(items, raw):
    if not isinstance(items, list) or not items:
        raise ValueError("Missing evidence")
    text = raw["transcript"]
    segments = raw["segments"] or _inline_segments(text)
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Evidence must be an object")
        quote = item.get("quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError("Evidence not in capture")
        start, end = item.get("start"), item.get("end")
        if start is not None and (not _number(start) or start < 0):
            raise ValueError("Invalid evidence span")
        if end is not None and (not _number(end) or end < (start or 0)):
            raise ValueError("Invalid evidence span")
        speaker = item.get("speaker")
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
            raise ValueError("Invalid speaker")
        if speaker or start is not None or end is not None:
            matches = [segment for segment in segments if quote in (segment.get("text") or "")
                       and (not speaker or speaker == segment.get("speaker"))]
            if not matches:
                raise ValueError("Speaker evidence mismatch")
            if start is not None or end is not None:
                def supported_span(segment):
                    lower, upper = segment.get("start"), segment.get("end")
                    if not _number(lower):
                        return False
                    if start is not None and start < lower:
                        return False
                    if _number(upper):
                        return (start is None or start <= upper) and (end is None or end <= upper)
                    # Inline timestamps provide a start, never an invented end.
                    return (start is None or start == lower) and end is None
                if not any(supported_span(segment) for segment in matches):
                    raise ValueError("Span evidence mismatch")


def _supported_due_dates(text, raw):
    """Resolve unambiguous date evidence in the source's own calendar timezone."""
    dates = set()
    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            dates.add(datetime.fromisoformat(value).date())
        except ValueError:
            pass
    try:
        occurred = datetime.fromisoformat(raw["startTime"])
        calendar_zone = ZoneInfo(raw.get("timezone") or os.getenv("JARVIS_TIMEZONE", "America/New_York"))
        base = (occurred.astimezone(calendar_zone) if occurred.tzinfo else occurred.replace(tzinfo=calendar_zone)).date()
    except (ValueError, TypeError, KeyError):
        base = None
    if base is not None:
        for word, delta in (("today", 0), ("tonight", 0), ("tomorrow", 1)):
            if re.search(r"\b" + word + r"\b", text, re.IGNORECASE):
                dates.add(base + timedelta(days=delta))
        weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        for weekday, name in enumerate(weekdays):
            if re.search(r"\b(?:next\s+|this\s+|by\s+|on\s+)?" + name + r"\b", text, re.IGNORECASE):
                offset = (weekday - base.weekday()) % 7
                if offset == 0 and re.search(r"\bnext\s+" + name + r"\b", text, re.IGNORECASE):
                    offset = 7
                dates.add(base + timedelta(days=offset))
        for count, unit in re.findall(r"\bin\s+(\d+)\s+(days?|weeks?)\b", text, re.IGNORECASE):
            dates.add(base + timedelta(days=int(count) * (7 if unit.lower().startswith("week") else 1)))
        for month in range(1, 13):
            name = datetime(2000, month, 1).strftime("%B")
            expression = r"\b" + name + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b"
            for day, year in re.findall(expression, text, re.IGNORECASE):
                try:
                    dates.add(datetime(int(year or base.year), month, int(day)).date())
                except ValueError:
                    pass
    return dates


def _validate_action(action, extraction, raw):
    if not isinstance(action, dict):
        raise ValueError("Action must be an object")
    kind = action.get("kind")
    if kind not in _ACTION_STATUSES:
        raise ValueError("Invalid action kind")
    for field in ("title", "text"):
        if not isinstance(action.get(field), str) or not action[field].strip():
            raise ValueError("Invalid action text")
    if action.get("domain") not in _SCOPES or action.get("status") not in _ACTION_STATUSES[kind]:
        raise ValueError("Invalid action state")
    _validate_evidence(action.get("evidence"), raw)
    domain = extraction["domain"]
    if domain in _SCOPES and action["domain"] != domain:
        raise ValueError("Action domain conflicts with capture")
    if domain == "unknown":
        raise ValueError("Unknown capture cannot validate actions")
    if domain == "mixed" and action["domain"] == "professional":
        professional_slice = extraction.get("professional_slice")
        if not professional_slice or any(item["quote"] not in professional_slice["text"] for item in action["evidence"]):
            raise ValueError("Professional action evidence is outside the work slice")
    owner = action.get("owner")
    if owner is not None:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Invalid action owner")
        named = re.compile(r"(?<!\w)" + re.escape(owner) + r"(?!\w)", re.IGNORECASE)
        if not any(named.search(item["quote"]) or owner == item.get("speaker") for item in action["evidence"]):
            raise ValueError("Owner lacks evidence")
    evidence_text = "\n".join(item["quote"] for item in action["evidence"])
    due = action.get("due_at")
    if due is not None:
        if not isinstance(due, str) or not due.strip():
            raise ValueError("Invalid due date")
        try:
            due_time = datetime.fromisoformat(due)
        except ValueError as error:
            raise ValueError("Invalid due date") from error
        if due_time.date() not in _supported_due_dates(evidence_text, raw):
            raise ValueError("Due date lacks source evidence")
        if "T" in due or " " in due:
            if due_time.strftime("%H:%M") not in evidence_text:
                raise ValueError("Due time lacks source evidence")
    status = action["status"]
    transitions = {"done": r"\b(sent|finished|completed|delivered|submitted|shipped|resolved|paid|emailed|done)\b",
                   "cancelled": r"\b(cancelled|canceled|cancel|no longer|won't|will not|abandon|dropped)\b",
                   "superseded": r"\b(superseded|replaced|instead|changed|switch|reversed|overturned)\b"}
    if status in transitions:
        future = re.search(r"\b(will|would|plan to|need to|not yet|haven't|have not|didn't|did not)\b", evidence_text, re.IGNORECASE)
        if not re.search(transitions[status], evidence_text, re.IGNORECASE) or (status == "done" and future):
            raise ValueError("Action status lacks source evidence")


def _validate_extraction(extraction, raw):
    if not isinstance(extraction, dict) or extraction.get("domain") not in _SCOPES | {"mixed", "unknown"}:
        raise ValueError("Invalid extraction domain")
    confidence = extraction.get("confidence")
    if not _number(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    topics = extraction.get("topics", [])
    if not isinstance(topics, list) or any(not isinstance(topic, str) or not topic.strip() for topic in topics):
        raise ValueError("Invalid topics")
    professional_slice = extraction.get("professional_slice")
    if professional_slice is not None:
        if not isinstance(professional_slice, dict) or not isinstance(professional_slice.get("text"), str) or not professional_slice["text"].strip():
            raise ValueError("Invalid slice")
        if not isinstance(professional_slice.get("title"), str) or not professional_slice["title"].strip():
            raise ValueError("Professional slice requires its own title")
        _validate_evidence(professional_slice.get("evidence"), raw)
        joined = "\n".join(item["quote"] for item in professional_slice["evidence"])
        if professional_slice["text"] not in raw["transcript"] and professional_slice["text"] != joined:
            raise ValueError("Slice text must be verbatim evidence")
    actions = extraction.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("Invalid actions")
    for action in actions:
        _validate_action(action, extraction, raw)
    # Persist only the schema, never arbitrary model-supplied provenance/approval.
    return {"domain": extraction["domain"], "confidence": confidence, "topics": topics,
            "professional_slice": professional_slice, "actions": actions}


class CaptureArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self._lock = RLock()
        self.db = sqlite3.connect(self.root / "captures.sqlite", timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS captures(source TEXT,id TEXT,content_hash TEXT,raw_path TEXT,
              status TEXT,extraction TEXT,attempts INTEGER DEFAULT 0,retry_at REAL DEFAULT 0,error TEXT,
              projected_hash TEXT,PRIMARY KEY(source,id));
            CREATE TABLE IF NOT EXISTS cursors(name TEXT PRIMARY KEY,value TEXT);""")
        os.chmod(self.root / "captures.sqlite", 0o600)

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def _get(self, source, source_id):
        row = self.db.execute("SELECT * FROM captures WHERE source=? AND id=?", (source, source_id)).fetchone()
        return dict(row) if row else None

    def get(self, source, id):
        with self._lock:
            return self._get(source, id)

    def _read_raw(self, row):
        if row is None:
            raise ValueError("Capture not found")
        path = Path(row["raw_path"])
        if not path.resolve().is_relative_to((self.root / "raw").resolve()):
            raise ValueError("Raw revision path is outside the archive")
        raw = json.loads(path.read_text())
        if digest(raw) != row["content_hash"]:
            raise ValueError("Raw revision integrity hash mismatch")
        return raw

    def raw(self, source, id):
        with self._lock:
            return self._read_raw(self._get(source, id))

    def archive(self, source, record):
        if not isinstance(source, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", source):
            raise ValueError("Invalid source")
        raw = normalize(record)
        key = source, raw["id"]
        content_hash = digest(raw)
        folder = self.root / "raw" / source / hashlib.sha256(raw["id"].encode()).hexdigest()[:24]
        with self._transaction():
            previous = self._get(*key)
            if previous and previous["content_hash"] == content_hash:
                try:
                    self._read_raw(previous)
                    return key
                except (ValueError, FileNotFoundError):
                    pass  # Recreate an identical authoritative revision.
            folder.mkdir(parents=True, exist_ok=True)
            os.chmod(folder, 0o700)
            path = folder / (content_hash + ".json")
            fd, temp_name = tempfile.mkstemp(prefix=".capture-", suffix=".tmp", dir=folder)
            try:
                with os.fdopen(fd, "w") as output:
                    json.dump(raw, output, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp_name, path)
            finally:
                Path(temp_name).unlink(missing_ok=True)
            if previous and previous["content_hash"] == content_hash:
                return key
            status = "pending" if raw["transcript"].strip() and (source != "fieldy" or raw.get("endTime")) else "awaiting_source"
            self.db.execute("""INSERT INTO captures(source,id,content_hash,raw_path,status) VALUES(?,?,?,?,?)
                ON CONFLICT(source,id) DO UPDATE SET content_hash=excluded.content_hash,raw_path=excluded.raw_path,
                status=excluded.status,extraction=NULL,attempts=0,retry_at=0,error=NULL,projected_hash=NULL""",
                (*key, content_hash, str(path), status))
        return key

    def fail(self, source, id, error, expected_content_hash=None):
        with self._transaction():
            row = self._get(source, id)
            if row is None:
                raise ValueError("Capture not found")
            if expected_content_hash is not None and row["content_hash"] != expected_content_hash:
                return False
            retry = time.time() + min(3600, 60 * 2 ** min(row["attempts"], 6))
            attempts = row["attempts"] + 1
            status = "dead" if attempts >= MAX_ATTEMPTS else "failed"
            self.db.execute("UPDATE captures SET status=?,error=?,attempts=?,retry_at=? WHERE source=? AND id=?",
                            (status, str(error)[:120], attempts, retry, source, id))
        return True

    def pending(self, limit=10):
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM captures WHERE status IN ('pending','failed') AND retry_at<=? ORDER BY attempts,id LIMIT ?",
                                                       (time.time(), max(0, limit)))]

    def validate(self, source, id, extraction, expected_content_hash=None):
        with self._transaction():
            row = self._get(source, id)
            if row is None:
                raise ValueError("Capture not found")
            if expected_content_hash is not None and row["content_hash"] != expected_content_hash:
                raise ValueError("Capture revision changed during extraction")
            raw = self._read_raw(row)
            if not raw["transcript"].strip() or (source == "fieldy" and not raw.get("endTime")):
                raise ValueError("Capture incomplete")
            validated = _validate_extraction(deepcopy(extraction), raw)
            previous = json.loads(row["extraction"]) if row["extraction"] else None
            self.db.execute("UPDATE captures SET status='validated',extraction=?,error=NULL,retry_at=0,projected_hash=? WHERE source=? AND id=?",
                            (json.dumps(validated, ensure_ascii=False), row["projected_hash"] if previous == validated else None, source, id))

    @staticmethod
    def _managed_ids(store, source, source_id):
        # A private connection must wait for the live store's writers instead of
        # failing the whole projection with "database is locked"; close it
        # explicitly so repeated reconciles cannot leak file descriptors.
        connection = sqlite3.connect(store.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            connection.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
            return {row[0] for row in connection.execute("SELECT id FROM entities WHERE source_app=? AND json_extract(properties,'$.source_id')=? AND json_extract(properties,'$.capture_managed')=1", (source, source_id))}
        finally:
            connection.close()

    @staticmethod
    def _checkpoint_complete(store, checkpoint, projection_hash, entities, relations):
        if not isinstance(checkpoint, dict) or checkpoint.get("hash") != projection_hash or checkpoint.get("store") != str(store.db_path.resolve()):
            return False
        for expected in entities:
            current = store.get_entity(expected.id)
            if current is None:
                return False
            expected.created_at = current.created_at
            if export_fingerprint(current) != export_fingerprint(expected):
                return False
        for expected in relations:
            current = store.get_relation(expected.id)
            if current is None or (current.source_id, current.target_id, current.relation_type) != (expected.source_id, expected.target_id, expected.relation_type):
                return False
            if expected.bidirectional and expected.get_inverse_type():
                inverse = store.get_relation(expected.get_inverse_id())
                if inverse is None or (inverse.source_id, inverse.target_id, inverse.relation_type) != (expected.target_id, expected.source_id, expected.get_inverse_type()):
                    return False
        return True

    @staticmethod
    def _projection_chunks(text, size=1200):
        """Tile the transcript without cutting a word in half at a chunk edge."""
        offset = 0
        while offset < len(text):
            end = min(offset + size, len(text))
            if end < len(text):
                floor = offset + size // 2
                boundary = max(text.rfind("\n", floor, end), text.rfind(" ", floor, end))
                if boundary >= floor:
                    end = boundary + 1
            yield offset, end
            offset = end

    @staticmethod
    def _chunk_evidence(raw, offset, end):
        evidence = []
        cursor = 0
        for segment in raw.get("segments") or _inline_segments(raw["transcript"]):
            text = segment.get("text")
            if not text:
                continue
            position = raw["transcript"].find(text, cursor)
            if position < 0:
                continue
            cursor = position + len(text)
            left, right = max(offset, position), min(end, cursor)
            if left < right:
                evidence.append({"quote": raw["transcript"][left:right], "speaker": segment.get("speaker"),
                                 "start": segment.get("start"), "end": segment.get("end")})
        return evidence

    @staticmethod
    def _legacy_quarantine(store, source, source_id, parent):
        entity = store.get_entity(parent)
        if (entity and entity.source_app == source and entity.properties.get("source_id") == source_id
                and entity.properties.get("migration_state") == "awaiting_validation"
                and entity.properties.get("legacy_origin") == "professional"):
            return {parent}
        return set()

    def project(self, source, id, stores):
        if not stores or set(stores) - _SCOPES:
            raise ValueError("Projection stores must be professional or personal")
        with self._transaction():
            row = self._get(source, id)
            if row is None or row["status"] != "validated":
                raise ValueError("Capture not validated")
            raw = self._read_raw(row)
            extraction = json.loads(row["extraction"])
            domain = extraction["domain"]
            try:
                created = as_utc(datetime.fromisoformat(raw["startTime"]))
            except (TypeError, ValueError, KeyError):
                created = datetime.now(timezone.utc)
            source_url = raw.get("source_url") or f"{source}://{quote_id(id, safe='')}"
            parent = f"conv-{source}-{quote_id(id, safe='')}"
            common = {"domain": domain, "corpus": "memory", "contains_raw": True, "export_policy": "blocked",
                      "export_permissions": [], "capture_managed": True, "source_id": id,
                      "source_version": row["content_hash"], "start": raw.get("startTime"), "end": raw.get("endTime"),
                      "archive_path": row["raw_path"]}
            target = "professional" if domain == "professional" else "personal"
            plans = {scope: [] for scope in stores}
            relations = {scope: [] for scope in stores}

            def put(scope, entity_id, label, title, text, properties, evidence=None, parent_id=None):
                if scope not in plans:
                    return
                props = deepcopy(properties)
                if evidence:
                    props["evidence"] = [dict(item, source_url=source_url, source_app=source, source_id=id) for item in evidence]
                entity = Entity(id=entity_id, label=label, name=title, description=text, source_app=source,
                                source_url=source_url, source_file_path=row["raw_path"], created_at=created,
                                tags=[scope, source], topics=extraction.get("topics") or [], properties=props)
                plans[scope].append(entity)
                if parent_id:
                    relations[scope].append(Relation(id=f"{entity_id}:part_of:{parent_id}", source_id=entity_id,
                                                     target_id=parent_id, relation_type="part_of"))

            put(target, parent, "Event", raw.get("title") or "Captured conversation", raw["transcript"], common)
            for index, (offset, end) in enumerate(self._projection_chunks(raw["transcript"])):
                put(target, f"{parent}:chunk:{index}", "Chunk", f"{raw.get('title') or source} · {index + 1}",
                    raw["transcript"][offset:end], dict(common, parent_id=parent, offset=offset, end_offset=end),
                    evidence=self._chunk_evidence(raw, offset, end), parent_id=parent)
            work_slice = extraction.get("professional_slice")
            if domain == "mixed" and work_slice:
                put("professional", parent + ":professional", "Event", work_slice["title"], work_slice["text"],
                    dict(common, domain="professional", origin_domain="mixed", contains_raw=False, record_kind="fact"), work_slice["evidence"])
            for action in extraction.get("actions") or []:
                scope = action["domain"]
                entity_id = parent + ":" + action["kind"].lower() + ":" + digest([action["kind"], action["title"], action.get("owner"), action["evidence"]])[:20]
                properties = dict(common, domain=scope, contains_raw=False, owner=action.get("owner"),
                                  due_at=action.get("due_at"), status=action["status"], record_kind="fact")
                put(scope, entity_id, action["kind"], action["title"], action["text"], properties, action["evidence"],
                    parent if scope == target else parent + ":professional")

            projection_hash = digest([_PROJECTION_VERSION, row["content_hash"], extraction])
            checkpoints = json.loads(row["projected_hash"]) if row["projected_hash"] and row["projected_hash"].startswith("{") else {}
            changed = 0
            # Verify checkpoints first: a fully projected capture must not open
            # private connections or rewrite entities on every reconcile run.
            complete = {}
            for scope, store in stores.items():
                complete[scope] = self._checkpoint_complete(store, checkpoints.get(scope), projection_hash, plans[scope], relations[scope])
            for scope, store in stores.items():
                if complete[scope]:
                    continue
                for entity in plans[scope]:
                    previous = store.get_entity(entity.id)
                    if previous:
                        entity.created_at = previous.created_at
                        if (export_allowed(previous) or export_allowed(previous, purpose="publish")) and export_fingerprint(previous) == export_fingerprint(entity):
                            for field in _APPROVAL_FIELDS:
                                if field in previous.properties:
                                    entity.properties[field] = previous.properties[field]
                    store.add_entity(entity)
                    changed += 1
                for relation in relations[scope]:
                    store.add_relation(relation)
                checkpoints[scope] = {"hash": projection_hash, "store": str(store.db_path.resolve())}
            # Write the complete desired route before deleting obsolete projections.
            # Only explicitly marked legacy canonical parents are managed here.
            for scope, store in stores.items():
                if complete[scope]:
                    continue
                desired = {entity.id for entity in plans[scope]}
                obsolete = self._managed_ids(store, source, id) | self._legacy_quarantine(store, source, id, parent)
                for entity_id in obsolete - desired:
                    store.delete_entity(entity_id)
            self.db.execute("UPDATE captures SET projected_hash=? WHERE source=? AND id=?", (json.dumps(checkpoints), source, id))
        return changed
