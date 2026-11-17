"""UTC timestamps, preserving the local-time meaning of legacy naive values."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    # datetime.astimezone uses the host's local zone for naive legacy records.
    return value.astimezone(timezone.utc)
