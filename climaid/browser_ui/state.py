"""Session-scoped temporary browser wizard state for ClimAID.

The original single global wizard_state is retained conceptually, but state is
now keyed by a client-generated session id so concurrent browser runs cannot
overwrite one another.
"""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from threading import RLock
import uuid

MAX_SESSIONS = 64
SESSION_TTL = timedelta(hours=2)
_lock = RLock()
_states: "OrderedDict[str, tuple[datetime, dict]]" = OrderedDict()


def _cleanup(now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    expired = [sid for sid, (ts, _) in _states.items() if now - ts > SESSION_TTL]
    for sid in expired:
        _states.pop(sid, None)
    while len(_states) > MAX_SESSIONS:
        _states.popitem(last=False)


def normalize_session_id(session_id: str | None) -> str:
    """Validate a client session id or create a new UUID4 id."""
    if session_id:
        try:
            return str(uuid.UUID(session_id))
        except (ValueError, AttributeError, TypeError):
            pass
    return str(uuid.uuid4())


def get_wizard_state(session_id: str | None = None) -> tuple[str, dict]:
    sid = normalize_session_id(session_id)
    with _lock:
        now = datetime.now(timezone.utc)
        _cleanup(now)
        entry = _states.get(sid)
        if entry is None:
            state = {}
        else:
            _, state = entry
        _states[sid] = (now, state)
        _states.move_to_end(sid)
        return sid, state


def clear_wizard_state(session_id: str | None = None) -> str:
    sid = normalize_session_id(session_id)
    with _lock:
        _states.pop(sid, None)
    return sid


# Backward-compatible symbol for legacy imports. New code should use get_wizard_state.
wizard_state = {}
