"""In-memory session conversation manager."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single dialogue turn in a session history."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


class ConversationMemory:
    """In-memory conversation store keyed by session_id with owner isolation and TTL auto-expiration."""

    def __init__(
        self,
        max_turns: int = 10,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 2000,
    ) -> None:
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._store: dict[str, list[Turn]] = {}
        self._session_owners: dict[str, str] = {}

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        owner_id: str | None = None,
    ) -> None:
        self._cleanup()

        # Enforce session ownership isolation
        if session_id in self._session_owners and owner_id:
            existing_owner = self._session_owners[session_id]
            if existing_owner != owner_id and owner_id != "admin":
                raise PermissionError(
                    f"Unauthorized session access: Session {session_id} is owned by another user."
                )
        elif owner_id:
            self._session_owners[session_id] = owner_id

        turns = self._store.setdefault(session_id, [])
        turns.append(Turn(role=role, content=content))
        if len(turns) > self.max_turns * 2:
            self._store[session_id] = turns[-self.max_turns * 2 :]

    def get_history(self, session_id: str, owner_id: str | None = None) -> list[Turn]:
        self._cleanup()

        # Prevent cross-user session history leakage
        if session_id in self._session_owners and owner_id:
            existing_owner = self._session_owners[session_id]
            if existing_owner != owner_id and owner_id != "admin":
                return []

        return list(self._store.get(session_id, []))

    def get_summarized_history(
        self,
        session_id: str,
        max_chars_per_turn: int = 350,
        owner_id: str | None = None,
    ) -> list[Turn]:
        """Return history where long assistant responses are summarized to conserve context tokens."""
        self._cleanup()

        if session_id in self._session_owners and owner_id:
            existing_owner = self._session_owners[session_id]
            if existing_owner != owner_id and owner_id != "admin":
                return []

        raw_turns = self._store.get(session_id, [])
        summarized: list[Turn] = []
        for t in raw_turns:
            if t.role == "assistant" and len(t.content) > max_chars_per_turn:
                summary = self._summarize_text(t.content, max_chars_per_turn)
                summarized.append(Turn(role=t.role, content=f"[Prior Assistant Summary: {summary}]", timestamp=t.timestamp))
            else:
                summarized.append(t)
        return summarized

    def _summarize_text(self, text: str, max_chars: int) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("---")]
        key_points = []
        for line in lines:
            if line.startswith("#") or line.startswith("-") or line.startswith("*") or len(key_points) == 0:
                clean = line.lstrip("#-* ").strip()
                if clean and clean not in key_points:
                    key_points.append(clean)
        summary_str = "; ".join(key_points[:5])
        if len(summary_str) > max_chars:
            summary_str = summary_str[:max_chars].rsplit(" ", 1)[0] + "..."
        return summary_str or text[:max_chars] + "..."

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, turns in self._store.items()
            if turns and (now - turns[-1].timestamp) > self.ttl_seconds
        ]
        for sid in expired:
            self._store.pop(sid, None)
            self._session_owners.pop(sid, None)

        # Enforce max sessions bound
        if len(self._store) > self.max_sessions:
            overflow = len(self._store) - self.max_sessions
            for sid in list(self._store.keys())[:overflow]:
                self._store.pop(sid, None)
                self._session_owners.pop(sid, None)
