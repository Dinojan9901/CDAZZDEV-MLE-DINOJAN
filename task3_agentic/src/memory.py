"""Session and persistent memory, Task 3C.

Two distinct things share the word memory here:

SessionMemory is short-term. It holds the running message list and the observations
already gathered, so a follow-up question inside one session is answered from what the
agent has seen rather than by calling the same tool twice.

BriefCache is persistent. A completed brief is written to disk keyed by ticker and date,
and a later run for the same ticker on the same day loads it instead of re-running every
tool. The date is part of the key on purpose: yesterday's brief is stale by definition,
so it must not satisfy today's request.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


@dataclass
class SessionMemory:
    messages: list = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_message(self, message) -> None:
        self.messages.append(message)

    def remember(self, key: str, value: Any) -> None:
        self.observations[key] = value

    def recall(self, key: str, default=None) -> Any:
        return self.observations.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.observations

    def note(self, text: str) -> None:
        self.notes.append(text)

    def summary(self) -> str:
        if not self.observations:
            return "Nothing gathered yet."
        parts = []
        for key, value in self.observations.items():
            if isinstance(value, dict):
                ok = value.get("ok")
                parts.append(f"{key}: {'available' if ok else 'failed'}")
            else:
                parts.append(f"{key}: available")
        return "; ".join(parts)


class BriefCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str, on: date | str | None = None) -> Path:
        stamp = on if isinstance(on, str) else (on or date.today()).isoformat()
        return self.directory / f"{ticker.strip().upper()}_{stamp}.json"

    def exists(self, ticker: str, on: date | str | None = None) -> bool:
        return self.path_for(ticker, on).exists()

    def load(self, ticker: str, on: date | str | None = None) -> dict | None:
        path = self.path_for(ticker, on)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        # A cache file from an older layout is worse than no cache at all.
        if payload.get("cache_version") != CACHE_VERSION:
            return None
        return payload

    def save(self, ticker: str, payload: dict, on: date | str | None = None) -> Path:
        path = self.path_for(ticker, on)
        stamp = on if isinstance(on, str) else (on or date.today()).isoformat()
        body = {
            "cache_version": CACHE_VERSION,
            "ticker": ticker.strip().upper(),
            "cached_on": stamp,
            **payload,
        }
        path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        return path

    def clear(self, ticker: str, on: date | str | None = None) -> bool:
        path = self.path_for(ticker, on)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_cached(self) -> list[str]:
        return sorted(p.name for p in self.directory.glob("*.json"))
