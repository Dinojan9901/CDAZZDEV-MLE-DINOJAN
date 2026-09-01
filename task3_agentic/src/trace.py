"""Structured tool-call tracing to agent_trace.jsonl, Task 3C.

Every tool invocation is recorded whether it succeeds or fails. A trace that only
contains successes hides exactly the runs worth debugging.

The active agent name travels in a ContextVar rather than a parameter, so tools stay
callable by any agent without threading an identity argument through every signature.
"""

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

OUTPUT_TRUNCATE = 200

_active_agent: ContextVar[str] = ContextVar("active_agent", default="root")


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return repr(value)


class ToolTracer:
    def __init__(self, path: Path, session_id: str | None = None, append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._lock = Lock()
        self._seq = 0
        if not append and self.path.exists():
            self.path.unlink()

    def record(self, tool: str, inputs: dict, output, duration_ms: float,
               error: str | None = None) -> dict:
        rendered = _stringify(output)
        with self._lock:
            self._seq += 1
            seq = self._seq
        entry = {
            "session_id": self.session_id,
            "seq": seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": _active_agent.get(),
            "tool": tool,
            "inputs": inputs,
            "output": rendered[:OUTPUT_TRUNCATE],
            "output_truncated": len(rendered) > OUTPUT_TRUNCATE,
            "output_chars": len(rendered),
            "duration_ms": round(duration_ms, 2),
            "status": "error" if error else "ok",
        }
        if error:
            entry["error"] = error[:OUTPUT_TRUNCATE]
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    @contextmanager
    def span(self, tool: str, inputs: dict):
        """Time a call and record it on both the success and the failure path.

        A tool that returns a handled failure rather than raising still has to land in
        the trace as an error. Setting box["error"], or returning a payload with
        ok=False, marks it so, otherwise the log would show a clean run that was not.
        """
        started = time.perf_counter()
        box: dict = {"output": None, "error": None}
        try:
            yield box
        except Exception as exc:
            self.record(tool, inputs, box.get("output"),
                        (time.perf_counter() - started) * 1000, error=f"{type(exc).__name__}: {exc}")
            raise

        error = box.get("error")
        output = box.get("output")
        if error is None and isinstance(output, dict) and output.get("ok") is False:
            error = str(output.get("error", "tool reported failure"))
        self.record(tool, inputs, output, (time.perf_counter() - started) * 1000, error=error)

    def read(self, session_only: bool = True) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_only and row.get("session_id") != self.session_id:
                continue
            rows.append(row)
        return rows

    def summary(self) -> dict:
        rows = self.read()
        by_tool: dict[str, dict] = {}
        for row in rows:
            stat = by_tool.setdefault(row["tool"], {"calls": 0, "errors": 0, "total_ms": 0.0})
            stat["calls"] += 1
            stat["errors"] += row["status"] == "error"
            stat["total_ms"] += row["duration_ms"]
        for stat in by_tool.values():
            stat["mean_ms"] = round(stat["total_ms"] / stat["calls"], 2)
            stat["total_ms"] = round(stat["total_ms"], 2)
        return {
            "session_id": self.session_id,
            "total_calls": len(rows),
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "wall_ms": round(sum(r["duration_ms"] for r in rows), 2),
            "by_tool": by_tool,
        }

    def render(self, limit: int = 40) -> str:
        rows = self.read()[:limit]
        if not rows:
            return "(no tool calls recorded)"
        lines = [f"{'#':>3}  {'agent':<16} {'tool':<20} {'ms':>8}  status  inputs"]
        for row in rows:
            args = ", ".join(f"{k}={v!r}" for k, v in (row["inputs"] or {}).items())
            lines.append(
                f"{row['seq']:>3}  {row['agent']:<16} {row['tool']:<20} "
                f"{row['duration_ms']:>8.1f}  {row['status']:<6}  {args[:60]}"
            )
        return "\n".join(lines)


@contextmanager
def acting_as(agent: str):
    token = _active_agent.set(agent)
    try:
        yield
    finally:
        _active_agent.reset(token)


def current_agent() -> str:
    return _active_agent.get()
