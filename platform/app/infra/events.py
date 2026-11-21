"""Incremental reader for outputs/<run>/events.jsonl (contract: README.md section 10.1).

Reads by byte offset: each poll costs only the new bytes, and a half-written last line is left for the next poll,
so the offset can be handed to the browser as an SSE event id and resumed exactly after a reconnect.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path


def read_events(path: Path, offset: int = 0, limit: int | None = None) -> tuple[list[dict], int]:
    """Complete events after `offset` (at most `limit`) and the offset to continue from."""
    if not path.exists():
        return [], offset
    events, pos = [], offset
    with path.open("rb") as f:
        f.seek(offset)
        for raw in f:
            if not raw.endswith(b"\n"):  # still being written
                break
            pos += len(raw)
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue  # a corrupt line never blocks the stream
            if limit and len(events) >= limit:
                break
    return events, pos


def tail_events(path: Path, n: int, window: int = 512_000) -> tuple[list[dict], int]:
    """The last `n` complete events (reads only the file's tail) and the end offset to stream from."""
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    start = max(0, size - window)
    events, end = read_events(path, start)
    if start > 0:
        events = events[1:]  # the first line read from mid-file may be partial
    return events[-n:], end


async def follow_events(path: Path, offset: int, is_live: Callable[[], bool],
                        poll_s: float = 0.5) -> AsyncIterator[tuple[list[dict], int]]:
    """Yield new event batches until the run has ended and nothing new arrives."""
    while True:
        events, offset = read_events(path, offset)
        if events:
            yield events, offset
            if any(e.get("type") == "run_end" for e in events):
                return
        elif not is_live():
            return
        await asyncio.sleep(poll_s)
