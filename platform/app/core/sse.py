"""Server-Sent Events helpers: one formatting function and one streaming response with heartbeats.

The browser's EventSource reconnects automatically and sends Last-Event-ID; streams use a byte offset as the
id so a reconnect resumes exactly where it stopped.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

HEARTBEAT_S = 15


def sse_message(data, event: str | None = None, id_: str | int | None = None) -> str:
    lines = []
    if id_ is not None:
        lines.append(f"id: {id_}")
    if event:
        lines.append(f"event: {event}")
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    lines += [f"data: {line}" for line in payload.splitlines() or [""]]
    return "\n".join(lines) + "\n\n"


async def _with_heartbeat(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Interleave `: ping` comments so proxies keep idle connections open."""
    queue: asyncio.Queue = asyncio.Queue()

    async def pump():
        try:
            async for item in source:
                await queue.put(item)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is None:
                return
            yield item
    finally:
        task.cancel()


def sse_response(source: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(_with_heartbeat(source), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
