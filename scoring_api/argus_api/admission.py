"""Admission control (LLD 6.3 deadlines, FR-API-01): a bounded number of scorings run at once.

Without it, a burst above capacity queues without limit: every request waits longer than its deadline and the caller
times out after the fact. With it, a request that cannot get a scoring slot before its deadline is turned away at
once with 503, so the Payment API applies its fail-open rule immediately, and the requests that are admitted keep
their latency. Shadow scoring (MLOPS-06) only takes a slot when another one stays free for the next live request.
"""
from __future__ import annotations

import asyncio


class Admission:
    def __init__(self, slots: int):
        self.slots = max(1, int(slots))
        self._sem = asyncio.Semaphore(self.slots)
        self.admitted = self.shed = 0

    @property
    def busy(self) -> int:
        return self.slots - self._sem._value  # noqa: SLF001 - read-only view for /v1/health

    async def acquire(self, budget_s: float) -> bool:
        """A slot within `budget_s` seconds, or False (shed) once the budget is spent."""
        if budget_s <= 0:
            self.shed += 1
            return False
        try:
            await asyncio.wait_for(self._sem.acquire(), budget_s)
        except TimeoutError:
            self.shed += 1
            return False
        self.admitted += 1
        return True

    async def try_acquire(self, keep_free: int = 1) -> bool:
        """Shadow scoring: a slot only if, after taking it, `keep_free` slots are still free for live traffic."""
        if self.slots - self.busy - 1 < keep_free:
            self.shed += 1
            return False
        await self._sem.acquire()
        self.admitted += 1
        return True

    def release(self) -> None:
        self._sem.release()

    def stats(self) -> dict:
        return {"slots": self.slots, "busy": self.busy, "admitted": self.admitted, "shed": self.shed}
