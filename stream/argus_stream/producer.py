"""Traffic: publishes transactions to Kinesis <env>-transactions while switched on.

It runs inside the stream service and is OFF until someone presses Start on the UI's Live stream page
(POST /live/api/producer; AUTO_START=1 starts it at boot). Standalone, until Ctrl-C:  python -m argus_stream.producer
Env: EVENTS_PER_SEC (2, Poisson arrivals), FRAUD_RATE (0.04), ATTACK_EVERY_S (300: a fraud ring fires a wave of
ATTACK_SIZE (15) transactions every ~N seconds while on; 0 = off), PRODUCER_SOURCE / DATASET_URI (see generator).
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time

from . import STREAM_TX
from .generator import make_source

log = logging.getLogger("argus.stream.producer")


def send(producer, events: list[dict]) -> int:
    for e in events:
        producer.produce(STREAM_TX, key=e["request"]["customer_id"], value=e)
    producer.poll()
    return len(events)


class Traffic:
    """Start / stop / re-rate a background thread that keeps sending transactions."""

    def __init__(self, producer=None, source=None):
        self._producer, self._source = producer, source
        self.rate = float(os.getenv("EVENTS_PER_SEC", "2"))
        self.attack_every = float(os.getenv("ATTACK_EVERY_S", "300"))
        self.attack_size = int(os.getenv("ATTACK_SIZE", "15"))
        self.sent = self.attacks = 0
        self.started_at: float | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def producer(self):
        if self._producer is None:
            from .bus import KinesisProducer
            self._producer = KinesisProducer()
        return self._producer

    @property
    def source(self):
        if self._source is None:
            self._source = make_source()
        return self._source

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        return {"running": self.running, "rate": self.rate, "sent": self.sent, "attacks": self.attacks,
                "source": type(self._source).__name__ if self._source else os.getenv("PRODUCER_SOURCE", "mix"),
                "attack_every_s": self.attack_every, "started_at": self.started_at, "error": self.error}

    def start(self, rate: float | None = None) -> dict:
        with self._lock:
            if rate is not None:
                self.rate = max(0.1, min(50.0, float(rate)))
            if not self.running:
                self._stop.clear()
                self.error, self.started_at = None, time.time()
                self._thread = threading.Thread(target=self._loop, daemon=True, name="traffic")
                self._thread.start()
                log.info("traffic ON at %.1f tx/s", self.rate)
        return self.status()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("traffic OFF after %d transactions", self.sent)
        return self.status()

    def burst(self, n: int, kind: str | None = None) -> int:
        k = send(self.producer, self.source.burst(n, kind))
        self.producer.flush()
        self.sent += k
        self.attacks += 1
        return k

    def _loop(self) -> None:
        try:
            next_attack = time.time() + self.attack_every
            while not self._stop.is_set():
                self.sent += send(self.producer, [self.source.next()])
                if self.attack_every and time.time() >= next_attack:
                    self.burst(self.attack_size)
                    next_attack = time.time() + self.attack_every * random.uniform(0.7, 1.3)
                self._stop.wait(random.expovariate(self.rate))
            self.producer.flush()
        except Exception as e:  # noqa: BLE001 - shown on the Live stream page instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
            log.exception("traffic generator stopped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    traffic = Traffic()
    traffic.start()
    try:
        while traffic.running:
            time.sleep(60)
            log.info("sent %d transactions", traffic.sent)
    except KeyboardInterrupt:
        traffic.stop()


if __name__ == "__main__":
    main()
