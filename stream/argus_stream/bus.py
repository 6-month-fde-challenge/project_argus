"""Amazon Kinesis Data Streams: the event bus between the producer, the scorer and downstream consumers.

Thread safety: the producer is shared by the traffic thread and API threads (attack waves), so its buffer is guarded
by a lock and swapped atomically on flush; nothing produced during a flush is lost.

    KinesisProducer  buffered PutRecords; records Kinesis throttles are retried with back-off
    ShardReader      GetRecords over every shard of a stream, follows resharding, reports lag (MillisBehindLatest)

Env: AWS_REGION; KINESIS_ENDPOINT only for a local emulator (docker compose: moto); KINESIS_CREATE=1 creates missing
streams (local only - on AWS Terraform owns the streams and the task role may not create them).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

log = logging.getLogger("argus.stream.bus")

MAX_BATCH = 500        # PutRecords limit per call
FLUSH_EVERY_S = 0.2


def aws_region() -> str:
    """AWS_REGION (what ECS / Fargate and our compose file set) or AWS_DEFAULT_REGION (what the SDK reads)."""
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"


def aws_client(service: str, **kw):
    """A boto3 client in this deployment's region (Kinesis, SNS, CloudWatch, S3)."""
    import boto3
    return boto3.client(service, region_name=aws_region(), **kw)


def kinesis_client():
    from botocore.config import Config
    return aws_client("kinesis", endpoint_url=os.getenv("KINESIS_ENDPOINT") or None,
                      config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, connect_timeout=5, read_timeout=20))


def ensure_streams(client, names: list[str], shards: int = 1, wait_s: float = 60) -> None:
    """Create any missing stream and wait until it is ACTIVE (local emulator only). Safe to call from several threads
    at once (the scorer and the traffic producer both do at start-up): whoever loses the create race just waits."""
    for name in names:
        try:
            client.describe_stream_summary(StreamName=name)
        except client.exceptions.ResourceNotFoundException:
            try:
                client.create_stream(StreamName=name, ShardCount=shards)
                log.info("created Kinesis stream %s", name)
            except client.exceptions.ResourceInUseException:
                pass  # created (or being created) by another thread or process: wait for it below
    deadline = time.time() + wait_s
    for name in names:
        while time.time() < deadline:
            if client.describe_stream_summary(StreamName=name)["StreamDescriptionSummary"]["StreamStatus"] == "ACTIVE":
                break
            time.sleep(0.5)


def stream_epoch(client, name: str) -> str:
    """The stream's identity: its creation time. A deleted and recreated stream (Terraform replace, a restarted local
    emulator) restarts its sequence numbers, so checkpoints taken on the old one must not be applied to it."""
    created = client.describe_stream_summary(StreamName=name)["StreamDescriptionSummary"]["StreamCreationTimestamp"]
    return created.isoformat() if hasattr(created, "isoformat") else str(created)


def _partition_key(key) -> str:
    k = key.decode() if isinstance(key, bytes) else str(key or "")
    return (k or "none")[:256]


class KinesisProducer:
    """produce() buffers; poll() sends when the buffer is full or old; flush() sends everything now."""

    def __init__(self, client=None):
        self.client = client or kinesis_client()
        self.buffer: dict[str, list[dict]] = {}
        self.last = time.time()
        self.sent = self.failed = 0
        self._lock = threading.Lock()
        if os.getenv("KINESIS_CREATE") == "1":
            from . import STREAM_DECISIONS, STREAM_TX
            ensure_streams(self.client, [STREAM_TX, STREAM_DECISIONS])

    def produce(self, stream: str, key=None, value=None) -> None:
        data = value if isinstance(value, bytes) else json.dumps(value, default=str).encode()
        with self._lock:
            self.buffer.setdefault(stream, []).append({"Data": data, "PartitionKey": _partition_key(key)})

    def poll(self) -> int:
        with self._lock:
            n = sum(len(v) for v in self.buffer.values())
        if n >= MAX_BATCH or (n and time.time() - self.last >= FLUSH_EVERY_S):
            return self.flush()
        return 0

    def flush(self) -> int:
        with self._lock:  # take everything buffered so far; records produced meanwhile go to the next flush
            pending, self.buffer = self.buffer, {}
            self.last = time.time()
        n = 0
        for stream, records in pending.items():
            for i in range(0, len(records), MAX_BATCH):
                n += self._put(stream, records[i : i + MAX_BATCH])
        return n

    def _put(self, stream: str, records: list[dict]) -> int:
        total = len(records)
        for attempt in range(5):
            resp = self.client.put_records(StreamName=stream, Records=records)
            failed = [rec for rec, res in zip(records, resp["Records"]) if res.get("ErrorCode")]
            self.sent += len(records) - len(failed)
            if not failed:
                return total
            records = failed  # only the throttled / failed records go round again
            time.sleep(0.2 * 2 ** attempt)
        self.failed += len(records)
        # never drop silently: the caller must not advance its checkpoint, so the batch is retried (at-least-once)
        raise RuntimeError(f"Kinesis refused {len(records)} of {total} records for {stream} after 5 attempts")


class ShardReader:
    """Reads every shard of one stream. start: LATEST | TRIM_HORIZON | a POSIX timestamp (AT_TIMESTAMP) for shards
    without a checkpoint; `checkpoints` {shard: sequence number} resumes right after what was already processed.
    Every decoded record gets `_shard` / `_seq`, its position, for the consumer's checkpoint."""

    def __init__(self, stream: str, client=None, start="LATEST", limit: int = 100, checkpoints: dict | None = None):
        self.stream, self.client, self.start, self.limit = stream, client or kinesis_client(), start, limit
        self.iterators: dict[str, str | None] = {}  # shard -> iterator (None: shard closed and drained)
        self.sequence: dict[str, str] = dict(checkpoints or {})  # shard -> last sequence number read
        self.lag_ms: dict[str, int] = {}
        self.next_call: dict[str, float] = {}
        self.listed = 0.0

    def _iterator(self, shard: str) -> str:
        kw: dict = {"StreamName": self.stream, "ShardId": shard}
        if shard in self.sequence:
            kw.update(ShardIteratorType="AFTER_SEQUENCE_NUMBER", StartingSequenceNumber=self.sequence[shard])
        elif isinstance(self.start, (int, float)):
            kw.update(ShardIteratorType="AT_TIMESTAMP", Timestamp=datetime.fromtimestamp(self.start, timezone.utc))
        else:
            kw.update(ShardIteratorType=self.start)
        return self.client.get_shard_iterator(**kw)["ShardIterator"]

    def _list(self) -> None:
        """Pick up new shards after resharding (re-listed every 60 s)."""
        shards, token = [], None
        while True:
            resp = self.client.list_shards(NextToken=token) if token else self.client.list_shards(StreamName=self.stream)
            shards += resp["Shards"]
            token = resp.get("NextToken")
            if not token:
                break
        for s in shards:
            if s["ShardId"] not in self.iterators:
                self.iterators[s["ShardId"]] = self._iterator(s["ShardId"])
        self.listed = time.time()

    @property
    def lag(self) -> int | None:
        return max(self.lag_ms.values()) if self.lag_ms else None

    def read(self) -> tuple[list[dict], bool]:
        """One GetRecords per open shard (at most 5 calls / s / shard). -> (decoded records, caught up)."""
        if not self.iterators or time.time() - self.listed > 60:
            self._list()
        out, caught_up = [], True
        for shard, it in list(self.iterators.items()):
            if it is None:  # closed and drained after resharding
                continue
            if time.time() < self.next_call.get(shard, 0):  # not read this round: cannot claim it is caught up
                caught_up = False
                continue
            try:
                resp = self.client.get_records(ShardIterator=it, Limit=self.limit)
            except self.client.exceptions.ExpiredIteratorException:
                self.iterators[shard] = self._iterator(shard)
                continue
            except self.client.exceptions.ProvisionedThroughputExceededException:
                self.next_call[shard] = time.time() + 1
                caught_up = False
                continue
            self.next_call[shard] = time.time() + 0.21
            for rec in resp["Records"]:
                self.sequence[shard] = rec["SequenceNumber"]
                try:
                    item = json.loads(rec["Data"])
                except ValueError:
                    log.warning("skipping a non-JSON record %s", rec["SequenceNumber"])
                    continue
                if isinstance(item, dict):  # where it came from: the consumer checkpoints by it after processing
                    item["_shard"], item["_seq"] = shard, rec["SequenceNumber"]
                out.append(item)
            self.iterators[shard] = resp.get("NextShardIterator")
            self.lag_ms[shard] = int(resp.get("MillisBehindLatest") or 0)
            if resp["Records"] or self.lag_ms[shard] > 0:
                caught_up = False
        return out, caught_up
