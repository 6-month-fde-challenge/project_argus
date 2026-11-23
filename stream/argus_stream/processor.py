"""Real-time scorer: Kinesis <env>-transactions -> Fraud Model API -> decision -> action -> <env>-decisions.

Every transaction is scored by the Fraud Model API (`POST /v1/fraud/score`, CRD 10.2 -> 10.3) of the run that is active
there, so the stream always uses exactly what the API serves (switch runs on the Scoring page; no restart here).
  unavailable  503, a timeout or a connection error: the CRD fail-open rule (FR-API-01) decides -
               approve below Rs 10,000, otherwise REVIEW - with decision_source FAIL_OPEN. A rejected token (401) is a
               misconfiguration, not an outage: REVIEW with AUTH_ERROR (alarm). After a failure the API is
               skipped for BREAKER_OPEN_S (circuit breaker, LLD 6.1) so a dead API never stalls the stream.
  outputs      Kinesis <env>-decisions: every scored transaction with its action (the hook for downstream systems;
               Kinesis Firehose archives it to the S3 audit bucket on AWS) - SNS: BLOCK actions - SQLite: analytics -
               CloudWatch: counters and API latency every 60 s when CW_NAMESPACE is set.
  shadow       with a shadow run set (POST /live/api/shadow, after the API loaded it as its shadow), every decided batch
               is also scored by that run in a separate pool (POST /v1/fraud/shadow/{run}), after it was stored and
               published; the store compares the two live. The shadow never decides or delays a decision (MLOPS-06).
Delivery is at-least-once and nothing is lost: each batch is published to the decisions stream and flushed (Kinesis
acknowledged every record), then stored, then its per-shard checkpoint advances. A failure anywhere raises before the
checkpoint moves, so the batch is read and scored again; the store ignores repeated event ids. After a restart the
transactions stream resumes right after the checkpoint; on Fargate (no disk) an empty store is first rebuilt from the
decisions stream, whose records carry their source position, so the checkpoints come back with it.
Checkpoints belong to one transactions stream (its creation time, bus.stream_epoch): a recreated stream (Terraform
replace, a restarted local emulator) numbers its records from the start again, so its checkpoints are dropped and it is
read from its start; a store without a recorded stream resumes by time instead.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import FAIL_OPEN_LIMIT, STREAM_DECISIONS, STREAM_TX, action_for
from .store import Store

log = logging.getLogger("argus.stream.processor")
BREAKER_OPEN_S = 15


class ScoringClient:
    """The Fraud Model API with a circuit breaker. score() never raises: a failure returns (None, status)."""

    def __init__(self, url: str, token: str, timeout_s: float = 10.0, deadline_ms: int | None = None, transport=None):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        headers = {"Authorization": f"Bearer {token}"}
        if deadline_ms:
            headers["X-Request-Deadline-Ms"] = str(deadline_ms)  # LLD 6.3: the caller's budget
        self.http = httpx.Client(base_url=self.url or "http://disabled", headers=headers, timeout=timeout_s,
                                 transport=transport)  # transport: tests inject a fake API
        self.open_until = 0.0
        self.status, self.model_version, self.active_run = "unknown", None, None
        self.scored = self.errors = 0
        self.auth_failed = False  # the last answer was 401: while the breaker is open, keep holding (never fail open)
        # shadow (MLOPS-06): a second run scores the same transaction; it never decides, and its own back-off keeps
        # a broken shadow from slowing the live path
        self.shadow_run: str | None = None
        self.shadow_open_until, self.shadow_status = 0.0, "off"

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def check(self) -> bool:
        try:
            body = self.http.get("/v1/health", timeout=3).json()
            self.status, self.model_version, self.active_run = body.get("status"), body.get("model_version"), body.get("active_run")
            ok = body.get("scorer") == "ok"
        except (httpx.HTTPError, ValueError) as e:
            self.status, ok = f"unreachable ({type(e).__name__})", False
        if not ok:
            self.open_until = time.monotonic() + BREAKER_OPEN_S
        return ok

    def ready(self) -> bool:
        return self.enabled and time.monotonic() >= self.open_until

    def score(self, request: dict) -> tuple[dict | None, int | str, float]:
        """-> (CRD 10.3 response or None, HTTP status or error name, client latency ms)."""
        if not self.ready():
            return None, "AUTH_FAILED" if self.auth_failed else "BREAKER_OPEN", 0.0
        start = time.perf_counter()
        try:
            resp = self.http.post("/v1/fraud/score", json=request)
        except httpx.HTTPError as e:
            self.errors += 1
            self.open_until, self.status = time.monotonic() + BREAKER_OPEN_S, f"error ({type(e).__name__})"
            return None, type(e).__name__, (time.perf_counter() - start) * 1000
        ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            body = valid_reply(resp)
            if body is None:  # a 200 that is not a CRD 10.3 response: treat the API as unavailable
                self.errors += 1
                self.open_until, self.status = time.monotonic() + BREAKER_OPEN_S, "bad reply"
                return None, "BAD_REPLY", ms
            self.scored += 1
            self.status, self.auth_failed = "ok", False
            return body, 200, ms
        self.errors += 1
        if resp.status_code == 503 and _retryable(resp):  # shed under load: this one fails open, the next may pass
            self.status = "overloaded (shedding)"
            return None, resp.status_code, ms
        if resp.status_code in (401, 503) or resp.status_code >= 500:  # down, no model, or a token mismatch
            self.open_until, self.status = time.monotonic() + BREAKER_OPEN_S, f"HTTP {resp.status_code}"
            self.auth_failed = resp.status_code == 401
            if self.auth_failed:
                log.error("the Fraud Model API rejected the token: check ARGUS_SCORING_TOKEN on both sides")
        return None, resp.status_code, ms

    def shadow_score(self, request: dict) -> dict | None:
        """The shadow run's answer for the same transaction, or None (off, backing off, or failed)."""
        run = self.shadow_run
        if not run or not self.enabled or time.monotonic() < self.shadow_open_until:
            return None
        try:
            resp = self.http.post(f"/v1/fraud/shadow/{run}", json=request)
        except httpx.HTTPError as e:
            self.shadow_open_until, self.shadow_status = time.monotonic() + BREAKER_OPEN_S, f"error ({type(e).__name__})"
            return None
        body = valid_reply(resp) if resp.status_code == 200 else None
        if body is None:
            self.shadow_open_until = time.monotonic() + BREAKER_OPEN_S
            self.shadow_status = f"HTTP {resp.status_code}" if resp.status_code != 200 else "bad reply"
            return None
        self.shadow_status = "ok"
        return body


def _retryable(resp: httpx.Response) -> bool:
    try:
        return bool(resp.json().get("retry"))
    except (ValueError, AttributeError):
        return False


REPLY_FIELDS = ("decision", "risk_level", "fraud_probability", "decision_source", "reason_codes", "model_version")


def valid_reply(resp: httpx.Response) -> dict | None:
    """The JSON body if it carries every CRD 10.3 field the stream uses and a known decision, else None."""
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict) or any(k not in body for k in REPLY_FIELDS):
        return None
    if body["decision"] not in ("ALLOW", "REVIEW", "BLOCK") or not isinstance(body["reason_codes"], list):
        return None
    return body


def fail_open(amount: float) -> tuple[str, str]:
    """CRD FR-API-01: the caller approves small payments and reviews the rest when the API cannot answer."""
    return ("ALLOW", "UNKNOWN") if amount < FAIL_OPEN_LIMIT else ("REVIEW", "UNKNOWN")


def score_event(event: dict, client: ScoringClient) -> dict:
    """One transaction -> the record that goes to the store, the decisions stream and the metrics."""
    req = event["request"]
    resp, status, ms = client.score(req)
    when = req["transaction_time"]
    record = {
        "event_id": event["event_id"], "ts": event["ts"], "source": event.get("source"),
        "transaction_id": req["transaction_id"], "customer_id": req["customer_id"], "device_id": req.get("device_id"),
        "merchant_category": req["merchant_category"], "amount": req["transaction_amount"],
        "city": (req.get("customer_location") or {}).get("city"), "merchant_city": req["merchant_location"]["city"],
        "hour": int(when[11:13]) if len(when) >= 13 else None, "fraud_label": event.get("fraud_label"),
        "fraud_type": event.get("fraud_type"), "ring_id": event.get("ring_id"),
        "http_status": status if isinstance(status, int) else None, "api_latency_ms": round(ms, 2) if ms else None,
    }
    if resp is not None:
        record.update(decision=resp["decision"], risk_level=resp["risk_level"], probability=resp["fraud_probability"],
                      decision_source=resp["decision_source"], reason_codes=resp["reason_codes"],
                      explanation=resp.get("explanation"), model_version=resp["model_version"])
    elif status in (401, "AUTH_FAILED"):  # a token mismatch is a misconfiguration, not an outage: never approve on it, hold everything
        record.update(decision="REVIEW", risk_level="UNKNOWN", probability=None, decision_source="AUTH_ERROR",
                      reason_codes=[], explanation="The Fraud Model API rejected this service's token (401): held for "
                                                   "review until ARGUS_SCORING_TOKEN matches on both sides.",
                      model_version=None)
    elif status == 400:  # the Payment API would reject it; hold it instead of guessing
        record.update(decision="REVIEW", risk_level="UNKNOWN", probability=None, decision_source="INVALID_REQUEST",
                      reason_codes=[], explanation="The Fraud Model API rejected the request as invalid (400).",
                      model_version=None)
    else:
        decision, risk = fail_open(req["transaction_amount"])
        record.update(decision=decision, risk_level=risk, probability=None, decision_source="FAIL_OPEN", reason_codes=[],
                      explanation=f"Fraud Model API unavailable ({status}): fail-open rule, "
                                  f"{'approved below' if decision == 'ALLOW' else 'held for review at or above'} "
                                  f"Rs {FAIL_OPEN_LIMIT:,.0f} (FR-API-01).", model_version=None)
    record["action"] = action_for(record["decision"], record["risk_level"])[0]
    record["scored_ts"] = time.time()
    return record


def shadow_fields(request: dict, client: ScoringClient) -> dict | None:
    """The shadow run's view of a transaction that has already been decided (never changes that decision)."""
    shadow = client.shadow_score(request)
    if not shadow:
        return None
    return {"shadow_run": shadow.get("shadow_run", client.shadow_run), "shadow_decision": shadow["decision"],
            "shadow_probability": shadow["fraud_probability"], "shadow_model_version": shadow["model_version"],
            "shadow_latency_ms": shadow.get("latency_ms")}


class Metrics:
    """Counters and API latency, pushed to CloudWatch (namespace CW_NAMESPACE) once a minute."""

    NAMES = ("Transactions", "Allowed", "Reviewed", "Blocked", "FailOpen", "AuthErrors", "LabelledFraud", "FraudMissed")

    def __init__(self, namespace: str = ""):
        self.namespace, self.cw = namespace, None
        if namespace:
            try:
                from .bus import aws_client
                self.cw = aws_client("cloudwatch")
            except Exception as e:  # noqa: BLE001 - metrics must never stop scoring
                log.warning("CloudWatch metrics disabled: %s", e)
        self.reset()

    def reset(self) -> None:
        self.counts = dict.fromkeys(self.NAMES, 0)
        self.latency: list[float] = []
        self.started = time.time()

    def add(self, records: list[dict]) -> None:
        for r in records:
            self.counts["Transactions"] += 1
            self.counts[{"ALLOW": "Allowed", "REVIEW": "Reviewed", "BLOCK": "Blocked"}[r["decision"]]] += 1
            self.counts["FailOpen"] += r["decision_source"] == "FAIL_OPEN"
            self.counts["AuthErrors"] += r["decision_source"] == "AUTH_ERROR"
            if r.get("fraud_label"):
                self.counts["LabelledFraud"] += 1
                self.counts["FraudMissed"] += r["decision"] == "ALLOW"
            if r.get("api_latency_ms") and r["http_status"] == 200:
                self.latency.append(r["api_latency_ms"])

    def flush(self, lag_ms: int | None = None, force: bool = False) -> None:
        if not force and time.time() - self.started < 60:
            return
        if self.cw:
            dims = [{"Name": "Service", "Value": "stream"}]
            data = [{"MetricName": k, "Dimensions": dims, "Value": float(v), "Unit": "Count"} for k, v in self.counts.items()]
            if lag_ms is not None:
                data.append({"MetricName": "StreamLagMs", "Dimensions": dims, "Value": float(lag_ms), "Unit": "Milliseconds"})
            if self.latency:  # raw values (an even sample of at most 150), so CloudWatch can alarm on p99 (OBJ-10)
                step = -(-len(self.latency) // 150)
                data.append({"MetricName": "ApiLatency", "Dimensions": dims, "Unit": "Milliseconds",
                             "Values": self.latency[::step]})
            try:
                for i in range(0, len(data), 20):
                    self.cw.put_metric_data(Namespace=self.namespace, MetricData=data[i : i + 20])
            except Exception as e:  # noqa: BLE001
                log.warning("put_metric_data failed: %s", e)
        log.info("metrics %s", json.dumps({**self.counts, "lag_ms": lag_ms}))
        self.reset()


def client_from_env() -> ScoringClient:
    client = ScoringClient(os.getenv("SCORING_URL", "http://scoring:8080"),
                           os.getenv("ARGUS_SCORING_TOKEN", "demo-service-token"),
                           float(os.getenv("SCORING_TIMEOUT_S", "10")),
                           int(os.getenv("SCORING_DEADLINE_MS", "0")) or None)
    return client  # the shadow run is set at runtime (UI -> platform -> POST /live/api/shadow), after the API loads it


def positions(events: list[dict]) -> dict[str, str]:
    """The furthest Kinesis sequence number per shard in a batch of transactions."""
    out: dict[str, str] = {}
    for e in events:
        shard, seq = e.get("_shard"), e.get("_seq")
        if shard and seq and (shard not in out or int(seq) > int(out[shard])):
            out[shard] = seq
    return out


def source_positions(decisions: list[dict]) -> dict[str, str]:
    """The same, from decision records (rebuild): each carries the transaction's source_shard / source_seq."""
    return positions([{"_shard": d.get("source_shard"), "_seq": d.get("source_seq")} for d in decisions])


class Processor(threading.Thread):
    def __init__(self, store: Store, client: ScoringClient | None = None, kinesis=None):
        super().__init__(daemon=True, name="processor")
        self.store, self.kinesis = store, kinesis
        self.client = client or client_from_env()
        self.batch = int(os.getenv("BATCH_SIZE", "32"))
        self.pool = ThreadPoolExecutor(max_workers=int(os.getenv("SCORING_CONCURRENCY", "8")))
        # shadow scoring has its own small pool and runs after the decision is stored and published: a slow or
        # broken shadow can never delay a live decision (MLOPS-06)
        self.shadow_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")
        self.last_shadow_job = None
        self.metrics = Metrics(os.getenv("CW_NAMESPACE", ""))
        self.sns_topic, self.sns, self.last_sns = os.getenv("SNS_ACTIONS_TOPIC_ARN", ""), None, 0.0
        self.pending_blocks: list[dict] = []  # BLOCK alerts waiting for the next SNS window (never dropped)
        self.stop_event = threading.Event()
        self.tx_epoch: str | None = None  # identity of the transactions stream being consumed (bus.stream_epoch)
        self.state = {"stream": "connecting", "consumed": 0, "lag_ms": None, "last_batch_ts": None,
                      "started": time.time(), "errors": 0, "rebuilt": 0, "dead_letters": 0}

    def _score(self, event: dict) -> dict | None:
        """One event; a malformed one is logged and counted (dead letter), never allowed to stop the stream."""
        try:
            return score_event(event, self.client)
        except Exception:  # noqa: BLE001
            self.state["dead_letters"] += 1
            log.exception("skipping a malformed event %s", str(event)[:300])
            return None

    def handle(self, events: list[dict], producer=None) -> list[dict]:
        """Score -> publish and flush -> store -> checkpoint. Any exception leaves the checkpoint where it was."""
        scored = list(self.pool.map(self._score, events))  # API calls in parallel, results in order
        records = [r for r in scored if r is not None]
        for e, r in zip(events, scored):
            if r is not None and e.get("_shard"):
                r["source_shard"], r["source_seq"], r["source_epoch"] = e["_shard"], e["_seq"], self.tx_epoch
        if producer is not None:
            for r in records:
                producer.produce(STREAM_DECISIONS, key=r["customer_id"], value=r)
            producer.flush()  # raises if Kinesis refuses records after its retries: nothing below runs
        self.store.insert(records)
        self.store.save_checkpoints(positions(events))  # dead letters too: a malformed event is never retried
        self._notify(records)
        self.metrics.add(records)
        self.state["consumed"] += len(records)
        self.state["last_batch_ts"] = time.time()
        if self.client.shadow_run and records:
            if self.last_shadow_job is not None and not self.last_shadow_job.done():
                self.state["shadow_skipped"] = self.state.get("shadow_skipped", 0) + len(records)  # never queue up
            else:
                requests = {e["event_id"]: e["request"] for e, r in zip(events, scored) if r is not None}
                self.last_shadow_job = self.shadow_pool.submit(self._shadow, requests)
        return records

    def _shadow(self, requests: dict[str, dict]) -> int:
        """Score an already-decided batch with the shadow run and attach the result to the stored records."""
        try:
            rows = [(event_id, fields) for event_id, req in requests.items()
                    if (fields := shadow_fields(req, self.client)) is not None]
            self.store.set_shadow(rows)
            return len(rows)
        except Exception:  # noqa: BLE001 - analytics only
            log.exception("shadow scoring failed")
            return 0

    def _notify(self, records: list[dict]) -> None:
        """Every BLOCK goes to SNS: at most one message a second, each carrying all BLOCKs since the last one."""
        if not self.sns_topic:
            return
        self.pending_blocks += [r for r in records if r["decision"] == "BLOCK"]
        if not self.pending_blocks or time.time() - self.last_sns < 1:
            return
        blocks = self.pending_blocks
        try:
            if self.sns is None:
                from .bus import aws_client
                self.sns = aws_client("sns")
            message = {"action": "BLOCK_AND_ALERT", "count": len(blocks),
                       "transactions": [{k: r.get(k) for k in ("transaction_id", "customer_id", "amount", "probability",
                                                               "reason_codes", "model_version")} for r in blocks[:50]]}
            self.sns.publish(TopicArn=self.sns_topic, Subject="ARGUS: payment blocked", Message=json.dumps(message, default=str))
            self.pending_blocks, self.last_sns = [], time.time()
        except Exception as e:  # noqa: BLE001 - kept for the next attempt
            self.pending_blocks = blocks[-1000:]
            log.warning("SNS publish failed (%d BLOCKs kept for retry): %s", len(blocks), e)

    def _client(self):
        if self.kinesis is None:
            from .bus import kinesis_client
            self.kinesis = kinesis_client()
        return self.kinesis

    def rebuild(self, max_seconds: float = 300) -> int:
        """Empty store after a restart -> reload past decisions from the decisions stream (read-only), until every
        shard has been read up to its end. Checkpoints are restored only from decisions taken on the transactions
        stream that exists now (source_epoch): never from a stream that was since recreated."""
        from .bus import ShardReader
        if self.store.head() or os.getenv("REBUILD_FROM_STREAM", "1") != "1":
            return 0
        reader, n, began = ShardReader(STREAM_DECISIONS, self._client(), start="TRIM_HORIZON", limit=1000), 0, time.time()
        while time.time() - began < max_seconds:
            records, caught_up = reader.read()
            self.store.insert(records)
            current = [d for d in records if self.tx_epoch and d.get("source_epoch") == self.tx_epoch]
            if current:
                self.store.restore_epoch(self.tx_epoch)
                self.store.save_checkpoints(source_positions(current))  # where the transactions stream resumes
            n += len(records)
            if caught_up:
                break
            time.sleep(0.05 if records else 0.25)
        else:
            log.warning("rebuild stopped after %d s with %d decisions: the oldest ones may be missing", max_seconds, n)
        log.info("rebuilt the analytics store from %s: %d decisions", STREAM_DECISIONS, n)
        self.state["rebuilt"] = n
        return n

    def run(self) -> None:
        from .bus import KinesisProducer, ShardReader, stream_epoch
        self.client.check()
        while not self.stop_event.is_set():
            try:
                producer = KinesisProducer(self._client())  # KINESIS_CREATE=1 creates the streams locally
                self.tx_epoch = stream_epoch(self._client(), STREAM_TX)
                self.rebuild()
                last = self.store.last_event_ts()
                start = self.resume_point(self.tx_epoch, last)
                # shards with a checkpoint resume right after it; any other shard (new after resharding) uses `start`
                reader = ShardReader(STREAM_TX, self._client(), start=start, limit=self.batch,
                                     checkpoints=self.store.checkpoints())
                log.info("consuming Kinesis %s -> %s", STREAM_TX, self.client.url)
                while not self.stop_event.is_set():
                    events, _ = reader.read()
                    self.state["stream"] = "connected"
                    for i in range(0, len(events), self.batch):
                        self.handle(events[i : i + self.batch], producer)
                    producer.poll()
                    self.state["lag_ms"] = reader.lag
                    self.metrics.flush(reader.lag)
                    if not events:
                        self._notify([])  # BLOCK alerts still waiting for their SNS window
                        if time.monotonic() >= self.client.open_until and self.client.status != "ok":
                            self.client.check()  # keep health fresh while idle
                        self.stop_event.wait(0.25)
                producer.flush()
            except Exception as e:  # noqa: BLE001 - the scorer thread must survive anything and retry
                self.state["stream"] = f"error: {type(e).__name__}"
                self.state["errors"] += 1
                log.exception("processor error, retrying in 5 s: %s", e)
                self.stop_event.wait(5)

    def resume_point(self, epoch: str, last_ts: float):
        """Where shards without a checkpoint start. Checkpoints only count on the stream they were taken on: a
        recreated stream numbers its records from the start again, so an old checkpoint would skip new records."""
        verdict = self.store.bind_checkpoints(epoch)
        if verdict == "new":
            log.warning("%s was recreated (%s): old checkpoints dropped, reading the new stream from its start",
                        STREAM_TX, epoch)
            return "TRIM_HORIZON"  # everything in a new stream is unprocessed; repeated event ids are ignored
        if verdict == "unknown" and last_ts:
            log.info("no stream identity recorded yet: resuming by time, just before the newest stored event")
        return (last_ts - 5) if last_ts else "LATEST"

    def stop(self) -> None:
        self.stop_event.set()
