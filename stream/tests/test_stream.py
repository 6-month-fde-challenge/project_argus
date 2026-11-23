"""Stream service: generator, Fraud Model API client + fail-open rule, actions, analytics store, API, and the Kinesis
bus (moto, in memory). No network, no AWS account, no model."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from argus_stream import FAIL_OPEN_LIMIT, action_for
from argus_stream.bus import stream_epoch
from argus_stream.generator import FRAUD_TYPES, HoldoutSource, SyntheticSource, make_source
from argus_stream.processor import BREAKER_OPEN_S, Metrics, Processor, ScoringClient, fail_open, score_event
from argus_stream.store import Store

ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MCC = {"education", "electronics", "food", "gaming", "gift_cards", "grocery", "healthcare", "other", "quick_commerce",
       "travel"}


def assert_crd_request(req: dict) -> None:
    """The CRD 10.2 / LLD 4.1 rules the Fraud Model API validates."""
    assert ID.match(req["transaction_id"]) and ID.match(req["customer_id"])
    assert 0 < req["transaction_amount"] <= 10_000_000 and round(req["transaction_amount"], 2) == req["transaction_amount"]
    assert req["merchant_category"] in MCC
    assert datetime.fromisoformat(req["transaction_time"]).utcoffset() is not None
    assert 18 <= req["customer_age"] <= 120 and req["account_age"] >= 0
    for loc in (req["customer_location"], req["merchant_location"]):
        assert -90 <= loc["lat"] <= 90 and -180 <= loc["lon"] <= 180 and loc["city"]


# ------------------------------------------------------------------ generator
def test_synthetic_events_are_valid_crd_requests_with_ground_truth():
    g = SyntheticSource(fraud_rate=0.5, seed=7)
    events = [g.next() for _ in range(300)]
    assert {e["fraud_label"] for e in events} == {0, 1}
    for e in events:
        assert_crd_request(e["request"])
        assert e["fraud_type"] in FRAUD_TYPES + ("genuine",)
    wave = g.burst(10, "card_testing")
    assert len(wave) == 10 and all(e["fraud_label"] == 1 and e["fraud_type"] == "card_testing" for e in wave)
    assert len({e["ring_id"] for e in wave}) == 1  # one ring per attack wave


def test_fraud_patterns_carry_the_signals_the_api_sees(monkeypatch):
    monkeypatch.setenv("TRANSACTION_TIME", "pattern")
    g = SyntheticSource(seed=1)
    takeover = g.fraud("account_takeover")["request"]
    assert datetime.fromisoformat(takeover["transaction_time"]).hour < 5  # R03 odd hour
    assert takeover["customer_location"]["city"] != takeover["merchant_location"]["city"]  # R02 geo mismatch
    assert g.fraud("mule_account")["request"]["account_age"] <= 20  # R06 new account


def test_transaction_time_now_mode_fits_the_15_minute_window(monkeypatch):
    monkeypatch.setenv("TRANSACTION_TIME", "now")
    g = SyntheticSource(seed=2)
    for e in [g.fraud("account_takeover") for _ in range(20)]:
        when = datetime.fromisoformat(e["request"]["transaction_time"])
        assert abs(datetime.now(timezone.utc) - when) < timedelta(minutes=1)


def test_holdout_replay_keeps_the_dataset_ids(tmp_path):
    pd.DataFrame([{
        "transaction_id": f"TX_{i:010d}", "customer_id": "C_abc", "transaction_amount": 999.5, "merchant_category": "food",
        "transaction_time": pd.Timestamp("2026-01-05T10:00:00Z"), "device_id": None if i == 1 else "D_1",
        "customer_age": 30, "customer_location": json.dumps({"city": "Pune", "lat": 18.5, "lon": 73.8}),
        "merchant_location": json.dumps({"city": "Pune", "lat": 18.5, "lon": 73.9}), "account_age": 400,
        "fraud_label": int(i == 0)} for i in range(3)]).to_parquet(tmp_path / "part-0.parquet")
    src = HoldoutSource(str(tmp_path), seed=3)
    e = src.next()
    assert e["source"] == "holdout" and e["request"]["transaction_id"].startswith("TX_")
    assert_crd_request(e["request"])
    assert all(x["fraud_label"] == 1 for x in src.burst(5))


def test_missing_holdout_falls_back_to_synthetic(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_URI", str(tmp_path / "nothing"))
    assert isinstance(make_source("mix"), SyntheticSource)
    with pytest.raises(FileNotFoundError):
        make_source("holdout")


# ------------------------------------------------------------------ actions and the fail-open rule
def test_actions_follow_crd_10_4():
    assert action_for("ALLOW", "LOW")[0] == "APPROVE"
    assert action_for("ALLOW", "MEDIUM")[0] == "APPROVE_STEP_UP"
    assert action_for("ALLOW", "UNKNOWN")[0] == "APPROVE"
    assert action_for("REVIEW", "HIGH")[0] == "HOLD_FOR_REVIEW"
    assert action_for("BLOCK", "CRITICAL")[0] == "BLOCK_AND_ALERT"
    assert action_for("???")[0] == "HOLD_FOR_REVIEW"


def test_fail_open_threshold():
    assert fail_open(FAIL_OPEN_LIMIT - 0.01)[0] == "ALLOW" and fail_open(FAIL_OPEN_LIMIT)[0] == "REVIEW"


# ------------------------------------------------------------------ the Fraud Model API client
def fake_api(status: int = 200, decision: str = "BLOCK", calls: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"status": "ok", "scorer": "ok", "model_version": "argus-lm-x-fp8", "active_run": "x"})
        if status != 200:
            return httpx.Response(status, json={"error": "UNAVAILABLE"})
        body = json.loads(request.content)
        if request.url.path.startswith("/v1/fraud/shadow/"):  # the shadow run is stricter: it blocks everything
            return httpx.Response(200, json={
                "transaction_id": body["transaction_id"], "shadow_run": request.url.path.rsplit("/", 1)[1],
                "fraud_probability": 0.99, "decision": "BLOCK", "risk_level": "CRITICAL", "reason_codes": [],
                "decision_source": "MODEL", "model_version": "argus-xgb-shadow", "latency_ms": 3})
        return httpx.Response(200, json={
            "transaction_id": body["transaction_id"], "fraud_probability": 0.97, "decision": decision,
            "risk_level": "CRITICAL" if decision == "BLOCK" else "LOW", "reason_codes": ["R01_NEW_DEVICE"],
            "explanation": "x", "decision_source": "MODEL", "model_version": "argus-lm-x-fp8", "latency_ms": 12})
    return httpx.MockTransport(handler)


def client_for(status: int = 200, decision: str = "BLOCK", calls: list | None = None) -> ScoringClient:
    return ScoringClient("http://scoring:8080", "tok", transport=fake_api(status, decision, calls))


def event(amount: float = 50_000.0, label: int = 1) -> dict:
    e = SyntheticSource(seed=5).fraud("account_takeover")
    e["request"]["transaction_amount"], e["fraud_label"] = amount, label
    return e


def test_a_scored_event_carries_the_api_decision_and_its_action():
    calls: list = []
    rec = score_event(event(), client_for(calls=calls))
    assert rec["decision"] == "BLOCK" and rec["action"] == "BLOCK_AND_ALERT" and rec["decision_source"] == "MODEL"
    assert rec["http_status"] == 200 and rec["model_version"] == "argus-lm-x-fp8"
    assert calls[0].headers["authorization"] == "Bearer tok"


@pytest.mark.parametrize("amount,decision", [(9_999.0, "ALLOW"), (10_000.0, "REVIEW")])
def test_api_down_applies_fail_open_and_opens_the_breaker(amount, decision):
    calls: list = []
    client = client_for(503, calls=calls)
    rec = score_event(event(amount), client)
    assert (rec["decision"], rec["decision_source"], rec["http_status"]) == (decision, "FAIL_OPEN", 503)
    n = len(calls)
    rec2 = score_event(event(amount), client)  # breaker open: no call, immediate fail-open
    assert len(calls) == n and rec2["decision_source"] == "FAIL_OPEN" and rec2["http_status"] is None
    assert BREAKER_OPEN_S >= 5


def test_a_shed_request_fails_open_without_opening_the_breaker():
    def shedding(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "UNAVAILABLE", "retry": True})

    client = ScoringClient("http://scoring:8080", "tok", transport=httpx.MockTransport(shedding))
    rec = score_event(event(5_000.0), client)
    assert rec["decision_source"] == "FAIL_OPEN" and rec["decision"] == "ALLOW"
    assert client.ready() and client.status == "overloaded (shedding)"  # the next transaction tries again


def test_invalid_request_is_held_not_guessed():
    rec = score_event(event(), client_for(400))
    assert (rec["decision"], rec["decision_source"], rec["action"]) == ("REVIEW", "INVALID_REQUEST", "HOLD_FOR_REVIEW")


def test_no_scoring_url_means_fail_open_everywhere():
    rec = score_event(event(500.0), ScoringClient("", "tok"))
    assert rec["decision"] == "ALLOW" and rec["decision_source"] == "FAIL_OPEN"


# ------------------------------------------------------------------ store and metrics
def records() -> list[dict]:
    """3 frauds blocked, 5 genuine allowed, 1 fraud missed, 1 genuine blocked."""
    out = []
    for decision, label, n in (("BLOCK", 1, 3), ("ALLOW", 0, 5), ("ALLOW", 1, 1), ("BLOCK", 0, 1)):
        client = client_for(decision=decision)
        out += [score_event(event(label=label), client) for _ in range(n)]
    return out


def test_store_is_idempotent_and_measures_live_accuracy():
    store = Store(":memory:")
    recs = records()
    store.insert(recs)
    store.insert(recs[:3])  # at-least-once redelivery
    s = store.summary(60)
    assert s["totals"]["n"] == 10 and s["totals"]["block"] == 4
    acc = s["live_accuracy"]
    assert acc["recall"] == 0.75 and acc["precision_at_block"] == 0.75 and acc["false_decline_rate"] == 1 / 6
    assert s["reasons"]["R01_NEW_DEVICE"] == 4 and s["api_latency"]["n"] == 10
    assert store.recent(1)[0]["reason_codes"] == ["R01_NEW_DEVICE"]


def test_metrics_count_decisions_and_misses():
    m = Metrics()
    m.add(records())
    assert m.counts["Transactions"] == 10 and m.counts["Blocked"] == 4 and m.counts["FraudMissed"] == 1


def test_processor_publishes_every_decision():
    class FakeProducer:
        def __init__(self):
            self.sent = []

        def produce(self, stream, key=None, value=None):
            self.sent.append((stream, key, value))

        def poll(self):
            return 0

        def flush(self):
            return 0

    store, producer = Store(":memory:"), FakeProducer()
    p = Processor(store, client_for(), kinesis=object())
    out = p.handle([event() for _ in range(4)], producer)
    assert len(out) == 4 and len(producer.sent) == 4 and store.head() == 4
    assert all(stream.endswith("decisions") and v["action"] == "BLOCK_AND_ALERT" for stream, _, v in producer.sent)


class NullProducer:
    def __init__(self):
        self.sent = []

    def produce(self, stream, key=None, value=None):
        self.sent.append(value)

    def poll(self):
        return 0

    def flush(self):
        return 0


def test_shadow_runs_after_the_decision_is_published_and_never_decides(tmp_path):
    client = client_for(decision="ALLOW")
    client.shadow_run = "argus_demo_hybrid"
    store, producer = Store(str(tmp_path / "s.db")), NullProducer()
    p = Processor(store, client, kinesis=object())
    records = p.handle([event(label=0), event(label=1)], producer)
    assert all(r["decision"] == "ALLOW" and r["action"] == "APPROVE" for r in records)  # the live model decides
    assert all("shadow_decision" not in v for v in producer.sent)  # published before the shadow is even asked
    assert p.last_shadow_job.result(timeout=10) == 2
    stored = store.recent(2)
    assert {(r["shadow_run"], r["shadow_decision"]) for r in stored} == {("argus_demo_hybrid", "BLOCK")}
    s = store.summary(60)["shadow"][0]
    assert s["n"] == 2 and s["agreement"] == 0 and s["live_recall"] == 0 and s["shadow_recall"] == 1
    assert s["shadow_precision_at_block"] == 0.5 and s["model_version"] == "argus-xgb-shadow"


def test_a_failing_shadow_backs_off_without_touching_the_decision():
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.startswith("/v1/fraud/shadow/"):
            return httpx.Response(409, json={"error": "NOT_SHADOW"})
        return fake_api(decision="REVIEW").handle_request(request)

    client = ScoringClient("http://scoring:8080", "tok", transport=httpx.MockTransport(handler))
    client.shadow_run = "not_designated"
    p = Processor(Store(":memory:"), client, kinesis=object())
    out = p.handle([event(), event()], NullProducer())
    assert [r["decision"] for r in out] == ["REVIEW", "REVIEW"]
    assert p.last_shadow_job.result(timeout=10) == 0
    assert client.shadow_status == "HTTP 409" and sum(c.startswith("/v1/fraud/shadow/") for c in calls) == 1  # backed off


def test_a_malformed_event_or_reply_never_stops_the_stream():
    p = Processor(Store(":memory:"), client_for(), kinesis=object())
    out = p.handle([event(), {"event_id": "broken"}, event()], NullProducer())  # no request at all
    assert len(out) == 2 and p.state["dead_letters"] == 1

    def bad_reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"decision": "MAYBE"}) if "/score" in request.url.path else \
            httpx.Response(200, json={"scorer": "ok"})

    client = ScoringClient("http://scoring:8080", "tok", transport=httpx.MockTransport(bad_reply))
    rec = score_event(event(500.0), client)  # a 200 that is not a CRD 10.3 response -> fail-open, breaker opens
    assert rec["decision_source"] == "FAIL_OPEN" and client.status == "bad reply"


def test_block_alerts_are_queued_not_dropped(monkeypatch):
    sent = []

    class FakeSns:
        def publish(self, **kw):
            sent.append(json.loads(kw["Message"]))

    p = Processor(Store(":memory:"), client_for(decision="BLOCK"), kinesis=object())
    p.sns_topic, p.sns = "arn:aws:sns:ap-south-1:0:argus-prod-fraud-actions", FakeSns()
    p.handle([event()], NullProducer())          # first window: published at once
    p.handle([event(), event()], NullProducer())  # within 1 s: queued
    assert [m["count"] for m in sent] == [1] and len(p.pending_blocks) == 2
    p.last_sns -= 2
    p._notify([])                                 # next window: everything queued goes out
    assert [m["count"] for m in sent] == [1, 2] and p.pending_blocks == []


def test_producer_flush_keeps_records_produced_meanwhile():
    from argus_stream.bus import KinesisProducer

    class SlowKinesis:
        def __init__(self):
            self.records = []

        def put_records(self, StreamName, Records):
            producer.produce("s", key="k", value={"late": True})  # arrives while a flush is sending
            self.records += Records
            return {"FailedRecordCount": 0, "Records": [{} for _ in Records]}

    kinesis = SlowKinesis()
    producer = KinesisProducer(kinesis)
    producer.produce("s", key="k", value={"early": True})
    assert producer.flush() == 1 and len(producer.buffer["s"]) == 1  # the record produced mid-flush is kept
    assert producer.flush() == 1 and [json.loads(r["Data"]) for r in kinesis.records] == [{"early": True}, {"late": True}]


def test_an_old_store_gains_the_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, ts REAL, decision TEXT)")
    conn.commit()
    conn.close()
    store = Store(str(path))
    store.insert(records())
    assert store.head() == 10 and store.summary(60)["totals"]["n"] == 10


# ------------------------------------------------------------------ API
class FakeTraffic:
    def __init__(self):
        self.on, self.rate, self.bursts = False, 2.0, []

    def status(self):
        return {"running": self.on, "rate": self.rate}

    def start(self, rate=None):
        self.on, self.rate = True, rate or self.rate
        return self.status()

    def stop(self):
        self.on = False
        return self.status()

    def burst(self, n, kind=None):
        self.bursts.append((n, kind))
        return n


@pytest.fixture
def api(monkeypatch):
    from argus_stream import api as module
    monkeypatch.delenv("KINESIS_ENABLED", raising=False)
    monkeypatch.setenv("CONTROL_API_KEY", "ctl")
    module.S.clear()
    module.S["store"] = Store(":memory:")
    module.S["store"].insert(records())
    with TestClient(module.app) as client:
        module.S["traffic"] = FakeTraffic()
        yield client, module.S["traffic"]
    module.S.clear()


def test_api_reads_need_the_key_except_health(api):
    client, _ = api
    key = {"X-Control-Key": "ctl"}
    assert client.get("/live/health").json()["events"] == 10
    for path in ("/live/api/summary", "/live/api/events", "/live/api/state", "/live/api/producer"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-Control-Key": "wrong"}).status_code == 401
    assert client.get("/live/api/summary?minutes=30", headers=key).json()["totals"]["n"] == 10
    assert len(client.get("/live/api/events?limit=3", headers=key).json()["events"]) == 3
    assert client.get("/live/api/state", headers=key).json()["stream"] == "disabled"


def test_api_control_needs_the_key(api):
    client, traffic = api
    key = {"X-Control-Key": "ctl"}
    assert client.post("/live/api/producer", json={"action": "start"}).status_code == 401
    assert client.post("/live/api/producer", json={"action": "start", "rate": 5}, headers=key).json()["running"]
    assert traffic.rate == 5
    assert client.post("/live/api/producer", json={"action": "jump"}, headers=key).status_code == 422
    assert client.post("/live/api/inject", json={"fraud_type": "x"}, headers=key).status_code == 400
    r = client.post("/live/api/inject", json={"n": 7, "fraud_type": "mule_account"}, headers=key)
    assert r.json()["sent"] == 7 and traffic.bursts == [(7, "mule_account")]
    assert client.post("/live/api/inject", json={"n": 7}, headers=key).status_code == 429  # one wave per 3 s


def test_api_sets_the_shadow_run(api):
    from argus_stream import api as module
    client, _ = api
    key = {"X-Control-Key": "ctl"}
    assert client.post("/live/api/shadow", json={"run": "run_b"}).status_code == 401
    assert client.post("/live/api/shadow", json={"run": "run_b"}, headers=key).status_code == 503  # no processor yet
    module.S["processor"] = Processor(module.S["store"], client_for(), kinesis=object())
    assert client.post("/live/api/shadow", json={"run": "../etc"}, headers=key).status_code == 422
    assert client.post("/live/api/shadow", json={"run": "argus_demo_hybrid"}, headers=key).json() == {"shadow_run": "argus_demo_hybrid"}
    assert client.get("/live/api/state", headers=key).json()["scoring"]["shadow_run"] == "argus_demo_hybrid"
    assert client.post("/live/api/shadow", json={"run": None}, headers=key).json() == {"shadow_run": None}


# ------------------------------------------------------------------ Kinesis bus (moto, in memory)
def test_clients_use_aws_region_like_ecs_and_compose_set_it(monkeypatch):
    from argus_stream.bus import aws_client
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "ap-south-1")  # the only one docker-compose.yml / ECS set
    for service in ("kinesis", "sns", "cloudwatch", "s3"):
        assert aws_client(service).meta.region_name == "ap-south-1"  # botocore alone raised NoRegionError here



def test_kinesis_round_trip(monkeypatch):
    from moto import mock_aws
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("KINESIS_CREATE", "1")
    monkeypatch.delenv("KINESIS_ENDPOINT", raising=False)
    with mock_aws():
        from argus_stream import STREAM_TX
        from argus_stream.bus import KinesisProducer, ShardReader, kinesis_client
        client = kinesis_client()
        producer = KinesisProducer(client)
        reader = ShardReader(STREAM_TX, client, start="TRIM_HORIZON")
        assert stream_epoch(client, STREAM_TX)
        sent = [SyntheticSource(seed=9).next() for _ in range(25)]
        for e in sent:
            producer.produce(STREAM_TX, key=e["request"]["customer_id"], value=e)
        assert producer.flush() == 25 and producer.sent == 25
        got: list = []
        for _ in range(5):
            batch, _ = reader.read()
            got += batch
            reader.next_call.clear()
        assert [e["event_id"] for e in got] == [e["event_id"] for e in sent]


def tagged(seq: int, shard: str = "shardId-000000000000", **kw) -> dict:
    e = event(**kw)
    e["_shard"], e["_seq"] = shard, str(seq)
    return e


def test_a_refused_publish_stores_nothing_and_keeps_the_checkpoint():
    class RefusingProducer(NullProducer):
        def flush(self):
            raise RuntimeError("Kinesis refused 2 of 2 records")

    store = Store(":memory:")
    p = Processor(store, client_for(), kinesis=object())
    with pytest.raises(RuntimeError):
        p.handle([tagged(5), tagged(6)], RefusingProducer())
    assert store.head() == 0 and store.checkpoints() == {}  # retried from the old checkpoint: nothing lost

    out = p.handle([tagged(5), tagged(6)], NullProducer())
    assert len(out) == 2 and store.checkpoints() == {"shardId-000000000000": "6"}
    assert {r["source_seq"] for r in store.recent(2)} == {"5", "6"}


def test_checkpoints_only_move_forward_and_compare_as_numbers():
    store = Store(":memory:")
    store.save_checkpoints({"a": "99"})
    store.save_checkpoints({"a": "100", "b": "7"})
    store.save_checkpoints({"a": "98"})
    assert store.checkpoints() == {"a": "100", "b": "7"}


def test_a_running_shadow_job_is_not_queued_behind():
    import threading as th
    gate = th.Event()
    client = client_for(decision="ALLOW")
    client.shadow_run = "argus_demo_hybrid"
    p = Processor(Store(":memory:"), client, kinesis=object())
    p._shadow = lambda requests: gate.wait(10) and len(requests)  # a slow shadow
    p.handle([event()], NullProducer())
    first = p.last_shadow_job
    p.handle([event(), event()], NullProducer())
    assert p.last_shadow_job is first and p.state["shadow_skipped"] == 2
    gate.set()
    first.result(timeout=10)


def test_restart_resumes_after_the_checkpoint(monkeypatch):
    from moto import mock_aws
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("KINESIS_CREATE", "1")
    monkeypatch.delenv("KINESIS_ENDPOINT", raising=False)
    with mock_aws():
        from argus_stream import STREAM_TX
        from argus_stream.bus import KinesisProducer, ShardReader, kinesis_client
        kin = kinesis_client()
        producer = KinesisProducer(kin)
        sent = [SyntheticSource(seed=3).next() for _ in range(6)]
        for e in sent:
            producer.produce(STREAM_TX, key=e["request"]["customer_id"], value=e)
        producer.flush()
        store = Store(":memory:")
        p = Processor(store, client_for(), kinesis=kin)
        p.tx_epoch = stream_epoch(kin, STREAM_TX)
        assert p.resume_point(p.tx_epoch, 0) == "LATEST"  # first start: this stream is recorded
        reader = ShardReader(STREAM_TX, kin, start="TRIM_HORIZON")
        got, _ = reader.read()
        p.handle(got[:4], producer)  # the process dies after 4 of 6
        resumed = ShardReader(STREAM_TX, kin, start="LATEST", checkpoints=store.checkpoints())
        rest, _ = resumed.read()
        assert [e["event_id"] for e in rest] == [e["event_id"] for e in sent[4:]]

        # Fargate: a new task with an empty store gets the checkpoints back from the decisions stream
        fresh = Store(":memory:")
        q = Processor(fresh, client_for(), kinesis=kin)
        q.tx_epoch = p.tx_epoch
        monkeypatch.setenv("REBUILD_FROM_STREAM", "1")
        assert q.rebuild(max_seconds=20) == 4 and fresh.checkpoints() == store.checkpoints()
        assert q.resume_point(q.tx_epoch, fresh.last_event_ts()) != "TRIM_HORIZON"
        assert fresh.checkpoints() == store.checkpoints()  # same stream: the precise resume point survives


@pytest.mark.parametrize("amount", [500.0, 50_000.0])
def test_a_rejected_token_holds_every_payment_and_never_fails_open(amount):
    calls: list = []
    client = client_for(401, calls=calls)
    first = score_event(event(amount), client)
    second = score_event(event(amount), client)  # breaker open: no call, still held
    assert len(calls) == 1
    for rec in (first, second):
        assert (rec["decision"], rec["decision_source"], rec["action"]) == ("REVIEW", "AUTH_ERROR", "HOLD_FOR_REVIEW")
    m = Metrics()
    m.add([first, second])
    assert m.counts["AuthErrors"] == 2 and m.counts["FailOpen"] == 0


def test_a_recreated_stream_never_inherits_old_checkpoints(monkeypatch):
    """A restarted Kinesis emulator or a replaced stream numbers records from the start again: an old checkpoint
    would silently skip the new stream's first records."""
    from moto import mock_aws
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("KINESIS_CREATE", "1")
    monkeypatch.delenv("KINESIS_ENDPOINT", raising=False)
    with mock_aws():
        import time as _time
        from argus_stream import STREAM_TX
        from argus_stream.bus import KinesisProducer, ShardReader, kinesis_client
        kin = kinesis_client()
        producer = KinesisProducer(kin)
        store = Store(":memory:")
        p = Processor(store, client_for(), kinesis=kin)
        p.tx_epoch = stream_epoch(kin, STREAM_TX)
        p.resume_point(p.tx_epoch, 0)
        for e in [SyntheticSource(seed=4).next() for _ in range(5)]:
            producer.produce(STREAM_TX, key="c", value=e)
        producer.flush()
        got, _ = ShardReader(STREAM_TX, kin, start="TRIM_HORIZON").read()
        p.handle(got, producer)
        assert store.checkpoints()

        _time.sleep(1.1)  # the creation time has one-second resolution
        kin.delete_stream(StreamName=STREAM_TX)
        _time.sleep(0.2)
        KinesisProducer(kin)  # KINESIS_CREATE=1 recreates it, as the restarted service does
        fresh = [SyntheticSource(seed=5).next() for _ in range(3)]
        for e in fresh:
            producer.produce(STREAM_TX, key="c", value=e)
        producer.flush()

        epoch = stream_epoch(kin, STREAM_TX)
        assert epoch != p.tx_epoch
        start = p.resume_point(epoch, store.last_event_ts())
        assert start == "TRIM_HORIZON" and store.checkpoints() == {}
        reader = ShardReader(STREAM_TX, kin, start=start, checkpoints=store.checkpoints())
        new, _ = reader.read()
        assert [e["event_id"] for e in new] == [e["event_id"] for e in fresh]  # none skipped


def test_a_store_without_a_recorded_stream_resumes_by_time():
    store = Store(":memory:")
    store.save_checkpoints({"shardId-000000000000": "20"})  # written before stream identities were recorded
    p = Processor(store, client_for(), kinesis=object())
    assert p.resume_point("2026-11-23T14:47:00+00:00", 1000.0) == 995.0 and store.checkpoints() == {}
    assert store.get_meta("tx_stream_epoch") == "2026-11-23T14:47:00+00:00"
