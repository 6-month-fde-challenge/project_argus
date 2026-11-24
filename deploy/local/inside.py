"""The rehearsal's view from inside AWS: runs in the stream image (boto3 + httpx), against the emulator.

    python inside.py describe            -> JSON: what the Fargate task would get (env + resolved secrets) and names
    python inside.py seed /data          -> the labelled holdout into the data bucket (as `ops.py dataset` does)
    python inside.py subscribe           -> an SQS queue on the actions topic, to catch BLOCK alerts
    python inside.py block <rules.json>  -> one payment from a blocklisted device into the transactions stream
    python inside.py verify              -> PASS / FAIL for every check (exit 1 on any FAIL)
AWS_ENDPOINT_URL points every boto3 client at the emulator; /work holds Terraform's plan (plan.json) and state.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
import httpx

REGION = os.environ.get("AWS_REGION", "ap-south-1")
WORK = Path("/work")
STREAM_URL = os.environ.get("REHEARSAL_STREAM_URL", "http://argus-aws-stream:8003")
QUEUE = "argus-rehearsal-actions"
# what the stream code calls on AWS (stream/argus_stream): each must be allowed by the task role (deploy/terraform)
CALLS_MADE = ["kinesis:PutRecords", "kinesis:GetRecords", "kinesis:GetShardIterator", "kinesis:ListShards",
              "kinesis:DescribeStreamSummary", "s3:ListBucket", "s3:GetObject", "sns:Publish", "cloudwatch:PutMetricData"]


def client(name: str):
    return boto3.client(name, region_name=REGION)


def state() -> dict:
    return json.loads((WORK / "state" / "terraform.tfstate").read_text())


def resources(kind: str) -> list[dict]:
    return [inst["attributes"] for r in state()["resources"] if r["type"] == kind and r["mode"] == "managed"
            for inst in r["instances"]]


def describe() -> dict:
    """The task definition Terraform registered: its environment, and its secrets resolved from SSM (as ECS does)."""
    td = resources("aws_ecs_task_definition")[0]
    container = json.loads(td["container_definitions"])[0]
    env = {e["name"]: e["value"] for e in container.get("environment", [])}
    ssm = client("ssm")
    for s in container.get("secrets", []):
        name = s["valueFrom"].split(":parameter")[-1]
        env[s["name"]] = ssm.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]
    topic = next(t["arn"] for t in resources("aws_sns_topic") if t["name"].endswith("-fraud-actions"))
    return {"env": env, "topic": topic, "buckets": [b["bucket"] for b in resources("aws_s3_bucket")],
            "image": container["image"], "task_role": td["task_role_arn"]}


def seed(folder: str) -> None:
    env = describe()["env"]
    bucket, prefix = env["DATASET_URI"][len("s3://"):].split("/", 1)
    files = sorted(Path(folder).glob("*.parquet"))
    s3 = client("s3")
    for f in files:
        s3.upload_file(str(f), bucket, f"{prefix}{f.name}")
    print(f"seeded {len(files)} holdout files into s3://{bucket}/{prefix}")


def subscribe() -> None:
    sqs, sns = client("sqs"), client("sns")
    url = sqs.create_queue(QueueName=QUEUE)["QueueUrl"]
    arn = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    sns.subscribe(TopicArn=describe()["topic"], Protocol="sqs", Endpoint=arn)
    print("subscribed", QUEUE, "to the actions topic")


def block(rules_path: str) -> None:
    """A payment from a blocklisted device: the hard rule BLOCKs it, so the SNS alert path runs."""
    device = json.loads(Path(rules_path).read_text())["blocked_devices"][0]
    env = describe()["env"]
    tx = {"event_id": f"rehearsal-{uuid.uuid4().hex[:12]}", "ts": time.time(), "source": "rehearsal", "fraud_label": 1,
          "request": {"transaction_id": f"TXREH{uuid.uuid4().hex[:8]}", "customer_id": "C_rehearsal",
                      "transaction_amount": 25000.0, "merchant_category": "electronics",
                      "transaction_time": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"), "device_id": device,
                      "customer_age": 30, "customer_location": {"lat": 18.52, "lon": 73.85, "city": "Pune"},
                      "merchant_location": {"lat": 28.61, "lon": 77.2, "city": "Delhi"}, "account_age": 12}}
    client("kinesis").put_record(StreamName=env["STREAM_TRANSACTIONS"], Data=json.dumps(tx).encode(), PartitionKey="C_rehearsal")
    print("sent one payment from blocklisted device", device)


# ------------------------------------------------------------------ verify
class Report:
    def __init__(self):
        self.rows: list[tuple[str, bool | None, str]] = []

    def check(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.rows.append((name, ok, detail))

    def print(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "n/a " if ok is None else "FAIL"
            print(f"  {mark}  {name.ljust(width)}  {detail}")
        failed = [n for n, ok, _ in self.rows if ok is False]
        na = sum(ok is None for _, ok, _ in self.rows)
        print(f"\n{len(self.rows) - len(failed) - na} of {len(self.rows)} checks passed"
              + (f", {na} not applicable" if na else "") + (f"; FAILED: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0


def verify() -> int:
    r = Report()
    d = describe()
    env = d["env"]
    plan = json.loads((WORK / "plan.json").read_text())
    planned = [x for x in plan["resource_changes"] if x["mode"] == "managed"]
    skipped = json.loads((WORK / "skipped.json").read_text())
    applied = {f"{x['type']}.{x['name']}" for x in state()["resources"] if x["mode"] == "managed"}
    missing = sorted({f"{x['type']}.{x['name']}" for x in planned} - applied - {s.split("[")[0] for s in skipped})
    r.check("Terraform applied the whole stack", not missing,
            f"{sum(len(x['instances']) for x in state()['resources'] if x['mode'] == 'managed')} resources applied; not creatable on the emulator: {', '.join(sorted({s.split('[')[0] for s in skipped}))}"
            + (f"; MISSING {missing}" if missing else ""))

    kin = client("kinesis")
    for key in ("STREAM_TRANSACTIONS", "STREAM_DECISIONS"):
        s = kin.describe_stream_summary(StreamName=env[key])["StreamDescriptionSummary"]
        r.check(f"Kinesis {key.split('_')[1].lower()} stream", s["StreamStatus"] == "ACTIVE", env[key])

    fh = client("firehose")
    name = fh.list_delivery_streams()["DeliveryStreamNames"][0]
    desc = fh.describe_delivery_stream(DeliveryStreamName=name)["DeliveryStreamDescription"]
    src = desc.get("Source", {}).get("KinesisStreamSourceDescription", {}).get("KinesisStreamARN", "")
    dest = desc["Destinations"][0].get("ExtendedS3DestinationDescription", {}).get("BucketARN", "")
    r.check("Firehose: decisions stream -> S3 audit bucket", src.endswith(f"/{env['STREAM_DECISIONS']}") and "audit" in dest,
            f"{src.rsplit('/', 1)[-1]} -> {dest.rsplit(':', 1)[-1]} (delivery itself: not emulated)")

    r.check("Secrets from SSM (scoring token, control key)", bool(env.get("ARGUS_SCORING_TOKEN")) and len(env.get("CONTROL_API_KEY", "")) >= 32,
            "resolved the way ECS injects them")

    http = httpx.Client(base_url=STREAM_URL, timeout=10, headers={"X-Control-Key": env["CONTROL_API_KEY"]})
    st = http.get("/live/api/state").json()
    r.check("Stream task (from the task definition) consumes Kinesis", st["stream"] == "connected" and st["consumed"] > 0,
            f"{st['consumed']} transactions, {st['errors']} errors")
    r.check("Stream errors / dead letters", st["errors"] == 0 and st.get("dead_letters", 0) == 0,
            f"{st['errors']} errors, {st.get('dead_letters', 0)} dead letters")
    sc = st.get("scoring") or {}
    r.check("Fraud Model API reached from the task", sc.get("status") == "ok" if sc.get("status") else None,
            f"{env.get('SCORING_URL')} -> {sc.get('status')} ({sc.get('active_run')})")

    summ = http.get("/live/api/summary", params={"minutes": 60}).json()
    sources = {g["key"] for g in summ.get("by_source", [])}
    r.check("Holdout replay from the S3 data bucket", "holdout" in sources if env.get("PRODUCER_SOURCE") in ("holdout", "mix") else None,
            f"sources seen: {sorted(s for s in sources if s)}")

    it = kin.get_shard_iterator(StreamName=env["STREAM_DECISIONS"], ShardId=kin.list_shards(StreamName=env["STREAM_DECISIONS"])["Shards"][0]["ShardId"],
                                ShardIteratorType="TRIM_HORIZON")["ShardIterator"]
    n = 0
    for _ in range(20):
        resp = kin.get_records(ShardIterator=it, Limit=1000)
        n += len(resp["Records"])
        it = resp.get("NextShardIterator")
        if not resp["Records"] or not it:
            break
    r.check("Decisions published to the decisions stream", n >= st["consumed"] > 0, f"{n} records")

    sqs = client("sqs")
    url = sqs.get_queue_url(QueueName=QUEUE)["QueueUrl"]
    msgs = []
    for _ in range(5):
        # VisibilityTimeout=0: only look, so `verify` can run again and still see the alert
        msgs += sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, WaitTimeSeconds=2,
                                    VisibilityTimeout=0).get("Messages", [])
        if msgs:
            break
    blocks = [json.loads(json.loads(m["Body"])["Message"]) for m in msgs]
    r.check("BLOCK alert published to SNS", any(b.get("action") == "BLOCK_AND_ALERT" for b in blocks),
            f"{len(blocks)} message(s) on the actions topic")

    cw = client("cloudwatch")
    ns = env["CW_NAMESPACE"]
    published = {m["MetricName"] for m in cw.list_metrics(Namespace=ns).get("Metrics", [])}
    r.check("Stream metrics in CloudWatch", {"Transactions", "DemoModelServing", "FailOpen"} <= published,
            f"{ns}: {', '.join(sorted(published))}")
    alarms = [x["change"]["after"] for x in planned if x["type"] == "aws_cloudwatch_metric_alarm"]
    ours = [a for a in alarms if a.get("namespace") == ns]
    unknown = sorted({a["metric_name"] for a in ours} - published - {"ApiLatency"} if published else set())
    r.check("Every stream alarm watches a metric the stream publishes", bool(ours) and not unknown,
            f"{len(ours)} alarms on {ns}" + (f"; never published: {unknown}" if unknown else ""))
    demo = cw.get_metric_statistics(Namespace=ns, MetricName="DemoModelServing", StartTime=time.time() - 3600,
                                    EndTime=time.time() + 60, Period=60, Statistics=["Maximum"],
                                    Dimensions=[{"Name": "Service", "Value": "stream"}]).get("Datapoints", [])
    r.check("DemoModelServing reflects what the API serves", bool(demo),
            f"max {max(p['Maximum'] for p in demo):.0f} (1 = demo setup behind live traffic)" if demo else "no datapoints")

    iam = client("iam")
    role = d["task_role"].rsplit("/", 1)[-1]
    allowed: set[str] = set()
    for pol in iam.list_role_policies(RoleName=role)["PolicyNames"]:
        doc = iam.get_role_policy(RoleName=role, PolicyName=pol)["PolicyDocument"]
        doc = json.loads(doc) if isinstance(doc, str) else doc
        for stmt in doc["Statement"]:
            if stmt.get("Effect") == "Allow":
                acts = stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
                allowed |= set(acts)
    denied = [c for c in CALLS_MADE if c not in allowed and f"{c.split(':')[0]}:*" not in allowed]
    r.check("Task role allows every AWS call the stream makes", not denied,
            "kinesis, s3, sns, cloudwatch" + (f"; MISSING {denied}" if denied else ""))
    return r.print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "describe":
        print(json.dumps(describe()))
    elif cmd == "seed":
        seed(sys.argv[2])
    elif cmd == "subscribe":
        subscribe()
    elif cmd == "block":
        block(sys.argv[2])
    else:
        sys.exit(verify())
