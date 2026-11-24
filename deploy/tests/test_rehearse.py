"""The local AWS rehearsal (deploy/local): the parts that need no Docker, and the static IAM check that keeps the
stream's AWS calls, the rehearsal's list of them and the task role in deploy/terraform/ecs.tf in step on every CI run."""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1]
ROOT = DEPLOY.parent
spec = importlib.util.spec_from_file_location("rehearse", DEPLOY / "local" / "rehearse.py")
rehearse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearse)

# boto3 method -> IAM action. None = only reached against the local emulator (on AWS, Terraform creates the streams).
BOTO_TO_IAM = {
    "put_records": "kinesis:PutRecords", "get_records": "kinesis:GetRecords",
    "get_shard_iterator": "kinesis:GetShardIterator", "list_shards": "kinesis:ListShards",
    "describe_stream_summary": "kinesis:DescribeStreamSummary", "create_stream": None,
    "get_paginator": "s3:ListBucket", "download_file": "s3:GetObject",
    "publish": "sns:Publish", "put_metric_data": "cloudwatch:PutMetricData",
}


def calls_made() -> list[str]:
    """CALLS_MADE from inside.py, read without importing it (it needs boto3, which the deploy tests do not)."""
    tree = ast.parse((DEPLOY / "local" / "inside.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CALLS_MADE")
    return ast.literal_eval(node.value)


def stream_boto_methods() -> set[str]:
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "stream" / "argus_stream").glob("*.py"))
    return {m for m in BOTO_TO_IAM if re.search(rf"\.{m}\(", src)}


# ------------------------------------------------------------------ IAM: code, rehearsal and task role agree
def test_every_aws_call_of_the_stream_is_in_the_rehearsal_list():
    used = stream_boto_methods()
    assert {"put_records", "get_records", "publish", "put_metric_data"} <= used     # the scan finds the real calls
    missing = sorted(BOTO_TO_IAM[m] for m in used if BOTO_TO_IAM[m] and BOTO_TO_IAM[m] not in calls_made())
    assert not missing, f"add to CALLS_MADE in deploy/local/inside.py (and the task role): {missing}"


def test_the_task_role_allows_every_call_the_stream_makes():
    ecs = (DEPLOY / "terraform" / "ecs.tf").read_text(encoding="utf-8")
    missing = [a for a in calls_made() if f'"{a}"' not in ecs]
    assert not missing, f"deploy/terraform/ecs.tf task role lacks {missing}"


# ------------------------------------------------------------------ the rehearsal's own logic
def plan(*resources):
    return {"resource_changes": [{"address": a, "mode": mode, "change": {"actions": acts}} for a, mode, acts in resources]}


def test_split_plan_applies_everything_the_emulator_can_create_and_lists_the_rest():
    targets, skipped = rehearse.split_plan(plan(
        ("aws_kinesis_stream.this[\"transactions\"]", "managed", ["create"]),
        ("aws_cloudwatch_metric_alarm.this[\"lag\"]", "managed", ["create"]),
        ("aws_cloudwatch_dashboard.this", "managed", ["create"]),
        ("aws_iam_role.task", "managed", ["create"]),
        ("data.aws_caller_identity.me", "data", ["read"]),
        ("aws_s3_bucket.old", "managed", ["delete"]),
    ))
    assert targets == ['aws_kinesis_stream.this["transactions"]', "aws_iam_role.task"]
    assert skipped == ['aws_cloudwatch_metric_alarm.this["lag"]', "aws_cloudwatch_dashboard.this"]


def test_scoring_token_from_env_file_else_compose_default(tmp_path, monkeypatch):
    monkeypatch.setattr(rehearse, "ROOT", tmp_path)
    assert rehearse.scoring_token() == "demo-service-token"
    (tmp_path / ".env").write_text("X=1\nARGUS_SCORING_TOKEN='s3cret'\n", encoding="utf-8")
    assert rehearse.scoring_token() == "s3cret"


def test_the_rehearsal_targets_the_compose_network_of_this_checkout():
    assert rehearse.NETWORK.endswith("_default") and rehearse.NETWORK == rehearse.NETWORK.lower()


# Terraform resource-type prefix -> the provider endpoint it calls. A new kind of resource fails this test until it is
# mapped here and given an endpoint in moto_override.tf (otherwise the rehearsal would call real AWS for it).
TYPE_TO_ENDPOINT = {
    "aws_cloudwatch_log": "logs", "aws_cloudwatch": "cloudwatch", "aws_ecr": "ecr", "aws_ecs": "ecs", "aws_iam": "iam",
    "aws_kinesis_firehose": "firehose", "aws_kinesis": "kinesis", "aws_lb": "elbv2", "aws_s3": "s3", "aws_sns": "sns",
    "aws_ssm": "ssm", "aws_sqs": "sqs", "aws_kms": "kms", "aws_caller_identity": "sts",
    "aws_internet_gateway": "ec2", "aws_route_table": "ec2", "aws_security_group": "ec2", "aws_subnet": "ec2",
    "aws_vpc": "ec2", "aws_availability_zones": "ec2", "aws_iam_policy_document": None,  # computed locally
}


def test_the_override_sends_every_resource_to_the_emulator_and_keeps_state_out_of_the_repo():
    tf = (DEPLOY / "local" / "moto_override.tf").read_text(encoding="utf-8")
    assert 'path = "/work/state/terraform.tfstate"' in tf
    endpoints = set(re.findall(r"^\s+([a-z0-9]+)\s+= \"http://aws:5000\"", tf, re.M))
    types = set(re.findall(r'^(?:resource|data) "(aws_[a-z0-9_]+)"', "\n".join(
        p.read_text(encoding="utf-8") for p in (DEPLOY / "terraform").glob("*.tf")), re.M))
    for t in sorted(types):
        prefix = max((p for p in TYPE_TO_ENDPOINT if t.startswith(p)), key=len, default=None)
        assert prefix, f"{t}: map it in TYPE_TO_ENDPOINT and add its endpoint to deploy/local/moto_override.tf"
        assert TYPE_TO_ENDPOINT[prefix] in endpoints | {None}, f"{t}: no emulator endpoint {TYPE_TO_ENDPOINT[prefix]}"


@pytest.mark.parametrize("value,ok", [("up", True), ("verify", True), ("down", True), ("rm -rf", False), ("", False)])
def test_operations_action_accepts_only_the_rehearsal_commands(value, ok):
    import sys
    sys.path.insert(0, str(DEPLOY))
    import ops_catalog
    actions = ops_catalog.load(ROOT)
    if ok:
        assert ops_catalog.validate(actions, "aws_local", {"command": value}, ROOT) == {"command": value}
    else:
        with pytest.raises(ops_catalog.ActionError):
            ops_catalog.validate(actions, "aws_local", {"command": value}, ROOT)
