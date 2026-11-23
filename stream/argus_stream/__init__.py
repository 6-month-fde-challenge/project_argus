"""ARGUS real-time stream: Amazon Kinesis -> Fraud Model API -> decision -> action, with live analytics.

    bus        Kinesis Data Streams producer (PutRecords, retries) and shard reader (GetRecords, lag)
    generator  CRD 10.2 transactions: synthetic traffic with fraud patterns, or a replay of the labelled holdout split
    producer   traffic switched on / off from the UI (POST /live/api/producer), plus attack waves
    processor  consumes the transactions stream, scores each one on the Fraud Model API, applies the fail-open rule
               when the API is down (FR-API-01), maps the decision to an action, publishes it on the decisions stream
    store      SQLite analytics over the scored stream (rebuilt from the decisions stream after a restart)
    api        FastAPI under /live: health, state, analytics, traffic control (the platform proxies it for the UI)
"""
from __future__ import annotations

import os

# Kinesis stream names (Terraform creates argus-<env>-transactions / argus-<env>-decisions)
STREAM_TX = os.getenv("STREAM_TRANSACTIONS", "argus-transactions")
STREAM_DECISIONS = os.getenv("STREAM_DECISIONS", "argus-decisions")

FAIL_OPEN_LIMIT = 10_000.0  # CRD FR-API-01: on 503 approve below Rs 10,000, otherwise REVIEW

# What happens after each decision (CRD 10.4): the action rides on every record of the decisions stream; BLOCK
# actions are also published to SNS for downstream automation.
ACTIONS = {
    "APPROVE": "Approved, no further action.",
    "APPROVE_STEP_UP": "Approved with step-up OTP where the merchant supports it.",
    "HOLD_FOR_REVIEW": "Held: case queued for a fraud analyst.",
    "BLOCK_AND_ALERT": "Blocked: payment declined, customer and fraud team alerted.",
}


def action_for(decision: str, risk_level: str | None = None) -> tuple[str, str]:
    """(action, text) for a decision. ALLOW at MEDIUM risk steps up; anything unknown is held for review."""
    if decision == "ALLOW":
        action = "APPROVE_STEP_UP" if risk_level == "MEDIUM" else "APPROVE"
    else:
        action = {"BLOCK": "BLOCK_AND_ALERT"}.get(decision, "HOLD_FOR_REVIEW")
    return action, ACTIONS[action]
