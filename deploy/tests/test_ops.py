"""deploy/ops.py logic that needs no AWS account: Terraform settings carried to CI, the allow-list, and the Fraud Model
API check (against a real local HTTP stub)."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ops  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    for k in [k for k in __import__("os").environ if k.startswith("TF_VAR_") or k in ("TF_VARS", "GITHUB_ACTIONS")]:
        monkeypatch.delenv(k)
    return monkeypatch


def test_ci_gets_every_non_secret_terraform_setting(clean_env):
    for k, v in {"TF_VAR_transaction_time": "now", "TF_VAR_certificate_arn": "arn:acm:x", "TF_VAR_scoring_token": "s3cret",
                 "TF_VAR_region": "ap-south-1", "TF_VAR_events_per_sec": "5"}.items():
        clean_env.setenv(k, v)
    saved = ops.saved_tf_vars()
    assert saved == {"transaction_time": "now", "certificate_arn": "arn:acm:x", "events_per_sec": "5"}  # no secret
    for k in list(saved):
        clean_env.delenv(f"TF_VAR_{k}")
    clean_env.setenv("TF_VARS", json.dumps(saved))
    clean_env.setenv("TF_VAR_events_per_sec", "9")  # an explicit value in CI still wins
    ops.restore_tf_vars()
    env = __import__("os").environ
    assert env["TF_VAR_transaction_time"] == "now" and env["TF_VAR_certificate_arn"] == "arn:acm:x"
    assert env["TF_VAR_events_per_sec"] == "9"


def test_allow_list(clean_env):
    clean_env.setenv("TF_VAR_allowed_cidrs", '["203.0.113.7/32"]')
    assert ops.allowed_cidrs() == ["203.0.113.7/32"]
    clean_env.delenv("TF_VAR_allowed_cidrs")
    clean_env.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(SystemExit):  # CI must never invent an allow-list (it would be the runner's own IP)
        ops.allowed_cidrs()


class Stub(BaseHTTPRequestHandler):
    scorer = "ok"
    mode = "production"   # the API's serving mode (production = demo runs refused)

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok", "scorer": Stub.scorer, "model_version": "argus-xgb-x", "active_run": "x",
                         "serving_mode": Stub.mode, "active_run_forced": False})

    def do_POST(self):
        ok = self.headers.get("Authorization") == "Bearer right"
        self._send(400 if ok else 401, {"error": "INVALID_REQUEST" if ok else "UNAUTHORISED"})

    def log_message(self, *a):
        pass


@pytest.fixture
def api(clean_env):
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    clean_env.setenv("TF_VAR_scoring_port", str(server.server_port))
    yield server
    server.shutdown()


def test_scoring_check_verifies_the_token_and_the_model(api):
    ok, msg = ops.check_scoring("127.0.0.1", "right")
    assert ok and "token accepted" in msg and "argus-xgb-x" in msg
    ok, msg = ops.check_scoring("127.0.0.1", "wrong")
    assert not ok and "rejects" in msg
    Stub.scorer = "down"
    try:
        ok, msg = ops.check_scoring("127.0.0.1", "right")
        assert not ok and "no active model" in msg
    finally:
        Stub.scorer = "ok"


def test_scoring_check_refuses_a_demo_setup_unless_allowed(api, monkeypatch):
    monkeypatch.delenv("ARGUS_ALLOW_DEMO", raising=False)
    Stub.mode = "demo"
    try:
        ok, msg = ops.check_scoring("127.0.0.1", "right")
        assert not ok and "demo mode" in msg and "--allow-demo" in msg
        monkeypatch.setenv("ARGUS_ALLOW_DEMO", "1")
        assert ops.check_scoring("127.0.0.1", "right")[0]
    finally:
        Stub.mode = "production"


def test_saved_tf_vars_never_carry_a_secret_whatever_the_spelling(clean_env):
    """Windows reports environment variables upper-case: the token must still be recognised as the secret."""
    clean_env.setenv("TF_VAR_SCORING_TOKEN", "s3cret")
    clean_env.setenv("TF_VAR_TRANSACTION_TIME", "now")
    saved = ops.saved_tf_vars()
    assert saved == {"transaction_time": "now"} and "s3cret" not in str(saved)


def test_scoring_check_when_nothing_listens(clean_env):
    clean_env.setenv("TF_VAR_scoring_port", "9")
    ok, msg = ops.check_scoring("127.0.0.1", "right")
    assert not ok and "unreachable" in msg


def test_an_empty_workflow_variable_does_not_hide_the_saved_value(clean_env):
    clean_env.setenv("TF_VARS", json.dumps({"certificate_arn": "arn:acm:x"}))
    clean_env.setenv("TF_VAR_certificate_arn", "")  # `TF_VAR_certificate_arn: ${{ vars.CERT }}` with CERT unset
    ops.restore_tf_vars()
    assert __import__("os").environ["TF_VAR_certificate_arn"] == "arn:acm:x"
    clean_env.setenv("TF_VAR_alert_email", "")
    assert "alert_email" not in ops.saved_tf_vars()  # an empty value is never carried to CI


def test_env_file_last_value_wins_and_survives_a_powershell_append(clean_env, tmp_path):
    clean_env.delenv("ARGUS_X", raising=False)
    clean_env.delenv("ARGUS_Y", raising=False)
    env = tmp_path / ".env"
    env.write_bytes(b"ARGUS_X=old\nARGUS_Y=keep\n" + "ARGUS_X=new\n".encode("utf-16"))  # Windows PowerShell >>
    ops.load_env(env)
    import os
    assert os.environ["ARGUS_X"] == "new" and os.environ["ARGUS_Y"] == "keep"
    clean_env.setenv("ARGUS_X", "from-shell")
    ops.load_env(env)
    assert os.environ["ARGUS_X"] == "from-shell"  # the real environment still wins
    clean_env.delenv("ARGUS_X")
    clean_env.delenv("ARGUS_Y")
