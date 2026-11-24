"""Every setting the Settings page can change: where it lives (.env), what it affects, how it is checked.

restart  services that read it at start-up and are recreated by "apply" (deploy/hyperstack/apply.sh / argus.ps1 apply)
approval needs two-person approval while the API runs in production mode (it changes what live traffic meets)
secret   written, never shown again (the page only says whether it is set)
locked   shown, not changeable here (with the reason)
Keys the UI does not know stay untouched in .env: the file is the single source of truth, edited by hand or here.
"""
from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, field

from ...core.errors import BadRequest

HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RUN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    label: str
    type: str                         # port | int | float | choice | text | url | host | email | repo | cidrs
    #                                   | secret | token (secret that can be generated) | serving_config | vm_profile | run
    help: str
    default: str | None = None
    restart: tuple[str, ...] = ()
    approval: bool = False
    options: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    locked: str | None = None         # reason it cannot be changed here
    note: str | None = None           # shown next to the field (consequences elsewhere, e.g. AWS)
    extra: dict = field(default_factory=dict)

    @property
    def secret(self) -> bool:
        return self.type in ("secret", "token")


PORT_KEYS = ("ARGUS_UI_PORT", "ARGUS_API_PORT", "ARGUS_MLFLOW_PORT", "ARGUS_SCORING_PORT", "ARGUS_EXPLAINER_PORT",
             "ARGUS_STREAM_PORT")

SETTINGS: list[Setting] = [
    # ------------------------------------------------------------------ serving
    Setting("SCORING_CONFIG", "Serving", "Serving profile", "serving_config",
            "The profile whose api: section the Fraud Model API runs with. A profile with allow_forced_runs: false "
            "(prod_a100) is production mode: demo runs are refused. local_demo serves demo runs (never behind live "
            "traffic).", default="configs/local_demo.yaml", restart=("scoring",), approval=True),
    Setting("RUN_NAME", "Serving", "Served run", "run",
            "The run the Fraud Model API loads at start-up. Change it with Operations -> Serve a run (or activate a "
            "run on the Scoring page for an instant switch).", default="argus_demo_llm_scorer",
            locked="set by Serve a run (serve.sh / argus.ps1 serve), which also restarts what is needed"),
    Setting("EXPLAINER_RUN", "Serving", "Explainer run", "run",
            "The run whose AWQ explainer vLLM serves.", default="argus_demo_llm_scorer",
            locked="set by Serve a run together with the served run"),
    Setting("VLLM_GPU_UTIL", "Serving", "vLLM GPU share", "float",
            "Fraction of GPU memory vLLM may take (the rest is the scorer's and training's).", default="0.5",
            restart=("explainer",), min=0.1, max=0.95),
    Setting("ARGUS_PROFILE", "Serving", "VM profile", "vm_profile",
            "What the GPU VM scripts (setup_vm.sh, pipeline.sh) use when no PROFILE is given: prod for live traffic, "
            "demo for demonstrations, smoke for a health run. Unset: the scripts stop and ask.", default=None),
    # ------------------------------------------------------------------ network
    Setting("ARGUS_UI_PORT", "Network and ports", "UI port", "port",
            "Host port of the UI. Changing it moves the address you open (and the SSH tunnel's).", default="3000",
            restart=("frontend", "platform", "mlflow"), min=1024, max=65535,
            note="the page reloads on the new port after Apply"),
    Setting("ARGUS_API_PORT", "Network and ports", "Platform API port", "port", "Host port of the platform API.",
            default="8090", restart=("platform",), min=1024, max=65535),
    Setting("ARGUS_MLFLOW_PORT", "Network and ports", "MLflow port", "port", "Host port of the MLflow UI.",
            default="5000", restart=("mlflow", "platform"), min=1024, max=65535),
    Setting("ARGUS_SCORING_PORT", "Network and ports", "Fraud Model API port", "port",
            "Host port of the Fraud Model API (open it in the Hyperstack firewall; AWS calls it).", default="8080",
            restart=("scoring",), approval=True, min=1024, max=65535,
            note="AWS calls this port: sync AWS settings and redeploy after changing it"),
    Setting("ARGUS_EXPLAINER_PORT", "Network and ports", "vLLM port", "port", "Host port of the vLLM explainer.",
            default="8000", restart=("explainer",), min=1024, max=65535),
    Setting("ARGUS_STREAM_PORT", "Network and ports", "Stream API port", "port",
            "Host port of the local stream service.", default="8003", restart=("stream",), min=1024, max=65535),
    Setting("ARGUS_BIND", "Network and ports", "UI / platform / MLflow address", "choice",
            "Where the UI, platform, MLflow, vLLM and stream listen. 127.0.0.1: this machine only (reach them through "
            "the SSH tunnel).", default="127.0.0.1", options=("127.0.0.1",), restart=("frontend", "platform", "mlflow"),
            locked="the UI has no login: it is never opened to the network from here (edit .env by hand if you must)"),
    Setting("ARGUS_SCORING_BIND", "Network and ports", "Fraud Model API address", "choice",
            "127.0.0.1: this machine only. 0.0.0.0: reachable from the network (token-protected; the AWS stream "
            "needs it on the GPU VM).", default="127.0.0.1", options=("127.0.0.1", "0.0.0.0"),
            restart=("scoring",), approval=True),
    # ------------------------------------------------------------------ secrets
    Setting("ARGUS_SCORING_TOKEN", "Secrets and keys", "Fraud Model API service token", "token",
            "Bearer token every caller of POST /v1/fraud/score sends (the stream, the platform, AWS).",
            restart=("scoring", "stream", "platform"), approval=True,
            note="the AWS stream uses it too: sync AWS settings and redeploy after a rotation"),
    Setting("ARGUS_SCORING_ADMIN_TOKEN", "Secrets and keys", "Fraud Model API admin token", "token",
            "Bearer token of the admin API (models, thresholds, lists, audit).", restart=("scoring", "platform"),
            approval=True),
    Setting("ARGUS_STREAM_CONTROL_KEY", "Secrets and keys", "Stream control key", "token",
            "X-Control-Key of the stream service (the platform sends it; the browser never sees it). On AWS the key "
            "lives in SSM: paste that one here when the platform uses the AWS stream.", restart=("stream", "platform")),
    Setting("SUPABASE_URL", "Secrets and keys", "Supabase URL", "url", "The project holding the ARGUS dataset.",
            restart=("platform",)),
    Setting("SUPABASE_KEY", "Secrets and keys", "Supabase key", "secret", "Read access to the dataset.",
            restart=("platform",)),
    Setting("HF_TOKEN", "Secrets and keys", "Hugging Face token", "secret",
            "Needed for gated models (prod_a100 pilots Llama 3.2 and Gemma 2).", restart=("platform",)),
    # ------------------------------------------------------------------ stream
    Setting("ARGUS_STREAM_URL", "Stream", "Stream service URL", "url",
            "Local: http://stream:8003. AWS: the load balancer URL (python deploy/ops.py platform-env).",
            default="http://stream:8003", restart=("platform",)),
    Setting("PRODUCER_SOURCE", "Stream", "Local traffic source", "choice",
            "What the local producer sends: synthetic patterns, the labelled holdout, or a mix.", default="mix",
            options=("synthetic", "holdout", "mix"), restart=("stream",)),
    Setting("EVENTS_PER_SEC", "Stream", "Local traffic rate", "float", "Default transactions per second.",
            default="2", restart=("stream",), min=0.1, max=50),
    Setting("ATTACK_EVERY_S", "Stream", "Automatic attack wave every (s)", "int",
            "0 = never; the Live stream page can still inject waves.", default="180", restart=("stream",), min=0, max=3600),
    # ------------------------------------------------------------------ AWS through GitHub
    Setting("ARGUS_GITHUB_REPO", "AWS (through GitHub Actions)", "GitHub repository", "repo",
            "owner/repo whose deploy / destroy workflows the AWS page runs (default: the git remote origin)."),
    Setting("ARGUS_GITHUB_TOKEN", "AWS (through GitHub Actions)", "GitHub token", "secret",
            "Fine-grained token for that repository: Actions, Variables and Secrets read/write. No AWS key is ever "
            "stored here: deploys run in GitHub Actions through the role `python deploy/ops.py up` created."),
    Setting("TF_VAR_scoring_host", "AWS (through GitHub Actions)", "Fraud Model API host for AWS", "host",
            "Public IP / DNS of this GPU VM as AWS sees it (empty: detected when syncing)."),
    Setting("TF_VAR_allowed_cidrs", "AWS (through GitHub Actions)", "Who may reach the AWS stream API", "cidrs",
            'JSON list, e.g. ["203.0.113.7/32", "198.51.100.0/24"]: this VM (the platform proxies the Live stream page) '
            "and your office."),
    Setting("TF_VAR_alert_email", "AWS (through GitHub Actions)", "Alarm e-mail", "email",
            "CloudWatch alarms (fail-open, demo model serving, latency, ...) go here (confirm the SNS e-mail once)."),
    Setting("TF_VAR_transaction_time", "AWS (through GitHub Actions)", "Producer transaction time", "choice",
            "now: transactions carry the current time (needed with prod_a100's +-15 min window); pattern: the dataset's "
            "time of day.", options=("pattern", "now")),
    Setting("TF_VAR_events_per_sec", "AWS (through GitHub Actions)", "AWS traffic rate", "float",
            "Default transactions per second of the AWS producer.", min=0.1, max=50),
    Setting("TF_VAR_certificate_arn", "AWS (through GitHub Actions)", "TLS certificate (ACM ARN)", "text",
            "HTTPS for the AWS stream API (optional; HTTP when empty)."),
]
BY_KEY = {s.key: s for s in SETTINGS}


def _number(s: Setting, value: str) -> str:
    try:
        n = int(value) if s.type in ("port", "int") else float(value)
    except ValueError as e:
        raise BadRequest(f"{s.key}: a {'whole ' if s.type in ('port', 'int') else ''}number") from e
    if (s.min is not None and n < s.min) or (s.max is not None and n > s.max):
        raise BadRequest(f"{s.key}: between {s.min:g} and {s.max:g}")
    return str(n)


def validate(s: Setting, value: str, serving_configs: list[str], vm_profiles: list[str]) -> str:
    """-> the value as it is written to .env. Never lets a value break the .env format for compose / bash."""
    if not isinstance(value, str):
        raise BadRequest(f"{s.key}: text expected")
    value = value.strip()
    if any(c in value for c in "\n\r\x00") or "$" in value or " #" in value or value[:1] in ("'", '"'):
        raise BadRequest(f"{s.key}: no line breaks, '$', ' #' or leading quotes (they break .env parsing)")
    if not value:
        raise BadRequest(f"{s.key}: empty (use 'reset' to remove a setting)")
    t = s.type
    if t in ("port", "int", "float"):
        return _number(s, value)
    if t == "choice":
        if value not in s.options:
            raise BadRequest(f"{s.key}: one of {list(s.options)}")
    elif t == "serving_config":
        if value not in serving_configs:
            raise BadRequest(f"{s.key}: one of {serving_configs}")
    elif t == "vm_profile":
        if value not in vm_profiles:
            raise BadRequest(f"{s.key}: one of {vm_profiles}")
    elif t == "run":
        if not RUN_RE.match(value):
            raise BadRequest(f"{s.key}: a run name")
    elif t == "url":
        if not re.match(r"^https?://[^\s]+$", value):
            raise BadRequest(f"{s.key}: an http(s):// URL")
    elif t == "host":
        try:
            ipaddress.ip_address(value)
        except ValueError:
            if not HOST_RE.match(value):
                raise BadRequest(f"{s.key}: an IP address or host name") from None
    elif t == "email":
        if not EMAIL_RE.match(value):
            raise BadRequest(f"{s.key}: an e-mail address")
    elif t == "repo":
        if not REPO_RE.match(value):
            raise BadRequest(f"{s.key}: owner/repo")
    elif t == "cidrs":
        try:
            items = json.loads(value)
            assert isinstance(items, list) and items and all(isinstance(c, str) for c in items)
            cidrs = [str(ipaddress.ip_network(c, strict=False)) for c in items]
        except (ValueError, AssertionError):
            raise BadRequest(f'{s.key}: a JSON list of CIDRs, e.g. ["203.0.113.7/32"]') from None
        return json.dumps(cidrs)
    elif t in ("secret", "token"):
        if len(value) < 8 or " " in value:
            raise BadRequest(f"{s.key}: at least 8 characters, no spaces")
    elif t == "text":
        if " " in value and s.key.startswith("TF_VAR_"):
            raise BadRequest(f"{s.key}: no spaces")
    return value
