#!/usr/bin/env python3
"""ARGUS on AWS, one command at a time (Windows, Linux, macOS, GitHub Actions). Needs: aws, terraform, docker (+ gh).

    python deploy/ops.py check              read-only: tools, AWS identity, the Fraud Model API + token, isolation
    python deploy/ops.py up                 everything: check -> bootstrap (state bucket, CI role, GitHub settings)
                                            -> deploy -> print the URLs
    python deploy/ops.py deploy [--tag T]   build + push the stream image, copy the holdout, terraform apply, wait
                                            (what CI runs on every push to main)
    python deploy/ops.py dataset            (re)copy the labelled holdout split to S3 for the producer
    python deploy/ops.py status             URLs, ECS service, health
    python deploy/ops.py platform-env       the 2 lines the ARGUS platform needs for the Live stream page
    python deploy/ops.py down [--all] [-y]  terraform destroy (only what this project created) [+ the bootstrap stack]
    python deploy/ops.py audit              any AWS resource still tagged Project=<project>

Settings come from the environment or the git-ignored .env next to docker-compose.yml:
  TF_VAR_scoring_host / TF_VAR_scoring_token   the GPU VM's IP and ARGUS_SCORING_TOKEN (deploy/hyperstack/pipeline.sh
                                               writes both to training/outputs/_deploy/aws.env on the VM)
  AWS_REGION (ap-south-1), PROJECT (argus), ENVIRONMENT (prod), GITHUB_REPO (owner/repo, default: the git remote)
  TF_VAR_allowed_cidrs   who may reach the stream API (default: the GPU VM + this machine's public IP)
Any Terraform variable can be set as TF_VAR_<name>, e.g. TF_VAR_alert_email=me@example.com.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # argus_solution/
INFRA, BOOT = ROOT / "deploy" / "terraform", ROOT / "deploy" / "terraform" / "bootstrap"
HOLDOUT = ROOT / "training" / "data" / "raw" / "parquet" / "transactions" / "holdout"
IMAGE = ("stream", ROOT / "stream")                   # ECR repo <name>/stream <- build context


def log(msg: str) -> None:
    print(f"\n\033[1;34m==> {msg}\033[0m", flush=True)


def run(cmd, cwd=None, capture=False, check=True, input_=None, quiet=False) -> str:
    if not quiet:
        print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, text=True, input=input_,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        if capture:
            sys.stderr.write((r.stderr or "") + (r.stdout or ""))
        sys.exit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd[:4])} ...")
    return (r.stdout or "").strip() if capture else ""


def load_env(path: Path) -> None:
    """KEY=VALUE lines of .env, without overriding variables already set in the environment. A key written twice
    takes its LAST value, like docker compose (a line appended later must win over the original one)."""
    if not path.exists():
        return
    values: dict[str, str] = {}
    raw = path.read_bytes()
    # Windows PowerShell 5.1 `>>` appends UTF-16: decode what is UTF-16, drop the NULs a mixed file would leave
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    text = text.replace("\x00", "").replace("\ufeff", "").replace("\ufffd", "")
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2).strip('"').strip("'")
    for key, value in values.items():
        if not os.getenv(key):
            os.environ[key] = value


def http(url: str, method: str = "GET", headers: dict | None = None, body: bytes | None = None, timeout: float = 6):
    """-> (status, text); status 0 when the host cannot be reached."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)


class Ctx:
    def __init__(self, a):
        self.region, self.project, self.env = a.region, a.project, a.environment
        self.name = f"{self.project}-{self.env}"
        self.repo = a.github_repo
        os.environ["AWS_REGION"] = os.environ["AWS_DEFAULT_REGION"] = self.region
        os.environ["TF_VAR_region"], os.environ["TF_VAR_project"], os.environ["TF_VAR_environment"] = \
            self.region, self.project, self.env
        os.environ.setdefault("TF_IN_AUTOMATION", "1")
        self._account = None

    @property
    def account(self) -> str:
        if not self._account:
            self._account = run(["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
                                capture=True, quiet=True)
        return self._account

    @property
    def registry(self) -> str:
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com"

    @property
    def state_bucket(self) -> str:
        return os.getenv("TF_STATE_BUCKET") or f"{self.project}-tfstate-{self.account}-{self.region}"

    @property
    def data_bucket(self) -> str:
        return f"{self.name}-data-{self.account}"


# ------------------------------------------------------------------ settings
def scoring_settings() -> tuple[str, str]:
    return os.getenv("TF_VAR_scoring_host", ""), os.getenv("TF_VAR_scoring_token", "")


def allowed_cidrs() -> list[str]:
    """Who may reach the stream API: TF_VAR_allowed_cidrs, else the GPU VM (its platform proxies the Live stream page)
    plus this machine's public IP. In CI there is no 'this machine': the value saved by `up` is used."""
    if os.getenv("TF_VAR_allowed_cidrs"):
        return json.loads(os.environ["TF_VAR_allowed_cidrs"])
    if os.getenv("GITHUB_ACTIONS") == "true":
        sys.exit("TF_VAR_allowed_cidrs is not set: run `python deploy/ops.py up` locally once (it saves ALLOWED_CIDRS)")
    cidrs = []
    host, _ = scoring_settings()
    if host:
        try:
            cidrs.append(f"{ipaddress.ip_address(socket.gethostbyname(host))}/32")
        except OSError:
            print(f"  note: cannot resolve {host}; the GPU VM is not added to allowed_cidrs")
    status, me = http("https://checkip.amazonaws.com", timeout=5)
    if status == 200:
        cidrs.append(f"{ipaddress.ip_address(me.strip())}/32")
    return sorted(set(cidrs)) or ["0.0.0.0/0"]


SECRET_TF_VARS = {"scoring_token"}                     # GitHub secret SCORING_TOKEN, never a variable
STACK_TF_VARS = {"region", "project", "environment"}   # set by Ctx from AWS_REGION / PROJECT / ENVIRONMENT


def saved_tf_vars() -> dict[str, str]:
    """Every non-secret TF_VAR_* of this machine, so CI deploys with exactly the same settings (transaction_time,
    certificate_arn, events_per_sec, ...) instead of silently falling back to the defaults."""
    return {k[len("TF_VAR_"):]: v for k, v in os.environ.items()
            if k.startswith("TF_VAR_") and v and k[len("TF_VAR_"):] not in SECRET_TF_VARS | STACK_TF_VARS}


def restore_tf_vars() -> None:
    """CI: the TF_VARS repository variable saved by `up` -> TF_VAR_* (an explicit, non-empty TF_VAR_* still wins; a
    workflow line like `TF_VAR_x: ${{ vars.X }}` with X unset gives an empty string, which counts as unset)."""
    for name, value in json.loads(os.getenv("TF_VARS") or "{}").items():
        if not os.getenv(f"TF_VAR_{name}"):
            os.environ[f"TF_VAR_{name}"] = value


def preflight(need_docker=True) -> None:
    missing = [t for t in ["aws", "terraform"] + (["docker"] if need_docker else []) if not shutil.which(t)]
    if missing:
        sys.exit(f"missing tools on PATH: {', '.join(missing)}")
    if need_docker and subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        sys.exit("docker is not running")


def check_scoring(host: str, token: str) -> tuple[bool, str]:
    """The Fraud Model API answers, has a model, and accepts the token (an empty body must give 400, not 401)."""
    base = f"{os.getenv('TF_VAR_scoring_scheme', 'http')}://{host}:{os.getenv('TF_VAR_scoring_port', '8080')}"
    status, body = http(f"{base}/v1/health")
    if status != 200:
        return False, f"{base}/v1/health unreachable ({body[:80]}): serve.sh on the VM, firewall open?"
    health = json.loads(body)
    status, _ = http(f"{base}/v1/fraud/score", "POST",
                     {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, b"{}")
    if status == 401:
        return False, "the Fraud Model API rejects TF_VAR_scoring_token (copy ARGUS_SCORING_TOKEN from the VM's .env)"
    if health.get("scorer") != "ok":
        return False, f"the Fraud Model API has no active model ({health.get('status')}): serve.sh <run> on the VM"
    return True, f"Fraud Model API on {host}: {health.get('model_version')} ({health.get('active_run')}), token accepted"


def check(c: Ctx) -> bool:
    """Every setting, and that this project cannot collide with anything in the account / region. Creates nothing."""
    bad, warn = [], []
    good = lambda m: print(f"  \033[32mOK\033[0m   {m}")  # noqa: E731

    log(f"Checking {c.name} in {c.region}")
    for t in ["aws", "terraform", "docker"]:
        good(f"{t} found") if shutil.which(t) else bad.append(f"{t} is not installed / not on PATH")
    if shutil.which("terraform"):
        v = json.loads(run(["terraform", "version", "-json"], capture=True, quiet=True))["terraform_version"]
        if tuple(int(x) for x in re.findall(r"\d+", v)[:2]) < (1, 10):
            bad.append(f"terraform {v} is too old: 1.10 or newer (S3 state locking)")
    if shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        bad.append("Docker Desktop is not running")
    if shutil.which("gh") and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0:
        good("GitHub CLI logged in: CI/CD variables and secrets will be set")
    else:
        warn.append("GitHub CLI not logged in (`gh auth login`): AWS deploys, but CI/CD is not wired to GitHub")

    signed_in = False
    if shutil.which("aws"):
        ident = subprocess.run(["aws", "sts", "get-caller-identity", "--output", "json"], capture_output=True, text=True)
        signed_in = ident.returncode == 0
        if signed_in:
            good(f"AWS identity {json.loads(ident.stdout)['Arn']}  region {c.region}")
        else:
            bad.append("AWS CLI has no valid credentials (`aws configure` or `aws login`)")

    host, token = scoring_settings()
    if not host or not token:
        warn.append("TF_VAR_scoring_host / TF_VAR_scoring_token missing: the stream deploys, but every transaction "
                    "is decided by the fail-open rule until the Fraud Model API is reachable")
    else:
        ok, msg = check_scoring(host, token)
        good(msg) if ok else warn.append(msg)
    if os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("TF_VAR_allowed_cidrs"):
        cidrs = allowed_cidrs()
        (warn.append if "0.0.0.0/0" in cidrs else good)(f"stream API reachable from {', '.join(cidrs)}")

    if signed_in:  # isolation (read-only)
        aws = lambda *a: subprocess.run(["aws", *a, "--region", c.region, "--output", "text"],  # noqa: E731
                                        capture_output=True, text=True)
        deployed = aws("s3api", "head-bucket", "--bucket", c.state_bucket).returncode == 0
        vpcs = len(aws("ec2", "describe-vpcs", "--query", "Vpcs[].VpcId").stdout.split())
        q = aws("service-quotas", "get-service-quota", "--service-code", "vpc", "--quota-code", "L-F678F1CE",
                "--query", "Quota.Value").stdout.strip()
        quota = int(float(q)) if q and q != "None" else 5
        if deployed:
            good(f"existing {c.name} state found: this updates the same stack")
        elif vpcs >= quota:
            bad.append(f"{vpcs}/{quota} VPCs used in {c.region}: raise the quota (this project needs its own VPC)")
        else:
            good(f"VPC quota: {vpcs}/{quota} used; this project adds its own VPC")
        if not deployed:
            clashes = []
            if c.name in [a.rsplit("/", 1)[-1] for a in aws("ecs", "list-clusters", "--query", "clusterArns").stdout.split()]:
                clashes.append(f"ECS cluster {c.name}")
            clashes += [f"ECR {r}" for r in aws("ecr", "describe-repositories", "--query",
                                                "repositories[].repositoryName").stdout.split() if r.startswith(c.name + "/")]
            clashes += [f"ALB {n}" for n in aws("elbv2", "describe-load-balancers", "--query",
                                                "LoadBalancers[].LoadBalancerName").stdout.split() if n.startswith(c.name)]
            clashes += [f"IAM role {n}" for n in aws("iam", "list-roles", "--query",
                                                     f"Roles[?starts_with(RoleName,'{c.name}')].RoleName").stdout.split()]
            clashes += [f"Kinesis stream {n}" for n in aws("kinesis", "list-streams", "--query", "StreamNames").stdout.split()
                        if n.startswith(c.name + "-")]
            clashes += [f"Firehose {n}" for n in aws("firehose", "list-delivery-streams", "--query",
                                                     "DeliveryStreamNames").stdout.split() if n.startswith(c.name + "-")]
            clashes += [f"log group {n}" for n in aws("logs", "describe-log-groups", "--log-group-name-prefix",
                                                      f"/ecs/{c.name}", "--query", "logGroups[].logGroupName").stdout.split()]
            if clashes:
                bad.append("names this project would create already exist (not managed by it): " + ", ".join(clashes))
            else:
                good(f"no existing resource uses the '{c.name}' names: nothing else in {c.region} is touched")

    for w in warn:
        print(f"  \033[33mWARN\033[0m {w}")
    for b in bad:
        print(f"  \033[31mFAIL\033[0m {b}")
    print("\n  " + ("\033[32mready to deploy\033[0m" if not bad else "\033[31mfix the FAIL lines, then re-run\033[0m"))
    return not bad


# ------------------------------------------------------------------ terraform helpers
def tf_init(c: Ctx) -> None:
    run(["terraform", f"-chdir={INFRA}", "init", "-input=false", "-reconfigure",
         f"-backend-config=bucket={c.state_bucket}", f"-backend-config=key={c.project}/{c.env}/terraform.tfstate",
         f"-backend-config=region={c.region}", "-backend-config=encrypt=true", "-backend-config=use_lockfile=true"])


def tf_outputs() -> dict:
    out = run(["terraform", f"-chdir={INFRA}", "output", "-json"], capture=True, check=False, quiet=True)
    try:
        return {k: v["value"] for k, v in json.loads(out or "{}").items()}
    except ValueError:
        return {}


def git_tag() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if sha.returncode == 0 and sha.stdout.strip():
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha.stdout.strip() + ("-dirty" + time.strftime("%H%M%S") if dirty else "")
    return time.strftime("build-%Y%m%d-%H%M%S")


def detect_repo() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"github\.com[:/]([^/]+/[^/.\s]+?)(?:\.git)?\s*$", url)
    return m.group(1) if m else ""


def owns_oidc() -> bool:
    """True if the GitHub OIDC provider is in the bootstrap state (created by this project)."""
    r = subprocess.run(["terraform", f"-chdir={BOOT}", "state", "list"], capture_output=True, text=True)
    return "aws_iam_openid_connect_provider.github" in (r.stdout or "")


# ------------------------------------------------------------------ commands
def bootstrap(c: Ctx) -> dict:
    log("Bootstrap: Terraform state bucket + GitHub Actions OIDC deploy role")
    providers = run(["aws", "iam", "list-open-id-connect-providers", "--output", "json"], capture=True, quiet=True)
    create = owns_oidc() or "token.actions.githubusercontent.com" not in providers  # never take over someone else's
    run(["terraform", f"-chdir={BOOT}", "init", "-input=false"])
    run(["terraform", f"-chdir={BOOT}", "apply", "-auto-approve", "-input=false",
         f"-var=github_repo={c.repo}", f"-var=create_oidc_provider={'true' if create else 'false'}"])
    out = {k: v["value"] for k, v in json.loads(run(["terraform", f"-chdir={BOOT}", "output", "-json"],
                                                    capture=True, quiet=True)).items()}
    if out.get("github_role_arn"):
        protect_environment(c)
        sync_github(c, {"AWS_ROLE_ARN": out["github_role_arn"], "AWS_REGION": c.region, "PROJECT": c.project,
                        "ENVIRONMENT": c.env})
    return out


def protect_environment(c: Ctx) -> None:
    """The deploy role trusts only the `production` environment: allow that environment on main only."""
    if not c.repo or not shutil.which("gh"):
        print("  gh not available: restrict the `production` environment to the main branch by hand "
              "(Settings -> Environments -> production -> Deployment branches)")
        return
    log(f"GitHub repo {c.repo}: environment `production` deploys from main only")
    run(["gh", "api", "-X", "PUT", f"repos/{c.repo}/environments/production", "--input", "-"], check=False,
        input_=json.dumps({"deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}}))
    run(["gh", "api", "-X", "POST", f"repos/{c.repo}/environments/production/deployment-branch-policies",
         "-f", "name=main", "-f", "type=branch"], check=False, quiet=True, capture=True)  # exists already: fine


def sync_github(c: Ctx, extra: dict | None = None) -> None:
    """The settings CI deploys with - scoring host, allow-list, TF_VARS, the token secret - set to what this machine
    just used, so the next push never redeploys with older ones. Local runs only: CI cannot (and must not) write them."""
    if os.getenv("GITHUB_ACTIONS") == "true" or not c.repo or not shutil.which("gh"):
        return
    log(f"GitHub repo {c.repo}: Actions variables + secrets")
    host, _ = scoring_settings()
    for k, v in {**(extra or {}), "SCORING_HOST": host, "ALLOWED_CIDRS": json.dumps(allowed_cidrs()),
                 "TF_VARS": json.dumps(saved_tf_vars())}.items():
        if v:
            run(["gh", "variable", "set", k, "--body", v, "-R", c.repo], check=False)
    if os.getenv("TF_VAR_scoring_token"):  # via stdin, never on a command line
        run(["gh", "secret", "set", "SCORING_TOKEN", "-R", c.repo], input_=os.environ["TF_VAR_scoring_token"], check=False)


def build_push(c: Ctx, tag: str, url: str) -> None:
    log(f"Build + push the stream image to {url}:{tag}")
    password = run(["aws", "ecr", "get-login-password", "--region", c.region], capture=True, quiet=True)
    run(["docker", "login", "--username", "AWS", "--password-stdin", c.registry], input_=password)
    run(["docker", "build", "--platform", "linux/amd64", "-t", f"{url}:{tag}", "-t", f"{url}:latest", IMAGE[1]])
    run(["docker", "push", f"{url}:{tag}"])
    run(["docker", "push", f"{url}:latest"])


def supabase_holdout(dest: Path) -> bool:
    """Download parquet/transactions/holdout/ from the Supabase bucket (the same files the trainer syncs)."""
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        return False
    auth = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    prefix = "parquet/transactions/holdout/"
    status, body = http(f"{url}/storage/v1/object/list/argus-data", "POST", auth,
                        json.dumps({"prefix": prefix, "limit": 1000}).encode(), timeout=60)
    if status != 200:
        print(f"  Supabase list failed ({status})")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for obj in (o for o in json.loads(body) if o["name"].endswith(".parquet")):
        req = urllib.request.Request(f"{url}/storage/v1/object/argus-data/{prefix}{obj['name']}", headers=auth)
        with urllib.request.urlopen(req, timeout=300) as r:
            (dest / obj["name"]).write_bytes(r.read())
    return any(dest.glob("*.parquet"))


def seed_dataset(c: Ctx, force: bool = False, folder: Path = HOLDOUT) -> None:
    """Copy the labelled holdout to s3://<data bucket>/holdout/ once (the producer replays it)."""
    if not force:
        r = subprocess.run(["aws", "s3api", "list-objects-v2", "--bucket", c.data_bucket, "--prefix", "holdout/",
                            "--max-items", "1", "--query", "KeyCount", "--output", "text"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() not in ("", "0", "None"):
            return
    if not any(folder.glob("*.parquet")) and not supabase_holdout(folder):
        print("  dataset: no holdout locally and no SUPABASE_URL / SUPABASE_KEY -> synthetic traffic only")
        return
    log(f"Copy the holdout to s3://{c.data_bucket}/holdout/")
    run(["aws", "s3", "sync", folder, f"s3://{c.data_bucket}/holdout/", "--only-show-errors"])


def deploy(c: Ctx, tag: str, wait: bool = True) -> dict:
    preflight()
    os.environ["TF_VAR_allowed_cidrs"] = json.dumps(allowed_cidrs())
    tf_init(c)
    log("Registry + buckets first (the image must exist before the service)")
    run(["terraform", f"-chdir={INFRA}", "apply", "-auto-approve", "-input=false",
         "-target=aws_ecr_repository.stream", "-target=aws_s3_bucket.this"])
    build_push(c, tag, tf_outputs().get("ecr") or f"{c.registry}/{c.name}/stream")
    seed_dataset(c)
    log("terraform apply (VPC, ALB, Kinesis, Firehose -> S3 audit, ECS Fargate, CloudWatch, SNS)")
    run(["terraform", f"-chdir={INFRA}", "apply", "-auto-approve", "-input=false", f"-var=image_tag={tag}"])
    out = tf_outputs()
    if wait and out.get("services"):
        log("Waiting for the ECS service to become stable (~3-5 min)")
        if subprocess.run(["aws", "ecs", "wait", "services-stable", "--cluster", c.name,
                           "--services", *out["services"]]).returncode != 0:
            print("  (not stable yet: `python deploy/ops.py status`)")
    summary(out)
    sync_github(c)  # a local deploy (new VM IP, token, settings) must not be undone by the next CI deploy
    return out


def summary(out: dict) -> None:
    log("ARGUS stream on AWS")
    for k in ("stream_url", "stream_api_docs", "scoring_url", "cloudwatch_dashboard", "kinesis_streams", "audit_bucket"):
        if out.get(k):
            print(f"  {k:22s} {out[k]}")
    print("  Next: `python deploy/ops.py platform-env` -> add the 2 lines to the platform's .env (GPU VM or this PC),\n"
          "        restart the platform, open the UI's Live stream page and press Start traffic.")


def control_key(out: dict) -> str:
    """The stream's X-Control-Key, from SSM (Terraform generated it)."""
    return run(["aws", "ssm", "get-parameter", "--name", out["control_key_parameter"], "--with-decryption",
                "--query", "Parameter.Value", "--output", "text"], capture=True, quiet=True, check=False)


def platform_env(c: Ctx) -> None:
    preflight(need_docker=False)
    tf_init(c)
    out = tf_outputs()
    if not out:
        sys.exit("not deployed (no Terraform outputs)")
    key = control_key(out)
    if not key:
        sys.exit(f"could not read {out['control_key_parameter']} from SSM (aws ssm get-parameter)")
    print(f"\n# ARGUS platform -> the AWS stream (append to .env, then: docker compose up -d platform)\n"
          f"ARGUS_STREAM_URL={out['stream_url']}\nARGUS_STREAM_CONTROL_KEY={key}")


def dataset(c: Ctx, folder: str | None) -> None:
    preflight(need_docker=False)
    seed_dataset(c, force=True, folder=Path(folder) if folder else HOLDOUT)
    run(["aws", "ecs", "update-service", "--cluster", c.name, "--service", f"{c.name}-stream", "--force-new-deployment",
         "--query", "service.serviceName", "--output", "text"], check=False)
    print("Service restarted: the producer now replays the holdout.")


def status(c: Ctx) -> None:
    preflight(need_docker=False)
    tf_init(c)
    out = tf_outputs()
    if not out:
        print("Not deployed (no Terraform outputs).")
        return
    summary(out)
    run(["aws", "ecs", "describe-services", "--cluster", c.name, "--services", *out["services"],
         "--query", "services[].{service:serviceName,status:status,running:runningCount,desired:desiredCount,"
                    "rollout:deployments[0].rolloutState}", "--output", "table"], check=False, quiet=True)
    code, body = http(f"{out['stream_url']}/live/api/state", headers={"X-Control-Key": control_key(out)}, timeout=8)
    if code == 200:
        st = json.loads(body)
        print(f"  stream: {st.get('stream')} | consumed {st.get('consumed')} | scoring {(st.get('scoring') or {}).get('status')}"
              f" | traffic {'on' if st.get('producer', {}).get('running') else 'off'}")
    else:
        print(f"  stream API not reachable from here ({code}): allowed_cidrs covers this machine?")


def down(c: Ctx, all_: bool = False, yes: bool = False) -> None:
    preflight(need_docker=False)
    if not yes and input(f"Destroy every resource of '{c.name}' in {c.region} (account {c.account}), incl. Kinesis "
                         f"streams and the S3 audit archive? Type the name to confirm: ").strip() != c.name:
        sys.exit("aborted")
    if subprocess.run(["aws", "s3api", "head-bucket", "--bucket", c.state_bucket], capture_output=True).returncode == 0:
        tf_init(c)
        log(f"terraform destroy {c.name} (only resources in this project's state)")
        os.environ.setdefault("TF_VAR_allowed_cidrs", json.dumps(["0.0.0.0/0"]))  # any value: destroy ignores it
        run(["terraform", f"-chdir={INFRA}", "destroy", "-auto-approve", "-input=false"])
    else:
        print(f"state bucket {c.state_bucket} not found: the main stack is already gone")
    if all_:
        log("Destroying the bootstrap stack (state bucket + CI role)")
        if (BOOT / "terraform.tfstate").exists():
            run(["terraform", f"-chdir={BOOT}", "init", "-input=false"])
            run(["terraform", f"-chdir={BOOT}", "destroy", "-auto-approve", "-input=false", f"-var=github_repo={c.repo}",
                 f"-var=create_oidc_provider={'true' if owns_oidc() else 'false'}"])
        else:
            print("  no local bootstrap state (run `down --all` on the machine that ran `up`)")
    audit(c)


def audit(c: Ctx) -> None:
    log(f"AWS resources still tagged Project={c.project} in {c.region}")
    r = run(["aws", "resourcegroupstaggingapi", "get-resources", "--tag-filters", f"Key=Project,Values={c.project}",
             "--query", "ResourceTagMappingList[].ResourceARN", "--output", "text"], capture=True, check=False, quiet=True)
    arns = [a for a in r.split() if a]
    print("\n".join("  " + a for a in arns) if arns else "  none")
    print("  (the tagging index can lag a few minutes after a destroy)")


def main(argv=None) -> None:
    load_env(ROOT / ".env")
    restore_tf_vars()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    p.add_argument("--project", default=os.getenv("PROJECT", "argus"))
    p.add_argument("--environment", default=os.getenv("ENVIRONMENT", "prod"))
    p.add_argument("--github-repo", default=os.getenv("GITHUB_REPO") or detect_repo(),
                   help="owner/repo allowed to deploy from GitHub Actions")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("up")
    s.add_argument("--tag")
    s.add_argument("--no-wait", action="store_true")
    sub.add_parser("check", help="read-only: settings, the Fraud Model API and isolation before deploying")
    sub.add_parser("bootstrap")
    s = sub.add_parser("deploy")
    s.add_argument("--tag")
    s.add_argument("--no-wait", action="store_true")
    s = sub.add_parser("dataset")
    s.add_argument("--dir", help="folder with the holdout *.parquet (default training/data/raw/.../holdout)")
    sub.add_parser("status")
    sub.add_parser("platform-env", help="ARGUS_STREAM_URL + ARGUS_STREAM_CONTROL_KEY for the platform")
    s = sub.add_parser("down")
    s.add_argument("--all", action="store_true")
    s.add_argument("-y", "--yes", action="store_true")
    sub.add_parser("audit")
    a = p.parse_args(argv)
    c = Ctx(a)
    if a.cmd == "check":
        sys.exit(0 if check(c) else 1)
    elif a.cmd == "up":
        if not check(c):
            sys.exit("pre-deploy check failed: nothing was created")
        preflight()
        bootstrap(c)
        deploy(c, a.tag or git_tag(), wait=not a.no_wait)
    elif a.cmd == "bootstrap":
        preflight(need_docker=False)
        bootstrap(c)
    elif a.cmd == "deploy":
        deploy(c, a.tag or git_tag(), wait=not a.no_wait)
    elif a.cmd == "dataset":
        dataset(c, a.dir)
    elif a.cmd == "status":
        status(c)
    elif a.cmd == "platform-env":
        platform_env(c)
    elif a.cmd == "down":
        down(c, a.all, a.yes)
    elif a.cmd == "audit":
        audit(c)


if __name__ == "__main__":
    main()
