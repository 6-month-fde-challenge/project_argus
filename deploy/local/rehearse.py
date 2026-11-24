"""Rehearse the AWS deployment on this machine, without an AWS account (full guide: deploy/README.md, section 4).

USAGE
    python deploy/local/rehearse.py up       # build the rehearsal and run every check (about 5 minutes)
    python deploy/local/rehearse.py verify   # run the checks again; the rehearsal keeps running until `down`
    python deploy/local/rehearse.py down     # remove everything the rehearsal created
    Same as: .\\argus.ps1 aws-local up|verify|down, or the UI: Operations -> Rehearse the AWS stack locally.
    On Linux (the GPU VM): python3 instead of python.

BEFORE `up`
    1. Docker is running.
    2. The stream image exists:   docker compose --profile stream build stream
    3. A run is served:           .\\argus.ps1 serve <run>   (else the "Fraud Model API reached" check fails)

WHAT `up` DOES
    1. Starts Moto (motoserver/moto), an open-source AWS emulator, as host "aws" on the compose network.
    2. Copies deploy/terraform + deploy/local/moto_override.tf into the Docker volume argus-aws-local and applies it
       there: every AWS endpoint goes to the emulator, the state stays in the volume. The repository is not modified.
    3. Runs the stream image (argus-stream:latest, the same Dockerfile as the ECR image) with the environment of the
       ECS task definition Terraform created and the secrets resolved from SSM, as ECS would inject them.
    4. Sends traffic: holdout replay from S3, synthetic payments, an attack wave, one payment from a blocklisted device.
    5. Prints PASS / FAIL per check and exits 1 on any FAIL.

WHAT THE EMULATOR CANNOT SHOW (confirm on the first real deployment)
    IAM enforcement, networking and the load balancer, Fargate itself, Firehose delivery to S3, and CloudWatch alarms
    and the dashboard (Moto does not speak the CloudWatch protocol of the Terraform AWS provider; each alarm's metric is
    checked against what the stream publishes instead).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# the compose network (where `scoring` runs): <project>_default, the project being the folder name unless overridden
NETWORK = os.environ.get("COMPOSE_PROJECT_NAME", ROOT.name).lower() + "_default"
MOTO_IMAGE = "motoserver/moto:5.2.5"
TF_IMAGE = "hashicorp/terraform:1.14.0"
STREAM_IMAGE = "argus-stream:latest"
VOLUME, MOTO, STREAM = "argus-aws-local", "argus-aws-local", "argus-aws-stream"
STREAM_PORT = 8013
# resource types the emulator cannot create (Moto vs the AWS provider's CloudWatch protocol): checked another way
EMULATOR_SKIPS = ("aws_cloudwatch_metric_alarm", "aws_cloudwatch_dashboard")
TF_VARS = {"environment": "local", "allowed_cidrs": '["127.0.0.1/32"]', "scoring_host": "scoring", "scoring_port": "8080",
           "scoring_scheme": "http", "image_tag": "local"}
EMULATOR_ENV = {"AWS_ENDPOINT_URL": "http://aws:5000", "KINESIS_ENDPOINT": "http://aws:5000",
                "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}


def log(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def docker(*args: str, check: bool = True, capture: bool = False, quiet: bool = False, stdin: str | None = None) -> str:
    if not quiet:
        print("$ docker " + " ".join(a if len(a) < 90 else a[:87] + "..." for a in args), flush=True)
    r = subprocess.run(["docker", *args], text=True, capture_output=capture, input=stdin, encoding="utf-8")
    if check and r.returncode != 0:
        if capture:
            sys.stderr.write((r.stderr or "") + (r.stdout or ""))
        sys.exit(f"docker {args[0]} failed ({r.returncode})")
    return (r.stdout or "").strip() if capture else ""


def scoring_token() -> str:
    """The token the local Fraud Model API accepts: ARGUS_SCORING_TOKEN from .env, else compose's default."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("ARGUS_SCORING_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "demo-service-token"


def terraform(*args: str, capture: bool = False) -> str:
    return docker("run", "--rm", "--network", NETWORK, "-v", f"{VOLUME}:/work", "-v", "argus-tf-plugins:/plugins",
                  "-e", "TF_PLUGIN_CACHE_DIR=/plugins", "-w", "/work/stack", TF_IMAGE, *args, capture=capture)


def inside(*args: str, mounts: tuple[str, ...] = (), capture: bool = False, check: bool = True) -> str:
    extra = [x for m in mounts for x in ("-v", m)]
    return docker("run", "--rm", "--network", NETWORK, "-v", f"{VOLUME}:/work:ro", "-v",
                  f"{ROOT / 'deploy' / 'local' / 'inside.py'}:/inside.py:ro", *extra,
                  *[x for k, v in {**EMULATOR_ENV, "AWS_REGION": "ap-south-1"}.items() for x in ("-e", f"{k}={v}")],
                  "--entrypoint", "python", STREAM_IMAGE, "/inside.py", *args, capture=capture, check=check)


def wait_http(url: str, seconds: int, what: str) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except OSError:
            time.sleep(1)
    sys.exit(f"{what} did not come up ({url})")


def control(path: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{STREAM_PORT}/live/api/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Control-Key": key}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=30))


def split_plan(plan: dict) -> tuple[list[str], list[str]]:
    """-> (addresses to apply on the emulator, addresses it cannot create). Every created resource is one or the other."""
    created = [c["address"] for c in plan["resource_changes"]
               if c["mode"] == "managed" and "create" in c["change"]["actions"]]
    skip = [a for a in created if a.split(".")[0] in EMULATOR_SKIPS]
    return [a for a in created if a not in skip], skip


# ------------------------------------------------------------------ steps
def preflight() -> None:
    log("Preflight")
    docker("info", capture=True, quiet=True)
    if not docker("images", "-q", STREAM_IMAGE, capture=True, quiet=True):
        sys.exit(f"{STREAM_IMAGE} is missing: docker compose --profile stream build stream")
    if not docker("network", "ls", "-q", "--filter", f"name=^{NETWORK}$", capture=True, quiet=True):
        docker("network", "create", NETWORK)
    print("docker ok, stream image present, network", NETWORK)
    if not docker("ps", "-q", "--filter", "label=com.docker.compose.service=scoring", capture=True, quiet=True):
        print("WARNING: no Fraud Model API is running (.\\argus.ps1 serve <run>): every decision will be the fail-open "
              "rule and the check 'Fraud Model API reached from the task' will fail")


def start_emulator() -> None:
    log(f"AWS emulator ({MOTO_IMAGE}) as 'aws' on {NETWORK}")
    docker("rm", "-f", MOTO, check=False, capture=True, quiet=True)
    docker("run", "-d", "--name", MOTO, "--network", NETWORK, "--network-alias", "aws", "-p", "127.0.0.1:4599:5000",
           "-e", "MOTO_IAM_LOAD_MANAGED_POLICIES=true", MOTO_IMAGE, capture=True)
    wait_http("http://127.0.0.1:4599/moto-api/", 60, "the emulator")


def apply_stack() -> None:
    log("Terraform: a copy of deploy/terraform, applied to the emulator")
    docker("volume", "create", VOLUME, capture=True, quiet=True)
    src = ROOT / "deploy"
    docker("run", "--rm", "-v", f"{VOLUME}:/work", "-v", f"{src}:/src:ro", "--entrypoint", "sh", TF_IMAGE, "-c",
           "rm -rf /work/stack /work/state && mkdir -p /work/stack /work/state && cp /src/terraform/*.tf "
           "/src/terraform/.terraform.lock.hcl /work/stack/ && cp /src/local/moto_override.tf /work/stack/")
    terraform("init", "-input=false", "-no-color")
    var_args = [x for k, v in {**TF_VARS, "scoring_token": scoring_token()}.items() for x in ("-var", f"{k}={v}")]
    terraform("plan", "-input=false", "-no-color", "-out=/work/plan.out", *var_args)
    plan = json.loads(terraform("show", "-json", "/work/plan.out", capture=True))
    targets, skipped = split_plan(plan)
    print(f"{len(targets) + len(skipped)} resources planned; {len(targets)} applied here, {len(skipped)} checked another way")
    # keep the plan (alarm definitions) and the skip list for the checks
    for name, data in (("plan.json", plan), ("skipped.json", skipped)):     # via stdin: no command-line size limit
        docker("run", "--rm", "-i", "-v", f"{VOLUME}:/work", "--entrypoint", "sh", TF_IMAGE, "-c",
               f"cat > /work/{name}", stdin=json.dumps(data), quiet=True)
    print("(Terraform will warn that -target is in effect: expected here, it is how the resources above are left out)")
    terraform("apply", "-auto-approve", "-input=false", "-no-color", *[f"-target={t}" for t in targets], *var_args)


def run_stream() -> dict:
    log("The stream task, as ECS would start it from the task definition")
    desc = json.loads(inside("describe", capture=True))
    env = {**desc["env"], **EMULATOR_ENV}
    print("task image (on AWS):", desc["image"])
    print("task environment:", ", ".join(sorted(desc["env"])))
    holdout = ROOT / "training" / "data" / "raw" / "parquet" / "transactions" / "holdout"
    if any(holdout.glob("*.parquet")):
        inside("seed", "/holdout", mounts=(f"{holdout}:/holdout:ro",))
    else:
        print("no local holdout (training/data): the producer uses synthetic traffic only")
    inside("subscribe")
    docker("rm", "-f", STREAM, check=False, capture=True, quiet=True)
    docker("run", "-d", "--name", STREAM, "--network", NETWORK, "-p", f"127.0.0.1:{STREAM_PORT}:8003",
           *[x for k, v in env.items() for x in ("-e", f"{k}={v}")], STREAM_IMAGE, capture=True, quiet=True)
    wait_http(f"http://127.0.0.1:{STREAM_PORT}/live/health", 90, "the stream task")
    return env


def traffic(env: dict, seconds: int) -> None:
    log(f"Traffic for {seconds} s (+ one payment from a blocklisted device, + one attack wave)")
    key = env["CONTROL_API_KEY"]
    time.sleep(8)
    print("producer:", control("producer", {"action": "start", "rate": 2}, key).get("running"))
    rules = next((p for p in (ROOT / "training" / "outputs" / "_scoring" / "rules_config.json",
                              ROOT / "training" / "data" / "raw" / "rules_config.json") if p.exists()), None)
    if rules:
        inside("block", "/rules.json", mounts=(f"{rules}:/rules.json:ro",))
    time.sleep(10)
    print("attack:", control("inject", {"n": 15, "fraud_type": "card_testing"}, key))
    time.sleep(seconds)
    print("producer stopped:", not control("producer", {"action": "stop"}, key).get("running"))
    time.sleep(5)


def verify() -> int:
    log("Checks")
    return subprocess.run(["docker", "run", "--rm", "--network", NETWORK, "-v", f"{VOLUME}:/work:ro", "-v",
                           f"{ROOT / 'deploy' / 'local' / 'inside.py'}:/inside.py:ro",
                           *[x for k, v in {**EMULATOR_ENV, "AWS_REGION": "ap-south-1",
                                            "REHEARSAL_STREAM_URL": f"http://{STREAM}:8003"}.items() for x in ("-e", f"{k}={v}")],
                           "--entrypoint", "python", STREAM_IMAGE, "/inside.py", "verify"]).returncode


def down() -> None:
    log("Removing the rehearsal (emulator, stream task, Terraform state)")
    docker("rm", "-f", STREAM, MOTO, check=False, capture=True)
    docker("volume", "rm", VOLUME, check=False, capture=True)
    print("removed; the Terraform provider cache (volume argus-tf-plugins) is kept for the next rehearsal")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="up", choices=["up", "verify", "down"])
    ap.add_argument("--traffic-seconds", type=int, default=75, help="CloudWatch metrics flush every 60 s")
    a = ap.parse_args(argv)
    if a.command == "down":
        down()
        return
    if a.command == "verify":
        sys.exit(verify())
    preflight()
    start_emulator()
    apply_stack()
    env = run_stream()
    traffic(env, a.traffic_seconds)
    code = verify()
    print(f"\nRehearsal {'PASSED' if code == 0 else 'FAILED (see deploy/README.md, section 10)'}.\n"
          f"Still running, for inspection:\n"
          f"  stream task API   http://127.0.0.1:{STREAM_PORT}/live/docs\n"
          f"  AWS emulator      http://127.0.0.1:4599/moto-api/\n"
          f"  checks again      python deploy/local/rehearse.py verify\n"
          f"  remove it         python deploy/local/rehearse.py down")
    sys.exit(code)


if __name__ == "__main__":
    main()
