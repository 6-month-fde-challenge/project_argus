# ARGUS deployment: GPU VM (Hyperstack) + AWS (ap-south-1)

The main [README](../README.md) runs everything on one machine with Docker. This folder takes the same solution to
real infrastructure:

- **a GPU VM** (Hyperstack, or any Ubuntu host with an NVIDIA GPU) trains the ladder and serves the Fraud Model API;
- **AWS ap-south-1** runs the real-time path around it: Amazon Kinesis → stream scorer → decision → action, with an
  S3 audit archive, SNS alerts and CloudWatch monitoring;
- **GitHub Actions** tests every push and deploys `main` to AWS without stored AWS keys.

```
 GPU VM (deploy/hyperstack)                           AWS ap-south-1 (deploy/terraform, own VPC 10.50.0.0/16)
 ┌───────────────────────────────────────┐            ┌──────────────────────────────────────────────────────────┐
 │ training jobs (argus_ft ladder)       │            │ Kinesis  argus-prod-transactions ──► stream scorer (ECS  │
 │ Fraud Model API :8080  (public,       │◄── HTTP ───┤          Fargate, /live API behind an ALB)               │
 │   bearer token) + vLLM explainer      │  + token   │            │ decision + action                          │
 │ UI :3000, platform :8090, MLflow :5000│            │ Kinesis  argus-prod-decisions ──► Firehose ──► S3 audit  │
 │   (127.0.0.1 only: SSH tunnel)        │── HTTP ───►│ SNS: BLOCK actions, alarms · CloudWatch dashboard + alarms│
 └───────────────────────────────────────┘  Live     └──────────────────────────────────────────────────────────┘
          ▲ the platform proxies the AWS stream to the UI's Live stream page
 GitHub ── push to main ──► ci.yml (all tests, Terraform, shellcheck) ──► deploy.yml (OIDC role) ──► ops.py deploy
```

### Contents

[1. What runs where](#1-what-runs-where) · [2. Isolation](#2-isolation-what-this-touches-in-the-aws-account) ·
[3. Configuration checklist](#3-configuration-checklist) · [4. Runbook](#4-runbook-in-this-order) ·
[5. Script reference](#5-script-reference) · [6. Data, audit and monitoring](#6-data-audit-and-monitoring) ·
[7. Security](#7-security) · [8. Cost](#8-cost) · [9. Troubleshooting](#9-troubleshooting)

---

## 1. What runs where

| Where | What | Why there |
|---|---|---|
| GPU VM | the whole Docker stack of the main README: training jobs, MLflow, Fraud Model API, vLLM explainer, UI, platform | training and serving need the GPU; CRD: models self-hosted |
| AWS Fargate | `stream/` (image `argus-<env>/stream`): Kinesis producer (off until started), scorer, analytics API under `/live` | the online event path of CRD 11 / LLD 3.3 in VKPayments' AWS account |
| AWS managed | Kinesis Data Streams, Kinesis Firehose, S3, SNS, CloudWatch, ECR, SSM | no servers to patch |
| GitHub | `.github/workflows/ci.yml`, `deploy.yml`, `destroy.yml` | every push tested, `main` deployed |

**One decision path.** The stream calls the same `POST /v1/fraud/score` as everyone else, so it always scores with
whatever run is active on the Fraud Model API (switch it on the Scoring page; the stream needs no restart). Each CRD
10.4 decision becomes an action on the decisions stream:

| Decision | Risk level | Action |
|---|---|---|
| ALLOW | LOW | `APPROVE` |
| ALLOW | MEDIUM | `APPROVE_STEP_UP` (step-up OTP where the merchant supports it) |
| REVIEW | HIGH | `HOLD_FOR_REVIEW` |
| BLOCK | CRITICAL | `BLOCK_AND_ALERT` (also published to SNS `argus-<env>-fraud-actions`) |
| API unavailable | – | CRD fail-open rule (FR-API-01): approve below ₹10,000, otherwise `HOLD_FOR_REVIEW`; `decision_source: FAIL_OPEN` |
| API rejects the token (401) | – | a misconfiguration, never an approval: `HOLD_FOR_REVIEW` for every amount; `decision_source: AUTH_ERROR`, `scoring-auth` alarm |

## 2. Isolation: what this touches in the AWS account

| Guarantee | How |
|---|---|
| Own network | new VPC `10.50.0.0/16` with its own subnets, internet gateway, ALB and security groups; no peering |
| Own names | every resource is named `argus-<env>-*` (cluster, ALB, ECR `argus-<env>/*`, IAM roles, log groups, SSM `/argus-<env>/*`, Kinesis, Firehose, buckets) |
| Own tags | every resource carries `Project=argus`, `ManagedBy=terraform` |
| Own state | Terraform state in its own bucket `argus-tfstate-<account>-<region>`; `down` can only remove what is in that state |
| Checked first | `ops.py check` (read-only) verifies that no existing resource uses these names, that a VPC slot is free (default quota 5 per region), that the Fraud Model API answers **and accepts the token**; `up` refuses to start if it fails |
| Clean teardown | `ops.py down` destroys exactly this stack (images, buckets and the audit archive included); `ops.py audit` lists anything still tagged `Project=argus` |

Account-wide items (IAM is global): the role `argus-github-deploy`, and the GitHub OIDC provider
`token.actions.githubusercontent.com`, created only if the account has none and removed by `down --all` only if this
project created it. The deploy role can be assumed only by this repository's `main` branch or `production`
environment, and may manage IAM only for names starting with `argus-`.

## 3. Configuration checklist

| # | Setting | Where it comes from | Goes into |
|---|---|---|---|
| 1 | `SUPABASE_URL`, `SUPABASE_KEY` | the Supabase project with the `argus-data` bucket | `.env` on the **VM** (and your PC) |
| 2 | `HF_TOKEN` | Hugging Face account that accepted the Llama 3.2 and Gemma 2 licences | `.env` on the VM, `PROFILE=prod` only |
| 3 | `ARGUS_SCORING_TOKEN`, `ARGUS_SCORING_ADMIN_TOKEN`, `ARGUS_STREAM_CONTROL_KEY` | generated once by `setup_vm.sh` | `.env` on the VM (never in git) |
| 4 | `TF_VAR_scoring_host`, `TF_VAR_scoring_token` | the VM: `training/outputs/_deploy/aws.env`, written by `pipeline.sh` | `.env` on your PC |
| 5 | AWS credentials | `aws configure` (admin rights for the first `up`) | your PC, once |
| 6 | GitHub CLI login | `gh auth login` | your PC, once |
| 7 | *(optional)* `TF_VAR_alert_email`, `AWS_REGION`, `TF_VAR_allowed_cidrs` | you | `.env` on your PC |

Terraform creates the rest: the Kinesis streams, Firehose, buckets, the stream control key in SSM, and (through
`ops.py up`) the GitHub variables and secrets for CI/CD.

**Tools on your PC:** Python 3.10+, Git, Docker Desktop, AWS CLI v2, Terraform ≥ 1.10, GitHub CLI.

---

## 4. Runbook (in this order)

### Step 1: the GPU VM

1. In the Hyperstack console create (or restore) a VM: one NVIDIA GPU (an A100 80GB for `PROFILE=prod`; ≥ 16 GB for
   the demo profile), an **Ubuntu image "… CUDA … with Docker"** (Docker and the driver preinstalled; `setup_vm.sh`
   installs whatever is missing), your SSH key and a public IP.
   **Disk:** ≥ 60 GB free for Docker images (Docker's data-root, normally on the root disk) and ≥ 40 GB for the
   repository (data, model cache, runs). Do **not** clone into `/ephemeral`: Hyperstack wipes the ephemeral disk on
   hibernate, and `setup_vm.sh` refuses it. If the root disk is too small, pick a large-root-disk flavour or attach a
   persistent volume.
2. Firewall: inbound **TCP 22 and 8080** only. The UI, platform and MLflow listen on 127.0.0.1 and are reached through
   an SSH tunnel; 8080 is the token-protected Fraud Model API (AWS Fargate has no fixed IP to allow-list).
3. Note the public IP. It can change after a hibernate / restore (Step 7).

### Step 2: train and serve on the VM (one command, unattended)

From your PC (Git Bash), pushing this repository:
```bash
export HS_HOST=<VM_IP> HS_KEY=~/.ssh/<key>
bash deploy/hyperstack/remote.sh push               # code + configs (+ .env the first time)
bash deploy/hyperstack/remote.sh pipeline demo      # or: prod | smoke
bash deploy/hyperstack/remote.sh logs pipeline      # wait for "PIPELINE DONE" (Ctrl-C only stops following)
```
Or on the VM itself:
```bash
git clone <this repo> argus_solution && cd argus_solution    # or remote.sh push
cp -n .env.example .env && nano .env                         # first time: SUPABASE_URL, SUPABASE_KEY (+ HF_TOKEN)
bash deploy/hyperstack/pipeline.sh                           # PROFILE=prod / smoke to change; runs in tmux
tail -f training/outputs/_logs/pipeline.log
```

`pipeline.sh` runs six steps by itself:

1. **Setup:** Docker, the NVIDIA Container Toolkit, `.env` secrets, all images, the data from Supabase and every model
   of the profile (gated access checked first, so paid GPU time never waits on a download or a licence).
2. **Health check** (`check.sh`).
3. **Train** the profile's runs with the normal ladder (`argus_ft.pipeline`, so they appear live in the UI):
   `demo` → `demo_hybrid` then `demo_llm_scorer`; `prod` → `prod_a100`; `smoke` → `smoke`.
4. **Reports:** `report.md` per run and `comparison.md`, including the holdout acceptance verdict (stage 9).
5. **Serve** the result: Fraud Model API on :8080 + vLLM explainer + UI, all restarting by themselves.
6. **Exercise it:** the CRD / LLD scenarios (`argus_api.demo`) and a 30-second load test (`argus_api.loadtest`),
   then write `training/outputs/_deploy/aws.env` for your PC.

| Variant | Command |
|---|---|
| demo profile (default) | `bash deploy/hyperstack/pipeline.sh` |
| production profile, one A100 80GB | `PROFILE=prod bash deploy/hyperstack/pipeline.sh` |
| quick health run of every stage | `PROFILE=smoke bash deploy/hyperstack/pipeline.sh` |
| VM restarted, models already trained | `SKIP_TRAIN=1 bash deploy/hyperstack/pipeline.sh` |
| serve another run | `SERVE_RUN=<run> SKIP_TRAIN=1 bash deploy/hyperstack/pipeline.sh` |
| health of GPU, data, images, runs, services | `bash deploy/hyperstack/check.sh` (`--full` adds CUDA, model access, test suites) |

**Open the UI** from your PC: `bash deploy/hyperstack/remote.sh tunnel`, then http://localhost:3000.

### Step 3: your PC (repository root)

```bash
# Git Bash (not PowerShell: Windows PowerShell's >> writes UTF-16, which docker compose cannot read)
bash deploy/hyperstack/remote.sh awsenv >> .env   # or paste the 2 lines of the VM's training/outputs/_deploy/aws.env
python deploy/ops.py check                        # read-only; must end with "ready to deploy"
```
A key written twice in `.env` takes its last value (as in docker compose), so appending a new value is enough.

### Step 4: deploy to AWS (one command)

```powershell
python deploy/ops.py up          # or: .\argus.ps1 aws up
```
It runs (about 15–20 minutes the first time):

1. **check:** tools, AWS identity, the Fraud Model API and its token, isolation in ap-south-1;
2. **bootstrap:** state bucket, GitHub OIDC deploy role, GitHub variables and the `SCORING_TOKEN` secret;
3. **ECR + buckets**, then **build and push** the stream image;
4. **dataset:** the labelled holdout (`parquet/transactions/holdout`) to the data bucket, for replay;
5. **terraform apply:** VPC, ALB, Kinesis ×2, Firehose → S3 audit, ECS Fargate, CloudWatch alarms + dashboard, SNS;
6. **wait** until the service is stable, then **print the URLs**.

### Step 5: use it

1. `python deploy/ops.py platform-env` prints `ARGUS_STREAM_URL` and `ARGUS_STREAM_CONTROL_KEY`. Append them to the
   VM's `.env` (where the platform runs) and restart the platform: `docker compose up -d platform`.
2. UI → **Live stream** → **Start traffic** (rate in transactions per second). **Inject attack wave** sends one fraud
   ring's burst (account takeover, card testing, mule account, impossible travel).
3. Watch the pipeline strip (producer → Kinesis → Fraud Model API → action → decisions stream), live recall and
   precision at BLOCK against the producer's ground truth, API p99, the decision mix and the live feed.
4. **Shadow a candidate (MLOPS-06):** on the same page pick another trained run under *Shadow run*. Every transaction
   is then also scored by it (`POST /v1/fraud/shadow/{run}`), never deciding, and the card compares it with the live
   model: agreement, recall, precision at BLOCK. The platform first has the Fraud Model API load the run as its shadow
   (admin token), so loading never happens on the scoring path. Promote it afterwards on Scoring → Deployments.
5. AWS console: CloudWatch dashboard `argus-prod` (decisions, API latency p50/p95/p99 against the 250 ms line, Kinesis
   lag, Firehose deliveries, errors), alarms to SNS `argus-prod-alerts`, and the archive in `s3://argus-prod-audit-*/decisions/`.
6. **Stop traffic** when done. The stream is idle (no records, no scoring) until started again.

### Step 6: change code → automatic deployment

`git push` to `main` runs `deploy.yml`: every test suite (platform, stream, training logic + scoring contract on CPU,
frontend), `terraform fmt` / `validate` for both stacks, shellcheck and the compose file, then OIDC → `ops.py deploy
--tag <sha>`: image build + push, `terraform apply`, Fargate rollout with automatic rollback on failure.

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | every PR / push | all tests + Terraform + shellcheck + compose |
| `deploy.yml` | push to `main`, or *Run workflow* | ci → OIDC role → `python deploy/ops.py deploy` (skipped until `up` ran once) |
| `destroy.yml` | manual, type `argus-prod` | `python deploy/ops.py down -y` |

The GPU VM is not redeployed by CI (it holds training state): after a push, `remote.sh push` then `serve.sh <run>`.

### Step 7: the VM restarted and its IP changed

```bash
# VM
cd argus_solution && SKIP_TRAIN=1 bash deploy/hyperstack/pipeline.sh
```
```powershell
# PC: put the new IP in .env (TF_VAR_scoring_host), then
python deploy/ops.py deploy      # updates SCORING_URL and the allow-list, and (with gh logged in) the GitHub variables
                                 # SCORING_HOST / ALLOWED_CIDRS / TF_VARS + the token secret, so CI deploys the same
```

### Step 8: shut down (stops all costs)

```powershell
python deploy/ops.py down        # type argus-prod; removes only this project's AWS resources (Kinesis, S3 audit included)
python deploy/ops.py audit       # confirms nothing tagged Project=argus is left
python deploy/ops.py down --all  # also the state bucket + the GitHub deploy role
```
On the VM: `bash deploy/hyperstack/serve.sh stop`, `tmux kill-server`, then **hibernate the VM** (it bills hourly).

---

## 5. Script reference

### GPU VM: `deploy/hyperstack/`

| Script | Runs on | Does |
|---|---|---|
| `setup_vm.sh` | VM | Docker + NVIDIA Container Toolkit, `.env` secrets and bindings, all images, data, model weights (`PROFILE=demo\|prod\|smoke`) |
| `check.sh [--full]` | VM | host, `.env`, data, images, runs (with holdout verdicts), services; `--full`: CUDA in the training image, gated-model access, test suites |
| `train.sh <profile> [args]` | VM | one ladder run in tmux `train` (stops serving first); args go to `argus_ft.pipeline` (`--stages`, `--set`) |
| `serve.sh [run\|stop]` | VM | UI + platform + MLflow + Fraud Model API + vLLM for a run (newest servable by default); waits until healthy |
| `pipeline.sh` | VM | everything above, unattended in tmux `pipeline`, then `aws.env` |
| `remote.sh <cmd>` | your PC | `push`, `setup`, `pipeline`, `train`, `logs`, `status`, `tunnel`, `pull`, `awsenv`, `ssh` over SSH |

Logs: `training/outputs/_logs/{pipeline,train}.log`. Every training run also writes `events.jsonl`, so the UI follows
CLI runs live.

### AWS: `deploy/ops.py` (also `.\argus.ps1 aws <command>`)

| Command | Does |
|---|---|
| `check` | read-only: tools, AWS identity, Fraud Model API health + token, allow-list, isolation |
| `up` | `check` → `bootstrap` → `deploy` |
| `bootstrap` | state bucket, OIDC provider (if none), deploy role, `production` environment on main only, GitHub variables + secret |
| `deploy [--tag T]` | image build + push, holdout to S3 (once), `terraform apply`, wait for the service |
| `dataset [--dir D]` | re-copy the holdout and restart the service |
| `status` | outputs, ECS service state, stream state |
| `platform-env` | `ARGUS_STREAM_URL` + `ARGUS_STREAM_CONTROL_KEY` for the platform |
| `down [--all] [-y]` | `terraform destroy` (+ bootstrap with `--all`), then `audit` |
| `audit` | resources still tagged `Project=argus` |

### Terraform: `deploy/terraform/` (main stack) and `deploy/terraform/bootstrap/`

| File | Resources |
|---|---|
| `network.tf` | VPC, 2 public subnets, internet gateway, ALB (HTTP, optional HTTPS with `certificate_arn`), security groups |
| `kinesis.tf` | `argus-<env>-transactions`, `argus-<env>-decisions` (KMS, 24 h) |
| `storage.tf` | data bucket (holdout), audit bucket (versioned, Glacier after 90 days, kept `audit_retention_days`), Firehose decisions → audit |
| `ecs.tf` | ECR, SSM secrets (scoring token, control key), IAM (least privilege), ECS cluster + Fargate service (circuit breaker) |
| `monitoring.tf` | log groups, SNS alerts + actions, alarms, dashboard |
| `variables.tf` / `outputs.tf` | every knob (defaults documented) / URLs and names `ops.py` prints |

## 6. Data, audit and monitoring

| Where | What | Kept |
|---|---|---|
| Kinesis `argus-<env>-transactions` | CRD 10.2 requests with the producer's ground truth | 24 h |
| Kinesis `argus-<env>-decisions` | every decision + its action, model version, decision source, API latency | 24 h |
| S3 `argus-<env>-audit-<account>/decisions/year=/month=/day=/` | the decisions stream, gzip JSON lines (Firehose) | `audit_retention_days` (5 years, NFR-11) |
| S3 `argus-<env>-data-<account>/holdout/` | the labelled holdout the producer replays | until `down` |
| SQLite in the task | analytics for the Live stream page | rebuilt from the decisions stream after a restart |
| SSM `/argus-<env>/*` | scoring token, control key (SecureString) | injected into the task, never in an image |

**Alarms** (to SNS `argus-<env>-alerts`; set `TF_VAR_alert_email` to receive them):

| Alarm | Fires when |
|---|---|
| `scoring-unreachable` | any transaction was decided by the fail-open rule (the API did not answer) |
| `api-latency-p99` | the Fraud Model API p99, as the stream sees it, exceeds 250 ms (OBJ-10) |
| `block-spike` | more than 60 BLOCKs in 5 minutes (attack wave, or a bad threshold change) |
| `missed-fraud` | more than 5 labelled frauds approved in 15 minutes |
| `kinesis-lag` | the scorer is more than 60 s behind |
| `task-cpu` / `task-memory` / `unhealthy` | the Fargate task is saturated or fails its health check |

## 7. Security

- **Tokens.** The VM never runs with the demo tokens from git: `setup_vm.sh` generates `ARGUS_SCORING_TOKEN` /
  `ARGUS_SCORING_ADMIN_TOKEN`, which the scoring service and the platform both read. On AWS the token lives in SSM.
  Token checks are constant-time.
- **Exposure.** On the VM only the token-protected Fraud Model API is public (`ARGUS_BIND=127.0.0.1` for the rest).
  The AWS load balancer accepts only `allowed_cidrs`: by default the VM (whose platform proxies the stream) and the
  IP of the machine that ran `ops.py`. The traffic controls also need the control key, which stays in the platform.
- **No keys in GitHub.** CI assumes an IAM role through OIDC, trusted only for this repository's `production`
  environment, which `bootstrap` restricts to the `main` branch (no other branch or pull request can assume it). The
  role manages only `argus-*` IAM roles / policies, can attach no AWS-managed policy except the ECS task-execution
  one, and can never change its own permissions.
- **Before real data: TLS on both hops.** As built, the scoring token (Fargate → VM) and the control key (platform →
  ALB) travel over plain HTTP. Close it with (1) `TF_VAR_certificate_arn=<ACM certificate>` for the ALB (HTTPS listener
  + HTTP→HTTPS redirect are already in `network.tf`) and `ARGUS_STREAM_URL=https://...`; (2) a DNS name for the VM and a
  TLS reverse proxy in front of :8080 (e.g. Caddy with Let's Encrypt), then `TF_VAR_scoring_host=<that name>` with an
  `TF_VAR_scoring_scheme=https` (and `TF_VAR_scoring_port=443`) - or keep the VM inside a VPN / private link to the VPC.
- **Production gap.** Plain HTTP between AWS and the VM is for this build only. Production (CRD NFR-07) is mTLS +
  15-minute service JWTs inside VKPayments' VPC, with the Fraud Model API on EKS (README section 22), and S3 Object
  Lock in compliance mode on the audit bucket.

## 8. Cost

| Item | Cost (approximate, if left running) |
|---|---|
| Fargate 0.5 vCPU / 1 GB | ~$18 / month |
| ALB | ~$18 / month |
| Kinesis: 2 streams × 1 shard, 24 h | ~$25 / month |
| Firehose + S3 audit | cents at demo volumes |
| CloudWatch (logs, 8 alarms, dashboard) | a few $ / month |
| GPU VM | billed by the hour by Hyperstack: hibernate it when idle |

`ops.py down` and hibernating the VM stop everything.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `check`: Fraud Model API unreachable | `pipeline.sh` done? `serve.sh <run>` running? TCP 8080 open in the Hyperstack firewall? IP current (Step 7)? `curl http://<VM_IP>:8080/v1/health` |
| `check`: the API rejects the token | `TF_VAR_scoring_token` must equal `ARGUS_SCORING_TOKEN` in the VM's `.env` (`remote.sh awsenv`) |
| `check`: FAIL VPC quota | raise "VPCs per Region" in Service Quotas |
| `check`: FAIL names already exist | a previous deployment lost its state: `ops.py audit`, remove the listed `argus-prod` items, or set `ENVIRONMENT=prod2` |
| Live stream page: "stream service is not reachable" | `ARGUS_STREAM_URL` set for the platform and the platform restarted? The VM's IP in `allowed_cidrs` (re-run `ops.py deploy` after an IP change)? |
| Every decision is `FAIL_OPEN` | the stream cannot reach the API: same fixes as the first row; the `scoring-unreachable` alarm says the same |
| Every decision is `AUTH_ERROR` | the API rejects the stream's token: `TF_VAR_scoring_token` = the VM's `ARGUS_SCORING_TOKEN`, then `ops.py deploy` |
| Synthetic transactions get HTTP 400 | the API runs a profile with the ±15 min window (`SCORING_CONFIG=configs/prod_a100.yaml`): `TF_VAR_transaction_time=now`, then `ops.py deploy` |
| `setup_vm.sh`: containers cannot see the GPU | the image's driver is too old or the toolkit failed: pick a newer NVIDIA image and re-run |
| `deploy.yml` deploy job skipped | `ops.py up` has not run yet (it sets `AWS_ROLE_ARN`) |
| A run is `interrupted` after the VM was hibernated | `train.sh <profile> --stages <remaining>` with the same run name (`--set run_name=...`) |
