# ARGUS deployment: GPU VM (Hyperstack) + AWS (ap-south-1)

The main [README](../README.md) runs everything on one machine with Docker ([SETUP.md](../SETUP.md): from scratch,
step by step). This folder takes the same solution to real infrastructure:

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

| Part | Sections |
|---|---|
| Understand it | [1. What runs where](#1-what-runs-where) · [2. Isolation](#2-isolation-what-this-touches-in-the-aws-account) |
| Prepare | [3. Configuration checklist](#3-configuration-checklist) · [4. Rehearse locally (optional)](#4-rehearse-the-aws-stack-locally-optional-no-aws-account) |
| Deploy and operate | [5. Runbook](#5-runbook-in-this-order) · [6. Script reference](#6-script-reference) · [7. Data, audit and monitoring](#7-data-audit-and-monitoring) |
| Reference | [8. Security](#8-security) · [9. Cost](#9-cost) · [10. Troubleshooting](#10-troubleshooting) |

**The shortest path:** [3](#3-configuration-checklist) (settings) → [4](#4-rehearse-the-aws-stack-locally-optional-no-aws-account)
(optional rehearsal on your PC) → [5](#5-runbook-in-this-order), Steps 1–5 (VM, then AWS) → tick the
[first-deployment checklist](#first-real-deployment-what-only-the-vm-and-aws-can-confirm).

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

### First real deployment: what only the VM and AWS can confirm

Everything else is covered by the test suites and CI (including the GPU VM scripts, run under bash with stubbed
tools) and by the local AWS rehearsal ([section 4](#4-rehearse-the-aws-stack-locally-optional-no-aws-account)).
These are the facts of the real environment; tick them once:

| # | Check | Why |
|---|---|---|
| 1 | Run `setup_vm.sh` / `pipeline.sh` as the normal VM user (`ubuntu`), **not** with `sudo` | the operations agent runs as that user and every file the platform writes is given to the repository's owner; under `sudo` both become root |
| 2 | `sudo -n true` works on the VM (passwordless sudo, the Hyperstack default) | the agent runs `setup_vm.sh` from the UI without a terminal to type a password in |
| 3 | `PROFILE=prod` (or `ARGUS_PROFILE=prod` saved) on the VM that AWS calls | only `prod_a100` is production mode; `ops.py check` / `deploy` refuse a demo setup |
| 4 | Hyperstack firewall: inbound TCP 22 and the scoring port (8080) | AWS Fargate calls the Fraud Model API; everything else goes through the SSH tunnel |
| 5 | `sudo systemctl status argus-agent` is active, and the UI's Operations page shows *agent on <vm> (linux)* | otherwise UI actions wait in the queue (the shown commands still work by hand) |
| 6 | `python deploy/ops.py check` ends with "ready to deploy" from your PC | tools, AWS identity, the API's token **and serving mode**, isolation |
| 7 | Confirm the SNS e-mail subscription AWS sends to `TF_VAR_alert_email` | alarms (fail-open, demo-model-serving, latency, ...) only arrive after the confirmation |
| 8 | For the UI's AWS page only: `ARGUS_GITHUB_TOKEN` (fine-grained, this repository: Actions, Variables, Secrets read/write) | deploy / destroy from the UI run the repository's workflows |
| 9 | After `ops.py up` and a few minutes of traffic: the Live stream page shows decisions; the CloudWatch dashboard `argus-prod` has data; no alarm is in ALARM; `s3://argus-prod-audit-*/decisions/` has objects | the parts the local rehearsal cannot emulate: Fargate, the load balancer, IAM enforcement, Firehose delivery, alarms and dashboard |

## 4. Rehearse the AWS stack locally (optional, no AWS account)

Runs the real Terraform stack against an AWS emulator on your PC, starts the stream the way its Fargate task would
start, sends traffic through it and checks the result. Use it before the first `ops.py up`, and after changing
`deploy/terraform` or the AWS calls in `stream/`. It costs nothing and touches no AWS account.

### 4.1 Before you start

| # | Requirement | How |
|---|---|---|
| 1 | Docker is running | Docker Desktop (Windows) or the Docker engine (VM) |
| 2 | The stream image exists | `docker compose --profile stream build stream` |
| 3 | A run is served (needed for every check to pass) | `.\argus.ps1 serve <run>` (VM: `bash deploy/hyperstack/serve.sh <run>`) |
| 4 | Ports 4599 and 8013 are free on 127.0.0.1 | used by the emulator and the rehearsal's stream task |

The first run downloads `motoserver/moto:5.2.5`, `hashicorp/terraform:1.14.0` and the Terraform AWS provider (cached
in the Docker volume `argus-tf-plugins`, so later runs are faster).

### 4.2 Run it

| Command | Does | Time |
|---|---|---|
| `.\argus.ps1 aws-local up` | build the rehearsal and run every check | about 5 min |
| `.\argus.ps1 aws-local verify` | run the checks again (the rehearsal keeps running after `up`) | under 1 min |
| `.\argus.ps1 aws-local down` | remove everything the rehearsal created | seconds |

The same three ways as everything else:

- **PowerShell:** `.\argus.ps1 aws-local up|verify|down`.
- **Python, any OS:** `python deploy/local/rehearse.py up|verify|down`. Use `python3` on the VM.
- **UI:** Operations → *Rehearse the AWS stack locally*, then choose `up`, `verify` or `down`.

### 4.3 What `up` does

1. **Emulator.** Starts [Moto](https://github.com/getmoto/moto), an open-source AWS emulator, as the container
   `argus-aws-local`, reachable as `aws` on the compose network.
2. **Terraform.** Copies `deploy/terraform` and `deploy/local/moto_override.tf` into the Docker volume
   `argus-aws-local`, then runs `init`, `plan` and `apply` there.
   - The override sends every AWS endpoint to the emulator and keeps the Terraform state inside the volume.
   - `deploy/terraform` and the repository are not modified.
   - Terraform warns that `-target` is in effect. That is expected: it is how the CloudWatch alarms and the dashboard
     are left out, because the emulator cannot create them.
3. **Stream task.** Runs `argus-stream:latest` (the same Dockerfile as the image pushed to ECR) as
   `argus-aws-stream`. It gets the environment of the ECS task definition Terraform created, and the secrets read
   from SSM, as ECS would inject them. It calls the Fraud Model API at `http://scoring:8080`.
4. **Traffic.** About 75 seconds of it, so the CloudWatch metrics are flushed at least once:
   - the labelled holdout, uploaded to the S3 data bucket and replayed from there;
   - synthetic payments;
   - one attack wave (card testing);
   - one payment from a blocklisted device, so a BLOCK goes out on SNS.
5. **Checks.** Prints PASS, FAIL or n/a for each check, and exits with 1 if any check fails.

### 4.4 The checks

| Check | Passes when |
|---|---|
| Terraform applied the whole stack | every planned resource exists, except the CloudWatch alarms and dashboard |
| Kinesis transactions stream / decisions stream | both streams are `ACTIVE` |
| Firehose: decisions stream → S3 audit bucket | the delivery stream reads the decisions stream and writes to the audit bucket |
| Secrets from SSM | the scoring token and control key (≥ 32 characters) resolve from SSM |
| Stream task consumes Kinesis | the task is connected and has consumed transactions |
| Stream errors / dead letters | 0 errors and 0 dead letters |
| Fraud Model API reached from the task | the API answered `ok` (n/a if the stream reports no API status) |
| Holdout replay from the S3 data bucket | holdout transactions were decided (n/a if the task does not replay the holdout) |
| Decisions published to the decisions stream | at least one decision per consumed transaction |
| BLOCK alert published to SNS | the blocklisted payment's `BLOCK_AND_ALERT` reached a test queue subscribed to the actions topic |
| Stream metrics in CloudWatch | `Transactions`, `FailOpen` and `DemoModelServing` were published |
| Every stream alarm watches a metric the stream publishes | each alarm Terraform defines on the stream's namespace names a metric the stream really sends |
| DemoModelServing reflects what the API serves | the metric has datapoints (1 = a demo setup is behind the traffic, 0 = production) |
| Task role allows every AWS call the stream makes | the task role's policy allows each call the stream code makes |

### 4.5 What the emulator cannot show

Confirm these on the first real deployment ([checklist row 9](#first-real-deployment-what-only-the-vm-and-aws-can-confirm)):

| Not emulated | Checked locally instead |
|---|---|
| IAM *enforcement* (Moto accepts every call) | the task role's policy lists every call the stream makes |
| VPC networking, security groups, the load balancer | – |
| Fargate itself (scheduling, health checks, rollback) | the stream runs as a container, with the task definition's environment |
| Firehose delivery into S3 | the delivery stream's source and destination |
| CloudWatch alarms and dashboard (Moto does not support the CloudWatch protocol the Terraform AWS provider uses) | every alarm's metric against the metrics the stream publishes |

### 4.6 In CI

Part of the rehearsal needs no Docker, so it runs on every push (`deploy/tests/test_rehearse.py`). Each of these
fails the build:

- the stream calls an AWS API that the rehearsal's list or the task role in `ecs.tf` does not cover;
- a Terraform resource type has no emulator endpoint in `moto_override.tf`, so a rehearsal would call real AWS;
- the Operations action accepts anything other than `up`, `verify` or `down`.

### 4.7 What it leaves running, and cleanup

After `up`, the rehearsal keeps running until `down`, so you can look around:

| Item | What it is | Look at |
|---|---|---|
| container `argus-aws-local` | the AWS emulator | http://127.0.0.1:4599/moto-api/ |
| container `argus-aws-stream` | the stream task | http://127.0.0.1:8013/live/docs |
| volume `argus-aws-local` | Terraform working copy, state and plan | – |
| volume `argus-tf-plugins` | Terraform provider cache (kept by `down`) | `docker volume rm argus-tf-plugins` to free it |

`down` removes both containers and the `argus-aws-local` volume. Nothing is written to the repository at any point.

## 5. Runbook (in this order)

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
PROFILE=prod bash deploy/hyperstack/pipeline.sh              # always name the profile (prod | demo | smoke); tmux
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

There is **no default profile**: the scripts take `PROFILE` from the command line, else `ARGUS_PROFILE` in `.env`
(the first `setup_vm.sh` saves it; the UI's Settings page changes it), else they stop and ask. A silent `demo` default
could put a forced demo model behind live AWS traffic. `PROFILE=prod` also switches the API to production mode
(`SCORING_CONFIG=configs/prod_a100.yaml`: demo runs refused); a demo profile never loosens a production setting.

| Variant | Command |
|---|---|
| production profile, one A100 80GB (the only one for live traffic) | `PROFILE=prod bash deploy/hyperstack/pipeline.sh` |
| demo profile (forced branches: demonstrations only) | `PROFILE=demo bash deploy/hyperstack/pipeline.sh` |
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

## 6. Script reference

### GPU VM: `deploy/hyperstack/`

| Script | Runs on | Does |
|---|---|---|
| `setup_vm.sh` | VM | Docker + NVIDIA Container Toolkit, `.env` secrets and bindings, all images, data, model weights, the operations agent service (`PROFILE=prod\|demo\|smoke`) |
| `check.sh [--full]` | VM | host, `.env`, data, images, runs (with holdout verdicts), services; `--full`: CUDA in the training image, gated-model access, test suites |
| `train.sh <profile> [args]` | VM | one ladder run in tmux `train` (stops serving first); args go to `argus_ft.pipeline` (`--stages`, `--set`) |
| `serve.sh [run\|stop]` | VM | UI + platform + MLflow + Fraud Model API + vLLM for a run (newest servable by default); waits until healthy |
| `pipeline.sh` | VM | everything above, unattended in tmux `pipeline`, then `aws.env` |
| `apply.sh <service>...` | VM | after a `.env` change: recreates those services (serving / stream ones only if running) |
| `deploy/agent.py` | VM (systemd `argus-agent`) / Windows (`.\argus.ps1 agent start`) | the operations agent: runs what the UI's Operations / Settings pages ask for, only from `deploy/ops_actions.json`, with validated parameters; logs in `training/outputs/_platform/ops/` |
| `remote.sh <cmd>` | your PC | `push`, `setup`, `pipeline`, `train`, `logs`, `status`, `tunnel`, `pull`, `awsenv`, `ssh` over SSH |

Logs: `training/outputs/_logs/{pipeline,train}.log`. Every training run also writes `events.jsonl`, so the UI follows
CLI runs live.

**The same from the UI.** Every script above is also an action on the UI's **Operations** page (through the agent,
with live log and cancel), and every `.env` setting is on the **Settings** page; the page shows the exact command, so
both ways always match. The first `setup_vm.sh` must run by hand: it installs Docker and the agent the UI relies on.
The agent needs `sudo` without a password for the steps that install packages (the default on Hyperstack images).

### AWS: `deploy/ops.py` (also `.\argus.ps1 aws <command>`)

| Command | Does |
|---|---|
| `check` | read-only: tools, AWS identity, Fraud Model API health + token + serving mode, allow-list, isolation |
| `up` | `check` → `bootstrap` → `deploy` |
| `--allow-demo` (on `check` / `up` / `deploy`) | accept a Fraud Model API that serves a demo setup (otherwise refused: live traffic must never meet a demo model). CI: repository variable `ALLOW_DEMO=true`, or the deploy workflow's `allow_demo` input for one run |
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

### Local rehearsal: `deploy/local/` (also `.\argus.ps1 aws-local <command>`; [section 4](#4-rehearse-the-aws-stack-locally-optional-no-aws-account))

| File | Runs on | Does |
|---|---|---|
| `rehearse.py up\|verify\|down` | your PC or the VM (standard library + the docker CLI) | starts the emulator, applies Terraform, runs the stream task, sends traffic, runs the checks; `down` removes it all |
| `inside.py` | inside the stream image, started by `rehearse.py` | talks to the emulator with boto3: reads the task definition and SSM secrets, uploads the holdout, sends the blocklisted payment, runs the checks |
| `moto_override.tf` | copied next to the Terraform copy in the volume | sends every AWS endpoint to the emulator; keeps the state in the volume |

## 7. Data, audit and monitoring

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

## 8. Security

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

## 9. Cost

| Item | Cost (approximate, if left running) |
|---|---|
| Fargate 0.5 vCPU / 1 GB | ~$18 / month |
| ALB | ~$18 / month |
| Kinesis: 2 streams × 1 shard, 24 h | ~$25 / month |
| Firehose + S3 audit | cents at demo volumes |
| CloudWatch (logs, 8 alarms, dashboard) | a few $ / month |
| GPU VM | billed by the hour by Hyperstack: hibernate it when idle |

`ops.py down` and hibernating the VM stop everything.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `check`: Fraud Model API unreachable | `pipeline.sh` done? `serve.sh <run>` running? TCP 8080 open in the Hyperstack firewall? IP current (Step 7)? `curl http://<VM_IP>:8080/v1/health` |
| `check`: the API rejects the token | `TF_VAR_scoring_token` must equal `ARGUS_SCORING_TOKEN` in the VM's `.env` (`remote.sh awsenv`) |
| `check`: FAIL VPC quota | raise "VPCs per Region" in Service Quotas |
| `check`: FAIL names already exist | a previous deployment lost its state: `ops.py audit`, remove the listed `argus-prod` items, or set `ENVIRONMENT=prod2` |
| Live stream page: "stream service is not reachable" | `ARGUS_STREAM_URL` set for the platform and the platform restarted? The VM's IP in `allowed_cidrs` (re-run `ops.py deploy` after an IP change)? |
| Every decision is `FAIL_OPEN` | the stream cannot reach the API: same fixes as the first row; the `scoring-unreachable` alarm says the same |
| `check` / `deploy` refused: "runs in demo mode" or "is a demo run" | the VM serves a demo setup: `PROFILE=prod bash deploy/hyperstack/pipeline.sh` (or Settings → Serving profile `prod_a100`, then Operations → Serve a measured run). Only for a deliberate demo: `--allow-demo` |
| Alarm `demo-model-serving` / red banner on the Live stream page | someone switched the API behind live traffic to a demo setup: same fix as the row above, then the alarm clears within two minutes |
| Every decision is `AUTH_ERROR` | the API rejects the stream's token: `TF_VAR_scoring_token` = the VM's `ARGUS_SCORING_TOKEN`, then `ops.py deploy` |
| Synthetic transactions get HTTP 400 | the API runs a profile with the ±15 min window (`SCORING_CONFIG=configs/prod_a100.yaml`): `TF_VAR_transaction_time=now`, then `ops.py deploy` |
| `setup_vm.sh`: containers cannot see the GPU | the image's driver is too old or the toolkit failed: pick a newer NVIDIA image and re-run |
| `deploy.yml` deploy job skipped | `ops.py up` has not run yet (it sets `AWS_ROLE_ARN`) |
| A run is `interrupted` after the VM was hibernated | `train.sh <profile> --stages <remaining>` with the same run name (`--set run_name=...`) |
| Rehearsal: `argus-stream:latest is missing` | `docker compose --profile stream build stream`, then `aws-local up` again |
| Rehearsal: FAIL *Fraud Model API reached from the task* (every decision `FAIL_OPEN`) | no run is served: `.\argus.ps1 serve <run>`, then `aws-local up` again |
| Rehearsal: `port is already allocated` (4599 or 8013) | another program uses the port (`up` already replaces an earlier rehearsal): stop it, then `aws-local up` |
| Rehearsal: FAIL *Task role allows every AWS call* | the stream makes a call the task role in `deploy/terraform/ecs.tf` does not allow: add the action there (and to `CALLS_MADE` in `deploy/local/inside.py`) |
| Rehearsal: FAIL *Terraform applied the whole stack* | read the `terraform apply` error above the checks; a new resource type may need its endpoint in `deploy/local/moto_override.tf` (CI says which) |
