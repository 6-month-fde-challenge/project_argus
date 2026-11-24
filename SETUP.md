# Setting up ARGUS from scratch

This guide takes a machine with nothing installed to a trained fraud model served behind the Fraud Model API, with
the UI, in numbered steps. Each step says what to run, how long it takes and how to check it worked. It is written for
anyone who has the repository and the project's Supabase credentials, whether or not they know the project.

- **What ARGUS is, how it is designed, every command and page:** [README.md](README.md).
- **Taking it to a GPU VM and AWS:** [deploy/README.md](deploy/README.md), after this guide (step 14).

Measured on the reference machine (Windows 11, Docker Desktop, 4 GB RTX 3050 Laptop GPU), 2026-11-24, unless stated.

---

## Contents

| Part | Steps |
|---|---|
| Prepare the machine | [0. Choose your path](#0-choose-your-path) · [1. Install the prerequisites](#1-install-the-prerequisites) · [2. Get the code](#2-get-the-code) · [3. Create `.env`](#3-create-env) |
| Run it locally | [4. Build](#4-build-the-images) · [5. Start the UI](#5-start-the-ui-stack) · [6. Download the dataset](#6-download-the-dataset) · [7. Train](#7-train-a-first-run) · [8. Serve](#8-serve-the-run) · [9. Try it](#9-try-it) |
| Optional | [10. Real-time stream](#10-optional-the-real-time-stream) · [11. Tests](#11-optional-run-the-tests) · [12. UI operations](#12-optional-let-the-ui-run-commands-the-operations-agent) · [13. AWS rehearsal](#13-optional-rehearse-the-aws-deployment-locally) |
| Beyond this machine | [14. GPU VM and AWS](#14-the-gpu-vm-and-aws) |
| Day to day | [15. Stop, restart, clean up](#15-stop-restart-and-clean-up) · [16. When something goes wrong](#16-when-something-goes-wrong) · [Appendix: Linux commands](#appendix-the-same-steps-on-linux-plain-docker-compose) |

**The short version** (Windows with an NVIDIA GPU, after steps 1–3):

```powershell
.\argus.ps1 build                      # step 4: ~20-30 min the first time
.\argus.ps1 up                         # step 5: UI at http://localhost:3000
.\argus.ps1 sync                       # step 6: the dataset
.\argus.ps1 train local_demo           # step 7: ~50 min on a 4 GB GPU
.\argus.ps1 serve argus_local_demo     # step 8: the Fraud Model API on :8080
.\argus.ps1 demo                       # step 9: every CRD / LLD scenario against it
```

---

## 0. Choose your path

| Your machine | What you can run | Follow |
|---|---|---|
| Windows 10/11 with an NVIDIA GPU (Ampere or newer, 4 GB or more) | everything: UI, training, serving, stream, tests | steps 1–16 with `.\argus.ps1` |
| Linux with an NVIDIA GPU | everything | steps 1–16, using the [Linux commands](#appendix-the-same-steps-on-linux-plain-docker-compose) |
| No NVIDIA GPU (a Mac, an AMD / Intel GPU) | the UI, platform and MLflow only; no training or serving | steps 1–5 (`argus.ps1` detects it and adds `docker-compose.nogpu.yml`) |
| A cloud GPU VM (Hyperstack or any Ubuntu GPU host) + AWS | the production layout | steps 1–3 here, then [deploy/README.md](deploy/README.md) |

A training run needs an NVIDIA GPU with bf16 support (Ampere, RTX 30-series or newer). Older GPUs fall back to fp32:
slower and more memory.

## 1. Install the prerequisites

| # | Install | Needed for | Check (expected result) |
|---|---|---|---|
| 1 | **Git** | getting the code | `git --version` prints a version |
| 2 | **Docker Desktop** (Windows / Mac, WSL 2 backend on Windows) or **Docker Engine + Compose v2** (Linux) | everything runs in containers | `docker compose version` prints v2.x |
| 3 | **NVIDIA driver** (Windows) / **driver + NVIDIA Container Toolkit** (Linux) | training and serving | `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` prints your GPU |
| 4 | **Python 3.10+** (standard library only, nothing to `pip install`) | optional tools on the host: the operations agent (step 12), the AWS rehearsal (step 13), AWS deployment | `python --version` (Linux: `python3 --version`) |
| 5 | **Free disk: about 80 GB** | images 53 GB (the vLLM explainer image alone is 35 GB), build cache ~13 GB, models + runs ~14 GB (measured) | Docker Desktop → Settings → Resources shows the disk image location |

On Windows, run every command in this guide in **PowerShell** from the repository folder. If PowerShell refuses to
run `argus.ps1` ("running scripts is disabled"), allow local scripts once for your user:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. Get the code

ARGUS is two repositories: this one (the solution) and the documents (client documents and write-ups). The UI's
Docs page reads the documents from the folder one level up, so clone them in this layout:

```
01_argus/            <- github.com/6-month-fde-challenge/argus           (documents)
  argus_solution/    <- github.com/6-month-fde-challenge/project_argus   (this repository)
```

```powershell
git clone https://github.com/6-month-fde-challenge/argus.git 01_argus
git clone https://github.com/6-month-fde-challenge/project_argus.git 01_argus/argus_solution
cd 01_argus/argus_solution
```

Without the documents repository everything still works; only the Docs page is empty.

**Check:** `git status` in `argus_solution` shows a clean working tree; `docker-compose.yml` and `argus.ps1` are
there.

## 3. Create `.env`

All machine-specific settings live in `.env` next to `docker-compose.yml`. It is git-ignored: never commit it.

```powershell
copy .env.example .env        # Linux: cp .env.example .env
notepad .env                  # Linux: nano .env
```

| Setting | Required? | Where to get it |
|---|---|---|
| `SUPABASE_URL` | **yes** | from the project owner: the Supabase project holding the `argus-data` bucket |
| `SUPABASE_KEY` | **yes** | from the project owner (read-only use: the code only downloads) |
| `HF_TOKEN` | only for the `prod_a100` profile | a Hugging Face token whose account accepted the Llama 3.2 and Gemma 2 licences. The local profiles use ungated models |
| everything else | no | leave commented out; the defaults work on a single machine. Each is explained in `.env.example`, and all of them can be changed later on the UI's Settings page |

Edit `.env` with an editor, not with PowerShell's `>>`: Windows PowerShell writes UTF-16, which `docker compose`
cannot read.

**Check:** `.env` exists and its two `SUPABASE_` lines have real values.

## 4. Build the images

```powershell
.\argus.ps1 build
```

Builds four images: the training image (CUDA, PyTorch; about 13 GB, most of the time), the scoring image (built on
the training image), the platform and UI, and the stream service. Every `argus.ps1` action prints the `docker compose`
command it runs.

- **Time:** about 20–30 minutes the first time, plus downloads. Later builds reuse every unchanged layer.
- **Not built here:** the vLLM explainer image (35 GB) is pulled the first time you serve (step 8).

**Check:** `.\argus.ps1 status` lists `argus-training`, `argus-scoring`, `argus-platform`, `argus-frontend` and
`argus-stream`.

## 5. Start the UI stack

```powershell
.\argus.ps1 up
```

| Opens | URL |
|---|---|
| the UI | http://localhost:3000 |
| the platform API (what the UI talks to) | http://localhost:8090/docs |
| MLflow (experiment tracking, model registry) | http://localhost:5000 |

Every port binds to 127.0.0.1 only: these pages are reachable from this machine, not the network. If a port is
already taken, set `ARGUS_UI_PORT` (and the others listed in `.env.example`) and run `up` again.

**Check:** http://localhost:3000 opens; **System** shows the GPU (or "no GPU" on a machine without one). Without an
NVIDIA GPU, stop here: the UI, platform and MLflow are what that machine can run.

## 6. Download the dataset

```powershell
.\argus.ps1 sync
```

Downloads the training files from the Supabase bucket into `training/data/raw/` and snapshots the feature, label,
transaction and customer tables into `training/data/db/`. Later runs of `sync` fetch only what changed.

UI alternative: **Operations → Sync the dataset** (needs the operations agent, step 12).

**Check:** the command ends with `Supabase sync -> data/raw: N downloaded, M up to date`, and `training/data/db/`
holds four `.parquet` files. A training run started without this step downloads the dataset itself first, but
running it here shows a credentials problem straight away.

## 7. Train a first run

Pick a profile. Each is a file in `training/configs/`.

| Profile | What it is | Time on a 4 GB RTX 3050 (measured, models already downloaded) |
|---|---|---|
| `local_demo` | **the one to start with**: the full ladder, sized for a 4 GB card, measured (its week-8 gate decides the branch) | 47 min |
| `smoke` | every stage with tiny sizes, to check the machine | a few minutes of training; the first run on a new machine also downloads the dataset and every model (101 min on the reference machine) |
| `demo_hybrid` | demo example 1: XGBoost scores, the LLM explains (branch forced) | 69 min |
| `demo_llm_scorer` | demo example 2: the LLM scorer; reuses `argus_demo_hybrid`, so train that first | 32 min |
| `prod_a100` | the production models; needs an A100 80GB and `HF_TOKEN` | GPU VM only ([deploy/README.md](deploy/README.md)) |

```powershell
.\argus.ps1 train local_demo
```

The first run downloads the models it uses (about 3 GB) into `training/.hf_cache/`. The run writes everything to
`training/outputs/argus_local_demo/`.

UI alternative: **Launch** → profile `local_demo` → **Start**. The run appears under **Runs** with its live ladder,
loss charts and logs. Command-line runs appear there live too.

**Check:** the command ends without an error; **Runs** shows `argus_local_demo` as *succeeded*; its page shows the
holdout verdict (GO / NO-GO) and `training/outputs/argus_local_demo/report.md` exists. A NO-GO verdict is a result,
not a failure: a 4 GB card trains small models, which do not reach the CRD targets.

## 8. Serve the run

```powershell
.\argus.ps1 serve argus_local_demo
```

Starts the **Fraud Model API** (`scoring`, port 8080) with the run's scorer, then the **vLLM explainer** (port 8000)
with the run's explainer.

- **First time:** pulls the 35 GB vLLM image. If the explainer cannot start (download cut off, not enough GPU memory),
  scoring still runs and explanations come from templates; run the same command again to retry.
- **One GPU, one owner:** serving and training cannot share the GPU. Stop serving before training again:
  `.\argus.ps1 unserve`.
- The served run is saved in `.env` (`RUN_NAME`), so a restart keeps serving it.

UI alternative: **System** → start scoring, or **Operations → Serve a run**. Switch the served run later without a
restart: **Scoring → Deployments**.

**Check:** http://localhost:8080/v1/health shows `"status": "ok"`, `"scorer": "ok"` and
`"active_run": "argus_local_demo"` (`"explainer": "ok"` once vLLM has loaded, which takes a minute or two).

## 9. Try it

```powershell
.\argus.ps1 demo
```

Runs the CRD / LLD scenarios against the served run, and writes the results to
`training/outputs/argus_local_demo/api/demo_results.json`. The scenarios are:
- the CRD worked example;
- known fraud and genuine payments;
- a blocked device and an allowlisted customer;
- invalid requests;
- a threshold change with two-person approval;
- the asynchronous explanation.

Then, in the UI:

| Page | Try |
|---|---|
| **Scoring** | send the ready-made samples; see the decision, risk level, reason codes and the explanation |
| **Runs** → `argus_local_demo` | the stage results, calibration, fairness, the GO / NO-GO checklist |
| **Rules & policy** | change a threshold (two names and a reason) and watch the change log |

Load test: `.\argus.ps1 loadtest 5 60` (5 requests a second for 60 s; results in the run's `api/loadtest.json`).

**Check:** `demo` prints each scenario with its HTTP status and response (the invalid requests answer 400, the rest
200), and ends with `wrote .../api/demo_results.json`. The Scoring page returns a decision for each sample.

## 10. Optional: the real-time stream

```powershell
.\argus.ps1 stream
```

Starts a local Kinesis emulator and the stream service: payments flow producer → Kinesis → Fraud Model API →
decision and action, as they will on AWS. Open **Live stream** → **Start traffic**; **Inject attack wave** sends a
fraud ring's burst. Stop with `.\argus.ps1 stream stop`.

**Check:** the Live stream page shows decisions arriving and the pipeline strip all green.

## 11. Optional: run the tests

```powershell
.\argus.ps1 test
```

Every test suite runs in containers (platform, training, scoring, stream, deploy scripts, frontend), the same suites
CI runs on every push. Serving can stay up.

**Check:** every suite reports `passed` with no failures (README section 19 lists what each suite covers).

## 12. Optional: let the UI run commands (the operations agent)

The **Operations** and **Settings** pages run host commands (serve, sync, build, apply settings) through a small agent
on this machine. The platform runs inside a container, so it cannot run them itself.

```powershell
.\argus.ps1 agent start       # runs in a minimized window; status: .\argus.ps1 agent status; stop: .\argus.ps1 agent stop
```

**Check:** the Operations page shows *agent on <this machine> (windows)*. Without the agent the UI still works:
actions wait in the queue, and each page shows the exact command to run by hand.

## 13. Optional: rehearse the AWS deployment locally

Before anyone deploys to AWS, the real Terraform stack can be applied to an AWS emulator on this machine, with no AWS
account. The command runs the stream as its Fargate task would, sends traffic through it and checks the result
(15 checks, about 5 minutes):

```powershell
.\argus.ps1 aws-local up      # needs a served run (step 8)
.\argus.ps1 aws-local down    # remove it
```

Details, the checks and what an emulator cannot show:
[deploy/README.md, section 4](deploy/README.md#4-rehearse-the-aws-stack-locally-optional-no-aws-account).

## 14. The GPU VM and AWS

The production layout trains and serves on a cloud GPU VM, and runs the real-time path on AWS (Kinesis, Fargate,
S3 audit, CloudWatch), deployed by GitHub Actions. [deploy/README.md](deploy/README.md) is the step-by-step runbook:

1. Configuration checklist, then the optional local rehearsal (step 13 here).
2. Create the VM, then run one command on it: `PROFILE=prod bash deploy/hyperstack/pipeline.sh`. It sets up, trains,
   serves and tests, unattended.
3. From your PC: `python deploy/ops.py check`, then `python deploy/ops.py up`.
4. Tick the first-deployment checklist.

## 15. Stop, restart and clean up

| To | Run | What is kept |
|---|---|---|
| stop serving (free the GPU) | `.\argus.ps1 unserve` | everything |
| stop everything | `.\argus.ps1 down` | images, data, models, runs, `.env` |
| start again later | `.\argus.ps1 up`, then `.\argus.ps1 serve` (serves the run saved in `.env`) | – |
| update to the latest code | `git pull`, then `.\argus.ps1 build` and `.\argus.ps1 up` | everything |

**Where state lives** (all git-ignored, all under this folder):

| Folder / file | Holds | Recreated by |
|---|---|---|
| `.env` | your settings and keys | step 3 (keep it: it cannot be regenerated) |
| `training/data/` | the dataset | `.\argus.ps1 sync` |
| `training/.hf_cache/` | downloaded models | the next training run |
| `training/outputs/<run>/` | each trained run | training it again |
| `training/outputs/mlruns/` | MLflow experiments and registry | training runs |
| `training/outputs/_platform/`, `_scoring/` | job history, operations log, change log; rules and audit log | the platform and API, on first use |

**Remove everything** (frees about 80 GB; afterwards, start again at step 4):

```powershell
.\argus.ps1 aws-local down                                                   # 1. the AWS rehearsal, if one ran
docker rm -f $(docker ps -aq --filter label=argus.job)                       # 2. finished training-job containers
docker compose --profile serve --profile stream --profile job down --volumes --rmi all   # 3. services, their volumes and images
docker rmi motoserver/moto:5.2.5 hashicorp/terraform:1.14.0                  # 4. the rehearsal's images
docker volume rm argus-tf-plugins                                            #    and its Terraform cache
docker rmi argus-platform:test argus-stream:test python:3.12.16-slim-bookworm   # 5. images built by `test`
docker builder prune --all --force                                           # 6. the build cache
```

An error on a line that has nothing to remove (`requires at least 1 argument`, `No such image`) is harmless.

Then delete `training/data`, `training/.hf_cache` and the run folders in `training/outputs` you no longer need. Keep
`.env`.

## 16. When something goes wrong

| Symptom | Fix |
|---|---|
| `argus.ps1 cannot be loaded ... running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `env file ... .env not found` | step 3: create `.env` next to `docker-compose.yml` |
| `could not select device driver "nvidia"` | no NVIDIA GPU visible to Docker: check step 1 #3. On a machine without one, use `argus.ps1` (it adds `docker-compose.nogpu.yml`) |
| `sync` or training fails with an authentication or DNS error from Supabase | the `SUPABASE_URL` / `SUPABASE_KEY` values in `.env`; a paused free-tier Supabase project must be restored by its owner |
| `Port ... is already allocated` | another program uses the port: set the matching `ARGUS_*_PORT` in `.env`, then `.\argus.ps1 up` |
| `no trained run '<name>' in training/outputs` on `serve` | train it first (step 7) or name a run that exists: the folders in `training/outputs/` |
| Training waits with "stop serving to free the GPU" | `.\argus.ps1 unserve`, then it starts by itself |
| Out of GPU memory during training | use `local_demo` or `smoke` on a small card; stop serving first |
| The explainer does not start | scoring still works (template explanations); `.\argus.ps1 logs explainer`, then `serve <run>` again |
| An Operations action stays *queued* | the agent is not running: step 12, or run the command the page shows |

More: [README.md section 20](README.md#20-troubleshooting), and for the VM and AWS
[deploy/README.md section 10](deploy/README.md#10-troubleshooting). Logs of any service: `.\argus.ps1 logs <service>`
(`platform`, `scoring`, `explainer`, `stream`, ...) or `.\argus.ps1 logs <job id>` for a training job.

---

## Appendix: the same steps on Linux (plain `docker compose`)

`argus.ps1` only wraps these commands. Run them from the repository folder.

| Step | Command |
|---|---|
| 4. Build | `docker compose build trainer && docker compose --profile serve build scoring && docker compose build platform frontend && docker compose --profile stream build stream` |
| 5. Start the UI | `docker compose up -d` |
| 6. Dataset | `docker compose run --rm --no-deps trainer python -m argus_ft.supabase_data` |
| 7. Train | `docker compose run --rm trainer python -m argus_ft.pipeline --config configs/local_demo.yaml --no-sync` |
| 8. Serve | put `RUN_NAME=argus_local_demo` and `EXPLAINER_RUN=argus_local_demo` in `.env`, then `docker compose --profile serve up -d --force-recreate scoring explainer` |
| 9. Try it | `docker compose --profile serve exec scoring python -m argus_api.demo --run argus_local_demo --url http://localhost:8080` |
| 10. Stream | `docker compose --profile stream up -d kinesis stream` |
| 12. Agent | `python3 deploy/agent.py` (the GPU VM installs it as the systemd service `argus-agent`) |
| 13. AWS rehearsal | `python3 deploy/local/rehearse.py up` |
| 15. Stop | `docker compose --profile serve --profile stream --profile job down` |

Without an NVIDIA GPU on Linux: `export COMPOSE_FILE=docker-compose.yml:docker-compose.nogpu.yml` before step 5.

On a GPU VM, use the scripts in `deploy/hyperstack/` instead. They also install Docker and the NVIDIA toolkit,
generate tokens and set up the agent: [deploy/README.md](deploy/README.md).
