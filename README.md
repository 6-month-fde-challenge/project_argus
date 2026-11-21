# Project ARGUS — solution

ARGUS is VKPayments' own fraud-intelligence LLM. It reads a card or UPI transaction in context and returns a fraud
probability, an ALLOW / REVIEW / BLOCK decision, reason codes and a plain-language explanation.
This folder holds the complete working solution:

- **the model training ladder** from Part 3 (Figure 4): data checks, XGBoost baseline, pilot, week-8 gate, CPT,
  SFT, DPO, distillation, quantisation, calibration and registry;
- **the Fraud Model API** from the CRD (Section 10) and the LLD: scoring, rules engine, explanations, audit;
- **a web control plane** to launch training, watch it live, compare runs, browse MLflow and test the API.

**Everything runs in Docker.** Nothing is installed on the host except Docker Desktop and the NVIDIA driver.

### Contents

| Part | Sections |
|---|---|
| What was asked | [1. The client brief](#1-the-client-brief) · [2. Requirements coverage](#2-requirements-coverage) |
| Design | [3. System design](#3-system-design) · [4. Training ladder](#4-training-ladder-design) · [5. Data and features](#5-data-and-features) · [6. Scoring path](#6-scoring-path-design) · [7. Control plane and UI](#7-control-plane-and-ui-design) |
| API design | [8. Fraud Model API](#8-api-design-fraud-model-api-scoring-8080) · [9. Platform API](#9-api-design-platform-api-8090apiv1) · [10. File contracts](#10-file-contracts) |
| Running it | [11. Prerequisites](#11-prerequisites) · [12. Quick start](#12-quick-start) · [13. Services](#13-services-and-ports) · [14. UI](#14-the-ui) · [15. Training](#15-training) · [16. Serving](#16-serving-and-api-testing) |
| Operating it | [17. Logs](#17-logs-where-to-look) · [18. Where to change things](#18-where-to-change-things) · [19. Tests](#19-tests) · [20. Troubleshooting](#20-troubleshooting) |
| Results and plan | [21. Results](#21-results-so-far) · [22. Local vs production](#22-local-demo-vs-production) · [23. Plan](#23-plan) · [24. Risks](#24-risks-and-mitigations) · [25. Folder layout](#25-folder-layout) · [26. Demo video](#26-demo-video) |

The client documents are in `01_argus/client_docs/`: CRD, Architecture, Capacity Planning, HLD and LLD.
The delivery team's own write-ups are in `01_argus/day_wise/part_01..05`: client understanding, technical
requirements, proposed solution with diagrams, client questions, mock meeting.
Section numbers below such as "CRD 10.2" or "LLD 4.1" point into those documents.

---

## 1. The client brief

**Client.** VKPayments Pvt. Ltd. is an RBI-authorised payment aggregator and gateway:

- 1.8 lakh merchants and 4.2 million transactions a day;
- 2,800 TPS at festive peaks;
- a target of 8 million transactions a day by March 2028.

**Problem** (CRD 3):

- **Losses doubled.** Fraud losses grew from 4.1 to 9.8 basis points in 18 months.
- **Too many good customers stopped.** A static engine of about 340 hand-written rules declines or reviews
  2.3% of transactions, and only 1 in 19 of those is fraud.
- **The review team can't keep up.** It handles 22,000 cases a day.
- **No explanations.** Support staff can't tell customers why a payment was blocked.

**Why an LLM of their own** (CRD 3.4):

- **Context, not columns.** It reads the transaction plus the customer's recent history the way an investigator
  does.
- **Plain-language explanations.**
- **Learning from unstructured text:** case notes, chargeback narratives and playbooks.
- **Data sovereignty.** PCI DSS and the DPDP Act forbid sending transaction data to third-party AI APIs, so the
  model must be self-hosted and trained in-house.

**Rules stay.** They sit after the model as a policy layer and as a safety net.

**Success criteria** (CRD 4):

| ID | Target | ID | Target |
|---|---|---|---|
| OBJ-01 | fraud recall ≥ 85% | OBJ-07 | fraud loss ≤ 6.4 bps (pilot) |
| OBJ-02 | precision at BLOCK ≥ 70% | OBJ-08 | false declines ≤ 1.4% |
| OBJ-03 | PR-AUC ≥ 0.80 | OBJ-09 | ≤ 12,000 REVIEW cases a day |
| OBJ-04 | PR-AUC uplift over XGBoost ≥ +0.03 | OBJ-10 | p99 latency ≤ 250 ms |
| OBJ-05 | ≥ 90% explanations factually correct | OBJ-11 | uptime ≥ 99.95% |
| OBJ-06 | 100% valid-JSON responses | OBJ-12 | ≤ ₹4.00 per 1,000 transactions |

**Hard constraints:**

- **Data and compute.** No data, prompts or weights leave VKPayments' AWS account. No third-party hosted AI API at
  any stage. Training is capped at 2,500 A100-hours, and serving at 8 inference GPUs (L4 / A10G).
- **Timeline.** The engagement runs 20 weeks, with a hard go/no-go in week 8. If the pilot LLM can't beat XGBoost,
  the fallback is a **hybrid**: XGBoost scores and the LLM only explains.

**Scope of this solution.** It is the delivery team's working build of that system. It covers the full offline
ladder and the online scoring path, sized to run on a 4 GB laptop GPU. A production profile for one A100 80GB uses
the models named in Part 3. AWS infrastructure (EKS, Redis, Aurora, Terraform, mTLS) is out of scope for this
build; section 22 maps each local stand-in to its production counterpart.

## 2. Requirements coverage

| Requirement | How this build meets it | Where |
|---|---|---|
| FR-ML-01 XGBoost baseline first | stage 1 trains it on the engineered DB features; every later model is compared to it | `training/argus_ft/stages/xgboost_baseline.py` |
| CRD 9.1 base model selection (≥ 3 models) | stage 2 pilots 3 model families × 2 prompt formats with the same LoRA budget and ranks them on validation PR-AUC | `stages/pilot.py`, `pilot.candidates` |
| FR-FE-07 two serialisation formats | `fmt-v1` (CRD 8.2) and `fmt-v2` (adds a `[SUMMARY]` line) are both piloted | `argus_ft/data.py` |
| Week-8 go/no-go, hybrid fallback | the `gate` stage measures PR-AUC uplift against `gate.min_uplift: 0.03` (OBJ-04); "no" skips the LLM scorer stages and XGBoost scores | `stages/gate.py` |
| CRD 9.1 CPT, SFT, DPO, distillation, quantisation | stages 3 to 7, LoRA / QLoRA throughout | `stages/*.py` |
| FR-ML-03 probability from a classification head | the SFT scorer has a fraud head plus 8 reason-code heads on the final hidden state; no free-text parsing | `stages/common.py`, `MultiHeadTrainer` |
| FR-ML-04 class imbalance | genuine rows down-sampled (`neg_per_pos`) **and** per-head `pos_weight` in the BCE loss | `stages/common.py` |
| FR-ML-05 time-based split | train Jan–Oct 2025, validate Nov–Dec 2025, holdout Jan–Feb 2026; stage 0 fails the run if splits overlap | `stages/data_checks.py` |
| FR-ML-06 calibration | stage 8 fits Platt and isotonic per scorer and keeps the lower ECE | `stages/calibrate.py` |
| FR-ML-07 grounded, schema-constrained explanations | JSON schema + grounding guard (every number must appear in the input) + reason-code consistency check, with a reason-code template as the fallback | `scoring_api/argus_api/explainer.py`, `argus_ft/grounding.py` |
| FR-ML-08 version every checkpoint | the registry stage logs data snapshot, prompt format, config, metrics and models to MLflow | `stages/registry.py` |
| FR-ML-11 fairness by age band and region | stage 8 reports error rates per band and region and flags gaps > 20% relative | `stages/calibrate.py` |
| FR-ML-12 threshold analysis | stage 8: precision, recall and review volume at the policy cut-offs and at "top 12,000 a day" (OBJ-09) | `report.md` → calibrate |
| CRD 9.3 reason codes R01–R08 | a code is returned only when its feature flag is true (and, for the LLM, its head ≥ 0.5); at most 3 | `scoring_api/argus_api/scorer.py` |
| CRD 10.1–10.3 API contract | `POST /v1/fraud/score`, request and response field for field | section 8 |
| CRD 10.4 decision policy | 0.30 / 0.60 / 0.85 → LOW / MEDIUM / HIGH / CRITICAL → ALLOW / ALLOW / REVIEW / BLOCK | `scoring_api/argus_api/rules.py` |
| FR-RE-01 configurable thresholds, logged | admin API + UI, two-person approval, append-only change log, hot reload | section 8.4 |
| FR-RE-02 hard rules override the model | blocked device → BLOCK with `decision_source: HARD_RULE`, before any model I/O | `rules.hard_match` |
| FR-RE-03 allowlist capped at REVIEW | allowlisted customers never get BLOCK | `rules.decide` |
| FR-RE-04 per-MCC thresholds | `mcc_thresholds` overrides, e.g. stricter for electronics | `rules.thresholds` |
| FR-RE-05 model vs hard rule | `decision_source` in every response | response schema |
| FR-API-02 400 / 503 | validation → 400 with the field; no model or a scorer error → 503 | `app.py` |
| FR-API-03 audit everything | every request, response, feature vector, prompt, versions and explanation → `audit.jsonl` | `scoring_api/argus_api/audit.py` |
| FR-API-04 health endpoint | `GET /v1/health` | section 8 |
| FR-API-05 probability never waits for the explanation | REVIEW/BLOCK wait at most `explain_cutoff_ms`; ALLOW and late explanations are written to the audit log asynchronously (≤ 5 s) | `app.py` |
| FR-API-06 schema failure → reason codes only | template explanation, `explanation_source: SCHEMA_FAILED`, logged | `explainer.py` |
| NFR-04 optimised inference server | the explainer runs on vLLM with dynamic batching | `docker-compose.yml` → `explainer` |
| CRD 13 prompt-injection robustness | free text (city) is NFKC-normalised, reduced to `[A-Za-z0-9 .,-]` and capped at 32 characters before it reaches a prompt (LLD 6.2) | `schemas.sanitise` |
| MLOPS-07 fast rollback | switch the served run at runtime: `POST /admin/v1/models/{run}/activate` (no restart) | `scoring_api/argus_api/deployments.py` |
| MLOPS-10 XGBoost kept warm | every run keeps its XGBoost model; a HYBRID run serves it | stage 1 |

## 3. System design

```
 browser :3000 ──► frontend (React SPA, nginx) ──/api──► platform :8090 (FastAPI control plane)
                                                            │  docker socket: launches / stops containers
                     ┌──────────────────────────────────────┼─────────────────────────┐
                     ▼                                      ▼                         ▼
          argus-job-<id> (training, GPU)        scoring :8080 (Fraud Model API) ─► explainer :8000 (vLLM, AWQ)
          writes training/outputs/<run>/        serves one finished run              (profile "serve")
                     │
                     └──► mlflow :5000 (tracking + model registry, SQLite under training/outputs/mlruns)
```

**Two halves, as in CRD 11:**

- **Offline.** A training job container runs the ladder, one process per stage. It writes models, reports and a
  live event stream to `training/outputs/<run>/`, then registers the result in MLflow.
- **Online.** The scoring service loads one finished run (scorer + calibrator), applies the rules engine, and calls
  the vLLM explainer for the explanation.

**The control plane** sits beside both. It reads the run folders and MLflow, launches and cancels training
containers through the Docker socket, and proxies the scoring API for the UI. Containers share only files on disk:
`training/outputs/` is bind-mounted into every service. No database server is needed for the demo.

**Design decisions:**

| Decision | Why |
|---|---|
| One container per training run, one process per stage | A stage crash or OOM can't poison the next stage's GPU memory, and a failed run can resume from any stage (`--stages`). |
| Events as an append-only JSON-lines file | Training needs no network dependency on the platform; CLI runs and UI runs look identical; the UI resumes after reconnect by byte offset. |
| The platform launches jobs through the Docker socket | The UI can start real training without a job server; the platform reuses its own container's mounts as the host paths for the job. |
| One GPU owner at a time | A 4 GB card can't hold training and serving together. A FIFO job queue holds jobs while serving runs; starting serving while a job runs is refused. |
| Scorer and explainer are separate models | The scorer must be fast and calibrated (classification heads, FP8). The explainer can be slower and asynchronous (generation, AWQ 4-bit on vLLM). This mirrors Part 3's "0.5B scorer + 1.5B explainer". |
| Profiles `extends:` a single `base.yaml` | Every setting has one default; a profile lists only what differs; per-run overrides are dotted `key=value`. |
| Feature modules on both backend and frontend | A new page or endpoint is one folder plus one registry line (section 18). |

## 4. Training ladder design

```
0 data checks → 1 XGBoost → 2 pilot (3 models × 2 prompt formats) → week-8 gate ──no──► HYBRID: XGBoost scores
                                                                          │ yes
3 CPT → 4 SFT (4a scorer heads, 4b explainer) → 5 DPO → 6 distil (6a scorer, 6b explainer)
      → 7 quantise (FP8 scorer, AWQ 4-bit explainer) → 8 calibrate → registry (MLflow)
```

On "no", the gate skips only the LLM *scorer* stages (4a, 6a). The explainer branch still runs, because in the
hybrid the LLM explains.

| # | Stage | What it does | Method (local demo → A100 profile) | Output |
|---|---|---|---|---|
| 0 | `data_checks` | Leakage tests: validation before holdout in time, and no training set (SFT, gold, DPO, CPT) contains a validation or holdout transaction (DR-02). Data-quality report. A failed check fails the run. | – | `data_checks/` |
| 1 | `xgboost` | Baseline on the DB features (FE-01..FE-10); the bar to beat (FR-ML-01). | XGBoost 400 trees, depth 6 | `xgboost/` |
| 2 | `pilot` | Each candidate × each prompt format gets the same short LoRA run; ranked on validation PR-AUC. The winner becomes the teacher and the explainer base. | Qwen2.5-0.5B, SmolLM2-360M, h2o-danube3-500m → Qwen2.5-7B, Llama-3.2-3B, Gemma-2-2B | `pilot/` |
| – | `gate` | Week 8: pilot winner PR-AUC − XGBoost PR-AUC ≥ 0.03? `mode: auto` follows the measurement; `hybrid` / `llm` force a branch for demos. | – | `gate.json` |
| 3 | `cpt` | Continued pre-training on the fraud text corpus (case notes, chargeback narratives, playbooks, circulars). | LoRA, causal LM | `cpt/` |
| 4a | `sft_scorer` | Teacher scorer: fraud head + 8 reason heads, weighted BCE. | QLoRA 4-bit → LoRA | `teacher_scorer/` |
| 4b | `sft_explainer` | Teacher explainer: JSON `{reason_codes, explanation}` on the gold cases. | LoRA | `teacher_explainer/` |
| 5 | `dpo` | Preference tuning of the explainer on reviewer pairs (better vs worse explanation). | DPO β 0.1 + RPO NLL term | `teacher_explainer_dpo/` |
| 6a | `distill_scorer` | Small scorer student: α · hard labels + (1−α) · teacher soft targets at T = 2. | SmolLM2-135M → Qwen2.5-0.5B | `scorer_student/` |
| 6b | `distill_explainer` | Small explainer student trained on the DPO teacher's answers. | SmolLM2-135M-Instruct → Qwen2.5-1.5B-Instruct | `explainer_student/` |
| 7 | `quantize` | FP8 scorer student; AWQ 4-bit explainer student (served by vLLM). | llm-compressor | `scorer_student/fp8/`, `explainer_student/awq/` |
| 8 | `calibrate` | Platt vs isotonic per scorer, ECE, PR-AUC, threshold analysis, policy simulation, fairness by age band and region, explainer evaluation (valid JSON, schema, reason F1, guard). | – | `calibration/` |
| – | `registry` | Logs data snapshot, prompt format, config, metrics and models to MLflow, and registers the servable scorer and explainer. | MLflow | MLflow model versions |

**Shared training settings** (`efficiency:` in `base.yaml`):

- Liger kernels (fused RMSNorm / SwiGLU / RoPE, plus fused linear+cross-entropy for causal models);
- TF32, `group_by_length` batching, bf16 throughout;
- a hard VRAM cap, `gpu_memory_fraction: 0.95`. It fails with OOM rather than spilling into system RAM.

**Reuse.** `reuse_from: <run>` seeds a run with another run's artefacts. That's how `demo_llm_scorer` adds only the
LLM-scorer branch on top of `demo_hybrid`.

## 5. Data and features

**Source.** The `.env` Supabase project, read-only (`training/argus_ft/supabase_data.py`):

- **Storage bucket `argus-data`.** SFT prompts, gold explanations, DPO pairs, the CPT corpus, the
  format-comparison set, raw transactions, plus `DATA_CARD.md`, `manifest.json`, `quality_report.json` and
  `rules_config.json` (the seed for the rules engine).
- **Postgres tables.** Transactions, labels, customers, features. Snapshotted to `training/data/db/*.parquet` for
  the XGBoost baseline and the fairness report.

The first run downloads everything into `training/data/`. Later runs only fetch changes, or nothing when started
with `--no-sync` (the UI default).

**Dataset.** A synthetic stand-in for the CRD's 14 months:

- 100,000 main transactions (Jan 2025 – Feb 2026) plus 10,000 warm-up events;
- fraud rate about 0.6%;
- 4,069 CPT documents, 7,600 gold explanations, 5,000 preference pairs;
- labels mature after 60 days.

**Splits** (FR-ML-05):

- train: Jan–Oct 2025;
- validate: Nov–Dec 2025;
- holdout: Jan–Feb 2026, never used for training or model selection.

**Features** (CRD 8; the same code serves training and live scoring, FR-FE-02):

| ID | Feature(s) | Reason code it supports |
|---|---|---|
| FE-01 | `amount_to_avg_ratio` | R05_AMOUNT_SPIKE |
| FE-02 | `is_new_device` (not used by the customer in 90 days) | R01_NEW_DEVICE |
| FE-03 | `customer_merchant_distance_km`, `is_new_city` | R02_GEO_MISMATCH |
| FE-04 | `hour_of_day`, `is_night` (00:00–05:00 IST) | R03_ODD_HOUR |
| FE-05 | `failed_transactions`, `has_recent_failures` (≥ 3) | R04_FAILED_ATTEMPTS |
| FE-06 | `account_age`, `is_new_account` (< 30 days) | R06_NEW_ACCOUNT |
| FE-07 | `txn_count_last_1h`, `txn_amount_last_24h` | R07_VELOCITY |
| FE-08 | MCC target-encoded fraud rate (training folds only) | R08_HIGH_RISK_MCC (electronics, gaming, gift cards) |
| FE-09 | `distinct_customers_per_device_7d` | – |
| FE-10 | `combined_risk_flags` (count of FE-02..FE-06 flags) | – |

**Serialisation** (CRD 8.2, FR-FE-05/06). Each transaction becomes one versioned prompt.

- **Sections:** `[TXN] [CUST] [CTX] [HIST] [TASK]`. Numbers are bucketed consistently, e.g. distance
  `<5 | 5-50 | 50-200 | 200-500 | 500-1000 | 1000+` km.
- **Length cap:** `max_length` tokens (384 locally, 1,024 in production). The oldest `[HIST]` entries are trimmed
  first.
- **Formats:** `fmt-v1` is the CRD format; `fmt-v2` adds a `[SUMMARY]` line. The pilot picks the better one, and
  the scorer records which one it was trained on, so serving uses the same format.

## 6. Scoring path design

Request flow inside `POST /v1/fraud/score`:

```
validate + sanitise (400 on error)
  → hard rules (blocked device → BLOCK / HARD_RULE, no model call)
  → features (snapshot store hit, or FEATURES_DEGRADED from request fields only)
  → scorer → raw logit → calibrator → probability
  → policy (global or per-MCC thresholds; allowlisted customer capped at REVIEW)
  → reason codes (flag true AND head ≥ 0.5, at most 3, strongest first)
  → explanation
      REVIEW / BLOCK: wait up to explain_cutoff_ms; ALLOW or a slow explainer: asynchronous, written to the audit log
      guard: valid JSON + schema → every number grounded in the input → explainer codes overlap the scorer's codes
      any failure → reason-code template (explanation_source says which check failed)
  → response
  → audit record (after the response leaves)
```

| Component | Local implementation | Production (LLD) |
|---|---|---|
| Feature store | offline snapshot of the Supabase `features` table + training prompts; unknown ids use request fields only | Redis, real-time velocity features |
| Scorer | the run's registered scorer: FP8 LLM student (LLM-scorer run) or XGBoost (HYBRID run) | FP8 0.5B scorer on L4 / A10G |
| Calibrator | `calibration/<scorer>.json` from stage 8 | same |
| Explainer | vLLM serving the AWQ explainer student | vLLM, AWQ 1.5B explainer |
| Rules config | `training/outputs/_scoring/rules_config.json`, re-read when the file changes | Aurora + 30 s hot reload |
| Audit | `training/outputs/_scoring/audit.jsonl` | 5-year immutable store (NFR-11) |
| Deployments | several runs loaded with LRU eviction, switch at runtime | registry-approved versions only, 7-day shadow |

**Time budgets.**

- **Production:** total deadline 250 ms (`X-Request-Deadline-Ms` can lower it, capped at 280 ms); REVIEW/BLOCK wait
  at most 220 ms for the explainer.
- **4 GB laptop** (`local_demo.yaml`): scoring takes about 0.3 s, so the budgets are 5,000 / 3,000 ms. The ±15 min
  `transaction_time` check is off, because the demo replays 2025 transactions.

## 7. Control plane and UI design

**Platform** (`platform/`, FastAPI) is split into three layers:

- **`app/core`:** settings (`ARGUS_*` env vars), dependency container, error model, SSE helper.
- **`app/infra`:** one adapter per outside system:
  - run folders and event tailing (`runs_repo`, `events`);
  - job store and Docker job runner (`job_store`, `job_runner`);
  - profiles, NVML GPU telemetry, HTTP clients for scoring and MLflow;
  - client documents and diagrams.
- **`app/features/<area>`:** one router + service per area: `jobs`, `runs`, `profiles`, `registry`, `scoring`,
  `system`, `knowledge`. Every router is mounted under `/api/v1`.

**Run state.**

- **Folding events.** The live state of a run (stage ladder, progress, loss series, status) is folded
  incrementally from `events.jsonl`.
- **Queued runs.** A queued job writes `job_queued` first, so the run page exists before the container starts.
- **Interrupted runs.** A run whose events stop for 15 minutes without `run_end` is shown as *interrupted*.

**Jobs.**

- **Launching.** `POST /jobs` validates the profile, run name, overrides and stages, then queues the job (FIFO, one
  GPU job at a time).
- **The job container.** It is started as `argus-job-<id>` with labels `argus.job` and `argus.run`. It gets the
  host paths of the platform's own mounts, plus `SUPABASE_URL`, `SUPABASE_KEY` and `HF_TOKEN` from `.env`.
- **Cancelling** stops the container. The run keeps everything written so far.

**Frontend** (`frontend/`, React 19 + Vite + TypeScript, TanStack Query, Tailwind, Radix):

- **`src/app`:** the shell, the router and the feature registry.
- **`src/shared`:** API client, UI kit, charts, formatters.
- **`src/features/<page>`:** one folder per page, each exporting a `FeatureDef`; the menu and routes come from the
  registry.
- **Proxy.** nginx serves the SPA and proxies `/api` to the platform on the same origin, with SSE unbuffered.

## 8. API design: Fraud Model API (scoring :8080)

Interactive docs: http://localhost:8080/docs. The contract is CRD 10, validated as in LLD 4.

### 8.1 Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /v1/fraud/score` | `Bearer demo-service-token` | score one transaction (CRD 10.2 → 10.3) |
| `GET /v1/health` | – | service, scorer, explainer, feature store and config versions (FR-API-04) |
| `GET /admin/v1/thresholds` | `Bearer demo-admin-token` | current global and per-MCC thresholds |
| `POST /admin/v1/thresholds` | admin | change thresholds (two-person approval) |
| `GET /admin/v1/lists/{name}` | admin | `blocked_devices` or `allowlisted_customers` |
| `POST` / `DELETE /admin/v1/lists/{name}` | admin | add / remove an item |
| `GET /admin/v1/decisions/{transaction_id}` | admin | the full audit record, including an explanation produced later |
| `GET /admin/v1/decisions?limit=` | admin | recent decisions |
| `GET /admin/v1/changelog` | admin | every rule, threshold and deployment change |
| `GET /admin/v1/models` | admin | runs with a servable scorer |
| `POST /admin/v1/models/{run}/activate` | admin | switch the served run without a restart |

Production replaces the bearer tokens with mTLS + a 15-minute service JWT, and SSO + MFA for admins (NFR-07/09).

### 8.2 Request (CRD 10.2)

```json
{
  "transaction_id": "TX98231",
  "customer_id": "C_7f31a9",
  "transaction_amount": 50000.00,
  "merchant_category": "electronics",
  "transaction_time": "2025-11-14T03:12:44+05:30",
  "device_id": "D_44be20",
  "customer_age": 27,
  "customer_location": {"lat": 18.52, "lon": 73.85, "city": "Pune"},
  "merchant_location": {"lat": 28.61, "lon": 77.20, "city": "Delhi"},
  "account_age": 6
}
```

| Field | Rule (LLD 4.1) |
|---|---|
| `transaction_id`, `customer_id` | 1–64 characters, `[A-Za-z0-9_-]` |
| `transaction_amount` | > 0, ≤ 10,000,000, at most 2 decimals |
| `merchant_category` | one of `education, electronics, food, gaming, gift_cards, grocery, healthcare, other, quick_commerce, travel`; anything else becomes `other` (warning in the audit record) |
| `transaction_time` | RFC 3339 **with** a UTC offset; within ±15 min of server time when `time_window_minutes` > 0 |
| `device_id` | optional, ≤ 64 characters |
| `customer_age` | 18–120 |
| `customer_location` | optional (about 3% of rows lack it); `lat` −90..90, `lon` −180..180 |
| `merchant_location` | required, same ranges |
| `city` | sanitised: NFKC, `[A-Za-z0-9 .,-]` only, ≤ 32 characters (prompt-injection defence) |
| `account_age` | ≥ 0 days |

History fields (`previous_transactions`, `average_transaction_value`, `failed_transactions`) are not sent by the
caller. The service looks them up in the feature store (CRD 10.2).

### 8.3 Response (CRD 10.3)

```json
{
  "transaction_id": "TX98231",
  "fraud_probability": 0.94,
  "decision": "BLOCK",
  "risk_level": "CRITICAL",
  "reason_codes": ["R01_NEW_DEVICE", "R04_FAILED_ATTEMPTS", "R03_ODD_HOUR"],
  "explanation": "First use of this device, 1,170 km from the merchant, at 3:12 AM, after 4 failed attempts ...",
  "decision_source": "MODEL",
  "model_version": "argus-lm-1.0.0-q8",
  "latency_ms": 118
}
```

| Field | Values |
|---|---|
| `fraud_probability` | calibrated, 4 decimals; `null` when a hard rule decided |
| `decision` / `risk_level` | see the policy below |
| `reason_codes` | up to 3 of R01–R08, strongest first |
| `explanation` | text, or `null` when it is produced asynchronously (it appears in the audit record within 5 s) |
| `decision_source` | `MODEL` or `HARD_RULE` (FR-RE-05) |
| `model_version` | the served run's scorer version |

**Explanation sources** (`explanation_source`, recorded in the audit record):

| Source | Meaning |
|---|---|
| `LLM` | the explainer's text passed every check |
| `EXPLANATION_ASYNC` | not ready in time; written later |
| `TEMPLATE` | explainer unavailable; sentence built from the reason codes |
| `SCHEMA_FAILED` | invalid JSON or schema |
| `GROUNDING_FAILED` | cites a number not in the input |
| `INCONSISTENT` | the explainer's codes don't overlap the scorer's |

Every source except `LLM` returns the reason-code template.

**Decision policy** (CRD 10.4, initial thresholds, changeable at runtime):

| Probability | Risk level | Decision |
|---|---|---|
| 0.00 – 0.29 | LOW | ALLOW |
| 0.30 – 0.59 | MEDIUM | ALLOW (step-up OTP if the merchant supports it) |
| 0.60 – 0.84 | HIGH | REVIEW |
| 0.85 – 1.00 | CRITICAL | BLOCK (allowlisted customer → REVIEW) |

**Reason codes** (CRD 9.3):

| Code | Meaning |
|---|---|
| R01_NEW_DEVICE | device not used by the customer before |
| R02_GEO_MISMATCH | far from the merchant or an unusual city |
| R03_ODD_HOUR | unusual time of day |
| R04_FAILED_ATTEMPTS | several failed attempts first |
| R05_AMOUNT_SPIKE | much higher than usual spend |
| R06_NEW_ACCOUNT | very new account |
| R07_VELOCITY | many transactions in a short period |
| R08_HIGH_RISK_MCC | high-risk merchant category |

**Errors:**

| Status | Body | When |
|---|---|---|
| 400 | `{"error": "INVALID_REQUEST", "field": ..., "message": ...}` | validation failure (FR-API-02) |
| 401 | `{"error": "UNAUTHORISED"}` | missing or wrong token |
| 404 | `{"error": "NOT_FOUND"}` | unknown decision or run |
| 503 | `{"error": "UNAVAILABLE", "retry": false}` | no active model or a scorer error |

On 503 the caller applies fail-open (FR-API-01): approve under ₹10,000, otherwise REVIEW.

### 8.4 Admin changes

**Thresholds** take a two-person body:

```json
{"medium": 0.30, "review": 0.60, "block": 0.85, "mcc": "electronics",
 "requested_by": "analyst.a", "approved_by": "lead.b", "reason": "festive-sale card testing"}
```

- `requested_by` and `approved_by` must be different people.
- `mcc` omitted means the global thresholds.
- Lists take `{"item": ..., "by": ..., "reason": ...}`.

Every change bumps `threshold_version` / `rules_version` and is appended to `changelog.jsonl`. It takes effect on
the next request; no restart is needed.

## 9. API design: Platform API (:8090/api/v1)

The UI calls these through nginx, as `/api/v1/...` on port 3000. Interactive docs: http://localhost:8090/docs.

| Area | Endpoints |
|---|---|
| system | `GET /health`, `GET /system/gpu` (NVML; each metric `null` if unavailable), `GET /services` (scoring, explainer, mlflow, platform), `POST /services/{service}/{action}` (start / stop serving) |
| profiles | `GET /profiles` (a broken profile is listed with an `error`), `GET /profiles/{name}` (raw YAML + resolved config) |
| jobs | `GET /jobs`, `POST /jobs`, `POST /jobs/preview` (the exact command and resolved config, without starting), `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `GET /jobs/{id}/logs/stream` (SSE) |
| runs | `GET /runs`, `GET /runs/{run}` (ladder, status, results), `GET /runs/{run}/report.md`, `GET /runs/{run}/series` (loss / metric series), `GET /runs/{run}/artifacts`, `GET /runs/{run}/files/{path}`, `GET /runs/{run}/events?after=`, `GET /runs/{run}/events/stream` (SSE) |
| compare | `GET /compare?runs=a,b` (up to 4 runs) |
| registry | `GET /registry/info`, `GET /registry/models`, `GET /registry/runs` (MLflow REST) |
| scoring | `POST /scoring/score`, `GET /scoring/health`, `GET /scoring/samples`, `GET /scoring/models`, `POST /scoring/models/{run}/activate`, `GET/POST /scoring/admin/thresholds`, `GET/POST/DELETE /scoring/admin/lists/{name}`, `GET /scoring/admin/decisions/{id}`, `GET /scoring/admin/decisions`, `GET /scoring/admin/changelog` |
| knowledge | `GET /data/overview` (data card, quality report, splits), `GET /docs/documents`, `GET /docs/documents/{name}` (docx → HTML), `GET /docs/diagrams`, `GET /docs/diagrams/{name}` |

**`POST /jobs` body:**

```json
{"profile": "demo_hybrid", "run_name": "my_run", "overrides": ["pilot.max_steps=30"],
 "stages": ["calibrate", "registry"], "reuse_from": "argus_demo_hybrid",
 "skip_data_sync": true, "allow_existing": false}
```

Only `profile` is required. `run_name` defaults to the profile's own. Without `allow_existing`, an existing run
folder is refused (409).

**Errors.** One model for the whole API: `{"error": CODE, "message": text}`.

| Status | Code |
|---|---|
| 400 | `INVALID_REQUEST` |
| 404 | `NOT_FOUND` |
| 409 | `CONFLICT`: run exists, GPU busy, serving active |
| 502 | upstream error (scoring / MLflow) |
| 503 | upstream down |

**SSE streams** send `id:` set to the byte offset. A reconnect with `Last-Event-ID` resumes exactly.

## 10. File contracts

### 10.1 Training events: `training/outputs/<run>/events.jsonl`

One JSON object per line, append-only. The pipeline and every stage process write to it:

```json
{"ts": "2026-11-08T01:02:03.456Z", "type": "train_step", "stage": "cpt", "data": {"step": 10, "loss": 2.31}}
```

| `type` | `data` |
|---|---|
| `job_queued` | `profile`, `stages[]` (written by the platform when the job is queued) |
| `run_start` | `profile`, `stages[]`, `run_name` |
| `stage_start` | `index`, `total` |
| `train_step` | `step`, `max_steps`, `epoch`, `loss`, `learning_rate`, `grad_norm`, `eval` (Trainer log step) |
| `progress` | `label`, `done`, `total` (long non-Trainer loops: scoring, generation, DPO, calibration), throttled |
| `metric` | `name`, `value` (e.g. `heldout_perplexity_before`) |
| `stage_end` | `status`, `minutes`, `peak_gpu_gb`, `error?` |
| `stage_skipped` | `reason` |
| `run_end` | `status`, `total_minutes` |

**How readers handle the file:**

- **Tailing.** Readers tail by byte offset and ignore a partial last line.
- **Older runs.** The runs trained before live events existed had their `events.jsonl` rebuilt from the console
  logs (`python -m argus_ft.backfill_events outputs/_logs/<log>`). The losses are the same, and every event is
  tagged `backfilled`.

### 10.2 Jobs: `training/outputs/_platform/jobs.json`

Newest first:

```
{id, run_name, profile, overrides[], stages[]?, reuse_from?, command[], status, message, container_id,
 created_at, started_at, finished_at, exit_code}
```

- **Status:** `queued → running → succeeded | failed | cancelled`.
- **Labels:** containers carry `argus.job=<id>` and `argus.run=<run>`.

### 10.3 Scoring state: `training/outputs/_scoring/`

| File | Contents |
|---|---|
| `rules_config.json` | thresholds, MCC overrides, lists, versions (seeded from the dataset's `rules_config.json`) |
| `audit.jsonl` | one decision record per request, plus explanation records appended later |
| `changelog.jsonl` | who changed what, when and why |

---

## 11. Prerequisites

| Need | Notes |
|---|---|
| Docker Desktop (WSL 2 backend) with GPU support | `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi` must print your GPU |
| NVIDIA GPU, bf16-capable (Ampere or newer) | local demo sized for a 4 GB RTX 3050 Laptop; production profile for an A100 80GB |
| `01_argus/.env` | `SUPABASE_URL`, `SUPABASE_KEY` (data sync); optional `HF_TOKEN` (gated models in production) |
| Disk | about 70 GB: training image 13 GB (scoring shares its layers), vLLM image 35 GB, build cache, models and outputs about 16 GB |

`client_docs/`, `day_wise/` and `.env` stay where they are in `01_argus/`; compose mounts them read-only.

## 12. Quick start

From `01_argus/argus_solution/` in PowerShell:

```powershell
.\argus.ps1 build          # all images (first time: 20-30 min plus downloads, mostly the training image)
.\argus.ps1 up             # UI, platform API, MLflow
start http://localhost:3000
```

Every `argus.ps1` action prints the plain `docker compose` command it runs, so you can always run that command
yourself instead. `.\argus.ps1` with no arguments lists the actions.

| Action | Plain docker command |
|---|---|
| `build` | `docker compose build trainer; docker compose --profile serve build scoring; docker compose build platform frontend` |
| `up` | `docker compose up -d` |
| `serve [run]` | `$env:RUN_NAME="<run>"; $env:EXPLAINER_RUN="<run>"; docker compose --profile serve up -d scoring explainer` |
| `unserve` | `docker compose --profile serve stop scoring explainer` |
| `train <profile> [run]` | `docker compose run --rm trainer python -m argus_ft.pipeline --config configs/<profile>.yaml --no-sync` |
| `report` | `docker compose run --rm --no-deps trainer python -m argus_ft.compare argus_demo_hybrid argus_demo_llm_scorer` |
| `demo` | `docker compose --profile serve exec scoring python -m argus_api.demo --run <active run> --url http://localhost:8080` |
| `logs [service or job id]` | `docker compose logs -f <service>` / `docker logs -f argus-job-<id>` |
| `status` | `docker compose --profile serve ps -a` + job containers + images |
| `test` | all three test suites in containers (section 19) |
| `down` | `docker compose --profile serve --profile job down` (files on disk are kept) |

## 13. Services and ports

| Service | URL | Image | GPU | Started by |
|---|---|---|---|---|
| frontend | http://localhost:3000 | `argus-frontend` (nginx) | – | `up` |
| platform | http://localhost:8090/docs | `argus-platform` | NVML read-only | `up` |
| mlflow | http://localhost:5000 | `argus-training` | – | `up` |
| scoring | http://localhost:8080/docs | `argus-scoring` (built on `argus-training`) | yes | `serve` (profile `serve`) |
| explainer | http://localhost:8000/v1/models | `vllm/vllm-openai:v0.33.0` | yes | `serve` (profile `serve`) |
| training job | – | `argus-training`, container `argus-job-<id>` | yes | the UI (Launch) or `train` |

**One GPU, one owner at a time.**

- **Queue.** On a 4 GB card the platform runs one training job at a time (FIFO queue).
- **Serving blocks training.** Queued jobs are held while `scoring`/`explainer` run, with the message *"stop
  serving to free the GPU"*. Stop serving from the UI (System page) or with `.\argus.ps1 unserve`.
- **Training blocks serving.** Starting serving while a job runs is refused.

## 14. The UI

| Page | What it shows / does |
|---|---|
| **Overview** | headline numbers of the latest runs, active jobs, services, GPU |
| **Launch** | pick a profile, name the run, edit parameters (only changed values become overrides), choose stages, reuse another run's artefacts, preview the exact command, start |
| **Runs** | every run with status and progress; a run page has the live ladder, loss charts per stage (SSE, resumes after reconnect), container logs, per-stage results (pilot table, gate, calibration, fairness, explainer samples), artefacts and `report.md` |
| **System** | services (start/stop scoring + explainer), GPU telemetry, the job queue (cancel) |
| **Pipeline** | Figure 4 as an interactive graph; click a stage for what it does and its latest result |
| **Compare** | up to 4 runs side by side (e.g. `argus_demo_hybrid` vs `argus_demo_llm_scorer`) |
| **Registry** | registered models and versions from MLflow, plus the full MLflow UI embedded |
| **Scoring** | API playground with ready samples (CRD worked example, known fraud, genuine, hard rule, invalid), deployments (switch the served run), recent decisions with async explanations |
| **Rules & policy** | thresholds (global and per MCC, two-person approval), blocked devices, allowlisted customers, change log |
| **Dataset** | data card, quality report, splits |
| **Docs & diagrams** | the client documents (`.docx` rendered in the browser) and the architecture diagrams |

## 15. Training

### From the UI

1. **Launch** → choose a profile.
2. Optionally change parameters and stages.
3. **Start.**

The job appears under **Runs** immediately and the page follows it live: stage ladder, loss, throughput, logs.
Cancel from the run page or from **System**.

### From the CLI
```powershell
.\argus.ps1 train smoke                      # every stage, tiny sizes (~20-30 min locally)
.\argus.ps1 train demo_hybrid                # demo example 1 (HYBRID, ~55 min locally)
.\argus.ps1 train demo_llm_scorer            # demo example 2 (LLM scorer, reuses example 1, ~30 min)
docker compose run --rm trainer python -m argus_ft.pipeline --config configs/local_demo.yaml --stages calibrate,registry --no-sync
```
CLI runs write the same `events.jsonl`, so they appear live in the UI too.

### Profiles (`training/configs/`)

| Profile | Purpose |
|---|---|
| `base.yaml` | the full ladder and every default; all others `extends:` it and override only what changes |
| `smoke.yaml` | every stage with tiny sizes and `gate.mode: llm` so no stage is skipped: catches crashes in minutes |
| `local_demo.yaml` | the demo on a 4 GB RTX 3050 (relaxed API deadlines) |
| `demo_hybrid.yaml` | demo example 1: full ladder, `gate.mode: hybrid` → XGBoost scores, the LLM explains |
| `demo_llm_scorer.yaml` | demo example 2: `gate.mode: llm`, `reuse_from: argus_demo_hybrid` → adds only the LLM scorer stages |
| `prod_a100.yaml` | one A100 80GB with the models named in Part 3 (Qwen2.5-7B / Llama-3.2-3B / Gemma-2-2B pilots, 0.5B scorer, 1.5B explainer), 1,024-token prompts, all validation rows |

**`gate.mode`:**

- `auto` follows the measured week-8 decision.
- `hybrid` / `llm` force a branch. The report still shows the measured decision.

**Overrides.** Any value can be overridden per run: `--set pilot.max_steps=30` on the CLI, or the parameter editor
in **Launch**.

### What a run produces (`training/outputs/<run>/`)

| File | Contents |
|---|---|
| `report.json` / `report.md` | every stage's metrics; written at the end of every run, also failed ones |
| `events.jsonl` | the live event stream the UI reads (section 10.1) |
| `xgboost/`, `pilot/`, `teacher*/`, `scorer_student/fp8/`, `explainer_student/awq/`, `calibration/` | models and artefacts per stage |
| `api/` | outputs of the scoring demo for this run (`demo_results.json`) |
| `../comparison.md` | hybrid vs LLM scorer side by side (`.\argus.ps1 report`) |

## 16. Serving and API testing

```powershell
.\argus.ps1 unserve; .\argus.ps1 serve argus_demo_llm_scorer   # or argus_demo_hybrid (XGBoost scorer)
.\argus.ps1 demo                                                # all CRD / LLD scenarios -> outputs/<run>/api/demo_results.json
```

- **First start.** The first `serve` creates the two containers. Afterwards the UI can stop and start them (System
  page).
- **Switching the served model** works at runtime, without a restart: **Scoring → Deployments**, or
  `POST /admin/v1/models/{run}/activate`.

```powershell
$h = @{ Authorization = "Bearer demo-service-token" }
$body = @{ transaction_id = "TX98231"; customer_id = "C10045"; transaction_amount = 48500.00
           merchant_category = "electronics"; transaction_time = "2026-04-12T02:14:00+05:30"; device_id = "D-7781"
           customer_age = 29; customer_location = @{ lat = 28.61; lon = 77.21; city = "Delhi" }
           merchant_location = @{ lat = 19.07; lon = 72.88; city = "Mumbai" }; account_age = 45 } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8080/v1/fraud/score -Headers $h -ContentType application/json -Body $body
```

**Through the platform** (no token needed; the UI uses this): `POST http://localhost:8090/api/v1/scoring/score`.

**What `demo` exercises:**

- the CRD worked example;
- known fraud and genuine transactions;
- a blocked device (hard rule) and an allowlisted customer;
- invalid requests (400);
- a threshold change with two-person approval;
- the async explanation lookup.

## 17. Logs: where to look

| What | Where |
|---|---|
| A training run, live | UI → **Runs** → the run (loss charts, stage ladder, *Logs* tab streams the container output) |
| A training run, terminal | `.\argus.ps1 logs <job id>` = `docker logs -f argus-job-<job id>` (job id shown in the UI) |
| A CLI training run | printed in your terminal; also `training/outputs/<run>/events.jsonl` |
| Run results | `training/outputs/<run>/report.md` (also UI → run → *Report*) |
| Platform / frontend / MLflow | `docker compose logs -f platform` (or `frontend`, `mlflow`) |
| Scoring API / vLLM | `docker compose --profile serve logs -f scoring` (or `explainer`) |
| Every scoring decision | `training/outputs/_scoring/audit.jsonl` (UI → Scoring → *Recent decisions*) |
| Rule / threshold / deployment changes | `training/outputs/_scoring/changelog.jsonl` (UI → Rules & policy) |
| Job queue state | `training/outputs/_platform/jobs.json` (UI → System) |
| Experiments, metrics, model versions | MLflow http://localhost:5000 (UI → Registry) |
| Older console logs | `training/outputs/_logs/` |

## 18. Where to change things

| To change | Edit | Then |
|---|---|---|
| Training sizes, steps, learning rates, models | `training/configs/*.yaml` (or override per run in Launch) | nothing: jobs mount `configs/` |
| A new profile | add `training/configs/<name>.yaml` with `extends: base.yaml` | appears in Launch automatically |
| Pipeline code / a stage | `training/argus_ft/stages/<stage>.py`; order in `stages/__init__.py` (`LADDER`) | nothing for jobs (code is mounted); `docker compose build trainer` for the image |
| Prompt serialisation | `training/argus_ft/data.py` (`PROMPT_FORMATS`, `to_format`) | retrain: the format is part of the model |
| Gate rule | `gate.min_uplift`, `gate.mode` in the profile | – |
| Decision thresholds, lists | UI → Rules & policy (persisted in `training/outputs/_scoring/rules_config.json`) | live, no restart |
| Production thresholds / API deadlines / tokens | `api:` section in `training/configs/base.yaml` / `local_demo.yaml` | `.\argus.ps1 serve` again |
| Scoring API code | `scoring_api/argus_api/` (`app.py` flow, `rules.py`, `explainer.py`, `features.py`, `scorer.py`) | `docker compose --profile serve build scoring` |
| Platform API (new endpoint) | add a router in `platform/app/features/<area>/router.py`, register it in `features/__init__.py` | `docker compose up -d --build platform` |
| UI (new page) | create `frontend/src/features/<name>/index.ts` exporting a `FeatureDef`, add it to `src/app/features.ts` | `docker compose up -d --build frontend` |
| Ports, GPU share for vLLM, mounted folders | `docker-compose.yml` (`VLLM_GPU_UTIL`, `RUN_NAME`, `EXPLAINER_RUN` env vars) | `docker compose up -d` |
| Platform settings (URLs, tokens, image names) | `ARGUS_*` env vars on the `platform` service (see `platform/app/core/config.py`) | `docker compose up -d platform` |

## 19. Tests

```powershell
.\argus.ps1 test
```
| Suite | Runs in | Covers |
|---|---|---|
| platform (pytest, 56 tests) | `argus-platform:test` | event tailing (offsets, partial/corrupt lines), live-state fold, interrupted runs, path traversal, profiles (extends, overrides, loops), job validation, FIFO GPU scheduler, cancel, every HTTP route incl. SSE streams and upstream-down errors |
| scoring (pytest, 51 tests) | `argus-scoring` | CRD request validation edge cases, sanitising, policy boundaries, hard rules, allowlist cap, MCC overrides, two-person approval, hot reload / corrupt config, reason-code rules, explanation consistency guard, deployments (LRU) |
| frontend (vitest, 25 tests) | `node` build stage | formatters, null-safe ratios, overrides diff, API client errors, UI states, `Button asChild`, feature registry |

## 20. Troubleshooting

| Symptom | Fix |
|---|---|
| Job stays *queued* with "stop serving to free the GPU" | `.\argus.ps1 unserve` (or System → stop scoring/explainer) |
| "service 'scoring' was never created" when starting serving from the UI | run `.\argus.ps1 serve` once from a terminal |
| `docker compose up` says "pull access denied" for an `argus-*` image | the images are local only (`pull_policy: never`): run `.\argus.ps1 build` first |
| Platform says Docker is not reachable | Docker Desktop running? The compose file mounts `/var/run/docker.sock`, which Docker Desktop provides on Windows too |
| `pip` downloads crawl or time out while building | build with a PyPI mirror: `$env:PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"; .\argus.ps1 build` (same pinned versions) |
| Build hangs on "load metadata" from Docker Hub | `$env:DOCKER_BUILDKIT=0` for that build, or retry later |
| CUDA out of memory on 4 GB | lower `batch_size` / raise `grad_accum` for the stage, or `max_length`; make sure serving is stopped |
| A run shows *interrupted* | its events stopped for 15 min without `run_end` (container killed / PC slept); open the run and press **Re-run stages** (same run name, pick the remaining stages) |
| MLflow panel empty in the UI | open http://localhost:5000 directly; `docker compose logs mlflow` |
| Scoring returns 503 "no model active" | activate a run: Scoring → Deployments, or `serve <run>` |
| Port already in use | change the left side of `ports:` in `docker-compose.yml` |
| C: drive still full after deleting images | Docker Desktop's virtual disk doesn't shrink by itself: Docker Desktop → Troubleshoot → Clean / Purge data |

## 21. Results so far

These are the local runs on a 4 GB RTX 3050, from `training/outputs/comparison.md` (`.\argus.ps1 report`). They are
demo-sized runs: 1 epoch or a few steps per stage, 135M–500M models, a random 5,000-row validation sample. They
prove the ladder works end to end, **not** the CRD accuracy targets. Those need the A100 profile and the client's
real data.

| Scorer | PR-AUC | ROC-AUC | Calibration (ECE) | Top 12,000/day: recall / precision |
|---|---|---|---|---|
| XGBoost (baseline, served by the HYBRID run) | 0.275 | 0.988 | isotonic 0.0024 | 4.8% / 14.3% |
| LLM teacher scorer (bf16) | 0.423 | 0.988 | Platt 0.0018 | 19.1% / 57.1% |
| LLM student scorer (bf16) | 0.329 | 0.988 | Platt 0.0016 | 4.8% / 14.3% |
| LLM student scorer (FP8, served) | 0.315 | 0.988 | Platt 0.0019 | 4.8% / 14.3% |

**The gate.**

- **Pilot result.** The pilot winner (h2o-danube3-500m, `fmt-v1`) reached 0.174 validation PR-AUC after its short
  pilot. That is below XGBoost, so the week-8 gate measured **HYBRID**.
- **The forced comparison.** `demo_llm_scorer` forces the LLM branch anyway, so both branches can be compared side
  by side.
- **What the comparison shows.** After CPT and full SFT, the LLM teacher beats XGBoost by +0.15 PR-AUC. That is
  exactly the kind of evidence the week-8 decision would revisit.

**Explainers.**

- 100% valid JSON and schema, reason-code F1 about 0.6, 100% accepted by the guard.
- Answers aren't varied yet: 1–2 distinct answers on 17 gold cases. The demo-length training learned the majority
  answer. More SFT rows and steps (the A100 profile) are the fix.
- The consistency guard falls back to the reason-code template whenever the text contradicts the scorer's codes.

## 22. Local demo vs production

| Concern | This build | Production (CRD 11 / Architecture / LLD) |
|---|---|---|
| Training compute | RTX 3050 4 GB, `local_demo` | A100 / H100 on-demand windows, `prod_a100`, ≤ 2,500 GPU-hours |
| Models | 135M–500M Apache-2.0 models | Qwen2.5 / Llama 3.2 / Gemma 2 pilots, 0.5B scorer (FP8), 1.5B explainer (AWQ) |
| Data | synthetic Supabase dataset (100k transactions) | 1.2 crore rows + 135M-token corpus in S3, ap-south-1 |
| Feature store | offline snapshot + degraded path | Redis, real-time velocity features (FR-FE-03) |
| Rules config | JSON file, hot reload on change | Aurora + 30 s reload, admin screen behind SSO + MFA |
| Auth | bearer tokens | mTLS + 15-min service JWT (NFR-07) |
| Serving | one GPU, one scoring container, vLLM explainer | multi-AZ EKS, L4 / A10G node group (≤ 8 GPUs), autoscaling to 2,800 TPS |
| Latency budgets | 5,000 / 3,000 ms (laptop) | 250 ms total, 220 ms explainer cutoff (`prod_a100.yaml`) |
| Audit | `audit.jsonl` | 5-year immutable store (NFR-11) |
| Registry | MLflow on SQLite | MLflow (or SageMaker registry), approved versions only, 7-day shadow (MLOPS-06) |
| Infrastructure | Docker Compose | Terraform, CI/CD, dashboards and drift alerts (NFR-13, MLOPS-01..04) |

**Moving to the A100.** Copy `argus_solution/` and `.env` to the GPU VM, then:
```bash
docker compose build trainer
docker compose run --rm trainer python -m argus_ft.pipeline --config configs/prod_a100.yaml
```

- **Only the profile changes:** same code, the models named in Part 3.
- **Gated models.** Llama 3.2 and Gemma 2 are gated on Hugging Face, so set `HF_TOKEN` in `.env` for an account
  that has accepted both licences.
- **Deadlines.** In `prod_a100.yaml` the API deadlines go back to the production 250 / 220 ms.

## 23. Plan

### 23.1 How this build was done (complete)

1. [x] Training code and scoring code brought into `argus_solution/`; the first fine-tuning workspace retired after
   its runs finished.
2. [x] Training events: `argus_ft/events.py`, Trainer callback, pipeline and stage hooks.
3. [x] Platform backend: core, infra adapters (runs, events, jobs/Docker, MLflow, scoring client), services,
   routers.
4. [x] Frontend: app shell, shared UI kit, one module per page (overview, pipeline, runs + live run page, launch,
   compare, registry/MLflow, scoring playground, rules, dataset, docs, system).
5. [x] Data, outputs and model cache moved under `training/`; all images built.
6. [x] Integration: smoke run launched and followed from the UI; serving stack (vLLM + scoring) started from the
   UI; MLflow embedded.
7. [x] Tests: platform pytest, scoring pytest, frontend vitest; end-to-end checks of every page and API.
8. [x] README.
9. [x] 4K demo video: docs → pipeline → live training → hybrid vs LLM → MLflow → API.

### 23.2 Delivery plan to production (CRD 15, 20 weeks)

| Weeks | Work | Deliverables | This build covers |
|---|---|---|---|
| 1–5 | Data access, PII redaction audit, EDA, corpus dedup and token counts, XGBoost baseline | D1 data report + corpus, D2 baseline | stages 0–1, data card / quality report |
| 6–8 | Pilot: 3 base models × 2 formats; licence review; **week-8 go/no-go** | D3 pilot report | stage 2 + gate (`prod_a100` pilots the Part 3 models) |
| 9–14 | CPT → SFT (scorer + explainer) → DPO → distillation → quantisation → calibration, thresholds, fairness | D4 checkpoints in the registry, D5 production model + reports | stages 3–8 + registry |
| 10–15 | Fraud Model API, rules engine, admin; Redis feature store; Terraform, CI/CD, dashboards | D6 API + tests, D7 IaC + monitoring | D6 done locally; D7 not in this build |
| 15–17 | Acceptance gates 1–4: offline accuracy on the holdout, 500-case explanation rating, red-team, 2-hour 2,800 TPS load test with an AZ failure | D8 acceptance results | evaluation code in stage 8; load and red-team tests to do |
| 17–20 | 4-week shadow pilot, go-live recommendation; model card, data card, runbook, KT | D9 shadow report, D10 docs + KT | docs; shadow pilot to do |

**Next steps on the real platform:**

- run `prod_a100` on the client's data and check OBJ-01..06 on the Jan–Feb 2026 holdout;
- replace the feature snapshot with Redis;
- put the API behind mTLS on EKS;
- add PSI drift checks and the BLOCK-rate alert (MLOPS-02/04);
- wire reviewer verdicts back into SFT and DPO data (MLOPS-08).

## 24. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM does not beat XGBoost | week-8 gate measured on every run; HYBRID branch ready (XGBoost scores, LLM explains); both branches comparable side by side |
| Latency at peak | distilled small scorer, FP8; explainer off the critical path (async for ALLOW, cutoff for REVIEW/BLOCK); fail-open on 503 |
| Hallucinated explanations | JSON schema + grounding guard + reason-code consistency check, with a template fallback; DPO; source recorded per decision |
| Prompt injection via free text | free-text fields sanitised before they reach a prompt; codes come from the scorer, not the text |
| PII memorised from case notes | redacted corpus (DR-01), dedup, holdout notes excluded (DR-02); memorisation test still to add on real data (FR-ML-13) |
| Training–serving skew | one feature and serialisation code path (`argus_ft.data`) shared by training and the API; the scorer records its prompt format |
| GPU contention on a small card | single FIFO job queue; jobs held while serving holds the GPU; hard VRAM cap per process |
| Bind mounts for launched jobs need host paths | the platform reads its own container's mounts from the Docker API and reuses their host sources |
| Browser CORS / SSE buffering | nginx proxies `/api` on the same origin with buffering off for SSE; CORS also enabled for direct :8090 use |
| MLflow UI refuses iframe / host header | permissive allowed-hosts / CORS; native registry views as the primary UI, iframe + new-tab link as extra |
| Dependency drift | exact versions pinned; frontend lockfile generated in the node:24 container; builds only in Docker |
| Windows paths / line endings | pathlib inside containers; LF line endings for shell entrypoints |

## 25. Folder layout

```
argus_solution/
  README.md             this file
  argus.ps1             the command wrapper (sections 12-19)
  docker-compose.yml    every service; profiles: default | serve | job
  training/             argus_ft (the ladder, stages/, events, data, supabase sync) + configs/  -> image argus-training
    data/  outputs/  .hf_cache/     runtime state, bind-mounted (not in the image)
  scoring_api/          argus_api (CRD 10 / LLD 2-6) + tests/  -> image argus-scoring
  platform/             FastAPI control plane: app/core, app/infra, app/features/<area> + tests/  -> image argus-platform
  frontend/             React 19 + Vite + TypeScript: src/app (shell, registry), src/shared, src/features/<page>
  demo/                 the final demo video, captions, YouTube text
  demo_video/           scripts to re-record the demo video chapters
```

## 26. Demo video

**Files.** The narrated walkthrough (British English voice, 4K) is
`demo/Project_01_ARGUS_Part_06_Live_Demo.mp4`, with captions (`.srt`) and YouTube text next to it.

**How it was made.** It was recorded from the running UI, chapter by chapter, with the technical-demo-video skill.
The scripts live in `demo_video/`:

| File | Purpose |
|---|---|
| `chNN_*/flow.json` | one chapter: the UI steps (navigate, click, fill, highlight) and the narration for each |
| `rec4k.py` + `rec4k_patches.py` | records a chapter (headless Chromium + edge-tts `gb-male`), with the networkidle cap live pages need |
| `dryrun.py` | runs a chapter's steps without narration to prove every selector resolves (steps marked `dry_skip` are skipped, e.g. the Launch click) |

**Re-recording one chapter after a UI change.** The stack must be up, with serving started for ch08b:
```powershell
C:\dv\Scripts\python demo_video\dryrun.py ch05_runs
C:\dv\Scripts\python demo_video\rec4k.py --flow-dir demo_video\ch05_runs --base-url http://localhost:3000 --voice gb-male --viewport 1920x1080 --out demo_video\ch05_runs\ch05_runs.mp4
```
