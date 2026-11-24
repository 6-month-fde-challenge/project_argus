/** Shapes returned by the ARGUS platform API (app/features/*). Kept loose where the training report is free-form. */
export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "interrupted" | "incomplete" | "empty";
export type StageStatus = "pending" | "running" | "ok" | "failed" | "skipped" | "reused";

export interface Progress { label: string; done: number; total: number; ts: string }
export interface LastStep { step: number; max_steps: number; loss: number | null; learning_rate: number | null; epoch: number | null }

export interface Job {
  id: string; run_name: string; profile: string; status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  overrides: string[]; stages: string[] | null; reuse_from: string | null; command: string[];
  created_at: string; started_at: string | null; finished_at: string | null; exit_code: number | null; message: string | null;
  container_id: string | null;
}

export interface Gate {
  decision: string;          // what the week-8 gate measured
  branch?: string;           // what the run actually did (LLM when its LLM scorer stage trained)
  mode?: string;             // the run's own gate.mode: auto, or hybrid / llm = forced (demo)
  forced?: boolean;          // not a measured run (same rule as acceptance / registry / scoring API): a demo
  forced_reason?: string | null;
  uplift: number; llm_pr_auc: number; xgboost_pr_auc: number;
}

export interface RunSummary {
  name: string; status: RunStatus; profile: string | null; reused_from: string | null; gpu: string | null;
  total_minutes: number | null; stages_done: number; stages_total: number; current_stage: string | null;
  progress: Progress | null; last_step: LastStep | null; gate: Gate | null;
  pilot_winner: { teacher: string; format: string; pr_auc: number } | null; scorer: "LLM" | "XGBoost" | null;
  updated_at: number; job: Job | null;
}

export interface LadderRow {
  index: number; name: string; status: StageStatus; minutes: number | null; peak_gpu_gb: number | null;
  reused_from: string | null; progress: Progress | null; last_step: LastStep | null; error: string | null;
}

export interface RunDetail extends RunSummary {
  ladder: LadderRow[];
  sections: Record<string, any>;
  config: Record<string, any> | null;
  overrides: string[];
  has_report_md: boolean;
}

export interface TrainEvent { ts: string; type: string; stage: string | null; data: Record<string, any> }
export interface LossPoint { step: number; loss: number | null; learning_rate: number | null; epoch: number | null; eval?: boolean; ts: string }

export interface ProfileSummary {
  name: string; extends: string | null; description: string; run_name: string; stages: string[];
  gate_mode: string | null; reuse_from: string | null;
  error?: string;          // set when the profile cannot be resolved (bad YAML, extends loop)
  built_in?: boolean;      // ships with the repository: editable, never deleted from the UI
  serving_mode?: "production" | "demo";   // its api: section (allow_forced_runs)
  models?: Record<string, string>;
}
export interface ProfileDetail extends ProfileSummary { yaml: string; resolved: Record<string, any> }

export interface ServicesStatus {
  platform: string; docker: string; containers: Record<string, string>; scoring: string; explainer: string;
  mlflow: string; mlflow_url: string; gpu_jobs: string[]; stream?: string;
  serving_mode?: "production" | "demo" | "unknown";  // production = demo runs refused
  active_run?: string | null;
  demo_model?: boolean;    // the API serves a demo setup right now
  agent?: boolean;         // the operations agent is running on the host
}
export interface GpuSnapshot {
  available: boolean;
  gpus: { index: number; name: string; memory_used_mb: number | null; memory_total_mb: number | null; utilization_pct: number | null; temperature_c: number | null; power_w: number | null }[];
}

export interface ScoreResponse {
  transaction_id: string; fraud_probability: number | null; decision: "ALLOW" | "REVIEW" | "BLOCK"; risk_level: string;
  reason_codes: string[]; explanation: string | null; decision_source: string; model_version: string; latency_ms: number;
}
export interface SampleRequest { label: string; request: Record<string, any>; note: string }

/* ----------------------------------------------------------------- governance (Settings / Operations / AWS) */
export interface Approval { requested_by: string; approved_by: string; reason: string }
export interface ServingMode {
  configured: { config: string; mode: string; error?: string }; live_mode: string | null; production: boolean;
  mode: "production" | "demo" | "unknown"; active_run: string | null; active_run_forced: boolean;
  active_run_forced_reason: string | null; mismatch: boolean;
}
export interface AgentStatus {
  online: boolean; seen: boolean; os?: "linux" | "windows"; host?: string; platform?: string; repo?: string;
  running?: string | null; age_s?: number; started?: string;
}
export interface SettingItem {
  key: string; label: string; type: string; help: string; note: string | null; default: string | null;
  value: string | null; is_set: boolean; secret: boolean; restart: string[]; approval: boolean; locked: string | null;
  options: string[]; min: number | null; max: number | null; warn: string | null;
  option_modes: Record<string, string> | null;
}
export interface SettingsResponse {
  env_path: string; env_exists: boolean; groups: { name: string; items: SettingItem[] }[]; unknown_keys: string[];
  serving: ServingMode; agent: AgentStatus;
}
export interface ChangeEntry {
  at: string; kind: string; summary: string; details: Record<string, unknown>;
  requested_by?: string; approved_by?: string; reason?: string;
}
export interface OpsParam {
  type: string; label: string; required?: boolean; options?: string[]; min?: number; max?: number; default?: unknown;
}
export interface OpsActionDef {
  name: string; label: string; group: string; description: string; params: Record<string, OpsParam>;
  linux: boolean; windows: boolean; approval_needed: boolean; timeout_min: number | null;
}
export interface OpsCatalog {
  actions: OpsActionDef[]; agent: AgentStatus; serving: ServingMode; runs: string[]; profiles: string[];
  defaults: { run: string | null; config: string | null; profile: string | null };
}
export interface OpsAction {
  id: string; action: string; label: string; params: Record<string, unknown>;
  commands: { linux: string | null; windows: string | null }; command?: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled"; created: string; started: string | null;
  finished: string | null; exit_code: number | null; message: string | null; cancel_requested?: boolean;
  requested_by?: string; approved_by?: string; reason?: string; origin?: string; agent_os?: string;
}
export interface WorkflowRun {
  id: number; status: string; conclusion: string | null; html_url: string; created_at: string; event: string;
  head_branch: string; display_title: string; sha: string;
}
export interface AwsStatus {
  serving: ServingMode; github: { ok: boolean; repo?: string; error?: string };
  variables: Record<string, string | null | undefined>; runs: Record<string, WorkflowRun[]>;
  stream: { url: string; remote: boolean; up: boolean }; bootstrapped?: boolean; stack?: string;
}
