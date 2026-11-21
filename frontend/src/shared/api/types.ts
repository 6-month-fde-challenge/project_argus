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

export interface Gate { decision: string; mode?: string; branch?: string; uplift: number; llm_pr_auc: number; xgboost_pr_auc: number }

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
}
export interface ProfileDetail extends ProfileSummary { yaml: string; resolved: Record<string, any> }

export interface ServicesStatus {
  platform: string; docker: string; containers: Record<string, string>; scoring: string; explainer: string;
  mlflow: string; mlflow_url: string; gpu_jobs: string[];
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
