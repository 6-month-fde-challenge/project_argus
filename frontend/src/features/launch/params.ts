/** The tunable knobs shown in the parameter editor, derived from the resolved profile (no hard-coded values). */
export interface Param {
  key: string;           // dotted path in the config, e.g. "sft_scorer.lr"
  label: string;
  kind: "number" | "select" | "text";
  options?: string[];
  step?: number;
}

const STAGE_KNOBS: Record<string, string[]> = {
  pilot: ["max_steps", "neg_per_pos", "val_rows", "batch_size", "lr"],
  cpt: ["method", "max_docs", "epochs", "max_steps", "batch_size", "grad_accum", "lr"],
  sft_scorer: ["method", "neg_per_pos", "epochs", "max_steps", "batch_size", "grad_accum", "lr"],
  sft_explainer: ["method", "max_rows", "epochs", "max_steps", "batch_size", "grad_accum", "lr"],
  dpo: ["max_pairs", "epochs", "beta", "rpo_alpha", "lr"],
  distill_scorer: ["neg_per_pos", "epochs", "max_steps", "alpha", "temperature", "lr"],
  distill_explainer: ["max_rows", "epochs", "max_steps", "lr"],
  quantize: ["calibration_samples", "awq_group_size"],
};

export const GROUPS: { id: string; label: string; params: Param[] }[] = [
  { id: "general", label: "General", params: [
    { key: "max_length", label: "prompt token cap", kind: "number" },
    { key: "eval.max_rows", label: "validation rows (null = all)", kind: "number" },
    { key: "gate.mode", label: "week-8 gate mode", kind: "select", options: ["auto", "hybrid", "llm"] },
    { key: "gate.min_uplift", label: "required PR-AUC uplift", kind: "number", step: 0.01 },
  ] },
  { id: "models", label: "Models", params: [
    { key: "models.teacher", label: "fallback teacher", kind: "text" },
    { key: "models.student_scorer", label: "scorer student", kind: "text" },
    { key: "models.student_explainer", label: "explainer student", kind: "text" },
  ] },
  ...Object.entries(STAGE_KNOBS).map(([stage, knobs]) => ({
    id: stage, label: stage.replaceAll("_", " "),
    params: knobs.map((k): Param => k === "method"
      ? { key: `${stage}.${k}`, label: "method", kind: "select", options: ["lora", "qlora"] }
      : { key: `${stage}.${k}`, label: k.replaceAll("_", " "), kind: "number", step: k === "lr" ? 1e-5 : k === "beta" || k === "alpha" ? 0.05 : 1 }),
  })),
];

export const getPath = (obj: any, key: string) => key.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);

/** Overrides = only the values the user actually changed, as `key=value` strings for the pipeline's --set. */
export function diffOverrides(resolved: any, edits: Record<string, string>): string[] {
  return Object.entries(edits).flatMap(([key, raw]) => {
    const original = getPath(resolved, key);
    if (raw === "" || raw === String(original ?? "")) return [];
    return [`${key}=${raw}`];
  });
}
