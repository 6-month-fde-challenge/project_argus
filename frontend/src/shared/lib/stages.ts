/** The ARGUS-LM model ladder (Part 3, Figure 4): one place for every stage's label, figure number and story. */
export type StageId =
  | "data_checks" | "xgboost" | "pilot" | "gate" | "cpt" | "sft_scorer" | "sft_explainer" | "dpo"
  | "distill_scorer" | "distill_explainer" | "quantize" | "calibrate" | "acceptance" | "registry";

export interface StageMeta {
  id: StageId;
  step: string;           // number shown in Figure 4
  label: string;
  short: string;
  group: "foundation" | "decide" | "adapt" | "compress" | "ship";
  branch: "both" | "llm-scorer" | "explainer";
  what: string;
  doc: string;            // where it comes from in the client docs
}

export const STAGES: StageMeta[] = [
  { id: "data_checks", step: "0", label: "Data checks", short: "Checks", group: "foundation", branch: "both",
    what: "Leakage tests (time order, disjoint splits, no holdout in training files) stop the run on failure; reports missing rates and prompt token lengths.",
    doc: "Part 2 TR-DATA-02/04, CRD DR-02" },
  { id: "xgboost", step: "1", label: "XGBoost baseline", short: "XGBoost", group: "foundation", branch: "both",
    what: "Gradient-boosted trees on the 10 engineered features from Supabase: the bar the LLM must beat, and the HYBRID scorer.",
    doc: "CRD FR-ML-01, Part 3 Step 2" },
  { id: "pilot", step: "2", label: "Pilot · 3 models × 2 formats", short: "Pilot", group: "decide", branch: "both",
    what: "Every candidate model is fine-tuned briefly on both prompt formats; the best (model, format) pair by validation PR-AUC becomes the teacher.",
    doc: "Part 3 Figure 4, TR-LLM-01, FR-FE-07" },
  { id: "gate", step: "W8", label: "Week-8 gate", short: "Gate", group: "decide", branch: "both",
    what: "Pilot winner vs XGBoost on the same rows: +0.03 PR-AUC continues the LLM path, otherwise HYBRID (XGBoost scores, LLM explains).",
    doc: "Part 3 Figure 4, OBJ-04" },
  { id: "cpt", step: "3", label: "Continued pre-training", short: "CPT", group: "adapt", branch: "both",
    what: "LoRA next-token training on investigator notes, chargeback narratives, playbooks and regulatory exercises.",
    doc: "CRD 9.1, TR-LLM-03" },
  { id: "sft_scorer", step: "4a", label: "SFT scorer · fraud + reason heads", short: "SFT scorer", group: "adapt", branch: "llm-scorer",
    what: "QLoRA fine-tune with a fraud head and 8 reason-code heads; probability always from a head, never from parsed text.",
    doc: "FR-ML-03/04, TR-LLM-04" },
  { id: "sft_explainer", step: "4b", label: "SFT explainer · JSON", short: "SFT explainer", group: "adapt", branch: "explainer",
    what: "LoRA fine-tune that writes {reason_codes, explanation} JSON from the transaction prompt; loss on the answer only.",
    doc: "FR-ML-07, OBJ-06" },
  { id: "dpo", step: "5", label: "DPO · preference pairs", short: "DPO", group: "adapt", branch: "explainer",
    what: "Preference tuning on chosen vs rejected answers with an anchoring NLL term; kept only if it beats SFT on held-out cases.",
    doc: "TR-LLM-05" },
  { id: "distill_scorer", step: "6a", label: "Distil scorer", short: "Distil scorer", group: "compress", branch: "llm-scorer",
    what: "Teacher scorer -> small student (hard labels + temperature-scaled soft targets): the ~0.5B production scorer.",
    doc: "TR-LLM-06, Part 3 capacity plan" },
  { id: "distill_explainer", step: "6b", label: "Distil explainer", short: "Distil explainer", group: "compress", branch: "explainer",
    what: "Teacher answers that pass the grounding guard train a small explainer student: the ~1.5B production explainer.",
    doc: "TR-LLM-06" },
  { id: "quantize", step: "7", label: "Quantise · FP8 + AWQ", short: "Quantise", group: "compress", branch: "both",
    what: "llm-compressor: scorer student to FP8, explainer student to AWQ 4-bit, both in vLLM-servable compressed-tensors format.",
    doc: "TR-LLM-06, Part 3 Figure 4" },
  { id: "calibrate", step: "8", label: "Calibrate + thresholds", short: "Calibrate", group: "ship", branch: "both",
    what: "Platt vs isotonic (lower ECE wins); PR-AUC, recall at review capacity, threshold sweep, fairness by age band and region.",
    doc: "FR-ML-06/11/12, TR-ML-02..05" },
  { id: "acceptance", step: "9", label: "Acceptance · holdout", short: "Acceptance", group: "ship", branch: "both",
    what: "The served scorer on the untouched Jan–Feb 2026 holdout with its frozen calibrator: OBJ checklist, recommended REVIEW / BLOCK thresholds, PR / ROC curves, p99 latency, GO / NO-GO.",
    doc: "CRD 4 OBJ-01..10, CRD 15 acceptance gate 1" },
  { id: "registry", step: "R", label: "Registry · MLflow", short: "Registry", group: "ship", branch: "both",
    what: "Config, data snapshot, every metric and the serving models logged and registered (argus-scorer, argus-explainer).",
    doc: "FR-ML-08, TR-DEP-03" },
];

export const STAGE_BY_ID: Record<string, StageMeta> = Object.fromEntries(STAGES.map((s) => [s.id, s]));
export const stageLabel = (id: string) => STAGE_BY_ID[id]?.label ?? id;
