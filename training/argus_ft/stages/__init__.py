"""The ARGUS-LM model ladder (Part 3, Section 5 / Figure 4), one module per stage, in run order.

    0 Data checks --> 1 XGBoost --> 2 Pilot (3 models x 2 formats) --> Week-8 gate --no--> Hybrid: XGBoost scores
                                                                           | yes
    3 CPT --> 4 SFT LoRA (4a fraud + reason heads, 4b explainer) --> 5 DPO
    6 Distil (6a scorer, 6b explainer) --> 7 Quantise (FP8, AWQ 4-bit) --> 8 Calibrate --> Registry (MLflow)

On "no", the gate skips the LLM scorer stages (4a, 6a); the explainer branch still runs ("the LLM explains").
"""
from . import (calibrate, cpt, data_checks, distill_explainer, distill_scorer, dpo, gate, pilot, quantize, registry,
               sft_explainer, sft_scorer, xgboost_baseline)

LADDER = {
    "data_checks":       data_checks.run,        # 0  leakage tests (fail the run) + data-quality report
    "xgboost":           xgboost_baseline.run,   # 1  baseline on DB features - the bar to beat
    "pilot":             pilot.run,              # 2  candidate models x prompt formats, pick the best pair
    "gate":              gate.run,               #    week 8: +0.03 PR-AUC over XGBoost, else HYBRID
    "cpt":               cpt.run,                # 3  continued pre-training on the fraud corpus
    "sft_scorer":        sft_scorer.run,         # 4a SFT: fraud head + 8 reason-code heads
    "sft_explainer":     sft_explainer.run,      # 4b SFT: JSON reason codes + explanation
    "dpo":               dpo.run,                # 5  preference tuning of the explainer
    "distill_scorer":    distill_scorer.run,     # 6a teacher scorer -> small scorer student
    "distill_explainer": distill_explainer.run,  # 6b DPO explainer -> small explainer student
    "quantize":          quantize.run,           # 7  FP8 scorer student, AWQ 4-bit explainer student
    "calibrate":         calibrate.run,          # 8  Platt / isotonic + thresholds + policy + fairness
    "registry":          registry.run,           #    data snapshot, config, metrics, models -> MLflow
}
