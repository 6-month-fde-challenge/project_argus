"""Stage 4a - SFT of the teacher scorer: fraud head + 8 reason-code heads on the final hidden state.

The fraud probability comes from a head, never from parsing generated text (FR-ML-03). Genuine rows are
down-sampled and the remaining imbalance is handled with per-head BCE weights (FR-ML-04).
After training it scores (a) the validation set and (b) the student's training rows - the soft targets
that stage 6a distils from.
"""
from __future__ import annotations

import numpy as np

from .. import data as D
from ..metrics import scorer_metrics
from ..modeling import predict_logits, save_adapter
from .common import train_scorer


def run(ctx) -> dict:
    c, cfg, tok = ctx.cfg.sft_scorer, ctx.cfg, ctx.tokenizer
    rows = ctx.fitted(ctx.scorer_train_rows("sft_scorer"))
    model = train_scorer(ctx, "scorer_teacher", ctx.teacher_base, tok, rows, c, method=c.method)
    save_adapter(model, tok, ctx.path("scorer_teacher", "adapter"))

    def score(rows_: list[dict], label: str) -> np.ndarray:
        return predict_logits(model, tok, [r["prompt"] for r in rows_], cfg.max_length, cfg.eval.batch_size,
                              progress=ctx.events.progress(label))

    val = ctx.fitted(ctx.sft("validation"))
    val_logits = score(val, "scoring validation")
    ctx.save_preds("teacher_val", [r["transaction_id"] for r in val], val_logits)

    distill_rows = ctx.fitted(ctx.scorer_train_rows("distill_scorer"))
    ctx.save_preds("teacher_distill_train", [r["transaction_id"] for r in distill_rows], score(distill_rows, "soft targets for the student"))

    return {
        "base": ctx.teacher_base, "method": c.method,
        "train_rows": len(rows), "train_fraud": sum(r["fraud_label"] for r in rows),
        "validation": scorer_metrics(val_logits, np.array([D.head_targets(r) for r in val])),
    }
