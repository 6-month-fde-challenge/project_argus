"""Stage 6a - Distil the teacher scorer into a much smaller student scorer (Part 3: the ~0.5B scorer).

The student learns from hard labels and from the teacher's soft per-head probabilities (temperature-scaled).
Its adapter and heads are merged into standalone weights, which stage 7 quantises. The student has its own
tokenizer, so prompts are fitted to its token budget.
"""
from __future__ import annotations

import numpy as np

from .. import data as D
from ..metrics import scorer_metrics
from ..modeling import free_gpu, load_model, load_tokenizer, merge_adapter, predict_logits, save_adapter
from .common import train_scorer


def run(ctx) -> dict:
    c, cfg = ctx.cfg.distill_scorer, ctx.cfg
    student_base = cfg.models.student_scorer
    tok = load_tokenizer(student_base)
    head_kw = dict(num_labels=D.NUM_HEADS, pad_token_id=tok.pad_token_id)

    rows = ctx.fitted(ctx.scorer_train_rows("distill_scorer"), tok)
    ids, teacher_logits = ctx.load_preds("teacher_distill_train")
    assert list(ids) == [r["transaction_id"] for r in rows], "teacher logits do not match the distillation rows"

    model = train_scorer(ctx, "scorer_student", student_base, tok, rows, c, lora={**cfg.lora, **c.lora},
                         teacher_logits=teacher_logits, alpha=c.alpha, temperature=c.temperature)
    adapter = ctx.path("scorer_student", "adapter")
    save_adapter(model, tok, adapter)
    del model
    free_gpu()

    merged = ctx.path("scorer_student", "merged")
    merge_adapter(student_base, str(adapter), merged, "scorer", tok, **head_kw)
    student = load_model(str(merged), "scorer", **head_kw)
    val = ctx.fitted(ctx.sft("validation"), tok)
    val_logits = predict_logits(student, tok, [r["prompt"] for r in val], cfg.max_length, cfg.eval.batch_size,
                                progress=ctx.events.progress("scoring validation"))
    ctx.save_preds("scorer_student_val", [r["transaction_id"] for r in val], val_logits)

    return {
        "student": student_base, "train_rows": len(rows), "alpha_hard": c.alpha, "temperature": c.temperature,
        "validation": scorer_metrics(val_logits, np.array([D.head_targets(r) for r in val])),
    }
