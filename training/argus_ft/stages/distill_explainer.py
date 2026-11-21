"""Stage 6b - Distil the DPO explainer into a smaller student explainer (Part 3: the ~1.5B explainer).

Sequence-level distillation: the teacher (explainer/merged from stage 5) answers gold-training prompts; an
answer is kept only if the guard accepts it (valid schema, grounded numbers), otherwise the gold answer is
used. The student is LoRA fine-tuned on these answers and merged into standalone weights for stage 7.
"""
from __future__ import annotations

import json
import random

from .. import data as D
from ..grounding import ExplanationGuard
from ..modeling import free_gpu, generate, load_model, load_tokenizer, merge_adapter, save_adapter
from .sft_explainer import evaluate_explainer, train_explainer


def teacher_answers(ctx, rows: list[dict]) -> tuple[list[dict], int]:
    """Rows with an `answer` field (teacher's if accepted by the guard, else gold); returns the accepted count."""
    tok, teacher_dir = ctx.tokenizer, str(ctx.run_dir / "explainer" / "merged")
    teacher = load_model(teacher_dir, "causal")
    fitted = ctx.fitted(rows)
    gens = generate(teacher, tok, [D.chat_prefix(tok, r["prompt"]) for r in fitted], ctx.cfg.eval.batch_size,
                    ctx.cfg.sft_explainer.answer_budget, progress=ctx.events.progress("teacher answers"))
    del teacher
    free_gpu()

    guard, accepted, out = ExplanationGuard(rows), 0, []
    for raw, fit, gen in zip(rows, fitted, gens):
        obj, status = guard.check(gen, fit["prompt"])
        accepted += status == "ok"
        out.append({**raw, "answer": json.dumps(obj, ensure_ascii=False) if obj else D.target_to_json(raw["target"])})
    return out, accepted


def run(ctx) -> dict:
    c, cfg = ctx.cfg.distill_explainer, ctx.cfg
    student_base = cfg.models.student_explainer
    gold_train, _ = D.load_gold(cfg.data_dir, cfg.sft_explainer.val_rows)
    rows = random.Random(cfg.seed + 2).sample(gold_train, min(c.max_rows, len(gold_train)))
    rows, accepted = teacher_answers(ctx, rows)

    tok = load_tokenizer(student_base)
    model = train_explainer(ctx, "explainer_student", student_base, tok, rows, c, answer_key="answer")
    adapter = ctx.path("explainer_student", "adapter")
    save_adapter(model, tok, adapter)
    metrics, samples = evaluate_explainer(ctx, model, tok)
    ctx.save_json("explainer_student/samples.json", samples)
    del model
    free_gpu()
    merge_adapter(student_base, str(adapter), ctx.path("explainer_student", "merged"), "causal", tok)

    return {"student": student_base, "train_rows": len(rows), "teacher_answers_accepted": accepted,
            "validation": metrics}
