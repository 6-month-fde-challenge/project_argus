"""Stage 4b - SFT of the explainer: transaction prompt -> JSON {reason_codes, explanation} (FR-ML-07).

Trained on the gold explanation set; loss covers only the answer tokens. The latest gold transactions are
held out to measure JSON validity (OBJ-06) and reason-code agreement on generated answers. The training and
evaluation helpers here are shared with DPO (5), explainer distillation (6b) and quantisation (7).
"""
from __future__ import annotations

import random
from collections import Counter

from transformers import DataCollatorForSeq2Seq, Trainer

from .. import data as D
from ..grounding import ExplanationGuard
from ..metrics import explanation_metrics
from ..modeling import add_lora, free_gpu, generate, load_model, quant_for, save_adapter
from .common import training_args

CHAT_OVERHEAD = 96  # system prompt + chat-template tokens around the transaction prompt


def explainer_max_length(cfg) -> int:
    return cfg.max_length + CHAT_OVERHEAD + cfg.sft_explainer.answer_budget


def train_explainer(ctx, stage: str, base: str, tok, rows: list[dict], c, answer_key: str = "target"):
    """LoRA SFT of a chat model on (prompt, JSON answer) rows; returns the trained model."""
    ds = D.build_explainer_dataset(ctx.fitted(rows, tok), tok, explainer_max_length(ctx.cfg), answer_key)
    model = load_model(base, "causal", quant_for(c.method))
    model = add_lora(model, "causal", c.method, {**ctx.cfg.lora, **c.get("lora", {})}, c.gradient_checkpointing)
    trainer = Trainer(
        model=model, args=training_args(ctx, stage, c, causal=True), train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[ctx.events.trainer_callback()],
    )
    trainer.train()
    del trainer
    free_gpu()
    return model


def explainer_eval_rows(ctx, tok=None) -> list[dict]:
    """The held-out gold cases (latest transactions) every explainer variant is judged on."""
    c = ctx.cfg.sft_explainer
    _, val_rows = D.load_gold(ctx.cfg.data_dir, c.val_rows)
    return ctx.fitted(val_rows[: c.eval_rows], tok)


def score_explanations(ctx, rows: list[dict], gens: list[str]) -> tuple[dict, list[dict]]:
    """Raw model quality (JSON validity, reason codes) plus what the guardrail did (TR-LLM-07)."""
    train_rows, _ = D.load_gold(ctx.cfg.data_dir, ctx.cfg.sft_explainer.val_rows)
    guard = ExplanationGuard(train_rows)
    # fallback codes: the gold codes stand in for the scorer's reason-code heads used in production
    guarded = [guard.guard(g, r["prompt"], r["target"]["reason_codes"]) for r, g in zip(rows, gens)]
    sources = Counter(src for _, src in guarded)
    metrics = {
        **explanation_metrics(gens, [r["target"] for r in rows]),
        "guard_accepted_rate": round(sources["model"] / max(len(rows), 1), 4),
        "guard_fallbacks": {k: v for k, v in sources.items() if k != "model"},
    }
    samples = [{"prompt": r["prompt"], "target": r["target"], "generated": g, "final": a, "source": s}
               for r, g, (a, s) in zip(rows[:10], gens, guarded)]
    return metrics, samples


def evaluate_explainer(ctx, model, tok=None) -> tuple[dict, list[dict]]:
    """Generate with transformers for the held-out gold cases and score the answers."""
    tok = tok or ctx.tokenizer
    rows = explainer_eval_rows(ctx, tok)
    prompts = [D.chat_prefix(tok, r["prompt"]) for r in rows]
    gens = generate(model, tok, prompts, ctx.cfg.eval.batch_size, ctx.cfg.sft_explainer.answer_budget,
                    progress=ctx.events.progress("generating explanations"))
    return score_explanations(ctx, rows, gens)


def run(ctx) -> dict:
    c, tok = ctx.cfg.sft_explainer, ctx.tokenizer
    train_rows, _ = D.load_gold(ctx.cfg.data_dir, c.val_rows)
    train_rows = random.Random(ctx.cfg.seed).sample(train_rows, min(c.max_rows, len(train_rows)))

    model = train_explainer(ctx, "explainer_sft", ctx.teacher_base, tok, train_rows, c)
    save_adapter(model, tok, ctx.path("explainer", "sft_adapter"))

    metrics, samples = evaluate_explainer(ctx, model)
    ctx.save_json("explainer/sft_samples.json", samples)
    return {"base": ctx.teacher_base, "method": c.method, "train_rows": len(train_rows), "validation": metrics}
