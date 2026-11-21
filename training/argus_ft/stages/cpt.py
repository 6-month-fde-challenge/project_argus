"""Stage 2 - Continued pre-training (domain adaptation) on the fraud text corpus.

LoRA on next-token prediction over investigator notes, chargeback narratives, playbooks and regulatory
exercises. The adapter is merged into the base weights, and every later teacher stage starts from them.
"""
from __future__ import annotations

import math

from transformers import DataCollatorForLanguageModeling, Trainer

from .. import data as D
from ..modeling import add_lora, free_gpu, load_model, merge_adapter, quant_for, save_adapter
from .common import training_args


def run(ctx) -> dict:
    c, base, tok = ctx.cfg.cpt, ctx.teacher_model, ctx.tokenizer
    train_ds, eval_ds = D.build_cpt_dataset(ctx.cfg.data_dir, tok, c.max_docs, c.block_size, ctx.cfg.seed)

    model = load_model(base, "causal", quant_for(c.method))
    model = add_lora(model, "causal", c.method, ctx.cfg.lora, c.gradient_checkpointing)
    trainer = Trainer(
        model=model, args=training_args(ctx, "cpt", c, causal=True), train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False), callbacks=[ctx.events.trainer_callback()],
    )
    loss_before = trainer.evaluate()["eval_loss"]
    ctx.events.emit("metric", name="heldout_perplexity_before", value=math.exp(loss_before))
    trainer.train()
    loss_after = trainer.evaluate()["eval_loss"]
    ctx.events.emit("metric", name="heldout_perplexity_after", value=math.exp(loss_after))

    adapter = ctx.path("cpt", "adapter")
    save_adapter(model, tok, adapter)
    del model, trainer
    free_gpu()
    merge_adapter(base, str(adapter), ctx.path("cpt", "merged"), "causal", tok)

    return {
        "train_tokens": len(train_ds) * c.block_size,
        "heldout_perplexity_before": round(math.exp(loss_before), 2),
        "heldout_perplexity_after": round(math.exp(loss_after), 2),
    }
