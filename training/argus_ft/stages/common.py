"""Training pieces shared by the stages: Trainer arguments and the multi-head scorer training used by
the pilot (2), the SFT teacher scorer (4a) and the distilled student (6)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .. import data as D
from ..modeling import DTYPE, add_lora, free_gpu, load_model, quant_for


def training_args(ctx, stage: str, c, causal: bool) -> TrainingArguments:
    """One place for every Trainer setting; speed/memory switches come from the `efficiency` config section.
    `causal`: text-generation model. Liger's fused linear+cross-entropy replaces the model's forward with a
    language-model one, so classification models (scorer heads) get only the norm / MLP / RoPE kernels."""
    eff = ctx.cfg.efficiency
    return TrainingArguments(
        output_dir=str(ctx.path(stage, "trainer")),
        num_train_epochs=c.get("epochs", 1),
        max_steps=c.get("max_steps", -1),
        per_device_train_batch_size=c["batch_size"],
        per_device_eval_batch_size=c["batch_size"],
        gradient_accumulation_steps=c["grad_accum"],
        learning_rate=c["lr"],
        warmup_steps=0.05,                    # a float in [0, 1) is a ratio of total steps (transformers v5)
        lr_scheduler_type="cosine",
        # QLoRA: paged 8-bit optimizer states; LoRA: fused AdamW kernel
        optim="paged_adamw_8bit" if c.get("method") == "qlora" else "adamw_torch_fused",
        bf16=DTYPE == torch.bfloat16,
        tf32=eff.tf32,                        # TF32 matmuls on Ampere+ for any fp32 math
        use_liger_kernel=eff.liger,           # fused RMSNorm / SwiGLU / RoPE (+ linear+cross-entropy for causal)
        liger_kernel_config=None if causal else {"fused_linear_cross_entropy": False, "cross_entropy": False},
        # batches of similar length -> less padding
        train_sampling_strategy="group_by_length" if eff.group_by_length else "random",
        dataloader_num_workers=eff.dataloader_workers,
        dataloader_pin_memory=True,
        # loss every 5 steps; every step for very short runs (smoke) so the live chart still gets points
        logging_steps=1 if 0 < c.get("max_steps", -1) <= 10 else 5,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=ctx.cfg.seed,
    )


def head_pos_weight(targets: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """Per-head neg/pos ratio for BCE (class imbalance, FR-ML-04)."""
    pos = targets.sum(0)
    return torch.tensor(np.clip((len(targets) - pos) / np.maximum(pos, 1), 1.0, cap), dtype=torch.float32)


def _heads(bce: torch.Tensor, reason_weight: float) -> torch.Tensor:
    return (bce[:, 0] + reason_weight * bce[:, 1:].mean(1)).mean()


class MultiHeadTrainer(Trainer):
    """Fraud head + 8 reason-code heads with weighted BCE. When the batch carries `teacher_logits`,
    the loss blends in knowledge distillation: alpha * hard-label loss + (1 - alpha) * soft-target loss."""

    def __init__(self, *args, pos_weight: torch.Tensor, reason_weight: float,
                 alpha: float = 1.0, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight, self.reason_weight = pos_weight, reason_weight
        self.alpha, self.temperature = alpha, temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        teacher = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)
        logits = outputs.logits.float()
        hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight.to(logits.device), reduction="none")
        loss = _heads(hard, self.reason_weight)
        if teacher is not None:
            t = self.temperature
            soft = F.binary_cross_entropy_with_logits(logits / t, torch.sigmoid(teacher.float() / t), reduction="none") * t * t
            loss = self.alpha * loss + (1 - self.alpha) * _heads(soft, self.reason_weight)
        return (loss, outputs) if return_outputs else loss


def train_scorer(ctx, stage: str, base: str, tok, rows: list[dict], c, *, method: str = "lora", lora: dict | None = None,
                 teacher_logits: np.ndarray | None = None, alpha: float = 1.0, temperature: float = 1.0):
    """Fine-tune `base` with LoRA + fraud/reason heads on pre-fitted `rows`; returns the trained model.
    Passing `teacher_logits` turns on knowledge distillation (stage 6)."""
    extra = {"teacher_logits": teacher_logits.tolist()} if teacher_logits is not None else None
    ds = D.build_scorer_dataset(rows, tok, ctx.cfg.max_length, extra=extra)
    model = load_model(base, "scorer", quant_for(method), num_labels=D.NUM_HEADS, pad_token_id=tok.pad_token_id)
    model = add_lora(model, "scorer", method, lora or ctx.cfg.lora, c.get("gradient_checkpointing", True))
    trainer = MultiHeadTrainer(
        model=model, args=training_args(ctx, stage, c, causal=False), train_dataset=ds,
        data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=8),
        pos_weight=head_pos_weight(np.array(ds["labels"])), reason_weight=ctx.cfg.sft_scorer.reason_weight,
        alpha=alpha, temperature=temperature, callbacks=[ctx.events.trainer_callback()],
    )
    trainer.train()
    del trainer  # drop optimizer state and cached activations before the model is used for scoring
    free_gpu()
    return model
