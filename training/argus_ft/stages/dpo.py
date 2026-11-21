"""Stage 5 - DPO on reviewer preference pairs, so explanations stay grounded in the input fields.

Continues training the SFT explainer adapter. The frozen SFT adapter is the reference policy; its
log-probabilities are computed once up front. Loss (Rafailov et al., 2023):
    -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])
plus `rpo_alpha` x the length-normalised NLL of the chosen answer (the "RPO" term, TRL's `rpo_alpha`). Without
it, easy preference pairs drive the DPO loss to 0 and the policy drifts away from well-formed answers.
Memory: sequences are left-padded so every answer ends at the last position, and the model computes vocabulary
logits only for the answer span (`logits_to_keep`) instead of the whole prompt.
Keep-the-best: the DPO explainer replaces the SFT explainer only if it is at least as good on the held-out gold
cases (guard acceptance, then reason-code F1). The kept one is merged into explainer/merged - the stage-6b teacher.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from .. import data as D
from ..modeling import DTYPE, GC_KWARGS, free_gpu, load_with_adapter, merge_adapter, quant_for, save_adapter
from .sft_explainer import evaluate_explainer, explainer_max_length


def _encode_pairs(ctx, pairs: list[dict]) -> list[tuple[dict, dict]]:
    tok, max_len = ctx.tokenizer, explainer_max_length(ctx.cfg)
    return [
        (D.encode_prompt_answer(tok, p["prompt"], p["chosen"], max_len), D.encode_prompt_answer(tok, p["prompt"], p["rejected"], max_len))
        for p in ctx.fitted(pairs)
    ]


def _pad_left(encs: list[dict], pad_id: int, device) -> dict:
    n = max(len(e["input_ids"]) for e in encs)
    pad = lambda key, value: torch.tensor([[value] * (n - len(e[key])) + e[key] for e in encs], device=device)
    batch = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)  # real positions despite left padding
    return batch


def _logps(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probability of the answer tokens (labels != -100) per sequence, and the answer lengths."""
    n_answer = int((batch["labels"] != -100).sum(-1).max())
    with torch.autocast("cuda", dtype=DTYPE):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       position_ids=batch["position_ids"], logits_to_keep=n_answer + 1).logits[:, :-1].float()
    target = batch["labels"][:, -n_answer:]
    token_lp = torch.gather(logits, 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    mask = target != -100
    return (token_lp * mask).sum(-1), mask.sum(-1)


def _pair_logps(model, pairs, pad_id, bs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chosen and rejected log-probs, plus chosen answer lengths; both halves of a pair share one padded batch."""
    chosen, rejected, chosen_len = [], [], []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i : i + bs]
        lp, n = _logps(model, _pad_left([c for c, _ in chunk] + [r for _, r in chunk], pad_id, model.device))
        chosen.append(lp[: len(chunk)])
        rejected.append(lp[len(chunk):])
        chosen_len.append(n[: len(chunk)])
    return torch.cat(chosen), torch.cat(rejected), torch.cat(chosen_len)


def train_dpo(model, pairs, ref_c, ref_r, c, pad_id: int, seed: int, events=None) -> list[float]:
    """Plain PyTorch loop: only the LoRA weights are trainable; returns the per-batch losses."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=c.lr)
    model.train()
    losses = []
    for epoch in range(c.epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(pairs))
        for step, i in enumerate(range(0, len(order), c.batch_size), 1):
            idx = torch.as_tensor(order[i : i + c.batch_size], device=ref_c.device)
            pol_c, pol_r, n_c = _pair_logps(model, [pairs[j] for j in idx.tolist()], pad_id, c.batch_size)
            margin = (pol_c - ref_c[idx]) - (pol_r - ref_r[idx])
            nll_chosen = -(pol_c / n_c.clamp(min=1)).mean()  # keeps the policy on well-formed, grounded answers
            loss = -F.logsigmoid(c.beta * margin).mean() + c.rpo_alpha * nll_chosen
            (loss / c.grad_accum).backward()
            losses.append(loss.item())
            if step % c.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                if events:
                    events.emit("train_step", step=step // c.grad_accum,
                                max_steps=len(order) // (c.batch_size * c.grad_accum), epoch=epoch + i / len(order),
                                loss=float(np.mean(losses[-c.grad_accum:])), learning_rate=c.lr)
            if step % 20 == 0:
                print(f"dpo step {step}: loss {np.mean(losses[-20:]):.4f}")
    model.eval()
    return losses


@torch.no_grad()
def frozen_logps(model, pairs, pad_id: int, bs: int):
    model.eval()
    chosen, rejected, _ = _pair_logps(model, pairs, pad_id, bs)
    return chosen, rejected


def _quality(metrics: dict) -> tuple[float, float]:
    return metrics["guard_accepted_rate"], metrics["reason_f1"]


def run(ctx) -> dict:
    c, tok = ctx.cfg.dpo, ctx.tokenizer
    pad, bs = tok.pad_token_id, c.batch_size
    train_pairs, eval_pairs = D.load_dpo(ctx.cfg.data_dir, c.eval_pairs)
    train_pairs = random.Random(ctx.cfg.seed).sample(train_pairs, min(c.max_pairs, len(train_pairs)))
    train_enc, eval_enc = _encode_pairs(ctx, train_pairs), _encode_pairs(ctx, eval_pairs)

    # policy = SFT explainer with its adapter trainable; reference = the same model before any DPO step
    model = load_with_adapter(ctx.teacher_base, str(ctx.run_dir / "explainer" / "sft_adapter"), "causal",
                              quant_for(c.method), trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
    model.enable_input_require_grads()
    model.config.use_cache = False

    ref_c, ref_r = frozen_logps(model, train_enc, pad, bs)
    ev_ref_c, ev_ref_r = frozen_logps(model, eval_enc, pad, bs)

    losses = train_dpo(model, train_enc, ref_c, ref_r, c, pad, ctx.cfg.seed, ctx.events)
    free_gpu()  # optimizer state is gone once train_dpo returns
    save_adapter(model, tok, ctx.path("explainer", "dpo_adapter"))

    ev_c, ev_r = frozen_logps(model, eval_enc, pad, bs)
    metrics, samples = evaluate_explainer(ctx, model)
    ctx.save_json("explainer/dpo_samples.json", samples)
    del model
    free_gpu()

    # keep-the-best: DPO must not make the explainer worse than SFT on the held-out gold cases
    report = json.loads((ctx.run_dir / "report.json").read_text())
    sft_metrics = report["stages"]["sft_explainer"]["validation"]
    kept = "dpo" if _quality(metrics) >= _quality(sft_metrics) else "sft"
    merge_adapter(ctx.teacher_base, str(ctx.run_dir / "explainer" / f"{kept}_adapter"), ctx.path("explainer", "merged"),
                  "causal", tok)
    return {
        "kept": kept, "sft_validation": {k: sft_metrics[k] for k in ("guard_accepted_rate", "reason_f1")},
        "train_pairs": len(train_enc), "eval_pairs": len(eval_enc),
        "loss_first_20": round(float(np.mean(losses[:20])), 4), "loss_last_20": round(float(np.mean(losses[-20:])), 4),
        "heldout_prefers_chosen_before": round(float((ev_ref_c > ev_ref_r).float().mean()), 4),
        "heldout_prefers_chosen_after": round(float((ev_c > ev_r).float().mean()), 4),
        "heldout_reward_accuracy": round(float((((ev_c - ev_ref_c) - (ev_r - ev_ref_r)) > 0).float().mean()), 4),
        "validation": metrics,
    }
