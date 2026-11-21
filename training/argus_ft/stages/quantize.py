"""Stage 7 - Quantise the two students for serving (Part 3 Figure 4: "FP8 · AWQ 4-bit"), with llm-compressor.

  scorer student    -> FP8 (W8A8, dynamic per-token activations; needs no calibration data). Native FP8 maths
                       on Ada/Hopper (the L4 serving fleet); Ampere runs it as FP8 weight-only.
  explainer student -> AWQ 4-bit weights (W4A16), calibrated on gold-training conversations.
Both are saved in compressed-tensors format, which vLLM serves directly. This stage measures what quantisation
costs in accuracy and saves in size; serving latency belongs to the serving stack (see serve/).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier as AWQTransform

from .. import data as D
from ..metrics import scorer_metrics
from ..modeling import free_gpu, load_model, load_tokenizer, predict_logits
from .sft_explainer import evaluate_explainer, explainer_max_length


def size_mb(model_dir: Path) -> float:
    return round(sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors")) / 1e6, 1)


def _save(model, tok, out_dir: Path) -> None:
    model.save_pretrained(out_dir, save_compressed=True)
    tok.save_pretrained(out_dir)


def quantize_scorer(ctx) -> dict:
    src, out = ctx.run_dir / "scorer_student" / "merged", ctx.path("scorer_student", "fp8")
    tok = load_tokenizer(str(src))
    head_kw = dict(num_labels=D.NUM_HEADS, pad_token_id=tok.pad_token_id)

    model = load_model(str(src), "scorer", **head_kw)
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["score"]))
    _save(model, tok, out)
    del model
    free_gpu()

    quantized = load_model(str(out), "scorer", **head_kw)
    val = ctx.fitted(ctx.sft("validation"), tok)
    targets = np.array([D.head_targets(r) for r in val])
    logits = predict_logits(quantized, tok, [r["prompt"] for r in val], ctx.cfg.max_length, ctx.cfg.eval.batch_size,
                            progress=ctx.events.progress("scoring validation (FP8)"))
    ctx.save_preds("scorer_student_fp8_val", [r["transaction_id"] for r in val], logits)
    del quantized
    free_gpu()
    _, bf16_logits = ctx.load_preds("scorer_student_val")  # stage 6a, same validation rows
    return {
        "bf16": {"size_mb": size_mb(src), "validation": scorer_metrics(bf16_logits, targets)},
        "fp8": {"size_mb": size_mb(out), "validation": scorer_metrics(logits, targets)},
    }


def quantize_explainer(ctx) -> dict:
    c = ctx.cfg.quantize
    src, out = ctx.run_dir / "explainer_student" / "merged", ctx.path("explainer_student", "awq")
    tok = load_tokenizer(str(src))

    # calibration: full conversations (prompt + gold answer) from the training period
    gold_train, _ = D.load_gold(ctx.cfg.data_dir, ctx.cfg.sft_explainer.val_rows)
    calib = ctx.fitted(random.Random(ctx.cfg.seed + 3).sample(gold_train, c.calibration_samples), tok)
    texts = [D.chat_prefix(tok, r["prompt"]) + D.answer_suffix(tok, r["prompt"], D.target_to_json(r["target"])) for r in calib]
    max_len = explainer_max_length(ctx.cfg)
    calib_ds = Dataset.from_dict(dict(tok(texts, truncation=True, max_length=max_len, add_special_tokens=False)))

    int4 = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=4, type="int", symmetric=False, strategy="group", group_size=c.awq_group_size))
    model = load_model(str(src), "causal")
    # AWQ = activation-aware rescaling of weight channels, then plain 4-bit group quantisation
    recipe = [AWQTransform(), QuantizationModifier(config_groups={"int4": int4}, ignore=["lm_head"])]
    oneshot(model=model, dataset=calib_ds, recipe=recipe, max_seq_length=max_len, num_calibration_samples=len(calib_ds))
    _save(model, tok, out)
    del model
    free_gpu()

    quantized = load_model(str(out), "causal")
    metrics, samples = evaluate_explainer(ctx, quantized, tok)
    ctx.save_json("explainer_student/awq_samples.json", samples)
    del quantized
    free_gpu()
    return {"bf16_size_mb": size_mb(src), "awq_size_mb": size_mb(out), "group_size": c.awq_group_size,
            "validation": metrics}


def run(ctx) -> dict:
    report = {}
    if (ctx.run_dir / "scorer_student" / "merged").exists():  # absent when the week-8 gate chose HYBRID
        report["scorer_fp8"] = quantize_scorer(ctx)
    report["explainer_awq"] = quantize_explainer(ctx)
    return report
