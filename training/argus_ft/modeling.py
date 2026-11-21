"""Model construction shared by every stage.

Training methods:
  lora   - bf16 base weights + LoRA adapters
  qlora  - 4-bit NF4 base weights (bitsandbytes) + LoRA adapters
Serving formats (FP8 scorer, AWQ 4-bit explainer) are produced by stage 7 with llm-compressor; checkpoints in
those formats load through the same `load_model` (transformers reads their quantization_config).

Efficiency defaults: bf16, PyTorch SDPA attention (FlashAttention kernels on Ampere and newer), gradient
checkpointing, Liger kernels via the Trainer (stages/common.py), and length-sorted batched inference.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig)

DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
GC_KWARGS = {"use_reentrant": False}


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def quant_for(method: str) -> str | None:
    """Weight precision a training method loads its base model in."""
    return "nf4" if method == "qlora" else None


def _nf4_config(scorer: bool) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE,
        llm_int8_skip_modules=["score"] if scorer else None,  # a classification head stays in full precision
    )


def attention_for(path: str) -> str:
    """SDPA everywhere, except Gemma-2, whose attention logit soft-capping needs the eager implementation."""
    return "eager" if AutoConfig.from_pretrained(path).model_type == "gemma2" else "sdpa"


def load_model(path: str, kind: str, quant: str | None = None, num_labels: int | None = None, pad_token_id: int | None = None):
    """kind: 'scorer' (sequence classification, multi-label heads) or 'causal' (text generation).
    quant: None (bf16) or "nf4" (QLoRA)."""
    kwargs = dict(dtype=DTYPE, attn_implementation=attention_for(path), device_map={"": 0})
    if quant == "nf4":  # only pass it when needed: an explicit None would override a checkpoint's own FP8/AWQ config
        kwargs["quantization_config"] = _nf4_config(kind == "scorer")
    if kind == "scorer":
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels, problem_type="multi_label_classification", **kwargs
        )
        model.config.pad_token_id = pad_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model


def add_lora(model, kind: str, method: str, lora_cfg: dict, gradient_checkpointing: bool):
    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                                gradient_checkpointing_kwargs=GC_KWARGS)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS if kind == "scorer" else TaskType.CAUSAL_LM,  # SEQ_CLS also trains `score`
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    ))
    for p in model.parameters():  # fp32 master copy for every trainable tensor
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def load_with_adapter(base_path: str, adapter_path: str, kind: str, quant: str | None = None, trainable: bool = False, **kw):
    model = load_model(base_path, kind, quant, **kw)
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)


def save_adapter(model, tokenizer, out_dir) -> None:
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def merge_adapter(base_path: str, adapter_path: str, out_dir, kind: str, tokenizer, **kw) -> None:
    """Fold a LoRA adapter (+ trained head) into bf16 base weights and save a standalone model."""
    model = load_with_adapter(base_path, adapter_path, kind, **kw).merge_and_unload()
    save_adapter(model, tokenizer, out_dir)
    del model
    free_gpu()


def free_gpu() -> None:
    """Release cached GPU memory. Callers must `del` their own model references first."""
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def predict_logits(model, tokenizer, prompts: list[str], max_length: int, batch_size: int, progress=None) -> np.ndarray:
    """Batched forward passes, longest prompts first to keep padding low. `progress(done, total)` is optional."""
    model.eval()
    order = np.argsort([-len(p) for p in prompts])
    out = np.zeros((len(prompts), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        enc = tokenizer([prompts[j] for j in idx], padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(model.device)
        with torch.autocast("cuda", dtype=DTYPE):
            out[idx] = model(**enc).logits.float().cpu().numpy()
        if progress:
            progress(min(i + batch_size, len(order)), len(order))
    return out



@torch.no_grad()
def generate(model, tokenizer, chat_prompts: list[str], batch_size: int, max_new_tokens: int, progress=None) -> list[str]:
    """Greedy batched generation, longest prompts first (less padding); results in input order."""
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    order = np.argsort([-len(p) for p in chat_prompts])
    outs = [""] * len(chat_prompts)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        enc = tokenizer([chat_prompts[j] for j in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for j, text in zip(idx, tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[j] = text
        if progress:
            progress(min(i + batch_size, len(order)), len(order))
    tokenizer.padding_side = "right"
    return outs
