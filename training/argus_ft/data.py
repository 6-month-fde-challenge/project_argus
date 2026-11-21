"""Load the pre-serialised ARGUS files and turn them into tokenised datasets.

  scorer    training/sft/{train,validation} -> fraud_label + 8 reason codes (classification heads)
  explainer training/gold/train -> JSON {reason_codes, explanation}
  dpo       training/dpo/train -> chosen / rejected JSON answers
  cpt       training/cpt/train -> raw domain text, packed into fixed blocks

Prompts arrive in fmt-v1 (CRD 8.2). fmt-v2 is derived from it exactly (verified against all 1,030 rows of
training/format_comparison), so the pilot can compare both formats on the same rows (FR-FE-07).
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

from datasets import Dataset
from jinja2 import TemplateError

REASON_CODES = [
    "R01_NEW_DEVICE", "R02_GEO_MISMATCH", "R03_ODD_HOUR", "R04_FAILED_ATTEMPTS",
    "R05_AMOUNT_SPIKE", "R06_NEW_ACCOUNT", "R07_VELOCITY", "R08_HIGH_RISK_MCC",
]
NUM_HEADS = 1 + len(REASON_CODES)  # head 0 = fraud, heads 1..8 = reason codes
SYSTEM_PROMPT = (
    "You are ARGUS-LM, VKPayments' fraud analyst model. Read the transaction context and reply with JSON only: "
    '{"reason_codes": [...], "explanation": "..."}. Cite only facts present in the input.'
)
PROMPT_FORMATS = ("fmt-v1", "fmt-v2")
NO_HISTORY = "none observed"  # the dataset's marker for an empty [HIST] block


def read_jsonl_gz(folder: Path) -> list[dict]:
    files = sorted(Path(folder).glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"No .jsonl.gz files in {folder} - run `python -m argus_ft.supabase_data` first")
    rows = []
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


# ---------------------------------------------------------------- prompt formats and length

def _history(line: str) -> list[str]:
    body = line[len("[HIST] "):]
    return [] if body == NO_HISTORY else body.split(" | ")


def to_format(prompt_v1: str, fmt: str) -> str:
    """fmt-v2 = fmt-v1 + a [SUMMARY] line (history count, distinct known cities) and history entries
    without their city."""
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r}; expected one of {PROMPT_FORMATS}")
    if fmt == "fmt-v1":
        return prompt_v1
    out = []
    for line in prompt_v1.split("\n"):
        if not line.startswith("[HIST]"):
            out.append(line)
            continue
        entries = _history(line)
        cities = {e.rsplit(" ", 1)[1] for e in entries} - {"unk"}
        out.append(f"[SUMMARY] hist_n={len(entries)} hist_cities={len(cities)}")
        out.append("[HIST] " + (" | ".join(e.rsplit(" ", 1)[0] for e in entries) if entries else NO_HISTORY))
    return "\n".join(out)


def fit_prompts(tokenizer, prompts: list[str], max_tokens: int) -> list[str]:
    """Drop the oldest [HIST] entries until each prompt fits (FR-FE-05). Entries are newest-first."""
    lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]
    return [p if n <= max_tokens else _trim_history(tokenizer, p, max_tokens) for p, n in zip(prompts, lengths)]


def _trim_history(tokenizer, prompt: str, max_tokens: int) -> str:
    lines = prompt.split("\n")
    idx = next((i for i, line in enumerate(lines) if line.startswith("[HIST]")), None)
    if idx is None:
        return prompt
    entries = _history(lines[idx])
    while entries:
        entries.pop()
        lines[idx] = "[HIST] " + (" | ".join(entries) if entries else NO_HISTORY)
        if len(tokenizer("\n".join(lines), add_special_tokens=False)["input_ids"]) <= max_tokens:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------- row selection

def downsample_negatives(rows: list[dict], neg_per_pos: int, seed: int) -> list[dict]:
    """Keep every fraud row and neg_per_pos genuine rows per fraud (FR-ML-04); 0 keeps everything."""
    if neg_per_pos <= 0:
        return rows
    pos = [r for r in rows if r["fraud_label"] == 1]
    neg = [r for r in rows if r["fraud_label"] == 0]
    out = pos + random.Random(seed).sample(neg, min(len(neg), neg_per_pos * len(pos)))
    random.Random(seed).shuffle(out)
    return out


# ---------------------------------------------------------------- scorer (classification heads)

def load_sft(data_dir: str, split: str) -> list[dict]:
    return read_jsonl_gz(Path(data_dir) / "training" / "sft" / split)


def head_targets(row: dict) -> list[float]:
    codes = set(row["reason_targets"] or [])
    return [float(row["fraud_label"])] + [float(c in codes) for c in REASON_CODES]


def build_scorer_dataset(rows: list[dict], tokenizer, max_length: int, extra: dict[str, list] | None = None) -> Dataset:
    """rows must carry a pre-fitted `prompt`; `extra` adds columns such as teacher_logits for distillation."""
    enc = tokenizer([r["prompt"] for r in rows], truncation=True, max_length=max_length)
    cols = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": [head_targets(r) for r in rows]}
    return Dataset.from_dict({**cols, **(extra or {})})


# ---------------------------------------------------------------- explainer / DPO (chat format)

def _messages(prompt: str, system: bool = True) -> list[dict]:
    if system:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}]  # template without a system role


def _render(tokenizer, messages_for, **kw) -> str:
    try:
        return tokenizer.apply_chat_template(messages_for(True), tokenize=False, **kw)
    except TemplateError:
        return tokenizer.apply_chat_template(messages_for(False), tokenize=False, **kw)


def chat_prefix(tokenizer, prompt: str) -> str:
    """Everything up to where the assistant's answer starts."""
    return _render(tokenizer, lambda s: _messages(prompt, s), add_generation_prompt=True)


def answer_suffix(tokenizer, prompt: str, answer: str) -> str:
    """The answer exactly as the chat template ends an assistant turn (end-of-turn token included)."""
    prefix = chat_prefix(tokenizer, prompt)
    full = _render(tokenizer, lambda s: _messages(prompt, s) + [{"role": "assistant", "content": answer}])
    return full[len(prefix):] if full.startswith(prefix) else answer + tokenizer.eos_token


def answer_codes(obj) -> list[str] | None:
    """`reason_codes` of a parsed answer if it is a list of strings, else None (models can emit anything)."""
    codes = obj.get("reason_codes") if isinstance(obj, dict) else None
    return codes if isinstance(codes, list) and all(isinstance(c, str) for c in codes) else None


def target_to_json(target: dict) -> str:
    return json.dumps({"reason_codes": target["reason_codes"], "explanation": target["explanation"]}, ensure_ascii=False)


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_length: int) -> dict:
    """Prompt tokens are masked with -100 so loss / log-probs cover only the answer."""
    prefix_ids = tokenizer(chat_prefix(tokenizer, prompt), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_suffix(tokenizer, prompt, answer), add_special_tokens=False)["input_ids"]
    input_ids = (prefix_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prefix_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_gold(data_dir: str, val_rows: int) -> tuple[list[dict], list[dict]]:
    """Time-ordered: the latest `val_rows` gold transactions are held out for validation."""
    rows = sorted(read_jsonl_gz(Path(data_dir) / "training" / "gold" / "train"), key=lambda r: r["transaction_id"])
    return rows[:-val_rows], rows[-val_rows:]


def build_explainer_dataset(rows: list[dict], tokenizer, max_length: int, answer_key: str = "target") -> Dataset:
    """`answer_key` = "target" (gold answers) or a key holding teacher answers (explainer distillation)."""
    def answer(r):
        a = r[answer_key]
        return a if isinstance(a, str) else target_to_json(a)
    return Dataset.from_list([encode_prompt_answer(tokenizer, r["prompt"], answer(r), max_length) for r in rows])


def load_dpo(data_dir: str, eval_pairs: int) -> tuple[list[dict], list[dict]]:
    """Time-ordered: the latest `eval_pairs` preference pairs are held out."""
    rows = sorted(read_jsonl_gz(Path(data_dir) / "training" / "dpo" / "train"), key=lambda r: r["transaction_id"])
    return rows[:-eval_pairs], rows[-eval_pairs:]


# ---------------------------------------------------------------- CPT

def build_cpt_dataset(data_dir: str, tokenizer, max_docs: int, block_size: int, seed: int) -> tuple[Dataset, Dataset]:
    """Concatenate documents (eos-separated) and cut into fixed blocks; last 5% of blocks for perplexity eval."""
    docs = read_jsonl_gz(Path(data_dir) / "training" / "cpt" / "train")
    docs = random.Random(seed).sample(docs, min(max_docs, len(docs)))
    ids = []
    for ids_doc in tokenizer([d["text"] for d in docs], add_special_tokens=False)["input_ids"]:
        ids += ids_doc + [tokenizer.eos_token_id]
    blocks = [ids[i : i + block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
    n_eval = max(1, len(blocks) // 20)
    to_ds = lambda bs: Dataset.from_dict({"input_ids": bs, "attention_mask": [[1] * block_size for _ in bs]})
    return to_ds(blocks[:-n_eval]), to_ds(blocks[-n_eval:])
