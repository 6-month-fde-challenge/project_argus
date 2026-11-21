"""Stage 0 - Data checks before any training (TR-DATA-02, TR-DATA-04, DR-02, FR-FE-05).

Leakage checks fail the run: the SFT splits must be disjoint and in time order, and the gold, DPO and
CPT training files may contain only training-period transactions. Data-quality numbers are reported only.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..db import transaction_facts
from ..supabase_data import fetch_tables

SPLITS = ("train", "validation", "holdout")


def _ids(rows: list[dict]) -> set[str]:
    return {r["transaction_id"] for r in rows if r.get("transaction_id")}


def run(ctx) -> dict:
    cfg = ctx.cfg
    sft = {s: D.load_sft(cfg.data_dir, s) for s in SPLITS}
    ids = {s: _ids(rows) for s, rows in sft.items()}
    facts = transaction_facts(cfg.db_dir)

    # ---- leakage (must all pass)
    span = {s: (facts.loc[list(ids[s]), "transaction_time"].min(), facts.loc[list(ids[s]), "transaction_time"].max())
            for s in SPLITS}
    training_files = {
        "gold": D.read_jsonl_gz(Path(cfg.data_dir) / "training" / "gold" / "train"),
        "dpo": D.read_jsonl_gz(Path(cfg.data_dir) / "training" / "dpo" / "train"),
        "cpt": D.read_jsonl_gz(Path(cfg.data_dir) / "training" / "cpt" / "train"),
    }
    leakage = {
        "sft_splits_disjoint": all(ids[a].isdisjoint(ids[b]) for a, b in combinations(SPLITS, 2)),
        "train_before_validation": span["train"][1] < span["validation"][0],
        "validation_before_holdout": span["validation"][1] < span["holdout"][0],
        **{f"{name}_only_train_split": all(r["split"] == "train" for r in rows) for name, rows in training_files.items()},
        **{f"{name}_has_no_eval_transactions": _ids(rows).isdisjoint(ids["validation"] | ids["holdout"])
           for name, rows in training_files.items()},
    }
    failed = [name for name, ok in leakage.items() if not ok]
    if failed:
        raise RuntimeError(f"leakage checks failed: {failed}")

    # ---- data quality and prompt size (reported)
    features = pd.read_parquet(fetch_tables(cfg.db_dir)["features"])
    sample = [r["prompt"] for r in sft["validation"][:2000]]
    lengths = np.array([len(x) for x in ctx.tokenizer(sample, add_special_tokens=False)["input_ids"]])
    return {
        "leakage": leakage,
        "time_spans": {s: [str(a), str(b)] for s, (a, b) in span.items()},
        "rows_and_fraud": {s: [len(rows), sum(r["fraud_label"] for r in rows)] for s, rows in sft.items()},
        "missing_rates": {c: round(float(features[c].mean()), 4) for c in ("missing_customer_location", "missing_device")},
        "prompt_tokens": {
            "tokenizer": ctx.teacher_model, "p50": int(np.percentile(lengths, 50)), "p99": int(np.percentile(lengths, 99)),
            "max": int(lengths.max()), "cap": cfg.max_length, "share_trimmed": round(float((lengths > cfg.max_length).mean()), 4),
        },
    }
