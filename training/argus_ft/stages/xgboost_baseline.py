"""Stage 1 - XGBoost baseline on the engineered features (FR-ML-01): the bar the LLM must beat (OBJ-04).

Reads the `features` and `labels` tables from Supabase Postgres (snapshotted to data/db/*.parquet), trains on
the train split and scores the validation split. In production it also stays warm as the fallback scorer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .. import data as D
from ..metrics import ranking
from ..supabase_data import fetch_tables

# FE-01 .. FE-10 plus data-quality flags; all computed point-in-time by the feature pipeline
FEATURES = [
    "amount_to_avg_ratio",              # FE-01 amount vs usual spend
    "is_new_device",                    # FE-02
    "customer_merchant_distance_km",    # FE-03
    "is_new_city",                      # FE-03
    "hour_of_day", "is_night",          # FE-04
    "has_recent_failures",              # FE-05
    "is_new_account",                   # FE-06
    "txn_count_last_1h", "txn_amount_last_24h_paise",  # FE-07 velocity
    "mcc_target_encoded_rate",          # FE-08
    "distinct_customers_per_device_7d", # FE-09 device rings
    "combined_risk_flags",              # FE-10
    "missing_customer_location", "missing_device", "cold_start",
]


def load_feature_table(ctx) -> pd.DataFrame:
    paths = fetch_tables(ctx.cfg.db_dir)
    labels = pd.read_parquet(paths["labels"])[["transaction_id", "fraud_label", "split"]]
    return pd.read_parquet(paths["features"]).merge(labels, on="transaction_id").sort_values("transaction_id")


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Booleans -> 0/1, missing values -> NaN (XGBoost learns a default branch for them)."""
    return np.column_stack([df[c].map(lambda v: np.nan if v is None else float(v)).to_numpy() for c in FEATURES])


def class_weight(n_rows: int, n_fraud: int, mode: str) -> float:
    """scale_pos_weight (FR-ML-04): 'ratio' = genuine / fraud; 'sqrt' = its square root, which ranks as well and
    keeps the raw scores less extreme (the stage-8 calibrator maps them to probabilities either way)."""
    ratio = (n_rows - n_fraud) / max(n_fraud, 1)
    return float(np.sqrt(ratio)) if mode == "sqrt" else ratio


def run(ctx) -> dict:
    c = ctx.cfg.xgboost
    df = load_feature_table(ctx)
    train, val = df[df.split == "train"], df[df.split == "validation"]
    immature = 0
    if (ctx.cfg.get("label_maturity") or {}).get("eligible_only"):  # the same rows the LLM scorers leave out
        not_mature = {r["transaction_id"] for r in ctx.sft("train") if not D.label_mature(r)}
        keep = ~train.transaction_id.isin(not_mature)
        immature, train = int((~keep).sum()), train[keep]
    n_fraud = int(train.fraud_label.sum())
    # Early stopping looks only at the EARLIER half of validation - the rows before the same boundary stage 8 uses
    # (the median id of the validation sample it calibrates on). Every row of the later half, where scorers are
    # judged and thresholds recommended, stays out-of-sample.
    sample_ids = sorted(r["transaction_id"] for r in ctx.sft("validation"))
    stop_on = val[val.transaction_id < sample_ids[len(sample_ids) // 2]]

    model = XGBClassifier(
        n_estimators=c.n_estimators, max_depth=c.max_depth, learning_rate=c.learning_rate,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=c.get("min_child_weight", 1),
        scale_pos_weight=class_weight(len(train), n_fraud, c.get("scale_pos_weight", "ratio")),
        early_stopping_rounds=c.get("early_stopping_rounds") or None,
        tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=ctx.cfg.seed,
    )
    fit_kw = {"eval_set": [(to_matrix(stop_on), stop_on.fraud_label.to_numpy())], "verbose": False} \
        if c.get("early_stopping_rounds") else {}
    model.fit(to_matrix(train), train.fraud_label.to_numpy(), **fit_kw)
    model.save_model(ctx.path("xgboost", "model.json"))  # keeps best_iteration: serving predicts with the same trees

    proba = np.clip(model.predict_proba(to_matrix(val))[:, 1], 1e-7, 1 - 1e-7)
    logit = np.log(proba / (1 - proba))  # same log-odds scale as the LLM scorers, for calibration
    ctx.save_preds("xgboost_val", val.transaction_id.tolist(), logit[:, None])

    top = sorted(zip(FEATURES, model.feature_importances_.tolist()), key=lambda kv: -kv[1])[:5]
    return {
        "source": "supabase tables: features + labels",
        "train_rows": len(train), "train_fraud": n_fraud, "dropped_not_mature": immature,
        "trees": model.best_iteration + 1 if fit_kw else c.n_estimators,
        "early_stopping": f"earlier half of validation, patience {c.early_stopping_rounds}" if fit_kw else "off",
        "validation": ranking(logit, val.fraud_label.to_numpy()),
        "top_features": {k: round(v, 3) for k, v in top},
    }
