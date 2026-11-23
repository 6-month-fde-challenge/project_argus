"""Stage 2 - Pilot: candidate models x prompt formats (Part 3 Figure 4: "3 models x 2 formats").

Each (model, format) pair gets the same short LoRA fine-tune with fraud + reason heads and is ranked on the PR-AUC
of the EARLIER half (by time) of its validation rows; the gate then judges the winner on the LATER half, so the pair
is never selected and judged on the same rows (picking the best of six on one slice flatters it). The best pair is always selected and written to pilot/selected.json: its model becomes the
teacher and its format the serialisation for every later stage (TR-LLM-01, FR-FE-07). Whether that winner is
good enough to continue on the LLM path is decided next by the week-8 gate.
"""
from __future__ import annotations

import random

import numpy as np

from .. import data as D
from ..metrics import ranking, scorer_metrics
from ..modeling import free_gpu, load_tokenizer, predict_logits
from .common import train_scorer


def _slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def pilot_one(ctx, model_name: str, fmt: str, train_rows: list[dict], val_rows: list[dict]) -> np.ndarray:
    """Short LoRA fine-tune of one (model, format) pair; returns its validation logits."""
    cfg, tok = ctx.cfg, load_tokenizer(model_name)
    stage_dir = f"pilot/{_slug(model_name)}_{fmt}"
    model = train_scorer(ctx, stage_dir, model_name, tok, ctx.fitted(train_rows, tok, fmt), cfg.pilot)
    val_rows = ctx.fitted(val_rows, tok, fmt)
    logits = predict_logits(model, tok, [r["prompt"] for r in val_rows], cfg.max_length, cfg.eval.batch_size,
                            progress=ctx.events.progress(f"scoring {_slug(model_name)} x {fmt}"))
    del model
    free_gpu()
    return logits


def run(ctx) -> dict:
    c = ctx.cfg.pilot
    train_rows = ctx.scorer_train_rows("pilot")
    val_rows = ctx.sft("validation")
    if c.val_rows and c.val_rows < len(val_rows):  # seeded sample across the whole validation period
        val_rows = sorted(random.Random(ctx.cfg.seed).sample(val_rows, c.val_rows), key=lambda r: r["transaction_id"])
    val_ids = [r["transaction_id"] for r in val_rows]
    val_targets = np.array([D.head_targets(r) for r in val_rows])

    half = len(val_rows) // 2  # rows are sorted by id = time: [:half] ranks the pairs, [half:] is the gate's
    early = val_targets[:half, 0]
    select_on = "earlier half" if half and early.min() != early.max() else "all rows (no fraud in the earlier half)"
    pick = slice(0, half) if select_on == "earlier half" else slice(None)

    grid = {}
    for model_name in c.candidates:
        for fmt in c.formats:
            print(f"--- pilot {model_name} x {fmt}")
            logits = pilot_one(ctx, model_name, fmt, train_rows, val_rows)
            preds = f"pilot_{_slug(model_name)}_{fmt}_val"
            ctx.save_preds(preds, val_ids, logits)
            grid[f"{model_name} | {fmt}"] = {"model": model_name, "format": fmt, "preds": preds,
                                             **scorer_metrics(logits, val_targets),
                                             "select_pr_auc": ranking(logits[pick, 0], val_targets[pick, 0])["pr_auc"]}

    # always select one pair; a missing PR-AUC (no fraud in the slice) ranks last
    best = max(grid.values(), key=lambda g: g["select_pr_auc"] if g["select_pr_auc"] is not None else -1.0)
    ctx.save_json("pilot/selected.json", {"teacher": best["model"], "format": best["format"], "preds": best["preds"],
                                          "selected_on": select_on, "boundary_id": val_ids[half] if half else None})
    return {
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "grid": {k: {m: v[m] for m in ("pr_auc", "select_pr_auc", "roc_auc", "reason_macro_f1")} for k, v in grid.items()},
        "selected_on": select_on,
        "selected": {"teacher": best["model"], "format": best["format"], "pr_auc": best["pr_auc"]},
    }
