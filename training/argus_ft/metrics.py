"""Evaluation shared by the XGBoost baseline and every LLM scorer (TR-ML-02), plus explanation checks."""
from __future__ import annotations

import json
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from .data import REASON_CODES, answer_codes

# rules_config.json: medium 0.30, review 0.60, block 0.85 (Part 3, Section 7)
POLICY_THRESHOLDS = (0.30, 0.60, 0.85)
SWEEP = tuple(round(t, 2) for t in np.arange(0.05, 0.951, 0.05))  # TR-ML-04: cut-offs 0.05 - 0.95
DAILY_TRANSACTIONS = 4_200_000  # CRD Section 2: average daily volume, to express flagged share as cases per day
REVIEW_CAPACITY_PER_DAY = 12_000  # OBJ-09: daily cases sent to REVIEW


def ranking(scores: np.ndarray, labels: np.ndarray) -> dict:
    if labels.min() == labels.max():  # a slice with only one class has no PR / ROC curve
        return {"pr_auc": None, "roc_auc": None, "n": int(len(labels)), "positives": int(labels.sum())}
    return {
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "n": int(len(labels)),
        "positives": int(labels.sum()),
    }


def scorer_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    """Head 0 = fraud (PR-AUC); heads 1..8 = reason codes (macro F1 at 0.5 vs weak feature-rule targets)."""
    reason_pred = (logits[:, 1:] > 0).astype(int)
    return {
        **ranking(logits[:, 0], targets[:, 0]),
        "reason_macro_f1": round(float(f1_score(targets[:, 1:].astype(int), reason_pred, average="macro", zero_division=0)), 4),
    }


class Calibrator:
    """Maps the raw fraud logit to a probability (FR-ML-06, TR-ML-03).

    platt    - temperature scaling with a bias term: p = sigmoid(a * logit + b). The bias also undoes the
               prior shift caused by down-sampling genuine rows in training.
    isotonic - monotone step function fitted to observed fraud rates.
    """

    def __init__(self, method: str):
        self.method = method

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "Calibrator":
        if self.method == "platt":
            self.model = LogisticRegression(C=1e6).fit(scores.reshape(-1, 1), labels)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, labels)
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return self.model.predict(scores)

    def to_dict(self) -> dict:
        if self.method == "platt":
            return {"method": "platt", "slope": float(self.model.coef_[0, 0]), "intercept": float(self.model.intercept_[0])}
        return {"method": "isotonic", "x": self.model.X_thresholds_.tolist(), "y": self.model.y_thresholds_.tolist()}


def apply_calibration(params: dict, scores: np.ndarray) -> np.ndarray:
    """A saved calibrator (`Calibrator.to_dict()`, i.e. calibration/<scorer>.json) applied to raw scores, exactly as the
    scoring API applies it at serving time."""
    scores = np.asarray(scores, dtype=np.float64)
    if params["method"] == "platt":
        return 1 / (1 + np.exp(-(params["slope"] * scores + params["intercept"])))
    return np.interp(scores, params["x"], params["y"])


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    return float(sum(abs(probs[idx == b].mean() - labels[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float, amounts: np.ndarray | None = None) -> dict:
    """Confusion-based metrics for 'flag when probability >= threshold'."""
    flagged, fraud = probs >= threshold, labels == 1
    tp, fp = int((flagged & fraud).sum()), int((flagged & ~fraud).sum())
    out = {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(int(fraud.sum()), 1), 4),
        "false_positive_rate": round(fp / max(int((~fraud).sum()), 1), 5),
        "flagged_per_day": int(flagged.mean() * DAILY_TRANSACTIONS),
    }
    if amounts is not None:  # share of fraud rupees caught rather than fraud count
        out["rupee_recall"] = round(float(amounts[flagged & fraud].sum() / max(amounts[fraud].sum(), 1e-9)), 4)
    return out


def at_review_capacity(scores: np.ndarray, labels: np.ndarray, amounts: np.ndarray) -> dict:
    """Send the riskiest 12,000 of 4.2M daily transactions (0.29%) to REVIEW: how much fraud is caught?
    Rank-based, so it compares scorers fairly whatever their calibrated probability range (OBJ-01, OBJ-09)."""
    k = max(1, round(len(scores) * REVIEW_CAPACITY_PER_DAY / DAILY_TRANSACTIONS))
    flagged = np.zeros(len(scores), bool)
    flagged[np.argsort(-scores, kind="stable")[:k]] = True
    fraud = labels == 1
    tp = int((flagged & fraud).sum())
    return {"reviewed": k, "share": round(k / len(scores), 5), "recall": round(tp / max(int(fraud.sum()), 1), 4),
            "precision": round(tp / k, 4),
            "rupee_recall": round(float(amounts[flagged & fraud].sum() / max(amounts[fraud].sum(), 1e-9)), 4)}


def threshold_report(probs: np.ndarray, labels: np.ndarray, thresholds=SWEEP) -> list[dict]:
    """Precision, recall and daily flagged volume per cut-off (FR-ML-12, TR-ML-04)."""
    return [at_threshold(probs, labels, t) for t in thresholds]


def threshold_for_recall(probs: np.ndarray, labels: np.ndarray, target: float) -> float | None:
    """The highest cut-off that still flags at least `target` of the fraud (OBJ-01 recall at REVIEW).
    None when the slice has no fraud to measure recall on."""
    fraud = np.sort(probs[labels == 1])[::-1]
    if not len(fraud):
        return None
    return float(fraud[max(1, math.ceil(target * len(fraud))) - 1])


def threshold_for_precision(probs: np.ndarray, labels: np.ndarray, target: float) -> float | None:
    """The lowest cut-off whose flagged set still has precision >= `target` (OBJ-02 precision at BLOCK), or the
    best-precision cut-off when the target is never reached. None when the slice has no fraud."""
    if not (labels == 1).any():
        return None
    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    precision = np.cumsum(labels[order]) / np.arange(1, len(order) + 1)
    # a cut-off flags every row with prob >= it, so only the last row of each run of tied scores is a real
    # operating point: judging a tie group part-way would claim a precision no threshold can deliver
    ends = np.flatnonzero(np.append(sorted_probs[1:] != sorted_probs[:-1], True))
    ok = ends[precision[ends] >= target]
    return float(sorted_probs[ok[-1] if len(ok) else ends[int(np.argmax(precision[ends]))]])


def curves(scores: np.ndarray, labels: np.ndarray, points: int = 100) -> dict:
    """PR and ROC curves thinned to at most `points` points each (small enough for report.json and the UI)."""
    if labels.min() == labels.max():
        return {}
    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    def thin(xs, ys):
        idx = np.unique(np.linspace(0, len(xs) - 1, min(points, len(xs))).astype(int))
        return [round(float(v), 5) for v in xs[idx]], [round(float(v), 5) for v in ys[idx]]

    r, p = thin(recall, precision)
    f, t = thin(fpr, tpr)
    return {"pr": {"recall": r, "precision": p}, "roc": {"fpr": f, "tpr": t}}


def per_code_f1(logits: np.ndarray, targets: np.ndarray) -> dict:
    """F1 of every reason-code head at 0.5 (columns 1..8 of the scorer output) against the weak feature-rule targets."""
    pred, gold = (logits[:, 1:] > 0).astype(int), targets[:, 1:].astype(int)
    return {code: round(float(f1_score(gold[:, i], pred[:, i], zero_division=0)), 4) for i, code in enumerate(REASON_CODES)}


def group_error_rates(probs: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float) -> dict:
    """Miss rate and false-positive rate per group, flagging gaps > 20% relative to overall (TR-ML-05)."""
    def rates(mask):
        fraud, flagged = labels[mask] == 1, probs[mask] >= threshold
        return {"n": int(mask.sum()), "fraud": int(fraud.sum()),
                "miss_rate": round(float((~flagged & fraud).sum() / max(fraud.sum(), 1)), 4),
                "false_positive_rate": round(float((flagged & ~fraud).sum() / max((~fraud).sum(), 1)), 5)}

    overall = rates(np.ones(len(labels), bool))
    report = {"overall": overall}
    for g in sorted(set(groups)):
        r = rates(groups == g)
        r["gap_over_20pct"] = any(
            overall[k] > 0 and abs(r[k] - overall[k]) / overall[k] > 0.2 for k in ("miss_rate", "false_positive_rate")
        ) if r["fraud"] else None  # too few fraud cases in the group to judge
        report[str(g)] = r
    return report


def explanation_metrics(generations: list[str], targets: list[dict]) -> dict:
    """OBJ-06 (valid JSON matching the schema) and reason-code agreement with the gold target."""
    valid_json = valid_schema = exact = tp = fp = fn = 0
    answers = set()  # distinct predicted code sets: 1 means the explainer ignores the input
    for text, target in zip(generations, targets):
        gold = set(target["reason_codes"])
        try:
            obj = json.loads(text.strip())
            valid_json += 1
        except json.JSONDecodeError:
            fn += len(gold)
            continue
        codes = answer_codes(obj)
        pred = set(codes or [])
        answers.add(frozenset(pred))
        if codes is not None and isinstance(obj.get("explanation"), str) and pred <= set(REASON_CODES):
            valid_schema += 1
        exact += pred == gold
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
    n = max(len(generations), 1)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "n": len(generations),
        "json_valid_rate": round(valid_json / n, 4),
        "schema_valid_rate": round(valid_schema / n, 4),
        "reason_exact_match": round(exact / n, 4),
        "reason_f1": round(2 * p * r / max(p + r, 1e-9), 4),
        "distinct_code_sets": len(answers),
        "gold_distinct_code_sets": len({frozenset(t["reason_codes"]) for t in targets}),
    }

