from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return model.predict(X)


def evaluate_classifier(model, X, y) -> dict[str, float]:
    scores = predict_scores(model, X)
    preds = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "pr_auc": average_precision_score(y, scores),
    }
    if len(set(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, scores)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def write_metrics_table(metrics: dict[str, dict[str, float]], path: str | Path) -> pd.DataFrame:
    out = pd.DataFrame(metrics).T.sort_values("pr_auc", ascending=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out


def write_classification_report(model, X, y, path: str | Path) -> None:
    preds = (predict_scores(model, X) >= 0.5).astype(int)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(classification_report(y, preds, zero_division=0), encoding="utf-8")
