"""Evaluation metrics — isolated from agent; uses only predictions + ground-truth after split."""

from __future__ import annotations

from typing import List, Dict, Any
from collections import Counter


def compute_metrics(
    predictions: List[int],
    actual: List[int],
) -> Dict[str, Any]:
    """Compute classification metrics for binary failure prediction.

    Predictions: list of 0/1 (agent outputs or baseline).
    Actual: ground-truth labels.

    Reports precision, recall, F1, confusion matrix, class distribution.
    """
    n = len(predictions)
    assert len(actual) == n, "Prediction/actual length mismatch"

    tp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 0)
    tn = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 0)
    fn = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    actual_dist = Counter(actual)
    pred_dist = Counter(predictions)

    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "actual_distribution": dict(actual_dist),
        "predicted_distribution": dict(pred_dist),
        "positive_rate_actual": round(actual_dist.get(1, 0) / n, 4) if n > 0 else 0.0,
        "positive_rate_predicted": round(pred_dist.get(1, 0) / n, 4) if n > 0 else 0.0,
    }
