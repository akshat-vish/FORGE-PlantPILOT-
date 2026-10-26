"""Deterministic baseline for failure classification.

Rule: if tool wear > 200 minutes, predict failure (1), else 0.
This is a deterministic rule — clearly labeled as baseline, not LLM.
Used only on evaluation split after predictions recorded; no agent access.
"""

from __future__ import annotations

from typing import List, Dict, Any


def baseline_predict(rows: List[Dict[str, Any]]) -> List[int]:
    """Predict 1 if Tool wear [min] > 200, else 0."""
    predictions = []
    for row in rows:
        try:
            wear = float(row.get("Tool wear [min]", 0))
        except (ValueError, TypeError):
            wear = 0.0
        predictions.append(1 if wear > 200 else 0)
    return predictions
