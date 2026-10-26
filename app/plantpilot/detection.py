"""Lightweight failure-risk inference and statistical anomaly evidence.

Models train lazily from the bundled AI4I 2020 benchmark. This is a prototype
benchmark, not a substitute for validation on a plant's own operating data.
"""
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
FEATURE_NAMES = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
DATASET = Path(__file__).resolve().parents[2] / "ai4i2020.csv"


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric")
    if not __import__('math').isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def normalize_reading(reading: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
    aliases = {
        "air_temperature_k": ("air_temperature_k", "Air temperature [K]"),
        "process_temperature_k": ("process_temperature_k", "Process temperature [K]"),
        "rotational_speed_rpm": ("rotational_speed_rpm", "Rotational speed [rpm]"),
        "torque_nm": ("torque_nm", "Torque [Nm]"),
        "tool_wear_min": ("tool_wear_min", "Tool wear [min]"),
    }
    values = {}
    for key, choices in aliases.items():
        found = next((reading[k] for k in choices if k in reading), None)
        if found is None:
            raise ValueError(f"Missing required measurement: {key}")
        values[key] = _number(found, key)
    return [values[k] for k in FEATURE_NAMES], values


@lru_cache(maxsize=1)
def _models():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import f1_score
        from .eval.split import load_dataset, split_dataset
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for model inference; install project dependencies") from exc
    if not DATASET.exists():
        raise RuntimeError(f"Benchmark dataset not found at {DATASET}")
    train, dev, _ = split_dataset(load_dataset(DATASET))
    X_train = [[float(r[c]) for c in FEATURE_COLUMNS] for r in train]
    y_train = [int(r["Machine failure"]) for r in train]
    X_dev = [[float(r[c]) for c in FEATURE_COLUMNS] for r in dev]
    y_dev = [int(r["Machine failure"]) for r in dev]
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=-1)
    hgb.fit(X_train, y_train); rf.fit(X_train, y_train)
    def threshold(model):
        probs = model.predict_proba(X_dev)[:, 1]
        candidates = sorted(set([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] + [float(p) for p in probs]))
        return max(candidates, key=lambda t: f1_score(y_dev, [int(p >= t) for p in probs], zero_division=0))
    return hgb, rf, threshold(hgb), threshold(rf)


def predict(reading: dict[str, Any], include_secondary: bool = False) -> dict[str, Any]:
    from .detector import fit_references, detect
    vector, values = normalize_reading(reading)
    hgb, rf, hgb_threshold, rf_threshold = _models()
    hgb_risk = float(hgb.predict_proba([vector])[0][1])
    iqr_refs = _iqr_references()
    row = {"UDI": reading.get("UDI", 0), **dict(zip(FEATURE_COLUMNS, vector))}
    anomaly = detect(row, iqr_refs).to_dict()
    result = {
        "model": "HistGradientBoostingClassifier",
        "failure_probability": round(hgb_risk, 6),
        "threshold": round(hgb_threshold, 6),
        "failure_risk": "elevated" if hgb_risk >= hgb_threshold else "not_elevated",
        "predicted_failure": bool(hgb_risk >= hgb_threshold),
        "anomaly_evidence": {"flagged_parameters": anomaly["flagged_parameters"], "detection_methods": anomaly["detection_methods"], "explanation": anomaly["explanation"]},
        "secondary_model": None,
        "provenance": "Models trained at runtime on the bundled AI4I 2020 dataset; benchmark only, not plant-validated.",
    }
    if include_secondary:
        rf_risk = float(rf.predict_proba([vector])[0][1])
        result["secondary_model"] = {"model": "RandomForestClassifier", "failure_probability": round(rf_risk, 6), "threshold": round(rf_threshold, 6), "predicted_failure": bool(rf_risk >= rf_threshold)}
        result["model_agreement"] = bool(result["predicted_failure"] == result["secondary_model"]["predicted_failure"])
    return result


@lru_cache(maxsize=1)
def _iqr_references():
    from .detector import fit_references
    from .eval.split import load_dataset, split_dataset
    train, _, _ = split_dataset(load_dataset(DATASET))
    return fit_references(train)
