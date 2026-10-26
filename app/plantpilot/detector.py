"""Multi-signal anomaly detector for AI4I 2020 operational data.

Separate from agent/evaluation. Uses dataset-derived reference statistics (fit on train split only).
No ground-truth failure labels as inputs. No fabricated measurements.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import statistics

# Reference columns with units (agent-facing only — no failure labels)
MEASUREMENTS = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}

# Threshold derivation documented per parameter
THRESHOLDS = {
    # Experimental heuristic: high wear indicates degradation
    # Source: existing baseline rule (wear > 200); documented as heuristic only
    "tool_wear_min": {"rule": "wear > 200", "method": "experimental_heuristic", "note": "Not a validated industrial safety limit"},
}


class DetectorResult:
    def __init__(
        self,
        record_id: int,
        flagged: List[str],
        values: Dict[str, float],
        methods: Dict[str, str],
        references: Dict[str, Any],
        explanation: str,
        provenance: str,
    ):
        self.record_id = record_id
        self.flagged = flagged
        self.values = values
        self.methods = methods
        self.references = references
        self.explanation = explanation
        self.provenance = provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "flagged_parameters": self.flagged,
            "observed_values": self.values,
            "detection_methods": self.methods,
            "reference_statistics": self.references,
            "explanation": self.explanation,
            "provenance": self.provenance,
            "source": "AI4I 2020 historical dataset",
        }


def compute_iqr_bounds(values: List[float]) -> tuple:
    """Return (lower_bound, upper_bound) using IQR method."""
    if not values or len(values) < 2:
        return (float("-inf"), float("inf"))
    s = sorted(values)
    n = len(s)
    q1 = s[int(n * 0.25)] if n >= 4 else s[0]
    q3 = s[int(n * 0.75)] if n >= 4 else s[-1]
    iqr = q3 - q1
    if iqr <= 0:
        # Zero IQR: use mean ± 2*std as fallback, else full range
        mean = statistics.mean(s)
        try:
            std = statistics.stdev(s)
        except statistics.StatisticsError:
            std = 0.0
        if std > 0:
            return (mean - 2 * std, mean + 2 * std)
        return (s[0], s[-1])
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def detect(
    row: Dict[str, Any],
    references: Dict[str, Any],
) -> DetectorResult:
    """Evaluate one dataset record using reference statistics from training data."""
    record_id = int(row.get("UDI", 0)) if row.get("UDI") else 0

    # Only measurement fields — never failure labels
    measurements_dict = {
        "air_temperature_k": float(row.get("Air temperature [K]", 0)),
        "process_temperature_k": float(row.get("Process temperature [K]", 0)),
        "rotational_speed_rpm": float(row.get("Rotational speed [rpm]", 0)),
        "torque_nm": float(row.get("Torque [Nm]", 0)),
        "tool_wear_min": float(row.get("Tool wear [min]", 0)),
    }

    flagged: List[str] = []
    methods: Dict[str, str] = {}
    ref_stats: Dict[str, Any] = {}

    # IQR-based detection on each parameter (using pre-computed train references)
    for key, val in measurements_dict.items():
        ref = references.get(key)
        if not ref:
            continue
        lower, upper = ref.get("lower"), ref.get("upper")
        if lower is not None and upper is not None and val < lower:
            flagged.append(key)
            methods[key] = "iqr_lower"
            ref_stats[key] = {"lower": lower, "upper": upper, "value": val, "mean_ref": ref.get("mean")}
        elif lower is not None and upper is not None and val > upper:
            flagged.append(key)
            methods[key] = "iqr_upper"
            ref_stats[key] = {"lower": lower, "upper": upper, "value": val, "mean_ref": ref.get("mean")}

        # Explicit rule for wear only (documented heuristic, not validated limit)
        if key == "tool_wear_min" and val > 200:
            if key not in flagged:
                flagged.append(key)
            methods[key] = methods.get(key, "") + "+rule_wear_gt_200"
            ref_stats[key] = {"threshold": 200, "value": val, "method": "experimental_heuristic", "note": "Not a validated industrial safety limit"}

    # Multi-signal: flag if >= 2 signals unusual
    multi = len(flagged) >= 2

    explanation = (
        f"Record UDI {record_id}: flagged parameters {flagged if flagged else 'none'}. "
        f"Method: {'multi-signal' if multi else ('single-signal' if flagged else 'none')}. "
        f"Reference: dataset-derived IQR (training split, seed=42), wear heuristic documented. "
        f"No ground-truth failure label used."
    )

    provenance = "AI4I 2020 historical dataset — statistical anomaly detection, not real-time sensor monitoring"

    return DetectorResult(
        record_id=record_id,
        flagged=flagged,
        values=measurements_dict,
        methods=methods,
        references=ref_stats,
        explanation=explanation,
        provenance=provenance,
    )


def fit_references(train_rows: list) -> Dict[str, Any]:
    """Compute IQR references from training split only."""
    refs = {}
    for key, col in [
        ("air_temperature_k", "Air temperature [K]"),
        ("process_temperature_k", "Process temperature [K]"),
        ("rotational_speed_rpm", "Rotational speed [rpm]"),
        ("torque_nm", "Torque [Nm]"),
        ("tool_wear_min", "Tool wear [min]"),
    ]:
        vals = [float(r[col]) for r in train_rows if r.get(col) is not None and str(r.get(col)).strip() != ""]
        if not vals:
            refs[key] = {"mean": None, "lower": None, "upper": None, "n": 0}
            continue
        lower, upper = compute_iqr_bounds(vals)
        refs[key] = {
            "mean": statistics.mean(vals),
            "lower": lower,
            "upper": upper,
            "n": len(vals),
        }
    return refs
