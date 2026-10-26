"""CSV-backed sensor-data adapter for PlantPilot.

Independent of the AI agent. Reads from ai4i2020.csv dataset.
Returns historical observations with provenance, units, and metadata.
Ground-truth failure labels are kept out of the agent interface.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Dataset path relative to project root
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "ai4i2020.csv"

# Agent-facing sensor parameters (measurement columns only)
AGENT_PARAMS = {
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

# Ground-truth evaluation-only labels (NOT exposed to agent)
EVAL_LABELS = {
    "machine_failure": "Machine failure",
    "twf": "TWF",
    "hdf": "HDF",
    "pwf": "PWF",
    "osf": "OSF",
    "rnf": "RNF",
}

# All measurement columns with units
COLUMN_MAP = {
    "udi": ("UDI", None),
    "product_id": ("Product ID", None),
    "type": ("Type", None),
    "air_temperature": ("Air temperature [K]", "K"),
    "process_temperature": ("Process temperature [K]", "K"),
    "rotational_speed": ("Rotational speed [rpm]", "rpm"),
    "torque": ("Torque [Nm]", "Nm"),
    "tool_wear": ("Tool wear [min]", "min"),
}


class DataAdapterError(ValueError):
    """Raised when data adapter encounters invalid input or missing data."""


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load CSV into list of dicts, validating header."""
    if not path.exists():
        raise DataAdapterError(f"Dataset not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise DataAdapterError("CSV has no header row")
        expected_headers = {
            "UDI", "Product ID", "Type",
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
        }
        actual = {h.replace("﻿", "").strip() for h in reader.fieldnames}
        if not expected_headers.issubset(actual):
            missing = expected_headers - actual
            raise DataAdapterError(f"CSV missing required columns: {missing}")
        rows = []
        for i, row in enumerate(reader, start=2):  # line number (1-based, header=1)
            # Validate numeric measurement columns
            for key, (col_name, _) in COLUMN_MAP.items():
                if key in ("udi", "product_id", "type"):
                    continue
                val_str = row.get(col_name, "").strip()
                if val_str == "":
                    raise DataAdapterError(
                        f"Row {i}: missing value for {col_name} "
                        f"(record UDI={row.get('UDI')})"
                    )
                try:
                    float(val_str)
                except ValueError:
                    raise DataAdapterError(
                        f"Row {i}: non-numeric value '{val_str}' for {col_name}"
                    )
            rows.append({k: v for k, v in row.items()})
    return rows


def get_sensor_readings(
    station_id: Union[int, str],
    parameter: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieve historical observations by record identifier.

    Args:
        station_id: Record UDI (int/str) or Product ID.
        parameter: Agent-facing parameter key (e.g., 'air_temperature').
                   If None, return all agent-facing measurements.
        csv_path: Path to CSV (defaults to project ai4i2020.csv).

    Returns:
        Dict with record info, measurements, provenance metadata.
        For unknown identifiers: raises DataAdapterError.
        For unsupported parameters: raises DataAdapterError.
    """
    path = csv_path or DEFAULT_CSV
    # Load once and cache? For simplicity reload per call (fast enough).
    # In production: cache loaded dataset in module or DB.
    try:
        all_rows = _load_csv(path)
    except DataAdapterError:
        raise

    # Find by UDI (primary) or Product ID
    record = None
    if isinstance(station_id, int) or (isinstance(station_id, str) and station_id.isdigit()):
        target_udi = str(int(station_id))
        for row in all_rows:
            if row.get("UDI", "").strip() == target_udi:
                record = row
                break
    else:
        target_pid = station_id.strip()
        for row in all_rows:
            if row.get("Product ID", "").strip() == target_pid:
                record = row
                break

    if record is None:
        raise DataAdapterError(f"Unknown record identifier: {station_id!r}")

    # Build measurements (agent-facing only)
    measurements: Dict[str, Any] = {}
    if parameter is None:
        for key, (col_name, unit) in COLUMN_MAP.items():
            if key in ("udi", "product_id", "type"):
                continue
            val_str = record.get(col_name, "").strip()
            if val_str:
                measurements[key] = {
                    "value": float(val_str),
                    "unit": unit,
                }
    else:
        # Validate parameter
        param_key = parameter.lower().replace(" ", "_").replace("[", "").replace("]", "")
        # Map common aliases
        param_map = {
            "air_temperature": "air_temperature",
            "air_temp": "air_temperature",
            "temperature": "air_temperature",
            "process_temperature": "process_temperature",
            "process_temp": "process_temperature",
            "rotational_speed": "rotational_speed",
            "speed": "rotational_speed",
            "rpm": "rotational_speed",
            "torque": "torque",
            "tool_wear": "tool_wear",
            "wear": "tool_wear",
        }
        mapped = param_map.get(param_key, param_key)
        if mapped not in AGENT_PARAMS:
            supported = list(AGENT_PARAMS.keys())
            raise DataAdapterError(
                f"Unsupported sensor parameter: {parameter!r}. "
                f"Supported: {supported}"
            )
        col_name = AGENT_PARAMS[mapped]
        val_str = record.get(col_name, "").strip()
        if val_str == "":
            raise DataAdapterError(
                f"Record {station_id}: no value for parameter {parameter!r}"
            )
        measurements[mapped] = {
            "value": float(val_str),
            "unit": COLUMN_MAP[mapped][1],
        }

    # Build result
    result = {
        "status": "retrieved",
        "record_identifier": {
            "udi": int(record.get("UDI", 0)),
            "product_id": record.get("Product ID", "").strip(),
            "type": record.get("Type", "").strip(),
        },
        "measurements": measurements,
        "source": "AI4I 2020 Predictive Maintenance Dataset (csv)",
        "provenance": "Historical dataset observation — not real-time sensor",
        "dataset_path": str(path),
        "dataset_rows": len(all_rows),
        "record_index": int(record.get("UDI", 0)),
    }

    # Add evaluation-only separation note
    result["evaluation_notes"] = (
        "Ground-truth labels (Machine failure, TWF, HDF, PWF, OSF, RNF) "
        "are kept separate and not included in agent-facing measurements."
    )
    return result


def get_evaluation_only(
    station_id: Union[int, str],
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Separate evaluation interface for ground-truth labels.

    Does NOT expose labels to the agent-facing tool.
    For evaluation engine use only.
    """
    path = csv_path or DEFAULT_CSV
    try:
        all_rows = _load_csv(path)
    except DataAdapterError:
        raise

    record = None
    if isinstance(station_id, int) or (isinstance(station_id, str) and station_id.isdigit()):
        target_udi = str(int(station_id))
        for row in all_rows:
            if row.get("UDI", "").strip() == target_udi:
                record = row
                break
    else:
        target_pid = station_id.strip()
        for row in all_rows:
            if row.get("Product ID", "").strip() == target_pid:
                record = row
                break

    if record is None:
        raise DataAdapterError(f"Unknown record identifier: {station_id!r}")

    return {
        "record_identifier": {
            "udi": int(record.get("UDI", 0)),
            "product_id": record.get("Product ID", "").strip(),
        },
        "labels": {
            "machine_failure": int(record.get("Machine failure", 0)),
            "twf": int(record.get("TWF", 0)),
            "hdf": int(record.get("HDF", 0)),
            "pwf": int(record.get("PWF", 0)),
            "osf": int(record.get("OSF", 0)),
            "rnf": int(record.get("RNF", 0)),
        },
        "source": "AI4I 2020 dataset — evaluation only",
    }
