import os
import pytest

os.environ.setdefault("NVIDIA_API_KEY", "test-key-not-real")

from plantpilot.detection import normalize_reading, predict
from plantpilot.digital_twin import get_twin_store, reset_twin_store
from plantpilot.config import reset_config

READING = {
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1550,
    "torque_nm": 40.2,
    "tool_wear_min": 120,
}

def test_normalize_reading_rejects_missing_measurement():
    with pytest.raises(ValueError, match="Missing required"):
        normalize_reading({"air_temperature_k": 298})

def test_normalize_reading_rejects_non_finite():
    bad = dict(READING, torque_nm=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        normalize_reading(bad)

def test_predict_primary_and_optional_secondary():
    result = predict(READING, include_secondary=False)
    assert 0 <= result["failure_probability"] <= 1
    assert result["model"] == "HistGradientBoostingClassifier"
    assert result["secondary_model"] is None
    assert "flagged_parameters" in result["anomaly_evidence"]
    result2 = predict(READING, include_secondary=True)
    assert result2["secondary_model"]["model"] == "RandomForestClassifier"
    assert isinstance(result2["model_agreement"], bool)

def test_twin_simulation_does_not_change_latest_observed(tmp_path, monkeypatch):
    from plantpilot import config
    # Use an isolated SQLite file for this test.
    monkeypatch.setenv("PLANTPILOT_DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    reset_config(); reset_twin_store()
    store = get_twin_store()
    twin = store.create("CNC-042", "CNC", {})
    observed = store.add_reading(twin["id"], READING, {"predicted_failure": False}, "observed")
    simulated = store.add_reading(twin["id"], dict(READING, torque_nm=70), {"predicted_failure": True}, "simulated")
    current = store.get(twin["id"])
    history = store.history(twin["id"])
    assert current["latest_observed_reading"]["id"] == observed["id"]
    assert simulated["evidence_type"] == "simulated"
    assert len(history) == 2
    reset_twin_store(); reset_config()
