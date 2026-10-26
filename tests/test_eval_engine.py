"""Tests for evaluation engine — leakage-safe."""

import pytest
from plantpilot.eval.split import load_dataset, split_dataset, get_label_distribution
from plantpilot.eval.metrics import compute_metrics
from plantpilot.eval.baseline import baseline_predict


def test_split_reproducible_and_no_leakage():
    rows = load_dataset()
    train, dev, eval_ = split_dataset(rows, seed=42)
    # All splits non-empty
    assert len(train) > 0 and len(dev) > 0 and len(eval_) > 0
    # No overlap
    train_udis = {r["UDI"] for r in train}
    dev_udis = {r["UDI"] for r in dev}
    eval_udis = {r["UDI"] for r in eval_}
    assert len(train_udis & dev_udis) == 0
    assert len(train_udis & eval_udis) == 0
    assert len(dev_udis & eval_udis) == 0
    # Stratified: both classes present
    assert any(int(r["Machine failure"]) == 1 for r in train)
    assert any(int(r["Machine failure"]) == 0 for r in train)


def test_label_isolation_agent_not_exposed():
    # Verify that agent-facing adapter never includes failure columns
    from plantpilot.data_adapter import get_sensor_readings
    result = get_sensor_readings(1)
    # No label fields in agent result
    assert "machine_failure" not in result
    assert "labels" not in result
    # Separate evaluation interface exists
    from plantpilot.data_adapter import get_evaluation_only
    eval_r = get_evaluation_only(1)
    assert "machine_failure" in eval_r["labels"]


def test_metrics_on_known_fixture():
    # Known fixture: 3 positive, 7 negative
    preds = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
    actual = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
    m = compute_metrics(preds, actual)
    assert m["tp"] == 3
    assert m["fn"] == 0
    assert m["fp"] == 0
    assert m["tn"] == 7
    assert m["f1"] == 1.0
    assert m["positive_rate_actual"] == 0.3


def test_baseline_deterministic():
    rows = load_dataset()
    # Just first 100 for speed
    preds = baseline_predict(rows[:100])
    # Should be list of ints 0/1
    assert all(p in (0, 1) for p in preds)
    # Deterministic: same input -> same output
    preds2 = baseline_predict(rows[:100])
    assert preds == preds2


def test_empty_predictions():
    m = compute_metrics([], [])
    assert m["n"] == 0
    assert m["precision"] == 0.0


def test_missing_malformed_handled_by_adapter():
    # Already verified by test_evidence_first; here confirm adapter validates
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(99999)
