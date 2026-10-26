"""Tests for safety policy enforcement."""

import pytest

from plantpilot.safety import (
    SafetyPolicy,
    ToolRisk,
    RiskLevel,
    check_safety,
    default_safety_policy,
)
from plantpilot.schemas import Recommendation, ApprovalStatus


def test_safety_policy_register_tool_risk():
    """Test registering tool risk metadata."""
    policy = SafetyPolicy()
    risk = ToolRisk(
        name="test_tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        description="A test tool",
    )
    policy.register_tool_risk(risk)

    retrieved = policy.get_tool_risk("test_tool")
    assert retrieved is not None
    assert retrieved.name == "test_tool"
    assert retrieved.risk_level == RiskLevel.NONE


def test_safety_policy_is_tool_permitted():
    """Test tool permission checking."""
    policy = SafetyPolicy()

    # NONE and LOW risk are permitted
    policy.register_tool_risk(
        ToolRisk("safe_tool", RiskLevel.NONE, False)
    )
    policy.register_tool_risk(
        ToolRisk("low_risk_tool", RiskLevel.LOW, False)
    )
    # MEDIUM and above are not
    policy.register_tool_risk(
        ToolRisk("risky_tool", RiskLevel.MEDIUM, True)
    )

    assert policy.is_tool_permitted("safe_tool") is True
    assert policy.is_tool_permitted("low_risk_tool") is True
    assert policy.is_tool_permitted("risky_tool") is False
    assert policy.is_tool_permitted("unknown_tool") is False


def test_safety_policy_check_tool_call():
    """Test tool call validation."""
    policy = SafetyPolicy()
    policy.register_tool_risk(
        ToolRisk("allowed", RiskLevel.NONE, False)
    )
    policy.register_tool_risk(
        ToolRisk("forbidden", RiskLevel.HIGH, True)
    )

    # Allowed tool passes
    assert policy.check_tool_call("allowed", {}) is True

    # Forbidden tool fails
    assert policy.check_tool_call("forbidden", {}) is False

    # Unknown tool fails
    assert policy.check_tool_call("unknown", {}) is False

    # Check audit events were recorded
    events = policy.audit_events()
    assert len(events) > 0
    assert any(e["event_type"] == "tool_approved" for e in events)
    assert any(e["event_type"] == "tool_rejected_risk" for e in events)
    assert any(e["event_type"] == "tool_rejected_unknown" for e in events)


def test_safety_policy_check_recommendation():
    """Test recommendation validation."""
    policy = SafetyPolicy()

    # Valid recommendation (requires approval, pending)
    valid_rec = Recommendation(
        id="rec_1",
        title="Test recommendation",
        description="Do something",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="medium",
    )
    assert policy.check_recommendation(valid_rec) is True

    # Invalid: does not require approval
    invalid_rec = Recommendation(
        id="rec_2",
        title="Test recommendation",
        description="Do something",
        requires_approval=False,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="medium",
    )
    assert policy.check_recommendation(invalid_rec) is False

    # Invalid: already approved
    approved_rec = Recommendation(
        id="rec_3",
        title="Test recommendation",
        description="Do something",
        requires_approval=True,
        approval_status=ApprovalStatus.APPROVED,
        priority="medium",
    )
    assert policy.check_recommendation(approved_rec) is False


def test_check_safety_convenience_wrapper():
    """Test the check_safety convenience function."""
    policy = SafetyPolicy()
    policy.register_tool_risk(
        ToolRisk("test", RiskLevel.NONE, False)
    )

    result = check_safety(policy, "test", {"arg": "value"})
    assert result is True


def test_default_safety_policy():
    """Test the default safety policy includes diagnostic_math."""
    policy = default_safety_policy()

    risk = policy.get_tool_risk("diagnostic_math")
    assert risk is not None
    assert risk.risk_level == RiskLevel.NONE
    assert risk.requires_approval is False

    # Should be permitted
    assert policy.is_tool_permitted("diagnostic_math") is True
