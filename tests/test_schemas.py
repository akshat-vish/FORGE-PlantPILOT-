"""Tests for Pydantic data models."""

import pytest
from datetime import datetime, timezone

from plantpilot.schemas import (
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Recommendation,
    ApprovalStatus,
    ToolExecution,
    Status,
    IncidentCreate,
    Incident,
    InvestigationCreate,
    Investigation,
    InvestigationResult,
)


def test_evidence_item_creation():
    """Test creating an evidence item."""
    evidence = EvidenceItem(
        id="ev_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Temperature reading: 88°C",
        source="sensor_temp_01",
        value=88.0,
        verified=True,
    )
    assert evidence.id == "ev_1"
    assert evidence.evidence_type == EvidenceType.OBSERVED
    assert evidence.verified is True


def test_evidence_item_defaults():
    """Test evidence item defaults."""
    evidence = EvidenceItem(
        id="ev_2",
        description="Something happened",
        source="system",
    )
    assert evidence.evidence_type == EvidenceType.OBSERVED
    assert evidence.verified is False
    assert evidence.value is None
    assert isinstance(evidence.timestamp, datetime)


def test_hypothesis_creation():
    """Test creating a hypothesis."""
    hypothesis = Hypothesis(
        id="hyp_1",
        title="Motor overheating due to bearing wear",
        explanation="Elevated temperature correlates with reduced lubrication",
        confidence="medium",
        supporting_evidence_ids=["ev_1", "ev_2"],
        contradicting_evidence_ids=[],
    )
    assert hypothesis.id == "hyp_1"
    assert hypothesis.confidence == "medium"
    assert len(hypothesis.supporting_evidence_ids) == 2


def test_recommendation_creation():
    """Test creating a recommendation."""
    recommendation = Recommendation(
        id="rec_1",
        title="Inspect motor bearings",
        description="Schedule maintenance check for motor bearings within 24 hours",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="high",
    )
    assert recommendation.requires_approval is True
    assert recommendation.approval_status == ApprovalStatus.PENDING_APPROVAL
    assert recommendation.priority == "high"


def test_tool_execution_creation():
    """Test creating a tool execution record."""
    execution = ToolExecution(
        id="exec_1",
        tool_name="diagnostic_math",
        arguments={"values": [1, 2, 3]},
        result='{"mean": 2.0}',
        status=Status.COMPLETED,
    )
    assert execution.tool_name == "diagnostic_math"
    assert execution.status == Status.COMPLETED
    assert execution.error_message is None


def test_incident_create_validation():
    """Test incident creation validates required fields."""
    with pytest.raises(Exception):  # Pydantic validation error
        IncidentCreate(description="")  # Too short


def test_incident_create_with_metadata():
    """Test incident creation with metadata."""
    create = IncidentCreate(
        description="Production line stopped",
        severity="high",
        category="operational",
        metadata={"station": "A", "line": 3},
    )
    assert create.metadata["station"] == "A"
    assert create.metadata["line"] == 3


def test_investigation_creation():
    """Test investigation creation."""
    investigation = Investigation(
        id="inv_1",
        incident_id="inc_1",
        status=Status.IN_PROGRESS,
        started_at=datetime.now(timezone.utc),
    )
    assert investigation.status == Status.IN_PROGRESS
    assert len(investigation.findings) == 0
    assert len(investigation.evidence) == 0
    assert len(investigation.tool_executions) == 0


def test_investigation_result_creation():
    """Test investigation result."""
    result = InvestigationResult(
        investigation_id="inv_1",
        status=Status.COMPLETED,
        findings=["Finding 1", "Finding 2"],
        evidence_ids=["ev_1", "ev_2"],
        hypothesis_ids=["hyp_1"],
        recommendation_ids=["rec_1"],
        errors=[],
        steps_taken=3,
        completed=True,
    )
    assert result.completed is True
    assert result.steps_taken == 3
    assert len(result.findings) == 2
