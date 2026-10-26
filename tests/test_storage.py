"""Tests for SQLite persistence layer."""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from plantpilot.storage import Database
from plantpilot.schemas import (
    IncidentCreate,
    InvestigationCreate,
    Status,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Recommendation,
    ApprovalStatus,
    ToolExecution,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    yield db
    db.close()


def test_database_initialization(test_db):
    """Test database schema is created."""
    # Should not raise
    assert test_db._conn is not None


def test_create_incident(test_db):
    """Test creating an incident."""
    create = IncidentCreate(
        description="Test incident",
        severity="high",
        category="operational",
        metadata={"key": "value"},
    )

    incident = test_db.create_incident(create)

    assert incident.id.startswith("inc_")
    assert incident.description == "Test incident"
    assert incident.severity == "high"
    assert incident.status == Status.PENDING
    assert incident.metadata["key"] == "value"


def test_get_incident(test_db):
    """Test retrieving an incident."""
    create = IncidentCreate(
        description="Test incident",
        severity="medium",
    )

    created = test_db.create_incident(create)
    retrieved = test_db.get_incident(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.description == created.description


def test_get_incident_not_found(test_db):
    """Test retrieving a non-existent incident returns None."""
    result = test_db.get_incident("inc_nonexistent")
    assert result is None


def test_list_incidents(test_db):
    """Test listing incidents."""
    for i in range(3):
        create = IncidentCreate(
            description=f"Incident {i}",
            severity="low",
        )
        test_db.create_incident(create)

    incidents = test_db.list_incidents()
    assert len(incidents) == 3
    # Newest first
    assert "Incident 2" in incidents[0].description


def test_list_incidents_pagination(test_db):
    """Test incident pagination."""
    for i in range(10):
        create = IncidentCreate(
            description=f"Incident {i}",
            severity="low",
        )
        test_db.create_incident(create)

    page1 = test_db.list_incidents(limit=5, offset=0)
    assert len(page1) == 5

    page2 = test_db.list_incidents(limit=5, offset=5)
    assert len(page2) == 5

    # Different incidents
    assert page1[0].id != page2[0].id


def test_create_investigation(test_db):
    """Test creating an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )

    create = InvestigationCreate(
        incident_id=incident.id,
        notes="Starting investigation",
    )

    investigation = test_db.create_investigation(create, incident)

    assert investigation.id.startswith("inv_")
    assert investigation.incident_id == incident.id
    assert investigation.status == Status.IN_PROGRESS
    assert investigation.notes == "Starting investigation"


def test_get_investigation(test_db):
    """Test retrieving an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)

    created = test_db.create_investigation(create, incident)
    retrieved = test_db.get_investigation(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.incident_id == incident.id


def test_get_investigation_not_found(test_db):
    """Test retrieving a non-existent investigation returns None."""
    result = test_db.get_investigation("inv_nonexistent")
    assert result is None


def test_update_investigation(test_db):
    """Test updating an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    test_db.update_investigation(
        investigation.id,
        status=Status.COMPLETED,
        findings=["Finding 1", "Finding 2"],
        completed_at=datetime.now(timezone.utc),
    )

    retrieved = test_db.get_investigation(investigation.id)
    assert retrieved.status == Status.COMPLETED
    assert len(retrieved.findings) == 2
    assert retrieved.completed_at is not None


def test_save_and_load_evidence(test_db):
    """Test saving and loading evidence."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    evidence = EvidenceItem(
        id="ev_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Temperature reading",
        source="sensor_01",
        value=88.0,
        verified=True,
    )

    test_db.save_evidence(investigation.id, evidence)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.evidence) == 1
    assert retrieved.evidence[0].id == "ev_1"
    assert retrieved.evidence[0].value == 88.0


def test_save_and_load_hypothesis(test_db):
    """Test saving and loading hypotheses."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    hypothesis = Hypothesis(
        id="hyp_1",
        title="Root cause hypothesis",
        explanation="Detailed explanation",
        confidence="medium",
        supporting_evidence_ids=["ev_1"],
    )

    test_db.save_hypothesis(investigation.id, hypothesis)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.hypotheses) == 1
    assert retrieved.hypotheses[0].title == "Root cause hypothesis"


def test_save_and_load_recommendation(test_db):
    """Test saving and loading recommendations."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    recommendation = Recommendation(
        id="rec_1",
        title="Take action",
        description="Detailed recommendation",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="high",
    )

    test_db.save_recommendation(investigation.id, recommendation)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.recommendations) == 1
    assert retrieved.recommendations[0].title == "Take action"


def test_save_and_load_tool_execution(test_db):
    """Test saving and loading tool executions."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    execution = ToolExecution(
        id="exec_1",
        tool_name="diagnostic_math",
        arguments={"values": [1, 2, 3]},
        result='{"mean": 2.0}',
        status=Status.COMPLETED,
    )

    test_db.save_tool_execution(investigation.id, execution)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].tool_name == "diagnostic_math"


def test_save_audit_event(test_db):
    """Test saving an audit event."""
    # Should not raise
    test_db.save_audit_event(
        event_type="test_event",
        message="Test message",
        details={"key": "value"},
    )


def test_database_persistence(tmp_path):
    """Test data persists across connections."""
    db_path = tmp_path / "test.db"

    # Create and save data
    db1 = Database(db_path)
    db1.connect()
    create = IncidentCreate(description="Test", severity="high")
    incident = db1.create_incident(create)
    incident_id = incident.id
    db1.close()

    # Reopen and verify
    db2 = Database(db_path)
    db2.connect()
    retrieved = db2.get_incident(incident_id)
    assert retrieved is not None
    assert retrieved.id == incident_id
    db2.close()



def test_conversation_round_trip_and_investigation_events(test_db):
    incident = test_db.create_incident(
        IncidentCreate(description="Temperature anomaly", severity="medium")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    conversation = test_db.create_conversation(investigation.id)
    assert conversation["id"].startswith("conv_")
    user_message = test_db.add_conversation_message(
        conversation["id"], "user", "Why was this flagged?"
    )
    assistant_message = test_db.add_conversation_message(
        conversation["id"], "assistant", "The stored evidence shows an unusual reading."
    )

    loaded = test_db.get_conversation(conversation["id"])
    assert loaded is not None
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][0]["id"] == user_message["id"]
    assert loaded["messages"][1]["id"] == assistant_message["id"]

    test_db.save_audit_event(
        "investigation_step_started", "Step 1 started", {"step": 1}, investigation.id
    )
    events = test_db.list_audit_events(investigation.id)
    assert any(event["type"] == "investigation_step_started" for event in events)


def test_create_conversation_rejects_unknown_investigation(test_db):
    with pytest.raises(KeyError):
        test_db.create_conversation("inv_does_not_exist")
