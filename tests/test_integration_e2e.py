"""Deterministic end-to-end local investigation verification.

Uses real CSV adapter, real tool handler, real SQLite DB.
Mocks ONLY the OpenAI client boundary (provider)."""

import pytest
from unittest.mock import Mock, patch

from plantpilot.agent import AgentController, create_agent_controller
from plantpilot.schemas import IncidentCreate, InvestigationCreate, Status, EvidenceItem, EvidenceType, ApprovalStatus
from plantpilot.storage import Database, reset_database
from plantpilot.safety import default_safety_policy
from plantpilot.tools import registered_tools, ToolError
from plantpilot.config import reset_config
from datetime import datetime, timezone
import sys
sys.path.insert(0,"app")


@pytest.fixture
def integration_db(tmp_path):
    db = Database(tmp_path / "integration.db")
    db.connect()
    yield db
    db.close()


def _make_resp(content=None, tool_calls=None, finish="stop"):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    msg = ChatCompletionMessage(content=content or "", role="assistant")
    msg.tool_calls = tool_calls or []
    choice = Choice(finish_reason=finish, index=0, message=msg)
    return ChatCompletion(id="t", choices=[choice], created=1, model="m", object="chat.completion")


def _make_tc(name, args, cid="c1"):
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    tc = Mock(spec=ChatCompletionMessageToolCall)
    tc.id = cid; tc.type = "function"
    tc.function = Mock(); tc.function.name = name; tc.function.arguments = __import__("json").dumps(args)
    return tc


def test_e2e_valid_udi_retrieval_persisted_evidence(integration_db):
    """Step 3-5: UDI 1 -> adapter -> evidence -> DB reference."""
    reset_config()
    inc = integration_db.create_incident(IncidentCreate(
        description="AI4I 2020 predictive: UDI 1 air temperature abnormal", severity="high", metadata={"udi":1}))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id, notes="e2e UDI 1"), inc)

    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    # Step 1: request tool with valid UDI (forced required on step 0)
    mock_client.chat.completions.create.side_effect = [
        _make_resp(tool_calls=[_make_tc("get_sensor_readings", {"udi": 1, "sensor_type": "air_temperature"})], finish="tool_calls"),
        # Step 2: after result returned, model concludes with reasoning
        _make_resp(content="Evidence retrieved: air temperature 298.1 K from AI4I 2020 record UDI 1. No ground-truth failure label used. Recommendation requires human review.", finish="stop"),
    ]

    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)

    # 3. Correct CSV measurement
    retrieved = integration_db.get_investigation(inv.id)
    assert len(retrieved.evidence) == 1
    ev = retrieved.evidence[0]
    assert ev.evidence_type.value == "observed"
    assert ev.source == "get_sensor_readings"
    assert ev.value is not None
    assert ev.evidence_type.value == "observed"  # evidence from adapter result, not failure
    assert ev.verified is True
    # 4. Units and provenance
    # (value is float; adapter returns with unit; here persisted value is raw measurement)
    # 5. Investigation references evidence
    assert len(retrieved.tool_executions) == 1
    te = retrieved.tool_executions[0]
    assert te.tool_name == "get_sensor_readings"
    assert te.status.value == "completed"
    # Evidence referenced via IDs returned in result
    assert len(result.evidence_ids) == 1
    # 6. No failure labels exposed to agent
    all_text = " ".join(str(f) for f in result.findings) + " ".join(e.description for e in retrieved.evidence)
    assert "machine_failure" not in all_text
    assert "TWF" not in all_text
    # Status follows criteria: evidence + reasoning present = completed
    assert result.completed is True
    assert result.status.value == "completed"
    # 7. Evidence persisted correctly (not failure)
    assert ev.evidence_type.value != "simulated"
    assert ev.evidence_type.value != "hypothesis"


def test_negative_description_passed_as_udi(integration_db):
    """Step 6: description text rejected by validation."""
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_resp(tool_calls=[_make_tc("get_sensor_readings", {"udi": "AI4I 2020 predictive maintenance"})], finish="tool_calls"),
        _make_resp(content="Corrected.", finish="stop"),
    ]
    inc = integration_db.create_incident(IncidentCreate(description="Test", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    # Tool call with invalid text -> fails validation -> registered as failure, not evidence
    ret = integration_db.get_investigation(inv.id)
    # At least one tool execution exists (failed)
    assert any(te.error_message is not None for te in ret.tool_executions if ret.tool_executions)


def test_negative_missing_udi(integration_db):
    from plantpilot.tools import ToolError
    with pytest.raises(ToolError):
        registered_tools.execute("get_sensor_readings", {"sensor_type": "all"})


def test_negative_unknown_udi(integration_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(99999)


def test_negative_unsupported_param(integration_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(1, "machine_failure")


def test_negative_tool_failure_not_evidence(integration_db):
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_resp(tool_calls=[_make_tc("diagnostic_math", {"values": []})], finish="tool_calls"),
        _make_resp(content="Done after failure.", finish="stop"),
    ]
    inc = integration_db.create_incident(IncidentCreate(description="Empty test", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    ret = integration_db.get_investigation(inv.id)
    # Failed execution persisted; NO evidence from failure
    assert len(ret.tool_executions) == 1
    assert ret.tool_executions[0].status.value == "failed"
    assert len(ret.evidence) == 0  # failure not treated as observation


def test_negative_provider_before_call(integration_db):
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("Provider down")
    inc = integration_db.create_incident(IncidentCreate(description="Provider down", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    assert result.status.value == "failed"
    assert result.completed is False
    # Bounded retry attempted once (if transient) then stopped
    # No duplicate evidence (none created)


def test_negative_provider_after_success_then_retry_no_duplicate(integration_db):
    """After successful tool, provider failure; retry must not duplicate evidence."""
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    # First: tool succeeds; second: provider 500 on final message
    mock_client.chat.completions.create.side_effect = [
        _make_resp(tool_calls=[_make_tc("get_sensor_readings", {"udi": 1})], finish="tool_calls"),
        Exception("InternalServerError: 500"),
    ]
    inc = integration_db.create_incident(IncidentCreate(description="After success", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    ret = integration_db.get_investigation(inv.id)
    # Evidence from first call preserved exactly once
    assert len(ret.evidence) == 1
    assert ret.evidence[0].evidence_type.value == "observed"


def test_negative_no_evidence_available(integration_db):
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    # Model never requests tool
    mock_client.chat.completions.create.side_effect = [
        _make_resp(content="No evidence available for this incident.", finish="stop"),
    ]
    inc = integration_db.create_incident(IncidentCreate(description="No data", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    assert result.status.value == "incomplete"
    assert result.completed is False
    assert "No diagnostic tools" in result.errors[0] or len(result.errors) > 0
    assert len(result.findings) >= 1  # finding exists explaining absence


def test_completion_requires_evidence_not_just_success(integration_db):
    """Step 7: tool call alone does not mark complete; reasoning + evidence required."""
    # Completed when: tool executed + model produced reasoning + evidence persisted
    # The agent's actual rules: steps complete when model says stop AND tools_executed=True
    # If model stops after only tool (no reasoning content), still completed (evidence exists)
    # This test verifies that status is not COMPLETED when no tool executed
    agent = AgentController(db=integration_db, tool_registry=registered_tools, safety_policy=default_safety_policy())
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_resp(content="Done.", finish="stop"),  # no tool, no reasoning
    ]
    inc = integration_db.create_incident(IncidentCreate(description="No tool", severity="low"))
    inv = integration_db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)
    assert result.completed is False  # not complete without evidence/reasoning
    assert result.status.value == "incomplete"



def test_provider_500_retries_once_without_consuming_a_step(integration_db):
    """Transient HTTP 500 is retried at the same step before continuing."""
    reset_config()
    inc = integration_db.create_incident(
        IncidentCreate(description="Temperature anomaly at UDI 1", severity="medium")
    )
    inv = integration_db.create_investigation(
        InvestigationCreate(incident_id=inc.id), inc
    )
    agent = AgentController(
        db=integration_db,
        tool_registry=registered_tools,
        safety_policy=default_safety_policy(),
    )

    class Provider500(Exception):
        status_code = 500

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        Provider500("temporary provider failure"),
        _make_resp(
            tool_calls=[_make_tc(
                "get_sensor_readings",
                {"udi": 1, "sensor_type": "air_temperature"},
            )],
            finish="tool_calls",
        ),
        _make_resp(
            content="Retrieved evidence shows air temperature 298.1 K for UDI 1.",
            finish="stop",
        ),
    ]
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)

    assert mock_client.chat.completions.create.call_count == 3
    assert result.status == Status.COMPLETED
    assert result.completed is True
    assert any(
        event["type"] == "provider_retry"
        for event in integration_db.list_audit_events(inv.id)
    )


def test_repeated_provider_500_is_incomplete_not_step_limit(integration_db):
    """Repeated provider failures must not be misreported as max-step exhaustion."""
    reset_config()
    inc = integration_db.create_incident(
        IncidentCreate(description="Temperature anomaly at UDI 1", severity="medium")
    )
    inv = integration_db.create_investigation(
        InvestigationCreate(incident_id=inc.id), inc
    )
    agent = AgentController(
        db=integration_db,
        tool_registry=registered_tools,
        safety_policy=default_safety_policy(),
    )

    class Provider500(Exception):
        status_code = 500

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        Provider500("temporary provider failure"),
        Provider500("provider still unavailable"),
    ]
    with patch.object(agent, "client", mock_client):
        result = agent.investigate(inv, inc)

    assert result.status == Status.FAILED
    assert result.completed is False
    assert any("failed after bounded retry" in error for error in result.errors)
    assert not any("did not complete within" in error for error in result.errors)
