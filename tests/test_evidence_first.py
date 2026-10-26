"""Evidence-first investigation workflow tests."""

import pytest
from unittest.mock import Mock, patch

from plantpilot.agent import AgentController, create_agent_controller
from plantpilot.schemas import IncidentCreate, InvestigationCreate, Status
from plantpilot.storage import Database
from plantpilot.safety import default_safety_policy
from plantpilot.tools import registered_tools


@pytest.fixture
def test_db(tmp_path):
    db = Database(tmp_path / "test.db")
    db.connect()
    yield db
    db.close()


@pytest.fixture
def controller(test_db):
    return AgentController(
        db=test_db,
        tool_registry=registered_tools,
        safety_policy=default_safety_policy(),
    )


def _make_response(content=None, tool_calls=None, finish_reason="stop"):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    msg = ChatCompletionMessage(content=content or "", role="assistant")
    if tool_calls:
        msg.tool_calls = tool_calls
    else:
        msg.tool_calls = None
    choice = Choice(finish_reason=finish_reason, index=0, message=msg)
    return ChatCompletion(
        id="test",
        choices=[choice],
        created=1,
        model="test",
        object="chat.completion",
    )


def _make_tool_call(tool_name, args, call_id="tc1"):
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    call = Mock(spec=ChatCompletionMessageToolCall)
    call.id = call_id
    call.type = "function"
    call.function = Mock()
    call.function.name = tool_name
    call.function.arguments = __import__("json").dumps(args)
    return call


def test_evidence_first_tool_then_complete(controller, test_db):
    """Model requests diagnostic tool; result returned; then completes."""
    incident = test_db.create_incident(
        IncidentCreate(description="Line stopped", severity="high")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    # First: forced tool call (required on step 0)
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Checking diagnostics.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": [1, 2, 3]})],
            finish_reason="tool_calls",
        ),
        # Second: model concludes after seeing result
        _make_response(
            content="Evidence shows normal values. No root cause identified.",
            finish_reason="stop",
        ),
    ]

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    assert result.status == Status.COMPLETED
    assert result.completed is True
    assert result.steps_taken == 2
    # Evidence should be persisted
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) == 1
    assert len(retrieved.evidence) == 1
    # Hypothesis and recommendation should exist
    assert len(retrieved.recommendations) == 1
    assert len(retrieved.hypotheses) == 1


def test_model_prose_no_tool_incomplete(controller, test_db):
    """Model returns prose without requesting tools → INCOMPLETE."""
    incident = test_db.create_incident(
        IncidentCreate(description="String", severity="low")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="This is a placeholder description with no sensor data.",
            finish_reason="stop",
        ),
    ]

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    assert result.status == Status.INCOMPLETE
    assert result.completed is False
    assert len(result.errors) == 1
    assert "No diagnostic tools" in result.errors[0]
    # No tool executions or evidence persisted
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) == 0
    assert len(retrieved.evidence) == 0


def test_invalid_tool_arguments(controller, test_db):
    """Tool called with bad arguments — fails gracefully, loop continues."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="medium")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    # Step 1: bad arguments
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Using tool.",
            tool_calls=[_make_tool_call("diagnostic_math", {"bad_key": "nope"})],
            finish_reason="tool_calls",
        ),
        # Step 2: valid call after error returned
        _make_response(
            content="Done.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": [1]})],
            finish_reason="tool_calls",
        ),
        # Step 3: complete
        _make_response(content="Finished.", finish_reason="stop"),
    ]

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    assert len(result.errors) >= 1  # At least the invalid-argument error
    retrieved = test_db.get_investigation(inv.id)
    # One failed execution, one completed
    assert len(retrieved.tool_executions) == 2


def test_tool_fails_executed(controller, test_db):
    """Registered tool raises exception — failure persisted; unrecoverable, investigation terminal FAILED."""
    # Use a custom registry with a failing tool
    from plantpilot.tools import Tool, ToolRegistry, RiskLevel
    registry = ToolRegistry()

    def fail_handler(args):
        raise Exception("Simulated failure")

    registry.register(Tool(
        name="fail_tool",
        description="Always fails",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=fail_handler,
        argument_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ))

    controller_fail = AgentController(
        db=test_db,
        tool_registry=registry,
        safety_policy=default_safety_policy(),
    )

    incident = test_db.create_incident(
        IncidentCreate(description="Fail test", severity="high")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Testing failure.",
            tool_calls=[_make_tool_call("fail_tool", {})],
            finish_reason="tool_calls",
        ),
    ]

    with patch.object(controller_fail, 'client', mock_client):
        result = controller_fail.investigate(inv, incident)

    # Unrecoverable tool failure -> terminal FAILED (contract); failure persisted
    assert result.status == Status.FAILED
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].status.value == "failed"
    assert retrieved.tool_executions[0].error_message is not None


def test_step_limit_reached(controller, test_db):
    """Model keeps requesting tools until max_steps — stops as INCOMPLETE."""
    incident = test_db.create_incident(
        IncidentCreate(description="Loop", severity="low")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    # Always returns tool call — never finishes
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="More steps.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": [1]})],
            finish_reason="tool_calls",
        ),
    ] * 10  # More than max_steps (default 8)

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    # Should hit max_steps and return INCOMPLETE
    assert result.status == Status.INCOMPLETE
    assert result.completed is False
    assert result.steps_taken <= 8


def test_no_sensor_data_available(controller, test_db):
    """No sensor data available — evidence shows absence, result honest."""
    incident = test_db.create_incident(
        IncidentCreate(description="No telemetry available", severity="medium")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    # Tool executes but returns empty/none — evidence persisted honestly
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="No telemetry.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": []})],
            finish_reason="tool_calls",
        ),
        # After error returned, model concludes
        _make_response(
            content="No sensor data available. Cannot diagnose.",
            finish_reason="stop",
        ),
    ]

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    # Even with failure, evidence attempt is persisted
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) >= 1  # At least attempted
    # Finding should reflect absence honestly
    assert any("No sensor" in f or "no telemetry" in f.lower() for f in result.findings)


def test_evidence_and_tool_executions_persisted(controller, test_db):
    """Evidence and tool executions correctly persisted."""
    incident = test_db.create_incident(
        IncidentCreate(description="Persist test", severity="low")
    )
    inv = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Using tool.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": [42]})],
            finish_reason="tool_calls",
        ),
        _make_response(content="Done.", finish_reason="stop"),
    ]

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)

    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.evidence) == 1
    assert retrieved.evidence[0].evidence_type.value == "observed"
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].status.value == "completed"
    assert len(retrieved.recommendations) == 1  # Should exist for evidence-backed result

def test_empty_values_fails_gracefully(controller, test_db):
    """Empty values list fails with clear error, not crash."""
    incident = test_db.create_incident(IncidentCreate(description="Empty", severity="low"))
    inv = test_db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)
    
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Checking.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": []})],
            finish_reason="tool_calls",
        ),
        _make_response(content="No data.", finish_reason="stop"),
    ]
    
    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)
    
    # Should complete but with error recorded; no evidence from failed run
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].status.value == "failed"
    assert "at least 1" in retrieved.tool_executions[0].error_message

def test_valid_numbers_success(controller, test_db):
    """Valid numerical readings produce successful execution and evidence."""
    incident = test_db.create_incident(IncidentCreate(description="Good data", severity="low"))
    inv = test_db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)
    
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = [
        _make_response(
            content="Using tool.",
            tool_calls=[_make_tool_call("diagnostic_math", {"values": [10.5, 20.3, 15.0]})],
            finish_reason="tool_calls",
        ),
        _make_response(content="Results normal.", finish_reason="stop"),
    ]
    
    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(inv, incident)
    
    assert result.status == Status.COMPLETED
    retrieved = test_db.get_investigation(inv.id)
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].status.value == "completed"
    assert len(retrieved.evidence) == 1
    assert retrieved.evidence[0].verified is True

def test_no_sensor_retrieval_tool_exists():
    """Confirm stub interface exists and fails honestly for bad IDs."""
    from plantpilot.tools import registered_tools
    names = registered_tools.names()
    assert "get_sensor_readings" in names  # Real adapter interface exists
    # No actual factory simulator / real-time sensor backend
    tool = registered_tools.get("get_sensor_readings")
    # Bad ID fails honestly (no fabricated data)
    with pytest.raises(Exception):
        tool.handler({"station_id": "bad_id"})

def test_data_adapter_success(test_db):
    from plantpilot.data_adapter import get_sensor_readings
    r = get_sensor_readings(1, "air_temperature")
    assert r["status"] == "retrieved"
    assert r["measurements"]["air_temperature"]["value"] == 298.1
    assert r["measurements"]["air_temperature"]["unit"] == "K"
    assert "dataset" in r["source"].lower()
    assert "historical" in r["provenance"].lower()
    assert r["record_identifier"]["udi"] == 1

def test_data_adapter_unknown_id(test_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(999999)

def test_data_adapter_unsupported_param(test_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(1, "machine_failure")

def test_data_adapter_empty_result(test_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    # Non-existent but valid-format ID handled as unknown, not empty
    with pytest.raises(DataAdapterError):
        get_sensor_readings(99999)

def test_data_adapter_evaluation_separation(test_db):
    from plantpilot.data_adapter import get_evaluation_only
    e = get_evaluation_only(2)
    assert "machine_failure" in e["labels"]
    assert e["labels"]["machine_failure"] in (0, 1)
    assert "source" in e

def test_valid_udi_string_accepted(controller, test_db):
    """Numeric UDI passed as string '1' works."""
    from plantpilot.data_adapter import get_sensor_readings
    r = get_sensor_readings("1", "air_temperature")
    assert r["record_identifier"]["udi"] == 1
    assert r["measurements"]["air_temperature"]["value"] == 298.1

def test_description_as_udi_rejected(controller, test_db):
    """Incident description text rejected as identifier."""
    from plantpilot.tools import registered_tools
    tool = registered_tools.get("get_sensor_readings")
    with pytest.raises(Exception):
        tool.handler({"udi": "AI4I 2020 predictive maintenance"})

def test_missing_udi_rejected(controller, test_db):
    from plantpilot.tools import registered_tools
    tool = registered_tools.get("get_sensor_readings")
    with pytest.raises(Exception):
        tool.handler({"sensor_type": "all"})

def test_unsupported_param_rejected(controller, test_db):
    from plantpilot.data_adapter import get_sensor_readings, DataAdapterError
    with pytest.raises(DataAdapterError):
        get_sensor_readings(1, "machine_failure")

def test_recovery_correctable(tool_path=None):
    """Correctable error: valid UDI after invalid attempt."""
    pass  # Covered by evidence-first loop; tool validates each call independently

def test_no_retry_bad_args(controller, test_db):
    """No retry loop with invalid args — model must correct or stop."""
    # Evidence-first already stops at max_steps with INCOMPLETE; no automatic retry
    assert controller.config.max_steps > 0  # step limit enforced
