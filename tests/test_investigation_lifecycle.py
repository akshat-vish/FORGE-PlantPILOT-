"""Focused lifecycle tests for permanent provider failure."""
from unittest.mock import Mock, patch
from plantpilot.agent import AgentController
from plantpilot.schemas import IncidentCreate, InvestigationCreate, Status
from plantpilot.storage import Database
from plantpilot.safety import default_safety_policy
from plantpilot.tools import registered_tools


def _resp(content=None, tool_calls=None, finish="stop"):
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    msg = ChatCompletionMessage(content=content or "", role="assistant")
    msg.tool_calls = tool_calls or []
    return ChatCompletion(
        id="t",
        choices=[Choice(finish_reason=finish, index=0, message=msg)],
        created=1, model="m", object="chat.completion",
    )


def test_permanent_failure_transitions_to_failed(tmp_path):
    db = Database(tmp_path / "l.db"); db.connect()
    controller = AgentController(
        db=db, tool_registry=registered_tools, safety_policy=default_safety_policy()
    )
    inc = db.create_incident(IncidentCreate(description="F", severity="low"))
    inv = db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("InternalServerError: 500")
    with patch.object(controller, "client", mock_client):
        result = controller.investigate(inv, inc)
    assert result.status == Status.FAILED
    assert result.completed is False
    events = db.list_audit_events(inv.id)
    types = [e["type"] for e in events]
    assert "api_error" in types
    assert types.count("investigation_failed") == 1
    assert types.index("api_error") < types.index("investigation_failed")
    assert db.get_investigation(inv.id).status == Status.FAILED


def test_success_still_complete(tmp_path):
    db = Database(tmp_path / "l2.db"); db.connect()
    controller = AgentController(
        db=db, tool_registry=registered_tools, safety_policy=default_safety_policy()
    )
    inc = db.create_incident(IncidentCreate(description="S", severity="low"))
    inv = db.create_investigation(InvestigationCreate(incident_id=inc.id), inc)
    mock_client = Mock()
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    tc = Mock(spec=ChatCompletionMessageToolCall)
    tc.id = "c1"; tc.type = "function"
    tc.function = Mock(); tc.function.name = "get_sensor_readings"; tc.function.arguments = '{"udi":1}'
    mock_client.chat.completions.create.side_effect = [
        _resp(content="Check.", tool_calls=[tc], finish="tool_calls"),
        _resp(content="Done.", finish="stop"),
    ]
    with patch.object(controller, "client", mock_client):
        result = controller.investigate(inv, inc)
    assert result.status == Status.COMPLETED
    assert result.completed is True
    events = db.list_audit_events(inv.id)
    assert any(e["type"] == "investigation_complete" for e in events)
