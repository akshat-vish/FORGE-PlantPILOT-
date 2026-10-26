"""Tests for agent controller behavior with mocked model client."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from plantpilot.agent import AgentController, AgentError
from plantpilot.schemas import (
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    Status,
)
from plantpilot.storage import Database
from plantpilot.tools import registered_tools
from plantpilot.safety import default_safety_policy


class MockModelClient:
    """Mock OpenAI client for testing."""

    def __init__(self, responses=None, error=None):
        self.responses = responses or []
        self.error = error
        self.call_count = 0

    def __getattr__(self, name):
        # Return a callable for any chat.completions.create
        if name == "chat":
            return self
        if name == "completions":
            return self
        if name == "create":
            return self._create
        return Mock()

    def _create(self, **kwargs):
        if self.error:
            raise self.error

        if self.call_count >= len(self.responses):
            # Return final message with no tool calls
            return self._make_response(content="Final findings and report.")

        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def _make_response(self, content=None, tool_calls=None, finish_reason="stop"):
        msg = ChatCompletionMessage(content=content or "", role="assistant")
        if tool_calls:
            msg.tool_calls = tool_calls
        else:
            msg.tool_calls = None

        choice = Choice(finish_reason=finish_reason, index=0, message=msg)
        return ChatCompletion(
            id="test",
            choices=[choice],
            created=int(datetime.now(timezone.utc).timestamp()),
            model="test",
            object="chat.completion",
        )


def _make_tool_call(tool_name, arguments, call_id="call_1"):
    """Create a mock tool call object."""
    call = Mock(spec=ChatCompletionMessageToolCall)
    call.id = call_id
    call.type = "function"
    call.function = Mock()
    call.function.name = tool_name
    call.function.arguments = json.dumps(arguments)
    return call


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


def test_agent_investigate_completes(controller, test_db):
    """Test investigation completes successfully with tool calls."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    # Mock the model client
    mock_client = MockModelClient()

    # First response: tool call
    tool_call = _make_tool_call("diagnostic_math", {"values": [1, 2, 3]})
    mock_response = mock_client._make_response(
        content="Let me check the diagnostics.",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )
    mock_client.responses.append(mock_response)

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    assert result.status == Status.COMPLETED
    assert result.completed is True
    # Verify actual tool executions persisted (accurate count from audit)
    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.tool_executions) == 1  # diagnostic_math executed


def test_agent_investigate_unknown_tool(controller, test_db):
    """Test investigation rejects unknown tool calls."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = MockModelClient()

    # Tool call to unknown tool
    tool_call = _make_tool_call("unknown_tool", {"arg": "value"})
    mock_response = mock_client._make_response(
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )
    mock_client.responses.append(mock_response)

    # Then a final message
    mock_client.responses.append(mock_client._make_response(content="Done"))

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    assert Status.INCOMPLETE in result.errors or len(result.errors) > 0


def test_agent_investigate_max_steps(controller, test_db):
    """Test investigation stops at max steps."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = MockModelClient()

    # Always return a tool call, never completes
    tool_call = _make_tool_call("diagnostic_math", {"values": [1, 2, 3]})

    for _ in range(20):
        mock_client.responses.append(
            mock_client._make_response(
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        )

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    assert result.status == Status.INCOMPLETE
    assert result.completed is False
    assert result.steps_taken == controller.config.max_steps


def test_agent_investigate_api_error(controller, test_db):
    """Test investigation handles API errors."""
    from openai import OpenAIError

    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = MockModelClient(error=RuntimeError("API error"))

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    assert result.status in (Status.INCOMPLETE, Status.FAILED)
    assert result.completed is False
    assert len(result.errors) > 0


def test_agent_investigate_invalid_tool_arguments(controller, test_db):
    """Test investigation handles malformed tool call arguments."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = MockModelClient()

    # Tool call with malformed arguments (invalid JSON)
    tool_call = Mock(spec=ChatCompletionMessageToolCall)
    tool_call.id = "call_1"
    tool_call.type = "function"
    tool_call.function = Mock()
    tool_call.function.name = "diagnostic_math"
    tool_call.function.arguments = "{invalid json"

    mock_client.responses.append(
        mock_client._make_response(
            tool_calls=[tool_call],
            finish_reason="tool_calls",
        )
    )
    mock_client.responses.append(mock_client._make_response(content="Done"))

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    # Should have recorded an error about invalid arguments
    assert len(result.errors) > 0


def test_agent_investigate_missing_content(controller, test_db):
    """Test investigation handles missing assistant content."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test incident", severity="high")
    )
    investigation = test_db.create_investigation(
        InvestigationCreate(incident_id=incident.id), incident
    )

    mock_client = MockModelClient()

    # Response with no content and no tool calls (model just stops)
    mock_client.responses.append(
        mock_client._make_response(
            content=None,
            tool_calls=None,
            finish_reason="stop",
        )
    )

    with patch.object(controller, 'client', mock_client):
        result = controller.investigate(investigation, incident)

    # Evidence-first: should be INCOMPLETE since no tools executed
    assert result.status == Status.INCOMPLETE
    assert result.completed is False
    assert len(result.errors) > 0
    assert "No diagnostic tools" in result.errors[0]


def test_agent_build_initial_messages(controller):
    """Test the prompt construction."""
    incident = Incident(
        id="inc_1",
        description="Test incident description",
        severity="high",
        category="operational",
        status=Status.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata={},
    )

    messages = controller._build_initial_messages(incident)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "PlantPilot" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Test incident description" in messages[1]["content"]
