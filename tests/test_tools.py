"""Tests for the tool registry and built-in tools."""

import pytest
import json

from plantpilot.tools import (
    Tool,
    ToolRegistry,
    ToolError,
    registered_tools,
)
from plantpilot.safety import RiskLevel


def test_tool_registry_register():
    """Test registering a tool."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {"result": "ok"}

    tool = Tool(
        name="dummy",
        description="A dummy tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    registry.register(tool)
    assert registry.get("dummy") is not None
    assert "dummy" in registry.names()


def test_tool_registry_duplicate_registration():
    """Test that duplicate registration raises an error."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {}

    tool = Tool(
        name="dummy",
        description="A dummy tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={"type": "object"},
    )

    registry.register(tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_tool_to_openai_format():
    """Test tool conversion to OpenAI format."""
    def dummy_handler(args):
        return {}

    tool = Tool(
        name="test_tool",
        description="Test description",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "required": ["value"],
        },
    )

    openai_format = tool.to_openai_format()
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "test_tool"
    assert openai_format["function"]["description"] == "Test description"
    assert "properties" in openai_format["function"]["parameters"]


def test_diagnostic_math_tool():
    """Test the built-in diagnostic_math tool."""
    tool = registered_tools.get("diagnostic_math")
    assert tool is not None
    assert tool.risk_level == RiskLevel.NONE

    result = tool.handler({"values": [1, 2, 3, 4, 5]})
    assert result["count"] == 5
    assert result["sum"] == 15
    assert result["mean"] == 3.0
    assert result["min"] == 1
    assert result["max"] == 5
    assert result["std_dev"] > 0


def test_diagnostic_math_validation():
    """Test diagnostic_math validates input."""
    tool = registered_tools.get("diagnostic_math")

    with pytest.raises(ToolError):
        tool.handler({"values": []})  # Empty list

    with pytest.raises(ToolError):
        tool.handler({"values": [1, "two", 3]})  # Non-numeric


def test_tool_registry_validate_arguments():
    """Test argument validation."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {}

    tool = Tool(
        name="test",
        description="Test",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "flag": {"type": "boolean"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    registry.register(tool)

    # Valid arguments
    validated = registry.validate_arguments("test", {"value": 42})
    assert validated["value"] == 42

    # Missing required argument
    with pytest.raises(ToolError, match="missing required argument"):
        registry.validate_arguments("test", {})

    # Unexpected argument
    with pytest.raises(ToolError, match="unexpected argument"):
        registry.validate_arguments("test", {"value": 42, "extra": "data"})

    # Wrong type
    with pytest.raises(ToolError, match="must be a number"):
        registry.validate_arguments("test", {"value": "not a number"})


def test_tool_registry_execute():
    """Test tool execution."""
    registry = ToolRegistry()

    def echo_handler(args):
        return {"echo": args.get("message", "")}

    tool = Tool(
        name="echo",
        description="Echo tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=echo_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    )
    registry.register(tool)

    result = registry.execute("echo", {"message": "hello"})
    assert result["echo"] == "hello"


def test_tool_registry_execute_unknown_tool():
    """Test executing an unknown tool raises an error."""
    registry = ToolRegistry()

    with pytest.raises(ToolError, match="Unknown tool"):
        registry.execute("nonexistent", {})


def test_tool_registry_to_openai_tools():
    """Test conversion of all tools to OpenAI format."""
    tools = registered_tools.to_openai_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(t["type"] == "function" for t in tools)
