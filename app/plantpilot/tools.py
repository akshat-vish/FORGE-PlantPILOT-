"""Tool registry — provider-independent.

Each tool is a registered, validated function with explicit permission and
risk metadata. Only tools registered by the backend may execute. The
registry is independent of the model provider and exposes the OpenAI
compatible tool-definition format for the NVIDIA endpoint.

Future sensor-data adapters should implement get_sensor_readings() and
register with a name like "get_station_readings" or "fetch_telemetry".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from plantpilot.safety import RiskLevel, ToolRisk
AGENT_PARAMS = {"air_temperature": "Air temperature [K]", "process_temperature": "Process temperature [K]", "rotational_speed": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]"}


from .data_adapter import get_sensor_readings, DataAdapterError


class ToolError(ValueError):
    """Raised when a tool call is rejected or fails."""


@dataclass
class Tool:
    """A registered tool."""
    name: str
    description: str
    risk_level: RiskLevel
    requires_approval: bool
    handler: Callable[[Dict[str, Any]], Any]
    argument_schema: Dict[str, Any]
    max_input_bytes: int = 4000

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to the OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.argument_schema,
            },
        }

    def to_risk(self) -> ToolRisk:
        return ToolRisk(
            name=self.name,
            risk_level=self.risk_level,
            requires_approval=self.requires_approval,
            description=self.description,
        )


class ToolRegistry:
    """Holds all registered tools and validates/executes calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Return all tools in OpenAI-compatible format."""
        return [t.to_openai_format() for t in self._tools.values()]

    def validate_arguments(self, tool_name: str, arguments: Any) -> Dict[str, Any]:
        """Validate and normalize tool arguments.

        Rejects unknown tools, non-dict arguments, and oversized inputs.
        Validates against the tool's declared JSON schema.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool: {tool_name!r}")
        if not isinstance(arguments, dict):
            raise ToolError(
                f"Tool {tool_name!r} arguments must be a JSON object, got {type(arguments).__name__}"
            )
        serialized = json.dumps(arguments, default=str, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > tool.max_input_bytes:
            raise ToolError(
                f"Tool {tool_name!r} input too large: {len(serialized.encode('utf-8'))} bytes"
            )
        # Enforce the declared JSON schema (basic checks).
        schema = tool.argument_schema
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ToolError(
                    f"Tool {tool_name!r} missing required argument: {key!r}"
                )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in arguments:
                if key not in properties:
                    raise ToolError(
                        f"Tool {tool_name!r} has unexpected argument: {key!r}"
                    )
        # Type-check declared properties.
        for key, value in arguments.items():
            prop = properties.get(key)
            if prop is None:
                continue
            # Check array minItems constraint
            if prop.get("type") == "array" and "minItems" in prop:
                if not isinstance(value, list) or len(value) < prop["minItems"]:
                    raise ToolError(
                        f"Tool {tool_name!r} argument {key!r} must have at least {prop['minItems']} items"
                    )
            prop_type = prop.get("type")
            if prop_type == "number" and not isinstance(value, (int, float)):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a number"
                )
            if prop_type == "integer" and not isinstance(value, int):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be an integer"
                )
            if prop_type == "string" and not isinstance(value, str):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a string"
                )
            if prop_type == "boolean" and not isinstance(value, bool):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a boolean"
                )
            if "enum" in prop and value not in prop["enum"]:
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be one of {prop['enum']}"
                )
        return arguments

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Validate and execute a tool call."""
        validated = self.validate_arguments(tool_name, arguments)
        tool = self._tools[tool_name]
        return tool.handler(validated)


def _sensor_data_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    udi_raw = arguments.get("udi")
    sensor_type = arguments.get("sensor_type", "all")
    
    if udi_raw is None:
        raise ToolError("udi is required. Provide a numeric UDI from AI4I 2020 dataset (e.g., 1). Do not pass incident description.")
    if isinstance(udi_raw, str) and not udi_raw.isdigit():
        raise ToolError(f"Invalid UDI: {udi_raw!r}. Must be positive integer, not text.")
    try:
        udi_int = int(udi_raw)
    except (ValueError, TypeError):
        raise ToolError(f"Invalid UDI: {udi_raw!r}. Must be positive integer.")
    if udi_int <= 0:
        raise ToolError(f"UDI must be positive: {udi_int}.")
    
    try:
        param_map = {
            "all": None,
            "air_temperature": "air_temperature",
            "process_temperature": "process_temperature",
            "rotational_speed": "rotational_speed",
            "torque": "torque",
            "tool_wear": "tool_wear",
        }
        mapped = param_map.get(str(sensor_type).lower(), str(sensor_type).lower())
        if mapped is not None and mapped not in AGENT_PARAMS:
            raise ToolError(f"Unsupported parameter: {sensor_type!r}. Supported: {list(AGENT_PARAMS.keys())}")
        result = get_sensor_readings(udi_int, parameter=mapped if mapped else None)
        return result
    except DataAdapterError as exc:
        raise ToolError(str(exc))


def _diagnostic_math_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Harmless deterministic math on caller-provided numbers.

    Computes sum, mean, min, max, and standard deviation. This tool
    requires VALID numeric input - it cannot compute statistics on empty data.
    For operational evidence, first retrieve sensor data using a data-adapter tool.
    """
    values = arguments["values"]
    if not isinstance(values, list) or not values:
        raise ToolError(
            "Cannot compute statistics on empty data. "
            "Provide non-empty values array, or retrieve sensor data first."
        )
    numeric = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ToolError("all values must be numeric")
        numeric.append(float(v))
    n = len(numeric)
    total = sum(numeric)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in numeric) / n
    std = math.sqrt(variance)
    return {
        "count": n,
        "sum": total,
        "mean": mean,
        "min": min(numeric),
        "max": max(numeric),
        "std_dev": std,
        "variance": variance,
    }


def _build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Stub: Sensor data retrieval (future implementation)
    registry.register(
        Tool(
            name="get_sensor_readings",
            description=(
                "Retrieve a historical observation from the AI4I 2020 predictive-maintenance dataset. "
                "Requires a numeric UDI (record identifier) - NOT incident description, station, or timestamp. "
                "Returns measurements with units/provenance. Supported: air_temperature, process_temperature, rotational_speed, torque, tool_wear, all."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_sensor_data_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "udi": {
                        "type": "integer",
                        "description": "Numeric UDI (record identifier) from AI4I 2020 dataset. Must be integer, not text.",
                    },
                    "sensor_type": {
                        "type": "string",
                        "enum": ["all", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"],
                        "description": "Type of sensor to query.",
                    },
                },
                "required": ["udi"],
                "additionalProperties": False,
            },
        )
    )

    # Diagnostic math tool (requires non-empty input)
    registry.register(
        Tool(
            name="diagnostic_math",
            description=(
                "Compute deterministic summary statistics (count, sum, mean, "
                "min, max, standard deviation) on a list of numeric values "
                "provided by the caller. Read-only, no side effects. "
                "Requires non-empty values array."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_diagnostic_math_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "description": "List of numeric values to summarize (must be non-empty).",
                    }
                },
                "required": ["values"],
                "additionalProperties": False,
            },
            max_input_bytes=4000,
        )
    )
    return registry


# Module-level default registry used by the app and tests.
registered_tools = _build_default_registry()
