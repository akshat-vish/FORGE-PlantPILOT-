"""Safety policy enforcement — independent of the LLM.

The safety layer is a Python gatekeeper. It inspects tool calls and
recommendations produced by the model and rejects anything that is not
explicitly permitted. It never asks the model whether an action is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .schemas import (
    ApprovalStatus,
    Recommendation,
    Status,
)


class RiskLevel(str, Enum):
    NONE = "none"          # read-only / harmless deterministic
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolRisk:
    """Risk metadata for a registered tool."""
    name: str
    risk_level: RiskLevel
    requires_approval: bool
    description: str = ""


class SafetyPolicy:
    """Enforces tool and recommendation safety rules."""

    # Tools that are always permitted (read-only / deterministic).
    PERMITTED_TOOL_RISKS = {RiskLevel.NONE, RiskLevel.LOW}

    def __init__(self) -> None:
        self._tool_risks: Dict[str, ToolRisk] = {}
        self._audit_events: List[Dict[str, Any]] = []

    def register_tool_risk(self, risk: ToolRisk) -> None:
        self._tool_risks[risk.name] = risk

    def get_tool_risk(self, tool_name: str) -> Optional[ToolRisk]:
        return self._tool_risks.get(tool_name)

    def is_tool_permitted(self, tool_name: str) -> bool:
        """Return True if the tool is registered and at an acceptable risk."""
        risk = self._tool_risks.get(tool_name)
        if risk is None:
            return False
        return risk.risk_level in self.PERMITTED_TOOL_RISKS

    def check_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate a tool call against policy. Returns True if allowed."""
        risk = self._tool_risks.get(tool_name)
        if risk is None:
            self._audit(
                "tool_rejected_unknown",
                f"Tool {tool_name!r} is not registered",
                {"tool_name": tool_name},
            )
            return False
        if risk.risk_level not in self.PERMITTED_TOOL_RISKS:
            self._audit(
                "tool_rejected_risk",
                f"Tool {tool_name!r} risk {risk.risk_level.value} not permitted",
                {"tool_name": tool_name, "risk": risk.risk_level.value},
            )
            return False
        self._audit(
            "tool_approved",
            f"Tool {tool_name!r} approved",
            {"tool_name": tool_name, "arguments": arguments},
        )
        return True

    def check_recommendation(self, recommendation: Recommendation) -> bool:
        """Validate a recommendation. Always requires approval."""
        if not recommendation.requires_approval:
            self._audit(
                "recommendation_rejected",
                "Recommendation must require approval",
                {"title": recommendation.title},
            )
            return False
        if recommendation.approval_status != ApprovalStatus.PENDING_APPROVAL:
            self._audit(
                "recommendation_rejected",
                "Recommendation must be pending approval",
                {"title": recommendation.title},
            )
            return False
        self._audit(
            "recommendation_pending_approval",
            f"Recommendation {recommendation.title!r} queued for human review",
            {"recommendation_id": recommendation.id},
        )
        return True

    def audit_events(self) -> List[Dict[str, Any]]:
        return list(self._audit_events)

    def _audit(self, event_type: str, message: str, details: Dict[str, Any]) -> None:
        self._audit_events.append(
            {
                "event_type": event_type,
                "message": message,
                "details": details,
            }
        )


def check_safety(
    policy: SafetyPolicy,
    tool_name: str,
    arguments: Dict[str, Any],
) -> bool:
    """Convenience wrapper around :meth:`SafetyPolicy.check_tool_call`."""
    return policy.check_tool_call(tool_name, arguments)


def default_safety_policy() -> SafetyPolicy:
    """Build a safety policy with the built-in diagnostic tool registered."""
    policy = SafetyPolicy()
    policy.register_tool_risk(
        ToolRisk(
            name="get_sensor_readings",
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            description="Retrieve historical CSV dataset observations - read-only",
        )
    )
    policy.register_tool_risk(
        ToolRisk(
            name="diagnostic_math",
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            description="Harmless deterministic math on caller-provided numbers",
        )
    )
    return policy