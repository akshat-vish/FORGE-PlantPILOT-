"""Agent controller — orchestrates bounded investigation loops.

The agent loads an incident, constructs a prompt distinguishing facts from
hypotheses, invokes the configured model with registered tools, validates
and executes tool calls, and produces a structured investigation report.

All investigation state and audit events are persisted in SQLite.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from openai.types.chat import ChatCompletion

from .config import get_config
from .safety import default_safety_policy, SafetyPolicy
from .schemas import (
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Incident,
    Investigation,
    Recommendation,
    Status,
    ToolExecution,
    InvestigationResult,
    ApprovalStatus,
)
from .storage import Database
from .tools import ToolRegistry, registered_tools, ToolError


class AgentError(RuntimeError):
    """Raised when the agent encounters an unrecoverable error."""


class AgentController:
    """Orchestrates AI investigations with safety controls."""

    def __init__(
        self,
        db: Database,
        tool_registry: ToolRegistry,
        safety_policy: SafetyPolicy,
    ) -> None:
        self.db = db
        self.tool_registry = tool_registry
        self.safety_policy = safety_policy
        self.config = get_config()

        self.client = OpenAI(
            base_url=self.config.nvidia_base_url,
            api_key=self.config.nvidia_api_key,
            timeout=self.config.api_timeout_seconds,
        )

    def investigate(
        self, investigation: Investigation, incident: Incident
    ) -> InvestigationResult:
        """Run a bounded investigation loop for the given incident.

        Enforces evidence-first workflow: the model must attempt diagnostic
        tools before concluding. If no tools are executed, the investigation
        is marked INCOMPLETE with an appropriate error message.
        """
        messages = self._build_initial_messages(incident)
        tools_openai = self.tool_registry.to_openai_tools()

        findings: List[str] = []
        evidence_ids: List[str] = []
        hypothesis_ids: List[str] = []
        recommendation_ids: List[str] = []
        errors: List[str] = []
        steps_taken = 0
        tool_call_count = 0
        tools_executed = False  # Track if any tool was successfully executed
        provider_failure = False

        for step in range(self.config.max_steps):
            steps_taken = step + 1
            self.db.save_audit_event(
                "investigation_step_started",
                f"Investigation step {steps_taken} started",
                {"step": steps_taken},
                investigation.id,
            )

            # Force tool call on first step to ensure evidence gathering
            tool_choice = "required" if step == 0 else "auto"

            # Retry a transient provider failure once at this step. A retry must
            # not consume an investigation step or silently skip the model call.
            response = None
            provider_error = None
            for attempt in range(2):
                try:
                    self.db.save_audit_event(
                        "model_request_started",
                        f"Requesting model response for investigation step {steps_taken}",
                        {"step": steps_taken, "attempt": attempt + 1},
                        investigation.id,
                    )
                    response = self.client.chat.completions.create(
                        model=self.config.model_identifier,
                        messages=messages,  # type: ignore
                        tools=tools_openai,
                        tool_choice=tool_choice,
                        temperature=0.2,
                        max_tokens=self.config.max_output_tokens,
                    )
                    provider_error = None
                    break
                except Exception as exc:
                    provider_error = exc
                    status_code = getattr(exc, "status_code", None)
                    transient = (
                        status_code in {429, 500, 502, 503, 504}
                        or isinstance(exc, (APIConnectionError, APITimeoutError))
                    )
                    if attempt == 0 and transient:
                        self.db.save_audit_event(
                            "provider_retry",
                            f"Transient provider error at step {steps_taken}; retrying once",
                            {"step": steps_taken, "error_type": type(exc).__name__,
                             "status_code": status_code},
                            investigation.id,
                        )
                        continue
                    break

            if response is None:
                exc = provider_error or RuntimeError("Provider returned no response")
                # Avoid persisting provider response bodies/API details that may
                # contain sensitive diagnostic data. Keep type/status for debugging.
                status_code = getattr(exc, "status_code", None)
                error_msg = (
                    f"AI provider request failed after bounded retry: "
                    f"{type(exc).__name__}"
                    + (f" (HTTP {status_code})" if status_code is not None else "")
                )
                errors.append(error_msg)
                provider_failure = True
                self.db.save_audit_event(
                    "api_error", error_msg,
                    {"step": steps_taken, "error_type": type(exc).__name__,
                     "status_code": status_code},
                    investigation.id,
                )
                break

            assistant_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Preserve the assistant's response.
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": (
                        [tc.model_dump() for tc in assistant_message.tool_calls]
                        if assistant_message.tool_calls
                        else None
                    ),
                }
            )

            tool_calls = assistant_message.tool_calls or []

            # If no tool calls, the model has finished.
            if not tool_calls:
                if assistant_message.content:
                    findings.append(assistant_message.content[:self.config.max_message_chars])

                # Evidence-first: if no tools were executed, mark as INCOMPLETE
                if not tools_executed:
                    error_msg = "No diagnostic tools were executed. Investigation cannot complete without evidence."
                    errors.append(error_msg)
                    self.db.save_audit_event(
                        "investigation_incomplete_no_evidence",
                        error_msg,
                        {"finish_reason": finish_reason, "steps": steps_taken},
                        investigation.id,
                    )
                    self.db.update_investigation(
                        investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        completed_at=datetime.now(timezone.utc),
                    )
                    return InvestigationResult(
                        investigation_id=investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        evidence_ids=evidence_ids,
                        hypothesis_ids=hypothesis_ids,
                        recommendation_ids=recommendation_ids,
                        errors=errors,
                        steps_taken=steps_taken,
                        completed=False,
                    )

                # Tools were executed - generate structured findings
                # Add a hypothesis based on evidence (model-generated claim)
                if findings and not any(findings):
                    pass  # findings already has content

                # Create a recommendation requiring human approval
                recommendation = Recommendation(
                    id=f"rec_{uuid.uuid4().hex[:12]}",
                    title="Review diagnostic results",
                    description="Investigate diagnostic results and determine root cause. Human approval required before any action.",
                    requires_approval=True,
                    approval_status=ApprovalStatus.PENDING_APPROVAL,
                    priority="high",
                )
                self.db.save_recommendation(investigation.id, recommendation)
                recommendation_ids.append(recommendation.id)

                # Add a model-generated hypothesis (explicitly unverified)
                if evidence_ids:
                    hypothesis = Hypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:12]}",
                        title="Diagnostic evidence requires interpretation",
                        explanation="Evidence was obtained through diagnostic tools. Root cause determination requires further human review.",
                        confidence="low",
                        supporting_evidence_ids=evidence_ids,
                    )
                    self.db.save_hypothesis(investigation.id, hypothesis)
                    hypothesis_ids.append(hypothesis.id)

                # Normal completion
                self.db.save_audit_event(
                    "investigation_complete",
                    f"Investigation finished at step {step + 1}",
                    {"finish_reason": finish_reason},
                    investigation.id,
                )
                self.db.update_investigation(
                    investigation.id,
                    status=Status.COMPLETED,
                    findings=findings,
                    completed_at=datetime.now(timezone.utc),
                )
                return InvestigationResult(
                    investigation_id=investigation.id,
                    status=Status.COMPLETED,
                    findings=findings,
                    evidence_ids=evidence_ids,
                    hypothesis_ids=hypothesis_ids,
                    recommendation_ids=recommendation_ids,
                    errors=errors,
                    steps_taken=steps_taken,
                    completed=True,
                )

            # Process tool calls.
            for tool_call in tool_calls:
                tool_call_count += 1
                if tool_call_count > self.config.max_tool_calls:
                    error_msg = f"Maximum tool calls ({self.config.max_tool_calls}) exceeded"
                    errors.append(error_msg)
                    self.db.save_audit_event(
                        "max_tool_calls_exceeded", error_msg, {}, investigation.id
                    )
                    self.db.update_investigation(
                        investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        completed_at=datetime.now(timezone.utc),
                    )
                    return InvestigationResult(
                        investigation_id=investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        evidence_ids=evidence_ids,
                        hypothesis_ids=hypothesis_ids,
                        recommendation_ids=recommendation_ids,
                        errors=errors,
                        steps_taken=steps_taken,
                        completed=False,
                    )

                tool_name = tool_call.function.name
                exec_id = f"exec_{uuid.uuid4().hex[:12]}"

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    result_str = json.dumps({"error": "Invalid tool arguments"})
                    errors.append(f"Tool {tool_name}: invalid arguments")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        {},
                        result_str,
                        Status.FAILED,
                        f"Invalid JSON: {exc}",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )
                    continue

                # Safety check.
                if not self.safety_policy.check_tool_call(tool_name, arguments):
                    result_str = json.dumps(
                        {"error": f"Tool {tool_name!r} not permitted"}
                    )
                    errors.append(f"Tool {tool_name}: safety check failed")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        "Safety policy rejected",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )
                    continue

                # Execute the tool.
                self.db.save_audit_event(
                    "tool_execution_started",
                    f"Executing registered tool: {tool_name}",
                    {"tool_name": tool_name, "arguments": arguments},
                    investigation.id,
                )
                try:
                    result = self.tool_registry.execute(tool_name, arguments)
                    result_str = self._serialize_tool_result(result)
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.COMPLETED,
                        None,
                    )
                    # Record as observed evidence.
                    evidence = EvidenceItem(
                        id=f"ev_{uuid.uuid4().hex[:12]}",
                        evidence_type=EvidenceType.OBSERVED,
                        description=f"Tool {tool_name} result",
                        source=tool_name,
                        value=result,
                        verified=True,
                    )
                    self.db.save_evidence(investigation.id, evidence)
                    evidence_ids.append(evidence.id)
                    tools_executed = True
                    self.db.save_audit_event(
                        "evidence_persisted",
                        f"Evidence persisted from tool: {tool_name}",
                        {"tool_name": tool_name, "evidence_id": evidence.id},
                        investigation.id,
                    )

                except ToolError as exc:
                    result_str = json.dumps({"error": str(exc)})
                    errors.append(f"Tool {tool_name}: {exc}")
                    # Do NOT set tools_executed = True for failed executions
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        str(exc),
                    )
                except Exception as exc:
                    result_str = json.dumps(
                        {"error": f"Tool execution failed: {type(exc).__name__}"}
                    )
                    errors.append(f"Tool {tool_name}: unexpected error")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        f"{type(exc).__name__}: {exc}",
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

        if provider_failure:
            # Terminal failure after bounded retry exhausted.
            self.db.save_audit_event(
                "investigation_failed",
                f"Investigation failed: {errors[-1]}" if errors else "Provider failure after retries",
                {"step": steps_taken, "error_category": "provider_failure", "status_code": getattr(exc, "status_code", None) if 'exc' in dir() else None},
                investigation.id,
            )
            self.db.update_investigation(
                investigation.id,
                status=Status.FAILED,
                findings=findings,
                completed_at=datetime.now(timezone.utc),
            )
            return InvestigationResult(
                investigation_id=investigation.id,
                status=Status.FAILED,
                findings=findings,
                evidence_ids=evidence_ids,
                hypothesis_ids=hypothesis_ids,
                recommendation_ids=recommendation_ids,
                errors=errors,
                steps_taken=steps_taken,
                completed=False,
            )

        # Step limit reached without finishing.
        error_msg = f"Investigation did not complete within {self.config.max_steps} steps"
        errors.append(error_msg)
        self.db.save_audit_event(
            "max_steps_exceeded", error_msg, {}, investigation.id
        )
        self.db.update_investigation(
            investigation.id,
            status=Status.INCOMPLETE,
            findings=findings,
            completed_at=datetime.now(timezone.utc),
        )
        return InvestigationResult(
            investigation_id=investigation.id,
            status=Status.INCOMPLETE,
            findings=findings,
            evidence_ids=evidence_ids,
            hypothesis_ids=hypothesis_ids,
            recommendation_ids=recommendation_ids,
            errors=errors,
            steps_taken=steps_taken,
            completed=False,
        )

    def _build_initial_messages(self, incident: Incident) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are PlantPilot, a human-supervised industrial operations AI copilot. "
            "Investigate operational incidents using available tools. "
            "Never invent sensor readings or historical data. "
            "Distinguish verified observations from model-generated hypotheses. "
            "Explain uncertainty and recommend the next useful check. "
            "Do not claim a root cause is proven without sufficient evidence. "
            "Never initiate physical actions. All consequential recommendations "
            "require human approval."
        )
        user_prompt = (
            f"Incident: {incident.description}\n\n"
            f"Severity: {incident.severity}\n"
            f"Category: {incident.category or 'operational'}\n\n"
            "Investigate this incident. The dataset uses numeric UDI identifiers (e.g., 1, 2, 3). Retrieve the actual record by passing the numeric UDI to get_sensor_readings (e.g., station_id: 1). Then use diagnostic_math on the retrieved measurements. Do NOT pass the incident description text as a UDI. Always use a numeric identifier from AI4I 2020."
            "Identify plausible explanations and state what additional evidence is needed. "
            "Do not assume that an abnormal reading proves the root cause."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize tool result to a size-limited JSON string."""
        try:
            serialized = json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(result)

        if len(serialized) > self.config.tool_result_max_chars:
            serialized = serialized[: self.config.tool_result_max_chars] + "...[truncated]"

        if len(serialized.encode("utf-8")) > self.config.tool_result_max_bytes:
            # Truncate by bytes.
            serialized_bytes = serialized.encode("utf-8")[
                : self.config.tool_result_max_bytes
            ]
            serialized = serialized_bytes.decode("utf-8", errors="ignore") + "...[truncated]"

        return serialized

    def _record_tool_execution(
        self,
        investigation_id: str,
        exec_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        status: Status,
        error_message: Optional[str],
    ) -> None:
        """Persist a tool execution audit record."""
        execution = ToolExecution(
            id=exec_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            error_message=error_message,
            executed_at=datetime.now(timezone.utc),
            input_size_bytes=len(json.dumps(arguments).encode("utf-8")),
            output_size_bytes=len(result.encode("utf-8")),
        )
        self.db.save_tool_execution(investigation_id, execution)


def create_agent_controller(db: Optional[Database] = None) -> AgentController:
    """Factory function to create an agent controller with defaults."""
    from .storage import get_database

    if db is None:
        db = get_database()
    return AgentController(
        db=db,
        tool_registry=registered_tools,
        safety_policy=default_safety_policy(),
    )
