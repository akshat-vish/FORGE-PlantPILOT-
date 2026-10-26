"""Pydantic data models for PlantPilot."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class EvidenceType(str, Enum):
    OBSERVED = "observed"           # verified measurement / reading
    DERIVED = "derived"             # calculation from observed values
    HYPOTHESIS = "hypothesis"       # model-generated claim (unverified)
    SIMULATED = "simulated"         # hypothetical / projected outcome


class ApprovalStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class EvidenceItem(BaseModel):
    """A single piece of evidence with explicit provenance."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Stable evidence identifier")
    evidence_type: EvidenceType = Field(
        EvidenceType.OBSERVED, description="Provenance category"
    )
    description: str = Field(
        ..., min_length=1, max_length=2000, description="What was observed or claimed"
    )
    source: str = Field(
        ..., min_length=1, max_length=500, description="Tool / sensor / source"
    )
    value: Optional[Any] = Field(None, description="Numeric or structured value")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of observation or claim",
    )
    verified: bool = Field(
        False, description="True only when independently verified, not model-only"
    )


class Hypothesis(BaseModel):
    """A model-generated explanation — explicitly NOT verified evidence."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    explanation: str = Field(..., min_length=1, max_length=4000)
    confidence: Optional[str] = Field(
        None, description="e.g. low / medium / high — model's own assessment"
    )
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Recommendation(BaseModel):
    """A proposal for human review — never an executed action."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=4000)
    requires_approval: bool = Field(
        True, description="Always True for consequential recommendations"
    )
    approval_status: ApprovalStatus = Field(
        ApprovalStatus.PENDING_APPROVAL
    )
    priority: str = Field("medium", description="low / medium / high")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    executed_at: Optional[datetime] = Field(
        None, description="Only set when approved and executed (not here)"
    )


class ToolExecution(BaseModel):
    """Audit record of a registered tool execution."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = Field(
        None, max_length=4000, description="String representation, size-limited"
    )
    status: Status = Field(Status.COMPLETED)
    error_message: Optional[str] = Field(None, max_length=1000)
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    input_size_bytes: int = Field(0, ge=0)
    output_size_bytes: int = Field(0, ge=0)


class IncidentCreate(BaseModel):
    """Payload to create a new incident."""
    model_config = ConfigDict(populate_by_name=True)

    description: str = Field(..., min_length=1, max_length=5000)
    severity: str = Field("medium", description="low / medium / high / critical")
    category: Optional[str] = Field("operational", max_length=200)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Incident(BaseModel):
    """Persisted incident."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    description: str
    severity: str
    category: Optional[str] = None
    status: Status = Field(Status.PENDING)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvestigationCreate(BaseModel):
    """Payload to start an investigation for an incident."""
    model_config = ConfigDict(populate_by_name=True)

    incident_id: str = Field(..., min_length=1)
    notes: Optional[str] = Field(None, max_length=2000)


class Investigation(BaseModel):
    """A completed or in-progress investigation."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    incident_id: str = Field(...)
    status: Status = Field(Status.IN_PROGRESS)
    findings: List[str] = Field(default_factory=list,
                                 description="Structured findings (text or JSON)")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    tool_executions: List[ToolExecution] = Field(default_factory=list,
                                                 description="Audit trail of all tool calls")
    recommendations: List[Recommendation] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list,
                               description="Explicit errors or failures")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    notes: Optional[str] = Field(None, max_length=2000)


class InvestigationResult(BaseModel):
    """Structured result returned by the agent controller."""
    model_config = ConfigDict(populate_by_name=True)

    investigation_id: str = Field(...)
    status: Status = Field(...)
    findings: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_ids: List[str] = Field(default_factory=list)
    recommendation_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    steps_taken: int = Field(0, ge=0)
    completed: bool = Field(False)




class ConversationCreate(BaseModel):
    """Create a conversational thread attached to an existing investigation."""
    investigation_id: str = Field(..., min_length=1)


class ConversationMessageCreate(BaseModel):
    """A user follow-up question for an investigation conversation."""
    content: str = Field(..., min_length=1, max_length=4000)


class ConversationMessage(BaseModel):
    """Persisted conversation turn."""
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime


class Conversation(BaseModel):
    """Conversation thread bound to one investigation."""
    id: str
    investigation_id: str
    created_at: datetime
    messages: List[ConversationMessage] = Field(default_factory=list)
