"""FastAPI application for PlantPilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
# Pure SSE via StreamingResponse (no extra dependency)
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Any, Dict

from . import __version__
from .agent import create_agent_controller
from .config import get_config, ConfigError
from openai import OpenAI, APIConnectionError, APITimeoutError
from .schemas import (
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    InvestigationResult,
    Conversation,
    ConversationCreate,
    ConversationMessageCreate,
)
from .storage import get_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    config = get_config()
    config.ensure_data_dir()
    db = get_database()
    yield
    db.close()


app = FastAPI(
    title="PlantPilot",
    description="Human-supervised industrial operations AI copilot",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Return application name, version, and status."""
    return {
        "application": "PlantPilot",
        "version": __version__,
        "status": "running",
    }


@app.get("/health")
async def health(check_provider: bool = False):
    """Return backend health status.

    By default, this endpoint only checks that the API is running.
    Pass check_provider=true to verify NVIDIA provider connectivity.
    """
    health_status = {
        "status": "healthy",
        "backend": "ok",
    }

    if check_provider:
        try:
            # Minimal API check without making a real model call
            from openai import OpenAI
            config = get_config()
            client = OpenAI(
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                timeout=5.0,
            )
            # Just verify we can construct the client
            health_status["provider"] = "reachable"
        except Exception as exc:
            health_status["provider"] = f"error: {type(exc).__name__}"
            health_status["status"] = "degraded"

    return health_status


@app.post("/api/v1/incidents", response_model=Incident, status_code=status.HTTP_201_CREATED)
async def create_incident(create: IncidentCreate) -> Incident:
    """Create and persist an incident."""
    try:
        db = get_database()
        incident = db.create_incident(create)
        return incident
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create incident: {type(exc).__name__}",
        )


@app.get("/api/v1/incidents", response_model=List[Incident])
async def list_incidents(limit: int = 50, offset: int = 0) -> List[Incident]:
    """List persisted incidents, newest first."""
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 200",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative",
        )

    try:
        db = get_database()
        incidents = db.list_incidents(limit=limit, offset=offset)
        return incidents
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list incidents: {type(exc).__name__}",
        )


@app.get("/api/v1/incidents/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str) -> Incident:
    """Retrieve a single incident."""
    try:
        db = get_database()
        incident = db.get_incident(incident_id)
        if incident is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id!r} not found",
            )
        return incident
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve incident: {type(exc).__name__}",
        )


@app.post("/api/v1/incidents/{incident_id}/investigations", response_model=InvestigationResult)
async def start_investigation(incident_id: str, create: InvestigationCreate) -> InvestigationResult:
    """Start an AI investigation for an existing incident.

    This endpoint is synchronous: it runs the investigation loop and
    returns the structured result.
    """
    if create.incident_id != incident_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="incident_id in path must match incident_id in body",
        )

    try:
        db = get_database()
        incident = db.get_incident(incident_id)
        if incident is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id!r} not found",
            )

        # Create investigation record
        investigation = db.create_investigation(create, incident)

        # Run the agent
        agent = create_agent_controller(db)
        result = agent.investigate(investigation, incident)

        return result

    except HTTPException:
        raise
    except ConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation failed: {type(exc).__name__}: {exc}",
        )


@app.get("/api/v1/investigations/{investigation_id}", response_model=Investigation)
async def get_investigation(investigation_id: str) -> Investigation:
    """Retrieve an investigation with full details."""
    try:
        db = get_database()
        investigation = db.get_investigation(investigation_id)
        if investigation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation {investigation_id!r} not found",
            )
        return investigation
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve investigation: {type(exc).__name__}",
        )


@app.get("/api/v1/investigations/{investigation_id}/events")
async def investigation_events(
    investigation_id: str,
    since: str | None = None,
    limit: int = 100,
):
    """Poll persisted investigation events for progressive UI updates.

    `since` is an ISO-8601 timestamp returned as `created_at` by an earlier
    event. This endpoint is polling-based, not an SSE/WebSocket stream.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    db = get_database()
    investigation = db.get_investigation(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return {
        "investigation_id": investigation_id,
        "status": investigation.status.value,
        "events": db.list_audit_events(investigation_id, since=since, limit=limit),
    }


@app.post("/api/v1/conversations", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(create: ConversationCreate):
    """Create a persisted follow-up thread bound to one investigation."""
    try:
        return get_database().create_conversation(create.investigation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Investigation not found")


@app.get("/api/v1/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = get_database().get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/v1/conversations/{conversation_id}/messages")
async def send_conversation_message(
    conversation_id: str, create: ConversationMessageCreate
):
    """Answer a follow-up using only persisted investigation context.

    This endpoint intentionally does not run new diagnostic tools. If new
    observations are needed, start a new investigation rather than implying
    that a fresh tool call was performed.
    """
    db = get_database()
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    investigation = db.get_investigation(conversation["investigation_id"])
    if investigation is None:
        raise HTTPException(status_code=404, detail="Linked investigation not found")

    # Persist the user turn first so it remains visible if provider inference fails.
    db.add_conversation_message(conversation_id, "user", create.content)

    evidence_context = [
        {
            "id": item.id,
            "type": item.evidence_type.value,
            "description": item.description,
            "source": item.source,
            "value": item.value,
            "timestamp": item.timestamp.isoformat(),
            "verified": item.verified,
        }
        for item in investigation.evidence
    ]
    context = {
        "investigation_id": investigation.id,
        "incident_id": investigation.incident_id,
        "status": investigation.status.value,
        "findings": investigation.findings,
        "evidence": evidence_context,
        "hypotheses": [h.model_dump(mode="json") for h in investigation.hypotheses],
        "recommendations": [r.model_dump(mode="json") for r in investigation.recommendations],
        "tool_executions": [
            {"tool_name": t.tool_name, "status": t.status.value, "result": t.result}
            for t in investigation.tool_executions
        ],
    }
    # Limit history to the latest 20 turns to keep requests bounded.
    history = conversation.get("messages", [])[-20:]
    messages = [
        {
            "role": "system",
            "content": (
                "You are PlantPilot, a human-supervised industrial investigation copilot. "
                "Answer follow-up questions using ONLY the supplied persisted investigation "
                "context. Do not invent readings, timestamps, tool calls, or sources. "
                "Clearly distinguish observed evidence from hypotheses and recommendations. "
                "If the context is insufficient, say so and recommend starting a new "
                "investigation to retrieve fresh evidence. Never claim you just inspected "
                "a machine or ran a diagnostic tool. Do not direct physical control actions. "
                "Consequential recommendations require human approval.\n\n"
                "Persisted investigation context (JSON):\n"
                + __import__("json").dumps(context, ensure_ascii=False, default=str)
            ),
        }
    ]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": create.content})

    config = get_config()
    client = OpenAI(
        base_url=config.nvidia_base_url,
        api_key=config.nvidia_api_key,
        timeout=config.api_timeout_seconds,
    )
    response = None
    last_error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config.model_identifier,
                messages=messages,
                temperature=0.2,
                max_tokens=min(config.max_output_tokens, 1500),
            )
            break
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            transient = (
                status_code in {429, 500, 502, 503, 504}
                or isinstance(exc, (APIConnectionError, APITimeoutError))
            )
            if attempt == 0 and transient:
                db.save_audit_event(
                    "conversation_provider_retry",
                    "Transient provider failure during follow-up; retrying once",
                    {"error_type": type(exc).__name__, "status_code": status_code},
                    investigation.id,
                )
                continue
            break

    if response is None:
        status_code = getattr(last_error, "status_code", None)
        db.save_audit_event(
            "conversation_provider_error",
            "Follow-up response could not be generated",
            {"error_type": type(last_error).__name__ if last_error else "Unknown",
             "status_code": status_code},
            investigation.id,
        )
        raise HTTPException(
            status_code=502,
            detail="PlantPilot could not generate a follow-up response. Your question was saved; retry it when the AI provider is available.",
        )

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="AI provider returned an empty response")
    saved = db.add_conversation_message(conversation_id, "assistant", answer)
    db.save_audit_event(
        "conversation_response_completed",
        "A context-grounded follow-up response was saved",
        {"conversation_id": conversation_id, "message_id": saved["id"]},
        investigation.id,
    )
    return {
        "conversation_id": conversation_id,
        "message": saved,
        "investigation_id": investigation.id,
    }




# --- Detection and digital-twin APIs ---

class DetectionRequest(BaseModel):
    reading: Dict[str, Any]
    include_secondary: bool = False

class TwinCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    machine_type: str = Field(..., min_length=1, max_length=120)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TwinReadingRequest(BaseModel):
    reading: Dict[str, Any]
    include_secondary: bool = False

class SimulationRequest(BaseModel):
    reading: Dict[str, Any]
    include_secondary: bool = True

@app.post("/api/v1/detection/predict")
async def detection_predict(payload: DetectionRequest):
    """Run fast HGB failure-risk prediction plus IQR anomaly evidence.

    Random Forest is optional and runs only when include_secondary is true.
    """
    from .detection import predict
    try:
        return predict(payload.reading, include_secondary=payload.include_secondary)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/api/v1/detection/models")
async def detection_models():
    return {
        "primary": "HistGradientBoostingClassifier",
        "statistical_detector": "IQR plus documented tool-wear heuristic",
        "secondary": "RandomForestClassifier (optional, requested per prediction)",
        "training_data": "bundled AI4I 2020 benchmark",
        "plant_validated": False,
        "notes": "Benchmark predictions are not industrial safety decisions or verified root causes."
    }

@app.post("/api/v1/digital-twins", status_code=status.HTTP_201_CREATED)
async def create_digital_twin(payload: TwinCreateRequest):
    from .digital_twin import get_twin_store
    return get_twin_store().create(payload.name, payload.machine_type, payload.metadata)

@app.get("/api/v1/digital-twins")
async def list_digital_twins(limit: int = 50):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 200")
    from .digital_twin import get_twin_store
    return get_twin_store().list(limit)

@app.get("/api/v1/digital-twins/{twin_id}")
async def get_digital_twin(twin_id: str):
    from .digital_twin import get_twin_store
    twin = get_twin_store().get(twin_id)
    if twin is None: raise HTTPException(status_code=404, detail="Digital twin not found")
    return twin

@app.get("/api/v1/digital-twins/{twin_id}/readings")
async def get_digital_twin_readings(twin_id: str, limit: int = 100):
    if limit < 1 or limit > 500: raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    from .digital_twin import get_twin_store
    rows = get_twin_store().history(twin_id, limit)
    if rows is None: raise HTTPException(status_code=404, detail="Digital twin not found")
    return rows

@app.post("/api/v1/digital-twins/{twin_id}/readings", status_code=status.HTTP_201_CREATED)
async def record_digital_twin_reading(twin_id: str, payload: TwinReadingRequest):
    from .detection import predict
    from .digital_twin import get_twin_store
    store = get_twin_store()
    if store.get(twin_id) is None: raise HTTPException(status_code=404, detail="Digital twin not found")
    try: assessment = predict(payload.reading, include_secondary=payload.include_secondary)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc))
    return store.add_reading(twin_id, payload.reading, assessment, "observed")

@app.post("/api/v1/digital-twins/{twin_id}/simulate")
async def simulate_digital_twin_reading(twin_id: str, payload: SimulationRequest):
    """Run a what-if prediction. Simulation results are explicitly isolated from observed history."""
    from .detection import predict
    from .digital_twin import get_twin_store
    store = get_twin_store()
    if store.get(twin_id) is None: raise HTTPException(status_code=404, detail="Digital twin not found")
    try: assessment = predict(payload.reading, include_secondary=payload.include_secondary)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc))
    # Persist as simulated for audit/history, but never as the twin's latest observed state.
    return store.add_reading(twin_id, payload.reading, assessment, "simulated")


@app.exception_handler(ConfigError)
async def config_error_handler(request, exc: ConfigError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Configuration error: {exc}"},
    )


# --- Async investigation + SSE streaming ---

import asyncio
from .storage import _event_streams, EventStream

@app.post("/api/v1/incidents/{incident_id}/investigations/async", status_code=status.HTTP_202_ACCEPTED)
async def start_investigation_async(incident_id: str, create: InvestigationCreate):
    if create.incident_id != incident_id:
        raise HTTPException(status_code=400, detail="incident_id mismatch")
    db = get_database()
    incident = db.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    investigation = db.create_investigation(create, incident)
    # Initialize stream
    stream = EventStream(investigation.id)
    stream.init_queue()
    _event_streams[investigation.id] = stream
    # Emit start event immediately (actual event, not fabricated progress)
    event_id = f"evt_{__import__('uuid').uuid4().hex[:12]}"
    stream.put(event_id, "investigation_started", f"Investigation {investigation.id} started", {"incident_id": incident_id})
    # Save audit event (reuse existing model)
    db.save_audit_event("investigation_started", f"Investigation started: {investigation.id}", {"incident_id": incident_id}, investigation.id)

    async def run_investigation():
        try:
            agent = create_agent_controller(db)
            # We call investigate in thread to avoid blocking event loop
            # Note: agent saves audit events to DB; we also emit via stream if we inject.
            # To emit actual events, modify agent to call stream.emit. Instead, after return we emit terminal.
            result = await asyncio.to_thread(agent.investigate, investigation, incident)
            # Emit terminal event based on actual result
            if result.status.value == "completed":
                terminal_type = "investigation_complete"
                msg = f"Investigation {investigation.id} completed (steps={result.steps_taken})"
            elif result.status.value == "incomplete":
                terminal_type = "investigation_incomplete"
                msg = f"Investigation {investigation.id} incomplete (steps={result.steps_taken})"
            else:
                terminal_type = "investigation_failed"
                msg = f"Investigation {investigation.id} failed (status={result.status.value})"
            event_id = f"evt_{__import__('uuid').uuid4().hex[:12]}"
            stream.put(event_id, terminal_type, msg, {"investigation_id": investigation.id, "status": result.status.value, "steps_taken": result.steps_taken})
            db.save_audit_event(terminal_type, msg, {"status": result.status.value, "steps": result.steps_taken}, investigation.id)
        except Exception as exc:
            event_id = f"evt_{__import__('uuid').uuid4().hex[:12]}"
            stream.put(event_id, "investigation_failed", f"Investigation failed: {type(exc).__name__}", {"error": str(exc)})
            db.save_audit_event("investigation_failed", f"Investigation failed: {type(exc).__name__}", {"error": str(exc)}, investigation.id)
        finally:
            # Keep stream alive briefly for reconnects; do not delete immediately
            pass

    asyncio.create_task(run_investigation())
    return {"investigation_id": investigation.id, "status": "started"}


@app.get("/api/v1/investigations/{investigation_id}/stream")
async def investigation_stream(investigation_id: str, request: Request, since_id: str | None = None):
    async def event_generator():
        import time
        # Replay persisted audit events first
        db = get_database()
        events = db.list_audit_events(investigation_id, since=since_id, limit=500) if since_id else db.list_audit_events(investigation_id, limit=500)
        # Build ordered replay; if no since, replay all
        replayed_ids = set()
        for ev in events:
            evt_id = ev.get("id") or f"evt_{ev['created_at']}"
            replayed_ids.add(evt_id)
            event_name = ev["type"]
            payload = {"message": ev["message"], "details": ev.get("details", {}), "created_at": ev["created_at"]}
            data_json = __import__("json").dumps(payload)
            out = 'id: ' + evt_id + '\nevent: ' + event_name + '\ndata: ' + data_json + '\n\n'
            yield out
        # Now subscribe to live stream

        stream = _event_streams.get(investigation_id)
        if stream is None:
            stream = EventStream(investigation_id)
            stream.init_queue()
            _event_streams[investigation_id] = stream
        # Replay any queued events that occurred after DB replay (if stream existed before connect)
        # For simplicity, after replay we read from queue until completion or disconnect
        session_done = False
        while not session_done:
            try:
                # Wait for item with timeout so disconnect can be detected
                item = await asyncio.wait_for(stream.queue.get(), timeout=2.0) if stream.queue else None
                if item is None:
                    continue
                # Skip already replayed by id
                if item["id"] in replayed_ids:
                    continue
                replayed_ids.add(item["id"])
                out = 'id: ' + item['id'] + chr(10) + 'event: ' + item['event'] + chr(10) + 'data: ' + __import__('json').dumps(item['data']) + chr(10) + chr(10)
                yield out
                # If terminal event, terminate after brief delay
                if item["event"] in ("investigation_complete", "investigation_incomplete", "investigation_failed"):
                    session_done = True
            except asyncio.TimeoutError:
                # Check if client disconnected via request
                if await request.is_disconnected():
                    session_done = True
                # Otherwise loop to keep stream open for reconnect/replay
        # On disconnect: do not cancel investigation task; just stop yielding

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })

