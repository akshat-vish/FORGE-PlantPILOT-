# PlantPilot Frontend Integration Contract

This document describes the backend API contract for a separate frontend client. It does not imply that a frontend application is included in this repository.

## Local startup

From the backend project root (the directory containing `pyproject.toml`):

```bash
uv sync
uv run uvicorn plantpilot.main:app --app-dir app --reload --port 8000
```

Base URL: `http://127.0.0.1:8000`  
Interactive API docs: `http://127.0.0.1:8000/docs`

Set `NVIDIA_API_KEY` in the backend's local `.env` when live LLM investigation is required. Never expose the key to browser code.

## Incidents and investigations

- `POST /api/v1/incidents` — create an incident (`description`, optional `severity`, `category`, `metadata`).
- `GET /api/v1/incidents?limit=50&offset=0` — list incidents.
- `GET /api/v1/incidents/{incident_id}` — retrieve an incident.
- `POST /api/v1/incidents/{incident_id}/investigations` — synchronous investigation; body includes matching `incident_id` and optional `notes`.
- `POST /api/v1/incidents/{incident_id}/investigations/async` — asynchronous investigation; returns `202` and `{ "investigation_id": "...", "status": "started" }`.
- `GET /api/v1/investigations/{investigation_id}` — retrieve investigation details.
- `GET /api/v1/investigations/{investigation_id}/events` — retrieve persisted events.

Check the running application's OpenAPI page (`/docs`) if a route or schema differs in the checked-out version.

## Detection

### `GET /api/v1/detection/models`

Returns information about the configured primary detector, statistical checks, optional secondary model, and benchmark limitations.

### `POST /api/v1/detection/predict`

Example request:

```json
{
  "reading": {
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1550,
    "torque_nm": 40.2,
    "tool_wear_min": 120
  },
  "include_secondary": false
}
```

All five measurements are required. The API also accepts the original AI4I CSV column names. `include_secondary` defaults to `false`; set it to `true` to request a Random Forest second assessment.

The response includes `failure_probability`, `threshold`, `failure_risk`, `predicted_failure`, `anomaly_evidence`, model provenance, and—when requested—`secondary_model` and `model_agreement`. Invalid inputs return `422`. Model/data unavailability may return `503`.

Models are trained from the AI4I 2020 benchmark with a reproducible split. This is not validated industrial equipment data. Predictions are estimates, not verified root causes or safety decisions.

## Digital twins and readings

- `POST /api/v1/digital-twins` — create a twin, for example `{ "name": "CNC-042", "machine_type": "CNC", "metadata": {} }`.
- `GET /api/v1/digital-twins?limit=50` — list twins.
- `GET /api/v1/digital-twins/{twin_id}` — retrieve twin metadata and observed-state summary.
- `GET /api/v1/digital-twins/{twin_id}/readings?limit=100` — retrieve reading history, with observed and simulated records labelled.
- `POST /api/v1/digital-twins/{twin_id}/readings` — assess and persist an observed reading.
- `POST /api/v1/digital-twins/{twin_id}/simulate` — run a what-if prediction and persist it as simulated.

Simulation records must remain labelled `simulated` and must not update the twin's latest observed state. These endpoints do not imply live telemetry ingestion.

## SSE investigation streaming

Connect to `GET /api/v1/investigations/{investigation_id}/stream` after starting an asynchronous investigation. The response uses `text/event-stream`. Persisted audit events are replayed before live events. The optional `since_id` parameter filters replay according to the backend implementation.

Terminal event types:
- `investigation_complete`
- `investigation_incomplete`
- `investigation_failed`

A client should handle each as a terminal outcome and stop waiting for additional events for that investigation.

### Successful completion

A successful investigation ends with status `completed` and an `investigation_complete` event. Only render findings/evidence/recommendations actually returned and persisted by the backend.

### Incomplete investigation

An investigation may end `incomplete` if the evidence-first completion requirements were not met or a configured workflow limit was reached. Do not display it as successful completion.

### Failed investigation

When an unrecoverable provider failure occurs after retry handling:
- Final status is `failed`.
- The original `api_error` audit event is preserved.
- An `investigation_failed` terminal event is persisted.
- The stream delivers the terminal event and then closes.
- The client must stop the stream and must not treat the investigation as completed.
- Do not fabricate findings, evidence, or recommendations.
- Surface safe error metadata only; never expose provider response bodies, credentials, or API keys.

Illustrative event shape:

```text
id: <event_id>
event: investigation_failed
data: {"message":"Investigation failed","details":{"investigation_id":"<id>","status":"failed","error_category":"provider_failure","status_code":500}}
```

Treat the payload above as illustrative; use the actual API response schema from the running version.

A disconnected client does not necessarily cancel the background investigation. SQLite audit replay is persisted; live fan-out is in-memory in this prototype, so multi-process deployments require additional coordination.

## Conversations

- `POST /api/v1/conversations` — create a thread linked to an investigation.
- `GET /api/v1/conversations/{conversation_id}` — retrieve thread and messages.
- `POST /api/v1/conversations/{conversation_id}/messages` — submit a follow-up message.

Follow-up answers use persisted investigation context. They must not be presented as fresh physical measurements or proof of a root cause. Check `/docs` for the exact schemas in the current checkout.

## UI guidance and errors

- Display failure probability as an estimate, not certainty.
- Display statistical anomaly flags separately from failure-risk prediction.
- Show secondary-model agreement only when the secondary model was requested and returned.
- Label simulated readings distinctly from observed readings.
- Handle `404` for unknown resources, `422` for invalid input, and `503` when model/data dependencies are unavailable, as applicable.
- Recommendations are proposals requiring human review.
