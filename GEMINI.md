# PlantPilot — Codebase Reference

> Persistent reference for AI-assisted development. **The source code and tests are authoritative.** If this document conflicts with the current checkout, inspect the implementation and update this document instead of assuming the prose is correct.

## 1. Project overview

PlantPilot is an evidence-first, human-supervised industrial operations AI copilot prototype. It combines:
- A FastAPI backend and SQLite persistence.
- A bounded investigation agent using approved tools and an NVIDIA-compatible OpenAI API client.
- Historical sensor-like measurements from the AI4I 2020 predictive-maintenance dataset.
- A primary HistGradientBoosting (HGB) failure-risk predictor and optional Random Forest assessment.
- IQR-based anomaly evidence and documented experimental heuristics.
- Persisted software digital twins with observed and simulated reading histories.
- Isolated what-if predictions.
- Persisted evidence, tool executions, findings/hypotheses, recommendations, and audit events.

It is not connected to physical sensors by default. The benchmark models are not validated on real factory equipment. Predictions are not safety decisions or verified root causes.

## 2. Runtime and configuration

- **API:** FastAPI
- **Validation:** Pydantic
- **Persistence:** SQLite
- **LLM client:** OpenAI-compatible SDK configured for NVIDIA inference
- **Provider base URL:** `https://integrate.api.nvidia.com/v1`
- **Default model:** `nvidia/nemotron-3.5-super-vl-120b-a12b` (configurable)
- **Python/package management:** use the versions and scripts in `pyproject.toml` and `uv.lock`.

Configuration includes `NVIDIA_API_KEY`, provider base URL/model settings, database path, bounded agent limits, and request/tool-output limits. Consult `app/plantpilot/config.py` and `.env.example` for the actual names and defaults in the current checkout.

Never print, commit, or expose `.env` contents. Do not pass API keys to frontend/browser code.

## 3. Project layout

The current upgraded project uses a root-level `app/` and `tests/` layout, rather than the older nested `backend/` layout described in historical notes.

```text
PlantPilot/
├── app/
│   └── plantpilot/
│       ├── main.py
│       ├── agent.py
│       ├── config.py
│       ├── schemas.py
│       ├── storage.py
│       ├── tools.py
│       ├── safety.py
│       ├── data_adapter.py
│       ├── detector.py
│       ├── detection.py
│       ├── digital_twin.py
│       └── eval/
├── tests/
├── data/                  # local/generated data may be ignored
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── DEVELOPMENT.md
├── FRONTEND_INTEGRATION.md
└── GEMINI.md
```

This is a high-level guide, not a promise that every checkout contains every path. Confirm the actual tree before editing or documenting a module.

## 4. Architecture and data flow

1. API creates/retrieves incidents and investigations.
2. The agent runs a bounded tool-calling loop.
3. Tool arguments are validated and calls pass through the code-enforced safety policy.
4. Successful tool execution can produce verified observed evidence; failed execution is not treated as successful evidence.
5. The agent reasons over persisted observations and may create hypotheses and recommendations.
6. SQLite stores investigation state, evidence, tool executions, and audit events.
7. Detection endpoints run benchmark-based failure-risk inference and statistical anomaly checks.
8. Digital-twin endpoints persist machine metadata and observed/simulated readings separately.

Ground-truth AI4I failure labels must remain isolated from agent-facing readings and be used only for evaluation.

## 5. Investigation lifecycle

The bounded agent loop requires evidence-first behavior. Successful completion must not be claimed without the required successful tool/evidence path and reasoning.

Relevant terminal statuses:
- `completed`: completion requirements satisfied.
- `incomplete`: requirements were not met or a workflow limit was reached.
- `failed`: unrecoverable provider failure was recorded after retry handling.

For new terminal provider failures in the current implementation:
- Status becomes `FAILED`.
- The original `api_error` audit event is preserved.
- An `investigation_failed` terminal event is persisted.
- The SSE stream emits the terminal event and closes.
- A failed investigation must not also be reported as completed.

Historical records are not retroactively rewritten. One historical investigation may remain `incomplete` under the behavior in effect when it was created; do not alter historical data merely to make old records match the new lifecycle contract.

## 6. Main modules

### `main.py`
FastAPI application, API routes, validation/error handling, investigation/event retrieval, SSE streaming, detection, digital-twin, and conversation endpoints. Confirm route names and schemas in the actual source or `/docs`.

### `agent.py`
Bounded investigation controller, provider call/retry handling, tool execution coordination, evidence-first enforcement, persisted audit events, and terminal status handling.

### `tools.py`
Tool registry, schemas, argument validation, and handlers. The documented core tools include historical sensor-data retrieval and deterministic diagnostic math; confirm the current registry before adding claims about tools.

### `data_adapter.py`
Loads AI4I 2020 measurements and separates agent-facing measurements from evaluation-only ground-truth labels.

### `detector.py`
IQR-based anomaly evidence and experimental heuristic checks. These are dataset-derived/experimental, not certified industrial limits.

### `detection.py`
HGB primary failure-risk prediction and optional Random Forest second assessment, according to the current project description. Check actual implementation and test results before changing model claims.

### `digital_twin.py`
Persisted software twin metadata, reading history, and observed/simulated separation. A simulation must not overwrite the latest observed state.

### `safety.py`
Code-enforced tool/recommendation constraints. Recommendations require human review; the model is not an authorization mechanism.

### `storage.py`
SQLite schema, persistence, migrations, and retrieval for supported entities/events. Preserve existing historical records.

### `eval/`
Reproducible dataset split and evaluation utilities. Keep evaluation-only labels isolated from the agent's observations.

## 7. API and integration

The API contract is documented in `FRONTEND_INTEGRATION.md`; use the live `/docs` page as the schema authority for the current checkout. API areas include incidents/investigations, detection prediction, digital twins, investigation events/SSE, and conversational follow-ups where implemented.

Terminal SSE event types include `investigation_complete`, `investigation_incomplete`, and `investigation_failed`. Frontend code must treat them as distinct outcomes.

## 8. Development and testing

From the directory containing `pyproject.toml`:

```bash
uv sync
uv run pytest tests/ -q
uv run uvicorn plantpilot.main:app --app-dir app --reload --port 8000
```

The latest reported test run from the active upgraded project was **104 passed, 0 failed**. This is a reported baseline, not an independently reproduced result for every clone/archive. Re-run tests after any change and report the actual output.

Tests using mocked provider calls do not prove live NVIDIA availability. Keep automated-test evidence and live-provider checks distinct.

## 9. Data, ML, and limitations

- AI4I 2020 contains 10,000 records.
- The HGB and optional Random Forest models are trained on benchmark data.
- The IQR detector and `tool_wear > 200` heuristic are not validated industrial operating limits.
- Predicted probabilities are estimates, not verified root causes or safety decisions.
- No live factory telemetry or physical actuation is implied by the software twin or simulation endpoints.
- Provider reliability depends on the external service.
- Authentication, authorization, rate limiting, and production-scale multi-process behavior must not be claimed unless implemented and independently verified.

## 10. Repository and security rules

- Keep `.env` and all real credentials out of Git and shared archives.
- Use `.env.example` with placeholders only.
- Do not include `.venv/`, caches, or local/generated databases unless deliberately reviewed and required.
- Do not replace a newer implementation with files from an older extracted project.
- Before changing Git history or pushing, inspect `git status`, branch history, remote state, and the complete diff.
- Preserve existing historical investigation records and backup branches.
- Update this document when the implementation changes; do not append a new status section while leaving contradictory older claims elsewhere.
