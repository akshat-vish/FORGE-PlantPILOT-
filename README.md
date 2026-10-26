# PlantPilot — Evidence-First Industrial Operations AI Copilot

PlantPilot is a human-supervised backend prototype for industrial incident investigation and predictive-maintenance experimentation. It combines evidence-first LLM investigations, benchmark-based failure-risk prediction, statistical anomaly evidence, persisted software digital twins, and isolated what-if predictions.

**PlantPilot is not connected to physical factory sensors by default.** Predictions and recommendations are decision-support outputs, not safety decisions or verified root causes.

## Current verified scope

- **API/backend:** FastAPI, Pydantic, SQLite, and an OpenAI-compatible client configured for NVIDIA inference.
- **Investigation agent:** bounded tool-calling workflow with evidence-first completion rules, persisted audit events, and explicit terminal failure handling.
- **Provider failure lifecycle:** after an unrecoverable provider failure/retry exhaustion, a new investigation terminates with status `failed` and an `investigation_failed` terminal event. The original `api_error` audit event is preserved. Historical records are not retroactively rewritten.
- **Sensor-data adapter:** reads historical measurements from the AI4I 2020 predictive-maintenance dataset. This is benchmark data, not live telemetry.
- **Anomaly evidence:** IQR-based statistical checks and documented experimental heuristics; these are not validated industrial operating limits.
- **Failure-risk prediction:** HistGradientBoosting (HGB) primary predictor and optional Random Forest secondary assessment, trained on AI4I 2020 data.
- **Digital twin:** persisted machine metadata and reading history, with observed and simulated readings labelled separately.
- **What-if simulation:** evaluates hypothetical readings without changing the twin's latest observed state.
- **Human oversight:** recommendations remain proposals requiring human review.

## Dataset and model limitations

The bundled AI4I 2020 dataset contains 10,000 records. The benchmark is useful for development and evaluation but does not establish performance on real industrial equipment. Ground-truth failure labels must remain isolated from agent-facing sensor observations and must only be used for evaluation.

A predicted failure probability is a model estimate. It is not a confirmed failure, root cause, or safety decision. The `tool_wear > 200` heuristic is experimental and must not be interpreted as an industrial safety threshold.

## Requirements

- Python 3.13 or newer (match `pyproject.toml`)
- `uv`
- An NVIDIA API key for live LLM investigations

## Setup

From the repository root:

```bash
uv sync
```

Create a local `.env` from the example and set your key:

```bash
cp .env.example .env
# Edit .env and set NVIDIA_API_KEY
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
# Edit .env and set NVIDIA_API_KEY
```

Never commit `.env`, print its contents, or expose `NVIDIA_API_KEY` to a browser/frontend.

## Run the API

```bash
uv run uvicorn plantpilot.main:app --app-dir app --reload --port 8000
```

- Base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Run commands from the directory containing `pyproject.toml`. If your checkout has a different package layout, follow that checkout's `pyproject.toml` and test configuration.

## Run tests

```bash
uv run pytest tests/ -q
```

The latest reported run from the active upgraded project was **104 passed, 0 failed**. Treat this as a reported baseline, not a guarantee that every checkout or clean environment will reproduce it. Re-run the suite after setup and before release. Unit/integration tests that mock the provider do not prove live NVIDIA availability.

## Main API areas

See [`FRONTEND_INTEGRATION.md`](FRONTEND_INTEGRATION.md) for request/response examples and streaming details. The API includes:

- Incidents and investigations
- Detection model metadata and failure-risk prediction
- Digital-twin creation, observed readings, history, and what-if simulation
- Investigation event retrieval and server-sent event (SSE) streaming
- Persisted conversational follow-ups linked to investigation context

## Investigation terminal states

- `completed`: the evidence-first completion requirements were met.
- `incomplete`: the workflow ended without satisfying completion requirements, such as no successful tool/evidence path or a step/tool limit.
- `failed`: an unrecoverable provider failure was recorded. A terminal `investigation_failed` event is emitted for new failures handled by the current lifecycle implementation.

A successful completion and a terminal failure must not be represented as the same outcome. See `FRONTEND_INTEGRATION.md`.

## Security and local artifacts

Do not commit:
- `.env` or real API credentials
- `.venv/`, Python caches, or test caches
- Local/generated SQLite databases unless deliberately reviewed and required
- IDE-specific state and temporary artifacts

Use `.env.example` for placeholder configuration. `.gitignore` protects Git operations only; it does not automatically exclude files from manually created ZIP/RAR archives.

## Known limitations

- No built-in physical sensor or factory telemetry connection.
- AI4I-trained models have not been validated on a real factory deployment.
- Provider availability and inference latency depend on the external NVIDIA service.
- SQLite and in-memory live SSE fan-out are intended for this prototype; the documented deployment assumptions should be reviewed before multi-process production use.
- No claim of production safety certification, real-world industrial validation, or guaranteed root-cause diagnosis is made.
