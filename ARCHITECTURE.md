# PlantPilot — Architecture

## Purpose

PlantPilot is a modular, human-supervised backend prototype for evidence-first incident investigation and predictive-maintenance experimentation. It combines deterministic data/tools, statistical and ML predictions, a bounded LLM investigation workflow, and persisted audit records.

It does not connect to physical sensors unless an external integration supplies readings.

## Core components

### 1. API layer — FastAPI

Exposes incident, investigation, detection, digital-twin, conversation, event-retrieval, and SSE streaming endpoints. Pydantic models validate API inputs and outputs.

### 2. Data adapter — AI4I 2020

Loads historical measurements from the AI4I 2020 predictive-maintenance CSV. Agent-facing sensor readings exclude ground-truth failure labels. Labels are reserved for evaluation.

### 3. Detection and evaluation

- **Primary predictor:** HistGradientBoosting (HGB), trained on benchmark data with threshold selection using development data.
- **Optional secondary predictor:** Random Forest, requested explicitly.
- **Anomaly evidence:** IQR-based checks and documented experimental heuristics.
- **Evaluation:** reproducible data split and classification metrics.

The models and thresholds are benchmark-derived, not validated industrial limits. A prediction is not a root-cause determination or a safety decision.

### 4. Investigation agent

The agent runs a bounded tool-calling loop. Tool arguments are validated, tool calls pass through the safety policy, and evidence is persisted only after successful tool execution. Completion requires the configured evidence-first criteria.

New terminal outcomes are distinct:
- `completed`: required evidence and reasoning were produced.
- `incomplete`: completion requirements were not met or a configured workflow limit was reached.
- `failed`: an unrecoverable provider failure was recorded after retry handling.

For terminal provider failure, the original `api_error` event is preserved and an `investigation_failed` terminal event is persisted. The SSE stream delivers the terminal event and closes. Historical investigations are not rewritten to match newer behavior.

### 5. Tools and safety layer

The registered diagnostic tools include historical sensor-data retrieval and deterministic numerical summaries. The safety policy is code-enforced and independent of the LLM. Recommendations require human review; the LLM cannot authorize an operational action merely by proposing it.

### 6. Digital twin and simulation

A digital twin is a persisted software representation containing machine metadata and reading history. Readings are labelled as `observed` or `simulated`. A what-if simulation must not overwrite the latest observed reading or be represented as physical telemetry. The simulator is separate from physical machine control.

### 7. Storage and audit

SQLite persists incidents, investigations, evidence, hypotheses, recommendations, tool executions, conversations, and audit events as supported by the current schema. The audit history supports event retrieval and SSE replay. Live SSE fan-out is in-memory; the prototype's single-process assumption must be considered for deployment.

## High-level data flow

```text
Historical dataset / externally supplied readings
                    |
                    v
          Data adapter and detection
                    |
                    v
             Incident / API
                    |
                    v
         Bounded investigation agent
                    |
         validated, safety-checked tools
                    |
                    v
       Observations + tool execution records
                    |
                    v
     Evidence -> hypotheses -> recommendations
                    |
                    v
       SQLite persistence and audit events
                    |
                    v
           Human review / API client
```

## Safety and provenance principles

1. Ground-truth failure labels are not exposed as agent observations.
2. Successful tool execution is required before evidence is treated as verified.
3. Failed tool calls are recorded as failures, not successful observations.
4. Recommendations require human review.
5. Simulated readings remain distinguishable from observed readings.
6. Provider failures must not be represented as successful investigations.
7. Predictions are decision-support estimates, not safety certifications.

## Current limitations

- AI4I benchmark data is not live factory telemetry.
- HGB, Random Forest, IQR bounds, and experimental heuristics have not been validated for a real industrial installation.
- External NVIDIA inference can fail or be unavailable.
- SQLite and in-memory SSE fan-out are prototype choices; multi-process deployment requires additional design/verification.
- No production authentication, authorization, or rate-limiting claims should be inferred unless implemented and verified in the current source.
