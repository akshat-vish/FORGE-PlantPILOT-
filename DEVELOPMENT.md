---
tags:
  - PLANTPILOT
---

# PlantPilot — Development and Verification Notes

## Current status

The latest reported automated test run from the active upgraded project was **104 passed, 0 failed**. This is a reported baseline; rerun tests in the current checkout/environment before release. A test run from one directory does not prove that a separate clone or packaged archive has the same result.

## Setup

Use the `pyproject.toml` and `uv.lock` in the project root as the source of truth for dependencies.

```bash
uv sync
```

Create `.env` from `.env.example` and set `NVIDIA_API_KEY` locally. Do not paste the key into chat, terminal logs, screenshots, or source files.

PowerShell:

```powershell
Copy-Item .env.example .env
```

## Run the backend

From the directory containing `pyproject.toml`:

```bash
uv run uvicorn plantpilot.main:app --app-dir app --reload --port 8000
```

If the package layout differs in a particular checkout, inspect that checkout's `pyproject.toml` and follow its configured import path rather than copying a command from another directory.

## Run tests

```bash
uv run pytest tests/ -q
```

For a focused lifecycle check:

```bash
uv run pytest tests/test_investigation_lifecycle.py -q
uv run pytest tests/test_evidence_first.py -q
```

Tests that mock the provider validate application behavior under the mocked conditions; they do not prove that the live NVIDIA endpoint is currently healthy.

## Investigation lifecycle

The current implementation distinguishes:
- `completed`: evidence-first completion requirements satisfied.
- `incomplete`: the workflow ended without meeting those requirements, for example when no usable evidence was produced or a configured limit was reached.
- `failed`: an unrecoverable provider failure was recorded after retry handling.

For a new terminal provider failure, the implementation preserves the original `api_error` audit event, persists an `investigation_failed` terminal event, and terminates the SSE stream. Historical investigation records are preserved; they are not automatically rewritten by the newer lifecycle code.

## What to verify after a change

1. Run focused tests for the affected module.
2. Run the complete test suite.
3. Review the full Git diff and ensure only intended files changed.
4. Check that `.env`, API keys, databases, virtual environments, and caches are not staged.
5. Verify setup commands against the actual package layout.
6. If changing provider handling, test successful, recoverable-error, and permanent-failure paths.
7. If changing simulations, verify that simulated readings do not mutate observed twin state.
8. Update documentation only to reflect behavior present in the current source.

## Important limitations

- AI4I 2020 is benchmark data, not proof of real-world industrial performance.
- Anomaly thresholds and the `tool_wear > 200` heuristic are not certified operating limits.
- A predicted failure probability is not a verified root cause or safety decision.
- Live NVIDIA availability must be checked separately from mocked tests.

## Historical development note

Older notes in the project's development history refer to smaller test counts and earlier `incomplete` behavior for provider failures. Those statements describe earlier snapshots and must not be treated as the current status. Keep historical records only when they are clearly dated and distinguished from current verification.
