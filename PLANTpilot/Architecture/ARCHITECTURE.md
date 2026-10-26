#PLANTPILOT / ARCHITECTURE

## Product Architecture

PlantPilot is designed as a modular, human-supervised industrial operations AI framework.

## Core Components

### 1. Data Adapter

Provides operational readings to the framework.

Future sources may include simulated readings or external industrial data systems.

### 2. Monitoring Engine

Evaluates operational readings and identifies abnormal patterns.

### 3. AI Investigation Agent

Uses the configured NVIDIA model and approved diagnostic tools to investigate incidents.

### 4. Safety Layer

Validates tool usage and enforces restrictions on proposed actions.

### 5. Storage Layer

Persists incidents, investigations, evidence, tool executions, and recommendations.

### 6. Operator Dashboard

Will display machine status, investigation evidence, and recommendations for human review.

## Intended Data Flow

Data Source  
→ Monitoring  
→ Incident  
→ AI Investigation  
→ Evidence Collection  
→ Hypotheses  
→ Recommendations  
→ Human Review  
→ Audit Record

## Current Gap

The framework still needs a validated sensor-data retrieval path so the agent can investigate actual numerical readings.

## Architectural Principle

The investigation framework must remain independent of the simulator so different data sources can be connected without rewriting the agent.