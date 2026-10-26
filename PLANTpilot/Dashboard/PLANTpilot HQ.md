# PLANTPILOT / PROJECT HQ

> [!info] Project Mission  
> Build a human-supervised industrial operations AI copilot that investigates machine incidents using operational data, presents evidence-backed findings, and recommends actions for human review.

## CURRENT STATUS

**Phase:** Stabilizing the investigation framework

**Backend:** Starts successfully  
**API:** Incident creation and investigation endpoints respond  
**AI:** NVIDIA integration has been exercised; the complete live workflow still needs verification  
**Simulator:** Not yet integrated  
**Dashboard:** Not yet built

## DEVELOPMENT ROADMAP

-  Start the FastAPI backend
    
-  Create an incident through the API
    
-  Run an investigation through the API
    
-  Add evidence-first investigation enforcement
    
-  Prevent diagnostic tools from receiving invalid inputs
    
-  Connect a sensor-data retrieval interface
    
-  Verify a successful diagnostic execution with valid readings
    
-  Verify the complete live NVIDIA tool-calling workflow
    
-  Build the simulator
    
-  Build the operator dashboard
    
-  Evaluate the system using repeatable test scenarios
    
-  Prepare the final hackathon presentation
    

## PROJECT NOTES

- [[ARCHITECTURE]]
    
- [[DEVELOPMENT]]
    
- [[EXPERIMENT & TEST RESULTS]]
    
- [[Hackathon Presentation]]
    

## CORE PRINCIPLES

1. Framework first, simulator second, dashboard third.
    
2. Never invent sensor readings or operational evidence.
    
3. Keep observations, hypotheses, and recommendations separate.
    
4. Validate tool arguments before execution.
    
5. Require human approval for risky actions.
    
6. Report measured results honestly.