# PLANTPILOT / EXPERIMENTS

## Experiment 001 — Placeholder Incident

**Input:** Incident description was `string`.

**Observed:**

- The investigation produced a textual finding.
    
- No evidence, hypotheses, or recommendations were stored.
    
- The earlier implementation marked the investigation completed.
    

**Conclusion:** The earlier workflow could complete without diagnostic evidence.

---

## Experiment 002 — Evidence-First Enforcement

**Input:** An incident with no numerical sensor readings.

**Observed:**

- `diagnostic_math` received `values=[]`.
    
- The tool rejected the arguments.
    
- A failed tool execution was persisted.
    
- The investigation status was `incomplete`.
    

**Conclusion:** Failure handling improved, but a successful diagnostic has not yet been demonstrated.

---

## Experiment 003 — Valid Sensor Readings

**Status:** Not yet run.

**Objective:** Provide valid numerical readings through an approved data source, execute a diagnostic tool, persist evidence, and generate a grounded investigation result.

### Results

To be completed after the experiment runs.

### Evidence

Paste actual responses, test outputs, or sanitized logs here.