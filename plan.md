# JPP Refactor Implementation Plan

## Findings

### Current Architecture (Two Pipelines)

**Legacy Controller** (`wargate.py:JointStaffPlanningController`): 6 sequential steps
- Step 1: J2 Intel Estimate → free-form text
- Step 2: J5+J3 COA Development → `[{"concepts": str, "details": str}]`
- Step 3: Functional Staff Reviews → `dict[role, str]`
- Step 4: SJA Review → free-form text
- Step 5: Commander Synthesis → `(str, str)`
- Step 6: Final Output → formatted string (hardcoded template)

**Orchestration** (`wargate_orchestration.py:MeetingOrchestrator`): 7 JPP phases, 4-step meetings
- Each phase: staff meeting → slides → commander brief → commander guidance
- State passes as `prior_context` (truncated text blob, not structured)
- Output: `PhaseResult(meeting, slides, brief, guidance)` — all text-oriented

### Key Problems
1. **No structured JSON artifacts** — steps produce prose, not parseable data
2. **No state graph** — step N+1 doesn't consume step N's structured output
3. **Steps don't align to JP 5-0** — legacy is 6 steps (not 7), missing Planning Initiation, COA Comparison, COA Approval as distinct phases
4. **No validators** — outputs are never checked for required elements
5. **Prompts don't mandate artifact production** — they ask for analysis but don't enforce structure

### Files to Create
- `jpp_artifacts.py` — Pydantic models for all 7 step outputs + state container
- `jpp_validators.py` — Validation functions per step
- `tests/test_jpp_validators.py` — Tests
- `examples/example_step2_output.json` — Sample Mission Analysis artifact
- `examples/example_step4_output.json` — Sample Wargaming artifact

### Files to Modify
- `wargate.py` — Refactor `JointStaffPlanningController` from 6-step to 7-step with structured artifacts
- `wargate_orchestration.py` — Update `PHASE_CONFIGS` prompts to mandate structured artifact production
- `wargate_backend.py` — Update to use new artifact types

## Implementation Approach

### Phase 1: Data Models (`jpp_artifacts.py`)
Define Pydantic models for each JPP step's output plus a `PlanningState` that accumulates across steps.

### Phase 2: Validators (`jpp_validators.py`)
Implement validation functions that check presence/count of required elements.

### Phase 3: Refactor Controller (wargate.py)
Rewrite each step to:
1. Build prompts that demand structured output (JSON extraction)
2. Parse LLM responses into Pydantic models
3. Store in PlanningState
4. Pass structured data to next step

### Phase 4: Update Orchestration
Update PHASE_CONFIGS key_outputs and focus_areas + meeting prompts to enforce artifact production.

### Phase 5: Tests + Examples
Validate with a sample scenario.
