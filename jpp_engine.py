"""
jpp_engine.py - Stateful JP 5-0 Planning Engine for ProjectWARGATE
==================================================================

Replaces the legacy 6-step JointStaffPlanningController with a 7-step
JP 5-0 aligned engine that:
  1. Produces structured JSON artifacts at each step (jpp_artifacts.py)
  2. Passes structured state between steps (PlanningState)
  3. Validates outputs against JP 5-0 requirements (jpp_validators.py)
  4. Generates both JSON artifacts AND human-readable reports

Each step:
  - Reads prior steps' structured artifacts from PlanningState
  - Builds prompts that demand structured output
  - Parses LLM responses into Pydantic models
  - Validates the output
  - Falls back to re-generation if validation fails (1 retry)

Usage:
    from jpp_engine import JPPEngine

    engine = JPPEngine(config)
    state = engine.run("Your scenario here...")
    # state.step2.ccirs, state.step3.coas, state.step7.opord, etc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from wargate import (
    WARGATEConfig,
    StaffRole,
    StaffAgent,
    create_all_staff_agents,
)
from jpp_artifacts import (
    PlanningState,
    Step1PlanningInitiation,
    Step2MissionAnalysis,
    Step3COADevelopment,
    Step4COAAnalysis,
    Step5COAComparison,
    Step6COAApproval,
    Step7PlanDevelopment,
    # Common types used in prompts
    Assumption,
    Task,
    CCIR,
    CenterOfGravity,
    RiskEntry,
    RunningEstimate,
    MissionStatement,
    CommandersIntent,
    ISROutline,
    COA,
    PhaseDetail,
    WargameResult,
    ARCEntry,
    DecisionPoint,
    ComparisonCriterion,
    COAScore,
    OPORDSkeleton,
    OPORDTask,
    AnnexStub,
    SyncMatrixRow,
    DSMEntry,
    AssessmentMeasure,
)
from jpp_validators import validate_step, ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# JSON EXTRACTION HELPERS
# =============================================================================

def _extract_json(text: str) -> dict | list:
    """Extract JSON from LLM response that may contain markdown fences or preamble."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { or [ and matching to the last } or ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from response (first 200 chars): {text[:200]}")


def _schema_hint(model_class: type) -> str:
    """Generate a concise JSON schema hint from a Pydantic model for the prompt."""
    schema = model_class.model_json_schema()
    # Simplify: just show the top-level properties
    props = schema.get("properties", {})
    hints = []
    for name, prop in props.items():
        ptype = prop.get("type", prop.get("anyOf", "object"))
        desc = prop.get("description", "")
        if isinstance(ptype, list):
            ptype = "/".join(str(t) for t in ptype)
        hint = f'  "{name}": <{ptype}>'
        if desc:
            hint += f"  // {desc}"
        hints.append(hint)
    return "{\n" + ",\n".join(hints) + "\n}"


# =============================================================================
# ANTI-GENERIC PROMPT DIRECTIVES
# =============================================================================

ANTI_GENERIC_RULES = """
HARD RULES — VIOLATIONS CAUSE REJECTION:
1. Ban vague verbs unless immediately operationalized with: object + method + owner + time + success condition.
   BANNED without qualifiers: "enhance", "bolster", "ensure", "integrate", "coordinate", "monitor".
   GOOD: "J6 establishes redundant SATCOM link (PACE: primary SATCOM, alternate HF, contingency mesh, emergency runner) NLT D-3"
   BAD: "Ensure communications resilience"

2. If info is unknown, output a clearly labeled ASSUMPTION with id and a TODO hook. Never generic filler.
   GOOD: {"id": "A-07", "text": "Adversary has no submarine-launched cruise missiles", "status": "UNVALIDATED", "impact_if_wrong": "Requires additional maritime AMD coverage"}
   BAD: "We assume the enemy has limited naval capability"

3. Every task must have: WHAT + WHO + WHEN + SUCCESS CONDITION.
4. Every risk must have: probability + severity + specific narrative (not generic).
5. Reference prior step outputs explicitly (PIR-01, COA-2, DP-03, etc.) — don't restate generically.
"""


# =============================================================================
# JPP ENGINE
# =============================================================================

class JPPEngine:
    """
    Stateful JP 5-0 planning engine.

    Runs 7 JPP steps sequentially. Each step:
    1. Builds a prompt referencing prior structured artifacts
    2. Calls the appropriate staff agent(s) via LLM
    3. Parses the response into a validated Pydantic artifact
    4. Stores in PlanningState for consumption by subsequent steps
    """

    def __init__(
        self,
        config: WARGATEConfig | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ):
        self.config = config or WARGATEConfig()
        self.staff: dict[StaffRole, StaffAgent] = {}
        self._initialized = False
        self.progress_callback = progress_callback

    def initialize(self) -> None:
        if self._initialized:
            return
        self.staff = create_all_staff_agents(self.config)
        self._initialized = True

    def _report_progress(self, step_name: str, step_num: int, total: int = 7) -> None:
        if self.progress_callback:
            self.progress_callback(step_name, step_num, total)
        logger.info("Step %d/%d: %s", step_num, total, step_name)

    def _invoke(self, role: StaffRole, prompt: str) -> str:
        """Invoke a staff agent and return the response."""
        return self.staff[role].invoke(prompt)

    def _invoke_and_parse(
        self,
        role: StaffRole,
        prompt: str,
        model_class: type,
        step_num: int,
        retry: bool = True,
    ) -> Any:
        """
        Invoke agent, extract JSON, parse into Pydantic model, validate.
        Retries once if validation fails.
        """
        response = self._invoke(role, prompt)

        try:
            data = _extract_json(response)
            artifact = model_class.model_validate(data)
        except Exception as e:
            logger.warning("Step %d: parse failed (%s), retrying with stricter prompt", step_num, e)
            if not retry:
                raise
            # Retry with explicit correction
            fix_prompt = (
                f"Your previous response could not be parsed as valid JSON.\n"
                f"Error: {e}\n\n"
                f"Please respond with ONLY valid JSON matching this schema:\n"
                f"{_schema_hint(model_class)}\n\n"
                f"Original task:\n{prompt}"
            )
            response = self._invoke(role, fix_prompt)
            data = _extract_json(response)
            artifact = model_class.model_validate(data)

        # Validate
        result = validate_step(step_num, artifact)
        if not result.passed and retry:
            logger.warning(
                "Step %d validation failed: %s — retrying", step_num, result.errors
            )
            fix_prompt = (
                f"Your output was missing required elements:\n"
                + "\n".join(f"- {e}" for e in result.errors)
                + "\n\nPlease regenerate with ALL required fields.\n\n"
                f"Original task:\n{prompt}"
            )
            response = self._invoke(role, fix_prompt)
            data = _extract_json(response)
            artifact = model_class.model_validate(data)

        return artifact

    # =========================================================================
    # STEP 1: PLANNING INITIATION
    # =========================================================================

    def run_step1(self, state: PlanningState) -> Step1PlanningInitiation:
        self._report_progress("Planning Initiation", 1)

        prompt = f"""You are the J5 Plans officer initiating the Joint Planning Process.

SCENARIO:
{state.scenario}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for PLANNING INITIATION with these required fields:

SCHEMA:
{_schema_hint(Step1PlanningInitiation)}

REQUIREMENTS:
- "higher_hq_direction": At least 3 bullets of higher HQ direction, constraints, and strategic guidance
- "initial_assumptions": At least 5 planning assumptions, each with id (A-01...), text, status ("UNVALIDATED"), basis, impact_if_wrong
- "problem_framing": One paragraph (3-5 sentences) framing the operational problem
- "tasks_to_staff": At least 4 tasks assigned to J2, J3, J5, J6 minimum. Each task has id, description, purpose, owner, nlt, success_condition, task_type
- "initial_information_requirements": At least 4 information requirements (these will become CCIRs later)
- "constraints": Operational constraints from higher HQ
- "restraints": Operational restraints from higher HQ
- "planning_timeline": Key planning milestones and deadlines

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J5, prompt, Step1PlanningInitiation, step_num=1
        )
        state.step1 = artifact
        return artifact

    # =========================================================================
    # STEP 2: MISSION ANALYSIS
    # =========================================================================

    def run_step2(self, state: PlanningState) -> Step2MissionAnalysis:
        self._report_progress("Mission Analysis", 2)

        # Build context from Step 1
        s1 = state.step1
        step1_context = ""
        if s1:
            step1_context = f"""
PRIOR STEP 1 OUTPUTS:
- Problem Framing: {s1.problem_framing}
- Assumptions: {json.dumps([a.model_dump() for a in s1.initial_assumptions], indent=1)}
- Higher HQ Direction: {json.dumps(s1.higher_hq_direction)}
- Tasks to Staff: {json.dumps([t.model_dump() for t in s1.tasks_to_staff], indent=1)}
- Initial Info Requirements: {json.dumps(s1.initial_information_requirements)}
- Constraints: {json.dumps(s1.constraints)}
- Restraints: {json.dumps(s1.restraints)}
"""

        prompt = f"""You are the J2 Intelligence officer leading Mission Analysis with J3, J4, J5, J6.

SCENARIO:
{state.scenario}

{step1_context}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for MISSION ANALYSIS with these required fields:

REQUIREMENTS:
1. "mission_statement": Joint format with who/what/when/where/why/full_statement
2. "commanders_intent_draft": purpose/method/end_state/key_tasks (draft — will be refined later)
3. "operational_limitations": {{"restraints": [...], "constraints": [...]}}
4. "assumptions": Carry forward Step 1 assumptions + add new ones. Validate/invalidate where possible. Min 5.
5. "tasks": Specified/implied/essential tasks with id, description, purpose, owner, nlt, success_condition, task_type
6. "cog_analysis": At least 2 entries (FRIENDLY + ADVERSARY). Each with cog, critical_capabilities, critical_requirements, critical_vulnerabilities
7. "risks": At least 3 risk entries with id, description, probability, severity, narrative, mitigation, owner
8. "ccirs": {{"PIR": [...min 3...], "FFIR": [...min 3...], "EEFI": [...min 3...]}} — each with id, text, indicator, collection_asset, timeliness
9. "isr_outline": At least 3 entries mapped to PIRs, with pir_id, named_area_of_interest, collection_asset, method, timeliness
10. "running_estimates": At least 5 entries from J2, J3, J4, J5, J6. Each with section, assessment, key_issues, recommendations
11. "facts": Known facts about the operational environment
12. "enemy_coas": MLCOA, MDCOA, and at least one other ECOA

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J2, prompt, Step2MissionAnalysis, step_num=2
        )
        state.step2 = artifact
        return artifact

    # =========================================================================
    # STEP 3: COA DEVELOPMENT
    # =========================================================================

    def run_step3(self, state: PlanningState) -> Step3COADevelopment:
        self._report_progress("COA Development", 3)

        s2 = state.step2
        step2_context = ""
        if s2:
            ccir_summary = json.dumps({
                k: [c.model_dump() for c in v] for k, v in s2.ccirs.items()
            }, indent=1)
            cog_summary = json.dumps([c.model_dump() for c in s2.cog_analysis], indent=1)
            step2_context = f"""
PRIOR STEP 2 OUTPUTS:
- Mission Statement: {s2.mission_statement.full_statement}
- Commander's Intent (draft): PURPOSE: {s2.commanders_intent_draft.purpose} | METHOD: {s2.commanders_intent_draft.method} | END STATE: {s2.commanders_intent_draft.end_state}
- COG Analysis: {cog_summary}
- CCIRs: {ccir_summary}
- Enemy COAs: {json.dumps(s2.enemy_coas)}
- Key Risks: {json.dumps([r.model_dump() for r in s2.risks], indent=1)}
- Tasks: {json.dumps([t.model_dump() for t in s2.tasks], indent=1)}
"""

        prompt = f"""You are the J5 Plans officer developing COAs with J3, Fires, Cyber/EW, J4, J6.

SCENARIO:
{state.scenario}

{step2_context}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for COA DEVELOPMENT with these requirements:

1. "coas": At least 3 DISTINCT courses of action. For EACH COA include:
   - "id": "COA-1", "COA-2", "COA-3"
   - "name": Descriptive name
   - "operational_approach": Lines of operation/effort
   - "phases": At least 3 phases (Phase 0-Shape, Phase 1-Deter, Phase 2-Seize Initiative, etc.)
     Each phase: phase_name, description, key_tasks (list), main_effort, cyber_integration, duration_estimate
   - "main_effort": Where we concentrate combat power
   - "supporting_efforts": List of supporting efforts
   - "task_organization": Task org and command relationships (supported/supporting)
   - "command_relationships": Supported/supporting designations
   - "cyber_integration": DCO, OCO, EMS, ISR linkages BY PHASE
   - "sustainment_concept": J4's initial sustainment concept
   - "comms_c2_concept": J6's initial C2/comms concept
   - "key_assumptions": Assumptions specific to this COA
   - "distinguishing_features": What makes this COA fundamentally different
   - "initial_risk": Risk entry for this COA

2. "development_rationale": Brief explanation of how COAs were derived from the mission analysis

COAs MUST be:
- FEASIBLE: Accomplishable with available means
- ACCEPTABLE: Worth the cost in resources and risk
- SUITABLE: Accomplishes the mission
- DISTINGUISHABLE: Significantly different from each other

Reference CCIRs and COG analysis from Step 2 by ID.

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J5, prompt, Step3COADevelopment, step_num=3
        )
        state.step3 = artifact
        return artifact

    # =========================================================================
    # STEP 4: COA ANALYSIS & WARGAMING
    # =========================================================================

    def run_step4(self, state: PlanningState) -> Step4COAAnalysis:
        self._report_progress("COA Analysis & Wargaming", 4)

        s2 = state.step2
        s3 = state.step3
        coa_summary = ""
        if s3:
            coa_summary = json.dumps([
                {
                    "id": c.id, "name": c.name,
                    "operational_approach": c.operational_approach,
                    "main_effort": c.main_effort,
                    "phases": [p.model_dump() for p in c.phases],
                }
                for c in s3.coas
            ], indent=1)

        ccir_summary = ""
        enemy_coas = ""
        if s2:
            ccir_summary = json.dumps({
                k: [c.model_dump() for c in v] for k, v in s2.ccirs.items()
            }, indent=1)
            enemy_coas = json.dumps(s2.enemy_coas)

        prompt = f"""You are the J2 Intelligence officer leading COA Analysis & Wargaming with J3 and Fires.

SCENARIO:
{state.scenario}

STEP 2 CCIRs: {ccir_summary}
STEP 2 ENEMY COAs: {enemy_coas}

STEP 3 COAs TO WARGAME:
{coa_summary}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for COA ANALYSIS & WARGAMING:

1. "wargame_method": Declare the method — "BELT", "BOX", "AVENUE_IN_DEPTH", or "HYBRID"

2. "wargame_results": One entry per COA. For EACH COA:
   - "coa_id": Reference the COA id from Step 3
   - "coa_name": COA name
   - "wargame_method": Method used for this COA
   - "arc_table": At least 4 Action-Reaction-Counteraction rows, each with: phase, friendly_action, enemy_reaction, friendly_counteraction, notes
   - "decision_points": At least 4 decision points with: id (DP-01...), trigger, decision, branch, sequel, owner, associated_ccir
   - "strengths": List of identified strengths
   - "weaknesses": List of identified weaknesses
   - "updated_risk": Updated risk entry for this COA after wargaming
   - "modifications": Recommended modifications based on wargame findings

3. "updated_ccirs": Any CCIRs changed or added during wargaming
4. "collection_gaps": Information collection gaps identified

Each ARC entry must reference a specific phase. Each DP must have a concrete trigger condition.
Reference Step 2 PIRs by ID when linking CCIRs to decision points.

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J2, prompt, Step4COAAnalysis, step_num=4
        )
        state.step4 = artifact
        return artifact

    # =========================================================================
    # STEP 5: COA COMPARISON
    # =========================================================================

    def run_step5(self, state: PlanningState) -> Step5COAComparison:
        self._report_progress("COA Comparison", 5)

        s3 = state.step3
        s4 = state.step4

        coa_names = ""
        wargame_summary = ""
        if s3:
            coa_names = json.dumps([{"id": c.id, "name": c.name} for c in s3.coas])
        if s4:
            wargame_summary = json.dumps([
                {
                    "coa_id": wr.coa_id,
                    "coa_name": wr.coa_name,
                    "strengths": wr.strengths,
                    "weaknesses": wr.weaknesses,
                    "decision_points_count": len(wr.decision_points),
                    "updated_risk": wr.updated_risk.model_dump() if wr.updated_risk else None,
                    "modifications": wr.modifications,
                }
                for wr in s4.wargame_results
            ], indent=1)

        prompt = f"""You are the J5 Plans officer leading COA Comparison with J3 and SJA.

SCENARIO:
{state.scenario}

COAs: {coa_names}
WARGAME RESULTS: {wargame_summary}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for COA COMPARISON:

1. "criteria": At least 6 weighted comparison criteria. Each with:
   - "name": e.g. "Mission Accomplishment", "Risk to Force", "Risk to Mission", "Logistics Feasibility", "C2 Resilience", "Legal Compliance", "Cyber Advantage", "Time to Achieve Objectives"
   - "weight": Numeric weight (e.g. 1-5 scale, higher = more important)
   - "description": What this criterion measures

2. "scoring_matrix": One entry per COA. Each with:
   - "coa_id": Reference COA from Step 3
   - "coa_name": Name
   - "scores": Dict mapping criterion name -> numeric score (1-10)
   - "weighted_total": Sum of (score * weight) for all criteria
   - "advantages": List of advantages
   - "disadvantages": List of disadvantages

3. "sensitivity_notes": What changes would alter the recommendation (at least 2)
4. "recommended_coa": COA ID of the recommended COA
5. "recommendation_justification": Tied to scores, risk, and wargame findings (at least 3 sentences)

The weighted_total must be correctly calculated from scores * weights.

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J5, prompt, Step5COAComparison, step_num=5
        )
        state.step5 = artifact
        return artifact

    # =========================================================================
    # STEP 6: COA APPROVAL
    # =========================================================================

    def run_step6(self, state: PlanningState) -> Step6COAApproval:
        self._report_progress("COA Approval", 6)

        s2 = state.step2
        s5 = state.step5

        comparison_summary = ""
        if s5:
            comparison_summary = json.dumps({
                "recommended_coa": s5.recommended_coa,
                "justification": s5.recommendation_justification,
                "scoring": [s.model_dump() for s in s5.scoring_matrix],
                "sensitivity": s5.sensitivity_notes,
            }, indent=1)

        ccir_summary = ""
        intent_draft = ""
        if s2:
            ccir_summary = json.dumps({
                k: [c.model_dump() for c in v] for k, v in s2.ccirs.items()
            }, indent=1)
            intent_draft = json.dumps(s2.commanders_intent_draft.model_dump(), indent=1)

        prompt = f"""You are the Commander making the COA approval decision.

SCENARIO:
{state.scenario}

STEP 5 COA COMPARISON:
{comparison_summary}

STEP 2 DRAFT INTENT:
{intent_draft}

STEP 2 CCIRs:
{ccir_summary}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for COA APPROVAL:

1. "decision_statement": The commander's formal COA decision (e.g. "I approve COA-2 'Thunder Strike' as the basis for plan development with the following refinements...")
2. "selected_coa_id": Which COA is approved (reference Step 3/5 COA ID)
3. "refinements_required": List of specific refinements the commander requires to the selected COA
4. "finalized_intent": Finalized Commander's intent with: purpose, method, end_state, key_tasks
5. "finalized_ccirs": Finalized CCIRs with PIR (>=3), FFIR (>=3), EEFI (>=3) — update from Step 2 based on wargaming
6. "risk_acceptance": Commander's risk acceptance statement (specific, not generic)
7. "delegated_authorities": Any delegated decision authorities
8. "planning_guidance": Specific guidance for OPORD development (priorities, timelines, focus areas)

Be decisive and specific. This is the commander's decision, not a staff recommendation.

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.COMMANDER, prompt, Step6COAApproval, step_num=6
        )
        state.step6 = artifact
        return artifact

    # =========================================================================
    # STEP 7: PLAN / ORDER DEVELOPMENT
    # =========================================================================

    def run_step7(self, state: PlanningState) -> Step7PlanDevelopment:
        self._report_progress("Plan/Order Development", 7)

        s2 = state.step2
        s3 = state.step3
        s4 = state.step4
        s6 = state.step6

        # Find the selected COA
        selected_coa = None
        if s3 and s6:
            for c in s3.coas:
                if c.id == s6.selected_coa_id:
                    selected_coa = c
                    break

        selected_coa_json = json.dumps(selected_coa.model_dump(), indent=1) if selected_coa else "SELECTED COA NOT FOUND"

        intent_json = ""
        if s6:
            intent_json = json.dumps(s6.finalized_intent.model_dump(), indent=1)

        dp_json = "[]"
        if s4:
            # Collect DPs from the selected COA's wargame
            for wr in s4.wargame_results:
                if s6 and wr.coa_id == s6.selected_coa_id:
                    dp_json = json.dumps([dp.model_dump() for dp in wr.decision_points], indent=1)
                    break

        ccir_json = ""
        if s6:
            ccir_json = json.dumps({
                k: [c.model_dump() for c in v] for k, v in s6.finalized_ccirs.items()
            }, indent=1)

        risks_json = "[]"
        if s2:
            risks_json = json.dumps([r.model_dump() for r in s2.risks], indent=1)

        prompt = f"""You are the J3 Operations officer developing the OPORD/OPLAN with J5, J4, J6, and all staff.

SCENARIO:
{state.scenario}

SELECTED COA (from Step 6):
{selected_coa_json}

FINALIZED COMMANDER'S INTENT (from Step 6):
{intent_json}

DECISION POINTS (from Step 4 wargaming):
{dp_json}

FINALIZED CCIRs (from Step 6):
{ccir_json}

IDENTIFIED RISKS (from Step 2):
{risks_json}

{ANTI_GENERIC_RULES}

Produce a JSON artifact for PLAN/ORDER DEVELOPMENT.

This is the most detailed artifact. It must include:

1. "opord": OPORD skeleton with:
   - "situation_adversary": Adversary situation (2+ paragraphs)
   - "situation_friendly": Friendly situation (2+ paragraphs)
   - "situation_assumptions": List of key assumptions still in effect
   - "situation_constraints": List of constraints and limitations
   - "mission": Full mission statement
   - "commanders_intent": {{purpose, method, end_state, key_tasks}}
   - "concept_of_operations": Phased concept (2+ paragraphs referencing phases from selected COA)
   - "tasks_to_subordinates": At least 3 tasks, each with: unit, task, purpose, nlt, coordination, success_condition
   - "coordinating_instructions": At least 3 entries (ROE placeholders, reporting, deconfliction, authorities)
   - "sustainment": Sustainment paragraph
   - "logistics_protection": Protection of logistics networks
   - "command_relationships": C2 relationships
   - "comms_plan": Communications plan with PACE
   - "cyber_defense_posture": Cyber defense posture

2. "annexes": At least 3 annex stubs:
   - "Annex B" (Intelligence): ISR plan mapped to PIRs
   - "Annex C" (Operations): Phasing + fires/cyber synchronization
   - "Annex K" (Communications): Comms + cyber defense posture
   Each annex: annex_id, title, key_content (list), responsible_section

3. "sync_matrix": At least 8 rows (D-7, D-5, D-3, D-1, D-Day, D+1, D+2, D+3). Each row:
   time_phase, maneuver, fires, intel, logistics, cyber_ew, protection, io_info, key_decision

4. "decision_support_matrix": At least 4 entries from Step 4 DPs:
   dp_id, trigger, decision, branch, owner, time_phase

5. "assessment_plan": At least 4 MOE/MOP entries:
   id, measure_type ("MOE" or "MOP"), description, metric, method_source, frequency, owner, threshold

Reference prior step IDs (PIR-01, DP-01, COA-2, etc.) throughout.

Respond with ONLY valid JSON. No commentary outside the JSON."""

        artifact = self._invoke_and_parse(
            StaffRole.J3, prompt, Step7PlanDevelopment, step_num=7
        )
        state.step7 = artifact
        return artifact

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    def run(
        self,
        scenario: str,
        operation_name: str = "UNNAMED_OP",
    ) -> PlanningState:
        """
        Execute the complete 7-step JP 5-0 planning pipeline.

        Returns:
            PlanningState with all 7 step artifacts populated
        """
        self.initialize()

        state = PlanningState(scenario=scenario, operation_name=operation_name)

        self.run_step1(state)
        self.run_step2(state)
        self.run_step3(state)
        self.run_step4(state)
        self.run_step5(state)
        self.run_step6(state)
        self.run_step7(state)

        logger.info("JPP Engine: All 7 steps complete for %s", operation_name)
        return state

    # =========================================================================
    # HUMAN-READABLE REPORT GENERATION
    # =========================================================================

    @staticmethod
    def render_report(state: PlanningState) -> str:
        """
        Generate a human-readable report from the structured planning state.
        Complements the JSON artifacts with a formatted text report.
        """
        lines: list[str] = []

        def section(title: str):
            lines.append("")
            lines.append("=" * 78)
            lines.append(f"  {title}")
            lines.append("=" * 78)

        def subsection(title: str):
            lines.append("")
            lines.append(f"--- {title} ---")

        lines.append("PROJECT WARGATE — JOINT STAFF PLANNING PRODUCT (JP 5-0 ALIGNED)")
        lines.append(f"Operation: {state.operation_name}")

        # Step 1
        s1 = state.step1
        if s1:
            section("STEP 1: PLANNING INITIATION")
            subsection("Higher HQ Direction")
            for bullet in s1.higher_hq_direction:
                lines.append(f"  • {bullet}")
            subsection("Problem Framing")
            lines.append(f"  {s1.problem_framing}")
            subsection("Initial Assumptions")
            for a in s1.initial_assumptions:
                lines.append(f"  [{a.id}] {a.text} (Status: {a.status})")
            subsection("Tasks to Staff")
            for t in s1.tasks_to_staff:
                lines.append(f"  [{t.id}] {t.owner}: {t.description} (NLT: {t.nlt})")

        # Step 2
        s2 = state.step2
        if s2:
            section("STEP 2: MISSION ANALYSIS")
            subsection("Mission Statement")
            lines.append(f"  {s2.mission_statement.full_statement}")
            subsection("Commander's Intent (Draft)")
            lines.append(f"  PURPOSE: {s2.commanders_intent_draft.purpose}")
            lines.append(f"  METHOD:  {s2.commanders_intent_draft.method}")
            lines.append(f"  END STATE: {s2.commanders_intent_draft.end_state}")
            subsection("Centers of Gravity")
            for cog in s2.cog_analysis:
                lines.append(f"  [{cog.entity}] COG: {cog.cog}")
                lines.append(f"    CC: {', '.join(cog.critical_capabilities)}")
                lines.append(f"    CR: {', '.join(cog.critical_requirements)}")
                lines.append(f"    CV: {', '.join(cog.critical_vulnerabilities)}")
            subsection("CCIRs")
            for category in ["PIR", "FFIR", "EEFI"]:
                lines.append(f"  {category}:")
                for c in s2.ccirs.get(category, []):
                    lines.append(f"    [{c.id}] {c.text}")
            subsection("Risks")
            for r in s2.risks:
                lines.append(f"  [{r.id}] {r.description} — P:{r.probability.value} S:{r.severity.value}")
                lines.append(f"    Mitigation: {r.mitigation}")

        # Step 3
        s3 = state.step3
        if s3:
            section("STEP 3: COA DEVELOPMENT")
            for coa in s3.coas:
                subsection(f"{coa.id}: {coa.name}")
                lines.append(f"  Approach: {coa.operational_approach}")
                lines.append(f"  Main Effort: {coa.main_effort}")
                lines.append(f"  Cyber: {coa.cyber_integration}")
                for phase in coa.phases:
                    lines.append(f"  {phase.phase_name}: {phase.description}")

        # Step 4
        s4 = state.step4
        if s4:
            section("STEP 4: COA ANALYSIS & WARGAMING")
            lines.append(f"  Method: {s4.wargame_method}")
            for wr in s4.wargame_results:
                subsection(f"Wargame: {wr.coa_id} — {wr.coa_name}")
                lines.append("  ARC Table:")
                for arc in wr.arc_table:
                    lines.append(f"    [{arc.phase}] Action: {arc.friendly_action}")
                    lines.append(f"           Reaction: {arc.enemy_reaction}")
                    lines.append(f"           Counter:  {arc.friendly_counteraction}")
                lines.append("  Decision Points:")
                for dp in wr.decision_points:
                    lines.append(f"    [{dp.id}] Trigger: {dp.trigger}")
                    lines.append(f"           Decision: {dp.decision}")
                    lines.append(f"           Branch: {dp.branch}")

        # Step 5
        s5 = state.step5
        if s5:
            section("STEP 5: COA COMPARISON")
            subsection("Criteria (weighted)")
            for c in s5.criteria:
                lines.append(f"  {c.name} (weight: {c.weight})")
            subsection("Scoring Matrix")
            for s in s5.scoring_matrix:
                lines.append(f"  {s.coa_id} ({s.coa_name}): weighted_total = {s.weighted_total}")
                for crit, score in s.scores.items():
                    lines.append(f"    {crit}: {score}")
            subsection("Recommendation")
            lines.append(f"  Recommended COA: {s5.recommended_coa}")
            lines.append(f"  Justification: {s5.recommendation_justification}")

        # Step 6
        s6 = state.step6
        if s6:
            section("STEP 6: COA APPROVAL")
            lines.append(f"  Decision: {s6.decision_statement}")
            lines.append(f"  Selected COA: {s6.selected_coa_id}")
            subsection("Finalized Commander's Intent")
            lines.append(f"  PURPOSE: {s6.finalized_intent.purpose}")
            lines.append(f"  METHOD:  {s6.finalized_intent.method}")
            lines.append(f"  END STATE: {s6.finalized_intent.end_state}")
            subsection("Risk Acceptance")
            lines.append(f"  {s6.risk_acceptance}")

        # Step 7
        s7 = state.step7
        if s7:
            section("STEP 7: PLAN/ORDER DEVELOPMENT")
            o = s7.opord
            subsection("1. SITUATION")
            lines.append(f"  Adversary: {o.situation_adversary[:300]}...")
            lines.append(f"  Friendly: {o.situation_friendly[:300]}...")
            subsection("2. MISSION")
            lines.append(f"  {o.mission}")
            subsection("3. EXECUTION")
            lines.append(f"  Intent: {o.commanders_intent.purpose}")
            lines.append(f"  Concept: {o.concept_of_operations[:500]}...")
            lines.append("  Tasks to Subordinates:")
            for t in o.tasks_to_subordinates:
                lines.append(f"    {t.unit}: {t.task} (Purpose: {t.purpose}, NLT: {t.nlt})")
            subsection("4. ADMIN & LOGISTICS")
            lines.append(f"  {o.sustainment[:300]}...")
            subsection("5. COMMAND & SIGNAL")
            lines.append(f"  C2: {o.command_relationships}")
            lines.append(f"  Comms: {o.comms_plan[:200]}...")

            subsection("SYNCHRONIZATION MATRIX")
            for row in s7.sync_matrix:
                lines.append(
                    f"  {row.time_phase:>6} | MAN: {row.maneuver[:20]:20} | "
                    f"FIRES: {row.fires[:20]:20} | CYBER: {row.cyber_ew[:20]:20}"
                )

            subsection("DECISION SUPPORT MATRIX")
            for dsm in s7.decision_support_matrix:
                lines.append(f"  [{dsm.dp_id}] {dsm.trigger} → {dsm.decision}")

            subsection("ASSESSMENT PLAN")
            for m in s7.assessment_plan:
                lines.append(f"  [{m.id}] {m.measure_type}: {m.description}")
                lines.append(f"    Metric: {m.metric} | Source: {m.method_source} | Threshold: {m.threshold}")

        lines.append("")
        lines.append("=" * 78)
        lines.append("  END OF PLANNING PRODUCT")
        lines.append("=" * 78)

        return "\n".join(lines)
