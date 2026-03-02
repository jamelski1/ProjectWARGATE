"""
jpp_artifacts.py - Structured Artifact Models for JP 5-0 Aligned JPP Pipeline
==============================================================================

Defines Pydantic models for the structured output of each JPP step.
Each step's output is a JSON-serializable artifact that:
  1. Contains all JP 5-0 required elements for that step
  2. Can be validated for completeness (see jpp_validators.py)
  3. Is consumed by subsequent steps via PlanningState

Usage:
    from jpp_artifacts import PlanningState, Step2MissionAnalysis, ...

    state = PlanningState(scenario="...")
    state.step2 = Step2MissionAnalysis(...)
    # step3 prompt can reference state.step2.ccirs, state.step2.cog_analysis, etc.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# COMMON / SHARED TYPES
# =============================================================================

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class Probability(str, Enum):
    UNLIKELY = "UNLIKELY"
    POSSIBLE = "POSSIBLE"
    LIKELY = "LIKELY"
    ALMOST_CERTAIN = "ALMOST_CERTAIN"


class Feasibility(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    GO_WITH_MITIGATION = "GO_WITH_MITIGATION"


class RiskEntry(BaseModel):
    """A single risk with probability, severity, narrative, and mitigation."""
    id: str = Field(description="Risk identifier, e.g. R-01")
    description: str = Field(description="What the risk is")
    probability: Probability
    severity: Severity
    narrative: str = Field(description="1-2 sentence explanation")
    mitigation: str = Field(default="", description="Proposed mitigation")
    owner: str = Field(default="", description="Staff section responsible")


class CCIR(BaseModel):
    """Commander's Critical Information Requirement."""
    id: str = Field(description="Identifier, e.g. PIR-01, FFIR-01, EEFI-01")
    text: str = Field(description="The requirement statement")
    indicator: str = Field(default="", description="Observable indicator")
    collection_asset: str = Field(default="", description="How to collect")
    timeliness: str = Field(default="", description="When needed by")


class Assumption(BaseModel):
    """Planning assumption with validation status."""
    id: str = Field(description="Identifier, e.g. A-01")
    text: str = Field(description="The assumption statement")
    status: Literal["VALID", "INVALID", "UNVALIDATED"] = "UNVALIDATED"
    basis: str = Field(default="", description="Why we believe this")
    impact_if_wrong: str = Field(default="", description="What changes if invalid")


class Task(BaseModel):
    """A task with purpose, owner, time, and success condition."""
    id: str = Field(description="Task identifier, e.g. T-01")
    description: str
    purpose: str = Field(default="")
    owner: str = Field(default="", description="Staff section or unit responsible")
    nlt: str = Field(default="", description="No later than time/phase")
    coordination: str = Field(default="", description="Required coordination")
    success_condition: str = Field(default="", description="How we know it's done")
    task_type: Literal["SPECIFIED", "IMPLIED", "ESSENTIAL"] = "IMPLIED"


class CenterOfGravity(BaseModel):
    """Center of Gravity analysis with critical factors."""
    entity: Literal["FRIENDLY", "ADVERSARY"]
    cog: str = Field(description="The center of gravity itself")
    critical_capabilities: list[str] = Field(default_factory=list, description="CC: What the COG can do")
    critical_requirements: list[str] = Field(default_factory=list, description="CR: What the COG needs")
    critical_vulnerabilities: list[str] = Field(default_factory=list, description="CV: Exploitable weaknesses")


class RunningEstimate(BaseModel):
    """A staff section's running estimate update."""
    section: str = Field(description="e.g. J2, J3, J4, J5, J6")
    assessment: str = Field(description="Short but specific assessment")
    key_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# STEP 1: PLANNING INITIATION
# =============================================================================

class Step1PlanningInitiation(BaseModel):
    """JP 5-0 Step 1 artifacts."""

    higher_hq_direction: list[str] = Field(
        description="Bullets: higher HQ direction and constraints"
    )
    initial_assumptions: list[Assumption] = Field(
        description="Minimum 5 initial planning assumptions"
    )
    problem_framing: str = Field(
        description="Initial problem framing (1 paragraph max)"
    )
    tasks_to_staff: list[Task] = Field(
        description="Initial tasks to staff (J2/J3/J5/J6 minimum)"
    )
    initial_information_requirements: list[str] = Field(
        description="Initial information requirements (not full CCIR yet)"
    )
    planning_timeline: str = Field(
        default="", description="Key planning milestones"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Operational constraints from higher HQ"
    )
    restraints: list[str] = Field(
        default_factory=list, description="Operational restraints from higher HQ"
    )


# =============================================================================
# STEP 2: MISSION ANALYSIS
# =============================================================================

class MissionStatement(BaseModel):
    """Joint mission statement format."""
    who: str = Field(description="The force executing")
    what: str = Field(description="The task")
    when: str = Field(description="Time or condition")
    where: str = Field(description="Location / AO")
    why: str = Field(description="Purpose")
    full_statement: str = Field(description="Complete formatted mission statement")


class CommandersIntent(BaseModel):
    """Commander's Intent in Purpose/Method/End State format."""
    purpose: str = Field(description="Why we are conducting this operation")
    method: str = Field(description="Broad approach to accomplish the mission")
    end_state: str = Field(description="Conditions that define success")
    key_tasks: list[str] = Field(default_factory=list, description="Tasks critical to intent")


class ISROutline(BaseModel):
    """ISR/collection outline entry mapped to a PIR."""
    pir_id: str = Field(description="PIR this supports, e.g. PIR-01")
    named_area_of_interest: str = Field(default="", description="NAI if applicable")
    collection_asset: str = Field(description="What collects")
    method: str = Field(description="How it collects")
    timeliness: str = Field(default="", description="When needed")


class Step2MissionAnalysis(BaseModel):
    """JP 5-0 Step 2 artifacts."""

    mission_statement: MissionStatement
    commanders_intent_draft: CommandersIntent
    operational_limitations: dict[str, list[str]] = Field(
        description="Keys: 'restraints', 'constraints'"
    )
    assumptions: list[Assumption] = Field(
        description="Validated/invalidated list from Step 1 + new"
    )
    tasks: list[Task] = Field(
        description="Specified, implied, and essential tasks"
    )
    cog_analysis: list[CenterOfGravity] = Field(
        description="Friendly and adversary COG with CC/CR/CV"
    )
    risks: list[RiskEntry] = Field(
        description="Risk identification with prob/sev + narrative"
    )
    ccirs: dict[str, list[CCIR]] = Field(
        description="Keys: 'PIR', 'FFIR', 'EEFI' — min 3 each"
    )
    isr_outline: list[ISROutline] = Field(
        description="Initial ISR/collection outline mapped to PIRs"
    )
    running_estimates: list[RunningEstimate] = Field(
        description="Updates by J2/J3/J4/J5/J6"
    )
    facts: list[str] = Field(default_factory=list, description="Known facts")
    enemy_coas: list[str] = Field(
        default_factory=list,
        description="MLCOA, MDCOA, and other ECOAs"
    )


# =============================================================================
# STEP 3: COA DEVELOPMENT
# =============================================================================

class PhaseDetail(BaseModel):
    """Detail for a single phase within a COA."""
    phase_name: str = Field(description="e.g. Phase 0 - Shape")
    description: str
    key_tasks: list[str] = Field(default_factory=list)
    main_effort: str = Field(default="")
    cyber_integration: str = Field(default="", description="DCO/OCO/EMS for this phase")
    duration_estimate: str = Field(default="")


class COA(BaseModel):
    """A single Course of Action with full JP 5-0 detail."""
    id: str = Field(description="COA identifier, e.g. COA-1")
    name: str = Field(description="Descriptive name")
    operational_approach: str = Field(
        description="Lines of operation/effort description"
    )
    phases: list[PhaseDetail] = Field(
        description="Phase 0-III+ as scenario dictates"
    )
    main_effort: str = Field(description="Where we concentrate combat power")
    supporting_efforts: list[str] = Field(default_factory=list)
    task_organization: str = Field(
        description="Task org / command relationships (supported/supporting)"
    )
    command_relationships: str = Field(
        default="",
        description="Supported/supporting designations"
    )
    cyber_integration: str = Field(
        description="DCO, OCO, EMS, ISR linkages across phases"
    )
    sustainment_concept: str = Field(
        description="Initial sustainment concept (J4)"
    )
    comms_c2_concept: str = Field(
        description="Initial comms/C2 concept (J6)"
    )
    key_assumptions: list[str] = Field(default_factory=list)
    distinguishing_features: str = Field(
        default="", description="What makes this COA distinct"
    )
    initial_risk: RiskEntry | None = Field(
        default=None, description="Initial risk assessment for this COA"
    )


class Step3COADevelopment(BaseModel):
    """JP 5-0 Step 3 artifacts."""

    coas: list[COA] = Field(
        description="Minimum 3 distinct COAs"
    )
    development_rationale: str = Field(
        default="",
        description="Brief explanation of how COAs were derived"
    )


# =============================================================================
# STEP 4: COA ANALYSIS & WARGAMING
# =============================================================================

class ARCEntry(BaseModel):
    """Action-Reaction-Counteraction table row."""
    phase: str = Field(description="Which phase this applies to")
    friendly_action: str
    enemy_reaction: str
    friendly_counteraction: str
    notes: str = Field(default="")


class DecisionPoint(BaseModel):
    """Friendly decision point with trigger and branch/sequel."""
    id: str = Field(description="e.g. DP-01")
    trigger: str = Field(description="Condition that activates this DP")
    decision: str = Field(description="What the commander must decide")
    branch: str = Field(default="", description="Contingency plan if triggered")
    sequel: str = Field(default="", description="Follow-on operation")
    owner: str = Field(default="", description="Who owns the decision")
    associated_ccir: str = Field(default="", description="Related PIR/FFIR")


class WargameResult(BaseModel):
    """Wargame results for a single COA."""
    coa_id: str
    coa_name: str
    wargame_method: Literal[
        "BELT", "BOX", "AVENUE_IN_DEPTH", "HYBRID"
    ] = Field(description="Wargame technique used")
    arc_table: list[ARCEntry] = Field(
        description="Action-Reaction-Counteraction entries"
    )
    decision_points: list[DecisionPoint] = Field(
        description="Minimum 4 decision points"
    )
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    updated_risk: RiskEntry | None = None
    modifications: list[str] = Field(
        default_factory=list,
        description="Recommended modifications to this COA"
    )


class Step4COAAnalysis(BaseModel):
    """JP 5-0 Step 4 artifacts."""

    wargame_method: str = Field(
        description="Primary wargame method declared"
    )
    wargame_results: list[WargameResult] = Field(
        description="Results per COA"
    )
    updated_ccirs: list[CCIR] = Field(
        default_factory=list,
        description="CCIRs changed or added during wargaming"
    )
    collection_gaps: list[str] = Field(
        default_factory=list,
        description="Information collection gaps identified"
    )


# =============================================================================
# STEP 5: COA COMPARISON
# =============================================================================

class ComparisonCriterion(BaseModel):
    """A single evaluation criterion with weight."""
    name: str = Field(description="e.g. 'Mission Accomplishment'")
    weight: float = Field(description="Relative weight (higher = more important)")
    description: str = Field(default="")


class COAScore(BaseModel):
    """A single COA's scores against all criteria."""
    coa_id: str
    coa_name: str
    scores: dict[str, float] = Field(
        description="criterion_name -> numeric score"
    )
    weighted_total: float = Field(description="Sum of (score * weight)")
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)


class Step5COAComparison(BaseModel):
    """JP 5-0 Step 5 artifacts."""

    criteria: list[ComparisonCriterion] = Field(
        description="Minimum 6 weighted criteria"
    )
    scoring_matrix: list[COAScore] = Field(
        description="One entry per COA with scores"
    )
    sensitivity_notes: list[str] = Field(
        default_factory=list,
        description="What changes would alter the recommendation"
    )
    recommended_coa: str = Field(
        description="COA ID of the recommended COA"
    )
    recommendation_justification: str = Field(
        description="Justification tied to scores + risk"
    )


# =============================================================================
# STEP 6: COA APPROVAL
# =============================================================================

class Step6COAApproval(BaseModel):
    """JP 5-0 Step 6 artifacts."""

    decision_statement: str = Field(
        description="Commander's COA decision statement"
    )
    selected_coa_id: str = Field(
        description="Which COA was approved"
    )
    refinements_required: list[str] = Field(
        default_factory=list,
        description="Refinements required to the selected COA"
    )
    finalized_intent: CommandersIntent = Field(
        description="Finalized Commander's intent"
    )
    finalized_ccirs: dict[str, list[CCIR]] = Field(
        description="Finalized CCIRs: PIR, FFIR, EEFI"
    )
    risk_acceptance: str = Field(
        description="Risk acceptance statement"
    )
    delegated_authorities: list[str] = Field(
        default_factory=list,
        description="Any delegated decision authorities"
    )
    planning_guidance: str = Field(
        description="Planning guidance for OPORD development"
    )


# =============================================================================
# STEP 7: PLAN / ORDER DEVELOPMENT
# =============================================================================

class OPORDTask(BaseModel):
    """A task to subordinate/supporting command."""
    unit: str = Field(description="Who")
    task: str = Field(description="What to do")
    purpose: str = Field(description="Why")
    nlt: str = Field(default="", description="No later than")
    coordination: str = Field(default="", description="Required coordination")
    success_condition: str = Field(default="", description="How we know it's done")


class AnnexStub(BaseModel):
    """Stub for an OPORD annex."""
    annex_id: str = Field(description="e.g. Annex B, Annex C, Annex K")
    title: str
    key_content: list[str] = Field(description="Key items this annex addresses")
    responsible_section: str = Field(default="")


class SyncMatrixRow(BaseModel):
    """A single row in the synchronization matrix."""
    time_phase: str = Field(description="e.g. D-7, D-3, D-Day, D+1")
    maneuver: str = Field(default="")
    fires: str = Field(default="")
    intel: str = Field(default="")
    logistics: str = Field(default="")
    cyber_ew: str = Field(default="")
    protection: str = Field(default="")
    io_info: str = Field(default="")
    key_decision: str = Field(default="", description="DP if applicable")


class DSMEntry(BaseModel):
    """Decision Support Matrix entry."""
    dp_id: str = Field(description="Decision Point ID, e.g. DP-01")
    trigger: str
    decision: str
    branch: str = Field(default="")
    owner: str = Field(default="")
    time_phase: str = Field(default="", description="When in the timeline")


class AssessmentMeasure(BaseModel):
    """MOE/MOP entry for the assessment plan."""
    id: str
    measure_type: Literal["MOE", "MOP"]
    description: str = Field(description="What we're measuring")
    metric: str = Field(description="How we measure it")
    method_source: str = Field(description="Data source or collection method")
    frequency: str = Field(default="", description="How often we assess")
    owner: str = Field(default="", description="Who is responsible")
    threshold: str = Field(default="", description="Success/failure threshold")


class OPORDSkeleton(BaseModel):
    """Draft OPORD/OPLAN skeleton with joint-style sections."""

    # Para 1: Situation
    situation_adversary: str
    situation_friendly: str
    situation_assumptions: list[str] = Field(default_factory=list)
    situation_constraints: list[str] = Field(default_factory=list)

    # Para 2: Mission
    mission: str

    # Para 3: Execution
    commanders_intent: CommandersIntent
    concept_of_operations: str = Field(
        description="Phased concept of operations"
    )
    tasks_to_subordinates: list[OPORDTask] = Field(default_factory=list)
    coordinating_instructions: list[str] = Field(
        default_factory=list,
        description="Authorities, ROE, reporting, deconfliction"
    )

    # Para 4: Administration & Logistics
    sustainment: str
    logistics_protection: str = Field(default="")

    # Para 5: Command & Signal
    command_relationships: str
    comms_plan: str
    cyber_defense_posture: str = Field(default="")


class Step7PlanDevelopment(BaseModel):
    """JP 5-0 Step 7 artifacts."""

    opord: OPORDSkeleton

    annexes: list[AnnexStub] = Field(
        description="At least Annex B (Intel), Annex C (Ops), Annex K (Comms)"
    )

    sync_matrix: list[SyncMatrixRow] = Field(
        description="Time-phased synchronization matrix, at least 8 rows (D-7..D+2)"
    )

    decision_support_matrix: list[DSMEntry] = Field(
        description="DSM with DP#, trigger, decision, branch, owner"
    )

    assessment_plan: list[AssessmentMeasure] = Field(
        description="MOE/MOP with metric, method, frequency, owner, threshold"
    )


# =============================================================================
# PLANNING STATE: ACCUMULATOR ACROSS ALL STEPS
# =============================================================================

class PlanningState(BaseModel):
    """
    Stateful container that accumulates artifacts across all JPP steps.

    Each step reads prior steps' artifacts and writes its own.
    This is the backbone of the structured planning pipeline.

    Example:
        state = PlanningState(scenario="...")
        state.step1 = run_step1(state)
        state.step2 = run_step2(state)  # reads state.step1.assumptions, etc.
        state.step3 = run_step3(state)  # reads state.step2.ccirs, cog_analysis
        ...
    """
    scenario: str = Field(description="The operational scenario")
    operation_name: str = Field(default="UNNAMED_OP")

    step1: Step1PlanningInitiation | None = None
    step2: Step2MissionAnalysis | None = None
    step3: Step3COADevelopment | None = None
    step4: Step4COAAnalysis | None = None
    step5: Step5COAComparison | None = None
    step6: Step6COAApproval | None = None
    step7: Step7PlanDevelopment | None = None

    def to_json(self) -> str:
        """Serialize the full planning state to JSON."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PlanningState:
        """Deserialize from JSON."""
        return cls.model_validate_json(json_str)

    def completed_steps(self) -> list[int]:
        """Return list of completed step numbers."""
        steps = []
        for i in range(1, 8):
            if getattr(self, f"step{i}") is not None:
                steps.append(i)
        return steps

    def get_step(self, step_num: int) -> BaseModel | None:
        """Get the artifact for a given step number."""
        return getattr(self, f"step{step_num}", None)
