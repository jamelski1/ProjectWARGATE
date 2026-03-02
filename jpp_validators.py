"""
jpp_validators.py - Artifact Validators for JP 5-0 Aligned JPP Pipeline
========================================================================

Validates that each JPP step's structured output contains required
elements per JP 5-0 doctrine. Validators check presence and count
thresholds; if validation fails, the caller should regenerate the step.

Usage:
    from jpp_validators import validate_step, validate_all, ValidationResult

    result = validate_step(2, step2_artifact)
    if not result.passed:
        print(result.errors)  # What's missing
        print(result.warnings)  # What's weak
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jpp_artifacts import (
    PlanningState,
    Step1PlanningInitiation,
    Step2MissionAnalysis,
    Step3COADevelopment,
    Step4COAAnalysis,
    Step5COAComparison,
    Step6COAApproval,
    Step7PlanDevelopment,
)


@dataclass
class ValidationResult:
    """Result from validating a JPP step artifact."""
    step: int
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = [f"Step {self.step}: {status}"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


# =============================================================================
# STEP 1 VALIDATOR
# =============================================================================

def validate_step1(artifact: Step1PlanningInitiation) -> ValidationResult:
    errors = []
    warnings = []

    if len(artifact.higher_hq_direction) < 2:
        errors.append(
            f"higher_hq_direction: need >=2 bullets, got {len(artifact.higher_hq_direction)}"
        )

    if len(artifact.initial_assumptions) < 5:
        errors.append(
            f"initial_assumptions: need >=5, got {len(artifact.initial_assumptions)}"
        )

    if not artifact.problem_framing or len(artifact.problem_framing.strip()) < 50:
        errors.append("problem_framing: must be a substantive paragraph (>=50 chars)")

    if len(artifact.tasks_to_staff) < 4:
        errors.append(
            f"tasks_to_staff: need >=4 (J2/J3/J5/J6 minimum), got {len(artifact.tasks_to_staff)}"
        )

    # Check staff sections covered
    owners = {t.owner.upper() for t in artifact.tasks_to_staff}
    for required in ["J2", "J3", "J5", "J6"]:
        if not any(required in o for o in owners):
            warnings.append(f"tasks_to_staff: no task assigned to {required}")

    if len(artifact.initial_information_requirements) < 3:
        errors.append(
            f"initial_information_requirements: need >=3, got {len(artifact.initial_information_requirements)}"
        )

    return ValidationResult(
        step=1,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 2 VALIDATOR
# =============================================================================

def validate_step2(artifact: Step2MissionAnalysis) -> ValidationResult:
    errors = []
    warnings = []

    # Mission statement
    if not artifact.mission_statement.full_statement or len(artifact.mission_statement.full_statement) < 20:
        errors.append("mission_statement: full_statement is missing or too short")

    # Commander's intent
    if not artifact.commanders_intent_draft.purpose:
        errors.append("commanders_intent_draft: purpose is empty")
    if not artifact.commanders_intent_draft.end_state:
        errors.append("commanders_intent_draft: end_state is empty")

    # Assumptions
    if len(artifact.assumptions) < 5:
        warnings.append(
            f"assumptions: only {len(artifact.assumptions)}, should carry forward >=5"
        )

    # Tasks
    task_types = {t.task_type for t in artifact.tasks}
    if "SPECIFIED" not in task_types:
        warnings.append("tasks: no SPECIFIED tasks found")
    if "IMPLIED" not in task_types:
        warnings.append("tasks: no IMPLIED tasks found")
    if "ESSENTIAL" not in task_types:
        warnings.append("tasks: no ESSENTIAL tasks found")
    if len(artifact.tasks) < 5:
        errors.append(f"tasks: need >=5 (specified+implied+essential), got {len(artifact.tasks)}")

    # COG analysis
    if len(artifact.cog_analysis) < 2:
        errors.append(
            f"cog_analysis: need >=2 (FRIENDLY + ADVERSARY), got {len(artifact.cog_analysis)}"
        )
    else:
        entities = {c.entity for c in artifact.cog_analysis}
        if "FRIENDLY" not in entities:
            errors.append("cog_analysis: missing FRIENDLY center of gravity")
        if "ADVERSARY" not in entities:
            errors.append("cog_analysis: missing ADVERSARY center of gravity")
        for cog in artifact.cog_analysis:
            if len(cog.critical_capabilities) == 0:
                warnings.append(f"cog_analysis ({cog.entity}): no critical capabilities")
            if len(cog.critical_requirements) == 0:
                warnings.append(f"cog_analysis ({cog.entity}): no critical requirements")
            if len(cog.critical_vulnerabilities) == 0:
                warnings.append(f"cog_analysis ({cog.entity}): no critical vulnerabilities")

    # CCIRs
    pir_count = len(artifact.ccirs.get("PIR", []))
    ffir_count = len(artifact.ccirs.get("FFIR", []))
    eefi_count = len(artifact.ccirs.get("EEFI", []))

    if pir_count < 3:
        errors.append(f"ccirs.PIR: need >=3, got {pir_count}")
    if ffir_count < 3:
        errors.append(f"ccirs.FFIR: need >=3, got {ffir_count}")
    if eefi_count < 3:
        errors.append(f"ccirs.EEFI: need >=3, got {eefi_count}")

    # ISR outline
    if len(artifact.isr_outline) < 3:
        warnings.append(
            f"isr_outline: only {len(artifact.isr_outline)} entries, should map to each PIR"
        )

    # Running estimates
    if len(artifact.running_estimates) < 3:
        errors.append(
            f"running_estimates: need >=3 (J2/J3/J4/J5/J6), got {len(artifact.running_estimates)}"
        )
    else:
        sections = {e.section.upper() for e in artifact.running_estimates}
        for required in ["J2", "J3"]:
            if not any(required in s for s in sections):
                warnings.append(f"running_estimates: missing {required}")

    # Risks
    if len(artifact.risks) < 3:
        warnings.append(f"risks: only {len(artifact.risks)}, recommend >=3")

    return ValidationResult(
        step=2,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 3 VALIDATOR
# =============================================================================

def validate_step3(artifact: Step3COADevelopment) -> ValidationResult:
    errors = []
    warnings = []

    if len(artifact.coas) < 3:
        errors.append(f"coas: need >=3 distinct COAs, got {len(artifact.coas)}")

    for coa in artifact.coas:
        prefix = f"COA '{coa.id}'"

        if len(coa.phases) < 2:
            errors.append(f"{prefix}: need >=2 phases, got {len(coa.phases)}")

        if not coa.task_organization or len(coa.task_organization.strip()) < 10:
            errors.append(f"{prefix}: task_organization is missing or too short")

        if not coa.cyber_integration or len(coa.cyber_integration.strip()) < 10:
            errors.append(f"{prefix}: cyber_integration is missing or too short")

        if not coa.sustainment_concept or len(coa.sustainment_concept.strip()) < 10:
            errors.append(f"{prefix}: sustainment_concept is missing or too short")

        if not coa.comms_c2_concept or len(coa.comms_c2_concept.strip()) < 10:
            errors.append(f"{prefix}: comms_c2_concept is missing or too short")

        if not coa.main_effort:
            warnings.append(f"{prefix}: main_effort is empty")

        if len(coa.supporting_efforts) == 0:
            warnings.append(f"{prefix}: no supporting_efforts listed")

        # Check phases have key_tasks
        for phase in coa.phases:
            if len(phase.key_tasks) == 0:
                warnings.append(
                    f"{prefix}, {phase.phase_name}: no key_tasks listed"
                )

    return ValidationResult(
        step=3,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 4 VALIDATOR
# =============================================================================

def validate_step4(artifact: Step4COAAnalysis) -> ValidationResult:
    errors = []
    warnings = []

    if not artifact.wargame_method:
        errors.append("wargame_method: must be declared")

    if len(artifact.wargame_results) < 1:
        errors.append("wargame_results: need at least 1 COA wargamed")

    total_dps = 0
    total_arc = 0

    for wr in artifact.wargame_results:
        prefix = f"Wargame '{wr.coa_id}'"

        if len(wr.arc_table) < 3:
            errors.append(
                f"{prefix}: ARC table needs >=3 entries, got {len(wr.arc_table)}"
            )
        total_arc += len(wr.arc_table)

        if len(wr.decision_points) < 4:
            errors.append(
                f"{prefix}: need >=4 decision points, got {len(wr.decision_points)}"
            )
        total_dps += len(wr.decision_points)

        if len(wr.strengths) == 0:
            warnings.append(f"{prefix}: no strengths identified")
        if len(wr.weaknesses) == 0:
            warnings.append(f"{prefix}: no weaknesses identified")

    if total_dps < 4:
        errors.append(
            f"Total decision points across all COAs: {total_dps}, need >=4"
        )

    if total_arc < 3:
        errors.append(f"Total ARC entries across all COAs: {total_arc}, need >=3")

    return ValidationResult(
        step=4,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 5 VALIDATOR
# =============================================================================

def validate_step5(artifact: Step5COAComparison) -> ValidationResult:
    errors = []
    warnings = []

    if len(artifact.criteria) < 6:
        errors.append(
            f"criteria: need >=6 weighted criteria, got {len(artifact.criteria)}"
        )

    if len(artifact.scoring_matrix) < 2:
        errors.append(
            f"scoring_matrix: need scores for >=2 COAs, got {len(artifact.scoring_matrix)}"
        )

    # Check that scores reference all criteria
    criterion_names = {c.name for c in artifact.criteria}
    for score in artifact.scoring_matrix:
        missing = criterion_names - set(score.scores.keys())
        if missing:
            warnings.append(
                f"COA '{score.coa_id}' missing scores for: {missing}"
            )

    if not artifact.recommended_coa:
        errors.append("recommended_coa: must specify which COA is recommended")

    if not artifact.recommendation_justification or len(artifact.recommendation_justification) < 30:
        errors.append("recommendation_justification: must be substantive (>=30 chars)")

    if len(artifact.sensitivity_notes) == 0:
        warnings.append("sensitivity_notes: should note what changes the recommendation")

    return ValidationResult(
        step=5,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 6 VALIDATOR
# =============================================================================

def validate_step6(artifact: Step6COAApproval) -> ValidationResult:
    errors = []
    warnings = []

    if not artifact.decision_statement or len(artifact.decision_statement) < 20:
        errors.append("decision_statement: must be substantive")

    if not artifact.selected_coa_id:
        errors.append("selected_coa_id: must specify which COA is approved")

    if not artifact.finalized_intent.purpose:
        errors.append("finalized_intent: purpose is empty")
    if not artifact.finalized_intent.end_state:
        errors.append("finalized_intent: end_state is empty")

    pir_count = len(artifact.finalized_ccirs.get("PIR", []))
    ffir_count = len(artifact.finalized_ccirs.get("FFIR", []))
    eefi_count = len(artifact.finalized_ccirs.get("EEFI", []))

    if pir_count < 3:
        warnings.append(f"finalized_ccirs.PIR: only {pir_count}, should have >=3")
    if ffir_count < 3:
        warnings.append(f"finalized_ccirs.FFIR: only {ffir_count}, should have >=3")
    if eefi_count < 3:
        warnings.append(f"finalized_ccirs.EEFI: only {eefi_count}, should have >=3")

    if not artifact.risk_acceptance or len(artifact.risk_acceptance) < 20:
        errors.append("risk_acceptance: must include a substantive risk acceptance statement")

    if not artifact.planning_guidance or len(artifact.planning_guidance) < 30:
        errors.append("planning_guidance: must include planning guidance for OPORD development")

    return ValidationResult(
        step=6,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# STEP 7 VALIDATOR
# =============================================================================

def validate_step7(artifact: Step7PlanDevelopment) -> ValidationResult:
    errors = []
    warnings = []

    # OPORD sections
    opord = artifact.opord
    if not opord.situation_adversary or len(opord.situation_adversary) < 30:
        errors.append("opord.situation_adversary: must be substantive")
    if not opord.situation_friendly or len(opord.situation_friendly) < 30:
        errors.append("opord.situation_friendly: must be substantive")
    if not opord.mission or len(opord.mission) < 20:
        errors.append("opord.mission: must be present")
    if not opord.commanders_intent.purpose:
        errors.append("opord.commanders_intent: purpose is empty")
    if not opord.concept_of_operations or len(opord.concept_of_operations) < 50:
        errors.append("opord.concept_of_operations: must be substantive")
    if len(opord.tasks_to_subordinates) < 2:
        errors.append(
            f"opord.tasks_to_subordinates: need >=2, got {len(opord.tasks_to_subordinates)}"
        )
    else:
        for task in opord.tasks_to_subordinates:
            if not task.purpose:
                warnings.append(f"Task to '{task.unit}': missing purpose")
            if not task.success_condition:
                warnings.append(f"Task to '{task.unit}': missing success_condition")

    if len(opord.coordinating_instructions) < 2:
        warnings.append("opord.coordinating_instructions: should have >=2")

    # Annexes
    annex_ids = {a.annex_id.upper().replace(" ", "") for a in artifact.annexes}
    for required in ["ANNEXB", "ANNEXC", "ANNEXK"]:
        if required not in annex_ids:
            errors.append(f"annexes: missing {required}")

    # Sync matrix
    if len(artifact.sync_matrix) < 8:
        errors.append(
            f"sync_matrix: need >=8 rows (D-7..D+2), got {len(artifact.sync_matrix)}"
        )

    # Decision support matrix
    if len(artifact.decision_support_matrix) < 3:
        errors.append(
            f"decision_support_matrix: need >=3 entries, got {len(artifact.decision_support_matrix)}"
        )

    # Assessment plan
    if len(artifact.assessment_plan) < 3:
        errors.append(
            f"assessment_plan: need >=3 MOE/MOP entries, got {len(artifact.assessment_plan)}"
        )
    else:
        types = {m.measure_type for m in artifact.assessment_plan}
        if "MOE" not in types:
            warnings.append("assessment_plan: no MOE (Measure of Effectiveness) entries")
        if "MOP" not in types:
            warnings.append("assessment_plan: no MOP (Measure of Performance) entries")

    return ValidationResult(
        step=7,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# DISPATCH
# =============================================================================

_VALIDATORS = {
    1: validate_step1,
    2: validate_step2,
    3: validate_step3,
    4: validate_step4,
    5: validate_step5,
    6: validate_step6,
    7: validate_step7,
}

_ARTIFACT_TYPES = {
    1: Step1PlanningInitiation,
    2: Step2MissionAnalysis,
    3: Step3COADevelopment,
    4: Step4COAAnalysis,
    5: Step5COAComparison,
    6: Step6COAApproval,
    7: Step7PlanDevelopment,
}


def validate_step(step_num: int, artifact: Any) -> ValidationResult:
    """
    Validate a single step's artifact.

    Args:
        step_num: JPP step number (1-7)
        artifact: The step's Pydantic model instance

    Returns:
        ValidationResult with pass/fail, errors, and warnings
    """
    validator = _VALIDATORS.get(step_num)
    if validator is None:
        return ValidationResult(
            step=step_num,
            passed=False,
            errors=[f"No validator for step {step_num}"],
        )
    return validator(artifact)


def validate_all(state: PlanningState) -> list[ValidationResult]:
    """
    Validate all completed steps in a PlanningState.

    Returns:
        List of ValidationResult, one per completed step
    """
    results = []
    for step_num in state.completed_steps():
        artifact = state.get_step(step_num)
        if artifact is not None:
            results.append(validate_step(step_num, artifact))
    return results


def get_artifact_type(step_num: int) -> type:
    """Return the Pydantic model class for a given step number."""
    return _ARTIFACT_TYPES[step_num]
