"""
test_jpp_validators.py - Tests for JPP artifact validators
==========================================================

Tests validators using a sample NATO Article 5 scenario.
Each test constructs a valid (or intentionally deficient) artifact
and verifies the validator catches the right issues.

Run with:
    python -m pytest tests/test_jpp_validators.py -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jpp_artifacts import (
    PlanningState,
    Step1PlanningInitiation,
    Step2MissionAnalysis,
    Step3COADevelopment,
    Step4COAAnalysis,
    Step5COAComparison,
    Step6COAApproval,
    Step7PlanDevelopment,
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
from jpp_validators import (
    validate_step,
    validate_all,
    ValidationResult,
)


# =============================================================================
# FIXTURES: Sample Artifacts
# =============================================================================

SAMPLE_SCENARIO = """
Operation NORTHERN SHIELD — NATO Article 5 Response

Background: Country X (near-peer adversary) has massed 3 combined arms armies
along the eastern border of a NATO ally (Country Z). SIGINT and GEOINT confirm
forward positioning of air defense, artillery, and cyber units. Country X has
conducted aggressive cyber probes against Country Z's power grid and military C2.
Intelligence assesses invasion within 72 hours.

Task: As the Joint Force Commander's staff, develop a defensive plan to deter
aggression and, if deterrence fails, defeat the invasion and restore
territorial integrity.
"""


def _make_assumptions(n: int) -> list[Assumption]:
    return [
        Assumption(
            id=f"A-{i:02d}",
            text=f"Assumption {i}: NATO allies provide air/naval support within 48 hours"
            if i == 1
            else f"Test assumption {i}",
            status="UNVALIDATED",
            basis=f"Based on NATO response timelines" if i == 1 else "Historical precedent",
            impact_if_wrong=f"Requires additional organic air defense" if i == 1 else "Adjusts planning",
        )
        for i in range(1, n + 1)
    ]


def _make_tasks(n: int) -> list[Task]:
    owners = ["J2", "J3", "J5", "J6", "J4", "CYBER_EW", "FIRES"]
    return [
        Task(
            id=f"T-{i:02d}",
            description=f"Task {i}: Conduct intelligence preparation of the battlefield"
            if i == 1
            else f"Task {i} description",
            purpose=f"To identify enemy disposition" if i == 1 else f"Purpose {i}",
            owner=owners[i % len(owners)],
            nlt="D-5",
            success_condition=f"IPB complete with ECOA overlay" if i == 1 else f"Done when {i}",
            task_type="SPECIFIED" if i <= 2 else "IMPLIED" if i <= 4 else "ESSENTIAL",
        )
        for i in range(1, n + 1)
    ]


def _make_ccirs() -> dict[str, list[CCIR]]:
    return {
        "PIR": [
            CCIR(id="PIR-01", text="When will Country X initiate ground offensive?", indicator="Movement of forward units", collection_asset="GEOINT", timeliness="Continuous"),
            CCIR(id="PIR-02", text="What is the main axis of advance?", indicator="Engineer bridging activity", collection_asset="SIGINT/GEOINT", timeliness="D-3"),
            CCIR(id="PIR-03", text="Will Country X employ chemical weapons?", indicator="NBC unit movements", collection_asset="MASINT", timeliness="D-2"),
        ],
        "FFIR": [
            CCIR(id="FFIR-01", text="Has NATO SACEUR approved OPLAN activation?", indicator="SHAPE message traffic", collection_asset="HQ liaison", timeliness="D-5"),
            CCIR(id="FFIR-02", text="Are allied air assets positioned within strike range?", indicator="Air deployment reports", collection_asset="AWACS", timeliness="D-3"),
            CCIR(id="FFIR-03", text="Is Country Z's rail network operational for reinforcement?", indicator="Rail traffic monitoring", collection_asset="OSINT/HUMINT", timeliness="D-4"),
        ],
        "EEFI": [
            CCIR(id="EEFI-01", text="Friendly force disposition and readiness levels", indicator="N/A", collection_asset="OPSEC program", timeliness="Continuous"),
            CCIR(id="EEFI-02", text="NATO OPLAN activation timeline", indicator="N/A", collection_asset="COMSEC", timeliness="Continuous"),
            CCIR(id="EEFI-03", text="Locations of forward logistics bases", indicator="N/A", collection_asset="Physical security", timeliness="Continuous"),
        ],
    }


def _make_cog_analysis() -> list[CenterOfGravity]:
    return [
        CenterOfGravity(
            entity="FRIENDLY",
            cog="NATO combined air superiority and precision strike capability",
            critical_capabilities=["Air superiority over the AO", "Precision guided munitions delivery"],
            critical_requirements=["Air bases within range", "AWACS/tanker support", "Integrated air defense suppression"],
            critical_vulnerabilities=["Limited air base hardening in Country Z", "Dependence on 3 airfields for forward ops"],
        ),
        CenterOfGravity(
            entity="ADVERSARY",
            cog="3rd Combined Arms Army as the main assault force",
            critical_capabilities=["Massed armored breakthrough", "Integrated fires + EW complex"],
            critical_requirements=["Continuous resupply via 2 MSRs", "Air defense umbrella over staging areas"],
            critical_vulnerabilities=["MSR choke points at 2 river crossings", "Centralized C2 vulnerable to decapitation strike"],
        ),
    ]


def _make_risks(n: int) -> list[RiskEntry]:
    return [
        RiskEntry(
            id=f"R-{i:02d}",
            description=f"Risk {i}: Enemy achieves air parity in first 48 hours"
            if i == 1
            else f"Risk {i} description",
            probability="LIKELY" if i == 1 else "POSSIBLE",
            severity="HIGH" if i == 1 else "MEDIUM",
            narrative=f"Country X has invested in advanced IADS; if not suppressed early, contests airspace"
            if i == 1
            else f"Narrative for risk {i}",
            mitigation="Priority SEAD/DEAD in Phase 1" if i == 1 else f"Mitigation {i}",
            owner="J3" if i == 1 else f"J{i+1}",
        )
        for i in range(1, n + 1)
    ]


def _make_running_estimates() -> list[RunningEstimate]:
    return [
        RunningEstimate(section="J2", assessment="Enemy likely to initiate within 72 hrs", key_issues=["ISR gap in northern sector"], recommendations=["Deploy additional SIGINT"]),
        RunningEstimate(section="J3", assessment="Current force posture insufficient for defense-in-depth", key_issues=["Need 2 additional BCTs"], recommendations=["Request SACEUR reserve"]),
        RunningEstimate(section="J4", assessment="Forward logistics bases need 96 hrs to establish", key_issues=["Class V shortfall"], recommendations=["Pre-position ammo at 3 sites"]),
        RunningEstimate(section="J5", assessment="Diplomatic track still active; 48-hr window before escalation", key_issues=["Coalition ROE differences"], recommendations=["Harmonize ROE NLT D-4"]),
        RunningEstimate(section="J6", assessment="C2 network partially degraded by cyber probes", key_issues=["SATCOM backup insufficient"], recommendations=["Deploy tactical mesh network"]),
    ]


# =============================================================================
# STEP 1 TESTS
# =============================================================================

class TestStep1Validator:
    def test_valid_step1(self):
        artifact = Step1PlanningInitiation(
            higher_hq_direction=[
                "SACEUR directs defense of Country Z territory IAW NATO OPLAN 40302",
                "Minimize collateral damage to civilian infrastructure",
                "No first use of nuclear weapons without NAC approval",
            ],
            initial_assumptions=_make_assumptions(5),
            problem_framing="Country X threatens NATO ally Country Z with imminent invasion by 3 combined arms armies. The JFC must rapidly deploy deterrent forces while preparing defensive operations to defeat aggression and restore territorial integrity, all while maintaining alliance cohesion and managing escalation risk.",
            tasks_to_staff=_make_tasks(4),
            initial_information_requirements=[
                "Enemy timeline for ground offensive",
                "Availability of allied air assets",
                "Status of Country Z infrastructure for force reception",
                "Enemy cyber capability assessment",
            ],
            constraints=["Do not strike targets within Country X territory without NAC approval"],
            restraints=["Must maintain freedom of navigation in international waters"],
        )
        result = validate_step(1, artifact)
        assert result.passed, result.summary

    def test_insufficient_assumptions(self):
        artifact = Step1PlanningInitiation(
            higher_hq_direction=["Direction 1", "Direction 2", "Direction 3"],
            initial_assumptions=_make_assumptions(2),  # Only 2, need 5
            problem_framing="Country X threatens an ally. We must defend." * 5,
            tasks_to_staff=_make_tasks(4),
            initial_information_requirements=["Req 1", "Req 2", "Req 3"],
        )
        result = validate_step(1, artifact)
        assert not result.passed
        assert any("initial_assumptions" in e for e in result.errors)

    def test_insufficient_tasks(self):
        artifact = Step1PlanningInitiation(
            higher_hq_direction=["Dir 1", "Dir 2"],
            initial_assumptions=_make_assumptions(5),
            problem_framing="A substantive paragraph about the operational problem that is long enough to pass validation requirements.",
            tasks_to_staff=_make_tasks(2),  # Only 2, need 4
            initial_information_requirements=["Req 1", "Req 2", "Req 3"],
        )
        result = validate_step(1, artifact)
        assert not result.passed
        assert any("tasks_to_staff" in e for e in result.errors)


# =============================================================================
# STEP 2 TESTS
# =============================================================================

class TestStep2Validator:
    def test_valid_step2(self):
        artifact = Step2MissionAnalysis(
            mission_statement=MissionStatement(
                who="JFC NORTH",
                what="defends Country Z",
                when="on order, NLT D-Day",
                where="along the Eastern Defense Line",
                why="to defeat Country X aggression and restore territorial integrity",
                full_statement="JFC NORTH defends Country Z along the Eastern Defense Line on order, NLT D-Day, to defeat Country X aggression and restore territorial integrity.",
            ),
            commanders_intent_draft=CommandersIntent(
                purpose="Defeat Country X's invasion and restore territorial integrity of Country Z",
                method="Establish defense-in-depth along Eastern Defense Line, attrite enemy forces through combined arms defense, then transition to counterattack once enemy culmination point is reached",
                end_state="Country X forces expelled from Country Z, deterrence re-established, alliance cohesion maintained",
                key_tasks=["Establish air superiority", "Defend key terrain at River Line", "Maintain alliance cohesion"],
            ),
            operational_limitations={
                "restraints": ["Must maintain LOCs to Country Z capital", "Cannot employ cluster munitions"],
                "constraints": ["No strikes on Country X territory without NAC approval", "Minimize civilian casualties"],
            },
            assumptions=_make_assumptions(6),
            tasks=_make_tasks(7),
            cog_analysis=_make_cog_analysis(),
            risks=_make_risks(4),
            ccirs=_make_ccirs(),
            isr_outline=[
                ISROutline(pir_id="PIR-01", collection_asset="GEOINT satellite", method="Continuous overhead imagery", timeliness="Every 6 hours"),
                ISROutline(pir_id="PIR-02", collection_asset="SIGINT ground station", method="Direction finding on military nets", timeliness="Continuous"),
                ISROutline(pir_id="PIR-03", collection_asset="MASINT", method="Chemical detection sensors", timeliness="Alert-based"),
            ],
            running_estimates=_make_running_estimates(),
            facts=["Country X has 3 combined arms armies forward-deployed", "Country Z has 1 mechanized division in AO"],
            enemy_coas=["MLCOA: Two-axis advance along highways 1 and 3 with main effort north", "MDCOA: Encirclement via rapid thrust through Gorski Gap"],
        )
        result = validate_step(2, artifact)
        assert result.passed, result.summary

    def test_missing_ccirs(self):
        artifact = Step2MissionAnalysis(
            mission_statement=MissionStatement(who="JFC", what="defends", when="on order", where="east", why="to defeat aggression", full_statement="JFC defends east on order to defeat aggression."),
            commanders_intent_draft=CommandersIntent(purpose="Defeat invasion", method="Defense in depth", end_state="Enemy expelled"),
            operational_limitations={"restraints": [], "constraints": []},
            assumptions=_make_assumptions(5),
            tasks=_make_tasks(5),
            cog_analysis=_make_cog_analysis(),
            risks=_make_risks(3),
            ccirs={"PIR": [CCIR(id="PIR-01", text="When attack?")], "FFIR": [], "EEFI": []},  # Only 1 PIR, 0 FFIR, 0 EEFI
            isr_outline=[ISROutline(pir_id="PIR-01", collection_asset="sat", method="imagery", timeliness="6hr")],
            running_estimates=_make_running_estimates(),
        )
        result = validate_step(2, artifact)
        assert not result.passed
        assert any("FFIR" in e for e in result.errors)
        assert any("EEFI" in e for e in result.errors)

    def test_missing_cog(self):
        artifact = Step2MissionAnalysis(
            mission_statement=MissionStatement(who="JFC", what="defends", when="on order", where="east", why="to defeat aggression", full_statement="JFC defends east on order to defeat aggression."),
            commanders_intent_draft=CommandersIntent(purpose="Defeat invasion", method="Defense in depth", end_state="Enemy expelled"),
            operational_limitations={"restraints": [], "constraints": []},
            assumptions=_make_assumptions(5),
            tasks=_make_tasks(5),
            cog_analysis=[_make_cog_analysis()[0]],  # Only FRIENDLY, missing ADVERSARY
            risks=_make_risks(3),
            ccirs=_make_ccirs(),
            isr_outline=[ISROutline(pir_id="PIR-01", collection_asset="sat", method="imagery", timeliness="6hr")],
            running_estimates=_make_running_estimates(),
        )
        result = validate_step(2, artifact)
        assert not result.passed
        assert any("ADVERSARY" in e for e in result.errors)


# =============================================================================
# STEP 3 TESTS
# =============================================================================

def _make_coa(coa_id: str, name: str, valid: bool = True) -> COA:
    phases = [
        PhaseDetail(phase_name="Phase 0 - Shape", description="Pre-conflict shaping ops", key_tasks=["Deploy ISR", "Establish C2"], main_effort="Intel collection", cyber_integration="DCO hardening"),
        PhaseDetail(phase_name="Phase 1 - Deter", description="Show of force", key_tasks=["Forward deploy BCTs", "Activate air defense"], main_effort="Forward presence", cyber_integration="OCO preparation"),
        PhaseDetail(phase_name="Phase 2 - Seize Initiative", description="Defensive operations", key_tasks=["Engage enemy at phase line", "SEAD/DEAD"], main_effort="Eastern Defense Line", cyber_integration="OCO against enemy C2"),
    ]
    return COA(
        id=coa_id,
        name=name,
        operational_approach="Defense-in-depth with phased counterattack along two lines of effort",
        phases=phases,
        main_effort="Eastern Defense Line — 2nd BCT sector",
        supporting_efforts=["Northern flank security by allied forces", "Deep fires against enemy staging areas"],
        task_organization="2nd BCT supported by allied air wing; 3rd BCT supporting as reserve" if valid else "",
        command_relationships="2nd BCT supported; allied air wing supporting",
        cyber_integration="Phase 0: DCO hardening of C2 networks. Phase 1: OCO preparation targeting enemy C2 nodes. Phase 2: Active OCO against enemy IADS and C2." if valid else "",
        sustainment_concept="Forward logistics base at Gridref XY1234 with 7-day stockpile; MSR via Highway 3 with alternate via Rail Line 7" if valid else "",
        comms_c2_concept="PACE: Primary SATCOM, Alternate HF, Contingency tactical mesh, Emergency runner. JOC at Location Alpha with alternate at Location Bravo." if valid else "",
        key_assumptions=["Allied air support arrives within 48 hours"],
        distinguishing_features="Emphasizes forward defense over depth",
        initial_risk=RiskEntry(id="R-COA1", description="Risk of forward elements being overrun", probability="POSSIBLE", severity="HIGH", narrative="If allied air delayed, forward BCTs exposed", mitigation="Pre-positioned fallback positions"),
    )


class TestStep3Validator:
    def test_valid_step3(self):
        artifact = Step3COADevelopment(
            coas=[
                _make_coa("COA-1", "Iron Shield"),
                _make_coa("COA-2", "Swift Hammer"),
                _make_coa("COA-3", "Deep Envelopment"),
            ],
            development_rationale="COAs derived from COG analysis targeting adversary CV at MSR choke points and dependence on centralized C2.",
        )
        result = validate_step(3, artifact)
        assert result.passed, result.summary

    def test_too_few_coas(self):
        artifact = Step3COADevelopment(
            coas=[_make_coa("COA-1", "Only One")],
        )
        result = validate_step(3, artifact)
        assert not result.passed
        assert any(">=3" in e for e in result.errors)

    def test_missing_cyber_integration(self):
        bad_coa = _make_coa("COA-1", "Bad Cyber", valid=False)
        artifact = Step3COADevelopment(
            coas=[bad_coa, _make_coa("COA-2", "Good 2"), _make_coa("COA-3", "Good 3")],
        )
        result = validate_step(3, artifact)
        assert not result.passed
        assert any("cyber_integration" in e for e in result.errors)


# =============================================================================
# STEP 4 TESTS
# =============================================================================

def _make_wargame_result(coa_id: str, coa_name: str) -> WargameResult:
    return WargameResult(
        coa_id=coa_id,
        coa_name=coa_name,
        wargame_method="BELT",
        arc_table=[
            ARCEntry(phase="Phase 1", friendly_action="Deploy BCTs to Eastern Defense Line", enemy_reaction="Accelerate staging", friendly_counteraction="Intensify ISR on staging areas", notes="Triggers PIR-01"),
            ARCEntry(phase="Phase 2", friendly_action="Engage enemy main body at phase line", enemy_reaction="Attempt flanking maneuver north", friendly_counteraction="Commit reserve BCT to northern sector", notes="DP-02 trigger"),
            ARCEntry(phase="Phase 2", friendly_action="SEAD/DEAD against enemy IADS", enemy_reaction="Disperse mobile SAMs", friendly_counteraction="Switch to hunt-and-kill with F-35s", notes="Degraded enemy air picture"),
            ARCEntry(phase="Phase 2", friendly_action="OCO against enemy C2 networks", enemy_reaction="Switch to backup comms", friendly_counteraction="SIGINT collection on backup nets", notes="Enables targeting of backup C2"),
        ],
        decision_points=[
            DecisionPoint(id="DP-01", trigger="Enemy crosses Phase Line Alpha", decision="Commit main defense or delay?", branch="If delay: fall back to Phase Line Bravo", sequel="Counterattack once culmination reached", owner="JFC", associated_ccir="PIR-01"),
            DecisionPoint(id="DP-02", trigger="Enemy flanking force detected in northern sector", decision="Commit reserve BCT north?", branch="If yes: thin southern sector; if no: accept risk north", owner="JFC", associated_ccir="PIR-02"),
            DecisionPoint(id="DP-03", trigger="Allied air assets arrive in theater", decision="Transition to offensive air operations?", branch="If delayed: extend defensive phase", owner="JFACC", associated_ccir="FFIR-02"),
            DecisionPoint(id="DP-04", trigger="Enemy culmination assessed", decision="Initiate counterattack?", branch="If not culminated: maintain defense", sequel="Phase 3 counterattack operations", owner="JFC", associated_ccir="PIR-01"),
        ],
        strengths=["Leverages terrain for defense", "Allied air superiority once deployed"],
        weaknesses=["48-hr window without full air support", "Logistics vulnerability on MSR"],
        updated_risk=RiskEntry(id="R-WG-01", description="Forward forces overrun before allied air arrives", probability="POSSIBLE", severity="HIGH", narrative="Wargaming showed 48-hr gap is critical", mitigation="Pre-position AMD assets; request early air deployment"),
        modifications=["Add AMD battery to forward positions", "Pre-position Class V at 2 additional sites"],
    )


class TestStep4Validator:
    def test_valid_step4(self):
        artifact = Step4COAAnalysis(
            wargame_method="BELT",
            wargame_results=[
                _make_wargame_result("COA-1", "Iron Shield"),
                _make_wargame_result("COA-2", "Swift Hammer"),
            ],
            updated_ccirs=[CCIR(id="PIR-04", text="Has enemy deployed electronic warfare assets forward?", indicator="SIGINT detection of EW emitters", collection_asset="ELINT", timeliness="Continuous")],
            collection_gaps=["No HUMINT source in enemy rear area", "Limited MASINT coverage of northern approach"],
        )
        result = validate_step(4, artifact)
        assert result.passed, result.summary

    def test_insufficient_decision_points(self):
        wr = _make_wargame_result("COA-1", "Test")
        wr.decision_points = wr.decision_points[:2]  # Only 2, need 4
        artifact = Step4COAAnalysis(
            wargame_method="BOX",
            wargame_results=[wr],
        )
        result = validate_step(4, artifact)
        assert not result.passed
        assert any("decision points" in e for e in result.errors)


# =============================================================================
# STEP 5 TESTS
# =============================================================================

class TestStep5Validator:
    def test_valid_step5(self):
        criteria = [
            ComparisonCriterion(name="Mission Accomplishment", weight=5, description="Likelihood of achieving objectives"),
            ComparisonCriterion(name="Risk to Force", weight=4, description="Potential casualties and losses"),
            ComparisonCriterion(name="Risk to Mission", weight=4, description="Probability of mission failure"),
            ComparisonCriterion(name="Logistics Feasibility", weight=3, description="Sustainment supportability"),
            ComparisonCriterion(name="C2 Resilience", weight=3, description="Command and control robustness"),
            ComparisonCriterion(name="Legal Compliance", weight=3, description="LOAC and ROE compliance"),
        ]
        artifact = Step5COAComparison(
            criteria=criteria,
            scoring_matrix=[
                COAScore(coa_id="COA-1", coa_name="Iron Shield", scores={c.name: 7.0 for c in criteria}, weighted_total=154.0, advantages=["Strong defense"], disadvantages=["Slow transition"]),
                COAScore(coa_id="COA-2", coa_name="Swift Hammer", scores={c.name: 6.0 for c in criteria}, weighted_total=132.0, advantages=["Fast counterattack"], disadvantages=["Logistics strain"]),
            ],
            sensitivity_notes=["If allied air delayed >72hrs, COA-2 becomes infeasible", "If enemy uses chemical weapons, COA-1 protection advantage increases"],
            recommended_coa="COA-1",
            recommendation_justification="COA-1 scores highest overall (154 vs 132) with particular advantage in risk management and C2 resilience. Wargaming confirmed its defensive approach is more robust against the 48-hour allied air gap.",
        )
        result = validate_step(5, artifact)
        assert result.passed, result.summary

    def test_too_few_criteria(self):
        artifact = Step5COAComparison(
            criteria=[
                ComparisonCriterion(name="Effectiveness", weight=5),
                ComparisonCriterion(name="Risk", weight=3),
            ],  # Only 2, need 6
            scoring_matrix=[
                COAScore(coa_id="COA-1", coa_name="Test", scores={"Effectiveness": 7, "Risk": 6}, weighted_total=53, advantages=[], disadvantages=[]),
                COAScore(coa_id="COA-2", coa_name="Test2", scores={"Effectiveness": 6, "Risk": 7}, weighted_total=51, advantages=[], disadvantages=[]),
            ],
            recommended_coa="COA-1",
            recommendation_justification="COA-1 scores highest overall with strong effectiveness rating.",
        )
        result = validate_step(5, artifact)
        assert not result.passed
        assert any(">=6" in e for e in result.errors)


# =============================================================================
# PLANNING STATE TESTS
# =============================================================================

class TestPlanningState:
    def test_serialization_roundtrip(self):
        state = PlanningState(scenario=SAMPLE_SCENARIO, operation_name="NORTHERN SHIELD")
        state.step1 = Step1PlanningInitiation(
            higher_hq_direction=["SACEUR directs defense", "Minimize collateral damage", "No first nuclear use"],
            initial_assumptions=_make_assumptions(5),
            problem_framing="Country X threatens NATO ally with imminent invasion requiring rapid force deployment and defense planning.",
            tasks_to_staff=_make_tasks(4),
            initial_information_requirements=["Enemy timeline", "Allied availability", "Infrastructure status"],
        )

        json_str = state.to_json()
        restored = PlanningState.from_json(json_str)

        assert restored.operation_name == "NORTHERN SHIELD"
        assert restored.step1 is not None
        assert len(restored.step1.initial_assumptions) == 5
        assert restored.step2 is None

    def test_completed_steps(self):
        state = PlanningState(scenario="test")
        assert state.completed_steps() == []

        state.step1 = Step1PlanningInitiation(
            higher_hq_direction=["a", "b"],
            initial_assumptions=_make_assumptions(5),
            problem_framing="x" * 60,
            tasks_to_staff=_make_tasks(4),
            initial_information_requirements=["a", "b", "c"],
        )
        assert state.completed_steps() == [1]

    def test_validate_all(self):
        state = PlanningState(scenario="test")
        state.step1 = Step1PlanningInitiation(
            higher_hq_direction=["a", "b", "c"],
            initial_assumptions=_make_assumptions(5),
            problem_framing="A substantive paragraph about the operational problem that is definitely long enough to pass the fifty-character minimum requirement.",
            tasks_to_staff=_make_tasks(4),
            initial_information_requirements=["a", "b", "c"],
        )
        results = validate_all(state)
        assert len(results) == 1
        assert results[0].step == 1
        assert results[0].passed


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
