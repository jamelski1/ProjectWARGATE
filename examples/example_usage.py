"""
Example usage of the WARGATE Multi-Agent Joint Staff Planning System.

This script demonstrates how to use WARGATE for various planning scenarios
using the new JointStaffPlanningController orchestration flow.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wargate import (
    # Primary entry point (new controller)
    run_joint_staff_planning,
    JointStaffPlanningController,
    # Legacy entry point
    run_wargate_planning,
    WARGATEOrchestrator,
    # Configuration and agents
    WARGATEConfig,
    StaffRole,
    create_staff_agent,
    create_all_staff_agents,
    # Persona generation
    MilitaryBranch,
    MilitaryPersona,
    generate_random_branch_and_rank,
)


# =============================================================================
# EXAMPLE 1: Basic Usage with New Controller
# =============================================================================

def example_basic_usage():
    """
    Basic usage example using the new JointStaffPlanningController.

    This follows the planning flow:
    1. J2 Intelligence Estimate
    2. J5 + J3 COA Development
    3. Functional Staff Reviews
    4. SJA Legal/Ethics Review
    5. Commander Synthesis
    6. Final Structured Output
    """
    scenario = """
    SITUATION: Hostile forces have seized a critical port city in a partner nation.
    The local government has requested assistance to restore sovereignty.

    ENEMY: Approximately 5,000 irregular fighters supported by foreign advisors,
    possessing MANPADS, ATGMs, and armed technicals. Limited air defense.

    FRIENDLY: One MEU afloat, one Army BCT in neighboring country,
    Air Force assets available from regional bases.

    MISSION: Develop options to restore partner nation control of the port city
    while minimizing civilian casualties and enabling rapid humanitarian aid.

    CONSTRAINTS: Host nation approval required. Minimize infrastructure damage.
    """

    print("Running WARGATE planning with new JointStaffPlanningController...")
    print("=" * 80)

    result = run_joint_staff_planning(
        scenario_text=scenario,
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("FINAL OUTPUT:")
    print("=" * 80)
    print(result)

    return result


# =============================================================================
# EXAMPLE 2: Using the Controller Directly
# =============================================================================

def example_controller_direct():
    """
    Example using JointStaffPlanningController directly for more control.
    """
    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.5,  # Lower temperature for more focused responses
        max_tokens=4096,
        verbose=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    controller = JointStaffPlanningController(config)

    scenario = """
    CYBER/HYBRID SCENARIO: A sophisticated state actor has launched a coordinated
    cyber campaign against critical infrastructure in multiple allied nations.

    SITUATION:
    - Power grid disruptions in three NATO member states
    - Financial system attacks affecting banking operations
    - Disinformation campaign amplifying social divisions
    - Attribution points to a specific state actor with high confidence
    - Conventional forces not directly engaged but posturing along borders

    MISSION: Develop response options that demonstrate resolve, impose costs,
    and deter escalation while maintaining Alliance unity.

    CONSTRAINTS: Response must be proportional. Avoid kinetic escalation.
    Must maintain Alliance consensus.
    """

    result = controller.run(scenario)
    return result


# =============================================================================
# EXAMPLE 3: Step-by-Step Execution
# =============================================================================

def example_step_by_step():
    """
    Example showing how to run planning steps individually for fine control.
    """
    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
    )

    controller = JointStaffPlanningController(config)
    controller.initialize()

    scenario = """
    MARITIME SCENARIO: Adversary naval forces are threatening freedom of
    navigation in international waters. Multiple merchant vessels have been
    harassed. Allies request US-led coalition response.

    ENEMY: Surface action group with destroyer, frigates, and support vessels.
    Submarine threat assessed as moderate. Shore-based anti-ship missiles.

    FRIENDLY: Carrier Strike Group, allied naval forces, regional air support.

    MISSION: Restore freedom of navigation while deterring escalation.
    """

    print("=" * 80)
    print("STEP-BY-STEP PLANNING EXECUTION")
    print("=" * 80)

    # Step 1: J2 Intelligence Estimate
    print("\n>>> STEP 1: J2 INTELLIGENCE ESTIMATE")
    j2_intel = controller.step_j2_intelligence_estimate(scenario)
    print(f"\nIntel Summary Length: {len(j2_intel)} characters")

    # Step 2: J5 + J3 COA Development
    print("\n>>> STEP 2: J5 + J3 COA DEVELOPMENT")
    coa_data = controller.step_coa_development(scenario, j2_intel)
    print(f"\nCOAs Developed: {len(coa_data)}")

    # Step 3: Functional Staff Reviews
    print("\n>>> STEP 3: FUNCTIONAL STAFF REVIEWS")
    staff_estimates = controller.step_functional_staff_reviews(
        scenario, j2_intel, coa_data
    )
    print(f"\nStaff Sections Reviewed: {len(staff_estimates)}")
    for role in staff_estimates.keys():
        print(f"  - {role}")

    # Step 4: SJA Review
    print("\n>>> STEP 4: SJA LEGAL/ETHICS REVIEW")
    sja_review = controller.step_sja_review(
        scenario, j2_intel, coa_data, staff_estimates
    )
    print(f"\nSJA Review Length: {len(sja_review)} characters")

    # Step 5: Commander Synthesis
    print("\n>>> STEP 5: COMMANDER SYNTHESIS")
    synthesis, intent = controller.step_commander_synthesis(
        scenario, j2_intel, coa_data, staff_estimates, sja_review
    )
    print(f"\nSynthesis Length: {len(synthesis)} characters")

    # Step 6: Final Output
    print("\n>>> STEP 6: GENERATING FINAL OUTPUT")
    final_output = controller.generate_final_output(
        scenario, j2_intel, coa_data, staff_estimates, sja_review, synthesis
    )

    return final_output


# =============================================================================
# EXAMPLE 4: Query Single Agent
# =============================================================================

def example_single_agent_query():
    """Example of querying a single staff agent directly."""

    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=False,
    )

    # Create just the J2 Intelligence agent
    j2_agent = create_staff_agent(StaffRole.J2, config)

    # Query the agent directly
    query = """
    Analyze the following intelligence report and provide your assessment:

    SUBJECT: Adversary Force Posture Update

    Recent imagery shows:
    - Additional tank battalion moved to forward assembly area
    - Fuel trucks observed at multiple staging points
    - Increased radio communications intercepts
    - Social media posts by adversary military members indicating movement orders

    Provide:
    1. Assessment of adversary intent
    2. Indicators to monitor
    3. Key intelligence gaps
    """

    print("Querying J2 Intelligence agent directly...")
    response = j2_agent.invoke(query)
    print("\nJ2 Response:")
    print(response)

    return response


# =============================================================================
# EXAMPLE 5: Compare Multiple Scenarios
# =============================================================================

def example_multi_scenario_comparison():
    """Example comparing planning outputs for different scenarios."""

    scenarios = {
        "conventional": """
        Conventional defense scenario: Large-scale ground invasion imminent.
        Enemy: 3 combined arms armies, air superiority contested.
        Friendly: 2 corps, allied air and naval support available.
        Mission: Defend and defeat attacking forces.
        """,

        "counterinsurgency": """
        COIN scenario: Armed insurgency in partner nation.
        Enemy: ~10,000 insurgents, popular support in rural areas.
        Friendly: Host nation forces + US advisory element.
        Mission: Support host nation to defeat insurgency.
        """,

        "humanitarian": """
        Humanitarian crisis: Major earthquake in densely populated region.
        Situation: 500,000 displaced, infrastructure destroyed.
        Friendly: JTF-HA forming with DOD and interagency partners.
        Mission: Provide humanitarian assistance, enable recovery.
        """,
    }

    results = {}

    for scenario_name, scenario_text in scenarios.items():
        print(f"\n{'='*80}")
        print(f"Processing scenario: {scenario_name.upper()}")
        print(f"{'='*80}")

        # Use lower verbosity for comparison run
        result = run_joint_staff_planning(
            scenario_text=scenario_text,
            model_name="gpt-4.1",
            temperature=0.7,
            verbose=False,  # Quiet mode for bulk processing
        )

        results[scenario_name] = result
        print(f"Completed: {scenario_name}")

    return results


# =============================================================================
# EXAMPLE 6: Using Military Personas
# =============================================================================

def example_with_personas():
    """
    Example demonstrating the military persona feature.

    Each staff agent can be assigned a random US military branch, rank, and name.
    This adds branch-specific cultural perspectives to their responses.
    """
    print("=" * 80)
    print("MILITARY PERSONA DEMONSTRATION")
    print("=" * 80)

    # Generate personas for all staff roles with a fixed seed (reproducible)
    print("\n--- Generating Staff Roster (seed=42) ---\n")

    for role in StaffRole:
        persona = generate_random_branch_and_rank(role.value, seed=42)
        print(f"{role.value:20s} -> {persona.full_designation}")
        print(f"                      Branch Culture: {persona.branch.value}")
        print()

    # Create a config with persona_seed for all agents
    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
        persona_seed=42,  # Ensures reproducible personas
    )

    # Create all agents with personas
    staff = create_all_staff_agents(config)

    # Show the created agents with their personas
    print("\n--- Created Staff Agents ---\n")
    for role, agent in staff.items():
        if agent.persona:
            print(f"{agent.persona.full_designation} - {agent._get_role_title()}")
        else:
            print(f"{role.value} (no persona)")

    # Example: Query a single agent with persona
    print("\n--- Querying J2 with Persona ---\n")

    j2_agent = staff[StaffRole.J2]
    print(f"J2 Officer: {j2_agent.persona.full_designation}")
    print(f"Branch perspective: {j2_agent.persona.branch.value}")
    print()

    # Note: In a real scenario, you would run:
    # response = j2_agent.invoke("Provide initial threat assessment")
    # print(response)

    return staff


def example_custom_persona():
    """Example of creating an agent with a specific custom persona."""

    # Generate a specific persona
    persona = generate_random_branch_and_rank("j5_plans", seed=123)
    print(f"Generated persona: {persona.full_designation}")
    print(f"  Branch: {persona.branch.value}")
    print(f"  Rank: {persona.rank_title} ({persona.rank_grade})")
    print(f"  Name: {persona.first_name} {persona.last_name}")

    # Create agent with explicit persona
    config = WARGATEConfig(model_name="gpt-4.1", verbose=True)
    agent = create_staff_agent(StaffRole.J5, config, persona=persona)

    print(f"\nCreated agent: {agent}")
    print(f"System prompt preview (first 500 chars):")
    print(agent.system_prompt[:500] + "...")

    return agent


# =============================================================================
# EXAMPLE 7: Legacy Orchestrator
# =============================================================================

def example_legacy_orchestrator():
    """Example using the legacy WARGATEOrchestrator for comparison."""

    scenario = """
    SITUATION: Border tensions escalating with neighboring state.
    Enemy massing forces, conducting aggressive patrols.
    Mission: Deter aggression, prepare defensive options.
    """

    print("Running with LEGACY WARGATEOrchestrator...")

    result = run_wargate_planning(
        scenario=scenario,
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WARGATE example usage")
    parser.add_argument(
        "--example",
        choices=["basic", "controller", "stepbystep", "single", "multi", "personas", "custom_persona", "legacy"],
        default="basic",
        help="Which example to run"
    )

    args = parser.parse_args()

    examples = {
        "basic": example_basic_usage,
        "controller": example_controller_direct,
        "stepbystep": example_step_by_step,
        "single": example_single_agent_query,
        "multi": example_multi_scenario_comparison,
        "personas": example_with_personas,
        "custom_persona": example_custom_persona,
        "legacy": example_legacy_orchestrator,
    }

    print(f"\n{'#'*80}")
    print(f"#  WARGATE EXAMPLE: {args.example.upper()}")
    print(f"{'#'*80}")

    result = examples[args.example]()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
