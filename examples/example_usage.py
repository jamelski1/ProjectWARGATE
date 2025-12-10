"""
Example usage of the WARGATE Multi-Agent Joint Staff Planning System.

This script demonstrates how to use WARGATE for various planning scenarios.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wargate import (
    run_wargate_planning,
    WARGATEConfig,
    WARGATEOrchestrator,
    StaffRole,
    create_staff_agent,
)


def example_basic_usage():
    """Basic usage example with default settings."""

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

    print("Running WARGATE planning with basic settings...")
    result = run_wargate_planning(
        scenario=scenario,
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("FINAL OUTPUT:")
    print("=" * 80)
    print(result)

    return result


def example_custom_config():
    """Example using custom configuration."""

    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.5,  # Lower temperature for more focused responses
        max_tokens=4096,
        verbose=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    orchestrator = WARGATEOrchestrator(config)

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

    result = orchestrator.run(scenario)
    return result


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
        result = run_wargate_planning(
            scenario=scenario_text,
            model_name="gpt-4.1",
            temperature=0.7,
            verbose=False,  # Quiet mode for bulk processing
        )

        results[scenario_name] = result
        print(f"Completed: {scenario_name}")

    return results


def example_phased_planning():
    """Example showing how to run planning phases individually."""

    config = WARGATEConfig(
        model_name="gpt-4.1",
        temperature=0.7,
        verbose=True,
    )

    orchestrator = WARGATEOrchestrator(config)
    orchestrator.initialize()

    scenario = """
    MARITIME SCENARIO: Adversary naval forces are threatening freedom of
    navigation in international waters. Multiple merchant vessels have been
    harassed. Allies request US-led coalition response.

    ENEMY: Surface action group with destroyer, frigates, and support vessels.
    Submarine threat assessed as moderate. Shore-based anti-ship missiles.

    FRIENDLY: Carrier Strike Group, allied naval forces, regional air support.

    MISSION: Restore freedom of navigation while deterring escalation.
    """

    # Run individual phases and examine outputs
    print("\n=== PHASE 1: MISSION ANALYSIS ===")
    mission_analysis = orchestrator.phase_mission_analysis(scenario)
    print("\nCommander's Guidance:")
    print(mission_analysis["commander_guidance"])

    print("\n=== PHASE 2: COA DEVELOPMENT ===")
    coa_development = orchestrator.phase_coa_development(mission_analysis)
    print("\nCOA Concepts:")
    print(coa_development["coa_concepts"])

    print("\n=== PHASE 3: COA ANALYSIS ===")
    coa_analysis = orchestrator.phase_coa_analysis(coa_development)
    print("\nRed Team Analysis:")
    print(coa_analysis["red_team_analysis"])

    # Continue with remaining phases...
    coa_comparison = orchestrator.phase_coa_comparison(coa_analysis)
    coa_selection = orchestrator.phase_coa_selection(coa_comparison)
    plan_development = orchestrator.phase_plan_development(coa_selection)

    final_output = orchestrator.generate_final_output()
    return final_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WARGATE example usage")
    parser.add_argument(
        "--example",
        choices=["basic", "custom", "single", "multi", "phased"],
        default="basic",
        help="Which example to run"
    )

    args = parser.parse_args()

    examples = {
        "basic": example_basic_usage,
        "custom": example_custom_config,
        "single": example_single_agent_query,
        "multi": example_multi_scenario_comparison,
        "phased": example_phased_planning,
    }

    # Run selected example
    print(f"\nRunning example: {args.example}")
    print("=" * 80)

    result = examples[args.example]()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
