"""
wargate_backend.py - Structured Output Backend for Project WARGATE

This module provides a structured output wrapper around the WARGATE
joint staff planning system. Instead of returning a single large text
report, it returns a dictionary with each planning phase as a separate key.

Usage:
    from wargate_backend import run_joint_staff_planning_structured

    result = run_joint_staff_planning_structured(
        scenario_text="Your scenario here...",
        model_name="gpt-4o",
        temperature=0.7
    )

    # Access individual phases
    print(result["intel_estimate"])
    print(result["coa_development"])
    print(result["commander_brief"])
"""

from __future__ import annotations

from typing import Any, TypedDict
from dataclasses import dataclass

# Import from the main wargate module
from wargate import (
    WARGATEConfig,
    JointStaffPlanningController,
    run_joint_staff_planning,
)


class PlanningResult(TypedDict):
    """Structured output from the joint staff planning process."""
    intel_estimate: str
    coa_development: str
    staff_estimates: str
    legal_ethics: str
    commander_brief: str
    full_report: str
    # Raw data for programmatic access
    raw_coa_data: list[dict[str, Any]]
    raw_staff_estimates: dict[str, str]


def format_staff_estimates(estimates: dict[str, str]) -> str:
    """
    Format the staff estimates dictionary into a readable markdown string.

    Args:
        estimates: Dictionary mapping role names to their estimate text

    Returns:
        Formatted markdown string with all staff estimates
    """
    if not estimates:
        return "No staff estimates available."

    sections = []

    # Define display order and friendly names
    role_display = {
        "J1 Personnel": ("J1 - Personnel/Manpower", "👥"),
        "J4 Logistics": ("J4 - Logistics/Sustainment", "📦"),
        "J6 Communications": ("J6 - Communications/C4I", "📡"),
        "Cyber/EW": ("Cyber/Electronic Warfare", "💻"),
        "Fires": ("Fires Integration", "🎯"),
        "Engineer": ("Engineer Support", "🔧"),
        "Protection": ("Force Protection", "🛡️"),
        "PAO": ("Public Affairs/Information", "📢"),
    }

    for role, estimate in estimates.items():
        display_name, icon = role_display.get(role, (role, "📋"))
        sections.append(f"### {icon} {display_name}\n\n{estimate}")

    return "\n\n---\n\n".join(sections)


def format_coa_development(coa_data: list[dict[str, Any]]) -> str:
    """
    Format the COA development data into a readable markdown string.

    Args:
        coa_data: List of COA dictionaries with concepts and details

    Returns:
        Formatted markdown string with COA information
    """
    if not coa_data:
        return "No COA data available."

    sections = []

    for i, coa in enumerate(coa_data, 1):
        sections.append(f"## COA Set {i}")

        if "concepts" in coa:
            sections.append("### Strategic Concepts (J5)\n")
            sections.append(coa["concepts"])

        if "details" in coa:
            sections.append("\n### Execution Details (J3)\n")
            sections.append(coa["details"])

    return "\n\n".join(sections)


def run_joint_staff_planning_structured(
    scenario_text: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    verbose: bool = False,
    api_key: str | None = None,
    persona_seed: int | None = None,
    progress_callback: callable = None,
) -> PlanningResult:
    """
    Run the joint staff planning process and return structured outputs.

    This function wraps the main WARGATE planning controller and captures
    all intermediate outputs, returning them in a structured dictionary
    that's suitable for UI display or further processing.

    Args:
        scenario_text: The operational scenario to plan for
        model_name: OpenAI model name (default: "gpt-4o")
        temperature: LLM temperature 0.0-1.0 (default: 0.7)
        verbose: Enable verbose console output (default: False)
        api_key: OpenAI API key (optional, uses env var if not provided)
        persona_seed: Optional seed for reproducible persona generation
        progress_callback: Optional callback function(step_name, step_number, total_steps)
                          Called after each major planning phase completes

    Returns:
        PlanningResult dictionary containing:
        - intel_estimate: J2 intelligence assessment
        - coa_development: Combined J5/J3 COA concepts and details
        - staff_estimates: Formatted estimates from all functional staff
        - legal_ethics: SJA legal and ethics review
        - commander_brief: Commander's synthesis and intent
        - full_report: Complete concatenated report
        - raw_coa_data: Raw COA data for programmatic access
        - raw_staff_estimates: Raw staff estimates dictionary

    Example:
        >>> result = run_joint_staff_planning_structured(
        ...     "A near-peer adversary threatens allied territory...",
        ...     model_name="gpt-4o",
        ...     temperature=0.7
        ... )
        >>> print(result["intel_estimate"])
        >>> print(result["commander_brief"])
    """
    # Create configuration
    config = WARGATEConfig(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        api_key=api_key,
        persona_seed=persona_seed,
    )

    # Create and initialize the controller
    controller = JointStaffPlanningController(config)
    controller.initialize()
    controller.scenario = scenario_text

    total_steps = 6

    def report_progress(step_name: str, step_num: int):
        if progress_callback:
            progress_callback(step_name, step_num, total_steps)

    # Step 1: J2 Intelligence Estimate
    report_progress("J2 Intelligence Estimate", 1)
    j2_intel = controller.step_j2_intelligence_estimate(scenario_text)

    # Step 2: COA Development (J5 + J3)
    report_progress("COA Development (J5/J3)", 2)
    coa_data = controller.step_coa_development(scenario_text, j2_intel)

    # Step 3: Functional Staff Reviews
    report_progress("Staff Estimates", 3)
    staff_estimates = controller.step_functional_staff_reviews(
        scenario_text, j2_intel, coa_data
    )

    # Step 4: SJA Legal/Ethics Review
    report_progress("Legal/Ethics Review (SJA)", 4)
    sja_review = controller.step_sja_review(
        scenario_text, j2_intel, coa_data, staff_estimates
    )

    # Step 5: Commander Synthesis
    report_progress("Commander's Synthesis", 5)
    synthesis, commanders_intent = controller.step_commander_synthesis(
        scenario_text, j2_intel, coa_data, staff_estimates, sja_review
    )

    # Step 6: Generate Final Output
    report_progress("Generating Final Report", 6)
    full_report = controller.generate_final_output(
        scenario_text, j2_intel, coa_data, staff_estimates, sja_review, synthesis
    )

    # Build and return the structured result
    return PlanningResult(
        intel_estimate=j2_intel,
        coa_development=format_coa_development(coa_data),
        staff_estimates=format_staff_estimates(staff_estimates),
        legal_ethics=sja_review,
        commander_brief=synthesis,
        full_report=full_report,
        raw_coa_data=coa_data,
        raw_staff_estimates=staff_estimates,
    )


# Preserve the original function as an alias
def run_planning(scenario_text: str, **kwargs) -> str:
    """
    Convenience wrapper that returns just the full report string.

    This is equivalent to the original run_joint_staff_planning function.
    """
    result = run_joint_staff_planning_structured(scenario_text, **kwargs)
    return result["full_report"]


# =============================================================================
# EXAMPLE USAGE / TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage
    example_scenario = """
    Operation GenAI - 2030 Scenario

    Background: Global tensions have escalated due to resource scarcity.
    Country Y has developed sophisticated AI-enabled cyber warfare capabilities.

    Situation: Intelligence reports suspicious activity including:
    - Social media posts about a new AI tool with physical-world effects
    - Government data appears copied with no trace in logs
    - Electric grid outage for 30 minutes in a major city
    - 5-minute delay in fighter jet communications
    - Pilots reporting small UFO-like objects

    Task: Develop strategy, plan, and 4 COAs to address these challenges.
    Recommend 1 COA with rationale. Analyze 2nd and 3rd order effects.
    """

    print("Running structured planning (this may take several minutes)...")

    def progress(step_name, step_num, total):
        print(f"  [{step_num}/{total}] {step_name}...")

    result = run_joint_staff_planning_structured(
        example_scenario,
        model_name="gpt-4o",
        temperature=0.7,
        verbose=False,
        progress_callback=progress,
    )

    print("\n" + "="*60)
    print("STRUCTURED OUTPUTS AVAILABLE:")
    print("="*60)

    for key in result.keys():
        if key.startswith("raw_"):
            continue
        content = result[key]
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n### {key.upper().replace('_', ' ')} ###")
        print(preview)
