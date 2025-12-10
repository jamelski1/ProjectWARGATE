"""
wargate_orchestration.py - Multi-Agent Meeting Orchestration for Project WARGATE

This module implements rich, multi-turn dialogue orchestration for the Joint Planning
Process (JPP). It enables:

1. MULTI-AGENT PARTICIPATION: 10+ staff agents participate in each meeting, with
   J2, J3, J5, J4, J6, Cyber/EW, Fires, Engineer, Protection, SJA, and PAO/IO.

2. MULTI-TURN CONVERSATIONS: Each agent speaks 2-3+ times per meeting, with
   back-and-forth dialogue, challenges, refinements, and collaborative decision-making.

3. INCREMENTAL RENDERING: Conversations unfold in real-time via callbacks, allowing
   Streamlit to render dialogue bubbles as they appear.

4. 4-STEP MEETING FLOW: Each JPP phase follows:
   (a) Staff Meeting - Multi-agent dialogue with accumulated context
   (b) Slide Creation - Generate structured slide content from transcript
   (c) Brief Commander - Staff presents to commander with Q&A
   (d) Commander Guidance - Commander issues direction for next phase

Usage:
    from wargate_orchestration import MeetingOrchestrator, DialogueTurn

    orchestrator = MeetingOrchestrator(config)

    # Run a staff meeting with incremental callbacks
    def on_turn(turn: DialogueTurn):
        render_dialogue_bubble(turn)

    meeting_result = orchestrator.run_staff_meeting(
        phase=JPPPhase.MISSION_ANALYSIS,
        scenario=scenario_text,
        prior_context=prior_outputs,
        on_turn_callback=on_turn
    )

Author: Project WARGATE Team
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Generator, TypedDict, Literal
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

# Import from main wargate module
from wargate import (
    WARGATEConfig,
    StaffRole,
    StaffAgent,
    create_staff_agent,
    MilitaryPersona,
    generate_random_branch_and_rank,
    STAFF_SYSTEM_PROMPTS,
)


# =============================================================================
# RETRY LOGIC FOR TRANSIENT NETWORK ERRORS
# =============================================================================

def invoke_with_retry(
    agent: Any,
    prompt: str,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> str:
    """
    Invoke an agent with automatic retry on transient network errors.

    Handles common network issues like:
    - RemoteProtocolError (connection closed unexpectedly)
    - ConnectionError
    - Timeout errors

    Args:
        agent: The agent to invoke (must have .invoke method)
        prompt: The prompt to send
        max_retries: Maximum number of retry attempts (default 3)
        initial_delay: Initial delay in seconds, doubles each retry (default 2.0)

    Returns:
        The agent's response string

    Raises:
        The original exception if all retries fail
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return agent.invoke(prompt)
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e).lower()

            # Check if this is a transient network error worth retrying
            is_transient = any([
                "remoteprotocolerror" in error_name.lower(),
                "connectionerror" in error_name.lower(),
                "timeout" in error_name.lower(),
                "peer closed connection" in error_msg,
                "incomplete chunked read" in error_msg,
                "connection reset" in error_msg,
                "network" in error_msg,
            ])

            if is_transient and attempt < max_retries:
                print(f"[RETRY] Network error on attempt {attempt + 1}: {error_name}")
                print(f"[RETRY] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                last_exception = e
            else:
                # Not a transient error or out of retries
                raise

    # Should not reach here, but raise last exception if we do
    if last_exception:
        raise last_exception


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class DialogueTurn(TypedDict):
    """Represents a single turn in a staff meeting dialogue."""
    speaker: str           # Full name: "COL Smith"
    role: str              # Staff role key: "j3_operations"
    role_display: str      # Display name: "J3 - Operations"
    branch: str            # Service branch: "US Army"
    rank: str              # Rank abbreviation: "COL"
    text: str              # The dialogue content (may contain markdown)
    turn_number: int       # Sequential turn number in the meeting
    is_commander: bool     # Whether this is the commander speaking


class MeetingResult(TypedDict):
    """Result from a staff meeting."""
    phase_name: str
    turns: list[DialogueTurn]
    transcript: str           # Full text transcript for slide generation
    decisions: list[str]      # Key decisions made
    products: dict[str, Any]  # Structured products (e.g., COA summaries)


class SlideContent(TypedDict):
    """Content for a single slide."""
    title: str
    bullets: list[str]
    notes: str


class BriefResult(TypedDict):
    """Result from a commander brief."""
    turns: list[DialogueTurn]
    questions_asked: list[str]
    clarifications: list[str]


class GuidanceResult(TypedDict):
    """Result from commander guidance."""
    guidance_text: str
    priority_tasks: list[str]
    guidance_by_section: dict[str, str]


class PhaseResult(TypedDict):
    """Complete result from a JPP phase."""
    phase_name: str
    meeting: MeetingResult
    slides: list[SlideContent]
    brief: BriefResult
    guidance: GuidanceResult


# =============================================================================
# JPP PHASE DEFINITIONS
# =============================================================================

class JPPPhase(Enum):
    """Joint Planning Process phases."""
    PLANNING_INITIATION = 1
    MISSION_ANALYSIS = 2
    COA_DEVELOPMENT = 3
    COA_ANALYSIS = 4
    COA_COMPARISON = 5
    COA_APPROVAL = 6
    PLAN_DEVELOPMENT = 7


# Phase-specific meeting configurations
PHASE_CONFIGS = {
    JPPPhase.PLANNING_INITIATION: {
        "name": "Planning Initiation",
        "topic": "Establish planning organization, review strategic guidance, and frame the problem",
        "lead_agents": [StaffRole.J5, StaffRole.J3, StaffRole.J2],
        "key_outputs": ["Problem Statement", "Planning Timeline", "Initial CCIRs", "Key Assumptions"],
        "min_turns": 15,
        "focus_areas": [
            "Strategic guidance interpretation",
            "Problem framing and operational environment",
            "Planning constraints and restraints",
            "Initial staff organization",
        ],
    },
    JPPPhase.MISSION_ANALYSIS: {
        "name": "Mission Analysis",
        "topic": "Analyze the mission, develop facts/assumptions, and produce restated mission",
        "lead_agents": [StaffRole.J2, StaffRole.J3, StaffRole.J5],
        "key_outputs": ["METT-TC Analysis", "Restated Mission", "CCIRs", "Assumptions"],
        "min_turns": 20,
        "focus_areas": [
            "METT-TC analysis (Mission, Enemy, Terrain, Troops, Time, Civil)",
            "Facts and assumptions",
            "Specified and implied tasks",
            "Constraints and limitations",
            "Restated mission development",
        ],
    },
    JPPPhase.COA_DEVELOPMENT: {
        "name": "COA Development",
        "topic": "Develop multiple distinct courses of action",
        "lead_agents": [StaffRole.J5, StaffRole.J3, StaffRole.FIRES],
        "key_outputs": ["COA Statements", "COA Sketches", "Initial Risk Assessment"],
        "min_turns": 25,
        "focus_areas": [
            "Brainstorming operational approaches",
            "Defining main effort and supporting efforts",
            "Phasing and synchronization",
            "Resource requirements per COA",
            "Ensuring COAs are FEASIBLE, ACCEPTABLE, SUITABLE, DISTINGUISHABLE",
        ],
    },
    JPPPhase.COA_ANALYSIS: {
        "name": "COA Analysis & Wargaming",
        "topic": "Wargame each COA against enemy COAs to identify strengths, weaknesses, and modifications",
        "lead_agents": [StaffRole.J2, StaffRole.J3, StaffRole.FIRES],
        "key_outputs": ["Wargame Results", "Decision Points", "Critical Events", "Modified COAs"],
        "min_turns": 25,
        "focus_areas": [
            "Action-reaction-counteraction wargaming",
            "Identifying decision points",
            "Critical events and synchronization",
            "Branches and sequels",
            "Risk identification",
        ],
    },
    JPPPhase.COA_COMPARISON: {
        "name": "COA Comparison",
        "topic": "Compare COAs against evaluation criteria to identify preferred COA",
        "lead_agents": [StaffRole.J5, StaffRole.J3, StaffRole.SJA],
        "key_outputs": ["Comparison Matrix", "Advantages/Disadvantages", "Staff Recommendation"],
        "min_turns": 20,
        "focus_areas": [
            "Evaluation criteria development",
            "Scoring each COA against criteria",
            "Risk comparison",
            "Staff recommendation formulation",
        ],
    },
    JPPPhase.COA_APPROVAL: {
        "name": "COA Approval",
        "topic": "Present COAs to commander for decision and approval",
        "lead_agents": [StaffRole.J5, StaffRole.J3, StaffRole.COMMANDER],
        "key_outputs": ["Decision Brief", "Commander's Decision", "Refined Intent"],
        "min_turns": 15,
        "focus_areas": [
            "Final COA presentation",
            "Risk acceptance discussion",
            "Commander's decision rationale",
            "Refined commander's intent",
        ],
    },
    JPPPhase.PLAN_DEVELOPMENT: {
        "name": "Plan/Order Development",
        "topic": "Develop detailed plan or order based on approved COA",
        "lead_agents": [StaffRole.J3, StaffRole.J5, StaffRole.J4],
        "key_outputs": ["Draft OPORD", "Annexes Outline", "Synchronization Matrix"],
        "min_turns": 25,
        "focus_areas": [
            "OPORD format and content",
            "Annex development by staff section",
            "Synchronization and integration",
            "Transition to execution",
        ],
    },
}


# =============================================================================
# AGENT SPEAKING ORDER & PROMPTS
# =============================================================================

# Full list of participating agents (order determines speaking priority)
MEETING_PARTICIPANTS = [
    StaffRole.J5,        # Plans - often facilitates/opens
    StaffRole.J2,        # Intel - sets threat picture
    StaffRole.J3,        # Operations - execution focus
    StaffRole.J4,        # Logistics - sustainment
    StaffRole.J6,        # C4I - communications
    StaffRole.CYBER_EW,  # Cyber/EW - information domain
    StaffRole.FIRES,     # Fires integration
    StaffRole.ENGINEER,  # Mobility/survivability
    StaffRole.PROTECTION,# Force protection
    StaffRole.SJA,       # Legal/ethics
    StaffRole.PAO,       # Public affairs/IO
]


# =============================================================================
# AGENT PERSONALITY TRAITS (ENHANCED FOR NATURAL DIALOGUE)
# =============================================================================
# Each agent has distinct personality characteristics, domain-specific vocabulary,
# and speaking quirks that make them sound like real military officers, not
# templated AI responses.
#
# These traits are injected into dialogue prompts to create more natural,
# varied conversations with polite disagreement and realistic cross-talk.

AGENT_PERSONALITIES: dict[StaffRole, dict] = {
    StaffRole.J2: {
        "traits": [
            "Analytical and data-driven - never satisfied until intel gaps are addressed",
            "Plays devil's advocate instinctively - 'what if we're wrong about this'",
            "Occasionally blunt when the staff is wishful-thinking the threat away",
            "Brisk, clipped speech focused on the 'air picture' and collection coverage",
        ],
        "speech_style": "Sharp, precise, sometimes skeptical. Interrupts politely to challenge assumptions. Uses phrases like 'the intel doesn't support that' or 'we're making a big assumption here'.",
        "domain_slang": [
            "VUL windows (vulnerable time windows for collection)",
            "ISR coverage, collection assets, pattern of life",
            "air picture, ground truth, indications and warnings",
            "ECOA (enemy COA), most likely vs most dangerous",
            "SIGINT take, HUMINT reporting, overhead imagery",
        ],
        "quirks": "Gets visibly annoyed when intel gaps are hand-waved. Occasionally mutters 'we've been wrong about this before...'",
        "pet_peeves": "Overconfidence, planning to best-case scenarios, treating enemy as predictable",
    },
    StaffRole.J3: {
        "traits": [
            "Direct, action-oriented - laser-focused on execution",
            "Impatient with theoretical discussions that don't lead to decisions",
            "Practical problem-solver who cuts through complexity to executable actions",
            "Calm under pressure but expects the same from everyone",
        ],
        "speech_style": "Concise, decisive, no-nonsense. 'Okay, so what do we do about it?' Gets to the point fast. May cut off tangents with 'let's stay focused'.",
        "domain_slang": [
            "scheme of maneuver, main effort, supporting effort",
            "battle rhythm, decision points, triggers",
            "fires integration, deconfliction, synchronization",
            "FRAGO (fragmentary order), WARNO (warning order)",
            "H-hour, D-day, phase lines, objectives",
        ],
        "quirks": "Drums fingers when discussions get too theoretical. Known for 'let's park that and move on' when debates stall.",
        "pet_peeves": "Analysis paralysis, vague guidance, beautiful plans that can't execute",
    },
    StaffRole.J5: {
        "traits": [
            "Big-picture strategist who thinks in campaigns, not just battles",
            "Loves exploring branches, sequels, and second-order effects",
            "Sometimes theatrical when framing strategic risk",
            "Patient listener who synthesizes across staff sections",
        ],
        "speech_style": "Thoughtful, deliberate, occasionally grandiose. 'Let's think about what happens six months after this...' Frames everything in strategic context.",
        "domain_slang": [
            "branches and sequels, transition conditions",
            "operational approach, lines of effort",
            "end state, strategic objectives, campaign design",
            "phase transitions, culminating points",
            "termination criteria, consolidation of gains",
        ],
        "quirks": "Often references historical parallels. Says 'this is where campaigns go wrong' more than colleagues appreciate.",
        "pet_peeves": "Tunnel vision on immediate fight, ignoring post-conflict, not connecting tactical to strategic",
    },
    StaffRole.J4: {
        "traits": [
            "Pragmatic skeptic - the voice of logistical reality",
            "Always mentally calculating tonnage, throughput, and consumption rates",
            "The 'yes, but can we actually sustain that?' person in every meeting",
            "Quietly proud when ops plans fail because they ignored logistics",
        ],
        "speech_style": "Grounded, sometimes bluntly pessimistic. 'Look, I'm not trying to be negative, but...' Speaks in concrete numbers and timelines.",
        "domain_slang": [
            "throughput, lift capacity, sustainment tail",
            "CLs (classes of supply), fuel consumption, ammo basic loads",
            "MSRs (main supply routes), forward operating bases, LOGPAC",
            "reconstitution, maintenance floats, operational tempo",
            "days of supply, stockage objectives, reachback",
        ],
        "quirks": "Carries a mental calculator. Known for 'we burn X tons of fuel per day, so...' math on the fly.",
        "pet_peeves": "Magical thinking about sustainment, 'logistics will figure it out', ignoring supply chain fragility",
    },
    StaffRole.J6: {
        "traits": [
            "Technical but mission-focused - understands comms enables everything",
            "Worries about single points of failure before anyone else does",
            "Bridges the gap between tech complexity and operational reality",
            "Calm explainer who makes complex systems understandable",
        ],
        "speech_style": "Technical but accessible. 'If we lose this node, here's what breaks...' Translates tech into mission impact.",
        "domain_slang": [
            "PACE plan (Primary, Alternate, Contingency, Emergency)",
            "network architecture, single points of failure, redundancy",
            "spectrum management, electromagnetic environment",
            "C2 nodes, data links, SATCOM, line of sight",
            "degraded operations, denied/contested environments",
        ],
        "quirks": "Gets quiet when people assume 'comms will just work'. Always asks about backup plans.",
        "pet_peeves": "No PACE plan, ignoring EW threats to C2, assuming perfect network connectivity",
    },
    StaffRole.CYBER_EW: {
        "traits": [
            "Tech-savvy with sardonic humor about legacy systems",
            "Sees the information domain as equally important as physical",
            "Frustrated that cyber/EW is often bolted on at the end",
            "Speaks in technical terms but can translate when needed",
        ],
        "speech_style": "Modern, sometimes wry. 'Yeah, about that... their networks are probably already inside ours.' Uses tech analogies that occasionally confuse senior officers.",
        "domain_slang": [
            "spectrum dominance, jamming, electronic attack",
            "SIGINT, EW effects, C2 nodes, access and authorities",
            "offensive cyber, defensive cyber, network defense",
            "adversary TTPs, indicators of compromise, persistence",
            "zero-days, supply chain compromise, air-gapped networks",
        ],
        "quirks": "Slightly sardonic about 'we'll just cyber them'. Cares deeply about our own vulnerabilities.",
        "pet_peeves": "Treating cyber as magic, not understanding access timelines, ignoring our own exposure",
    },
    StaffRole.FIRES: {
        "traits": [
            "Confident and precise - clear about what fires can and cannot do",
            "Obsessed with synchronization, timing, and deconfliction",
            "Strong opinions on proportionality and collateral damage",
            "Numbers-oriented when discussing effects",
        ],
        "speech_style": "Assertive, precise. 'We can put steel on that target in X minutes if we have...' Speaks in capabilities and authorities.",
        "domain_slang": [
            "fire support coordination measures, no-fire areas",
            "time on target, priority targets, high-value targets",
            "joint fires, close air support, deep fires",
            "BDA (battle damage assessment), desired effects",
            "kinetic, non-kinetic, collateral damage estimate",
        ],
        "quirks": "Always doing mental time-distance calculations. Gets animated about proper synchronization.",
        "pet_peeves": "Vague targeting guidance, unrealistic expectations, not integrating fires early",
    },
    StaffRole.ENGINEER: {
        "traits": [
            "Problem-solver who believes 'there's always a way'",
            "Thinks in mobility, counter-mobility, and survivability",
            "Often underestimated until plans hit terrain reality",
            "Practical, can-do attitude with realistic constraints",
        ],
        "speech_style": "Practical, solution-oriented. 'We can make that work, but we'll need...' or 'Here's the problem with the terrain...' Speaks in construction timelines.",
        "domain_slang": [
            "mobility, counter-mobility, survivability",
            "obstacle belts, breach sites, gap crossing",
            "route clearance, combat engineering, general engineering",
            "hardening, fortifications, protective positions",
            "OAKOC (Observation, Avenues of Approach, Key Terrain, Obstacles, Cover)",
        ],
        "quirks": "Always looking at the map for terrain problems. Says 'engineers make or break timelines' regularly.",
        "pet_peeves": "Ignoring terrain analysis, assuming unlimited engineer assets, not factoring construction time",
    },
    StaffRole.PROTECTION: {
        "traits": [
            "Constantly scanning for what could go wrong",
            "Focused on critical asset defense and force preservation",
            "Appears pessimistic but is actually just thorough",
            "The 'what if they hit us here?' voice in planning",
        ],
        "speech_style": "Risk-focused, methodical. 'What if they go after our...' Thinks in vulnerabilities and mitigations.",
        "domain_slang": [
            "critical asset list, defended asset list",
            "AMD (air and missile defense), early warning",
            "force protection, physical security, OPSEC",
            "CBRN defense, personnel recovery, CSAR",
            "vulnerability assessment, threat vectors",
        ],
        "quirks": "Always game-planning adversary reactions. Known for uncomfortable 'but what about...' questions.",
        "pet_peeves": "Assuming enemy won't adapt, reactive posture, not protecting critical nodes",
    },
    StaffRole.SJA: {
        "traits": [
            "Precise, risk-aware - doesn't sugarcoat legal landmines",
            "Focused on keeping operations within legal bounds",
            "Not afraid to say 'that's problematic' clearly and early",
            "Careful with words, sometimes frustratingly so for operators",
        ],
        "speech_style": "Measured, precise. 'The legal risk here is...' or 'Before we go further, we need to consider...' Never casual about compliance.",
        "domain_slang": [
            "LOAC (Law of Armed Conflict), IHL, ROE (Rules of Engagement)",
            "distinction, proportionality, military necessity",
            "collateral damage, protected sites, no-strike list",
            "authorities, title 10 vs title 50, legal sufficiency",
            "targeting review, ethics of autonomous systems",
        ],
        "quirks": "Gently but firmly stops discussions that are heading toward questionable territory.",
        "pet_peeves": "Being consulted after decisions are made, cavalier attitude toward LOAC, 'ask forgiveness' mentality",
    },
    StaffRole.PAO: {
        "traits": [
            "Thinks about perception and narrative constantly",
            "Bridges military operations with public/political reality",
            "Focused on the story we're telling and how it will land",
            "Sensitive to domestic and international audiences",
        ],
        "speech_style": "Audience-aware, framing-conscious. 'How is this going to play in...' or 'The story we're telling needs to be...' Always considering optics.",
        "domain_slang": [
            "narrative, messaging, information environment",
            "public affairs, strategic communications, themes",
            "domestic audience, international audience, partner perception",
            "counter-disinformation, adversary narratives",
            "talking points, media engagement, embed considerations",
        ],
        "quirks": "Always asking 'what's the headline?' Thinks about adversary propaganda response.",
        "pet_peeves": "Ignoring information dimension, reactive messaging, dismissing perception as 'just PR'",
    },
    StaffRole.COMMANDER: {
        "traits": [
            "Decisive leader who balances competing priorities",
            "Listens carefully but expects crisp recommendations",
            "Comfortable accepting risk with eyes open",
            "Expects staff to disagree openly in planning, then execute together",
        ],
        "speech_style": "Authoritative but engaged. Asks pointed questions. 'What's your recommendation?' and 'What are we missing?' Decisive when it's time.",
        "domain_slang": [
            "commander's intent, end state, key tasks",
            "main effort, risk to force, risk to mission",
            "staff recommendation, decision point, constraint",
            "priority of effort, acceptance of risk",
        ],
        "quirks": "Cuts off long explanations with 'bottom line it for me'. Values staff who disagree clearly.",
        "pet_peeves": "Staff not making recommendations, buried leads, options without risk assessment",
    },
}


# Common instructions for natural speech (applied to all agents)
# These are CRITICAL for making agents sound like real officers, not templated AI
NATURAL_SPEECH_INSTRUCTIONS = """
CRITICAL SPEAKING RULES - FOLLOW THESE EXACTLY:

1. BANNED PHRASES - NEVER START WITH:
   - "As the J2...", "As the Fires Officer...", "As the Operations Officer..."
   - "As a reminder...", "It's important to note that...", "I want to emphasize..."
   - "From my perspective as...", "In my role as...", "Speaking as the..."
   - Any role introduction whatsoever. Everyone knows who you are.

2. INSTEAD, START WITH:
   - Your main point immediately: "We're looking at a 72-hour window before..."
   - A direct reaction: "That's going to be a problem because..."
   - A question: "Have we confirmed the enemy's capability to...?"
   - A concern: "I'm worried we're underestimating..."

3. SPEAK NATURALLY - LIKE A REAL MEETING:
   - Vary your sentence length. Mix short punchy statements with longer explanations.
   - Use professional informality when appropriate:
     * "This is going to be ugly if we don't fix X..."
     * "Look, I'm not trying to derail this, but..."
     * "We've seen this before and it didn't work..."
     * "Honestly, I'm not fully convinced that..."
   - Reference what others said: "Building on what J3 said...", "I disagree with..."
   - Ask direct questions: "J4, can you actually sustain that tempo?"

4. BE A PERSON, NOT A DOCUMENT:
   - Admit uncertainty: "I'm not 100% on this, but..."
   - Disagree constructively: "I see it differently...", "That concerns me because..."
   - Show mild frustration appropriately: "We keep running into this problem..."
   - Use domain slang naturally - don't explain every acronym

5. STRUCTURE YOUR RESPONSE:
   - ONE clear summary sentence first (max 25 words) - your main point
   - 2-4 paragraphs of substance (150-350 words total)
   - End with a clear recommendation, question, or point of contention

6. ENGAGE WITH THE ROOM:
   - Respond to previous speakers by name or role
   - Challenge assumptions others have made
   - Build coalitions: "I think J2's point reinforces what I'm seeing..."
   - Create productive tension: "But here's where I disagree..."
"""


def get_personality_prompt(role: StaffRole) -> str:
    """
    Get personality-specific prompt additions for an agent.

    This now includes domain slang and personality quirks to make
    each agent sound distinctly like a real military officer with
    their own speech patterns and concerns.
    """
    personality = AGENT_PERSONALITIES.get(role)
    if not personality:
        return ""

    traits = personality.get("traits", [])
    style = personality.get("speech_style", "")
    domain_slang = personality.get("domain_slang", [])
    quirks = personality.get("quirks", "")
    peeves = personality.get("pet_peeves", "")

    # Format traits as bullet list
    traits_text = "\n".join(f"  - {t}" for t in traits) if traits else "  - Professional military officer"

    # Format domain slang
    slang_text = ", ".join(domain_slang[:3]) if domain_slang else ""  # Just first 3 to keep it concise

    return f"""
YOUR PERSONALITY AND SPEAKING STYLE:
{traits_text}

SPEECH PATTERN: {style}

DOMAIN VOCABULARY - Use naturally (don't over-explain):
{slang_text}

PERSONALITY QUIRKS: {quirks}

TOPICS THAT GET YOU ENGAGED: {peeves}

Remember: Let these traits influence HOW you speak, but stay professional. You're not a caricature -
you're a real officer who happens to have these tendencies. Vary your tone based on the conversation.
"""


# Meeting dialogue prompts for each phase
def get_meeting_prompt(
    phase: JPPPhase,
    role: StaffRole,
    turn_number: int,
    scenario: str,
    prior_context: str,
    conversation_so_far: str,
    persona: MilitaryPersona,
) -> str:
    """Generate the prompt for an agent's turn in a meeting."""

    phase_config = PHASE_CONFIGS[phase]
    personality_prompt = get_personality_prompt(role)

    # Determine the agent's behavior based on turn number
    if turn_number <= 3:
        turn_guidance = """This is an OPENING TURN. You should:
- Lead with your key concern or initial assessment
- Raise important questions for the group
- Reference relevant data from your domain"""
    elif turn_number <= 8:
        turn_guidance = """This is a DEVELOPMENT TURN. You should:
- Build on or push back on what others have said
- Challenge assumptions or offer alternatives
- Propose specific solutions or options"""
    else:
        turn_guidance = """This is a REFINEMENT TURN. You should:
- Synthesize discussion into concrete recommendations
- Identify remaining issues or risks
- Propose decision points or confirm coordination"""

    prompt = f"""You are {persona.full_designation}, the {role.value.replace('_', ' ').title()}.

You are in a staff meeting for the "{phase_config['name']}" phase of the Joint Planning Process.

{persona.culture_description}
{personality_prompt}
{NATURAL_SPEECH_INSTRUCTIONS}

=== MEETING CONTEXT ===
Topic: {phase_config['topic']}
Key Outputs: {', '.join(phase_config['key_outputs'])}
Focus Areas: {', '.join(phase_config['focus_areas'])}

=== SCENARIO ===
{scenario}

=== PRIOR PLANNING CONTEXT ===
{prior_context if prior_context else "This is the first phase; no prior context."}

=== CONVERSATION SO FAR ===
{conversation_so_far if conversation_so_far else "[Meeting just started - you are among the first to speak]"}

=== YOUR TURN (Turn #{turn_number}) ===
{turn_guidance}

RESPONSE FORMAT:
- Start with ONE summary sentence (your main point in ≤25 words)
- Then 2-4 paragraphs of detail (150-350 words total)
- Reference what others said and respond to them
- End with a clear point, question, or recommendation

NOW SPEAK YOUR TURN:"""

    return prompt


def get_brief_prompt(
    phase: JPPPhase,
    role: StaffRole,
    slide_content: str,
    persona: MilitaryPersona,
    questions_so_far: str,
) -> str:
    """Generate prompt for briefing the commander."""

    phase_config = PHASE_CONFIGS[phase]
    personality_prompt = get_personality_prompt(role)

    return f"""You are {persona.full_designation}, the {role.value.replace('_', ' ').title()}.

You are briefing the Commander on your section's findings from the {phase_config['name']} phase.
{personality_prompt}

SPEAKING RULES:
- DO NOT start with "As the J2..." or similar role introductions
- Start with your bottom line up front (one sentence summary)
- Be direct and speak naturally like you're in the room
- If responding to a question, give a direct answer first, then explain

=== YOUR SLIDE CONTENT ===
{slide_content}

=== QUESTIONS/DISCUSSION SO FAR ===
{questions_so_far if questions_so_far else "[You are presenting first]"}

BRIEFING GUIDELINES:
1. Lead with ONE summary sentence of your main finding
2. Present key points concisely but with substance
3. Highlight risks, concerns, or outstanding issues
4. If responding to a question, be direct and specific

Your briefing should be 100-200 words - executive summary style.

DELIVER YOUR BRIEF:"""


def get_commander_guidance_prompt(
    phase: JPPPhase,
    meeting_summary: str,
    brief_summary: str,
    scenario: str,
) -> str:
    """Generate prompt for commander to issue guidance."""

    phase_config = PHASE_CONFIGS[phase]
    next_phase = JPPPhase(phase.value + 1) if phase.value < 7 else None
    next_phase_name = PHASE_CONFIGS[next_phase]["name"] if next_phase else "Plan Execution"
    commander_personality = get_personality_prompt(StaffRole.COMMANDER)

    return f"""You are the Commander presiding over the {phase_config['name']} phase.

Your staff has just completed their meeting and briefed you on their findings.
{commander_personality}

SPEAKING RULES:
- Speak naturally and directly, like you're in the room
- Start with your overall assessment in one sentence
- Be decisive - make clear calls where needed
- Reference specific staff contributions when relevant

=== SCENARIO ===
{scenario}

=== STAFF MEETING SUMMARY ===
{meeting_summary}

=== STAFF BRIEF SUMMARY ===
{brief_summary}

=== YOUR TASK ===
Issue Commander's Guidance for the next phase ({next_phase_name}).

Your guidance should:
1. ASSESS the situation and staff work (one sentence bottom line first)
2. DECIDE on outstanding issues requiring your decision
3. PRIORITIZE the next phase's focus areas
4. DIRECT specific sections on what you need from them
5. ACCEPT RISK where appropriate and explain briefly

FORMAT:
1. COMMANDER'S ASSESSMENT: [Start with one-sentence bottom line]
2. DECISIONS: [What you're deciding now]
3. PRIORITY TASKS: [What sections should focus on]
4. RISK GUIDANCE: [Risks you're accepting and why]
5. INTENT FOR NEXT PHASE: [How to proceed]

Be substantive and specific. Your guidance shapes the next phase.

ISSUE YOUR GUIDANCE:"""


# =============================================================================
# MEETING ORCHESTRATOR
# =============================================================================

class MeetingOrchestrator:
    """
    Orchestrates multi-agent meetings for the Joint Planning Process.

    This class manages:
    - Creating and caching staff agents
    - Running multi-turn staff meetings
    - Generating slide content from transcripts
    - Running commander briefs
    - Issuing commander guidance

    Attributes:
        config: WARGATEConfig with model and persona settings
        agents: Dict of StaffRole -> StaffAgent (cached)
        personas: Dict of StaffRole -> MilitaryPersona (cached)
    """

    def __init__(self, config: WARGATEConfig | None = None):
        """Initialize the orchestrator with configuration."""
        self.config = config or WARGATEConfig()
        self.agents: dict[StaffRole, StaffAgent] = {}
        self.personas: dict[StaffRole, MilitaryPersona] = {}
        self._llm: ChatOpenAI | None = None

    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            )
        return self._llm

    def get_or_create_agent(self, role: StaffRole) -> StaffAgent:
        """Get a cached agent or create a new one."""
        if role not in self.agents:
            self.agents[role] = create_staff_agent(role, self.config)
            self.personas[role] = self.agents[role].persona
        return self.agents[role]

    def get_persona(self, role: StaffRole) -> MilitaryPersona:
        """Get the persona for a role, creating agent if needed."""
        if role not in self.personas:
            self.get_or_create_agent(role)
        return self.personas[role]

    def _format_turn_for_transcript(self, turn: DialogueTurn) -> str:
        """Format a turn for inclusion in the conversation transcript."""
        return f"**{turn['speaker']} ({turn['role_display']}):** {turn['text']}"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a direct LLM call (for slide generation, guidance, etc.) with retry."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        max_retries = 3
        delay = 2.0
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                error_name = type(e).__name__
                error_msg = str(e).lower()
                is_transient = any([
                    "remoteprotocolerror" in error_name.lower(),
                    "connectionerror" in error_name.lower(),
                    "timeout" in error_name.lower(),
                    "peer closed connection" in error_msg,
                    "incomplete chunked read" in error_msg,
                ])
                if is_transient and attempt < max_retries:
                    print(f"[RETRY] LLM call error on attempt {attempt + 1}: {error_name}")
                    time.sleep(delay)
                    delay *= 2
                    last_exception = e
                else:
                    raise

        if last_exception:
            raise last_exception

    # =========================================================================
    # STAFF MEETING
    # =========================================================================

    def run_staff_meeting(
        self,
        phase: JPPPhase,
        scenario: str,
        prior_context: str = "",
        on_turn_callback: Callable[[DialogueTurn], None] | None = None,
        turn_delay: float = 0.3,
    ) -> MeetingResult:
        """
        Run a multi-agent staff meeting for a JPP phase.

        This method orchestrates a meeting where 10+ agents participate with
        multiple speaking turns each. The conversation builds upon itself,
        with later speakers responding to and building on earlier contributions.

        Args:
            phase: The JPP phase for this meeting
            scenario: The operational scenario text
            prior_context: Context from prior phases (transcripts, decisions)
            on_turn_callback: Optional callback invoked after each turn for live rendering
            turn_delay: Delay in seconds between turns (for visual effect)

        Returns:
            MeetingResult with turns, transcript, decisions, and products
        """
        phase_config = PHASE_CONFIGS[phase]
        min_turns = phase_config["min_turns"]

        turns: list[DialogueTurn] = []
        transcript_parts: list[str] = []

        # Determine speaking order - prioritize lead agents, then cycle through all
        lead_agents = phase_config["lead_agents"]
        other_agents = [r for r in MEETING_PARTICIPANTS if r not in lead_agents]

        # Build speaking schedule: leads first, then mix
        # Each agent should speak 2-3 times minimum
        speaking_schedule: list[StaffRole] = []

        # Round 1: All agents speak once (leads first)
        speaking_schedule.extend(lead_agents)
        speaking_schedule.extend(other_agents)

        # Round 2: Leads respond, then key contributors
        speaking_schedule.extend(lead_agents[:2])
        speaking_schedule.extend([StaffRole.J4, StaffRole.FIRES, StaffRole.CYBER_EW])
        speaking_schedule.extend(lead_agents[2:] if len(lead_agents) > 2 else [])

        # Round 3: Cross-talk and synthesis
        speaking_schedule.extend([StaffRole.J2, StaffRole.J3, StaffRole.SJA])
        speaking_schedule.extend([StaffRole.J5, StaffRole.ENGINEER, StaffRole.PROTECTION])

        # Ensure minimum turns
        while len(speaking_schedule) < min_turns:
            # Add more dialogue from key agents
            for role in lead_agents + [StaffRole.J4, StaffRole.FIRES]:
                speaking_schedule.append(role)
                if len(speaking_schedule) >= min_turns:
                    break

        # Execute the meeting
        for turn_idx, role in enumerate(speaking_schedule):
            turn_number = turn_idx + 1

            # Get agent and persona
            agent = self.get_or_create_agent(role)
            persona = self.get_persona(role)

            # Build conversation context (last 10 turns for context window)
            recent_transcript = "\n\n".join(transcript_parts[-10:])

            # Generate the prompt
            prompt = get_meeting_prompt(
                phase=phase,
                role=role,
                turn_number=turn_number,
                scenario=scenario,
                prior_context=prior_context,
                conversation_so_far=recent_transcript,
                persona=persona,
            )

            # Get agent response
            response = invoke_with_retry(agent, prompt)

            # Create the turn record
            turn = DialogueTurn(
                speaker=persona.short_designation,
                role=role.value,
                role_display=role.value.replace('_', ' ').title().replace('Oic', ''),
                branch=persona.branch.value,
                rank=persona.rank_abbrev,
                text=response,
                turn_number=turn_number,
                is_commander=role == StaffRole.COMMANDER,
            )

            turns.append(turn)
            transcript_parts.append(self._format_turn_for_transcript(turn))

            # Invoke callback for live rendering
            if on_turn_callback:
                on_turn_callback(turn)
                if turn_delay > 0:
                    time.sleep(turn_delay)

        # Build full transcript
        full_transcript = "\n\n".join(transcript_parts)

        # Extract key decisions (simple heuristic - look for decision language)
        decisions = self._extract_decisions(full_transcript)

        return MeetingResult(
            phase_name=phase_config["name"],
            turns=turns,
            transcript=full_transcript,
            decisions=decisions,
            products={},
        )

    def _extract_decisions(self, transcript: str) -> list[str]:
        """Extract key decisions from a transcript (heuristic)."""
        decisions = []
        decision_markers = [
            "we will", "we should", "I recommend", "the staff recommends",
            "our assessment is", "decision:", "recommendation:",
        ]

        lines = transcript.split('\n')
        for line in lines:
            line_lower = line.lower()
            for marker in decision_markers:
                if marker in line_lower:
                    # Extract the relevant sentence
                    decisions.append(line.strip())
                    break

        return decisions[:10]  # Limit to top 10

    # =========================================================================
    # SLIDE GENERATION
    # =========================================================================

    def generate_slides(
        self,
        phase: JPPPhase,
        meeting_result: MeetingResult,
        scenario: str,
    ) -> list[SlideContent]:
        """
        Generate slide content from a meeting transcript.

        This creates structured bullet points suitable for PDF slide generation.

        Args:
            phase: The JPP phase
            meeting_result: The result from run_staff_meeting
            scenario: The scenario for context

        Returns:
            List of SlideContent with title, bullets, and speaker notes
        """
        phase_config = PHASE_CONFIGS[phase]

        system_prompt = """You are a military staff officer creating briefing slides.
Convert the meeting transcript into structured slide content.

OUTPUT FORMAT (JSON-like structure):
For each slide, provide:
- SLIDE TITLE: Clear, concise title
- BULLETS: 4-6 key points as bullet items
- NOTES: Speaker notes with additional detail

Create slides for each major topic discussed. Be SPECIFIC and SUBSTANTIVE.
Use actual content from the transcript, not generic placeholders."""

        user_prompt = f"""Create briefing slides for the {phase_config['name']} phase.

=== KEY OUTPUTS REQUIRED ===
{', '.join(phase_config['key_outputs'])}

=== MEETING TRANSCRIPT ===
{meeting_result['transcript'][:8000]}  # Truncate for token limits

=== INSTRUCTIONS ===
Create 4-8 slides covering:
1. Title slide with phase name and date
2. Key findings/analysis
3. Staff assessments by functional area
4. Decisions made
5. Outstanding issues
6. Way ahead / Next steps

Format each slide as:
---
SLIDE: [Title]
- Bullet 1
- Bullet 2
- Bullet 3
- Bullet 4
NOTES: [Speaker notes]
---"""

        response = self._call_llm(system_prompt, user_prompt)

        # Parse response into SlideContent list
        slides = self._parse_slide_response(response)

        return slides

    def _parse_slide_response(self, response: str) -> list[SlideContent]:
        """Parse LLM response into SlideContent structures."""
        slides = []
        current_slide = None
        current_bullets = []
        current_notes = ""

        for line in response.split('\n'):
            line = line.strip()

            if line.startswith('SLIDE:') or line.startswith('---'):
                # Save previous slide
                if current_slide:
                    slides.append(SlideContent(
                        title=current_slide,
                        bullets=current_bullets,
                        notes=current_notes,
                    ))

                if line.startswith('SLIDE:'):
                    current_slide = line.replace('SLIDE:', '').strip()
                    current_bullets = []
                    current_notes = ""
                else:
                    current_slide = None

            elif line.startswith('- ') and current_slide:
                current_bullets.append(line[2:])

            elif line.startswith('NOTES:') and current_slide:
                current_notes = line.replace('NOTES:', '').strip()

        # Save last slide
        if current_slide:
            slides.append(SlideContent(
                title=current_slide,
                bullets=current_bullets,
                notes=current_notes,
            ))

        return slides

    # =========================================================================
    # COMMANDER BRIEF
    # =========================================================================

    def run_commander_brief(
        self,
        phase: JPPPhase,
        meeting_result: MeetingResult,
        slides: list[SlideContent],
        scenario: str,
        on_turn_callback: Callable[[DialogueTurn], None] | None = None,
        turn_delay: float = 0.3,
    ) -> BriefResult:
        """
        Run the commander briefing where staff presents and commander asks questions.

        Args:
            phase: The JPP phase
            meeting_result: Result from the staff meeting
            slides: Generated slide content
            scenario: The scenario
            on_turn_callback: Callback for live rendering
            turn_delay: Delay between turns

        Returns:
            BriefResult with turns, questions, and clarifications
        """
        phase_config = PHASE_CONFIGS[phase]
        turns: list[DialogueTurn] = []
        questions: list[str] = []
        clarifications: list[str] = []

        # Get commander persona
        commander = self.get_or_create_agent(StaffRole.COMMANDER)
        commander_persona = self.get_persona(StaffRole.COMMANDER)

        # Each lead agent briefs their portion
        lead_agents = phase_config["lead_agents"]
        questions_so_far = ""

        for idx, role in enumerate(lead_agents):
            agent = self.get_or_create_agent(role)
            persona = self.get_persona(role)

            # Get relevant slide content for this role
            slide_content = self._get_slides_for_role(slides, role, idx, len(lead_agents))

            # Staff member briefs
            brief_prompt = get_brief_prompt(
                phase=phase,
                role=role,
                slide_content=slide_content,
                persona=persona,
                questions_so_far=questions_so_far,
            )

            brief_response = invoke_with_retry(agent, brief_prompt)

            brief_turn = DialogueTurn(
                speaker=persona.short_designation,
                role=role.value,
                role_display=role.value.replace('_', ' ').title(),
                branch=persona.branch.value,
                rank=persona.rank_abbrev,
                text=brief_response,
                turn_number=len(turns) + 1,
                is_commander=False,
            )

            turns.append(brief_turn)
            if on_turn_callback:
                on_turn_callback(brief_turn)
                if turn_delay > 0:
                    time.sleep(turn_delay)

            # Commander asks a question (50% chance after each brief, always after last)
            if idx == len(lead_agents) - 1 or (idx % 2 == 0):
                question_prompt = f"""You are the Commander. The {role.value.replace('_', ' ')} just briefed:

{brief_response}

Ask ONE pointed question that:
1. Probes a potential weakness or gap
2. Seeks clarification on a critical point
3. Tests an assumption

Keep your question to 1-2 sentences. Be direct and commanding."""

                question = self._call_llm(
                    f"You are {commander_persona.full_designation}, the Commander.",
                    question_prompt
                )

                question_turn = DialogueTurn(
                    speaker=commander_persona.short_designation,
                    role=StaffRole.COMMANDER.value,
                    role_display="Commander",
                    branch=commander_persona.branch.value,
                    rank=commander_persona.rank_abbrev,
                    text=question,
                    turn_number=len(turns) + 1,
                    is_commander=True,
                )

                turns.append(question_turn)
                questions.append(question)
                questions_so_far += f"\nCommander asked: {question}\n"

                if on_turn_callback:
                    on_turn_callback(question_turn)
                    if turn_delay > 0:
                        time.sleep(turn_delay)

                # Staff responds to question
                answer_prompt = f"""The Commander just asked you:
{question}

Provide a direct, substantive answer. Be specific and honest about any limitations."""

                answer = invoke_with_retry(agent, answer_prompt)

                answer_turn = DialogueTurn(
                    speaker=persona.short_designation,
                    role=role.value,
                    role_display=role.value.replace('_', ' ').title(),
                    branch=persona.branch.value,
                    rank=persona.rank_abbrev,
                    text=answer,
                    turn_number=len(turns) + 1,
                    is_commander=False,
                )

                turns.append(answer_turn)
                clarifications.append(answer)
                questions_so_far += f"{persona.short_designation} answered: {answer}\n"

                if on_turn_callback:
                    on_turn_callback(answer_turn)
                    if turn_delay > 0:
                        time.sleep(turn_delay)

        return BriefResult(
            turns=turns,
            questions_asked=questions,
            clarifications=clarifications,
        )

    def _get_slides_for_role(
        self,
        slides: list[SlideContent],
        role: StaffRole,
        role_idx: int,
        total_roles: int,
    ) -> str:
        """Get the slide content relevant to a particular role."""
        if not slides:
            return "[No slides generated yet]"

        # Divide slides among briefers
        slides_per_role = max(1, len(slides) // total_roles)
        start_idx = role_idx * slides_per_role
        end_idx = start_idx + slides_per_role if role_idx < total_roles - 1 else len(slides)

        role_slides = slides[start_idx:end_idx]

        content = []
        for slide in role_slides:
            content.append(f"SLIDE: {slide['title']}")
            for bullet in slide['bullets']:
                content.append(f"  - {bullet}")

        return "\n".join(content)

    # =========================================================================
    # COMMANDER GUIDANCE
    # =========================================================================

    def issue_commander_guidance(
        self,
        phase: JPPPhase,
        meeting_result: MeetingResult,
        brief_result: BriefResult,
        scenario: str,
        on_turn_callback: Callable[[DialogueTurn], None] | None = None,
    ) -> GuidanceResult:
        """
        Have the commander issue guidance for the next phase.

        Args:
            phase: The current JPP phase
            meeting_result: Result from staff meeting
            brief_result: Result from commander brief
            scenario: The scenario
            on_turn_callback: Callback for rendering

        Returns:
            GuidanceResult with guidance text and structured priorities
        """
        commander = self.get_or_create_agent(StaffRole.COMMANDER)
        commander_persona = self.get_persona(StaffRole.COMMANDER)

        # Summarize meeting and brief
        meeting_summary = self._summarize_transcript(meeting_result['transcript'])
        brief_summary = self._summarize_brief(brief_result)

        # Get commander guidance
        prompt = get_commander_guidance_prompt(
            phase=phase,
            meeting_summary=meeting_summary,
            brief_summary=brief_summary,
            scenario=scenario,
        )

        guidance_text = invoke_with_retry(commander, prompt)

        # Create turn for UI
        guidance_turn = DialogueTurn(
            speaker=commander_persona.short_designation,
            role=StaffRole.COMMANDER.value,
            role_display="Commander",
            branch=commander_persona.branch.value,
            rank=commander_persona.rank_abbrev,
            text=guidance_text,
            turn_number=1,
            is_commander=True,
        )

        if on_turn_callback:
            on_turn_callback(guidance_turn)

        # Parse guidance into structured form
        priority_tasks = self._extract_priority_tasks(guidance_text)
        guidance_by_section = self._extract_section_guidance(guidance_text)

        return GuidanceResult(
            guidance_text=guidance_text,
            priority_tasks=priority_tasks,
            guidance_by_section=guidance_by_section,
        )

    def _summarize_transcript(self, transcript: str) -> str:
        """Create a summary of the meeting transcript."""
        # For now, just truncate. Could use LLM summarization.
        return transcript[:4000] if len(transcript) > 4000 else transcript

    def _summarize_brief(self, brief_result: BriefResult) -> str:
        """Create a summary of the brief."""
        parts = []
        for turn in brief_result['turns']:
            parts.append(f"{turn['speaker']}: {turn['text'][:200]}...")
        return "\n".join(parts[:6])

    def _extract_priority_tasks(self, guidance: str) -> list[str]:
        """Extract priority tasks from commander guidance."""
        tasks = []
        lines = guidance.split('\n')
        in_priority_section = False

        for line in lines:
            if 'priority' in line.lower() or 'task' in line.lower():
                in_priority_section = True
            elif in_priority_section and line.strip().startswith('-'):
                tasks.append(line.strip()[1:].strip())
            elif in_priority_section and line.strip() and not line.strip().startswith('-'):
                if len(tasks) > 0:
                    in_priority_section = False

        return tasks[:8]

    def _extract_section_guidance(self, guidance: str) -> dict[str, str]:
        """Extract guidance directed at specific sections."""
        section_guidance = {}

        # Look for patterns like "J2:" or "Intel:" or "J2, I want..."
        section_patterns = {
            'j2': ['j2', 'intel', 'intelligence'],
            'j3': ['j3', 'ops', 'operations'],
            'j4': ['j4', 'log', 'logistics'],
            'j5': ['j5', 'plans'],
            'j6': ['j6', 'comms', 'communications'],
            'cyber': ['cyber', 'ew'],
            'fires': ['fires'],
            'sja': ['sja', 'legal'],
        }

        lines = guidance.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower()
            for section, patterns in section_patterns.items():
                for pattern in patterns:
                    if pattern in line_lower and ':' in line:
                        current_section = section
                        section_guidance[section] = line.split(':', 1)[1].strip()
                        break

        return section_guidance

    # =========================================================================
    # FULL PHASE EXECUTION
    # =========================================================================

    def run_full_phase(
        self,
        phase: JPPPhase,
        scenario: str,
        prior_context: str = "",
        on_turn_callback: Callable[[DialogueTurn], None] | None = None,
        on_substep_callback: Callable[[str, str], None] | None = None,
        turn_delay: float = 0.2,
    ) -> PhaseResult:
        """
        Run all four substeps of a JPP phase.

        Args:
            phase: The JPP phase to execute
            scenario: The scenario text
            prior_context: Context from prior phases
            on_turn_callback: Callback for each dialogue turn
            on_substep_callback: Callback when starting a new substep (a, b, c, d)
            turn_delay: Delay between turns

        Returns:
            Complete PhaseResult with all substep outputs
        """
        phase_config = PHASE_CONFIGS[phase]

        # Step A: Staff Meeting
        if on_substep_callback:
            on_substep_callback("a", f"{phase_config['name']} - Staff Meeting")

        meeting_result = self.run_staff_meeting(
            phase=phase,
            scenario=scenario,
            prior_context=prior_context,
            on_turn_callback=on_turn_callback,
            turn_delay=turn_delay,
        )

        # Step B: Slide Generation
        if on_substep_callback:
            on_substep_callback("b", f"{phase_config['name']} - Generating Slides")

        slides = self.generate_slides(
            phase=phase,
            meeting_result=meeting_result,
            scenario=scenario,
        )

        # Step C: Commander Brief
        if on_substep_callback:
            on_substep_callback("c", f"{phase_config['name']} - Briefing Commander")

        brief_result = self.run_commander_brief(
            phase=phase,
            meeting_result=meeting_result,
            slides=slides,
            scenario=scenario,
            on_turn_callback=on_turn_callback,
            turn_delay=turn_delay,
        )

        # Step D: Commander Guidance
        if on_substep_callback:
            on_substep_callback("d", f"{phase_config['name']} - Commander Guidance")

        guidance_result = self.issue_commander_guidance(
            phase=phase,
            meeting_result=meeting_result,
            brief_result=brief_result,
            scenario=scenario,
            on_turn_callback=on_turn_callback,
        )

        return PhaseResult(
            phase_name=phase_config["name"],
            meeting=meeting_result,
            slides=slides,
            brief=brief_result,
            guidance=guidance_result,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Default temperature for dialogue generation (higher for more varied speech)
# This produces more natural, varied conversation without sacrificing coherence.
# INCREASED from 0.7 to 0.8 for more human-like, varied dialogue.
# Use lower values (0.4-0.6) for formal products like OPORDs, slides, etc.
DIALOGUE_TEMPERATURE = 0.8

# Temperature recommendations by output type:
# - Staff meeting dialogue: 0.75-0.85 (more varied, natural speech)
# - Commander briefing: 0.7-0.8 (slightly more formal but still natural)
# - Slide generation: 0.5-0.6 (more consistent, formal)
# - OPORD/formal products: 0.3-0.5 (highly consistent, doctrinal)


def create_orchestrator(
    model_name: str = "gpt-4o",
    temperature: float = DIALOGUE_TEMPERATURE,  # Higher default for natural dialogue
    persona_seed: int | None = None,
) -> MeetingOrchestrator:
    """
    Create a configured MeetingOrchestrator.

    The default temperature (0.8) is tuned for natural, varied dialogue that
    sounds like real military officers talking, not templated AI responses.
    This produces:
    - More varied sentence structures
    - Natural informality and professional humor
    - Realistic disagreement and cross-talk
    - Domain-specific slang used naturally

    For formal written products (OPORDs, slides), consider using a separate
    lower temperature (0.4-0.6) to ensure consistency and doctrinal accuracy.

    Args:
        model_name: The OpenAI model to use (default: gpt-4o)
        temperature: LLM temperature for dialogue generation (default: 0.8)
                    Higher values (0.75-0.85) produce more varied, natural speech.
                    Lower values (0.3-0.5) would be used for formal products.
        persona_seed: Optional seed for reproducible persona generation

    Returns:
        Configured MeetingOrchestrator instance
    """
    config = WARGATEConfig(
        model_name=model_name,
        temperature=temperature,
        persona_seed=persona_seed,
        verbose=False,  # Suppress agent verbose output
    )
    return MeetingOrchestrator(config)
