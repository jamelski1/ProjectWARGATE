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

    # Determine the agent's behavior based on turn number
    if turn_number <= 3:
        turn_guidance = """This is an OPENING TURN. You should:
- Introduce your key concerns and initial assessment
- Raise important questions for the group
- Reference relevant doctrine or data from your domain"""
    elif turn_number <= 8:
        turn_guidance = """This is a DEVELOPMENT TURN. You should:
- Build on what others have said
- Challenge assumptions or offer alternative perspectives
- Propose specific solutions or options
- Ask clarifying questions of other staff sections"""
    else:
        turn_guidance = """This is a REFINEMENT TURN. You should:
- Synthesize discussion points into concrete recommendations
- Identify remaining issues or risks
- Propose decision points or next steps
- Confirm coordination requirements with other sections"""

    prompt = f"""You are {persona.full_designation}, the {role.value.replace('_', ' ').title()}.

You are in a staff meeting for the "{phase_config['name']}" phase of the Joint Planning Process.

{persona.culture_description}

=== MEETING CONTEXT ===
Topic: {phase_config['topic']}
Key Outputs Required: {', '.join(phase_config['key_outputs'])}
Focus Areas: {', '.join(phase_config['focus_areas'])}

=== SCENARIO ===
{scenario}

=== PRIOR PLANNING CONTEXT ===
{prior_context if prior_context else "This is the first phase; no prior context."}

=== CONVERSATION SO FAR ===
{conversation_so_far if conversation_so_far else "[Meeting just started - you are among the first to speak]"}

=== YOUR TURN (Turn #{turn_number}) ===
{turn_guidance}

IMPORTANT SPEAKING GUIDELINES:
1. Speak in FIRST PERSON as if you are actually in the meeting room
2. Be SPECIFIC and SUBSTANTIVE - use real doctrine, realistic numbers, actual considerations
3. Your response should be 150-400 words - enough for meaningful contribution but not a monologue
4. Reference what OTHER STAFF SECTIONS have said and respond to them
5. When you disagree, say so professionally but clearly
6. Use your service branch perspective where relevant
7. End with a clear point, question, or recommendation

DO NOT:
- Write "I think we should..." without substance
- Give generic responses that any staff could give
- Ignore what others have said
- Speak as if writing a memo (this is a CONVERSATION)

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

    return f"""You are {persona.full_designation}, the {role.value.replace('_', ' ').title()}.

You are briefing the Commander on your section's findings from the {phase_config['name']} phase.

=== YOUR SLIDE CONTENT ===
{slide_content}

=== QUESTIONS/DISCUSSION SO FAR ===
{questions_so_far if questions_so_far else "[You are presenting first]"}

BRIEFING GUIDELINES:
1. Present your key points CONCISELY but with SUBSTANCE
2. Highlight risks, concerns, or outstanding issues
3. Be prepared to answer commander's questions
4. If responding to a question, be direct and specific
5. Use your service branch expertise where relevant

Your briefing should be 100-250 words - executive summary style.

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

    return f"""You are the Commander presiding over the {phase_config['name']} phase.

Your staff has just completed their meeting and briefed you on their findings.

=== SCENARIO ===
{scenario}

=== STAFF MEETING SUMMARY ===
{meeting_summary}

=== STAFF BRIEF SUMMARY ===
{brief_summary}

=== YOUR TASK ===
Issue Commander's Guidance for the next phase ({next_phase_name}).

Your guidance should:
1. ACKNOWLEDGE key points from the staff work (be specific)
2. DECIDE on any outstanding issues requiring your decision
3. PRIORITIZE the next phase's focus areas
4. DIRECT specific staff sections on what you need from them
5. ACCEPT RISK where appropriate and explain your rationale
6. ISSUE INTENT for how you want the staff to approach the next phase

FORMAT YOUR GUIDANCE AS:
1. COMMANDER'S ASSESSMENT: Your view of the situation and staff work
2. DECISIONS: Any decisions you're making now
3. PRIORITY TASKS: What staff sections should focus on next
4. RISK GUIDANCE: What risks you're accepting and why
5. INTENT FOR NEXT PHASE: How you want the staff to proceed

Be SUBSTANTIVE and SPECIFIC. Your guidance shapes the entire next phase.

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
        """Make a direct LLM call (for slide generation, guidance, etc.)."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

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
            response = agent.invoke(prompt)

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

            brief_response = agent.invoke(brief_prompt)

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

                answer = agent.invoke(answer_prompt)

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

        guidance_text = commander.invoke(prompt)

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

def create_orchestrator(
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    persona_seed: int | None = None,
) -> MeetingOrchestrator:
    """Create a configured MeetingOrchestrator."""
    config = WARGATEConfig(
        model_name=model_name,
        temperature=temperature,
        persona_seed=persona_seed,
        verbose=False,  # Suppress agent verbose output
    )
    return MeetingOrchestrator(config)
