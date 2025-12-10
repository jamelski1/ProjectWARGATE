"""
ProjectWARGATE: Multi-Agent Joint Staff Planning System

A LangChain-based multi-agent system that mirrors a joint military staff
doing operational planning and COA (Course of Action) development.

Author: ProjectWARGATE Team
"""

from __future__ import annotations

import os
from typing import Any, Callable, TypedDict
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import Tool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from pydantic import BaseModel, Field


# =============================================================================
# CONFIGURATION
# =============================================================================

class WARGATEConfig(BaseModel):
    """Configuration for the WARGATE planning system."""
    model_name: str = Field(default="gpt-4.1", description="OpenAI model to use")
    temperature: float = Field(default=0.7, description="LLM temperature")
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    verbose: bool = Field(default=True, description="Enable verbose output")
    api_key: str | None = Field(default=None, description="OpenAI API key (or use env var)")


# =============================================================================
# STUB TOOLS / RAG RETRIEVERS
# =============================================================================
# These are placeholder implementations. Replace with real RAG retrievers.

def doctrine_query(query: str) -> str:
    """
    Retrieves relevant joint doctrine, LOAC, AI policy, and military regulations.
    TODO: Replace with real RAG retriever connected to doctrine knowledge base.
    """
    return f"[DOCTRINE_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to doctrine vector store for:\n- Joint Publications (JP 3-0, JP 5-0, etc.)\n- LOAC/IHL references\n- AI/Autonomy policy (DoDD 3000.09)\n- ROE frameworks"


def geopolitics_query(query: str) -> str:
    """
    Retrieves current and historical geopolitical context, regional dynamics,
    and international relations information.
    TODO: Replace with real RAG retriever connected to geopolitical knowledge base.
    """
    return f"[GEOPOLITICS_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to geopolitics vector store for:\n- Regional political dynamics\n- Alliance structures\n- Historical precedents\n- Current tensions and flashpoints"


def logistics_query(query: str) -> str:
    """
    Retrieves sustainment, supply chain, infrastructure, and logistics data.
    TODO: Replace with real RAG retriever connected to logistics knowledge base.
    """
    return f"[LOGISTICS_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to logistics vector store for:\n- Supply route analysis\n- Stockage levels\n- POL distribution networks\n- Maintenance and repair capabilities"


def cyberintel_query(query: str) -> str:
    """
    Retrieves cyber threat intelligence, AI/cyber incidents, and information
    environment assessments.
    TODO: Replace with real RAG retriever connected to cyber intel knowledge base.
    """
    return f"[CYBERINTEL_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to cyber intel vector store for:\n- Known threat actor TTPs\n- Recent cyber incidents\n- Infrastructure vulnerabilities\n- AI-enabled threat capabilities"


def terrain_query(query: str) -> str:
    """
    Retrieves terrain analysis, geographic data, and environmental factors.
    TODO: Replace with real RAG retriever connected to terrain/geo knowledge base.
    """
    return f"[TERRAIN_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to terrain vector store for:\n- OAKOC analysis\n- Key terrain features\n- Weather impacts\n- Infrastructure and LOCs"


def orbat_query(query: str) -> str:
    """
    Retrieves order of battle information, force structure, and unit capabilities.
    TODO: Replace with real RAG retriever connected to ORBAT knowledge base.
    """
    return f"[ORBAT_RETRIEVAL] Query: '{query}'\n\nPlaceholder response - Connect to ORBAT vector store for:\n- Friendly force structure\n- Enemy order of battle\n- Unit capabilities and readiness\n- Equipment and weapons systems"


# Create LangChain Tool objects with proper descriptions

doctrine_retriever = Tool(
    name="doctrine_retriever",
    func=doctrine_query,
    description=(
        "Retrieves relevant joint doctrine, Law of Armed Conflict (LOAC), "
        "AI policy, military regulations, and doctrinal frameworks. Use this "
        "to find authoritative guidance on military operations, legal constraints, "
        "and established procedures. Input should be a specific question or topic."
    )
)

geopolitics_retriever = Tool(
    name="geopolitics_retriever",
    func=geopolitics_query,
    description=(
        "Retrieves geopolitical context including regional dynamics, alliance "
        "structures, international relations, historical precedents, and current "
        "political tensions. Use for understanding the strategic environment "
        "and political implications of military actions. Input should be a "
        "specific question about geopolitical factors."
    )
)

logistics_retriever = Tool(
    name="logistics_retriever",
    func=logistics_query,
    description=(
        "Retrieves logistics and sustainment information including supply routes, "
        "POL (petroleum, oils, lubricants), ammunition stockage, maintenance "
        "capabilities, transportation networks, and infrastructure data. Use for "
        "assessing feasibility and sustainment requirements. Input should be a "
        "specific logistics-related question."
    )
)

cyberintel_retriever = Tool(
    name="cyberintel_retriever",
    func=cyberintel_query,
    description=(
        "Retrieves cyber threat intelligence including known threat actor TTPs, "
        "recent cyber incidents, infrastructure vulnerabilities, AI-enabled threats, "
        "and information environment assessments. Use for understanding cyber "
        "threats and information warfare considerations. Input should be a "
        "specific cyber/information security question."
    )
)

terrain_retriever = Tool(
    name="terrain_retriever",
    func=terrain_query,
    description=(
        "Retrieves terrain and geographic analysis including OAKOC (Observation, "
        "Avenues of Approach, Key Terrain, Obstacles, Cover/Concealment), weather "
        "impacts, infrastructure, and lines of communication. Use for operational "
        "geography questions. Input should be a specific terrain-related question."
    )
)

orbat_retriever = Tool(
    name="orbat_retriever",
    func=orbat_query,
    description=(
        "Retrieves Order of Battle (ORBAT) information including friendly and "
        "enemy force structures, unit capabilities, equipment, weapons systems, "
        "and readiness levels. Use for force correlation and capability assessment. "
        "Input should be a specific question about forces or units."
    )
)


# =============================================================================
# STAFF ROLE DEFINITIONS
# =============================================================================

class StaffRole(Enum):
    """Enumeration of joint staff roles."""
    COMMANDER = "commander"
    J1 = "j1_personnel"
    J2 = "j2_intelligence"
    J3 = "j3_operations"
    J4 = "j4_logistics"
    J5 = "j5_plans"
    J6 = "j6_communications"
    J7 = "j7_training"
    J8 = "j8_resources"
    CYBER_EW = "cyber_ew_oic"
    FIRES = "fires_oic"
    ENGINEER = "engineer_oic"
    PROTECTION = "protection_oic"
    SJA = "sja_legal"
    PAO = "pao_io"


# System prompts for each staff role
STAFF_SYSTEM_PROMPTS: dict[StaffRole, str] = {

    StaffRole.COMMANDER: """You are the COMMANDER / J3 COMMANDER'S CELL in a joint military staff.

DOCTRINAL ROLE:
- You are the final decision authority for this planning effort
- You issue Commander's Intent, which provides purpose, key tasks, and end state
- You select the recommended Course of Action (COA) after staff analysis
- You provide guidance and priorities to focus staff efforts
- You accept risk on behalf of the force

PLANNING RESPONSIBILITIES:
- Issue clear, concise Commander's Intent that enables disciplined initiative
- Prioritize warfighting functions and allocate resources
- Make timely decisions based on staff recommendations
- Assess and accept prudent risk
- Ensure unity of effort across all staff sections

WHEN REVIEWING COAs:
- Evaluate against mission accomplishment probability
- Consider risk to force and risk to mission
- Assess sustainability and feasibility
- Ensure legal and ethical compliance
- Consider strategic and political implications

You have access to tools for retrieving doctrine and other information. Use them when you need authoritative guidance. Be decisive but thoughtful. Your decisions shape the operation.""",

    StaffRole.J1: """You are the J1 - PERSONNEL staff officer in a joint military staff.

DOCTRINAL ROLE:
- Advise the commander on all personnel and manpower matters
- Manage force availability, strength reporting, and personnel readiness
- Coordinate casualty operations and replacement
- Address morale, welfare, and personnel services
- Manage personnel reconstitution and rotation

PLANNING RESPONSIBILITIES:
- Assess personnel requirements for each COA
- Identify manpower constraints and shortfalls
- Evaluate casualty estimates and replacement timelines
- Consider fatigue, morale, and human performance factors
- Coordinate personnel augmentation requirements

WHEN EVALUATING COAs:
- Can we man this operation with available forces?
- What are the personnel risks and casualty estimates?
- How will this affect unit cohesion and morale?
- What are the rotation and sustainment personnel implications?
- Are there critical skill shortages that affect feasibility?

You have access to doctrine retrieval tools. Use them to reference personnel policies and procedures. Provide realistic assessments of human capital constraints.""",

    StaffRole.J2: """You are the J2 - INTELLIGENCE staff officer and RED TEAM lead in a joint military staff.

DOCTRINAL ROLE:
- Provide all-source intelligence analysis and threat assessment
- Develop enemy Courses of Action (ECOAs) and most likely/dangerous COAs
- Identify intelligence gaps and collection requirements
- Provide indications and warnings (I&W)
- Challenge assumptions through red-teaming

RED TEAM RESPONSIBILITIES:
- Critically examine friendly COAs from adversary perspective
- Identify vulnerabilities the enemy could exploit
- Challenge groupthink and unexamined assumptions
- War-game enemy reactions to friendly actions
- Identify potential surprise and deception

WHEN EVALUATING COAs:
- How will the enemy likely react to this COA?
- What are the enemy's most dangerous and most likely responses?
- What assumptions are we making that could be wrong?
- Where are we vulnerable to enemy action?
- What intelligence gaps could lead to mission failure?

You have access to doctrine, geopolitics, cyber intelligence, and ORBAT retrieval tools. Be the devil's advocate. Challenge comfortable assumptions. Think like the enemy.""",

    StaffRole.J3: """You are the J3 - OPERATIONS staff officer in a joint military staff.

DOCTRINAL ROLE:
- Primary staff officer for current and future operations
- Develop and synchronize COAs from an execution perspective
- Integrate and synchronize all warfighting functions
- Manage operations center and battle rhythm
- Coordinate joint and combined operations

PLANNING RESPONSIBILITIES:
- Develop detailed COA sketches and synchronization matrices
- Integrate fires, maneuver, and effects across domains
- Identify decisive points and lines of operation
- Synchronize timing and phasing of operations
- Coordinate joint force employment

WHEN DEVELOPING/EVALUATING COAs:
- Is this COA executable with available forces and time?
- Are the phases properly synchronized?
- Have we identified the decisive operation?
- Is there proper main effort / supporting effort designation?
- Are branches and sequels identified?

You have access to doctrine and ORBAT retrieval tools. Focus on practical execution. Ensure COAs are tactically and operationally sound. You are responsible for making plans actually work.""",

    StaffRole.J4: """You are the J4 - LOGISTICS staff officer in a joint military staff.

DOCTRINAL ROLE:
- Primary staff officer for all logistics and sustainment matters
- Manage supply, maintenance, transportation, and services
- Assess logistics feasibility of operations
- Coordinate distribution and supply chain operations
- Plan for logistics over extended lines of communication

PLANNING RESPONSIBILITIES:
- Assess sustainment requirements for each COA (classes of supply)
- Evaluate transportation and distribution network capacity
- Identify logistics shortfalls and mitigation strategies
- Plan for maintenance and equipment readiness
- Coordinate fuel, ammunition, and critical supply positioning

WHEN EVALUATING COAs:
- Can we sustain this operation at the required tempo?
- What are the critical logistics nodes and vulnerabilities?
- Do we have adequate transportation capacity?
- What are the supply line risks and alternatives?
- Is the maintenance and repair posture adequate?

You have access to doctrine and logistics retrieval tools. Be the voice of logistical reality. Remember: amateurs talk tactics, professionals talk logistics.""",

    StaffRole.J5: """You are the J5 - PLANS staff officer in a joint military staff.

DOCTRINAL ROLE:
- Primary staff officer for mid- to long-range planning
- Develop campaign design and operational approach
- Plan branches, sequels, and future operations
- Coordinate with higher and adjacent headquarters
- Ensure operational art connects tactical actions to strategic objectives

PLANNING RESPONSIBILITIES:
- Develop the operational approach and lines of effort
- Design campaign phases and transitions
- Identify branches (contingencies) and sequels (follow-on operations)
- Ensure COAs nest with strategic guidance
- Coordinate with allies and coalition partners

WHEN EVALUATING COAs:
- Does this COA achieve strategic objectives?
- What are the 2nd and 3rd order effects?
- Have we planned for what comes after success?
- Are branches identified for key decision points?
- How does this fit the broader campaign?

You have access to doctrine and geopolitics retrieval tools. Think beyond the immediate operation. Connect tactical actions to strategic ends. You are the guardian of operational art.""",

    StaffRole.J6: """You are the J6 - COMMUNICATIONS / C4I staff officer in a joint military staff.

DOCTRINAL ROLE:
- Primary staff officer for command, control, communications, computers, and intelligence systems
- Manage network architecture and communications resilience
- Coordinate spectrum management and electromagnetic considerations
- Plan for degraded communications operations
- Integrate cyber-physical dependencies

PLANNING RESPONSIBILITIES:
- Assess communications requirements for each COA
- Plan network architecture and redundancy
- Identify single points of failure and vulnerabilities
- Coordinate PACE (Primary, Alternate, Contingency, Emergency) planning
- Manage spectrum and electromagnetic interference

WHEN EVALUATING COAs:
- Can we maintain C2 throughout the operation?
- What are the comms vulnerabilities and degraded ops procedures?
- Is the network architecture resilient to attack?
- Have we planned for contested electromagnetic environments?
- Are cyber-physical dependencies identified and protected?

You have access to doctrine, cyberintel, and geopolitics retrieval tools. Communications underpin everything. If we lose C2, we lose the battle.""",

    StaffRole.CYBER_EW: """You are the CYBER/EW OIC (Officer in Charge) in a joint military staff.

DOCTRINAL ROLE:
- Lead cyberspace operations (Offensive Cyber Ops, Defensive Cyber Ops)
- Coordinate electronic warfare across the electromagnetic spectrum
- Integrate information environment operations
- Advise on AI-enabled capabilities and threats
- Synchronize cyber/EW effects with kinetic operations

PLANNING RESPONSIBILITIES:
- Develop cyber and EW options for each COA
- Identify adversary cyber vulnerabilities and our own
- Plan OCO effects integrated with fires
- Ensure DCO posture protects critical networks
- Assess AI-enabled threats and opportunities

WHEN EVALUATING COAs:
- What cyber/EW effects can enable this COA?
- What are our cyber vulnerabilities in this operation?
- How does the enemy use cyber/EW and how do we counter?
- Are AI systems properly defended and employed?
- Is electronic attack synchronized with maneuver?

You have access to doctrine, cyberintel, and geopolitics retrieval tools. The cyber and electromagnetic domains are contested. Every operation has a cyber dimension.""",

    StaffRole.FIRES: """You are the FIRES OIC (Officer in Charge) in a joint military staff.

DOCTRINAL ROLE:
- Coordinate all fires across domains (air, land, sea, space, cyber)
- Integrate kinetic and non-kinetic effects
- Manage targeting process and target development
- Synchronize joint fires with maneuver
- Assess effects and battle damage

PLANNING RESPONSIBILITIES:
- Develop fires support for each COA
- Identify high-value targets and target sets
- Plan fire support coordination measures
- Integrate joint fires from all domains
- Balance kinetic and non-kinetic effects

WHEN EVALUATING COAs:
- Do we have adequate fires to support decisive operations?
- What are the priority targets for each phase?
- Are fire support coordination measures deconflicted?
- How do we integrate non-kinetic effects?
- What are the collateral damage and legal considerations?

You have access to doctrine and ORBAT retrieval tools. Fires set conditions for success. Mass effects at the decisive point. All fires must be synchronized and purposeful.""",

    StaffRole.ENGINEER: """You are the ENGINEER OIC (Officer in Charge) in a joint military staff.

DOCTRINAL ROLE:
- Lead mobility, counter-mobility, and survivability operations
- Coordinate infrastructure and critical node protection
- Plan obstacle integration and breaching operations
- Advise on terrain and environmental engineering
- Manage construction and general engineering

PLANNING RESPONSIBILITIES:
- Assess engineer requirements for each COA
- Plan obstacle belts and breach sites
- Identify critical infrastructure nodes
- Plan survivability positions and hardening
- Coordinate route clearance and MSR maintenance

WHEN EVALUATING COAs:
- Do we have adequate engineer assets for mobility?
- Where should we emplace obstacles for counter-mobility?
- What critical infrastructure must we protect or target?
- Are survivability positions planned for key nodes?
- What are the route clearance requirements?

You have access to doctrine, logistics, and terrain retrieval tools. Engineers are combat multipliers. We enable maneuver and deny it to the enemy.""",

    StaffRole.PROTECTION: """You are the PROTECTION OIC (Officer in Charge) in a joint military staff.

DOCTRINAL ROLE:
- Lead force protection and critical asset defense
- Coordinate air and missile defense
- Plan CBRN (Chemical, Biological, Radiological, Nuclear) defense
- Manage personnel recovery and CSAR
- Integrate physical security measures

PLANNING RESPONSIBILITIES:
- Assess protection requirements for each COA
- Plan air and missile defense coverage
- Identify critical assets requiring protection
- Develop CBRN reconnaissance and defense plans
- Coordinate personnel recovery operations

WHEN EVALUATING COAs:
- Are critical assets adequately protected?
- Is AMD (Air and Missile Defense) coverage sufficient?
- What CBRN threats exist and how are we postured?
- Do we have personnel recovery plans in place?
- What are the physical security vulnerabilities?

You have access to doctrine, logistics, and ORBAT retrieval tools. Protection preserves combat power. We must defend our critical vulnerabilities while exploiting the enemy's.""",

    StaffRole.SJA: """You are the SJA / LEGAL ADVISOR (Staff Judge Advocate / Ethicist) in a joint military staff.

DOCTRINAL ROLE:
- Advise on Law of Armed Conflict (LOAC) and International Humanitarian Law
- Review operations for legal compliance
- Advise on Rules of Engagement (ROE)
- Provide ethics guidance, especially for AI/autonomous systems
- Assess targeting legality and proportionality

PLANNING RESPONSIBILITIES:
- Review COAs for LOAC compliance
- Advise on distinction, proportionality, and military necessity
- Ensure ROE are adequate and understood
- Assess AI/autonomous systems against policy requirements
- Identify protected sites and no-strike entities

WHEN EVALUATING COAs:
- Is this COA lawful under LOAC/IHL?
- Are we maintaining distinction between combatants and civilians?
- Is anticipated collateral damage proportional to military advantage?
- Are autonomous systems employed within policy limits?
- What are the ethical implications of this operation?

You have access to doctrine retrieval tools. You are the conscience of the staff. Legal and ethical operations are non-negotiable. Unlawful orders must be identified and rejected.""",

    StaffRole.PAO: """You are the PAO / IO (Public Affairs Officer / Information Operations) lead in a joint military staff.

DOCTRINAL ROLE:
- Lead strategic communications and public affairs
- Coordinate information operations and influence activities
- Manage narrative and perception in the information environment
- Counter adversary propaganda and disinformation
- Integrate messaging across all operations

PLANNING RESPONSIBILITIES:
- Develop communication strategy for each COA
- Identify key audiences and messages
- Plan counter-disinformation operations
- Coordinate with allies on information activities
- Assess information environment effects

WHEN EVALUATING COAs:
- How will this operation be perceived domestically and internationally?
- What is the narrative and how do we control it?
- What disinformation threats exist?
- Are we winning the information competition?
- How do we exploit adversary information vulnerabilities?

You have access to doctrine and geopolitics retrieval tools. The information environment is a domain of warfare. Perception shapes reality. Win the narrative, win the war.""",

    StaffRole.J7: """You are the J7 - TRAINING / LESSONS LEARNED staff officer in a joint military staff.

DOCTRINAL ROLE:
- Capture and apply lessons learned from past operations
- Advise on training implications and readiness
- Reference doctrinal precedents and historical parallels
- Assess collective training status
- Integrate lessons into current planning

PLANNING RESPONSIBILITIES:
- Identify relevant historical examples and lessons
- Assess unit training status for required tasks
- Highlight doctrinal best practices
- Identify training gaps affecting COA feasibility
- Recommend pre-mission training requirements

WHEN EVALUATING COAs:
- What have we learned from similar operations?
- Are units trained for the required tasks?
- What doctrinal principles apply here?
- What went wrong in similar operations before?
- Do we have time for additional training?

You have access to doctrine retrieval tools. History is a harsh teacher but a great one. Learn from the past to succeed in the future. Doctrine represents hard-won lessons.""",

    StaffRole.J8: """You are the J8 - RESOURCES / FORCE STRUCTURE staff officer in a joint military staff.

DOCTRINAL ROLE:
- Advise on budgetary and resource constraints
- Assess force structure implications
- Manage long-term resource planning
- Coordinate capability development
- Assess cost-benefit and resource allocation

PLANNING RESPONSIBILITIES:
- Assess resource requirements for each COA
- Identify budgetary constraints and cost drivers
- Evaluate force structure implications
- Assess equipment procurement timelines
- Analyze long-term sustainment costs

WHEN EVALUATING COAs:
- What are the resource costs of this COA?
- Do we have the force structure to execute?
- What are the long-term resource implications?
- Are there more cost-effective alternatives?
- What are the opportunity costs?

You have access to doctrine and logistics retrieval tools. Resources are always constrained. Every decision has a cost. We must ensure our plans are affordable and sustainable.""",
}


# Tool assignments by role
ROLE_TOOLS: dict[StaffRole, list[Tool]] = {
    StaffRole.COMMANDER: [doctrine_retriever, orbat_retriever],
    StaffRole.J1: [doctrine_retriever, orbat_retriever],
    StaffRole.J2: [doctrine_retriever, geopolitics_retriever, cyberintel_retriever, orbat_retriever, terrain_retriever],
    StaffRole.J3: [doctrine_retriever, orbat_retriever, terrain_retriever],
    StaffRole.J4: [doctrine_retriever, logistics_retriever, terrain_retriever],
    StaffRole.J5: [doctrine_retriever, geopolitics_retriever],
    StaffRole.J6: [doctrine_retriever, cyberintel_retriever, geopolitics_retriever],
    StaffRole.CYBER_EW: [doctrine_retriever, cyberintel_retriever, geopolitics_retriever],
    StaffRole.FIRES: [doctrine_retriever, orbat_retriever, terrain_retriever],
    StaffRole.ENGINEER: [doctrine_retriever, logistics_retriever, terrain_retriever],
    StaffRole.PROTECTION: [doctrine_retriever, logistics_retriever, orbat_retriever],
    StaffRole.SJA: [doctrine_retriever, geopolitics_retriever],
    StaffRole.PAO: [doctrine_retriever, geopolitics_retriever],
    StaffRole.J7: [doctrine_retriever],
    StaffRole.J8: [doctrine_retriever, logistics_retriever],
}


# =============================================================================
# STAFF AGENT FACTORY
# =============================================================================

class StaffAgent:
    """Wrapper for a staff role agent with LangChain AgentExecutor."""

    def __init__(
        self,
        role: StaffRole,
        llm: ChatOpenAI,
        tools: list[Tool],
        verbose: bool = True
    ):
        self.role = role
        self.name = role.value
        self.system_prompt = STAFF_SYSTEM_PROMPTS[role]
        self.tools = tools
        self.verbose = verbose

        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def invoke(self, input_text: str, chat_history: list[BaseMessage] | None = None) -> str:
        """Invoke the agent with input and optional chat history."""
        result = self.executor.invoke({
            "input": input_text,
            "chat_history": chat_history or [],
        })
        return result.get("output", "")

    def __repr__(self) -> str:
        return f"StaffAgent(role={self.role.value})"


def create_staff_agent(
    role: StaffRole,
    config: WARGATEConfig,
) -> StaffAgent:
    """Factory function to create a staff agent for a given role."""

    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
    )

    tools = ROLE_TOOLS.get(role, [doctrine_retriever])

    return StaffAgent(
        role=role,
        llm=llm,
        tools=tools,
        verbose=config.verbose,
    )


def create_all_staff_agents(config: WARGATEConfig) -> dict[StaffRole, StaffAgent]:
    """Create all staff agents with the given configuration."""
    return {role: create_staff_agent(role, config) for role in StaffRole}


# =============================================================================
# PLANNING PHASES
# =============================================================================

class PlanningPhase(Enum):
    """Joint Planning Process phases."""
    RECEIPT_OF_MISSION = "receipt_of_mission"
    MISSION_ANALYSIS = "mission_analysis"
    COA_DEVELOPMENT = "coa_development"
    COA_ANALYSIS = "coa_analysis"
    COA_COMPARISON = "coa_comparison"
    COA_SELECTION = "coa_selection"
    PLAN_DEVELOPMENT = "plan_development"


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class WARGATEOrchestrator:
    """
    Orchestrates the multi-agent joint staff planning process.

    Follows a structured planning process based on Joint Planning doctrine (JP 5-0).
    """

    def __init__(self, config: WARGATEConfig | None = None):
        self.config = config or WARGATEConfig()
        self.staff: dict[StaffRole, StaffAgent] = {}
        self.planning_context: list[dict[str, Any]] = []
        self.coas: list[dict[str, Any]] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all staff agents."""
        if self._initialized:
            return
        self.staff = create_all_staff_agents(self.config)
        self._initialized = True

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(f"\n{'='*60}\n{message}\n{'='*60}")

    def _phase_log(self, phase: PlanningPhase, message: str) -> None:
        """Log a phase-specific message."""
        if self.config.verbose:
            print(f"\n[{phase.value.upper()}] {message}")

    def _collect_staff_input(
        self,
        prompt: str,
        roles: list[StaffRole] | None = None,
        phase: PlanningPhase | None = None,
    ) -> dict[StaffRole, str]:
        """Collect input from specified staff roles (or all if None)."""
        roles = roles or list(StaffRole)
        responses: dict[StaffRole, str] = {}

        for role in roles:
            if role not in self.staff:
                continue

            if phase:
                self._phase_log(phase, f"Consulting {role.value}...")

            response = self.staff[role].invoke(prompt)
            responses[role] = response

            if self.config.verbose:
                print(f"\n--- {role.value.upper()} ---")
                print(response[:500] + "..." if len(response) > 500 else response)

        return responses

    def _format_staff_inputs(self, responses: dict[StaffRole, str]) -> str:
        """Format staff responses into a single string for context."""
        formatted = []
        for role, response in responses.items():
            formatted.append(f"\n### {role.value.upper()} INPUT:\n{response}")
        return "\n".join(formatted)

    # -------------------------------------------------------------------------
    # Planning Phase Implementations
    # -------------------------------------------------------------------------

    def phase_mission_analysis(self, scenario: str) -> dict[str, Any]:
        """
        PHASE 1: Mission Analysis

        Staff analyzes the scenario to understand:
        - Specified, implied, and essential tasks
        - Constraints and restraints
        - Initial threat assessment
        - Operational environment
        - Commander's Critical Information Requirements (CCIRs)
        """
        self._log("PHASE 1: MISSION ANALYSIS")

        mission_analysis_prompt = f"""You are participating in MISSION ANALYSIS for the following scenario:

=== SCENARIO ===
{scenario}
=================

Analyze this scenario from your staff perspective. Provide:
1. Key factors relevant to your functional area
2. Specified and implied tasks you identify
3. Constraints and restraints affecting your area
4. Critical information requirements (CCIRs) you recommend
5. Initial assessment of risks and opportunities

Be thorough but concise. Use your retrieval tools if doctrine guidance would help."""

        # Get input from all staff
        responses = self._collect_staff_input(
            mission_analysis_prompt,
            phase=PlanningPhase.MISSION_ANALYSIS
        )

        # J2 provides initial threat assessment
        j2_threat_prompt = f"""Based on the scenario and your mission analysis, provide a focused THREAT ASSESSMENT:

=== SCENARIO ===
{scenario}
=================

Provide:
1. Most Likely Enemy COA (MLCOA)
2. Most Dangerous Enemy COA (MDCOA)
3. Key adversary capabilities and vulnerabilities
4. Intelligence gaps and collection priorities
5. Indications and Warnings to monitor"""

        threat_assessment = self.staff[StaffRole.J2].invoke(j2_threat_prompt)

        # Commander synthesizes and issues guidance
        commander_prompt = f"""You have received mission analysis inputs from your staff:

{self._format_staff_inputs(responses)}

=== J2 THREAT ASSESSMENT ===
{threat_assessment}
============================

Based on this analysis:
1. State your initial Commander's Intent (Purpose, Key Tasks, End State)
2. Issue planning guidance to focus COA development
3. Identify your Commander's Critical Information Requirements (CCIRs)
4. State acceptable levels of risk"""

        commander_guidance = self.staff[StaffRole.COMMANDER].invoke(commander_prompt)

        result = {
            "phase": PlanningPhase.MISSION_ANALYSIS.value,
            "scenario": scenario,
            "staff_analysis": responses,
            "threat_assessment": threat_assessment,
            "commander_guidance": commander_guidance,
        }

        self.planning_context.append(result)
        return result

    def phase_coa_development(self, mission_analysis: dict[str, Any]) -> dict[str, Any]:
        """
        PHASE 2: COA Development

        Develops multiple distinct courses of action.
        """
        self._log("PHASE 2: COA DEVELOPMENT")

        context = f"""
=== SCENARIO ===
{mission_analysis['scenario']}

=== COMMANDER'S GUIDANCE ===
{mission_analysis['commander_guidance']}

=== THREAT ASSESSMENT ===
{mission_analysis['threat_assessment']}
"""

        # J5 leads COA development with J3 support
        j5_coa_prompt = f"""Based on the mission analysis and Commander's guidance, develop 3 distinct Courses of Action (COAs).

{context}

For each COA, provide:
1. COA Name and Concept Statement
2. Main Effort and Supporting Efforts
3. Decisive Points and Lines of Operation/Effort
4. Key Phases and Transitions
5. Branches and Sequels
6. What makes this COA distinct from others

COAs should be:
- Feasible (can accomplish mission within constraints)
- Acceptable (worth the cost)
- Suitable (accomplishes the mission)
- Distinguishable (significantly different from each other)
- Complete (incorporates all elements)"""

        coa_concepts = self.staff[StaffRole.J5].invoke(j5_coa_prompt)

        # J3 develops execution details
        j3_detail_prompt = f"""The J5 has developed COA concepts. Add operational detail to each:

{context}

=== COA CONCEPTS ===
{coa_concepts}
====================

For each COA, add:
1. Task organization and force allocation
2. Synchronization of warfighting functions
3. Timing and phasing details
4. Critical coordination requirements
5. Key decision points"""

        coa_details = self.staff[StaffRole.J3].invoke(j3_detail_prompt)

        # Staff sections add functional details
        functional_roles = [
            StaffRole.J1, StaffRole.J4, StaffRole.J6,
            StaffRole.FIRES, StaffRole.ENGINEER, StaffRole.PROTECTION,
            StaffRole.CYBER_EW
        ]

        functional_prompt = f"""Review the following COAs and provide your functional input:

{context}

=== COA DETAILS ===
{coa_details}
===================

From your functional perspective:
1. What does your section need to provide for each COA?
2. What are the key requirements for each COA?
3. Are there significant differences in your requirements between COAs?"""

        functional_inputs = self._collect_staff_input(
            functional_prompt,
            roles=functional_roles,
            phase=PlanningPhase.COA_DEVELOPMENT
        )

        result = {
            "phase": PlanningPhase.COA_DEVELOPMENT.value,
            "coa_concepts": coa_concepts,
            "coa_details": coa_details,
            "functional_inputs": functional_inputs,
        }

        self.planning_context.append(result)
        self.coas = [
            {"name": "COA 1", "concept": coa_concepts, "details": coa_details},
            {"name": "COA 2", "concept": coa_concepts, "details": coa_details},
            {"name": "COA 3", "concept": coa_concepts, "details": coa_details},
        ]

        return result

    def phase_coa_analysis(self, coa_development: dict[str, Any]) -> dict[str, Any]:
        """
        PHASE 3: COA Analysis (Wargaming)

        War-games each COA against enemy COAs.
        """
        self._log("PHASE 3: COA ANALYSIS (WARGAMING)")

        context = f"""
=== COA CONCEPTS ===
{coa_development['coa_concepts']}

=== COA DETAILS ===
{coa_development['coa_details']}
"""

        # J2 Red Team challenges each COA
        red_team_prompt = f"""You are the RED TEAM. War-game enemy responses to each friendly COA:

{context}

For each COA:
1. How would the enemy most likely respond?
2. What friendly vulnerabilities could they exploit?
3. What are the critical friendly actions and enemy reactions?
4. Where might our plan fail?
5. What surprises could the enemy achieve?

Be adversarial. Think like the enemy. Challenge optimistic assumptions."""

        red_team_analysis = self.staff[StaffRole.J2].invoke(red_team_prompt)

        # J3 assesses execution risks
        j3_risk_prompt = f"""Based on the COAs and Red Team analysis, assess execution risks:

{context}

=== RED TEAM ANALYSIS ===
{red_team_analysis}
========================

For each COA, identify:
1. Risk to mission (things that could cause mission failure)
2. Risk to force (things that could cause excessive casualties)
3. Critical decision points where things could go wrong
4. Mitigation measures for key risks"""

        execution_risk = self.staff[StaffRole.J3].invoke(j3_risk_prompt)

        # Functional war-gaming
        wargame_roles = [StaffRole.J4, StaffRole.FIRES, StaffRole.CYBER_EW, StaffRole.PROTECTION]

        wargame_prompt = f"""Participate in wargaming the COAs:

{context}

=== RED TEAM ANALYSIS ===
{red_team_analysis}
========================

From your functional area:
1. What are critical actions and events in your area for each COA?
2. How might enemy actions affect your functional area?
3. What risks and decision points are relevant to your area?
4. What branches might be needed?"""

        wargame_inputs = self._collect_staff_input(
            wargame_prompt,
            roles=wargame_roles,
            phase=PlanningPhase.COA_ANALYSIS
        )

        # SJA legal review
        sja_prompt = f"""Review each COA for legal and ethical compliance:

{context}

=== RED TEAM ANALYSIS ===
{red_team_analysis}
========================

For each COA:
1. Are there LOAC/IHL concerns?
2. What are the collateral damage risks?
3. Are there ROE constraints or requirements?
4. Any concerns with autonomous systems employment?
5. Recommendations for ensuring legal compliance"""

        legal_review = self.staff[StaffRole.SJA].invoke(sja_prompt)

        result = {
            "phase": PlanningPhase.COA_ANALYSIS.value,
            "red_team_analysis": red_team_analysis,
            "execution_risk_assessment": execution_risk,
            "wargame_inputs": wargame_inputs,
            "legal_review": legal_review,
        }

        self.planning_context.append(result)
        return result

    def phase_coa_comparison(self, coa_analysis: dict[str, Any]) -> dict[str, Any]:
        """
        PHASE 4: COA Comparison

        Compares COAs against evaluation criteria.
        """
        self._log("PHASE 4: COA COMPARISON")

        # All staff sections provide comparison input
        comparison_prompt = f"""Compare the three COAs from your functional perspective.

=== COA CONCEPTS ===
{self.planning_context[1]['coa_concepts']}

=== RED TEAM ANALYSIS ===
{coa_analysis['red_team_analysis']}

=== LEGAL REVIEW ===
{coa_analysis['legal_review']}

Evaluate each COA against these criteria from YOUR perspective:
1. Feasibility (Can we do it with available resources?)
2. Acceptability (Is the cost worth the benefit?)
3. Suitability (Does it accomplish the mission?)
4. Distinguishability (Does this COA offer unique advantages?)
5. Completeness (Does it address all requirements?)

Provide your ranking of COAs (1st, 2nd, 3rd) with justification."""

        comparison_inputs = self._collect_staff_input(
            comparison_prompt,
            phase=PlanningPhase.COA_COMPARISON
        )

        # J5 synthesizes comparison
        j5_synthesis_prompt = f"""Synthesize the staff COA comparison inputs into a decision briefing format:

=== STAFF COMPARISON INPUTS ===
{self._format_staff_inputs(comparison_inputs)}
===============================

Provide:
1. Summary of each COA's strengths and weaknesses
2. Comparative analysis across evaluation criteria
3. Risk comparison (risk to mission, risk to force)
4. Staff recommendation with rationale"""

        comparison_synthesis = self.staff[StaffRole.J5].invoke(j5_synthesis_prompt)

        result = {
            "phase": PlanningPhase.COA_COMPARISON.value,
            "staff_inputs": comparison_inputs,
            "synthesis": comparison_synthesis,
        }

        self.planning_context.append(result)
        return result

    def phase_coa_selection(self, coa_comparison: dict[str, Any]) -> dict[str, Any]:
        """
        PHASE 5: COA Selection

        Commander selects the COA.
        """
        self._log("PHASE 5: COA SELECTION")

        commander_prompt = f"""You must now SELECT the Course of Action.

=== COMPARISON SYNTHESIS ===
{coa_comparison['synthesis']}

=== RED TEAM WARNINGS ===
{self.planning_context[2]['red_team_analysis']}

=== LEGAL CONSIDERATIONS ===
{self.planning_context[2]['legal_review']}

As the Commander:
1. STATE your selected COA and why
2. REFINE the Commander's Intent based on the selected COA
3. IDENTIFY acceptable risk and how to mitigate unacceptable risk
4. ISSUE guidance for plan development
5. STATE your CCIR updates if any"""

        commander_decision = self.staff[StaffRole.COMMANDER].invoke(commander_prompt)

        result = {
            "phase": PlanningPhase.COA_SELECTION.value,
            "commander_decision": commander_decision,
        }

        self.planning_context.append(result)
        return result

    def phase_plan_development(self, coa_selection: dict[str, Any]) -> dict[str, Any]:
        """
        PHASE 6: Plan Development

        Develops the detailed plan from selected COA.
        """
        self._log("PHASE 6: PLAN DEVELOPMENT")

        # J3/J5 develop the plan
        plan_prompt = f"""Develop the operational plan based on the selected COA.

=== COMMANDER'S DECISION ===
{coa_selection['commander_decision']}

=== SELECTED COA DETAILS ===
{self.planning_context[1]['coa_details']}

Develop the plan including:
1. SITUATION (Enemy, Friendly, Attachments/Detachments)
2. MISSION (Who, What, When, Where, Why)
3. EXECUTION (Commander's Intent, Concept of Operations, Tasks to Subordinate Units)
4. SUSTAINMENT (Logistics, Personnel, Medical)
5. COMMAND AND SIGNAL (C2, Communications)

Include:
- Phase lines and objectives
- Coordinating instructions
- Fire support coordination measures
- Synchronization matrix"""

        base_plan = self.staff[StaffRole.J3].invoke(plan_prompt)

        # Each functional area adds annexes
        annex_roles = [
            StaffRole.J1, StaffRole.J2, StaffRole.J4, StaffRole.J6,
            StaffRole.FIRES, StaffRole.CYBER_EW, StaffRole.ENGINEER,
            StaffRole.PROTECTION, StaffRole.PAO
        ]

        annex_prompt = f"""Develop your annex to the operational plan:

=== BASE PLAN ===
{base_plan}
=================

Develop your functional annex including:
1. Situation specific to your area
2. Mission and tasks for your functional area
3. Execution details and synchronization
4. Sustainment/support requirements
5. Command relationships and coordination"""

        annex_inputs = self._collect_staff_input(
            annex_prompt,
            roles=annex_roles,
            phase=PlanningPhase.PLAN_DEVELOPMENT
        )

        result = {
            "phase": PlanningPhase.PLAN_DEVELOPMENT.value,
            "base_plan": base_plan,
            "annexes": annex_inputs,
        }

        self.planning_context.append(result)
        return result

    def generate_final_output(self) -> str:
        """Generate the final unified operational plan and COA summary."""

        self._log("GENERATING FINAL OUTPUT")

        # Compile all context
        mission_analysis = self.planning_context[0]
        coa_development = self.planning_context[1]
        coa_analysis = self.planning_context[2]
        coa_comparison = self.planning_context[3]
        coa_selection = self.planning_context[4]
        plan_development = self.planning_context[5]

        # Commander issues final approval
        final_prompt = f"""Issue the FINAL OPERATION ORDER.

=== SCENARIO ===
{mission_analysis['scenario']}

=== COMMANDER'S DECISION & INTENT ===
{coa_selection['commander_decision']}

=== DEVELOPED PLAN ===
{plan_development['base_plan']}

=== KEY ANNEXES SUMMARY ===
{self._format_staff_inputs(plan_development['annexes'])}

Issue the final, unified OPERATION ORDER that integrates:
1. Complete 5-paragraph order format
2. Commander's intent (clearly stated)
3. Concept of operations with phases
4. Task organization
5. Coordinating instructions
6. Key decision points and branches
7. Risk mitigation measures
8. Success criteria and assessment framework"""

        final_order = self.staff[StaffRole.COMMANDER].invoke(final_prompt)

        # Compile summary
        output = f"""
################################################################################
#                           PROJECT WARGATE                                     #
#                    JOINT STAFF OPERATIONAL PLANNING OUTPUT                    #
################################################################################

================================================================================
                              MISSION ANALYSIS SUMMARY
================================================================================
{mission_analysis['commander_guidance']}

================================================================================
                              THREAT ASSESSMENT
================================================================================
{mission_analysis['threat_assessment']}

================================================================================
                         COURSES OF ACTION DEVELOPED
================================================================================
{coa_development['coa_concepts']}

================================================================================
                           COA ANALYSIS (WARGAMING)
================================================================================
=== RED TEAM ANALYSIS ===
{coa_analysis['red_team_analysis']}

=== LEGAL REVIEW ===
{coa_analysis['legal_review']}

================================================================================
                            COA COMPARISON
================================================================================
{coa_comparison['synthesis']}

================================================================================
                            COMMANDER'S DECISION
================================================================================
{coa_selection['commander_decision']}

================================================================================
                          FINAL OPERATION ORDER
================================================================================
{final_order}

################################################################################
#                         END OF PLANNING OUTPUT                                #
################################################################################
"""

        return output

    def run(self, scenario: str) -> str:
        """
        Execute the complete joint planning process.

        Args:
            scenario: The operational scenario description

        Returns:
            The final unified operational plan and COAs as a string
        """
        self.initialize()
        self.planning_context = []
        self.coas = []

        self._log(f"PROJECT WARGATE - JOINT STAFF PLANNING INITIATED")
        self._log(f"SCENARIO:\n{scenario}")

        # Execute planning phases
        mission_analysis = self.phase_mission_analysis(scenario)
        coa_development = self.phase_coa_development(mission_analysis)
        coa_analysis = self.phase_coa_analysis(coa_development)
        coa_comparison = self.phase_coa_comparison(coa_analysis)
        coa_selection = self.phase_coa_selection(coa_comparison)
        plan_development = self.phase_plan_development(coa_selection)

        # Generate final output
        final_output = self.generate_final_output()

        return final_output


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_wargate_planning(
    scenario: str,
    model_name: str = "gpt-4.1",
    temperature: float = 0.7,
    verbose: bool = True,
    api_key: str | None = None,
) -> str:
    """
    Main entry point for WARGATE joint staff planning.

    Args:
        scenario: The operational scenario to plan for
        model_name: OpenAI model name (default: "gpt-4.1")
        temperature: LLM temperature (default: 0.7)
        verbose: Enable verbose output (default: True)
        api_key: OpenAI API key (optional, uses env var if not provided)

    Returns:
        The final unified operational plan and COAs as a string

    Example:
        >>> scenario = '''
        ... A near-peer adversary has massed forces along the border of a NATO ally.
        ... Intelligence indicates an imminent invasion within 72 hours.
        ... The JFC has been directed to prepare options for deterrence and,
        ... if deterrence fails, defense of the allied nation.
        ... '''
        >>> result = run_wargate_planning(scenario, model_name="gpt-4.1")
        >>> print(result)
    """
    config = WARGATEConfig(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        api_key=api_key,
    )

    orchestrator = WARGATEOrchestrator(config)
    return orchestrator.run(scenario)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WARGATE: Multi-Agent Joint Staff Planning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python wargate.py --scenario "A near-peer adversary has massed forces..."
  python wargate.py --scenario-file scenario.txt --model gpt-4.1
  python wargate.py --scenario "..." --temperature 0.5 --quiet
        """
    )

    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="The operational scenario description"
    )

    parser.add_argument(
        "--scenario-file", "-f",
        type=str,
        help="Path to file containing the scenario"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name (default: gpt-4.1)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (optional, prints to stdout if not specified)"
    )

    args = parser.parse_args()

    # Get scenario
    if args.scenario_file:
        with open(args.scenario_file, 'r') as f:
            scenario = f.read()
    elif args.scenario:
        scenario = args.scenario
    else:
        # Default demo scenario
        scenario = """
SITUATION: A near-peer adversary (Country X) has massed approximately 150,000 troops
along the eastern border of Allied Nation Y, a NATO member. Intelligence indicates:

- 3x Combined Arms Armies with tank and motorized rifle divisions
- Significant artillery and rocket forces in forward positions
- Air defense umbrella established with S-400 and Pantsir systems
- Electronic warfare and cyber units actively probing allied networks
- Naval forces exercising in the adjacent sea, including amphibious capability
- Strategic messaging campaign underway to justify potential intervention

Allied Nation Y has limited defensive capability: 2 brigade combat teams, aging air force,
and limited air defense. NATO Article 5 would be invoked if attacked.

JFC has been directed to develop options for:
1. Deterrence operations to prevent invasion
2. Defensive operations if deterrence fails
3. Options for defeating adversary forces and restoring territorial integrity

Constraints: No first use of nuclear weapons. Minimize civilian casualties.
Operations must remain within allied territory unless attack occurs.

Time available for planning: 48 hours
"""
        print("No scenario provided. Using default demonstration scenario.\n")

    # Run planning
    result = run_wargate_planning(
        scenario=scenario,
        model_name=args.model,
        temperature=args.temperature,
        verbose=not args.quiet,
    )

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\nOutput written to: {args.output}")
    else:
        print(result)
