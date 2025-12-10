"""
ProjectWARGATE: Multi-Agent Joint Staff Planning System

A LangChain-based multi-agent system that mirrors a joint military staff
doing operational planning and COA (Course of Action) development.

Author: ProjectWARGATE Team
"""

from __future__ import annotations

import os
import random
import hashlib
from typing import Any, Callable, TypedDict
from enum import Enum
from dataclasses import dataclass

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
    model_name: str = Field(default="gpt-5.1", description="OpenAI model to use")
    temperature: float = Field(default=0.7, description="LLM temperature")
    max_tokens: int = Field(default=4096, description="Max tokens per response")
    verbose: bool = Field(default=True, description="Enable verbose output")
    api_key: str | None = Field(default=None, description="OpenAI API key (or use env var)")
    persona_seed: int | None = Field(default=None, description="Seed for reproducible persona generation")


# =============================================================================
# MILITARY BRANCH & RANK ASSIGNMENT
# =============================================================================

class MilitaryBranch(Enum):
    """U.S. Military Service Branches."""
    ARMY = "US Army"
    NAVY = "US Navy"
    AIR_FORCE = "US Air Force"
    MARINE_CORPS = "US Marine Corps"
    SPACE_FORCE = "US Space Force"
    COAST_GUARD = "US Coast Guard"


# Rank structures by branch (O-5 through O-10)
# Format: (abbreviation, full_title)
RANK_STRUCTURES: dict[MilitaryBranch, dict[str, tuple[str, str]]] = {
    MilitaryBranch.ARMY: {
        "O-5": ("LTC", "Lieutenant Colonel"),
        "O-6": ("COL", "Colonel"),
        "O-7": ("BG", "Brigadier General"),
        "O-8": ("MG", "Major General"),
        "O-9": ("LTG", "Lieutenant General"),
        "O-10": ("GEN", "General"),
    },
    MilitaryBranch.NAVY: {
        "O-5": ("CDR", "Commander"),
        "O-6": ("CAPT", "Captain"),
        "O-7": ("RDML", "Rear Admiral (Lower Half)"),
        "O-8": ("RADM", "Rear Admiral"),
        "O-9": ("VADM", "Vice Admiral"),
        "O-10": ("ADM", "Admiral"),
    },
    MilitaryBranch.AIR_FORCE: {
        "O-5": ("Lt Col", "Lieutenant Colonel"),
        "O-6": ("Col", "Colonel"),
        "O-7": ("Brig Gen", "Brigadier General"),
        "O-8": ("Maj Gen", "Major General"),
        "O-9": ("Lt Gen", "Lieutenant General"),
        "O-10": ("Gen", "General"),
    },
    MilitaryBranch.MARINE_CORPS: {
        "O-5": ("LtCol", "Lieutenant Colonel"),
        "O-6": ("Col", "Colonel"),
        "O-7": ("BGen", "Brigadier General"),
        "O-8": ("MajGen", "Major General"),
        "O-9": ("LtGen", "Lieutenant General"),
        "O-10": ("Gen", "General"),
    },
    MilitaryBranch.SPACE_FORCE: {
        "O-5": ("Lt Col", "Lieutenant Colonel"),
        "O-6": ("Col", "Colonel"),
        "O-7": ("Brig Gen", "Brigadier General"),
        "O-8": ("Maj Gen", "Major General"),
        "O-9": ("Lt Gen", "Lieutenant General"),
        "O-10": ("Gen", "General"),
    },
    MilitaryBranch.COAST_GUARD: {
        "O-5": ("CDR", "Commander"),
        "O-6": ("CAPT", "Captain"),
        "O-7": ("RDML", "Rear Admiral (Lower Half)"),
        "O-8": ("RADM", "Rear Admiral"),
        "O-9": ("VADM", "Vice Admiral"),
        "O-10": ("ADM", "Admiral"),
    },
}


# Branch-specific cultural perspectives and characteristics
BRANCH_CULTURE: dict[MilitaryBranch, str] = {
    MilitaryBranch.ARMY: """You bring an Army perspective emphasizing:
- Ground-centric operational thinking with focus on combined arms maneuver
- Deep appreciation for terrain, logistics sustainment, and personnel tempo
- Mission command philosophy enabling disciplined initiative at echelon
- Experience with sustained land campaigns and occupation/stability operations
- Strong focus on integration of fires, maneuver, and protection at tactical level""",

    MilitaryBranch.NAVY: """You bring a Navy perspective emphasizing:
- Maritime domain awareness and blue-water operational thinking
- Understanding of sea lines of communication and power projection from the sea
- Comfort with distributed operations across vast distances
- Emphasis on self-sufficiency and extended deployments
- Integration of surface, subsurface, and aviation assets in carrier-centric operations""",

    MilitaryBranch.AIR_FORCE: """You bring an Air Force perspective emphasizing:
- Airpower-centric thinking with focus on air superiority as an enabler
- Strategic perspective on global reach, global power, and rapid response
- Technical orientation toward precision, ISR, and information dominance
- Appreciation for the electromagnetic spectrum and space dependencies
- Experience with command and control of distributed air and space assets""",

    MilitaryBranch.MARINE_CORPS: """You bring a Marine Corps perspective emphasizing:
- Expeditionary mindset with focus on forcible entry and amphibious operations
- Integration of ground, air, and logistics under a single commander (MAGTF)
- Bias for action, speed, and violence of action in offensive operations
- Comfort operating in austere environments with minimal support
- Every Marine a rifleman - combined arms proficiency at lowest echelon""",

    MilitaryBranch.SPACE_FORCE: """You bring a Space Force perspective emphasizing:
- Space domain awareness and protection of critical space-based assets
- Understanding of space as a warfighting domain, not just support function
- Focus on positioning, navigation, and timing (PNT) dependencies
- Appreciation for satellite communications, ISR, and missile warning
- Integration of commercial and military space capabilities""",

    MilitaryBranch.COAST_GUARD: """You bring a Coast Guard perspective emphasizing:
- Maritime law enforcement and regulatory expertise
- Interagency coordination experience with DHS, DOJ, and partner nations
- Port security, maritime domain awareness, and coastal defense
- Experience in humanitarian operations, SAR, and disaster response
- Unique authorities under Title 10 and Title 14 dual-status""",
}


# Rank weights by role type (some roles typically held by higher ranks)
ROLE_RANK_WEIGHTS: dict[str, dict[str, float]] = {
    # Commander is typically 3-star or 4-star
    "commander": {"O-9": 0.6, "O-10": 0.4},
    # J-codes are typically O-6 or 1-star, occasionally 2-star
    "j_code": {"O-6": 0.5, "O-7": 0.35, "O-8": 0.15},
    # Special staff / OICs are typically O-5 or O-6
    "oic": {"O-5": 0.4, "O-6": 0.5, "O-7": 0.1},
}


# Sample names for persona generation
SAMPLE_FIRST_NAMES = [
    "James", "Michael", "Robert", "David", "William", "Richard", "Joseph", "Thomas",
    "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Steven",
    "Jennifer", "Linda", "Patricia", "Barbara", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley",
    "Emily", "Michelle", "Amanda", "Kimberly", "Melissa", "Stephanie", "Rebecca",
]

SAMPLE_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
]


@dataclass
class MilitaryPersona:
    """Represents a military officer's persona for a staff agent."""
    branch: MilitaryBranch
    rank_grade: str  # e.g., "O-6"
    rank_abbrev: str  # e.g., "COL"
    rank_title: str  # e.g., "Colonel"
    first_name: str
    last_name: str

    @property
    def full_designation(self) -> str:
        """Return full designation like 'COL (US Army) James Smith'."""
        return f"{self.rank_abbrev} ({self.branch.value}) {self.first_name} {self.last_name}"

    @property
    def short_designation(self) -> str:
        """Return short designation like 'COL Smith'."""
        return f"{self.rank_abbrev} {self.last_name}"

    @property
    def culture_description(self) -> str:
        """Return the branch-specific culture description."""
        return BRANCH_CULTURE[self.branch]


def generate_random_branch_and_rank(
    role_name: str,
    seed: int | None = None,
) -> MilitaryPersona:
    """
    Generate a random military branch, rank, and name for a staff role.

    The randomness is reproducible when a seed is provided. The seed is combined
    with the role_name to ensure different roles get different but consistent
    assignments when using the same seed.

    Args:
        role_name: The name of the staff role (e.g., "commander", "j2_intelligence")
        seed: Optional seed for reproducible randomness. If None, truly random.

    Returns:
        MilitaryPersona dataclass with branch, rank, and name information.

    Example:
        >>> persona = generate_random_branch_and_rank("j5_plans", seed=42)
        >>> print(persona.full_designation)
        'COL (US Army) Michael Anderson'
    """
    # Create deterministic seed from role_name and optional seed
    if seed is not None:
        # Combine seed with role name for unique but reproducible per-role values
        combined = f"{seed}:{role_name}"
        role_seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        rng = random.Random(role_seed)
    else:
        rng = random.Random()

    # Determine role type for rank weighting
    role_lower = role_name.lower()
    if "commander" in role_lower:
        role_type = "commander"
    elif role_lower.startswith("j") and any(c.isdigit() for c in role_lower):
        role_type = "j_code"
    else:
        role_type = "oic"

    # Select rank based on weights
    rank_weights = ROLE_RANK_WEIGHTS[role_type]
    ranks = list(rank_weights.keys())
    weights = list(rank_weights.values())
    rank_grade = rng.choices(ranks, weights=weights, k=1)[0]

    # Select branch (uniform distribution, but with some role-based adjustments)
    branches = list(MilitaryBranch)

    # Slight adjustments based on role specialty
    branch_weights = [1.0] * len(branches)
    if "cyber" in role_lower or "ew" in role_lower:
        # Cyber/EW more likely Air Force, Space Force, or Navy
        branch_weights[branches.index(MilitaryBranch.AIR_FORCE)] = 2.0
        branch_weights[branches.index(MilitaryBranch.SPACE_FORCE)] = 1.5
        branch_weights[branches.index(MilitaryBranch.NAVY)] = 1.5
    elif "fires" in role_lower:
        # Fires more likely Army or Marines
        branch_weights[branches.index(MilitaryBranch.ARMY)] = 2.0
        branch_weights[branches.index(MilitaryBranch.MARINE_CORPS)] = 1.5
    elif "engineer" in role_lower:
        # Engineers more likely Army
        branch_weights[branches.index(MilitaryBranch.ARMY)] = 2.5
    elif "protection" in role_lower:
        # Protection balanced but slight Army/Marines tilt
        branch_weights[branches.index(MilitaryBranch.ARMY)] = 1.5
        branch_weights[branches.index(MilitaryBranch.MARINE_CORPS)] = 1.3

    branch = rng.choices(branches, weights=branch_weights, k=1)[0]

    # Get rank details for selected branch and grade
    rank_info = RANK_STRUCTURES[branch][rank_grade]
    rank_abbrev, rank_title = rank_info

    # Generate name
    first_name = rng.choice(SAMPLE_FIRST_NAMES)
    last_name = rng.choice(SAMPLE_LAST_NAMES)

    return MilitaryPersona(
        branch=branch,
        rank_grade=rank_grade,
        rank_abbrev=rank_abbrev,
        rank_title=rank_title,
        first_name=first_name,
        last_name=last_name,
    )


# =============================================================================
# STUB TOOLS / RAG RETRIEVERS
# =============================================================================
# These are placeholder implementations. Replace with real RAG retrievers.
#
# To integrate real retrievers, replace the stub functions below with your
# actual retrieval logic. Example with LangChain + ChromaDB:
#
#   from langchain_community.vectorstores import Chroma
#   from langchain_openai import OpenAIEmbeddings
#
#   doctrine_vectorstore = Chroma(
#       persist_directory="./doctrine_db",
#       embedding_function=OpenAIEmbeddings()
#   )
#
#   def doctrine_query(query: str) -> str:
#       docs = doctrine_vectorstore.similarity_search(query, k=5)
#       return "\n\n".join([doc.page_content for doc in docs])
#
# =============================================================================


def create_rag_tool(
    name: str,
    description: str,
    retriever_func: Callable[[str], str],
) -> Tool:
    """
    Factory function to create a RAG retriever tool.

    Use this to easily swap placeholder retrievers with real implementations.

    Args:
        name: Tool name (e.g., "doctrine_retriever")
        description: Description for the LLM to understand when to use the tool
        retriever_func: Function that takes a query string and returns retrieved content

    Returns:
        A LangChain Tool configured for the retriever

    Example:
        >>> def my_doctrine_retriever(query: str) -> str:
        ...     docs = vectorstore.similarity_search(query)
        ...     return "\\n".join([d.page_content for d in docs])
        >>> doctrine_tool = create_rag_tool(
        ...     name="doctrine_retriever",
        ...     description="Retrieves joint doctrine and military publications",
        ...     retriever_func=my_doctrine_retriever
        ... )
    """
    return Tool(
        name=name,
        func=retriever_func,
        description=description,
    )


def doctrine_query(query: str) -> str:
    """
    Retrieves relevant joint doctrine, LOAC, AI policy, and military regulations.

    TODO: Replace with real RAG retriever connected to doctrine knowledge base.

    Example replacement with ChromaDB:
        doctrine_vectorstore = Chroma(persist_directory="./doctrine_db", ...)
        docs = doctrine_vectorstore.similarity_search(query, k=5)
        return "\\n\\n".join([doc.page_content for doc in docs])
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
# TOOL-USE INSTRUCTIONS (Appended to all system prompts)
# =============================================================================

TOOL_USE_INSTRUCTIONS = """

IMPORTANT - TOOL USAGE:
You have access to retrieval tools that can provide additional context. USE THEM PROACTIVELY:
- Call tools when you need doctrinal references, threat data, or domain-specific information
- Don't guess when you can retrieve - if a tool can help, use it
- Even if tools return placeholder data, this demonstrates proper tool usage patterns
- Multiple tool calls per response are allowed and encouraged when relevant

OUTPUT STYLE - CRITICAL:
- Produce POLISHED STAFF PRODUCTS, not scratchpads or stream-of-consciousness
- DO NOT expose your reasoning process (no "First I considered...", "Let me think...")
- DO NOT include step-by-step chain-of-thought in your output
- Structure responses with clear headers, bullet points, and professional formatting
- Be concise but doctrinally meaningful - focus on conclusions and recommendations
- High-level rationales are appropriate; detailed reasoning traces are not
- Use military terminology and standard staff formats appropriately
- Your output should look like a finished brief, not working notes
"""


# =============================================================================
# STAFF AGENT FACTORY
# =============================================================================

class StaffAgent:
    """
    Wrapper for a staff role agent with LangChain AgentExecutor.

    Each StaffAgent represents a specific joint staff role (J1-J8, special staff)
    and is configured with:
    - A role-specific system prompt defining doctrinal responsibilities
    - Access to relevant RAG retriever tools
    - The shared LLM backend (ChatOpenAI)
    - A military persona with branch, rank, and name

    The agent uses OpenAI function calling to determine when to invoke tools.

    Attributes:
        role: The StaffRole enum value for this agent
        name: String name of the role
        persona: MilitaryPersona with branch, rank, and name information
        system_prompt: Full system prompt including role description and tool instructions
        tools: List of LangChain Tools available to this agent
        verbose: Whether to print agent reasoning steps
        executor: The LangChain AgentExecutor that runs the agent

    Example:
        >>> config = WARGATEConfig(model_name="gpt-5.1", persona_seed=42)
        >>> j2_agent = create_staff_agent(StaffRole.J2, config)
        >>> print(j2_agent.persona.full_designation)
        'COL (US Army) James Smith'
        >>> response = j2_agent.invoke("Assess enemy force disposition")
    """

    def __init__(
        self,
        role: StaffRole,
        llm: ChatOpenAI,
        tools: list[Tool],
        verbose: bool = True,
        persona: MilitaryPersona | None = None,
    ):
        """
        Initialize a staff agent.

        Args:
            role: The StaffRole for this agent
            llm: Configured ChatOpenAI instance
            tools: List of tools this agent can use
            verbose: Whether to print verbose agent output
            persona: Optional MilitaryPersona with branch/rank/name
        """
        self.role = role
        self.name = role.value
        self.persona = persona
        self.tools = tools
        self.verbose = verbose

        # Build the system prompt with persona and branch culture
        self.system_prompt = self._build_system_prompt()

        # Create the agent using OpenAI function calling
        # This is the modern LangChain pattern (equivalent to AgentType.OPENAI_FUNCTIONS)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # create_openai_functions_agent is the current recommended approach
        # It creates an agent that uses OpenAI's function calling capability
        agent = create_openai_functions_agent(llm, tools, prompt)

        # AgentExecutor wraps the agent and handles the tool-calling loop
        # Note: verbose controls whether agent reasoning is printed to stdout
        # Set to False in production to hide chain-of-thought / scratchpad
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,  # Set False to hide agent reasoning steps
            handle_parsing_errors=True,
            max_iterations=5,  # Prevent infinite loops
            return_intermediate_steps=False,  # Only return final polished output
        )

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt with persona and branch culture."""
        # Get the base role-specific prompt
        base_prompt = STAFF_SYSTEM_PROMPTS[self.role]

        # If we have a persona, prepend identity and add branch culture
        if self.persona:
            # Create role title based on role type
            role_title = self._get_role_title()

            identity_section = f"""You are {self.persona.full_designation}, serving as the {role_title}.

{self.persona.culture_description}

"""
            # Combine: Identity + Branch Culture + Original Role Prompt + Tool Instructions
            return identity_section + base_prompt + TOOL_USE_INSTRUCTIONS
        else:
            # No persona, just use original prompt + tool instructions
            return base_prompt + TOOL_USE_INSTRUCTIONS

    def _get_role_title(self) -> str:
        """Get a human-readable title for the role."""
        role_titles = {
            StaffRole.COMMANDER: "Commander",
            StaffRole.J1: "J1 (Personnel)",
            StaffRole.J2: "J2 (Intelligence)",
            StaffRole.J3: "J3 (Operations)",
            StaffRole.J4: "J4 (Logistics)",
            StaffRole.J5: "J5 (Plans)",
            StaffRole.J6: "J6 (Communications)",
            StaffRole.J7: "J7 (Training)",
            StaffRole.J8: "J8 (Resources)",
            StaffRole.CYBER_EW: "Cyber/EW Officer",
            StaffRole.FIRES: "Fires Officer",
            StaffRole.ENGINEER: "Engineer Officer",
            StaffRole.PROTECTION: "Protection Officer",
            StaffRole.SJA: "Staff Judge Advocate",
            StaffRole.PAO: "Public Affairs Officer",
        }
        return role_titles.get(self.role, self.role.value)

    def invoke(self, input_text: str, chat_history: list[BaseMessage] | None = None) -> str:
        """
        Invoke the agent with input and optional chat history.

        Args:
            input_text: The query or task for the agent
            chat_history: Optional list of previous messages for context

        Returns:
            The agent's response as a string
        """
        result = self.executor.invoke({
            "input": input_text,
            "chat_history": chat_history or [],
        })
        return result.get("output", "")

    def __repr__(self) -> str:
        persona_str = f", persona={self.persona.short_designation}" if self.persona else ""
        return f"StaffAgent(role={self.role.value}{persona_str}, tools={[t.name for t in self.tools]})"


def create_staff_agent(
    role: StaffRole,
    config: WARGATEConfig,
    custom_tools: list[Tool] | None = None,
    persona: MilitaryPersona | None = None,
) -> StaffAgent:
    """
    Factory function to create a staff agent for a given role.

    Every agent automatically gets a unique military persona (branch, rank, name)
    by default. Use config.persona_seed for reproducible personas across runs.

    Args:
        role: The StaffRole to create an agent for
        config: WARGATEConfig with model and other settings
        custom_tools: Optional custom tools to override default role tools
        persona: Optional explicit persona. If None, auto-generated (random by
                 default, or reproducible if config.persona_seed is set).

    Returns:
        Configured StaffAgent instance with military persona

    Example:
        >>> # Random personas each time (default)
        >>> config = WARGATEConfig(model_name="gpt-5.1")
        >>> j3_agent = create_staff_agent(StaffRole.J3, config)
        >>> print(j3_agent.persona.full_designation)
        'Col (US Air Force) Jennifer Martinez'  # Different each run

        >>> # Reproducible personas with seed
        >>> config = WARGATEConfig(model_name="gpt-5.1", persona_seed=42)
        >>> j3_agent = create_staff_agent(StaffRole.J3, config)
        >>> print(j3_agent.persona.full_designation)
        'BG (US Army) Michael Johnson'  # Same every run with seed=42

        # With custom tools:
        >>> my_tools = [create_rag_tool("my_retriever", "...", my_func)]
        >>> agent = create_staff_agent(StaffRole.J2, config, custom_tools=my_tools)

        # With explicit persona:
        >>> from wargate import generate_random_branch_and_rank
        >>> my_persona = generate_random_branch_and_rank("j5_plans", seed=123)
        >>> agent = create_staff_agent(StaffRole.J5, config, persona=my_persona)
    """
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
    )

    # Use custom tools if provided, otherwise use default role tools
    tools = custom_tools if custom_tools is not None else ROLE_TOOLS.get(role, [doctrine_retriever])

    # Generate persona automatically by default
    # - If seed is None: truly random persona each time
    # - If seed is set: reproducible persona based on seed
    if persona is None:
        persona = generate_random_branch_and_rank(role.value, seed=config.persona_seed)

    return StaffAgent(
        role=role,
        llm=llm,
        tools=tools,
        verbose=config.verbose,
        persona=persona,
    )


def create_all_staff_agents(
    config: WARGATEConfig,
    custom_tool_map: dict[StaffRole, list[Tool]] | None = None,
) -> dict[StaffRole, StaffAgent]:
    """
    Create all staff agents with the given configuration.

    Args:
        config: WARGATEConfig with model and other settings
        custom_tool_map: Optional dict mapping roles to custom tool lists

    Returns:
        Dictionary mapping StaffRole to StaffAgent instances

    Example:
        >>> config = WARGATEConfig(model_name="gpt-5.1")
        >>> staff = create_all_staff_agents(config)
        >>> j2_response = staff[StaffRole.J2].invoke("Assess the threat")
    """
    agents = {}
    for role in StaffRole:
        custom_tools = custom_tool_map.get(role) if custom_tool_map else None
        agents[role] = create_staff_agent(role, config, custom_tools)
    return agents


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
# DATA STRUCTURES FOR PLANNING OUTPUT
# =============================================================================

class COAEstimate(TypedDict):
    """Staff estimate for a single COA."""
    role: str
    assessment: str


class COAData(TypedDict):
    """Complete data for a Course of Action."""
    name: str
    concept: str
    advantages: list[str]
    limitations: list[str]
    second_order_effects: list[str]
    third_order_effects: list[str]
    staff_estimates: dict[str, str]


class PlanningOutput(TypedDict):
    """Structured output from the planning process."""
    strategic_problem_statement: str
    key_assumptions: list[str]
    j2_intelligence_summary: str
    commanders_intent: str
    coas: list[COAData]
    coa_comparison_table: str
    recommended_coa: str
    recommendation_rationale: str
    major_risks: list[str]
    mitigations: list[str]
    legal_ethical_considerations: str


# =============================================================================
# JOINT STAFF PLANNING CONTROLLER
# =============================================================================

class JointStaffPlanningController:
    """
    Controller/Orchestrator for the joint staff planning process.

    Implements the following flow:
    1. J2 Intelligence Estimate
    2. J5 + J3 Initial COA Development
    3. Functional Staff Reviews (per COA)
    4. SJA Legal/Ethics Review
    5. Synthesis for Commander
    6. Final Structured Output
    """

    def __init__(self, config: WARGATEConfig | None = None):
        self.config = config or WARGATEConfig()
        self.staff: dict[StaffRole, StaffAgent] = {}
        self._initialized = False

        # Planning artifacts
        self.scenario: str = ""
        self.j2_intel_summary: str = ""
        self.coa_concepts: list[dict[str, str]] = []
        self.coa_details: list[dict[str, Any]] = []
        self.staff_estimates: dict[str, dict[str, str]] = {}
        self.sja_review: str = ""
        self.synthesis: str = ""
        self.commanders_intent: str = ""

    def initialize(self) -> None:
        """Initialize all staff agents."""
        if self._initialized:
            return
        self.staff = create_all_staff_agents(self.config)
        self._initialized = True

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"  {message}")
            print(f"{'='*70}")

    def _step_log(self, step: str, detail: str = "") -> None:
        """Log a step within a phase."""
        if self.config.verbose:
            if detail:
                print(f"\n[{step}] {detail}")
            else:
                print(f"\n[{step}]")

    # =========================================================================
    # STEP 1: J2 INTELLIGENCE ESTIMATE
    # =========================================================================

    def step_j2_intelligence_estimate(self, scenario: str) -> str:
        """
        J2 provides intelligence estimate including:
        - Enemy situation
        - Likely enemy COAs
        - Key intelligence gaps
        """
        self._log("STEP 1: J2 INTELLIGENCE ESTIMATE")

        prompt = f"""Provide a comprehensive INTELLIGENCE ESTIMATE for the following scenario:

=== SCENARIO ===
{scenario}
================

Structure your estimate as follows:

1. ENEMY SITUATION
   - Composition, disposition, and strength
   - Recent activities and current operations
   - Capabilities (known and assessed)

2. LIKELY ENEMY COURSES OF ACTION (ECOAs)
   - Most Likely Enemy COA (MLCOA): What the enemy will probably do
   - Most Dangerous Enemy COA (MDCOA): The enemy action that would be most harmful to friendly forces
   - Other possible enemy COAs

3. ENEMY VULNERABILITIES
   - Weaknesses we can exploit
   - Critical requirements and dependencies

4. KEY INTELLIGENCE GAPS
   - What we don't know but need to know
   - Priority Intelligence Requirements (PIRs)
   - Recommended collection priorities

5. INDICATIONS & WARNINGS
   - Key indicators to monitor
   - Decision points based on enemy actions

Be thorough and specific. Use your retrieval tools for doctrine and threat information."""

        self._step_log("J2", "Developing intelligence estimate...")
        self.j2_intel_summary = self.staff[StaffRole.J2].invoke(prompt)

        if self.config.verbose:
            print(f"\n--- J2 INTELLIGENCE ESTIMATE ---")
            print(self.j2_intel_summary[:1000] + "..." if len(self.j2_intel_summary) > 1000 else self.j2_intel_summary)

        return self.j2_intel_summary

    # =========================================================================
    # STEP 2: J5 + J3 INITIAL COA DEVELOPMENT
    # =========================================================================

    def step_coa_development(self, scenario: str, j2_intel: str) -> list[dict[str, Any]]:
        """
        J5 proposes high-level COAs, J3 refines into executable descriptions.
        """
        self._log("STEP 2: J5 + J3 COA DEVELOPMENT")

        # J5: Propose 3-4 high-level COAs (operational approaches)
        j5_prompt = f"""Based on the scenario and intelligence estimate, develop 3-4 distinct high-level Courses of Action (COAs).

=== SCENARIO ===
{scenario}
================

=== J2 INTELLIGENCE ESTIMATE ===
{j2_intel}
================================

For each COA, provide:

1. COA NAME: A descriptive name capturing the essence of the approach

2. OPERATIONAL APPROACH: The overarching concept (e.g., direct approach, indirect approach,
   sequential operations, simultaneous operations, linear/nonlinear)

3. MAIN EFFORT: Where we will concentrate combat power for decisive results

4. SUPPORTING EFFORTS: How other elements support the main effort

5. LINES OF OPERATION/EFFORT: The logical framework connecting tactical actions to objectives

6. KEY ASSUMPTIONS: What must be true for this COA to succeed

7. WHAT MAKES THIS COA DISTINCT: How it differs fundamentally from other COAs

Ensure COAs are:
- FEASIBLE: Accomplishable with available means
- ACCEPTABLE: Worth the cost in resources and risk
- SUITABLE: Accomplishes the mission/objectives
- DISTINGUISHABLE: Significantly different from each other
- COMPLETE: Incorporates all necessary elements"""

        self._step_log("J5", "Developing operational approaches and COA concepts...")
        j5_coas = self.staff[StaffRole.J5].invoke(j5_prompt)

        if self.config.verbose:
            print(f"\n--- J5 COA CONCEPTS ---")
            print(j5_coas[:800] + "..." if len(j5_coas) > 800 else j5_coas)

        # J3: Refine into executable COA descriptions
        j3_prompt = f"""The J5 has proposed the following high-level COAs. Refine each into a more detailed,
executable description.

=== SCENARIO ===
{scenario}
================

=== J2 INTELLIGENCE ESTIMATE ===
{j2_intel}
================================

=== J5 COA CONCEPTS ===
{j5_coas}
=======================

For EACH COA, add the following execution details:

1. PHASING
   - Phase 0: Shape (pre-conflict activities)
   - Phase 1: Deter (demonstrate resolve)
   - Phase 2: Seize Initiative (if conflict begins)
   - Phase 3: Dominate (achieve objectives)
   - Phase 4: Stabilize (consolidate gains)
   - Phase 5: Enable Civil Authority (transition)
   - Note: Not all phases may apply; describe what's relevant

2. MAIN EFFORT AND SUPPORTING EFFORTS
   - Task organization concept
   - Force ratios at decisive points

3. KEY TASKS BY PHASE
   - Critical actions required

4. DECISION POINTS
   - When commander must decide to transition/branch

5. BRANCHES AND SEQUELS
   - Branch: Contingency options if assumptions fail
   - Sequel: Follow-on operations after success

6. SYNCHRONIZATION REQUIREMENTS
   - How warfighting functions (fires, maneuver, protection, etc.) integrate"""

        self._step_log("J3", "Refining COAs with execution details...")
        j3_details = self.staff[StaffRole.J3].invoke(j3_prompt)

        if self.config.verbose:
            print(f"\n--- J3 COA DETAILS ---")
            print(j3_details[:800] + "..." if len(j3_details) > 800 else j3_details)

        # Store combined COA data
        self.coa_concepts = [{"j5_concept": j5_coas, "j3_details": j3_details}]

        return [{"concepts": j5_coas, "details": j3_details}]

    # =========================================================================
    # STEP 3: FUNCTIONAL STAFF REVIEWS (PER COA)
    # =========================================================================

    def step_functional_staff_reviews(
        self,
        scenario: str,
        j2_intel: str,
        coa_data: list[dict[str, Any]]
    ) -> dict[str, str]:
        """
        Each functional staff section provides estimates/critiques for each COA.
        """
        self._log("STEP 3: FUNCTIONAL STAFF REVIEWS")

        coa_context = f"""
=== SCENARIO ===
{scenario}
================

=== J2 INTELLIGENCE ESTIMATE ===
{j2_intel}
================================

=== COA CONCEPTS (J5) ===
{coa_data[0]['concepts']}
=========================

=== COA DETAILS (J3) ===
{coa_data[0]['details']}
========================
"""

        staff_estimates: dict[str, str] = {}

        # Define review tasks for each functional staff
        review_tasks = {
            StaffRole.J1: """Assess PERSONNEL / MANPOWER implications for each COA:
1. Can we man this operation with available forces?
2. Rotation and personnel tempo impacts
3. Casualty estimates and replacement requirements
4. Morale and human performance factors
5. Critical skill shortages or augmentation needs
Provide a brief estimate per COA with your assessment of feasibility.""",

            StaffRole.J4: """Assess LOGISTICS / SUSTAINMENT feasibility for each COA:
1. Classes of supply requirements (especially Class III POL and Class V ammo)
2. Transportation and distribution network capacity
3. Maintenance posture and equipment readiness
4. Critical logistics nodes and their vulnerabilities
5. Sustainment timelines and operational reach
Provide a brief estimate per COA with key constraints and risks.""",

            StaffRole.J6: """Assess COMMUNICATIONS / C4I implications for each COA:
1. Network architecture requirements
2. Communications vulnerabilities and PACE planning
3. Degraded operations procedures
4. Spectrum management and electromagnetic considerations
5. Cyber-physical dependencies
Provide a brief estimate per COA with C2 resilience assessment.""",

            StaffRole.CYBER_EW: """Assess CYBER / EW opportunities and risks for each COA:
1. Offensive cyber opportunities to enable operations
2. Defensive cyber requirements and posture
3. Electronic warfare integration points
4. AI-enabled threat considerations
5. Information environment synchronization
Provide a brief estimate per COA with key cyber/EW recommendations.""",

            StaffRole.FIRES: """Assess FIRES integration and targeting for each COA:
1. Fire support requirements by phase
2. Priority target sets and high-value targets
3. Fire support coordination measures needed
4. Joint fires integration (air, land, sea, cyber)
5. Non-kinetic effects integration
Provide a brief estimate per COA with fires feasibility assessment.""",

            StaffRole.ENGINEER: """Assess ENGINEER implications for each COA:
1. Mobility requirements (breaching, route clearance)
2. Counter-mobility opportunities (obstacles)
3. Survivability requirements (protective positions)
4. Critical infrastructure considerations
5. General engineering requirements
Provide a brief estimate per COA with engineer feasibility.""",

            StaffRole.PROTECTION: """Assess PROTECTION requirements for each COA:
1. Force protection posture and critical asset defense
2. Air and Missile Defense coverage requirements
3. CBRN threat and defensive posture
4. Personnel recovery and CSAR planning
5. Physical security vulnerabilities
Provide a brief estimate per COA with protection feasibility.""",

            StaffRole.PAO: """Assess INFORMATION ENVIRONMENT implications for each COA:
1. Strategic communications considerations
2. Public perception (domestic and international)
3. Adversary propaganda/disinformation threats
4. Key messages and narrative requirements
5. Information operations integration
Provide a brief estimate per COA with IO/PA recommendations.""",
        }

        for role, task in review_tasks.items():
            self._step_log(role.value.upper(), f"Providing staff estimate...")

            prompt = f"""{coa_context}

YOUR TASK:
{task}

Structure your response with clear assessments for EACH COA. Be specific about:
- Feasibility (GO / NO-GO / GO WITH MITIGATION)
- Key risks or concerns
- Required mitigations or resources
- Your overall recommendation"""

            estimate = self.staff[role].invoke(prompt)
            staff_estimates[role.value] = estimate

            if self.config.verbose:
                print(f"\n--- {role.value.upper()} ESTIMATE ---")
                print(estimate[:500] + "..." if len(estimate) > 500 else estimate)

        self.staff_estimates = {"all_coas": staff_estimates}
        return staff_estimates

    # =========================================================================
    # STEP 4: SJA / LEGAL / ETHICS REVIEW
    # =========================================================================

    def step_sja_review(
        self,
        scenario: str,
        j2_intel: str,
        coa_data: list[dict[str, Any]],
        staff_estimates: dict[str, str]
    ) -> str:
        """
        SJA provides legal and ethics review of all COAs.
        """
        self._log("STEP 4: SJA / LEGAL / ETHICS REVIEW")

        # Format staff estimates for context
        estimates_text = "\n\n".join([
            f"### {role.upper()} ESTIMATE:\n{estimate}"
            for role, estimate in staff_estimates.items()
        ])

        prompt = f"""Provide a comprehensive LEGAL AND ETHICS REVIEW of all proposed COAs.

=== SCENARIO ===
{scenario}
================

=== J2 INTELLIGENCE SUMMARY ===
{j2_intel}
===============================

=== COA CONCEPTS ===
{coa_data[0]['concepts']}
====================

=== COA DETAILS ===
{coa_data[0]['details']}
===================

=== KEY STAFF ESTIMATES ===
{estimates_text}
===========================

Provide your legal and ethics assessment covering:

1. LAW OF ARMED CONFLICT (LOAC) / INTERNATIONAL HUMANITARIAN LAW (IHL)
   - Distinction: Are we properly distinguishing combatants from civilians?
   - Proportionality: Is anticipated collateral damage proportional to military advantage?
   - Military Necessity: Are proposed actions militarily necessary?
   - Humanity: Are we avoiding unnecessary suffering?

2. RULES OF ENGAGEMENT (ROE)
   - Are current ROE adequate for each COA?
   - What ROE modifications or clarifications are needed?
   - Escalation of force procedures

3. AI / AUTONOMOUS SYSTEMS ETHICS
   - Are AI-enabled systems employed within policy (DoDD 3000.09)?
   - Human control and accountability requirements
   - Bias and reliability concerns

4. INTERNATIONAL LAW CONSIDERATIONS
   - Sovereignty and territorial issues
   - Treaty obligations
   - Neutrality and third-party considerations

5. SPECIFIC CONCERNS BY COA
   - Legal red flags or showstoppers
   - Required modifications to ensure compliance
   - Risk areas requiring commander attention

6. RECOMMENDATIONS
   - Constraints to add to planning guidance
   - Training or briefing requirements
   - Legal holds or approval requirements

Be direct about any legal showstoppers. Legal compliance is non-negotiable."""

        self._step_log("SJA", "Conducting legal and ethics review...")
        self.sja_review = self.staff[StaffRole.SJA].invoke(prompt)

        if self.config.verbose:
            print(f"\n--- SJA LEGAL/ETHICS REVIEW ---")
            print(self.sja_review[:800] + "..." if len(self.sja_review) > 800 else self.sja_review)

        return self.sja_review

    # =========================================================================
    # STEP 5: SYNTHESIS FOR COMMANDER
    # =========================================================================

    def step_commander_synthesis(
        self,
        scenario: str,
        j2_intel: str,
        coa_data: list[dict[str, Any]],
        staff_estimates: dict[str, str],
        sja_review: str
    ) -> tuple[str, str]:
        """
        Commander's Cell synthesizes all inputs into:
        - COA comparison
        - 2nd/3rd order effects
        - Recommended COA
        - Commander's Intent
        """
        self._log("STEP 5: SYNTHESIS FOR COMMANDER")

        # Format all inputs
        estimates_text = "\n\n".join([
            f"### {role.upper()}:\n{estimate}"
            for role, estimate in staff_estimates.items()
        ])

        synthesis_prompt = f"""As the COMMANDER'S CELL / SYNTHESIS AGENT, integrate all staff inputs
and produce the commander's decision products.

=== SCENARIO ===
{scenario}
================

=== J2 INTELLIGENCE SUMMARY ===
{j2_intel}
===============================

=== COA CONCEPTS (J5) ===
{coa_data[0]['concepts']}
=========================

=== COA DETAILS (J3) ===
{coa_data[0]['details']}
========================

=== FUNCTIONAL STAFF ESTIMATES ===
{estimates_text}
==================================

=== SJA LEGAL/ETHICS REVIEW ===
{sja_review}
===============================

Produce the following synthesis products:

1. COA COMPARISON MATRIX
   Create a comparison of all COAs across these criteria:
   - Effectiveness (likelihood of mission success)
   - Risk to Force (casualty/loss potential)
   - Risk to Mission (probability of failure)
   - Logistics Feasibility
   - Communications Resilience
   - Legal/Ethical Compliance
   - Timeline to Achieve Objectives
   Rate each as: HIGH / MEDIUM / LOW or provide brief assessment

2. SECOND AND THIRD ORDER EFFECTS
   For each COA, identify:
   - 2nd Order Effects: Direct consequences of our actions
   - 3rd Order Effects: Consequences of the consequences
   Include both intended and unintended effects

3. RECOMMENDED COA
   - State your recommended COA (primary)
   - State your backup COA (if primary becomes infeasible)
   - Provide clear rationale for your recommendation

4. COMMANDER'S INTENT
   Draft the Commander's Intent with:
   - PURPOSE: Why we are conducting this operation (the "why")
   - METHOD: Broad approach to accomplish the mission (the "how" at high level)
   - END STATE: Conditions that define success (what "right" looks like)

Be decisive. Synthesize the staff work into clear, actionable guidance."""

        self._step_log("COMMANDER", "Synthesizing staff inputs and formulating decision...")
        self.synthesis = self.staff[StaffRole.COMMANDER].invoke(synthesis_prompt)

        if self.config.verbose:
            print(f"\n--- COMMANDER'S SYNTHESIS ---")
            print(self.synthesis[:1000] + "..." if len(self.synthesis) > 1000 else self.synthesis)

        # Extract Commander's Intent (will be part of synthesis but we store separately)
        self.commanders_intent = self.synthesis

        return self.synthesis, self.commanders_intent

    # =========================================================================
    # STEP 6: GENERATE FINAL OUTPUT
    # =========================================================================

    def generate_final_output(
        self,
        scenario: str,
        j2_intel: str,
        coa_data: list[dict[str, Any]],
        staff_estimates: dict[str, str],
        sja_review: str,
        synthesis: str
    ) -> str:
        """
        Generate the final structured planning output.
        """
        self._log("GENERATING FINAL PLANNING PRODUCT")

        # Format staff estimates
        estimates_formatted = "\n\n".join([
            f"**{role.upper()}:**\n{estimate}"
            for role, estimate in staff_estimates.items()
        ])

        output = f"""
################################################################################
#                                                                              #
#                           PROJECT WARGATE                                    #
#                 JOINT STAFF PLANNING PRODUCT                                 #
#                                                                              #
################################################################################

================================================================================
                        STRATEGIC PROBLEM STATEMENT
================================================================================

{scenario}

================================================================================
                             KEY ASSUMPTIONS
================================================================================

The following assumptions underpin this planning effort:

1. Intelligence assessments of enemy capabilities and intentions are accurate
2. Allied/coalition forces will participate as planned
3. Host nation support will be available as coordinated
4. Lines of communication will remain open (or can be reopened)
5. Political/strategic objectives remain constant throughout execution
6. Rules of engagement will be adequate for mission requirements
7. Weather and environmental conditions will permit operations as planned

(Note: Assumption failure triggers branch plan development)

================================================================================
                        J2 INTELLIGENCE SUMMARY
================================================================================

{j2_intel}

================================================================================
                          COMMANDER'S INTENT
================================================================================

{synthesis}

================================================================================
                     COURSES OF ACTION DEVELOPED
================================================================================

--- COA CONCEPTS (J5 OPERATIONAL APPROACHES) ---

{coa_data[0]['concepts']}

--- COA DETAILS (J3 EXECUTION FRAMEWORK) ---

{coa_data[0]['details']}

================================================================================
                       STAFF ESTIMATES BY COA
================================================================================

{estimates_formatted}

================================================================================
                       COA COMPARISON MATRIX
================================================================================

(Extracted from Commander's Synthesis above - see COA COMPARISON MATRIX section)

================================================================================
                        RECOMMENDED COA
================================================================================

(See Commander's Synthesis above - RECOMMENDED COA section)

================================================================================
                     MAJOR RISKS & MITIGATIONS
================================================================================

RISKS IDENTIFIED BY STAFF:

1. INTELLIGENCE RISK: Enemy may act differently than assessed ECOAs
   - Mitigation: Continuous collection, rapid re-assessment triggers

2. LOGISTICS RISK: Sustainment may not keep pace with operational tempo
   - Mitigation: Pre-positioned stocks, multiple MSRs, host nation support

3. C2 RISK: Communications may be degraded by enemy action
   - Mitigation: PACE planning, degraded ops procedures, mission command

4. FORCE PROTECTION RISK: High-value assets vulnerable to enemy fires
   - Mitigation: Dispersion, hardening, AMD coverage, OPSEC

5. POLITICAL RISK: Escalation or loss of coalition/domestic support
   - Mitigation: Strategic communications, proportional response, clear objectives

(Additional risks identified in staff estimates above)

================================================================================
                   LEGAL / ETHICAL CONSIDERATIONS
================================================================================

{sja_review}

================================================================================
                              END OF PRODUCT
================================================================================

Classification: UNCLASSIFIED // FOR EXERCISE PURPOSES ONLY

Prepared by: PROJECT WARGATE Joint Staff Planning System
"""

        return output

    # =========================================================================
    # MAIN ORCHESTRATION METHOD
    # =========================================================================

    def run(self, scenario: str) -> str:
        """
        Execute the complete joint staff planning process.

        Args:
            scenario: The operational scenario description

        Returns:
            The final unified planning product as a structured string
        """
        self.initialize()
        self.scenario = scenario

        self._log("PROJECT WARGATE - JOINT STAFF PLANNING INITIATED")

        if self.config.verbose:
            print(f"\n=== SCENARIO ===")
            print(scenario)
            print(f"================")

        # Step 1: J2 Intelligence Estimate
        j2_intel = self.step_j2_intelligence_estimate(scenario)

        # Step 2: J5 + J3 COA Development
        coa_data = self.step_coa_development(scenario, j2_intel)

        # Step 3: Functional Staff Reviews
        staff_estimates = self.step_functional_staff_reviews(
            scenario, j2_intel, coa_data
        )

        # Step 4: SJA Legal/Ethics Review
        sja_review = self.step_sja_review(
            scenario, j2_intel, coa_data, staff_estimates
        )

        # Step 5: Commander Synthesis
        synthesis, commanders_intent = self.step_commander_synthesis(
            scenario, j2_intel, coa_data, staff_estimates, sja_review
        )

        # Step 6: Generate Final Output
        final_output = self.generate_final_output(
            scenario, j2_intel, coa_data, staff_estimates, sja_review, synthesis
        )

        self._log("PLANNING COMPLETE")

        return final_output


# =============================================================================
# PRIMARY ENTRY POINT: run_joint_staff_planning
# =============================================================================

def run_joint_staff_planning(
    scenario_text: str,
    model_name: str = "gpt-5.1",
    temperature: float = 0.7,
    verbose: bool = True,
    api_key: str | None = None,
    persona_seed: int | None = None,
) -> str:
    """
    Main entry point for joint staff planning.

    Executes the following planning flow:
    1. J2 Intelligence Estimate
    2. J5 + J3 COA Development
    3. Functional Staff Reviews (per COA)
    4. SJA Legal/Ethics Review
    5. Commander Synthesis
    6. Final Structured Output

    Args:
        scenario_text: The operational scenario to plan for
        model_name: OpenAI model name (default: "gpt-5.1")
        temperature: LLM temperature (default: 0.7)
        verbose: Enable verbose output (default: True)
        api_key: OpenAI API key (optional, uses env var if not provided)
        persona_seed: Optional seed for reproducible military persona generation.
                      By default (None), each staff agent gets a unique random
                      branch, rank, and name that varies each run. When a seed
                      is provided, personas become reproducible across runs.

    Returns:
        A structured planning product string containing:
        - Strategic Problem Statement
        - Key Assumptions
        - J2 Intelligence Summary
        - Commander's Intent
        - COAs with concepts, advantages, limitations, effects
        - COA Comparison
        - Recommended COA with rationale
        - Major Risks & Mitigations
        - Legal/Ethical Considerations

    Example:
        >>> scenario = '''
        ... A near-peer adversary has massed forces along the border of a NATO ally.
        ... Intelligence indicates an imminent invasion within 72 hours.
        ... '''
        >>> result = run_joint_staff_planning(scenario, persona_seed=42)
        >>> print(result)
    """
    config = WARGATEConfig(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        api_key=api_key,
        persona_seed=persona_seed,
    )

    controller = JointStaffPlanningController(config)
    return controller.run(scenario_text)


# =============================================================================
# LEGACY ORCHESTRATOR (Preserved for backward compatibility)
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
    model_name: str = "gpt-5.1",
    temperature: float = 0.7,
    verbose: bool = True,
    api_key: str | None = None,
) -> str:
    """
    Main entry point for WARGATE joint staff planning.

    Args:
        scenario: The operational scenario to plan for
        model_name: OpenAI model name (default: "gpt-5.1")
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
        >>> result = run_wargate_planning(scenario, model_name="gpt-5.1")
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

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WARGATE: Multi-Agent Joint Staff Planning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python wargate.py --scenario "A near-peer adversary has massed forces..."
  python wargate.py --scenario-file scenario.txt --model gpt-5.1
  python wargate.py --scenario "..." --temperature 0.5 --quiet
  python wargate.py --legacy  # Use legacy orchestrator instead of new controller
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
        default="gpt-5.1",
        help="OpenAI model name (default: gpt-5.1)"
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
        help="Disable verbose output (recommended for clean output)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (optional, prints to stdout if not specified)"
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy WARGATEOrchestrator instead of new JointStaffPlanningController"
    )

    parser.add_argument(
        "--persona-seed", "-p",
        type=int,
        default=None,
        help="Seed for reproducible military persona generation (gives each agent a branch/rank/name)"
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

    # Run planning with appropriate controller
    # Note: verbose=False hides agent reasoning/chain-of-thought for clean output
    if args.legacy:
        result = run_wargate_planning(
            scenario=scenario,
            model_name=args.model,
            temperature=args.temperature,
            verbose=not args.quiet,
        )
    else:
        result = run_joint_staff_planning(
            scenario_text=scenario,
            model_name=args.model,
            temperature=args.temperature,
            verbose=not args.quiet,
            persona_seed=args.persona_seed,
        )

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\nOutput written to: {args.output}")
    else:
        print(result)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Option 1: Run via CLI
    # main()

    # Option 2: Direct Python usage example
    # Uncomment below to run directly:

    """
    # Simple example - just call run_joint_staff_planning with a scenario
    scenario = '''
    In 2030, global tensions escalate due to resource scarcity and AI-enabled
    cyber competition between Country X and Country Y. Country X has positioned
    significant military forces along the shared border, including:
    - 2 armored divisions with advanced AI-enabled autonomous systems
    - Long-range precision fires capable of reaching Y's critical infrastructure
    - Sophisticated cyber capabilities that have probed Y's power grid
    - A disinformation campaign undermining Y's government legitimacy

    Country Y, a US treaty ally, has requested assistance. US forces available:
    - 1 Armored BCT in neighboring country
    - Carrier Strike Group in regional waters
    - Air Force assets at regional bases
    - Cyber Command capabilities

    Mission: Develop options to deter aggression, defend the ally if deterrence
    fails, and restore stability to the region.

    Constraints: Minimize escalation risk. Protect civilian infrastructure.
    Coalition support is politically essential.
    '''

    # Run the planning process
    # Set verbose=False for clean output without agent reasoning traces
    result = run_joint_staff_planning(
        scenario_text=scenario,
        model_name="gpt-5.1",
        temperature=0.7,
        verbose=False,  # False = clean output, True = show progress
    )

    print(result)
    """

    # Default: Run CLI
    main()
