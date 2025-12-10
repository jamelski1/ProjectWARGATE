# ProjectWARGATE

**W**argaming **A**gent for **R**apid **G**eneration of **A**ctions, **T**hreats, and **E**stimates

A multi-agent LangChain system that mirrors a joint military staff doing operational planning and COA (Course of Action) development using OpenAI models.

## Overview

WARGATE implements a complete joint staff planning cell with 15 specialized AI agents representing different staff functions. The system follows Joint Planning Process (JP 5-0) doctrine to analyze scenarios, develop courses of action, war-game options, and produce unified operational plans.

## Features

- **15 Specialized Staff Agents**: Commander, J1-J8, Cyber/EW, Fires, Engineer, Protection, SJA, and PAO
- **Structured Planning Flow**: J2 Intel → J5/J3 COA Dev → Staff Reviews → SJA Review → Commander Synthesis
- **Red Team Integration**: J2 Intelligence agent challenges assumptions and war-games enemy responses
- **RAG-Ready Architecture**: Stub tools ready for doctrine, geopolitics, logistics, and cyber intel retrievers
- **Configurable LLM Backend**: Use any OpenAI model with adjustable parameters

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Quick Start

```python
from wargate import run_joint_staff_planning

scenario = """
A near-peer adversary has massed forces along the border of a NATO ally.
Intelligence indicates an imminent invasion within 72 hours.
Develop options for deterrence and defense.
"""

result = run_joint_staff_planning(
    scenario_text=scenario,
    model_name="gpt-4.1",
    temperature=0.7,
)

print(result)
```

## CLI Usage

```bash
# Run with inline scenario (uses new JointStaffPlanningController)
python wargate.py --scenario "Your scenario description here"

# Run with scenario from file
python wargate.py --scenario-file scenario.txt

# Customize model and temperature
python wargate.py -s "Scenario..." --model gpt-4.1 --temperature 0.5

# Save output to file
python wargate.py -s "Scenario..." --output plan.txt

# Run quietly (minimal console output)
python wargate.py -s "Scenario..." --quiet

# Use legacy orchestrator
python wargate.py -s "Scenario..." --legacy
```

## Planning Flow

The `run_joint_staff_planning()` function executes the following orchestrated flow:

### Step 1: J2 Intelligence Estimate
- Enemy situation assessment
- Likely enemy COAs (MLCOA/MDCOA)
- Key intelligence gaps and PIRs
- Indications & warnings

### Step 2: J5 + J3 COA Development
- **J5**: Proposes 3-4 high-level operational approaches
- **J3**: Refines into executable COA descriptions with:
  - Phasing (Shape → Deter → Seize Initiative → Dominate → Stabilize → Enable)
  - Main effort and supporting efforts
  - Decision points, branches, and sequels

### Step 3: Functional Staff Reviews
Each staff section provides estimates per COA:
- **J1**: Personnel/manpower feasibility
- **J4**: Logistics/sustainment constraints
- **J6**: Communications/C4I resilience
- **Cyber/EW**: Cyber and EW opportunities/risks
- **Fires**: Targeting and fires integration
- **Engineer**: Mobility/counter-mobility/survivability
- **Protection**: Force protection and AMD
- **PAO/IO**: Information environment implications

### Step 4: SJA Legal/Ethics Review
- LOAC/IHL compliance
- ROE adequacy
- AI/autonomous systems ethics
- International law considerations

### Step 5: Synthesis for Commander
- COA comparison matrix
- 2nd and 3rd order effects analysis
- Recommended COA with rationale
- Commander's Intent (Purpose, Method, End State)

### Step 6: Final Output
Structured planning product containing:
- Strategic Problem Statement
- Key Assumptions
- J2 Intelligence Summary
- Commander's Intent
- COAs (concepts, advantages, limitations, effects)
- COA Comparison
- Recommended COA
- Major Risks & Mitigations
- Legal/Ethical Considerations

## Staff Roles

| Role | Description |
|------|-------------|
| **Commander** | Final decision authority, Commander's Intent, COA selection |
| **J1** | Personnel, manpower, force availability, morale |
| **J2** | Intelligence, threat assessment, Red Team lead |
| **J3** | Operations, COA execution, synchronization |
| **J4** | Logistics, sustainment, supply chain |
| **J5** | Plans, campaign design, branches & sequels |
| **J6** | Communications, C4I, network resilience |
| **J7** | Training, lessons learned, doctrine |
| **J8** | Resources, force structure, budget |
| **Cyber/EW** | Cyber operations, electronic warfare, AI threats |
| **Fires** | Kinetic/non-kinetic fires, targeting |
| **Engineer** | Mobility, counter-mobility, survivability |
| **Protection** | Force protection, AMD, CBRN |
| **SJA** | Legal advisor, LOAC, ethics |
| **PAO/IO** | Strategic communications, information ops |

## Step-by-Step Execution

For fine-grained control, use the controller directly:

```python
from wargate import JointStaffPlanningController, WARGATEConfig

config = WARGATEConfig(model_name="gpt-4.1", temperature=0.7)
controller = JointStaffPlanningController(config)
controller.initialize()

# Execute steps individually
j2_intel = controller.step_j2_intelligence_estimate(scenario)
coa_data = controller.step_coa_development(scenario, j2_intel)
staff_estimates = controller.step_functional_staff_reviews(scenario, j2_intel, coa_data)
sja_review = controller.step_sja_review(scenario, j2_intel, coa_data, staff_estimates)
synthesis, intent = controller.step_commander_synthesis(
    scenario, j2_intel, coa_data, staff_estimates, sja_review
)
final_output = controller.generate_final_output(
    scenario, j2_intel, coa_data, staff_estimates, sja_review, synthesis
)
```

## Extending with RAG

The system includes stub retriever tools ready for connection to vector stores:

```python
def doctrine_query(query: str) -> str:
    # TODO: Connect to your doctrine vector store
    # Example with ChromaDB:
    # docs = doctrine_vectorstore.similarity_search(query)
    # return "\n".join([d.page_content for d in docs])
    return f"[DOCTRINE_PLACEHOLDER] {query}"
```

Available retriever stubs:
- `doctrine_retriever` - Joint doctrine, LOAC, AI policy
- `geopolitics_retriever` - Regional dynamics, international relations
- `logistics_retriever` - Sustainment, supply chain
- `cyberintel_retriever` - Cyber threat intelligence
- `terrain_retriever` - Geographic analysis
- `orbat_retriever` - Order of battle, force structure

## Architecture

```
run_joint_staff_planning()
└── JointStaffPlanningController
    ├── WARGATEConfig (model, temperature, etc.)
    ├── StaffAgent[] (15 agents)
    │   ├── StaffRole (enum)
    │   ├── System Prompt (doctrinal role)
    │   ├── Tools (RAG retrievers)
    │   └── AgentExecutor (LangChain)
    └── Planning Steps
        ├── step_j2_intelligence_estimate()
        ├── step_coa_development()
        ├── step_functional_staff_reviews()
        ├── step_sja_review()
        ├── step_commander_synthesis()
        └── generate_final_output()
```

## Example Output Structure

```
================================================================================
                        STRATEGIC PROBLEM STATEMENT
================================================================================
[Scenario description]

================================================================================
                             KEY ASSUMPTIONS
================================================================================
1. Intelligence assessments accurate...

================================================================================
                        J2 INTELLIGENCE SUMMARY
================================================================================
[Enemy situation, ECOAs, vulnerabilities, intel gaps, I&W]

================================================================================
                          COMMANDER'S INTENT
================================================================================
[COA comparison, 2nd/3rd order effects, recommended COA, intent]

================================================================================
                     COURSES OF ACTION DEVELOPED
================================================================================
[J5 concepts + J3 execution details]

================================================================================
                       STAFF ESTIMATES BY COA
================================================================================
[J1, J4, J6, Cyber/EW, Fires, Engineer, Protection, PAO estimates]

================================================================================
                   LEGAL / ETHICAL CONSIDERATIONS
================================================================================
[SJA review: LOAC, ROE, AI ethics, international law]
```

## License

MIT License
