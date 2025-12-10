# ProjectWARGATE

**W**argaming **A**gent for **R**apid **G**eneration of **A**ctions, **T**hreats, and **E**stimates

A multi-agent LangChain system that mirrors a joint military staff doing operational planning and COA (Course of Action) development using OpenAI models.

## Overview

WARGATE implements a complete joint staff planning cell with 15 specialized AI agents representing different staff functions. The system follows Joint Planning Process (JP 5-0) doctrine to analyze scenarios, develop courses of action, war-game options, and produce unified operational plans.

## Features

- **15 Specialized Staff Agents**: Commander, J1-J8, Cyber/EW, Fires, Engineer, Protection, SJA, and PAO
- **Doctrinal Planning Process**: Follows JP 5-0 joint planning phases
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
from wargate import run_wargate_planning

scenario = """
A near-peer adversary has massed forces along the border of a NATO ally.
Intelligence indicates an imminent invasion within 72 hours.
Develop options for deterrence and defense.
"""

result = run_wargate_planning(
    scenario=scenario,
    model_name="gpt-4.1",
    temperature=0.7,
)

print(result)
```

## CLI Usage

```bash
# Run with inline scenario
python wargate.py --scenario "Your scenario description here"

# Run with scenario from file
python wargate.py --scenario-file scenario.txt

# Customize model and temperature
python wargate.py -s "Scenario..." --model gpt-4.1 --temperature 0.5

# Save output to file
python wargate.py -s "Scenario..." --output plan.txt

# Run quietly (minimal console output)
python wargate.py -s "Scenario..." --quiet
```

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

## Planning Phases

1. **Mission Analysis**: Staff analyzes scenario, J2 provides threat assessment, Commander issues guidance
2. **COA Development**: J5/J3 develop 3 distinct COAs with functional inputs
3. **COA Analysis**: War-gaming, Red Team challenge, legal review
4. **COA Comparison**: Staff evaluation against criteria
5. **COA Selection**: Commander selects COA
6. **Plan Development**: Detailed OPORD with annexes

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
WARGATEOrchestrator
├── WARGATEConfig (model, temperature, etc.)
├── StaffAgent[] (15 agents)
│   ├── StaffRole (enum)
│   ├── System Prompt (doctrinal role)
│   ├── Tools (RAG retrievers)
│   └── AgentExecutor (LangChain)
└── Planning Phases (6 phases)
```

## License

MIT License
