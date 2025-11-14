# Agentic AI Framework Examples

Unified implementations of **CrewAI** and **Pydantic AI** with intelligent orchestration. Both frameworks use domain-agnostic agents that automatically route tasks through general capabilities, not hardcoded logic.

## Architecture

Each framework has **1 orchestrator** + **3 specialized agents**:

1. **Analytical Reasoning Specialist** - logical reasoning, calculations, problem-solving
2. **Knowledge Retrieval Specialist** - semantic search through knowledge base
3. **Structured Data Specialist** - query and analyze structured data

The orchestrator analyzes requests and routes to the appropriate specialist. No if/then logic - the LLM decides based on capabilities.

## Framework Comparison

| Item (Scale 1-10, higher the better) | CrewAI | Pydantic AI | Comments |
|---------------------------------------|--------|-------------|----------|
| **Framework Design** | 8 | 9 | Both well-architected; Pydantic AI more Pythonic |
| **Documentation** | 7 | 9 | CrewAI has good docs; Pydantic AI documentation is excellent and comprehensive |
| **Framework Learning Curve** | 6 | 8 | CrewAI has more concepts to learn; Pydantic AI is straightforward |
| **System Prompt** | 8 | 9 | Both support rich system prompts; Pydantic AI's agent.system_prompt is cleaner |
| **Out of Box Coding** | 9 | 7 | CrewAI can generate Python with right packages; Pydantic AI needs explicit tools |
| **Latency** | 7 | 7 | Similar performance on complex tasks; both use same LLM backend |
| **Agents as Tools (Orchestration)** | 9 | 8 | CrewAI uses delegation naturally; Pydantic AI requires wrapping agents as tools |
| **Independent Task Execution** | Yes | Possible but manual | CrewAI agents can work in parallel; Pydantic AI would need custom implementation |
| **Graph Capability** | No | No | Neither has built-in graph workflows; would need LangGraph integration |
| **Native Tools** | RagTool, inject_date | None | CrewAI has more built-in tools |
| **LangChain Integration** | BaseTool wrappers required | Direct compatibility | Pydantic AI easier to integrate LangChain tools |
| **Type Safety** | No | Yes (Pydantic models) | Pydantic AI has structured, validated outputs |
| **Best For** | Multi-agent orchestration | Type-safe AI applications | Choose based on use case |

## Tool Priority

Both implementations follow this priority:
1. **Native framework tools first** (e.g., CrewAI's inject_date, RagTool)
2. **LangChain tools second** (e.g., SQL tools)
3. **Custom tools last** (only if no alternative exists)

## Setup

### Prerequisites
- Python 3.10+
- `uv` package manager
- AWS credentials configured for Bedrock

### Installation

```bash
sh setup.sh
```

### Manual Setup

```bash
uv venv
uv sync  # Install from uv.lock
python sql.py  # Create sample database
```

## Usage

### Interactive Runner (Recommended)

```bash
sh run.sh
```

Select framework, choose sample prompts, or enter your own.

### Direct Execution

```bash
# CrewAI
uv run python crewai_unified.py "What time is it in Austin?"

# Pydantic AI  
uv run python pydantic_ai_unified.py "Which model was used in Attorney Demand Classification?"
```

### Generate Embeddings (Optional)

For knowledge retrieval tasks:

```bash
# Place DS_Projects_Docs.json in data/
python embedding.py
```

This creates `text_embeddings.json` and `metadata.json` for semantic search.

## Project Structure

```
framework-learning-examples/
├── crewai_unified.py          # CrewAI implementation
├── pydantic_ai_unified.py     # Pydantic AI implementation
├── embedding.py               # Embedding utilities
├── run.py                     # Interactive runner
├── sql.py                     # Database setup
├── data/                      # Sample data
│   ├── text_embeddings.json   # Pre-computed embeddings
│   ├── metadata.json          # Embedding metadata
│   └── doc.db                 # SQLite database
└── tasks/                     # Sample prompts (for run.py)
```

## Key Features

### Domain-Agnostic Design

All agents use general capabilities and methodologies, not domain-specific knowledge:

- **Analytical Reasoning**: Applies systematic reasoning to any problem
- **Knowledge Retrieval**: Uses semantic search strategy for any topic
- **Structured Data**: Queries any structured data source

### Semantic Search Strategy

The Knowledge Retrieval Specialist uses a general approach:
- Start with conceptual queries, not just keywords
- Refine searches based on initial results  
- Think about synonyms and related concepts
- Cast a wide semantic net, then narrow down

This works for any domain without hardcoding.

### Observability

Both frameworks use Phoenix for tracing:
- Launches at `http://localhost:6006`
- Tracks orchestration flow
- Visualizes tool usage
- LangSmith tracing disabled (Phoenix only)

## Example Tasks

The agents can handle diverse requests:

```bash
# Time/Date reasoning
"What time is it in Austin right now?"

# Semantic search
"Which model was used in the Attorney Demand Classification project?"

# Database queries
"How many police reports were filed in 2024?"

# Puzzles/Riddles
"Take the 5th letter in the State Farm jingle and multiply by 10"
```

No task-specific routing needed - the orchestrator analyzes and delegates based on the request's nature.

## AWS Bedrock Configuration

Uses AWS Bedrock with Claude Sonnet 4.5:

```bash
export AWS_REGION=us-east-1
```

Ensure you have:
1. AWS credentials configured
2. Bedrock access in us-east-1
3. Claude model access granted

## Dependencies

Dependencies are managed with `uv` and locked in `uv.lock` for reproducible installations.

Key packages:
- `crewai>=0.5.0`
- `pydantic-ai>=0.0.14`
- `arize-phoenix>=4.0.0`
- `langchain-community>=0.3.0`
- `boto3>=1.34.0`

See `pyproject.toml` for complete dependency list.

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
