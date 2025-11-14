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
| **Framework Design** | 8 | 9 | CrewAI: Built from scratch, LangChain-independent. Pydantic AI: Built by Pydantic team, powers OpenAI/Anthropic SDKs |
| **Documentation** | 7 | 9 | CrewAI: 1000+ pages but examples don't always work (native RagTool incompatible with precomputed embeddings). Pydantic AI: Excellent, clear |
| **Framework Learning Curve** | 6 | 8 | CrewAI: More concepts (Agents, Tasks, Crews, Flows, Processes). Pydantic AI: Simpler, Pythonic, familiar patterns |
| **System Prompt** | 9 | 9 | CrewAI: Rich role/goal/backstory system. Pydantic AI: Clean system_prompt with dependency injection |
| **Out of Box Coding** | 10 | 7 | CrewAI: CodeInterpreterTool generates/executes Python code. Pydantic AI: Needs explicit @agent.tool definitions |
| **Latency** | 6 | 7 | CrewAI: Slower RAG searches (LangSmith tracing errors slowed execution). Pydantic AI: Faster, cleaner execution |
| **Agents as Tools (Orchestration)** | 10 | 8 | CrewAI: Native delegation with allow_delegation=True works well. Pydantic AI: Wrap agents as tools manually |
| **Independent Task Execution** | Yes (async_execution) | Possible (asyncio) | CrewAI: Built-in async task execution. Pydantic AI: Manual async implementation required |
| **Graph Capability** | Yes (Flows with @router) | Yes (graph with type hints) | CrewAI: Flows with conditional routing. Pydantic AI: Type-hint based graph definitions |
| **Native Tools** | 50+ tools (many require workarounds) | MCP, A2A protocol support | CrewAI: RagTool doesn't support precomputed embeddings, had to use LangChain FAISS. Pydantic AI: Protocol-based |
| **LangChain Integration** | Native (BaseTool wrappers) | Direct (seamless) | CrewAI: Requires wrapping LangChain tools. Pydantic AI: Pydantic is LangChain's validation layer, no wrapping needed |
| **Type Safety** | No | Yes (Full type checking) | CrewAI: Dynamic, runtime errors. Pydantic AI: Compile-time type safety, IDE autocomplete catches errors early |
| **Observability** | 5 (Phoenix issues) | 9 (Phoenix works) | CrewAI: Phoenix tracing incomplete, had instrumentation errors. Pydantic AI: Phoenix integration smooth |
| **RAG Implementation** | 6 (complex workarounds) | 7 (custom implementation) | Both struggled with RAG tasks completing. CrewAI native RagTool failed, custom FAISS worked. Pydantic AI needed custom tools |
| **Best For** | Multi-agent delegation workflows | Type-safe apps, clean implementations | CrewAI: Good when delegation works. Pydantic AI: Better developer experience, fewer surprises |

## Known Issues & Workarounds

### CrewAI
- ❌ **Phoenix Tracing Incomplete**: CrewAIInstrumentor doesn't fully capture all traces - some agent interactions missing
- ❌ **Native RagTool Limitation**: Doesn't support precomputed embeddings - had to use custom LangChain FAISS tool instead
- ❌ **LangSmith API Errors**: Had to disable LangSmith tracing (`LANGCHAIN_TRACING_V2=false`) to prevent 400 errors during execution
- ⚠️ **RAG Task Completion**: Knowledge retrieval agent sometimes struggles with complex semantic searches

### Pydantic AI
- ✅ **Phoenix Tracing Works**: PydanticAIInstrumentor correctly traces all operations
- ⚠️ **RAG Task Completion**: Similar to CrewAI, complex retrieval queries sometimes incomplete
- ⚠️ **Manual Tool Wrapping**: Need to manually wrap agents as tools for orchestration (more boilerplate)

### Both Frameworks
- ⚠️ **Semantic Search Quality**: Both rely on LLM's ability to refine queries - results vary with prompt engineering
- ⚠️ **Bedrock Rate Limits**: Both can hit AWS Bedrock throttling on complex multi-agent workflows

## Tool Priority

Both implementations follow this priority:
1. **Native framework tools first** (e.g., CrewAI's inject_date, RagTool)
2. **LangChain tools second** (e.g., SQL tools)
3. **Custom tools last** (only if no alternative exists)

**Note**: Due to issues encountered, we had to deviate from this priority for RAG - using custom LangChain FAISS tools instead of native options.

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
