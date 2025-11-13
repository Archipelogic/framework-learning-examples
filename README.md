# Agentic AI Framework Examples

Unified implementations of **CrewAI** and **Pydantic AI** with intelligent orchestration agents that automatically route tasks.

## Overview

This repository demonstrates unified implementations of AI frameworks with intelligent orchestration:
- **CrewAI**: Uses hierarchical process with delegation for task routing
- **Pydantic AI**: Uses tool-based orchestration where specialists are tools

Each framework has a single script with **only 3 specialized agents**:
1. **Reasoning Specialist** (no tools) - handles calculations, puzzles, logical reasoning
2. **Data Researcher** (RAG/file tools) - handles file searches
3. **Database Analyst** (SQL tools) - handles database queries

An orchestration agent analyzes prompts and delegates to the appropriate specialist.

## Project Structure

```
framework-learning-examples/
├── crewai.py            # Unified CrewAI with hierarchical orchestration
├── pydantic_ai.py       # Unified Pydantic AI with tool-based orchestration
├── tasks/               # Task configurations (single source of truth)
│   ├── austin.yaml      # Time calculation task - name, description, prompt
│   ├── statefarm.yaml   # Jingle puzzle task - name, description, prompt
│   ├── rag.yaml         # File search task - name, description, prompt
│   └── sql.yaml         # Database query task - name, description, prompt
├── data/                # Sample data files
│   ├── projects/        # JSON files for RAG tasks
│   └── doc.csv          # SQLite database for SQL tasks
├── run.py               # Interactive task runner with sample prompts
├── setup.sh             # Automated setup script
└── run.sh               # Quick run script
```

## Architecture

### Minimal Agent Design
Both frameworks use only **3 specialized agents**:

1. **Reasoning Specialist** (no tools)
   - Handles: Time calculations, puzzles, riddles, mathematical reasoning
   - Examples: Austin time calculation, StateFarm jingle puzzle
   - Flexible enough to handle any logical reasoning task

2. **Data Researcher** (RAG/file tools)
   - Handles: File searches, document analysis
   - Tools: CrewAI uses native `RagTool`, Pydantic AI uses custom file reading
   - Examples: Searching through project JSON files

3. **Database Analyst** (SQL tools)
   - Handles: Database queries, structured data analysis
   - Tools: Both use LangChain SQL tools (no native available)
   - Examples: Querying police reports database

### Orchestration
- **CrewAI**: Hierarchical process with `allow_delegation=True`
- **Pydantic AI**: Tool-based where each specialist is a callable tool
- **No if/then logic**: The orchestration agent decides which specialist to use based on prompt analysis

## How It Works

### Execution Flow

1. **User provides a prompt** (e.g., "What time is it in Austin?")

2. **Orchestration agent analyzes the prompt**
   - CrewAI: Manager agent in hierarchical process
   - Pydantic AI: Agent with specialist tools

3. **Agent decides which specialist to use**
   - Reasoning Specialist: for calculations, puzzles, logic
   - Data Researcher: for file/document searches
   - Database Analyst: for database queries

4. **Specialist executes the task**
   - Uses appropriate tools (RAG, SQL, or none)
   - Returns structured result

5. **Result returned to user**
   - With reasoning/explanation
   - Traced in Phoenix for observability

### Key Advantage

The system is **flexible and scalable**:
- Can handle the 4 example tasks (Austin, StateFarm, RAG, SQL)
- Can handle 1000s of similar tasks without modification
- Agent decides based on prompt content, not predefined categories

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured (for Bedrock)

### Installation

```bash
# Clone the repository
git clone https://github.com/Archipelogic/framework-learning-examples.git
cd framework-learning-examples

# Run setup script
sh setup.sh
```

The setup script will:
1. Create a virtual environment
2. Install all dependencies
3. Set up sample database
4. Launch interactive task runner

### Manual Setup

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Create sample database
uv run python sql.py
```

## Usage

### Option 1: Interactive Runner (Recommended)

```bash
sh run.sh
# or
uv run python run.py
```

This launches an interactive menu where you can:
1. Select a framework (CrewAI or Pydantic AI)
2. Choose from sample prompts (dynamically loaded from `tasks/` YAML files) or enter your own
3. Watch the orchestration agent delegate to specialists

### Option 2: Direct Script Execution

```bash
# CrewAI - orchestration agent will analyze and delegate
uv run python crewai.py "What is the current time in Austin?"

# Pydantic AI - orchestration agent will select appropriate tool
uv run python pydantic_ai.py "How many police reports in 2024?"

# File search example
uv run python crewai.py "Which model was used in the Attorney Demand Classification project?"

# Puzzle example
uv run python pydantic_ai.py "Take the 5th letter in the State Farm jingle and multiply by 10"
```

The orchestration agent automatically determines which specialist to use - no manual task selection needed!

## Key Features Demonstrated

### CrewAI
- **Hierarchical Delegation**: Orchestrator delegates to 3 specialized agents
- **Minimal Agent Set**: Only 3 agents handle all task types
- **Native RAG Tool**: Built-in RagTool for file search
- **LangChain Integration**: SQL tools from langchain_community
- **Flexible Architecture**: Reasoning agent handles multiple task types
- **Phoenix Observability**: Tracing with Arize Phoenix

### Pydantic AI
- **Tool-Based Orchestration**: Specialists exposed as callable tools
- **Minimal Agent Set**: Only 3 agents handle all task types
- **Structured Outputs**: Type-safe results with Pydantic models
- **Custom Tools**: File reading tools with @agent.tool decorator
- **LangChain Integration**: SQL tools compatibility
- **Flexible Architecture**: Reasoning agent handles multiple task types
- **Phoenix Observability**: Tracing with Arize Phoenix

## Observability

Both unified scripts include Phoenix observability:

- **Phoenix (Arize)**: Used in both frameworks
  - Launches at `http://localhost:6006`
  - Provides detailed trace visualization
  - Tracks orchestration flow and tool usage

## Dependencies

Key packages:
- `crewai>=0.80.0` - Multi-agent orchestration
- `pydantic-ai>=0.0.14` - Type-safe agent framework
- `arize-phoenix>=4.0.0` - Observability platform
- `logfire>=0.40.0` - Pydantic observability
- `langchain-community>=0.3.0` - LangChain tools
- `boto3>=1.34.0` - AWS Bedrock integration

See `requirements.txt` for complete list.

## Known Issues

### huggingface-hub Warning
You may see this warning during installation:
```
warning: The package `huggingface-hub==1.1.2` does not have an extra named `inference`
```

This is **expected and harmless**. The `tokenizers` package pins `huggingface-hub==1.1.2`, which doesn't have the `inference` extra. All packages install successfully despite the warning.

## AWS Bedrock Configuration

All examples use AWS Bedrock with Claude Sonnet 4.5. Ensure you have:

1. AWS credentials configured
2. Bedrock access enabled in `us-east-1`
3. Claude model access granted

Set environment variable:
```bash
export AWS_REGION=us-east-1
```

## Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Arize Phoenix](https://docs.arize.com/phoenix)
- [Logfire](https://docs.pydantic.dev/logfire/)
