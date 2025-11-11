# AI Framework Examples

Comparative examples of **CrewAI** and **Pydantic AI** implementations across different task types.

## Overview

This repository demonstrates how to implement the same AI tasks using two popular Python frameworks:
- **CrewAI**: Multi-agent orchestration framework
- **Pydantic AI**: Type-safe agent framework with structured outputs

## Project Structure

```
framework-learning-examples/
├── crewai/              # CrewAI implementations
│   ├── crewai_austin_single_agent.py
│   ├── crewai_austin_multi_agent.py
│   ├── crewai_statefarm_single_agent.py
│   ├── crewai_statefarm_multi_agent.py
│   ├── crewai_statefarm_multi_agent_listen.py  # Flow with @listen
│   ├── crewai_rag_single_agent.py
│   └── crewai_sql_single_agent.py
├── pydantic/            # Pydantic AI implementations
│   ├── pydantic_austin_single_agent.py
│   ├── pydantic_austin_no_tools.py
│   ├── pydantic_statefarm_single_agent.py
│   ├── pydantic_statefarm_no_tools.py
│   ├── pydantic_statefarm_logfire.py
│   ├── pydantic_rag_single_agent.py
│   └── pydantic_sql_single_agent.py
├── tasks/               # Task configurations
│   ├── austin.yaml
│   ├── statefarm.yaml
│   ├── rag.yaml
│   └── sql.yaml
├── data/                # Sample data files
└── run.py               # Interactive task runner
```

## Tasks

### 1. Austin Time Calculation
Get current time in Austin, TX and calculate: `minutes^(1/hour)`

**Implementations:**
- Single agent (CrewAI & Pydantic AI)
- Multi-agent (CrewAI)
- No tools - LLM knowledge only (Pydantic AI)

### 2. StateFarm Jingle Puzzle
Find 5th letter in State Farm jingle, convert to alphabet position, multiply by question length.

**Implementations:**
- Single agent (CrewAI & Pydantic AI)
- Multi-agent sequential (CrewAI)
- Multi-agent with Flow + @listen decorator (CrewAI)
- No tools - LLM knowledge only (Pydantic AI)
- Logfire observability (Pydantic AI)

### 3. RAG (Retrieval-Augmented Generation)
Search through JSON files to find model information.

**Implementations:**
- File reading tools (both frameworks)
- LangChain tools integration (Pydantic AI)

### 4. SQL Database Query
Query SQLite database for police reports by month.

**Implementations:**
- Custom SQL tools (both frameworks)

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

### Interactive Runner

```bash
sh run.sh
# or
uv run python run.py
```

This launches an interactive menu to run any example.

### Run Individual Scripts

```bash
# CrewAI examples
uv run python crewai/crewai_statefarm_multi_agent_listen.py

# Pydantic AI examples
uv run python pydantic/pydantic_statefarm_logfire.py
```

## Key Features Demonstrated

### CrewAI
- **Sequential Process**: Tasks execute in order
- **Multi-Agent Collaboration**: Specialized agents working together
- **Flow with @listen**: Event-driven reactive workflows
- **Built-in Tools**: DirectoryReadTool, FileReadTool
- **Custom Tools**: SQL query tools with @tool decorator
- **Phoenix Observability**: Tracing with Arize Phoenix

### Pydantic AI
- **Structured Outputs**: Type-safe results with Pydantic models
- **Tool Integration**: LangChain tools compatibility
- **Dependency Injection**: Clean separation of concerns
- **Logfire Observability**: Pydantic's native observability platform
- **Streaming Support**: Real-time response streaming
- **No-Tools Mode**: Pure LLM reasoning without external tools

## Observability

All examples include observability integration:

- **Phoenix (Arize)**: Used in most examples
  - Launches at `http://localhost:6006`
  - Provides detailed trace visualization

- **Logfire**: Used in Pydantic AI examples
  - Pydantic's native observability platform
  - Console logging + optional cloud integration

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
