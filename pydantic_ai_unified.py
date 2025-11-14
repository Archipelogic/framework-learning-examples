#!/usr/bin/env python3
"""
Unified Pydantic AI Implementation with Intelligent Orchestration

Architecture:
- 1 Orchestration Agent (with specialists as callable tools)
- 3 Specialized Agents:
  1. Reasoning Specialist (no tools) - calculations, puzzles, logical reasoning
  2. Data Researcher (custom file tools) - file searches and document analysis
  3. Database Analyst (LangChain SQL tools) - database queries

The orchestration agent analyzes prompts and calls the appropriate specialist tool.
NO if/then logic - the agent decides everything through tool selection.
"""

import os
import sys
import json
import warnings
import webbrowser
from datetime import datetime
import pytz
from pathlib import Path

os.environ["AWS_REGION"] = "us-east-1"
# Disable LangSmith tracing (we use Phoenix for observability)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.pydantic_ai import PydanticAIInstrumentor
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from langchain_community.utilities import SQLDatabase
from embedding import create_langchain_faiss_vectorstore, get_embedding_for_vectordb
import boto3

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=Warning, message='.*Skipped unsupported reflection.*')
warnings.filterwarnings('ignore', message='.*BatchSpanProcessor.*')
warnings.filterwarnings('ignore', message='.*TracerProvider.*global.*')

# Launch Phoenix and setup tracing
session = px.launch_app()
tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
PydanticAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Open Phoenix in browser
webbrowser.open(session.url)


# ============================================================
# VECTORSTORE SETUP (Load pre-computed embeddings only)
# ============================================================
text_embeddings_file = Path(__file__).parent / 'data' / 'text_embeddings.json'
metadata_file = Path(__file__).parent / 'data' / 'metadata.json'

# Load vectorstore from pre-computed embeddings (or None if files don't exist)
vectorstore = None
try:
    # Create bedrock client and embedding function (only for query embedding)
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    embedding_model_id = "amazon.titan-embed-text-v1"
    embedding_function = get_embedding_for_vectordb(bedrock_client, embedding_model_id)
    
    # Load vectorstore from pre-computed embeddings
    vectorstore = create_langchain_faiss_vectorstore(text_embeddings_file, metadata_file, embedding_function)
    print(f"✅ Loaded vectorstore with {vectorstore.index.ntotal} vectors from pre-computed embeddings")
except FileNotFoundError:
    print(f"⚠️  Warning: Embedding files not found in data/ directory.")
except Exception as e:
    print(f"⚠️  Warning: Could not load vectorstore: {e}")


# ============================================================
# DEPENDENCIES
# ============================================================
class OrchestratorDeps(BaseModel):
    """Dependencies for orchestration"""
    project_root: Path
    user_prompt: str


# ============================================================
# SPECIALIZED AGENTS (Minimal Set)
# ============================================================
def create_reasoning_agent(model: BedrockConverseModel) -> Agent:
    """
    Create analytical reasoning agent.
    Handles calculations, logical reasoning, and general problem-solving.
    """
    agent = Agent(
        model=model,
        system_prompt="""You are an expert in analytical reasoning and problem-solving.

Your Capabilities:
- Logical deduction and multi-step reasoning
- Pattern recognition and abstract thinking
- Quantitative analysis and calculations
- Temporal reasoning with current date/time context
- Breaking down complex problems into manageable components

Your Approach:
- Start by understanding the core question or problem
- Identify relevant information and constraints
- Apply systematic reasoning to derive solutions
- Verify conclusions for logical consistency
- Explain your reasoning process clearly"""
    )

    @agent.tool
    def get_current_time(ctx: RunContext[None], timezone: str = "UTC") -> str:
        """Get the current time in any timezone for temporal reasoning and time-based calculations."""
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except Exception as e:
            return f"Error getting time: {str(e)}"

    return agent


def create_research_agent(model: BedrockConverseModel, data_dir: Path) -> Agent:
    """
    Create data research agent (with RAG search tool).
    Handles semantic search through project documentation.
    """
    agent = Agent(
        model=model,
        system_prompt="""You are an expert at semantic search and information retrieval.

Search Strategy:
- Start with conceptual queries that capture the intent, not just keywords
- If initial results are unclear, broaden or refine your search terms
- Think about synonyms, related concepts, and how information might be described
- Consider the context: technical documentation uses precise terminology
- Cast a wide semantic net first, then narrow based on what you find

You understand that:
- Similar concepts may be described with different words
- Project documentation often contains rich contextual information
- The best answers come from understanding the question's intent, not just matching words""",
        deps_type=Path
    )
    
    @agent.tool
    def semantic_search(ctx: RunContext[Path], query: str) -> str:
        """Search knowledge base using semantic similarity to find relevant information.
        
        Args:
            query: Conceptual search query to find relevant content
        """
        try:
            if vectorstore is None:
                return "Error: Vectorstore not loaded. Embedding files (text_embeddings.json, metadata.json) must be in data/ directory."
            
            if not query:
                return "Error: No search query provided. Please provide a search query."
            
            docs = vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No relevant documents found for your query."
            
            # Include metadata for better context
            results = []
            for doc in docs:
                result = ""
                if doc.metadata:
                    result += f"[Metadata: {doc.metadata}]\n"
                result += doc.page_content
                results.append(result)
            
            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    return agent


def create_database_agent(model: BedrockConverseModel, db_path: Path) -> Agent:
    """
    Create database agent (with SQL tools).
    Handles querying databases and structured data.
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Create agent that will expose simple SQL tools
    agent = Agent(
        model=model,
        system_prompt="""You are an expert in querying and analyzing structured data.

Your Expertise:
- Understanding data schemas and relationships
- Constructing efficient queries to extract information
- Interpreting query results in context
- Identifying relevant data patterns and insights

Your Approach:
- First understand the data structure available
- Formulate precise queries based on the question
- Execute queries and interpret results
- Present findings in a clear, meaningful way""",
        deps_type=Path,
    )

    @agent.tool
    def get_schema(ctx: RunContext[Path]) -> str:
        """Get data structure information including available tables, fields, and relationships."""
        try:
            return db.get_table_info()
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    @agent.tool
    def execute_query(ctx: RunContext[Path], query: str) -> str:
        """Execute a query against structured data and return results."""
        try:
            raw = db.run(query)
            return str(raw)
        except Exception as e:
            return f"Error executing SQL: {str(e)}"

    return agent


# ============================================================
# ORCHESTRATION
# ============================================================
def run_orchestration(user_prompt: str, project_root: Path) -> str:
    """
    Orchestration agent that has access to 3 specialized agents as tools.
    Only 3 agents: Reasoning (no tools), Research (file tools), Database (SQL)
    """
    print("\n" + "=" * 60)
    print("PYDANTIC AI ORCHESTRATION")
    print("=" * 60)
    
    model = BedrockConverseModel(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    
    # Create minimal set of specialized agents (only 3!)
    reasoning_agent = create_reasoning_agent(model)
    research_agent = create_research_agent(model, project_root / 'data' / 'projects')
    database_agent = create_database_agent(model, project_root / 'data' / 'doc.db')
    
    # Create orchestration agent with all specialists as tools
    # Each specialist is exposed as a callable tool that the orchestrator can invoke
    orchestrator = Agent(
        model=model,
        system_prompt="""You are an intelligent orchestrator with access to specialized capabilities.
        
Analyze each request and route it to the appropriate specialist:

- use_analytical_reasoning: Apply logical reasoning, pattern analysis, and problem-solving
- use_knowledge_retrieval: Search and retrieve information from the knowledge base
- use_data_analysis: Query and analyze structured data sources

Your Role:
- Understand the core intent of each request
- Select the most appropriate specialist for the task
- Delegate the full request context to the chosen specialist
- Synthesize the specialist's response into a clear answer""",
        deps_type=OrchestratorDeps
    )
    
    # Define tools that expose each specialist agent
    # The orchestrator will call these based on prompt analysis
    
    @orchestrator.tool
    def use_analytical_reasoning(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Apply analytical reasoning and problem-solving capabilities.
        
        Args:
            task: The analytical task to perform
        """
        print("\n→ Using Analytical Reasoning")
        result = reasoning_agent.run_sync(task)
        payload = getattr(result, "output", result)
        return str(payload)
    
    @orchestrator.tool
    def use_knowledge_retrieval(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Search and retrieve information from the knowledge base.
        
        Args:
            task: The information retrieval task to perform
        """
        print("\n→ Using Knowledge Retrieval")
        data_dir = ctx.deps.project_root / 'data' / 'projects'
        result = research_agent.run_sync(task, deps=data_dir)
        payload = getattr(result, "output", result)
        return str(payload)
    
    @orchestrator.tool
    def use_data_analysis(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Query and analyze structured data sources.
        
        Args:
            task: The data analysis task to perform
        """
        print("\n→ Using Data Analysis")
        result = database_agent.run_sync(task)
        payload = getattr(result, "output", result)
        return str(payload)
    
    # Run orchestration - agent will analyze prompt and call appropriate tool
    deps = OrchestratorDeps(project_root=project_root, user_prompt=user_prompt)
    result = orchestrator.run_sync(
        f"Analyze this request and call the appropriate specialist: {user_prompt}",
        deps=deps
    )
    payload = getattr(result, "output", result)
    return str(payload)


def main():
    """Main entry point"""
    # ============================================================
    # OBSERVABILITY SETUP (Phoenix already launched at module level)
    # ============================================================
    print(f"Phoenix UI: {session.url}")
    
    # ============================================================
    # GET USER PROMPT
    # ============================================================
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("\nEnter your question:")
        user_prompt = input("> ").strip()
    
    if not user_prompt:
        print("Error: No prompt provided")
        sys.exit(1)
    
    # ============================================================
    # RUN ORCHESTRATION
    # ============================================================
    print("\n" + "=" * 60)
    print("PYDANTIC AI UNIFIED ORCHESTRATOR")
    print("=" * 60)
    print(f"Prompt: {user_prompt}")
    
    project_root = Path(__file__).parent
    result = run_orchestration(user_prompt, project_root)
    
    # ============================================================
    # DISPLAY FINAL RESULT
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
    print()
    print(f"View traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
