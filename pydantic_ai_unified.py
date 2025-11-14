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
from datetime import datetime
import pytz
from pathlib import Path

os.environ["AWS_REGION"] = "us-east-1"

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.bedrock import BedrockInstrumentor
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from langchain_community.utilities import SQLDatabase
import numpy as np
import boto3
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_aws import BedrockEmbeddings

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=Warning, message='.*Skipped unsupported reflection.*')
warnings.filterwarnings('ignore', message='.*BatchSpanProcessor.*')
warnings.filterwarnings('ignore', message='.*TracerProvider.*global.*')

# Launch Phoenix and setup Bedrock tracing (align with crewai_unified)
session = px.launch_app()
tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
BedrockInstrumentor().instrument(tracer_provider=tracer_provider)


# ============================================================
# VECTORSTORE SETUP (Module Level - matches CrewAI)
# ============================================================
def create_langchain_faiss_vectorstore(text_embeddings_path, metadata_path, embedding_function):
    """Create FAISS vectorstore from pre-computed embeddings"""
    text_embeddings = json.load(open(text_embeddings_path))
    text_embeddings = [(text, np.array(vec)) for text, vec in text_embeddings]
    metadata_list = json.load(open(metadata_path))
    
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_function,
        metadatas=metadata_list,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        relevance_score_fn=lambda distance: (distance + 1.0) / 2.0
    )
    return vectorstore


def get_embedding_for_vectordb(client, model_id):
    """Get embedding function for vectorstore"""
    embedding_function = BedrockEmbeddings(client=client, model_id=model_id)
    return embedding_function.embed_query


# Setup vectorstore at module level
embedding_model_id = "amazon.titan-embed-text-v1"
text_embeddings_file = Path(__file__).parent / 'data' / 'text_embeddings.json'
metadata_file = Path(__file__).parent / 'data' / 'metadata.json'

# Create bedrock client and embedding function
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
embedding_function = get_embedding_for_vectordb(bedrock_client, embedding_model_id)

# Create vectorstore (or None if files don't exist)
try:
    vectorstore = create_langchain_faiss_vectorstore(text_embeddings_file, metadata_file, embedding_function)
    print(f"✅ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
except FileNotFoundError as e:
    print(f"⚠️  Warning: Embedding files not found. Copy text_embeddings.json and metadata.json to data/ directory.")
    print(f"   RAG tool will not work until files are present.")
    vectorstore = None


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
    """General reasoning agent (with time tool)."""
    agent = Agent(
        model=model,
        system_prompt="""You excel at reasoning, calculations, puzzles, and problem-solving.
        You can handle time calculations, mathematical problems, riddles, multi-step reasoning,
        and any task that requires logical thinking. Show your reasoning step by step."""
    )

    @agent.tool
    def get_current_time(ctx: RunContext[None], timezone: str = "UTC") -> str:
        """Get the current time in a specific timezone like 'America/Chicago' or 'UTC'."""
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
        system_prompt="You search project documentation using semantic similarity to find relevant information.",
        deps_type=Path
    )
    
    @agent.tool
    def search_project_docs(ctx: RunContext[Path], query: str) -> str:
        """Search project documentation using semantic similarity.
        
        Args:
            query: The search query to find relevant documents
        """
        try:
            if vectorstore is None:
                return "Error: Vectorstore not loaded. Embedding files (text_embeddings.json, metadata.json) must be in data/ directory."
            
            if not query:
                return "Error: No search query provided. Please provide a search query."
            
            docs = vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No relevant documents found for your query."
            
            return "\n\n---\n\n".join([doc.page_content for doc in docs])
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
        system_prompt="You query databases using SQL and format results as JSON.",
        deps_type=Path,
    )

    @agent.tool
    def get_schema(ctx: RunContext[Path]) -> str:
        """Get database schema information for available tables and columns."""
        try:
            return db.get_table_info()
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    @agent.tool
    def run_sql(ctx: RunContext[Path], query: str) -> str:
        """Execute a SQL query against the SQLite database and return raw text results."""
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
        system_prompt="""You are a master orchestrator with access to specialized agents.
        Analyze the user's request and call the appropriate specialist:
        
        - call_reasoning_specialist: for calculations, puzzles, riddles, and logical reasoning
        - call_data_researcher: for searching through files and documents
        - call_database_analyst: for querying databases and structured data
        
        Choose the most appropriate specialist and pass them the user's request.""",
        deps_type=OrchestratorDeps
    )
    
    # Define tools that expose each specialist agent
    # The orchestrator will call these based on prompt analysis
    
    @orchestrator.tool
    def call_reasoning_specialist(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Call the Reasoning Specialist for calculations, puzzles, and logical reasoning.
        
        Args:
            task: The reasoning task to perform
        """
        print("\n→ Delegating to Reasoning Specialist")
        result = reasoning_agent.run_sync(task)
        payload = getattr(result, "output", result)
        return str(payload)
    
    @orchestrator.tool
    def call_data_researcher(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Call the Data Researcher for searching through files and documents.
        
        Args:
            task: The file search task to perform
        """
        print("\n→ Delegating to Data Researcher")
        data_dir = ctx.deps.project_root / 'data' / 'projects'
        result = research_agent.run_sync(task, deps=data_dir)
        payload = getattr(result, "output", result)
        return str(payload)
    
    @orchestrator.tool
    def call_database_analyst(ctx: RunContext[OrchestratorDeps], task: str) -> str:
        """Call the Database Analyst for database queries.
        
        Args:
            task: The database query task to perform
        """
        print("\n→ Delegating to Database Analyst")
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
