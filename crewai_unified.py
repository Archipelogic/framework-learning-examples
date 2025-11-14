#!/usr/bin/env python3
"""
Unified CrewAI Implementation with Intelligent Orchestration

Architecture:
- 1 Orchestration Agent (with delegation enabled)
- 3 Specialized Agents:
  1. Reasoning Specialist (inject_date=True) - calculations, puzzles, logical reasoning, time queries
  2. Data Researcher (native RagTool) - file searches and document analysis
  3. Database Analyst (LangChain SQL tools wrapped in BaseTool) - database queries

The orchestration agent analyzes prompts and delegates to the appropriate specialist.
Uses sequential process with allow_delegation=True for dynamic task routing.
NO if/then logic - the agent decides everything through delegation.
"""

import os
import sys
import webbrowser
from datetime import datetime
import pytz
from pathlib import Path

# MUST set these BEFORE importing CrewAI
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-bedrock"
os.environ["AWS_REGION"] = "us-east-1"
# Disable CrewAI's own telemetry (not Phoenix/OpenTelemetry)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
# Disable SSL verification for local Phoenix (localhost doesn't need SSL)
os.environ["OTEL_EXPORTER_OTLP_INSECURE"] = "true"
# Disable LangSmith tracing (we use Phoenix for observability)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor

# Launch Phoenix and setup tracing
session = px.launch_app()
tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

# Open Phoenix in browser
webbrowser.open(session.url)

# Import CrewAI after instrumentation
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.utilities import SQLDatabase
import boto3
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool as LCInfoTool,
    ListSQLDatabaseTool as LCListTool,
    QuerySQLDatabaseTool as LCQueryTool,
)
from embedding import create_langchain_faiss_vectorstore, get_embedding_for_vectordb


# ============================================================
# VECTORSTORE SETUP (Load pre-computed embeddings)
# ============================================================
# Note: RagTool doesn't work with pre-computed embeddings, so we use LangChain FAISS
text_embeddings_file = Path(__file__).parent / 'data' / 'text_embeddings.json'
metadata_file = Path(__file__).parent / 'data' / 'metadata.json'

vectorstore = None
try:
    # Create bedrock client and embedding function (only for query embedding)
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    embedding_model_id = "amazon.titan-embed-text-v1"
    embedding_function = get_embedding_for_vectordb(bedrock_client, embedding_model_id)
    
    # Load vectorstore from pre-computed embeddings
    vectorstore = create_langchain_faiss_vectorstore(text_embeddings_file, metadata_file, embedding_function)
    print(f"✅ Loaded vectorstore with {vectorstore.index.ntotal} vectors")
except FileNotFoundError:
    print(f"⚠️  Warning: Embedding files not found in data/ directory.")
except Exception as e:
    print(f"⚠️  Warning: Could not load vectorstore: {e}")


# ============================================================
# SPECIALIZED AGENTS (Minimal Set)
# ============================================================
def create_specialized_agents(project_root: Path) -> tuple[Agent, Agent, Agent]:
    """
    Create minimal set of specialized agents:
    1. Reasoning Agent (time tool) - handles calculations, puzzles, general reasoning
    2. Data Research Agent (RAG tool) - handles file searches
    3. Database Agent (SQL tools) - handles database queries
    """
    
    # 1. Reasoning Agent (General problem-solving with time tool)
    # Note: inject_date only provides DATE, not actual TIME - need explicit tool
    class CurrentTimeTool(BaseTool):
        name: str = "get_current_time"
        description: str = "Get the current time in a specific timezone. Input should be a timezone name like 'America/Chicago', 'America/New_York', or 'UTC'."
        
        def _run(self, timezone: str = "UTC") -> str:
            try:
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                return f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            except Exception as e:
                return f"Error getting time: {str(e)}"
    
    reasoning_agent = Agent(
        role="Analytical Reasoning Specialist",
        goal="Apply logical reasoning and problem-solving to any challenge",
        backstory="""You are an expert in analytical reasoning and problem-solving.

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
- Explain your reasoning process clearly""",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=[CurrentTimeTool()],  # Explicit time tool (inject_date only gives date, not time)
        inject_date=True,  # Additional date context
        verbose=True,
        allow_delegation=False
    )
    
    # 2. Data Research Agent (use LangChain FAISS with pre-computed embeddings)
    # Note: Native RagTool doesn't support pre-computed embeddings
    class SearchTool(BaseTool):
        name: str = "semantic_search"
        description: str = "Search knowledge base using semantic similarity to find relevant information. Input: conceptual search query."
        
        def _run(self, query: str = "") -> str:
            if vectorstore is None:
                return "Error: Vectorstore not loaded. Embedding files must be in data/ directory."
            if not query:
                return "Error: No search query provided."
            try:
                docs = vectorstore.similarity_search(query, k=3)
                if not docs:
                    return "No relevant documents found."
                
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
                return f"Error: {str(e)}"
    
    research_agent = Agent(
        role="Knowledge Retrieval Specialist",
        goal="Find and retrieve relevant information through intelligent search",
        backstory="""You are an expert at semantic search and information retrieval.

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
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=[SearchTool()],
        verbose=True,
        allow_delegation=False
    )
    
    # 3. Database Agent (with SQL tools wrapped in BaseTool)
    db_path = project_root / 'data' / 'doc.db'
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # Wrap LangChain tools in CrewAI BaseTool
    # Create instances of LangChain tools first
    lc_query_tool = LCQueryTool(db=db)
    lc_info_tool = LCInfoTool(db=db)
    lc_list_tool = LCListTool(db=db)
    
    class QuerySQLTool(BaseTool):
        name: str = "execute_query"
        description: str = "Execute queries against structured data and return results"
        
        def _run(self, query: str) -> str:
            return lc_query_tool.run(query)
    
    class InfoSQLTool(BaseTool):
        name: str = "get_schema_info"
        description: str = "Get information about data structure, tables, and available fields"
        
        def _run(self, table_names: str = "") -> str:
            return lc_info_tool.run(table_names)
    
    class ListSQLTool(BaseTool):
        name: str = "list_data_sources"
        description: str = "List all available data sources and tables"
        
        def _run(self, tool_input: str = "") -> str:
            return lc_list_tool.run(tool_input)
    
    database_agent = Agent(
        role="Structured Data Specialist",
        goal="Extract insights from structured data sources",
        backstory="""You are an expert in querying and analyzing structured data.

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
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=[ListSQLTool(), InfoSQLTool(), QuerySQLTool()],
        verbose=True,
        allow_delegation=False
    )
    
    return reasoning_agent, research_agent, database_agent


# ============================================================
# ORCHESTRATION
# ============================================================
def run_orchestration(user_prompt: str, project_root: Path) -> str:
    """
    Orchestration agent that analyzes the prompt and delegates to specialized agents.
    Only 3 agents: Reasoning (no tools), Research (RAG), Database (SQL)
    """
    print("\n" + "=" * 60)
    print("CREWAI ORCHESTRATION")
    print("=" * 60)
    
    # Create minimal set of specialized agents (only 3!)
    reasoning_agent, research_agent, database_agent = create_specialized_agents(project_root)
    
    # Orchestration agent with delegation enabled
    # This agent will analyze the prompt and delegate to the appropriate specialist
    orchestrator = Agent(
        role="Task Orchestrator",
        goal="Analyze user requests and delegate to the appropriate specialist",
        backstory="""You are a master orchestrator who analyzes user requests and delegates
        to specialized agents. You have access to:
        
        - Reasoning Specialist: for calculations, puzzles, riddles, and logical reasoning
        - Data Researcher: for searching through files and documents
        - Database Analyst: for querying databases and structured data
        
        Analyze the user's request carefully and delegate to the most appropriate specialist.""",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=True  # KEY: Enables delegation to other agents
    )
    
    # Create task for orchestrator with delegation enabled
    # Orchestrator will analyze and delegate to appropriate specialists
    orchestration_task = Task(
        description=f"""Analyze this user request and delegate to the appropriate specialist:
        
        User Request: {user_prompt}
        
        You have access to these specialists:
        - Reasoning Specialist: for calculations, puzzles, riddles, and logical reasoning
        - Data Researcher: for searching through files and documents  
        - Database Analyst: for querying databases and structured data
        
        Analyze the request, determine which specialist is needed, and delegate the work to them.""",
        expected_output="The final answer from the appropriate specialist",
        agent=orchestrator  # Orchestrator with delegation enabled
    )
    
    # Create sequential crew with delegation
    # Orchestrator can delegate to any of the specialists as needed
    orchestration_crew = Crew(
        agents=[orchestrator, reasoning_agent, research_agent, database_agent],
        tasks=[orchestration_task],
        process=Process.sequential,  # Sequential process with delegation
        verbose=True
    )
    
    # Execute - orchestrator analyzes prompt and delegates automatically
    result = orchestration_crew.kickoff()
    return str(result)


def main():
    """Main entry point"""
    # Phoenix tracing already set up at module level
    
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
    print("CREWAI UNIFIED ORCHESTRATOR")
    print("=" * 60)
    print(f"Phoenix UI: {session.url}")
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
