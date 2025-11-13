#!/usr/bin/env python3
"""
Unified CrewAI Implementation with Intelligent Orchestration

Architecture:
- 1 Orchestration Agent (hierarchical manager with delegation)
- 3 Specialized Agents:
  1. Reasoning Specialist (no tools) - calculations, puzzles, logical reasoning
  2. Data Researcher (native RagTool) - file searches and document analysis
  3. Database Analyst (LangChain SQL tools) - database queries

The orchestration agent analyzes prompts and delegates to the appropriate specialist.
NO if/then logic - the agent decides everything through delegation.
"""

import os
import sys
from pathlib import Path

# MUST set these BEFORE importing CrewAI
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-bedrock"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["AWS_REGION"] = "us-east-1"

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import RagTool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool as LCInfoTool,
    ListSQLDatabaseTool as LCListTool,
    QuerySQLDatabaseTool as LCQueryTool,
)
from pydantic import Field


# ============================================================
# SPECIALIZED AGENTS (Minimal Set)
# ============================================================
def create_specialized_agents(project_root: Path) -> tuple[Agent, Agent, Agent]:
    """
    Create minimal set of specialized agents:
    1. Reasoning Agent (no tools) - handles calculations, puzzles, general reasoning
    2. Data Research Agent (RAG tool) - handles file searches
    3. Database Agent (SQL tools) - handles database queries
    """
    
    # 1. General Reasoning Agent (no tools needed)
    reasoning_agent = Agent(
        role="Reasoning Specialist",
        goal="Solve problems through logical reasoning and calculation",
        backstory="""You excel at reasoning, calculations, puzzles, and problem-solving.
        You can handle time calculations, mathematical problems, riddles, multi-step reasoning,
        and any task that requires logical thinking.""",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    # 2. Data Research Agent (with RAG tool)
    data_dir = project_root / 'data' / 'projects'
    rag_tool = RagTool()
    rag_tool.add(data_type="directory", path=str(data_dir))
    
    research_agent = Agent(
        role="Data Researcher",
        goal="Search through files and documents to find information",
        backstory="You use RAG tools to search through documents and extract relevant information.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=[rag_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # 3. Database Agent (with SQL tools wrapped in BaseTool)
    db_path = project_root / 'data' / 'doc.csv'
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    # Wrap LangChain tools in CrewAI BaseTool
    # Create instances of LangChain tools first
    lc_query_tool = LCQueryTool(db=db)
    lc_info_tool = LCInfoTool(db=db)
    lc_list_tool = LCListTool(db=db)
    
    class QuerySQLTool(BaseTool):
        name: str = "Query SQL Database"
        description: str = "Execute SQL queries and return results"
        
        def _run(self, query: str) -> str:
            return lc_query_tool.run(query)
    
    class InfoSQLTool(BaseTool):
        name: str = "Get SQL Table Info"
        description: str = "Get information about database tables and schema"
        
        def _run(self, table_names: str = "") -> str:
            return lc_info_tool.run(table_names)
    
    class ListSQLTool(BaseTool):
        name: str = "List SQL Tables"
        description: str = "List all available tables in the database"
        
        def _run(self, tool_input: str = "") -> str:
            return lc_list_tool.run(tool_input)
    
    database_agent = Agent(
        role="Database Analyst",
        goal="Query databases to answer questions",
        backstory="You use SQL tools to query databases and format results.",
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
    
    # Create task - in hierarchical mode, don't assign to manager
    # The manager will analyze and delegate automatically
    orchestration_task = Task(
        description=f"""Analyze this user request and provide the answer:
        
        User Request: {user_prompt}
        
        Determine what type of task this is and provide a comprehensive answer.""",
        expected_output="The final answer to the user's question",
        agent=reasoning_agent  # Assign to a worker agent, manager will delegate as needed
    )
    
    # Create hierarchical crew with orchestrator as manager
    # The orchestrator will analyze the task and delegate to appropriate specialists
    # Note: manager_agent should NOT be in the agents list
    orchestration_crew = Crew(
        agents=[reasoning_agent, research_agent, database_agent],
        tasks=[orchestration_task],
        process=Process.hierarchical,  # Enables manager-worker delegation
        manager_agent=orchestrator,    # Orchestrator is the manager
        verbose=True
    )
    
    # Execute - orchestrator analyzes prompt and delegates automatically
    result = orchestration_crew.kickoff()
    return str(result)


def main():
    """Main entry point"""
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="crewai-orchestrator",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
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
