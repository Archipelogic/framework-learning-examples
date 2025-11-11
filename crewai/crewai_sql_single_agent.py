#!/usr/bin/env python3
"""
SQL Task - CrewAI Single Agent Implementation

TASK: Query database for police reports by month

APPROACH: Single agent with SQL query tool
- Agent has a tool to execute SQL queries
- Formats results as JSON
- External tool for database access

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os

# MUST set these BEFORE importing CrewAI to prevent OpenAI connections
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-bedrock"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from pathlib import Path
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
)

# NOTE: Using LangChain's built-in SQL tools instead of custom tools


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    project_root = Path(__file__).parent.parent
    task_file = project_root / 'tasks' / 'sql.yaml'
    
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="sql-single-agent-crewai",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # SQL DATABASE SETUP
    # ============================================================
    db = SQLDatabase.from_uri(f"sqlite:///{project_root}/data/doc.csv")
    
    # Create LangChain SQL tools
    query_tool = QuerySQLDataBaseTool(db=db)
    info_tool = InfoSQLDatabaseTool(db=db)
    list_tool = ListSQLDatabaseTool(db=db)
    
    # ============================================================
    # AGENT: With LangChain SQL tools
    # ============================================================
    agent = Agent(
        role="Database Analyst",
        goal="Query databases to answer questions",
        backstory="You use SQL tools to query databases and format results.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tools=[list_tool, info_tool, query_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # ============================================================
    # TASK: Uses prompt from config file
    # ============================================================
    task = Task(
        description=task_prompt,
        agent=agent,
        expected_output="JSON object with police reports count by month"
    )
    
    # ============================================================
    # CREW: Single agent crew
    # ============================================================
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("SQL TASK - Single Agent CrewAI")
    print("=" * 60)
    print("Approach: Single agent with LangChain SQL tools")
    print("Tools: QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool")
    print("Note: Using LangChain's built-in SQL tools (not custom)")
    print("=" * 60)
    print()
    
    result = crew.kickoff()
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)
    print()
    print(f"View traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
