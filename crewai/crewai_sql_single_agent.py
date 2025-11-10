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
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool


# NOTE: CrewAI doesn't have built-in SQL tools, so we create minimal custom tools
# These are EXTERNAL tools that actually query a database (not LLM knowledge)

@tool("execute_sql_query")
def execute_sql_query(query: str) -> str:
    """
    Execute a SQL query on the police reports database.
    
    Args:
        query: SQL query to execute
    
    Returns:
        Query results as JSON string
    """
    import sqlite3
    import json
    try:
        conn = sqlite3.connect('../data/doc.csv')
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        # Convert to list of dicts
        result_dicts = [dict(zip(columns, row)) for row in results]
        return json.dumps(result_dicts, indent=2)
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool("get_database_schema")
def get_database_schema() -> str:
    """
    Get the schema of the documents database.
    
    Returns:
        Database schema information
    """
    import sqlite3
    try:
        conn = sqlite3.connect('../data/doc.csv')
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schema = cursor.fetchall()
        conn.close()
        return "\n".join(s[0] for s in schema if s[0])
    except Exception as e:
        return f"Error getting schema: {str(e)}"


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    with open('../tasks/sql.yaml', 'r') as f:
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
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    
    # ============================================================
    # AGENT: With SQL tools
    # ============================================================
    agent = Agent(
        role="Database Analyst",
        goal="Query databases to answer questions",
        backstory="You use SQL tools to query databases and format results.",
        llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        tools=[get_database_schema, execute_sql_query],
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
    print("Approach: Single agent with custom SQL tools")
    print("Tools: get_database_schema, execute_sql_query (custom - no built-in SQL tools)")
    print("Note: These are EXTERNAL tools that query actual database")
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
