#!/usr/bin/env python3
"""
SQL Task - Pydantic AI Single Agent Implementation

TASK: Query database for police reports by month

APPROACH: Single agent with SQL query tool
- Agent has tools to execute SQL queries
- Formats results as JSON
- External tools for database access

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os
from pathlib import Path
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.pydantic_ai import PydanticAIInstrumentor
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModel
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    QuerySQLDataBaseTool
)


# ============================================================
# OUTPUT STRUCTURE
# ============================================================
class SQLResult(BaseModel):
    """Structured output for SQL query results."""
    query_executed: str = Field(description="The SQL query that was executed")
    results: dict = Field(description="Results as JSON object with year_month: count format")


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
        project_name="sql-single-agent-pydantic",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    PydanticAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    model = BedrockModel(
        model_id="anthropic.claude-sonnet-4-5-v1:0",
        region_name="us-east-1"
    )
    
    # ============================================================
    # AGENT with LangChain SQL TOOLS
    # ============================================================
    # Using LangChain's built-in SQL tools instead of custom implementations
    db = SQLDatabase.from_uri("sqlite:///../data/doc.csv")
    
    # LangChain SQL tools
    schema_tool = InfoSQLDatabaseTool(db=db)
    query_tool = QuerySQLDataBaseTool(db=db)
    
    agent = Agent(
        model=model,
        result_type=SQLResult,
        system_prompt="You query databases using SQL and format results.",
        tools=[schema_tool, query_tool]
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("SQL TASK - Pydantic AI with LangChain Tools")
    print("=" * 60)
    print("Approach: Single agent with LangChain SQL tools")
    print("Tools:")
    print("  - InfoSQLDatabaseTool: Gets database schema (from langchain_community)")
    print("  - QuerySQLDataBaseTool: Executes SQL queries (from langchain_community)")
    print("=" * 60)
    print()
    
    result = agent.run_sync(task_prompt)
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Query Executed:\n{result.data.query_executed}\n")
    print(f"Results:")
    print(json.dumps(result.data.results, indent=2))
    print()
    
    if hasattr(result, 'usage'):
        usage = result.usage()
        print(f"Token Usage: {usage.total_tokens} total")
    
    print(f"\nView traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
