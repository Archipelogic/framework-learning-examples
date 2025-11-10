#!/usr/bin/env python3
"""
RAG Task - Pydantic AI Single Agent Implementation

TASK: Search through JSON files to find model information

APPROACH: Single agent with file reading tools
- Agent has tools to read JSON files
- Searches through project data
- External tools for file I/O

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.pydantic_ai import PydanticAIInstrumentor
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModel
from langchain_community.tools import ReadFileTool, ListDirectoryTool


# ============================================================
# OUTPUT STRUCTURE
# ============================================================
class RAGResult(BaseModel):
    """Structured output for RAG search."""
    project_name: str = Field(description="Name of the project found")
    model_type: str = Field(description="Model type used in the project")
    source_file: str = Field(description="File where information was found")


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    with open('../tasks/rag.yaml', 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    data_dir = '../' + task_config['parameters']['data_directory']
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="rag-single-agent-pydantic",
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
    # AGENT with LangChain FILE TOOLS
    # ============================================================
    # Using LangChain's built-in tools instead of custom implementations
    read_file_tool = ReadFileTool()
    list_dir_tool = ListDirectoryTool()
    
    agent = Agent(
        model=model,
        result_type=RAGResult,
        system_prompt="You search through files to find information.",
        tools=[read_file_tool, list_dir_tool]
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("RAG TASK - Pydantic AI with LangChain Tools")
    print("=" * 60)
    print("Approach: Single agent with LangChain tools")
    print("Tools:")
    print("  - ReadFileTool: Reads file contents (from langchain_community)")
    print("  - ListDirectoryTool: Lists directory contents (from langchain_community)")
    print("=" * 60)
    print()
    
    prompt = f"{task_prompt}\n\nSearch in directory: {data_dir}"
    result = agent.run_sync(prompt)
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Project: {result.data.project_name}")
    print(f"Model Type: {result.data.model_type}")
    print(f"Source: {result.data.source_file}")
    print()
    
    if hasattr(result, 'usage'):
        usage = result.usage()
        print(f"Token Usage: {usage.total_tokens} total")
    
    print(f"\nView traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
