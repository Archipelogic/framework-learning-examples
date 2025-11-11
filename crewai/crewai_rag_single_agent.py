#!/usr/bin/env python3
"""
RAG Task - CrewAI Single Agent Implementation

TASK: Search through JSON files to find model information

APPROACH: Single agent with file reading tool
- Agent has a tool to read JSON files
- Searches through project data
- External tool for file I/O

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os
from pathlib import Path
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, DirectoryReadTool


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    project_root = Path(__file__).parent.parent
    task_file = project_root / 'tasks' / 'rag.yaml'
    
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    data_dir = str(project_root / task_config['parameters']['data_directory'])
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="rag-single-agent-crewai",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    
    # ============================================================
    # AGENT: With CrewAI's built-in file tools
    # ============================================================
    file_read_tool = FileReadTool()
    directory_tool = DirectoryReadTool(directory=data_dir)
    
    agent = Agent(
        role="Data Researcher",
        goal="Search through files to answer questions",
        backstory="You use tools to read and search through data files.",
        llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        tools=[directory_tool, file_read_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # ============================================================
    # TASK: Uses prompt from config file
    # ============================================================
    task = Task(
        description=f"{task_prompt}\n\nSearch in directory: {data_dir}",
        agent=agent,
        expected_output="The model type used in the project"
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
    print("RAG TASK - Single Agent CrewAI")
    print("=" * 60)
    print("Approach: Single agent with CrewAI built-in tools")
    print("Tools: DirectoryReadTool, FileReadTool (from crewai_tools)")
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
