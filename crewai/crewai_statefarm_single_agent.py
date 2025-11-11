#!/usr/bin/env python3
"""
StateFarm Task - CrewAI Single Agent Implementation

TASK: Find 5th letter in State Farm jingle, convert to alphabet position, 
      multiply by number of letters in the question

APPROACH: Single agent with reasoning (no tools)
- Agent recalls the State Farm jingle
- Performs letter counting and calculation
- Minimal complexity

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os

# MUST set these BEFORE importing CrewAI to prevent OpenAI connections
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-bedrock"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from pathlib import Path
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    project_root = Path(__file__).parent.parent
    task_file = project_root / 'tasks' / 'statefarm.yaml'
    
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="statefarm-single-agent-crewai",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    
    # ============================================================
    # AGENT: Uses ONLY the task prompt
    # ============================================================
    agent = Agent(
        role="Task Executor",
        goal="Complete the task as specified",
        backstory="You follow instructions precisely.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    # ============================================================
    # TASK: Uses prompt from config file
    # ============================================================
    task = Task(
        description=task_prompt,
        agent=agent,
        expected_output="The calculated result with clear reasoning"
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
    print("STATEFARM TASK - Single Agent CrewAI")
    print("=" * 60)
    print("Approach: Single agent with reasoning")
    print("Tools: None (LLM reasoning only)")
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
