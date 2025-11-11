#!/usr/bin/env python3
"""
StateFarm Task - CrewAI Multi-Agent Implementation

TASK: Find 5th letter in State Farm jingle, convert to alphabet position,
      multiply by number of letters in the question

APPROACH: Multiple specialized agents
- Knowledge Agent: Recalls the State Farm jingle
- Counter Agent: Counts letters in the question
- Math Agent: Performs the calculation
- Shows agent collaboration

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


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    # Get project root (parent of crewai directory)
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
        project_name="statefarm-multi-agent-crewai",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    
    # ============================================================
    # AGENTS: Specialized agents
    # ============================================================
    knowledge_agent = Agent(
        role="Knowledge Specialist",
        goal="Handle knowledge-related parts of the task",
        backstory="You focus on recall and information.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    counter_agent = Agent(
        role="Counter Specialist",
        goal="Handle counting parts of the task",
        backstory="You focus on counting.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    math_agent = Agent(
        role="Math Specialist",
        goal="Handle mathematical parts of the task",
        backstory="You focus on calculations.",
        llm="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    # ============================================================
    # TASKS: Break down the main prompt
    # ============================================================
    jingle_task = Task(
        description="Find the 5th letter in the State Farm jingle and convert it to its alphabet position.",
        agent=knowledge_agent,
        expected_output="The 5th letter and its position"
    )
    
    count_task = Task(
        description=f"Count the number of letters in this question: {task_prompt}",
        agent=counter_agent,
        expected_output="The letter count"
    )
    
    multiply_task = Task(
        description="Multiply the alphabet position from task 1 by the letter count from task 2.",
        agent=math_agent,
        expected_output="The final result"
    )
    
    # ============================================================
    # CREW: Multi-agent collaboration
    # ============================================================
    crew = Crew(
        agents=[knowledge_agent, counter_agent, math_agent],
        tasks=[jingle_task, count_task, multiply_task],
        process=Process.sequential,
        verbose=True
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("STATEFARM TASK - Multi-Agent CrewAI")
    print("=" * 60)
    print("Approach: Specialized agents (Knowledge + Counter + Math)")
    print("Tools: None (agent reasoning)")
    print("Orchestration: Sequential task execution")
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
