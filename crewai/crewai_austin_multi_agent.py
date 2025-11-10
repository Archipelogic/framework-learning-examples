#!/usr/bin/env python3
"""
Austin Task - CrewAI Multi-Agent Implementation

TASK: Get current time in Austin, TX and calculate: minutes^(1/hour)

APPROACH: Multiple specialized agents working sequentially
- Time Agent: Determines current Austin time
- Math Agent: Performs the calculation
- Shows agent collaboration without tools

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew, Process


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    with open('../tasks/austin.yaml', 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="austin-multi-agent-crewai",
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
    time_agent = Agent(
        role="Time Specialist",
        goal="Handle time-related parts of the task",
        backstory="You focus on time determination.",
        llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    math_agent = Agent(
        role="Math Specialist",
        goal="Handle mathematical parts of the task",
        backstory="You focus on calculations.",
        llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
        verbose=True,
        allow_delegation=False
    )
    
    # ============================================================
    # TASKS: Break down the main prompt
    # ============================================================
    time_task = Task(
        description="What is the current time in Austin, Texas? Provide hour and minutes.",
        agent=time_agent,
        expected_output="Current Austin time"
    )
    
    math_task = Task(
        description="Using the time from the previous task, " + task_prompt.split('?')[1].strip(),
        agent=math_agent,
        expected_output="Calculated result"
    )
    
    # ============================================================
    # CREW: Multi-agent collaboration
    # ============================================================
    crew = Crew(
        agents=[time_agent, math_agent],
        tasks=[time_task, math_task],
        process=Process.sequential,  # Tasks execute in order
        verbose=True
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("AUSTIN TASK - Multi-Agent CrewAI")
    print("=" * 60)
    print("Approach: Specialized agents (Time + Math)")
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
