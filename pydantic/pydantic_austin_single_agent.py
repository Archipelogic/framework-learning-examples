#!/usr/bin/env python3
"""
Austin Task - Pydantic AI Single Agent Implementation

TASK: Get current time in Austin, TX and calculate: minutes^(1/hour)

APPROACH: Single agent with structured output (no tools)
- Agent uses reasoning to determine time and calculate
- Pydantic model ensures structured, type-safe output

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


# ============================================================
# OUTPUT STRUCTURE: Type-safe result
# ============================================================
class AustinResult(BaseModel):
    """Structured output for Austin time calculation."""
    current_time: str = Field(description="Current time in Austin, TX")
    hour: int = Field(description="Hour value extracted")
    minutes: int = Field(description="Minutes value extracted")
    calculation: str = Field(description="The calculation performed: minutes^(1/hour)")
    result: float = Field(description="Final numerical result")


def main():
    # ============================================================
    # LOAD TASK CONFIG
    # ============================================================
    project_root = Path(__file__).parent.parent
    task_file = project_root / 'tasks' / 'austin.yaml'
    
    with open(task_file, 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="austin-single-agent-pydantic",
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
    # AGENT: Uses ONLY the task prompt
    # ============================================================
    agent = Agent(
        model=model,
        result_type=AustinResult,
        system_prompt="You follow instructions precisely and provide structured output."
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("AUSTIN TASK - Single Agent Pydantic AI")
    print("=" * 60)
    print("Approach: Single agent with structured output")
    print("Tools: None (LLM reasoning only)")
    print("Output: Type-safe Pydantic model")
    print("=" * 60)
    print()
    
    result = agent.run_sync(task_prompt)
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Current Time: {result.data.current_time}")
    print(f"Hour: {result.data.hour}")
    print(f"Minutes: {result.data.minutes}")
    print(f"Calculation: {result.data.calculation}")
    print(f"Result: {result.data.result}")
    print()
    
    if hasattr(result, 'usage'):
        usage = result.usage()
        print(f"Token Usage: {usage.total_tokens} total")
    
    print(f"\nView traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
