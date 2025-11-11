#!/usr/bin/env python3
"""
StateFarm Task - Pydantic AI with NO Tools Implementation

TASK: Find 5th letter in State Farm jingle, convert to alphabet position,
      multiply by number of letters in the question

APPROACH: Single agent with NO tools - LLM has this knowledge
- LLM knows the State Farm jingle
- LLM can count letters
- LLM knows alphabet positions
- No tools needed!

OBSERVABILITY: Phoenix (Arize) for tracing

NOTE: This shows that tools aren't always necessary. The LLM already has
the capabilities needed for this task.
"""

import os
from pathlib import Path
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.pydantic_ai import PydanticAIInstrumentor
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockModel


# ============================================================
# OUTPUT STRUCTURE
# ============================================================
class StateFarmResult(BaseModel):
    """Structured output for StateFarm puzzle."""
    reasoning: str = Field(description="Step-by-step reasoning")
    final_answer: int = Field(description="Final numerical result")


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
        project_name="statefarm-no-tools-pydantic",
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
    # AGENT with NO TOOLS - LLM has this knowledge
    # ============================================================
    agent = Agent(
        model=model,
        result_type=StateFarmResult,
        system_prompt="You follow instructions precisely."
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("STATEFARM TASK - Pydantic AI (NO Tools)")
    print("=" * 60)
    print("Approach: Single agent, NO tools")
    print("Why no tools? LLM already knows:")
    print("  - State Farm jingle")
    print("  - Letter counting")
    print("  - Alphabet positions")
    print("=" * 60)
    print()
    
    result = agent.run_sync(task_prompt)
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"Reasoning:\n{result.data.reasoning}")
    print(f"\nFinal Answer: {result.data.final_answer}")
    print()
    
    if hasattr(result, 'usage'):
        usage = result.usage()
        print(f"Token Usage: {usage.total_tokens} total")
    
    print(f"\nView traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
