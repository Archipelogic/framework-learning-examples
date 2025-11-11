#!/usr/bin/env python3
"""
StateFarm Task - Pydantic AI Single Agent Implementation

TASK: Find 5th letter in State Farm jingle, convert to alphabet position,
      multiply by number of letters in the question

APPROACH: Single agent with structured output (no tools)
- Agent uses reasoning for all steps
- Pydantic model ensures structured output

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
# OUTPUT STRUCTURE
# ============================================================
class StateFarmResult(BaseModel):
    """Structured output for StateFarm puzzle."""
    jingle: str = Field(description="The State Farm jingle")
    fifth_letter: str = Field(description="The 5th letter in the jingle")
    alphabet_position: int = Field(description="Position of letter in alphabet")
    question_letter_count: int = Field(description="Number of letters in the question")
    calculation: str = Field(description="The multiplication performed")
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
        project_name="statefarm-single-agent-pydantic",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    PydanticAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        region_name="us-east-1"
    )
    
    # ============================================================
    # AGENT: Uses ONLY the task prompt
    # ============================================================
    agent = Agent(
        model=model,
        result_type=StateFarmResult,
        system_prompt="You follow instructions precisely and provide structured output."
    )
    
    # ============================================================
    # EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("STATEFARM TASK - Single Agent Pydantic AI")
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
    print(f"Jingle: {result.data.jingle}")
    print(f"5th Letter: {result.data.fifth_letter}")
    print(f"Alphabet Position: {result.data.alphabet_position}")
    print(f"Question Letter Count: {result.data.question_letter_count}")
    print(f"Calculation: {result.data.calculation}")
    print(f"Final Answer: {result.data.final_answer}")
    print()
    
    if hasattr(result, 'usage'):
        usage = result.usage()
        print(f"Token Usage: {usage.total_tokens} total")
    
    print(f"\nView traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
