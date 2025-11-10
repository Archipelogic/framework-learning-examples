#!/usr/bin/env python3
"""
StateFarm Task - Pydantic AI with Logfire Observability

TASK: Find 5th letter in State Farm jingle, convert to alphabet position,
      multiply by number of letters in the question

APPROACH: Single agent with NO tools, using Logfire for observability
- LLM has all needed knowledge
- Logfire provides detailed tracing and logging
- Alternative to Phoenix for observability

OBSERVABILITY: Logfire (Pydantic's observability platform)

NOTE: Logfire is Pydantic's own observability solution. It provides
detailed insights into Pydantic AI execution without requiring login
for local development.
"""

import os
import yaml
import logfire
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
    with open('../tasks/statefarm.yaml', 'r') as f:
        task_config = yaml.safe_load(f)['task']
    
    task_prompt = task_config['prompt']
    
    # ============================================================
    # OBSERVABILITY SETUP: Logfire
    # ============================================================
    logfire.configure(
        send_to_logfire='if-token-present',
        console=True,
    )
    
    logfire.instrument_pydantic_ai()
    
    print("=" * 60)
    print("Logfire Observability Enabled")
    print("Logs will appear in console")
    print("Set LOGFIRE_TOKEN env var to send to Logfire cloud")
    print("=" * 60)
    print()
    
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
        result_type=StateFarmResult,
        system_prompt="You follow instructions precisely."
    )
    
    # ============================================================
    # EXECUTION with Logfire tracing
    # ============================================================
    print("\n" + "=" * 60)
    print("STATEFARM TASK - Pydantic AI with Logfire")
    print("=" * 60)
    print("Approach: Single agent, NO tools")
    print("Observability: Logfire (Pydantic's platform)")
    print("=" * 60)
    print()
    
    # Logfire automatically traces this execution
    with logfire.span("statefarm_puzzle_solving"):
        logfire.info("Starting StateFarm puzzle", prompt=task_prompt)
        
        result = agent.run_sync(task_prompt)
        
        logfire.info(
            "Puzzle solved",
            jingle=result.data.jingle,
            fifth_letter=result.data.fifth_letter,
            answer=result.data.final_answer
        )
    
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
        logfire.info("Token usage", total=usage.total_tokens)
    
    print("\n" + "=" * 60)
    print("LOGFIRE OBSERVABILITY")
    print("=" * 60)
    print("✓ All execution traced automatically")
    print("✓ Logs visible in console above")
    print("✓ Set LOGFIRE_TOKEN to send to cloud")
    print("=" * 60)


if __name__ == "__main__":
    main()
