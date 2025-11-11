#!/usr/bin/env python3
"""
StateFarm Task - CrewAI Flow with @listen Decorator

TASK: Find 5th letter in State Farm jingle, convert to alphabet position,
      multiply by number of letters in the question

APPROACH: Event-driven Flow with @listen decorators
- Uses @start() and @listen() for automatic method chaining
- Each step triggers the next automatically
- Demonstrates reactive workflow orchestration

OBSERVABILITY: Phoenix (Arize) for tracing
"""

import os
import yaml
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.crewai import CrewAIInstrumentor
from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start


class StateFarmFlow(Flow):
    """Flow-based implementation with @listen decorator for event-driven execution"""
    
    def __init__(self):
        super().__init__()
        
        # Load task configuration
        with open('../tasks/statefarm.yaml', 'r') as f:
            task_config = yaml.safe_load(f)['task']
        
        self.task_prompt = task_config['prompt']
        
        # Initialize agents
        self.knowledge_agent = Agent(
            role="Knowledge Specialist",
            goal="Handle knowledge-related parts of the task",
            backstory="You focus on recall and information.",
            llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
            verbose=True,
            allow_delegation=False
        )
        
        self.counter_agent = Agent(
            role="Counter Specialist",
            goal="Handle counting parts of the task",
            backstory="You focus on counting.",
            llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
            verbose=True,
            allow_delegation=False
        )
        
        self.math_agent = Agent(
            role="Math Specialist",
            goal="Handle mathematical parts of the task",
            backstory="You focus on calculations.",
            llm="bedrock/anthropic.claude-sonnet-4-5-v1:0",
            verbose=True,
            allow_delegation=False
        )
    
    @start()
    def find_jingle_letter(self):
        """
        Step 1: Find the 5th letter in State Farm jingle and convert to alphabet position
        This is the entry point marked with @start()
        """
        print("\n" + "=" * 60)
        print("🚀 STEP 1: Finding 5th letter in State Farm jingle")
        print("=" * 60)
        
        task = Task(
            description="Find the 5th letter in the State Farm jingle and convert it to its alphabet position.",
            agent=self.knowledge_agent,
            expected_output="The 5th letter and its position in the alphabet"
        )
        
        crew = Crew(
            agents=[self.knowledge_agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        print(f"\n✅ Step 1 Complete: {result}")
        
        return {
            "jingle_result": str(result),
            "step": "jingle_letter"
        }
    
    @listen(find_jingle_letter)
    def count_question_letters(self, jingle_output):
        """
        Step 2: Count letters in the question
        Automatically triggered when find_jingle_letter completes
        """
        print("\n" + "=" * 60)
        print("🔢 STEP 2: Counting letters in question")
        print("=" * 60)
        print(f"Previous result: {jingle_output['jingle_result']}")
        
        task = Task(
            description=f"Count the number of letters in this question: {self.task_prompt}",
            agent=self.counter_agent,
            expected_output="The total number of letters in the question"
        )
        
        crew = Crew(
            agents=[self.counter_agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        print(f"\n✅ Step 2 Complete: {result}")
        
        return {
            "jingle_result": jingle_output["jingle_result"],
            "count_result": str(result),
            "step": "letter_count"
        }
    
    @listen(count_question_letters)
    def calculate_final_answer(self, count_output):
        """
        Step 3: Multiply the alphabet position by the letter count
        Automatically triggered when count_question_letters completes
        """
        print("\n" + "=" * 60)
        print("🧮 STEP 3: Calculating final answer")
        print("=" * 60)
        print(f"Jingle result: {count_output['jingle_result']}")
        print(f"Count result: {count_output['count_result']}")
        
        task = Task(
            description=f"""
            Based on these results:
            1. Jingle letter analysis: {count_output['jingle_result']}
            2. Letter count: {count_output['count_result']}
            
            Multiply the alphabet position from step 1 by the letter count from step 2.
            Provide the final numerical answer.
            """,
            agent=self.math_agent,
            expected_output="The final calculated result"
        )
        
        crew = Crew(
            agents=[self.math_agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        print("\n" + "=" * 60)
        print("🏁 FINAL RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        return {
            "final_answer": str(result),
            "step": "final_calculation"
        }


def main():
    # ============================================================
    # OBSERVABILITY SETUP
    # ============================================================
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    
    tracer_provider = register(
        project_name="statefarm-flow-listen",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # ============================================================
    # AWS BEDROCK SETUP
    # ============================================================
    os.environ["AWS_REGION"] = "us-east-1"
    
    # ============================================================
    # FLOW EXECUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("STATEFARM TASK - Flow with @listen Decorator")
    print("=" * 60)
    print("Approach: Event-driven Flow")
    print("Pattern: @start() → @listen() → @listen()")
    print("Orchestration: Automatic method chaining")
    print("=" * 60)
    print()
    
    # Create and run the flow
    flow = StateFarmFlow()
    result = flow.kickoff()
    
    print("\n" + "=" * 60)
    print("FLOW EXECUTION COMPLETE")
    print("=" * 60)
    print(f"View traces in Phoenix: {session.url}")


if __name__ == "__main__":
    main()
