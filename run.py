#!/usr/bin/env python3
"""
Interactive Task Runner for AI Framework Examples
Run tasks grouped by framework with an interactive menu
"""

import os
import sys
import subprocess
from pathlib import Path

# Task definitions grouped by framework
TASKS = {
    "CrewAI": {
        "Austin - Single Agent": "crewai/crewai_austin_single_agent.py",
        "Austin - Multi Agent": "crewai/crewai_austin_multi_agent.py",
        "StateFarm - Single Agent": "crewai/crewai_statefarm_single_agent.py",
        "StateFarm - Multi Agent": "crewai/crewai_statefarm_multi_agent.py",
        "RAG - Single Agent": "crewai/crewai_rag_single_agent.py",
        "SQL - Single Agent": "crewai/crewai_sql_single_agent.py",
    },
    "Pydantic AI": {
        "Austin - Single Agent": "pydantic/pydantic_austin_single_agent.py",
        "Austin - No Tools": "pydantic/pydantic_austin_no_tools.py",
        "StateFarm - Single Agent": "pydantic/pydantic_statefarm_single_agent.py",
        "StateFarm - No Tools": "pydantic/pydantic_statefarm_no_tools.py",
        "StateFarm - Logfire": "pydantic/pydantic_statefarm_logfire.py",
        "RAG - Single Agent": "pydantic/pydantic_rag_single_agent.py",
        "SQL - Single Agent": "pydantic/pydantic_sql_single_agent.py",
    }
}


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print header"""
    print("=" * 70)
    print("AI FRAMEWORK EXAMPLES - INTERACTIVE TASK RUNNER")
    print("=" * 70)
    print()


def select_framework():
    """Let user select framework"""
    frameworks = list(TASKS.keys())
    
    print("SELECT FRAMEWORK:")
    print()
    for i, framework in enumerate(frameworks, 1):
        print(f"  {i}. {framework}")
    print(f"  {len(frameworks) + 1}. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-{}): ".format(len(frameworks) + 1)).strip()
            choice_num = int(choice)
            
            if choice_num == len(frameworks) + 1:
                return None
            
            if 1 <= choice_num <= len(frameworks):
                return frameworks[choice_num - 1]
            
            print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Try again.")


def select_task(framework):
    """Let user select task for the framework"""
    tasks = TASKS[framework]
    task_list = list(tasks.items())
    
    clear_screen()
    print_header()
    print(f"FRAMEWORK: {framework}")
    print()
    print("SELECT TASK:")
    print()
    
    for i, (name, _) in enumerate(task_list, 1):
        print(f"  {i}. {name}")
    print(f"  {len(task_list) + 1}. Back to framework selection")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-{}): ".format(len(task_list) + 1)).strip()
            choice_num = int(choice)
            
            if choice_num == len(task_list) + 1:
                return None
            
            if 1 <= choice_num <= len(task_list):
                return task_list[choice_num - 1][1]
            
            print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Try again.")


def run_task(script_name):
    """Run the selected task script"""
    clear_screen()
    print_header()
    print(f"RUNNING: {script_name}")
    print("=" * 70)
    print()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=False
        )
        
        print()
        print("=" * 70)
        if result.returncode == 0:
            print("TASK COMPLETED SUCCESSFULLY")
        else:
            print(f"TASK FAILED (exit code: {result.returncode})")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
    except Exception as e:
        print(f"\nError running task: {e}")
    
    print()
    input("Press Enter to continue...")


def main():
    """Main interactive loop"""
    while True:
        clear_screen()
        print_header()
        
        framework = select_framework()
        if framework is None:
            print("\nExiting...")
            break
        
        while True:
            task = select_task(framework)
            if task is None:
                break
            
            run_task(task)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
