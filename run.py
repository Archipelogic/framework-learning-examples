#!/usr/bin/env python3
"""
Interactive Task Runner for AI Framework Examples
Run unified framework scripts with orchestration agents
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

# Unified framework scripts
FRAMEWORKS = {
    "CrewAI": "crewai_unified.py",
    "Pydantic AI": "pydantic_ai_unified.py"
}


def load_sample_prompts():
    """
    Dynamically load sample prompts from task YAML files.
    Returns dict of {task_name: prompt}
    """
    tasks_dir = Path(__file__).parent / 'tasks'
    prompts = {}
    
    # Load all YAML files from tasks directory
    for task_file in sorted(tasks_dir.glob('*.yaml')):
        try:
            with open(task_file, 'r') as f:
                task_data = yaml.safe_load(f)
                task = task_data.get('task', {})
                
                # Use task name as key, prompt as value
                name = task.get('name', task_file.stem.title())
                prompt = task.get('prompt', '').strip()
                
                if prompt:
                    prompts[name] = prompt
        except Exception as e:
            print(f"Warning: Could not load {task_file.name}: {e}")
    
    return prompts


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
    frameworks = list(FRAMEWORKS.keys())
    
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


def get_prompt():
    """Get prompt from user - either custom or sample (loaded from task YAMLs)"""
    print()
    print("ENTER YOUR PROMPT:")
    print()
    print("Sample prompts from task definitions (or enter your own):")
    print()
    
    # Dynamically load prompts from task YAML files
    sample_prompts = load_sample_prompts()
    prompts = list(sample_prompts.items())
    
    if not prompts:
        print("Warning: No task files found in tasks/ directory")
        print()
    
    for i, (name, prompt) in enumerate(prompts, 1):
        print(f"  {i}. {name}")
        print(f"     \"{prompt[:80]}...\"" if len(prompt) > 80 else f"     \"{prompt}\"")
        print()
    
    print(f"  {len(prompts) + 1}. Enter custom prompt")
    print(f"  {len(prompts) + 2}. Back to framework selection")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-{}): ".format(len(prompts) + 2)).strip()
            choice_num = int(choice)
            
            if choice_num == len(prompts) + 2:
                return None
            
            if choice_num == len(prompts) + 1:
                print()
                custom_prompt = input("Enter your prompt: ").strip()
                if custom_prompt:
                    return custom_prompt
                print("Empty prompt. Try again.")
                continue
            
            if 1 <= choice_num <= len(prompts):
                return prompts[choice_num - 1][1]
            
            print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Try again.")


def run_task(script_name, prompt):
    """Run the selected framework script with the given prompt"""
    clear_screen()
    print_header()
    print(f"FRAMEWORK: {script_name}")
    print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")
    print("=" * 70)
    print()
    
    try:
        # Run the script with the prompt as argument
        result = subprocess.run(
            [sys.executable, script_name, prompt],
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
        
        script_name = FRAMEWORKS[framework]
        
        while True:
            clear_screen()
            print_header()
            print(f"FRAMEWORK: {framework}")
            
            prompt = get_prompt()
            if prompt is None:
                break
            
            run_task(script_name, prompt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
