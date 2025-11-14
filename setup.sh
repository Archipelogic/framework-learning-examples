#!/bin/sh

# Setup script for AI Framework Examples
# This script:
# 1. Installs dependencies using uv
# 2. Sets up sample data
# 3. Launches interactive task runner

set -e  # Exit on error

echo "=========================================="
echo "AI Framework Examples - Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run from framework-learning-examples directory."
    exit 1
fi

# Step 1: Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Step 2: Create virtual environment and install dependencies
echo ""
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Installing dependencies with uv..."
uv sync
echo "✓ Dependencies installed (from uv.lock)"

# Step 3: Setup sample database
echo ""
echo "Setting up sample database..."
uv run python sql.py
echo "✓ Database created"

# Step 4: Check AWS credentials
echo ""
echo "Checking AWS configuration..."
if [ -z "$AWS_REGION" ]; then
    export AWS_REGION=us-east-1
    echo "✓ AWS_REGION set to us-east-1"
else
    echo "✓ AWS_REGION already set to $AWS_REGION"
fi

# Step 5: Display summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project Structure:"
echo "  - 2 unified scripts (crewai.py & pydantic_ai.py)"
echo "  - Orchestration agents for intelligent task routing"
echo "  - 4 task configs (austin, statefarm, rag, sql)"
echo "  - Sample data (JSON files & SQLite database)"
echo ""
echo "Next Steps:"
echo "  1. Ensure AWS credentials are configured"
echo "  2. Run: sh run.sh (or uv run python run.py)"
echo ""
echo "=========================================="
echo ""

# Step 6: Ask if user wants to run interactive menu
printf "Launch interactive task runner now? (y/n): "
read -r reply
if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
    echo ""
    uv run python run.py
fi
