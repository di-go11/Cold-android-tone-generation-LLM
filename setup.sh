#!/bin/bash
"""
Setup Script for ELYZA LLM Fine-tuning Environment

This script sets up the environment for fine-tuning ELYZA's LLM.
"""

echo "🤖 ELYZA LLM Fine-tuning Setup"
echo "=============================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p output logs data config

# Generate sample data
echo "Generating sample data..."
python prepare_data.py --create_sample

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit config/training_config.yaml if needed"
echo "3. Run fine-tuning: python finetune.py"
echo "4. Generate text: python generate.py --interactive"
echo ""
echo "For more information, see README.md"