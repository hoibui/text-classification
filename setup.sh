#!/bin/bash

# Setup script for text-classification project
echo "Setting up text-classification project..."

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your specific configuration!"
fi

# Create docker-compose override file if it doesn't exist
if [ ! -f "docker-compose.override.yml" ]; then
    echo "Creating docker-compose.override.yml from example..."
    cp docker-compose.override.yml.example docker-compose.override.yml
    echo "📝 You can customize docker-compose.override.yml for your local setup"
fi

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

echo ""
echo "Setup complete! 🎉"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your database credentials and paths"
echo "2. To activate the environment in the future, run: source venv/bin/activate"
echo "3. To run the trainer: ./run_trainer.sh"