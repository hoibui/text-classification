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
echo "✅ Clean Architecture Text Classification System Setup Complete! 🎉"
echo ""
echo "🚀 Next steps:"
echo "1. Edit the .env file with your database credentials and paths"
echo "2. Setup your database: python setup_database.py"
echo ""
echo "📚 Available Commands:"
echo "  Training:"
echo "    ./run_trainer.sh                    # Train with default settings"
echo "    MODEL_NAME=my_model ./run_trainer.sh # Train with custom name"
echo ""
echo "  API Server:"
echo "    ./run_api.sh                        # Start API server"
echo ""
echo "  Model Management:"
echo "    ./list_models.sh                    # List all models"
echo "    ./list_models.sh ready             # List models by status"
echo ""
echo "  Development:"
echo "    source venv/bin/activate            # Activate virtual environment"
echo "    python src/main.py --help          # See all CLI options"
echo ""
echo "🐳 Docker:"
echo "    docker-compose up -d               # Start all services"
echo ""
echo "📖 Documentation: Check README.md for detailed usage instructions"