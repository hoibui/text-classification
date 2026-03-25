#!/bin/bash

# Script to run API server with Clean Architecture
source venv/bin/activate

# Default values from environment or fallback
API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8001}
API_WORKERS=${API_WORKERS:-1}

echo "Starting Text Classification API server with Clean Architecture..."
echo "Host: $API_HOST"
echo "Port: $API_PORT"
echo "Workers: $API_WORKERS"

# Set PYTHONPATH and run from src directory
export PYTHONPATH="$PWD/src:$PYTHONPATH"
cd src && python main.py serve --host "$API_HOST" --port "$API_PORT" --workers "$API_WORKERS"