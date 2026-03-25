#!/bin/bash

# Script to list models with Clean Architecture
source venv/bin/activate

# Optional status filter
STATUS_FILTER=${1:-""}

# Set PYTHONPATH and run from src directory
export PYTHONPATH="$PWD/src:$PYTHONPATH"
cd src

if [ -n "$STATUS_FILTER" ]; then
    echo "Listing models with status: $STATUS_FILTER"
    python main.py list --status "$STATUS_FILTER"
else
    echo "Listing all models:"
    python main.py list
fi