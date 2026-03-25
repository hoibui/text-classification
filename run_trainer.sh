#!/bin/bash

# ML Training Script for Text Classification
# Usage: ./run_trainer.sh [model_name] [model_version] [data_path]

# Configuration
MODEL_NAME="${1:-ml_model}"
MODEL_VERSION="${2:-1.0.0}"
DATA_PATH="${3:-data/train.csv}"

echo "🚀 Starting ML Model Training"
echo "=============================="
echo "Model Name: $MODEL_NAME"
echo "Model Version: $MODEL_VERSION"
echo "Data Path: $DATA_PATH"
echo "=============================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found. Make sure dependencies are installed."
fi

# Set Python path for imports
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data file not found at $DATA_PATH"
    echo "Please make sure the data file exists."
    exit 1
fi

# Run ML training
echo "🏋️ Starting ML training..."
python src/main.py train \
    --name "$MODEL_NAME" \
    --version "$MODEL_VERSION" \
    --data-path "$DATA_PATH"

if [ $? -eq 0 ]; then
    echo "✅ ML training completed successfully!"
    echo "Model saved and ready for inference."
    echo ""
    echo "To test the model:"
    echo "curl -X POST http://localhost:8001/classify/ -H \"Content-Type: application/json\" -d '{\"text\": \"test text\", \"return_confidence\": true}'"
else
    echo "❌ ML training failed!"
    exit 1
fi