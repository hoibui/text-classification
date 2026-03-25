#!/usr/bin/env python3
"""
Main entry point for the text classification system.
Supports both training and API server modes with Clean Architecture.
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path and set up proper Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir
root_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Set PYTHONPATH environment variable for subprocesses
os.environ['PYTHONPATH'] = str(src_dir)

from infrastructure.config.settings import Settings
from infrastructure.config.dependencies import get_container
from application.dto.model_dto import ModelTrainingRequestDTO


async def train_model(args):
    """Train a new model using the Clean Architecture."""
    try:
        # Get dependencies
        container = get_container()
        training_use_case = container.get_train_model_use_case()

        # Create training request
        training_request = ModelTrainingRequestDTO(
            name=args.name,
            version=args.version,
            model_type="traditional_ml",
            training_data_path=args.data_path,
            config=container.settings.config,
            description=f"Model trained on {args.data_path}"
        )

        print(f"Starting training for model {args.name} v{args.version}...")

        # Execute training
        model = await training_use_case.execute(training_request)

        print(f"Training completed successfully!")
        print(f"Model ID: {model.id}")
        print(f"Model Type: {model.model_type}")
        print(f"Accuracy: {model.metrics.accuracy:.4f}")
        print(f"F1 Score: {model.metrics.f1_score:.4f}")
        print(f"File Path: {model.file_path}")

        # Set as active model
        model_info_use_case = container.get_model_info_use_case()
        if model.id:
            success = await model_info_use_case.set_active_model(model.id, args.name)
            if success:
                print(f"Model {args.name} v{args.version} is now active!")
            else:
                print("Warning: Could not set model as active")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)



def run_api_server(args):
    """Run the API server."""
    import uvicorn
    from presentation.api.main import app

    settings = Settings()

    print(f"Starting Text Classification API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=settings.log_level.lower()
    )


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description="Text Classification System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--name", required=True, help="Model name")
    train_parser.add_argument("--version", required=True, help="Model version")
    train_parser.add_argument("--data-path", default="data/train.csv", help="Training data path")


    # API server command
    api_parser = subparsers.add_parser("serve", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    # List models command
    list_parser = subparsers.add_parser("list", help="List all models")
    list_parser.add_argument("--status", help="Filter by status")

    args = parser.parse_args()

    if args.command == "train":
        asyncio.run(train_model(args))
    elif args.command == "serve":
        run_api_server(args)
    elif args.command == "list":
        asyncio.run(list_models(args))
    else:
        parser.print_help()


async def list_models(args):
    """List all models."""
    try:
        container = get_container()
        model_info_use_case = container.get_model_info_use_case()

        if args.status:
            models = await model_info_use_case.list_models_by_status(args.status)
        else:
            models = await model_info_use_case.list_models()

        if not models:
            print("No models found.")
            return

        print(f"{'ID':<5} {'Name':<20} {'Version':<10} {'Type':<15} {'Status':<10} {'Accuracy':<10}")
        print("-" * 80)

        for model in models:
            accuracy = f"{model.metrics.accuracy:.4f}" if model.metrics else "N/A"
            print(f"{model.id:<5} {model.name:<20} {model.version:<10} {model.model_type:<15} {model.status:<10} {accuracy:<10}")

    except Exception as e:
        print(f"Failed to list models: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()